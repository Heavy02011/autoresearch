# AutoResearch × DonkeyCar — Adaptation Plan

> **Goal**: Apply the AutoResearch "one GPU, one file, one metric" autonomous-experimentation loop to
> optimising the neural-network autopilot of a [DonkeyCar](https://github.com/autorope/donkeycar)
> RC vehicle.  An AI agent iterates overnight on `train_donkey.py`, training for a fixed time budget,
> measuring a clear scalar metric, and keeping or discarding each experiment — exactly as AutoResearch
> does today for GPT pretraining.

---

## 1. Background — How Each System Works

### 1.1 AutoResearch (this repo)

| Concern | Current implementation |
|---|---|
| Task | GPT language-model pretraining |
| Data | ClimbMix 400 B text tokens (parquet shards) |
| Model | Transformer (GPT) — modified by the agent |
| Metric | `val_bpb` — validation bits-per-byte (lower = better) |
| Time budget | 5 minutes wall-clock training |
| Fixed file | `prepare.py` — data prep, tokeniser, evaluation harness |
| Agent file | `train.py` — model, optimiser, hyperparams |
| Instructions | `program.md` |

### 1.2 DonkeyCar

DonkeyCar is a behavioural-cloning platform for RC cars:

1. A human drives the car around a track while a camera records frames.
2. Each frame is stored together with the steering angle and throttle that the human applied at that moment — this forms a **tub** (DonkeyCar's data format: a folder of JPEG images + a JSON-line catalogue).
3. A neural network is trained to predict `(steering, throttle)` from a camera frame.
4. At inference time the trained network is loaded onto a Raspberry Pi / Jetson Nano and controls the car in real time.

Standard DonkeyCar model: a small Keras/TF CNN (`Linear` model, ~50 k parameters).  
The adaptation below uses **PyTorch** throughout so that it integrates cleanly with the AutoResearch stack.

---

## 2. What Stays the Same

* **Autonomous experiment loop** — agent edits one file, runs for 5 minutes, logs results, keeps or discards.
* **Single-file agent scope** — `train_donkey.py` is the only file the agent touches.
* **Fixed time budget** — still 5 minutes wall-clock training time.
* **`results.tsv` logging** — same five-column format: `commit`, `val_metric`, `memory_gb`, `status`, `description`.
* **`program_donkey.md`** — same structure as `program.md`, adapted for the driving task.

---

## 3. What Changes

| Concern | AutoResearch | DonkeyCar adaptation |
|---|---|---|
| Input data | Text tokens (1-D sequences) | Camera frames (H×W×3 images) + optional IMU/speed |
| Model family | Transformer / GPT | CNN encoder → regression head |
| Output | Next-token logits | `(steering, throttle)` ∈ [−1,1] × [0,1] |
| Loss | Cross-entropy | MSE (or Huber) on steering + throttle |
| Metric | `val_bpb` | `val_mse` — mean steering MSE on held-out laps |
| Data format | Parquet shards | DonkeyCar **tub** directories |
| On-car hardware | n/a | Raspberry Pi 4 / Jetson Nano |
| Deployment | n/a | Export model → ONNX or TorchScript, load with `donkeycar` |

---

## 4. New File Structure

```
prepare_donkey.py   — fixed: tub loading, pre-processing, evaluation (do NOT modify)
train_donkey.py     — agent modifies: CNN model, optimiser, hyperparameters
program_donkey.md   — agent instructions for the driving task
results_donkey.tsv  — experiment log (not tracked by git)
```

The existing `prepare.py` / `train.py` / `program.md` remain untouched.

---

## 5. `prepare_donkey.py` — Specification

This file is **read-only** for the agent, just like `prepare.py` today.

### 5.1 Constants (fixed for all experiments)

```python
IMG_H, IMG_W   = 120, 160        # camera resolution: height=120, width=160 (DonkeyCar default)
IMG_CHANNELS   = 3               # RGB
MAX_SEQ_LEN    = 1               # single-frame model (extend later for temporal)
TIME_BUDGET    = 300             # 5-minute wall-clock training budget (seconds)
VAL_FRACTION   = 0.15            # fraction of tubs reserved for validation
STEERING_SCALE = 1.0             # steering already in [−1, 1]
THROTTLE_SCALE = 1.0             # throttle already in [0, 1]
```

### 5.2 Tub data loader

```python
def make_dataloader(tub_paths: list[str], split: str,
                    batch_size: int) -> DataLoader:
    """
    Returns a PyTorch DataLoader over DonkeyCar tub directories.

    Each sample: (image_tensor [3,H,W] float32, label_tensor [2] float32)
    label_tensor = [steering, throttle]

    'split' is "train" or "val" — split is done deterministically by tub index.
    Images are normalised to [0, 1] and optionally augmented (train only).
    """
```

Key implementation notes:
* Parse each tub's `manifest.json` / `catalogue` to enumerate `(image_path, steering, throttle)` records.
* Deterministic train/val split at the **tub** level (not frame level) to avoid leakage from temporally-correlated frames.
* Augmentation (horizontal flip + mirrored steering, brightness jitter) applied during training only.

### 5.3 Evaluation harness

```python
def evaluate_mse(model: torch.nn.Module,
                 dataloader: DataLoader,
                 device: str) -> float:
    """
    Returns the mean squared error on steering prediction over the validation set.
    This is the ground-truth metric — do not modify.
    Throttle is excluded from the primary metric so that the agent focuses on
    steering quality (throttle can be fixed or tuned separately).
    """
```

> **Why steering MSE only?**  Steering accuracy is the primary determinant of lap quality.
> Throttle can be held constant or tuned with a simple heuristic after the model is deployed.
> Using a single scalar metric keeps the experiment loop identical to AutoResearch.

---

## 6. `train_donkey.py` — Specification

This file is **fully editable by the agent**, mirroring `train.py`.

### 6.1 Baseline model (DonkeyCar Linear equivalent in PyTorch)

```python
class DonkeyNet(nn.Module):
    """Baseline CNN autopilot.  Agent may replace entirely."""
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3,  24, 5, stride=2), nn.ReLU(),
            nn.Conv2d(24, 32, 5, stride=2), nn.ReLU(),
            nn.Conv2d(32, 64, 5, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
            nn.Flatten(),
        )
        self.head = nn.Sequential(
            nn.Linear(1152, 100), nn.ReLU(),
            nn.Linear(100, 50),  nn.ReLU(),
            nn.Linear(50, 2),    # [steering, throttle]
        )
    def forward(self, x):
        return self.head(self.encoder(x))
```

### 6.2 Editable hyperparameters (lines ~50–80)

```python
# ── Editable hyperparameters ────────────────────────────────────────────────
BATCH_SIZE      = 64
LR              = 1e-3
WEIGHT_DECAY    = 1e-4
OPTIMIZER       = "adam"        # "adam" | "sgd" | "adamw"
AUGMENT         = True
STEERING_WEIGHT = 1.0           # relative loss weight for steering
THROTTLE_WEIGHT = 0.5           # relative loss weight for throttle
# ────────────────────────────────────────────────────────────────────────────
```

### 6.3 Training loop

Mirrors the AutoResearch training loop:

```python
start = time.time()
while time.time() - start < TIME_BUDGET:
    for batch in train_loader:
        # forward / backward / step
        ...
    # lightweight mid-run logging (optional)

val_mse = evaluate_mse(model, val_loader, device)
print(f"val_mse: {val_mse:.6f}")
print(f"peak_vram_mb: {torch.cuda.max_memory_allocated() // 2**20}")
```

### 6.4 Output format

```
---
val_mse:          0.012345
training_seconds: 300.1
total_seconds:    315.8
peak_vram_mb:     1240.0
num_epochs:       42
num_samples:      12800
num_params_k:     427
```

---

## 7. `program_donkey.md` — Agent Instructions

The agent instructions file has the same structure as `program.md` with driving-specific details:

```markdown
# autoresearch — DonkeyCar edition

## Setup
1. Agree on a run tag and create branch `autoresearch/donkey-<tag>`.
2. Read prepare_donkey.py and train_donkey.py in full.
3. Verify tub data exists at ~/donkeycar/data/.  If not, ask the human to
   collect driving data with `donkey drive`.
4. Run baseline: `python train_donkey.py > run.log 2>&1`
5. Initialise results_donkey.tsv with header row.

## Goal
Minimise val_mse (steering MSE on held-out laps).

## Experiment loop
Loop forever:
1. Modify train_donkey.py (model, optimiser, augmentation, …).
2. git commit.
3. python train_donkey.py > run.log 2>&1
4. grep "^val_mse:\|^peak_vram_mb:" run.log
5. Log to results_donkey.tsv.
6. If improved: keep.  If not: git reset --hard HEAD~1.

## Deployment check (optional, manual)
After a significant improvement, export and test on the real car:
  python export_donkey.py   # produces autopilot.onnx
  # copy to Raspberry Pi and run donkey drive --model autopilot.onnx
```

---

## 8. Deployment Pipeline

Once the agent finds a good model, it needs to run on the car's onboard computer:

```
┌─────────────────────────────────────────────────────────┐
│                    Training machine (GPU)                │
│  train_donkey.py  ──►  best_model.pth                   │
│  export_donkey.py ──►  autopilot.onnx  (or TorchScript) │
└───────────────────────────┬─────────────────────────────┘
                            │ scp / rsync
┌───────────────────────────▼─────────────────────────────┐
│                  Raspberry Pi 4 / Jetson Nano            │
│  donkeycar runtime  loads  autopilot.onnx               │
│  camera frame ──► model ──► (steering, throttle) ──► car│
└─────────────────────────────────────────────────────────┘
```

`export_donkey.py` (a small fixed utility, not agent-editable):

```python
import torch, sys
from train_donkey import DonkeyNet

model = DonkeyNet()
model.load_state_dict(torch.load("best_model.pth", map_location="cpu"))
model.eval()
dummy = torch.zeros(1, 3, 120, 160)
torch.onnx.export(model, dummy, "autopilot.onnx",
                  input_names=["image"], output_names=["controls"],
                  opset_version=17)
print("Exported autopilot.onnx")
```

On the Raspberry Pi the DonkeyCar `manage.py` is configured with a custom
`ONNXPilot` part that wraps the ONNX model using `onnxruntime`.

---

## 9. Hardware Recommendations

| Component | Minimum | Recommended |
|---|---|---|
| Training GPU | GTX 1060 6 GB | RTX 3090 / A100 |
| Car computer | Raspberry Pi 4 (4 GB) | Jetson Orin Nano |
| Camera | 160×120 USB / CSI | Wide-angle CSI |
| Track | Indoor carpet loop | DIY Robocars outdoor |

For a laptop-only setup (no GPU): lower `IMG_H×IMG_W` to `66×200`
(NVIDIA DAVE-2 style: height=66, width=200) and reduce the CNN depth — smaller images train faster on CPU.

---

## 10. Data Collection Tips

The quality of the training tub is the biggest factor in model performance:

1. **Drive smoothly** — abrupt steering changes are hard to learn.
2. **Collect recovery data** — deliberately drive toward the edge, then
   correct.  This teaches the model to recover, not just follow the centre.
3. **Multiple lighting conditions** — collect data at different times of day.
4. **Augmentation compensates** — horizontal flip + mirrored steering
   doubles effective dataset size and improves left/right symmetry.
5. **Aim for ≥ 10 k frames** before serious training.

---

## 11. Suggested Experiment Ideas for the Agent

The following are good starting experiments (analogous to the GPT
hyperparameter sweeps in AutoResearch):

| Category | Experiment idea |
|---|---|
| **Architecture** | Add batch normalisation after each conv layer |
| **Architecture** | Replace CNN encoder with MobileNetV2 (pretrained ImageNet) |
| **Architecture** | Add recurrent head (GRU) to capture temporal context |
| **Architecture** | Replace regression head with categorical bins (like DonkeyCar `Categorical` model) |
| **Optimiser** | Switch from Adam to AdamW with weight decay |
| **Optimiser** | Try cosine LR schedule with warm-up |
| **Optimiser** | Try SGD + momentum |
| **Data** | Tune augmentation strength (flip probability, brightness range) |
| **Data** | Add Gaussian noise to images as regularisation |
| **Loss** | Upweight steering in the loss (most important output) |
| **Loss** | Try Huber loss instead of MSE (more robust to outliers) |
| **Regularisation** | Add dropout to the fully-connected head |

---

## 12. Implementation Checklist

- [ ] Write `prepare_donkey.py` (tub loader, augmentation, `evaluate_mse`)
- [ ] Write `train_donkey.py` (baseline `DonkeyNet`, training loop, output format)
- [ ] Write `program_donkey.md` (agent instructions)
- [ ] Write `export_donkey.py` (ONNX export utility)
- [ ] Collect ≥ 10 k frames of training data on the target track
- [ ] Run baseline experiment and record in `results_donkey.tsv`
- [ ] Kick off autonomous agent loop
- [ ] (Optional) Set up a sim-based validation using the DonkeyCar simulator
  to replace wall-clock lap tests with a fully automated metric

---

## 13. Stretch Goal: Simulator-Based Metric

The DonkeyCar project ships a
[Gym environment](https://docs.donkeycar.com/guide/deep_learning/simulator/)
backed by a Unity simulator.  A more ambitious version of this plan would:

1. Replace `evaluate_mse` with `evaluate_lap_time` — run the model in the
   simulator and return the fastest lap time (lower = better).
2. This gives the agent a **deployable** metric rather than a proxy metric.
3. Downside: simulator rollout is slower than a forward pass over a fixed
   validation set, so the 5-minute budget buys fewer training steps.

This is left as a future extension; `val_mse` on a held-out tub is a
sufficient proxy to get started.
