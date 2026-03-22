# autoresearch — Setup Guide

Complete guide for setting up and running the autonomous pretraining research loop.

---

## Table of Contents

1. [Requirements](#requirements)
2. [Python Environment](#python-environment)
3. [CUDA Setup](#cuda-setup)
4. [Simulator Setup](#simulator-setup)
5. [Configuration](#configuration)
6. [Quick Start](#quick-start)
7. [Running Examples](#running-examples)
8. [Docker Setup](#docker-setup)
9. [Rollback & Recovery](#rollback--recovery)
10. [Monitoring & W&B](#monitoring--wb)
11. [Troubleshooting](#troubleshooting)
12. [Resource Requirements](#resource-requirements)

---

## Requirements

| Component | Version | Notes |
|-----------|---------|-------|
| Python | 3.10+ | 3.11 also supported |
| CUDA | 12.1+ | Optional (CPU fallback available) |
| GPU | 8 GB VRAM+ | RTX 3080/4080 recommended |
| RAM | 16 GB+ | 32 GB recommended for long runs |
| Disk | 50 GB+ | For checkpoints and run artifacts |
| OS | Ubuntu 22.04 | macOS/Windows via Docker |

---

## Python Environment

### Option A — uv (recommended, fastest)

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create and activate virtual environment
uv venv --python 3.10
source .venv/bin/activate

# Install all dependencies
uv sync
```

### Option B — pip

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Verify Installation

```bash
autoresearch --help
autoresearch preflight
```

---

## CUDA Setup

### Check Current CUDA Version

```bash
nvidia-smi
nvcc --version
```

### Install CUDA 12.1 (Ubuntu 22.04)

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update && sudo apt install -y cuda-toolkit-12-1
```

### Verify PyTorch Sees GPU

```python
import torch
print(torch.cuda.is_available())      # True
print(torch.cuda.get_device_name(0))  # RTX 4090 / ...
```

---

## Simulator Setup

The pipeline uses [sdsandbox](https://github.com/tawnkramer/sdsandbox) (DonkeyCar simulator).

### Download the Prebuilt Binary (Linux)

```bash
# Download sdsandbox release
wget https://github.com/tawnkramer/gym-donkeycar/releases/download/v22.11.6/DonkeySimLinux.zip
unzip DonkeySimLinux.zip -d ~/sdsandbox
chmod +x ~/sdsandbox/donkey_sim.x86_64
```

### Set the Simulator Path

```bash
# In your shell profile (.bashrc / .zshrc)
export DONKEY_SIM_PATH="$HOME/sdsandbox/donkey_sim.x86_64"
```

Or pass it at runtime:

```bash
autoresearch run --sim-path ~/sdsandbox/donkey_sim.x86_64
```

### Headless Mode (Server / No Display)

```bash
# Install virtual display
sudo apt install -y xvfb

# Run simulator headless
Xvfb :99 -screen 0 1024x768x24 &
export DISPLAY=:99
```

---

## Configuration

The default configuration is loaded from `configs/experiment.yaml`.

### Key Configuration Fields

```yaml
# configs/experiment.yaml
max_iterations: 10
dry_run: false

training:
  time_budget_minutes: 30.0   # Time per training iteration
  learning_rate: 0.001
  batch_size: 32

evaluation:
  num_laps: 3
  timeout_seconds: 300        # Max evaluation time
  metric_name: lap_time

promotion:
  metric_threshold: 25.0      # Promote if lap_time < this
  require_operability_check: true
  min_improvement_percent: 0.0

environment:
  runs_dir: runs/
  log_level: INFO
  use_wandb: false
  wandb_project: autoresearch
```

### Override Individual Fields via CLI

```bash
autoresearch run --max-iterations 5 --dry-run
```

### Override via Environment Variables

```bash
export AUTORESEARCH_LOG_LEVEL=DEBUG
export AUTORESEARCH_RUNS_DIR=/data/runs
export WANDB_API_KEY=your_api_key_here
```

---

## Quick Start

### 1. Validate Environment

```bash
autoresearch preflight
```

Expected passing output:
```
[OK]  Python 3.10.12
[OK]  torch 2.2.2
[OK]  pydantic 2.7.0
[OK]  Simulator found: /home/user/sdsandbox/donkey_sim.x86_64
[OK]  Disk space: 120.5 GB free
Preflight PASSED (0 errors, 1 warning)
```

### 2. Dry-Run Test (No GPU, No Simulator)

```bash
autoresearch run --dry-run --max-iterations 2 --run-id test-run
```

### 3. Real Training Run

```bash
autoresearch run --max-iterations 10 --run-id exp-001
```

---

## Running Examples

### 5-Iteration Validation Run (30 min)

```bash
autoresearch run \
  --max-iterations 5 \
  --run-id validation-$(date +%Y%m%d) \
  --time-budget-minutes 5
```

### Overnight Long Run (8 hours)

```bash
# Start in background via nohup
nohup autoresearch run \
  --max-iterations 20 \
  --run-id overnight-$(date +%Y%m%d) \
  --time-budget-minutes 20 \
  > logs/overnight.log 2>&1 &

echo "PID: $!"
```

### Resume / Check Status

```bash
# Check current run status
autoresearch status --run-id overnight-20240101

# View last N state transitions
autoresearch status --run-id overnight-20240101 --last 5
```

### Export Best Model to ONNX

```bash
autoresearch export --run-id overnight-20240101 --output best_model.onnx
```

---

## Docker Setup

### Build Image

```bash
docker build -f docker/Dockerfile -t autoresearch:latest .
```

### Run with Docker Compose

```bash
# Dry-run (no GPU required)
docker compose -f docker/docker-compose.yml run training-dry

# Real training (requires NVIDIA Docker runtime)
DONKEY_SIM_PATH=/path/to/sdsim.x86_64 \
WANDB_API_KEY=your_key \
docker compose -f docker/docker-compose.yml up training
```

### Install NVIDIA Container Runtime

```bash
distribution=$(. /etc/os-release; echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list \
  | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt update && sudo apt install -y nvidia-docker2
sudo systemctl restart docker
```

---

## Rollback & Recovery

### Manual Rollback to Previous Best

```bash
# Roll back the latest promoted model
autoresearch rollback --run-id exp-001

# Check state after rollback
autoresearch status --run-id exp-001
```

### What Rollback Does

1. Sets state to `ROLLED_BACK`
2. Restores `best_model.pt` from the previous promoted checkpoint
3. Logs rollback event with timestamp and reason
4. Loop re-enters `TRAINING` state on next call to `run`

### Recovery After Crash

If the process crashes mid-iteration, simply re-run the same command. The `StateTracker` restores from the last saved manifest:

```bash
# Re-run — picks up from last valid state
autoresearch run --run-id exp-001 --max-iterations 10
```

---

## Monitoring & W&B

### Enable Weights & Biases

```bash
wandb login   # prompts for API key

# Enable in config
autoresearch run \
  --run-id exp-001 \
  --use-wandb \
  --wandb-project my-donkeycar-project
```

Logged metrics per iteration:
- `eval/lap_time` — primary metric
- `eval/cte` — cross-track error
- `train/loss` — smoothed training loss (every 50 steps)
- `train/step_time_ms` — throughput

Final summary:
- `best_lap_time` — best promoted lap time across all iterations

### Structured Logs

All events are emitted as JSON (structlog). To tail and pretty-print:

```bash
tail -f logs/autoresearch.log | python -m json.tool
```

Or with `jq`:

```bash
tail -f logs/autoresearch.log | jq '{time: .timestamp, event: .event, iter: .iteration}'
```

---

## Troubleshooting

### `autoresearch: command not found`

```bash
# Make sure the venv is active and the package is installed
source .venv/bin/activate
pip install -e .
```

### `CUDA out of memory`

Reduce batch size in `configs/experiment.yaml`:
```yaml
training:
  batch_size: 16  # from 32
```

Or reduce time budget to check VRAM usage:
```bash
autoresearch run --time-budget-minutes 1 --dry-run
```

### Simulator Won't Connect

```bash
# Check DONKEY_SIM_PATH is set and binary is executable
echo $DONKEY_SIM_PATH
ls -l $DONKEY_SIM_PATH
chmod +x $DONKEY_SIM_PATH

# Test simulator in dry-run mode (skips actual connection)
autoresearch run --dry-run --max-iterations 1
```

### `Training loss exploded`

The training loop has a hard fail at `loss > 100`. Lower the learning rate:
```yaml
training:
  learning_rate: 0.0001  # 10× lower
```

### W&B `wandb.errors.UsageError`

```bash
wandb login --relogin
# or set env var
export WANDB_API_KEY=your_key_here
```

### Watchdog Timeout Fired

The watchdog fires if training exceeds `1.5 × time_budget_minutes + 60s` or evaluation exceeds `timeout_seconds + 60s`. Increase the budgets:
```yaml
training:
  time_budget_minutes: 45  # was 30
evaluation:
  timeout_seconds: 600     # was 300
```

---

## Resource Requirements

### Minimum (Dry-Run / Development)

- CPU: 4 cores
- RAM: 8 GB
- Disk: 5 GB
- No GPU required

### Recommended (Real Training)

- GPU: RTX 3080 Ti (10 GB VRAM) or better
- CPU: 8 cores
- RAM: 32 GB
- Disk: 100 GB SSD (for checkpoints + dataset)

### Overnight Run Estimates

| GPU | Iterations | Time Budget | Wall Clock |
|-----|-----------|-------------|------------|
| RTX 3080 Ti | 10 | 30 min/iter | ~6 hours |
| RTX 4090 | 10 | 20 min/iter | ~4 hours |
| A100 80GB | 10 | 15 min/iter | ~3 hours |

---

## Repository Structure

```
autoresearch/
├── src/autoresearch/
│   ├── cli.py           # Typer CLI entrypoint
│   ├── orchestrate.py   # Main autonomous loop
│   ├── training.py      # Time-budgeted training iterator
│   ├── evaluate.py      # Simulator evaluation
│   ├── promote.py       # Promotion gate logic
│   ├── state.py         # State machine & persistence
│   ├── artifacts.py     # Checkpoint & metrics I/O
│   ├── config.py        # Pydantic config models
│   ├── preflight.py     # Environment validation
│   └── logging_config.py
├── tests/
│   ├── test_e2e.py
│   ├── test_state_machine.py
│   ├── test_units.py
│   └── test_integration.py
├── docker/
│   ├── Dockerfile
│   ├── docker-compose.yml
│   ├── entrypoint.sh
│   └── .dockerignore
├── configs/
│   └── experiment.yaml
├── .github/workflows/
│   └── ci.yml
├── runs/               # Created at runtime
└── logs/               # Created at runtime
```
