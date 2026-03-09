# Autoresearch — Detailed Code Explanation

This document explains every component of the codebase in detail.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [File Structure](#file-structure)
3. [`prepare.py` — Data & Evaluation](#preparepy--data--evaluation)
   - [Constants](#constants-fixed-do-not-modify)
   - [Data Download](#data-download)
   - [Tokenizer Training](#tokenizer-training)
   - [Runtime Utilities](#runtime-utilities)
   - [Evaluation Metric](#evaluation-metric-evaluate_bpb)
4. [`train.py` — Model, Optimizer, Training Loop](#trainpy--model-optimizer-training-loop)
   - [GPT Model](#gpt-model)
   - [MuonAdamW Optimizer](#muonadamw-optimizer)
   - [Hyperparameters](#hyperparameters)
   - [Setup Phase](#setup-phase)
   - [Learning Rate & Schedule Functions](#learning-rate--schedule-functions)
   - [Training Loop](#training-loop)
5. [`program.md` — Agent Instructions](#programmd--agent-instructions)
6. [Workflow Summary](#workflow-summary)

---

## Project Overview

AutoResearch is an autonomous AI research framework: an LLM agent repeatedly
edits `train.py`, runs a 5-minute training experiment, evaluates the result,
and keeps or discards the change — overnight, indefinitely.

The evaluation metric is **val_bpb** (validation bits per byte). Lower is better.
It is vocabulary-size independent, so experiments with different tokenizers or
vocab sizes are directly comparable.

The agent never touches `prepare.py` (data prep & evaluation) or `program.md`
(human-written instructions), only `train.py`.

---

## File Structure

```
autoresearch/
├── prepare.py      — Fixed constants, one-time data prep, runtime utilities
│                     (dataloader, tokenizer). DO NOT MODIFY.
├── train.py        — GPT model, MuonAdamW optimizer, training loop.
│                     This is the file the agent edits.
├── program.md      — Human-written instructions for the AI agent.
├── EXPLAIN.md      — This file.
├── analysis.ipynb  — Jupyter notebook for analyzing results.tsv
├── pyproject.toml  — Python dependencies (managed with uv)
└── uv.lock         — Locked dependency versions
```

---

## `prepare.py` — Data & Evaluation

### Constants (fixed, do not modify)

```python
MAX_SEQ_LEN = 2048        # Context window in tokens
TIME_BUDGET = 300         # Training time budget: 5 minutes (wall clock)
EVAL_TOKENS = 40 * 524288 # Number of tokens used for validation evaluation
VOCAB_SIZE  = 8192        # BPE vocabulary size
```

These constants are imported by `train.py`. Changing them would break
comparability between experiment runs.

Other configuration:

```python
CACHE_DIR   = ~/.cache/autoresearch/    # Root for all cached data
DATA_DIR    = ~/.cache/autoresearch/data/
TOKENIZER_DIR = ~/.cache/autoresearch/tokenizer/
BASE_URL    = "https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle/..."
MAX_SHARD   = 6542  # Total available parquet shards in the dataset
VAL_SHARD   = 6542  # Pinned validation shard (always shard_06542.parquet)
```

---

### Data Download

**`download_single_shard(index)`**

Downloads one `.parquet` file by index from HuggingFace with up to 5 retries
(exponential backoff). Writes to a `.tmp` file first, then renames atomically.
Returns `True` on success.

**`download_data(num_shards, download_workers=8)`**

Downloads `num_shards` training shards in parallel (using `multiprocessing.Pool`)
plus the pinned validation shard. Skips shards already on disk. Running with
`--num-shards -1` downloads all 6 542 shards (~200 GB).

---

### Tokenizer Training

**`train_tokenizer()`**

1. Reads up to 1 billion characters from training shards using `text_iterator()`.
2. Trains a BPE tokenizer via `rustbpe` with a GPT-4-style split pattern and
   vocabulary size `VOCAB_SIZE - 4` (leaving room for 4 special tokens).
3. Wraps the result in a `tiktoken.Encoding` and pickles it to
   `tokenizer.pkl`.
4. Builds a `token_bytes` tensor: for each token ID, stores the UTF-8 byte
   length of the decoded string (0 for special tokens). Saved as
   `token_bytes.pt`; used by `evaluate_bpb()` to count bytes.
5. Runs a round-trip sanity check.

The BOS (beginning-of-sequence) token is `<|reserved_0|>`.

**`Tokenizer` class**

Thin wrapper around the pickled `tiktoken.Encoding`. Exposes:
- `encode(text_or_list, prepend=None)` — encodes a string or batch of strings,
  optionally prepending the BOS token ID.
- `decode(ids)` — decodes token IDs back to a string.
- `get_vocab_size()` — returns `enc.n_vocab`.
- `get_bos_token_id()` — returns the integer BOS token ID.

---

### Runtime Utilities

**`_document_batches(split)`**

Infinite iterator over batches of raw document strings from parquet files.
The training split excludes `shard_06542.parquet`; the validation split uses
only that shard. Yields `(list_of_texts, epoch)`.

**`make_dataloader(tokenizer, B, T, split, buffer_size=1000)`**

Yields `(inputs, targets, epoch)` tensors of shape `(B, T)` on GPU.

Uses **BOS-aligned best-fit bin packing**:
- Each row starts with a BOS token.
- A buffer of `buffer_size` tokenized documents is maintained.
- For each row position, the largest document that fits the remaining space is
  chosen ("best fit"). If no document fits, the shortest document is truncated
  to fill exactly.
- Result: 100% token utilization — no padding wasted.

Pre-allocates pinned CPU memory and GPU memory for zero-copy transfers
(`non_blocking=True`).

---

### Evaluation Metric: `evaluate_bpb()`

```python
@torch.no_grad()
def evaluate_bpb(model, tokenizer, batch_size):
```

**DO NOT MODIFY** — this is the fixed comparison metric.

Computes **bits per byte (BPB)** on the pinned validation shard:

1. Runs `EVAL_TOKENS // (batch_size * MAX_SEQ_LEN)` forward passes with
   `reduction='none'` to get per-token cross-entropy loss (in nats).
2. Looks up the UTF-8 byte length of each target token via the `token_bytes`
   tensor. Special tokens (byte length 0) are masked out from both the
   numerator and denominator.
3. Returns `total_nats / (ln(2) * total_bytes)`.

Why BPB is useful: it is independent of vocabulary size. A model with a larger
vocabulary produces fewer tokens per byte, so cross-entropy per token would
naturally be lower even with no improvement in understanding — BPB corrects for
this.

---

## `train.py` — Model, Optimizer, Training Loop

### GPT Model

#### Architecture Overview

```
Input tokens (B, T)
  │
  ▼
wte: nn.Embedding(vocab_size, n_embd)   — token embedding lookup
  │
RMSNorm                                  — normalize input
  │
  ├── save as x0                         — skip connection source
  ▼
┌─────────────────────────────────────────────────────┐
│  Layer i (repeated n_layer times):                   │
│                                                      │
│  x = resid_lambdas[i] * x + x0_lambdas[i] * x0     │
│                                                      │
│      ↑ per-layer learnable scalars; allow the model  │
│        to blend the residual stream with the raw     │
│        post-embedding signal at each depth           │
│                                                      │
│  x = x + CausalSelfAttention(RMSNorm(x), ve, ...)   │
│  x = x + MLP(RMSNorm(x))                            │
└─────────────────────────────────────────────────────┘
  │
RMSNorm
  │
lm_head: nn.Linear(n_embd, vocab_size, bias=False)
  │
softcap: logits = 15 * tanh(logits / 15)   — prevents logit explosion
  │
cross-entropy loss (or raw logits)
```

---

#### `GPTConfig` — Model Shape Parameters

| Field | Default | Description |
|-------|---------|-------------|
| `sequence_len` | 2048 | Context window length (tokens) |
| `vocab_size` | 32768 | Vocabulary size (set from tokenizer at runtime) |
| `n_layer` | 12 | Number of transformer blocks (primary size knob) |
| `n_head` | 6 | Number of query attention heads |
| `n_kv_head` | 6 | Number of key/value heads. Equal to `n_head` = standard MHA; smaller = GQA |
| `n_embd` | 768 | Hidden dimension (must equal `n_head * head_dim`) |
| `window_pattern` | `"SSSL"` | Per-layer window type: `'L'` = full context, `'S'` = half context. Cycles over layers; last layer always forced to `'L'`. |

At runtime `n_layer` is set to `DEPTH`, and `n_embd` is derived as
`round_up(DEPTH * ASPECT_RATIO, HEAD_DIM)`.

---

#### `norm(x)` — RMS Normalization

```python
def norm(x):
    return F.rms_norm(x, (x.size(-1),))
```

Normalizes `x` to unit root-mean-square along its last dimension. Used as
pre-norm before every attention and MLP sub-layer, and on the input embedding
and final hidden state.

---

#### `has_ve(layer_idx, n_layer)` — Value Embedding Selector

```python
def has_ve(layer_idx, n_layer):
    return layer_idx % 2 == (n_layer - 1) % 2
```

Returns `True` for every other layer (the same parity as the last layer).
Determines which layers get a Value Embedding table.

---

#### `apply_rotary_emb(x, cos, sin)` — Rotary Position Embeddings (RoPE)

Rotates pairs of channels in `x` (shape `(B, T, n_head, head_dim)`) using
precomputed cosine/sine frequency tables.

The rotation is applied to both queries and keys before the dot-product
attention, which causes the attention score between position `m` and position
`n` to depend only on their relative distance `m - n`. This encodes position
without adding absolute positional embeddings to the residual stream.

The rotation formula per pair `(x1, x2)`:
```
y1 = x1 * cos + x2 * sin
y2 = -x1 * sin + x2 * cos
```

---

#### `CausalSelfAttention` — Multi-Head Attention with RoPE, QK-Norm, and Value Embeddings

**Parameters:**
- `c_q`, `c_k`, `c_v`: linear projections to query, key, and value spaces.
  Query is `n_head`-wide; key/value are `n_kv_head`-wide (GQA-ready).
- `c_proj`: output projection back to `n_embd`.
- `ve_gate`: small linear layer (32 input channels → `n_kv_head` outputs) that
  computes a per-head sigmoid gate for the Value Embedding residual.
  Only present on Value Embedding layers.

**Forward pass:**
1. Project input to Q, K, V.
2. If this layer has a Value Embedding (`ve` is not None):
   - Reshape `ve` to `(B, T, n_kv_head, head_dim)`.
   - Compute gate = `2 * sigmoid(ve_gate(x[..., :32]))` (range [0, 2],
     neutral at 1.0 because `sigmoid(0)=0.5`).
   - Add `gate * ve` to V.
3. Apply RoPE to Q and K.
4. Apply RMSNorm to Q and K (QK-norm — stabilizes attention logit scale).
5. Run FlashAttention-3 with causal masking and the layer's window size.
6. Output projection.

**Window sizes** are computed by `_compute_window_sizes()`: each layer character
in `WINDOW_PATTERN` maps to `(sequence_len, 0)` for `'L'` or
`(sequence_len//2, 0)` for `'S'`. The last layer is always `'L'`.

**Value Embeddings (ResFormer-style):** Token-level residuals in the value
space allow direct token-identity information to flow through V without going
through the Q–K attention bottleneck. This is particularly beneficial in early
layers.

---

#### `MLP` — Feed-Forward Block

```python
class MLP(nn.Module):
    # c_fc:   Linear(n_embd → 4*n_embd)
    # c_proj: Linear(4*n_embd → n_embd)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()   # ReLU² activation
        x = self.c_proj(x)
        return x
```

**ReLU²** (squaring the ReLU output) provides a beneficial inductive bias:
- Sparse activations (zero for negative inputs, like ReLU).
- Super-linear growth (quadratic near the boundary, unlike linear ReLU).
- Simpler and faster to compute than GELU or SwiGLU.

Output projection `c_proj` is initialized to zero, so every block starts as an
identity-passthrough, enabling stable training at any depth.

---

#### `Block` — Transformer Block

```python
class Block(nn.Module):
    def forward(self, x, ve, cos_sin, window_size):
        x = x + self.attn(norm(x), ve, cos_sin, window_size)
        x = x + self.mlp(norm(x))
        return x
```

Standard pre-norm residual transformer block. The per-layer skip-connection
blend (`resid_lambdas`, `x0_lambdas`) is applied by the parent `GPT.forward`,
not here, keeping `Block` a clean and reusable unit.

---

#### `GPT` — Full Language Model

**`__init__`:** Builds token embedding, a list of `Block`s, the unembedding
head, per-layer scalar parameters, Value Embedding tables for alternating
layers, and a precomputed RoPE buffer (10× context length for potential
length generalization).

**`init_weights()`:** Custom initialization:
- Token embedding: `N(0, 1)` (large std because input is normalized).
- lm_head: `N(0, 0.001)` (nearly-zero outputs at init).
- Q, K, V, MLP-fc weights: `Uniform(-s, s)` where `s = √3 / √n_embd`.
- Projection (output) weights: zero-initialized (identity blocks at start).
- `resid_lambdas`: initialized to 1.0 (pure residual).
- `x0_lambdas`: initialized to 0.1 (small skip contribution).
- `ve_gate` weights: zero-initialized (gate = sigmoid(0) * 2 = 1.0, neutral).
- Embeddings cast to bfloat16.

**`_precompute_rotary_embeddings(seq_len, head_dim)`:**
Computes `inv_freq = 1 / (10000 ^ (2i/head_dim))` for each frequency band,
then builds a `(1, seq_len, 1, head_dim//2)` table of cosines and sines in
bfloat16. Registered as a non-persistent buffer (not saved in checkpoints).

**`_compute_window_sizes(config)`:**
Maps each layer index to a `(window, 0)` tuple by cycling through
`window_pattern` characters. The last layer is always forced to `(sequence_len, 0)`.

**`estimate_flops()`:**
Estimates FLOPs per token for throughput / MFU calculation:
- Standard formula: `6 * (non-embedding params)` for linear layers.
- Per-layer attention FLOPs: `12 * n_head * head_dim * effective_window`.

**`num_scaling_params()`:**
Returns a dict of parameter counts broken out by type (wte, lm_head, matrices,
value_embeds, scalars, total).

**`setup_optimizer()`:** Constructs `MuonAdamW` with separate parameter groups:
- lm_head → AdamW with `unembedding_lr`
- wte → AdamW with `embedding_lr`
- value_embeds → AdamW with `embedding_lr`
- resid_lambdas → AdamW with very small LR (`scalar_lr * 0.01`)
- x0_lambdas → AdamW with `scalar_lr` and slower β₁
- Each unique weight shape in transformer blocks → Muon group

All LRs are scaled by `1/√(model_dim/768)` so the same nominal LR works across
model sizes.

**`forward(idx, targets=None, reduction='mean')`:**
1. Token embedding lookup + RMSNorm → `x`, save as `x0`.
2. For each layer `i`:
   - `x = resid_lambdas[i] * x + x0_lambdas[i] * x0`
   - Retrieve Value Embedding `ve` if applicable.
   - `x = block(x, ve, cos_sin, window_sizes[i])`
3. RMSNorm → lm_head → `logits.float()`.
4. Logit soft-cap: `15 * tanh(logits / 15)`.
5. If `targets` provided: return `cross_entropy(logits, targets, reduction=reduction)`.
6. Otherwise: return `logits`.

---

### MuonAdamW Optimizer

A combined optimizer that dispatches different update rules by parameter type.

#### AdamW step (`_step_adamw`)

Standard AdamW with bias correction. Uses fused, `torch.compile`-compatible
scalar tensors on CPU to avoid recompilation when hyperparameters change.

#### Muon step (`_step_muon` + `muon_step_fused`)

Applied to all 2D weight matrices (attention Q/K/V/proj and MLP fc/proj).
All matrices of the same shape are updated in a single batched operation.

The step proceeds in four stages:

1. **Nesterov momentum**
   ```
   momentum_buffer = lerp(momentum_buffer, grad, 1-β)
   g = lerp(grad, momentum_buffer, β)          # lookahead
   ```
   Smoother gradient estimates than heavy-ball momentum.

2. **Polar Express orthogonalization**
   Computes the polar factor (nearest orthogonal matrix) of `g` using a
   polynomial Newton iteration — specifically the coefficients in
   `polar_express_coeffs`:
   ```
   X = g / (||g|| * 1.02 + ε)    # normalize
   for a, b, c in polar_express_coeffs[:ns_steps]:
       if tall (rows > cols):  A = Xᵀ X;  X = a*X + X @ (b*A + c*A²)
       else:                    A = X Xᵀ;  X = a*X + (b*A + c*A²) @ X
   g = X
   ```
   Each iteration converges the singular values toward 1.0. After `ns_steps=5`
   iterations the result closely approximates the polar factor. This scales
   updates uniformly across singular value directions, preventing large
   singular values from dominating.

3. **NorMuon variance reduction**
   Per-row (or per-column for wide matrices) adaptive rescaling, similar to
   Adam's second-moment denominator but applied to the already-orthogonalized
   gradient. Stored in `second_momentum_buffer` with exponential moving average.

4. **Cautious weight decay + parameter update**
   ```
   mask = (g * params) >= 0      # agree in sign
   params -= lr * g + lr * wd * params * mask
   ```
   Weight decay is only applied where the decay direction agrees with the
   gradient direction, preventing regularization from fighting the update.

#### LR scaling

The Muon LR for each group is additionally scaled by
`max(1.0, rows/cols)^0.5` to account for non-square matrices.

---

### Hyperparameters

All hyperparameters are plain module-level constants at the top of the
"Hyperparameters" section in `train.py`. No CLI flags are needed — the agent
edits them directly.

```python
# ── Architecture ────────────────────────────────────────────────────────────
ASPECT_RATIO  = 64       # model_dim = DEPTH * ASPECT_RATIO
                         # (rounded up to next HEAD_DIM multiple)
HEAD_DIM      = 128      # attention head dimension; n_head = n_embd // HEAD_DIM
WINDOW_PATTERN = "SSSL"  # attention window per layer: 'L'=full, 'S'=half context

# ── Optimization ─────────────────────────────────────────────────────────────
TOTAL_BATCH_SIZE = 2**19 # ~524K tokens per optimizer step (gradient-accumulated)
EMBEDDING_LR    = 0.6    # AdamW LR for token embeddings & value embeddings
UNEMBEDDING_LR  = 0.004  # AdamW LR for lm_head
MATRIX_LR       = 0.04   # Muon LR for 2D weight matrices
SCALAR_LR       = 0.5    # AdamW LR for per-layer scalars
WEIGHT_DECAY    = 0.2    # Muon cautious weight decay (linearly decays to 0)
ADAM_BETAS      = (0.8, 0.95)  # Adam β₁, β₂
WARMUP_RATIO    = 0.0    # fraction of TIME_BUDGET for LR warm-up (0 = none)
WARMDOWN_RATIO  = 0.5    # fraction of TIME_BUDGET for LR warm-down
FINAL_LR_FRAC   = 0.0   # final LR = initial LR * FINAL_LR_FRAC

# ── Model size ────────────────────────────────────────────────────────────────
DEPTH           = 8      # number of transformer layers (primary size knob)
DEVICE_BATCH_SIZE = 128  # micro-batch size per GPU; reduce if OOM
```

**Derived values (computed automatically):**
```python
model_dim  = round_up(DEPTH * ASPECT_RATIO, HEAD_DIM)
n_head     = model_dim // HEAD_DIM
grad_accum = TOTAL_BATCH_SIZE // (DEVICE_BATCH_SIZE * MAX_SEQ_LEN)
```

---

### Setup Phase

Executed once at module load (before the training loop):

1. Set random seeds (`torch.manual_seed(42)`, `torch.cuda.manual_seed(42)`).
2. Set float32 matmul precision to `"high"` (enables TF32 on Ampere+).
3. Load the tokenizer from disk.
4. Build `GPTConfig` via `build_model_config(DEPTH)`.
5. Instantiate `GPT` on the meta device, move to CUDA, call `init_weights()`.
6. Compute and print parameter counts and estimated FLOPs/token.
7. Assert `TOTAL_BATCH_SIZE` is divisible by `DEVICE_BATCH_SIZE * MAX_SEQ_LEN`.
8. Create the `MuonAdamW` optimizer; store `initial_lr` on each group.
9. Compile the model with `torch.compile(dynamic=False)`.
10. Create the training dataloader and prefetch the first batch.

---

### Learning Rate & Schedule Functions

**`build_model_config(depth)`**

Computes `model_dim = round_up(depth * ASPECT_RATIO, HEAD_DIM)`,
`n_head = model_dim // HEAD_DIM`, and returns a `GPTConfig`.

**`get_lr_multiplier(progress)`**

Three-phase schedule based on `progress = elapsed_training_time / TIME_BUDGET`:

| Phase | Condition | Multiplier |
|-------|-----------|------------|
| Warm-up | `[0, WARMUP_RATIO)` | Linear ramp `0 → 1` |
| Constant | `[WARMUP_RATIO, 1 - WARMDOWN_RATIO)` | `1.0` |
| Warm-down | `[1 - WARMDOWN_RATIO, 1]` | Linear decay `1 → FINAL_LR_FRAC` |

Applied as `group["lr"] = group["initial_lr"] * get_lr_multiplier(progress)`.

**`get_muon_momentum(step)`**

Linearly ramps Muon momentum from `0.85` to `0.95` over the first 300 steps,
then holds at `0.95`. Lower momentum at the start stabilizes the noisy early
gradient estimates.

**`get_weight_decay(progress)`**

Returns `WEIGHT_DECAY * (1 - progress)`. Decays weight decay to zero by end of
training, allowing the model to converge tightly to the loss minimum without
regularization interference.

---

### Training Loop

```
t_start_training = now()
step = 0

while True:
    ┌─────────────────────────────────────────────────────────┐
    │  Gradient accumulation (grad_accum_steps micro-steps)   │
    │  ┌──────────────────────────────────────────────────┐   │
    │  │  with bfloat16 autocast:                         │   │
    │  │      loss = model(x, y)                          │   │
    │  │  (loss / grad_accum_steps).backward()            │   │
    │  │  prefetch next (x, y, epoch)                     │   │
    │  └──────────────────────────────────────────────────┘   │
    └─────────────────────────────────────────────────────────┘

    progress = min(total_training_time / TIME_BUDGET, 1.0)
    update LR for all param groups
    update Muon momentum & weight decay
    optimizer.step(); model.zero_grad()

    if train_loss > 100: print("FAIL"); exit(1)   # loss explosion

    if step > 10:
        total_training_time += step_wall_time     # exclude warmup steps

    # Logging (overwriting same line with \r):
    print step, smoothed_loss, LR multiplier, tok/sec, MFU, epoch, remaining

    if step == 0:
        gc.collect(); gc.freeze(); gc.disable()   # prevent GC stalls

    step += 1

    if step > 10 and total_training_time >= TIME_BUDGET:
        break

# ── Final evaluation ──────────────────────────────────────────────────────
model.eval()
val_bpb = evaluate_bpb(model, tokenizer, DEVICE_BATCH_SIZE)

# ── Print summary ─────────────────────────────────────────────────────────
print("---")
print(f"val_bpb:          {val_bpb:.6f}")
print(f"training_seconds: {total_training_time:.1f}")
...
```

**Key implementation details:**

- The first 10 steps are excluded from time tracking to absorb
  `torch.compile()` JIT compilation latency.
- `total_training_time` only counts wall-clock time after step 10, making the
  budget fair even on slow-to-compile configurations.
- Python's garbage collector is frozen after step 0 because GC pauses can
  cause ~500 ms stalls that waste training budget.
- The EMA of the training loss uses `β=0.9` with bias correction to report a
  smooth, reliable loss estimate.
- MFU (model FLOPs utilization) is computed as
  `FLOPs_per_token * batch_size / step_time / H100_BF16_PEAK_FLOPS`.
  Peak H100 bfloat16 throughput is 989.5 TFLOPS.

---

## `program.md` — Agent Instructions

Written by the human. Tells the AI agent how to conduct experiments:

1. **Setup**: create a git branch, read `prepare.py`, verify data, initialize
   `results.tsv`.
2. **Experiment loop** (runs indefinitely until manually stopped):
   - Read the current git state.
   - Modify `train.py` with one experimental change.
   - `git commit`.
   - `uv run train.py > run.log 2>&1`
   - Parse results: `grep "^val_bpb:\|^peak_vram_mb:" run.log`
   - Log to `results.tsv`: `commit | val_bpb | memory_gb | status | description`
   - If `val_bpb` improved: keep; otherwise `git revert`.
3. **Constraints**:
   - Do not modify `prepare.py` or `evaluate_bpb()`.
   - Do not install new packages.
   - Kill runs exceeding 10 minutes.
   - Prefer simpler changes when performance is equal.

---

## Workflow Summary

```
Human writes program.md
        │
        ▼
Agent reads program.md + prepare.py
        │
        ▼
uv run prepare.py       (one-time: downloads data, trains tokenizer)
        │
        ▼
┌───────────────────────────────────────────────┐
│  LOOP (never stops):                          │
│                                               │
│  1. Modify train.py                           │
│  2. git commit                                │
│  3. uv run train.py > run.log 2>&1            │
│     (~5 min training + ~25s startup/eval)     │
│  4. grep val_bpb from run.log                 │
│  5. Append to results.tsv                     │
│  6. Improved? Keep : git revert               │
│                                               │
│  ≈12 experiments/hour                        │
│  ≈100 experiments while you sleep            │
└───────────────────────────────────────────────┘
        │
        ▼
Human reviews results.tsv + improved train.py
```

### Output format

Each run prints a final summary block that the agent parses:

```
---
val_bpb:          0.997900    # ← primary metric (lower is better)
training_seconds: 300.1       # should be ≈300 (5 min)
total_seconds:    325.9       # includes startup and evaluation
peak_vram_mb:     45060.2     # GPU memory high watermark
mfu_percent:      39.80       # model FLOPs utilization
total_tokens_M:   499.6       # millions of training tokens seen
num_steps:        953         # optimizer steps completed
num_params_M:     50.3        # model parameter count
depth:            8           # DEPTH hyperparameter used
```
