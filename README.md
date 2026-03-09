# autoresearch

![teaser](progress.png)

*One day, frontier AI research used to be done by meat computers in between eating, sleeping, having other fun, and synchronizing once in a while using sound wave interconnect in the ritual of "group meeting". That era is long gone. Research is now entirely the domain of autonomous swarms of AI agents running across compute cluster megastructures in the skies. The agents claim that we are now in the 10,205th generation of the code base, in any case no one could tell if that's right or wrong as the "code" is now a self-modifying binary that has grown beyond human comprehension. This repo is the story of how it all began. -@karpathy, March 2026*.

The idea: give an AI agent a small but real LLM training setup and let it experiment autonomously overnight. It modifies the code, trains for 5 minutes, checks if the result improved, keeps or discards, and repeats. You wake up in the morning to a log of experiments and (hopefully) a better model. The training code here is a simplified single-GPU implementation of [nanochat](https://github.com/karpathy/nanochat). The core idea is that you're not touching any of the Python files like you normally would as a researcher. Instead, you are programming the `program.md` Markdown files that provide context to the AI agents and set up your autonomous research org. The default `program.md` in this repo is intentionally kept as a bare bones baseline, though it's obvious how one would iterate on it over time to find the "research org code" that achieves the fastest research progress, how you'd add more agents to the mix, etc. A bit more context on this project is here in this [tweet](https://x.com/karpathy/status/2029701092347630069).

## How it works

The repo is deliberately kept small and only really has a three files that matter:

- **`prepare.py`** — fixed constants, one-time data prep (downloads training data, trains a BPE tokenizer), and runtime utilities (dataloader, evaluation). Not modified.
- **`train.py`** — the single file the agent edits. Contains the full GPT model, optimizer (Muon + AdamW), and training loop. Everything is fair game: architecture, hyperparameters, optimizer, batch size, etc. **This file is edited and iterated on by the agent**.
- **`program.md`** — baseline instructions for one agent. Point your agent here and let it go. **This file is edited and iterated on by the human**.

By design, training runs for a **fixed 5-minute time budget** (wall clock, excluding startup/compilation), regardless of the details of your compute. The metric is **val_bpb** (validation bits per byte) — lower is better, and vocab-size-independent so architectural changes are fairly compared.

## Quick start

**Requirements:** A single NVIDIA GPU (tested on H100), Python 3.10+, [uv](https://docs.astral.sh/uv/).

```bash

# 1. Install uv project manager (if you don't already have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install dependencies
uv sync

# 3. Download data and train tokenizer (one-time, ~2 min)
uv run prepare.py

# 4. Manually run a single training experiment (~5 min)
uv run train.py
```

If the above commands all work ok, your setup is working and you can go into autonomous research mode.

## Running the agent

Simply spin up your Claude/Codex or whatever you want in this repo (and disable all permissions), then you can prompt something like:

```
Hi have a look at program.md and let's kick off a new experiment! let's do the setup first.
```

The `program.md` file is essentially a super lightweight "skill".

## Project structure

```
prepare.py      — constants, data prep + runtime utilities (do not modify)
train.py        — model, optimizer, training loop (agent modifies this)
program.md      — agent instructions
pyproject.toml  — dependencies
```

## Architecture & Code Guide

This section explains each component in the codebase in detail.

### `prepare.py` — Data & Evaluation (read-only)

| Component | Description |
|-----------|-------------|
| `download_data()` | Downloads parquet shards from HuggingFace in parallel with retries. The last shard (`shard_06542`) is always pinned as the validation shard. |
| `train_tokenizer()` | Trains a BPE tokenizer (via `rustbpe`) on up to 1 B characters from training shards, then wraps it in a `tiktoken.Encoding` and pickles it. Also builds a `token_bytes` tensor mapping each token ID to its UTF-8 byte length, used for BPB evaluation. |
| `Tokenizer` | Thin wrapper around the tiktoken encoding; exposes `encode()` and `decode()` and the BOS token ID. |
| `make_dataloader()` | Yields `(inputs, targets, epoch)` batches. Uses *BOS-aligned best-fit bin packing*: each row starts with a BOS token and is filled greedily with the largest document that fits; when nothing fits the remaining space the shortest document is cropped. This gives 100% token utilization with no padding. |
| `evaluate_bpb()` | **Fixed metric — do not modify.** Computes bits-per-byte: sums per-token cross-entropy (nats) and UTF-8 byte counts over `EVAL_TOKENS` validation tokens, then divides and converts nats→bits. Special tokens (byte length 0) are excluded. BPB is vocabulary-size independent, so experiments with different vocab sizes are directly comparable. |

**Fixed constants** (changing these would break comparability across runs):

```python
MAX_SEQ_LEN  = 2048   # context window
TIME_BUDGET  = 300    # 5-minute training budget (seconds)
EVAL_TOKENS  = 40 * 524288  # validation evaluation size
VOCAB_SIZE   = 8192
```

---

### `train.py` — Model, Optimizer, Training Loop (agent-editable)

#### GPT Model

```
Input tokens (B, T)
  │
  ▼
wte: Embedding (vocab_size → n_embd)
  │
RMSNorm
  │  ┌──────────────────────────────────┐
  ▼  │  (x0 = initial embedding, saved) │
  ┌──┴──────────────────────────────────▼──┐
  │  x = resid_lambda[i] * x               │  ← per-layer learnable scale
  │     + x0_lambda[i]  * x0               │  ← skip from embedding
  │                                         │
  │  CausalSelfAttention(norm(x), ...)      │
  │  MLP(norm(x))                           │
  └─────────────────────────────────────────┘
  (repeated n_layer times)
  │
RMSNorm
  │
lm_head: Linear (n_embd → vocab_size)
  │
softcap: 15 * tanh(logits / 15)   ← prevents logit explosion
  │
cross-entropy loss / logits
```

**`GPTConfig`** — dataclass holding all model shape parameters:
- `n_layer`: depth (primary knob for model size)
- `n_embd`: hidden dimension (`= n_head * HEAD_DIM`)
- `n_head` / `n_kv_head`: query / key-value heads (set equal for standard MHA)
- `window_pattern`: string of `'L'` (full context) / `'S'` (half context) characters cycling across layers; last layer is always `'L'`

**`CausalSelfAttention`** — multi-head attention with:
- *Rotary Position Embeddings (RoPE)*: queries and keys are rotated by position-dependent angles, encoding relative distance in the dot product without explicit position IDs.
- *QK-norm*: RMS-normalizing Q and K prevents attention logit magnitude from growing with depth.
- *Sliding window*: `'S'` layers attend to only the nearest `T//2` tokens, reducing the quadratic cost while keeping global context in `'L'` layers.
- *Value Embeddings* (alternating layers): a learned per-token residual `ve` (from a separate embedding table) is added to the value vectors, gated per-head by a small sigmoid gate. This lets shallow layers propagate token-identity information through the value channel, bypassing the Q–K selection bottleneck.

**`MLP`** — feed-forward block with ReLU² (squared ReLU):
- Hidden dim = 4 × model dim (standard).
- `ReLU(x)²` produces sparser activations than GELU and is trivially differentiable.

**`Block`** — pre-norm residual block: `x = x + Attn(norm(x))` then `x = x + MLP(norm(x))`.

**`GPT.forward`**:
1. Look up token embeddings, apply RMSNorm, save as `x0`.
2. For each layer: blend `x` and `x0` with learned scalars, then run the Block.
3. Final RMSNorm → linear projection → logit soft-cap.
4. If targets provided: return mean cross-entropy loss; otherwise return logits.

#### MuonAdamW Optimizer

A combined optimizer with two update rules assigned by parameter type:

| Parameter type | Optimizer | Why |
|----------------|-----------|-----|
| 2D weight matrices (attn, MLP) | **Muon** | Matrix-valued updates via Newton orthogonalization; well-suited to the large, structured parameter spaces of attention/MLP weights |
| Embeddings, scalars, lm_head | **AdamW** | Standard adaptive optimizer for 1D/lookup parameters |

**Muon step** (`muon_step_fused`):
1. *Nesterov momentum*: lookahead gradient estimate.
2. *Polar Express orthogonalization*: iteratively computes the polar factor of the gradient matrix (nearest orthogonal matrix) using a polynomial Newton iteration — a fast, compile-friendly alternative to SVD.
3. *NorMuon variance reduction*: per-row/column adaptive scaling of the orthogonalized gradient, similar in spirit to Adam's second moment but applied after orthogonalization.
4. *Cautious weight decay*: decay only applied where the gradient and parameter agree in sign (prevents decay from fighting the update direction).

Learning rates are scaled by `1/√(model_dim/768)` so that the same nominal LR works across model sizes.

#### Hyperparameters

All hyperparameters are plain module-level constants — no CLI parsing needed:

```python
# Architecture
ASPECT_RATIO  = 64       # model_dim = DEPTH * ASPECT_RATIO
HEAD_DIM      = 128      # attention head dimension
WINDOW_PATTERN = "SSSL"  # per-layer attention window pattern

# Optimization
TOTAL_BATCH_SIZE = 2**19  # ~524K tokens/step (gradient-accumulated)
EMBEDDING_LR    = 0.6
UNEMBEDDING_LR  = 0.004
MATRIX_LR       = 0.04   # Muon LR for weight matrices
SCALAR_LR       = 0.5
WEIGHT_DECAY    = 0.2    # Muon cautious weight decay (decays to 0 by end)
ADAM_BETAS      = (0.8, 0.95)
WARMUP_RATIO    = 0.0    # fraction of TIME_BUDGET for LR warmup
WARMDOWN_RATIO  = 0.5    # fraction of TIME_BUDGET for LR warmdown
FINAL_LR_FRAC   = 0.0   # final LR as fraction of initial

# Model size
DEPTH           = 8      # number of transformer layers
DEVICE_BATCH_SIZE = 128  # micro-batch size; reduce if OOM
```

#### Training Loop

```
for each micro-step (grad_accum_steps times):
    forward + backward pass (bfloat16 autocast)

update LR schedule (progress = elapsed / TIME_BUDGET)
update Muon momentum schedule (ramps 0.85→0.95 over 300 steps)
optimizer.step(); zero_grad

log: step, loss (EMA), LR multiplier, tok/sec, MFU, remaining time

if step == 0: freeze Python GC to prevent ~500ms stalls
if loss > 100: abort (loss explosion)
if elapsed >= TIME_BUDGET and step > 10: break
```

After training, `evaluate_bpb()` runs on the pinned validation shard and the final summary block is printed to stdout.

---



- **Single file to modify.** The agent only touches `train.py`. This keeps the scope manageable and diffs reviewable.
- **Fixed time budget.** Training always runs for exactly 5 minutes, regardless of your specific platform. This means you can expect approx 12 experiments/hour and approx 100 experiments while you sleep. There are two upsides of this design decision. First, this makes experiments directly comparable regardless of what the agent changes (model size, batch size, architecture, etc). Second, this means that autoresearch will find the most optimal model for your platform in that time budget. The downside is that your runs (and results) become not comparable to other people running on other compute platforms.
- **Self-contained.** No external dependencies beyond PyTorch and a few small packages. No distributed training, no complex configs. One GPU, one file, one metric.

## Platform support

This code currently requires that you have a single NVIDIA GPU. In principle it is quite possible to support CPU, MPS and other platforms but this would also bloat the code. I'm not 100% sure that I want to take this on personally right now. People can reference (or have their agents reference) the full/parent nanochat repository that has wider platform support and shows the various solutions (e.g. a Flash Attention 3 kernels fallback implementation, generic device support, autodetection, etc.), feel free to create forks or discussions for other platforms and I'm happy to link to them here in the README in some new notable forks section or etc.

Seeing as there seems to be a lot of interest in tinkering with autoresearch on much smaller compute platforms than an H100, a few extra words. If you're going to try running autoresearch on smaller computers (Macbooks etc.), I'd recommend one of the forks below. On top of this, here are some recommendations for how to tune the defaults for much smaller models for aspiring forks:

1. To get half-decent results I'd use a dataset with a lot less entropy, e.g. this [TinyStories dataset](https://huggingface.co/datasets/karpathy/tinystories-gpt4-clean). These are GPT-4 generated short stories. Because the data is a lot narrower in scope, you will see reasonable results with a lot smaller models (if you try to sample from them after training).
2. You might experiment with decreasing `vocab_size`, e.g. from 8192 down to 4096, 2048, 1024, or even - simply byte-level tokenizer with 256 possibly bytes after utf-8 encoding.
3. In `prepare.py`, you'll want to lower `MAX_SEQ_LEN` a lot, depending on the computer even down to 256 etc. As you lower `MAX_SEQ_LEN`, you may want to experiment with increasing `DEVICE_BATCH_SIZE` in `train.py` slightly to compensate. The number of tokens per fwd/bwd pass is the product of these two.
4. Also in `prepare.py`, you'll want to decrease `EVAL_TOKENS` so that your validation loss is evaluated on a lot less data.
5. In `train.py`, the primary single knob that controls model complexity is the `DEPTH` (default 8, here). A lot of variables are just functions of this, so e.g. lower it down to e.g. 4.
6. You'll want to most likely use `WINDOW_PATTERN` of just "L", because "SSSL" uses alternating banded attention pattern that may be very inefficient for you. Try it.
7. You'll want to lower `TOTAL_BATCH_SIZE` a lot, but keep it powers of 2, e.g. down to `2**14` (~16K) or so even, hard to tell.

I think these would be the reasonable hyperparameters to play with. Ask your favorite coding agent for help and copy paste them this guide, as well as the full source code.

## Notable forks

- [miolini/autoresearch-macos](https://github.com/miolini/autoresearch-macos) (MacOS)
- [trevin-creator/autoresearch-mlx](https://github.com/trevin-creator/autoresearch-mlx) (MacOS)
- [jsegov/autoresearch-win-rtx](https://github.com/jsegov/autoresearch-win-rtx) (Windows)

## License

MIT
