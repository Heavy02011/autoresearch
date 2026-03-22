# autoresearch — DonkeyCar × sdsandbox edition

This is an experiment to have the LLM do its own research on autonomous driving.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `donkey-mar5`). The branch
   `autoresearch/donkey-<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/donkey-<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context.
   - `prepare_donkey.py` — fixed constants, tub data loading, simulator evaluation harness. Do not modify.
   - `train_donkey.py` — the file you modify. CNN model, optimiser, training loop.
4. **Verify the simulator is running**:
   `~/donkey_sim/donkey_sim.x86_64 --headless --port 9091 &`
   If it's not running, tell the human to start it.
5. **Verify sim tub data exists** at `~/donkeycar/data/sim_tub`.
   If not, tell the human to run: `python prepare_donkey.py --generate`

   **Using real-world data**: If you have an existing tub recorded from a real DonkeyCar (or any
   other source), you can use it directly — no simulation required.  Simply point the training
   script at it:
   ```bash
   python train_donkey.py --tub /path/to/real_world_tub
   # or, using the environment variable:
   DONKEY_TUB=/path/to/real_world_tub python train_donkey.py
   ```
   To inspect a tub (count records, verify format) without training:
   ```bash
   python prepare_donkey.py --tub /path/to/real_world_tub
   ```
   Both catalogue-based tubs (`manifest.json` + JSONL catalogue files) and legacy
   `record_*.json` tubs are supported.
6. **Initialise results_donkey.tsv**: Create `results_donkey.tsv` with just the header row.
   The baseline will be recorded after the first run.
7. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on a single GPU. The training script runs for a **fixed time
budget of 5 minutes** (wall-clock training time, excluding startup overhead). You
launch it simply as: `python train_donkey.py`.

**What you CAN do:**
- Modify `train_donkey.py` — this is the only file you edit. Everything is fair game:
  model architecture, optimiser, hyperparameters, augmentation, loss function, batch size, etc.

**What you CANNOT do:**
- Modify `prepare_donkey.py`. It is read-only. It contains the fixed evaluation harness,
  tub data loading, simulator interface, and training constants (time budget, image size, etc).
- Install new packages or add dependencies.
- Modify the evaluation harness. The `evaluate_sim` function in `prepare_donkey.py` is the
  ground-truth metric.

**The goal is simple: get the lowest val_cte.** Since the time budget is fixed, you don't
need to worry about training time — it's always 5 minutes. Everything is fair game: change
the architecture, the optimiser, the hyperparameters, the augmentation, the loss function.
The only constraint is that the code runs without crashing and finishes within the time budget.

**VRAM** is a soft constraint. Some increase is acceptable for meaningful val_cte gains, but
it should not blow up dramatically.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds
ugly complexity is not worth it. Conversely, removing something and getting equal or better
results is a great outcome — that's a simplification win. When evaluating whether to keep a
change, weigh the complexity cost against the improvement magnitude. A 0.001 val_cte improvement
that adds 20 lines of hacky code? Probably not worth it. A 0.001 val_cte improvement from
deleting code? Definitely keep. An improvement of ~0 but much simpler code? Keep.

**The first run**: Your very first run should always be to establish the baseline, so you will
run the training script as is.

## Output format

Once the script finishes it prints a summary like this:

```
---
val_cte:          0.045231
training_seconds: 300.1
total_seconds:    320.4
peak_vram_mb:     1240.0
num_epochs:       42
num_samples:      12800
num_params_k:     427
```

You can extract the key metric from the log file:

```
grep "^val_cte:" run.log
```

## Logging results

When an experiment is done, log it to `results_donkey.tsv` (tab-separated, NOT comma-separated).

The TSV has a header row and 5 columns:

```
commit	val_cte	memory_gb	status	description
```

1. git commit hash (short, 7 chars)
2. val_cte achieved (e.g. 0.045231) — use 0.000000 for crashes
3. peak memory in GB, round to .1f (e.g. 1.2 — divide peak_vram_mb by 1024) — use 0.0 for crashes
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried

Example:

```
commit	val_cte	memory_gb	status	description
a1b2c3d	0.045231	1.2	keep	baseline
b2c3d4e	0.038400	1.3	keep	add batch norm to encoder
c3d4e5f	0.052000	1.2	discard	switch to SGD (worse)
d4e5f6g	0.000000	0.0	crash	double model width (OOM)
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/donkey-mar5`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on.
2. Tune `train_donkey.py` with an experimental idea by directly hacking the code.
3. git commit.
4. Run the experiment: `python train_donkey.py > run.log 2>&1`
   (redirect everything — do NOT use tee or let output flood your context)
5. Read out the results: `grep "^val_cte:\|^peak_vram_mb:" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the
   Python stack trace and attempt a fix.
7. Record the results in the TSV (NOTE: do not commit results_donkey.tsv, leave it untracked).
8. If val_cte improved (lower), you "advance" the branch, keeping the git commit.
9. If val_cte is equal or worse, you git reset back to where you started.

**Timeout**: Each experiment should take ~5 minutes total (+ a few seconds for startup and
eval overhead). If a run exceeds 10 minutes, kill it and treat it as a failure (discard and
revert).

**Crashes**: If a run crashes (OOM, sim connection failure, or a bug), use your judgement:
If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run.
If the idea itself is fundamentally broken, just skip it, log "crash" as the status, and
move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause
to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a
good stopping point?". The human might be asleep, or gone from a computer and expects you
to continue working *indefinitely* until you are manually stopped. You are autonomous. If
you run out of ideas, think harder — re-read the in-scope files for new angles, try
combining previous near-misses, try more radical architectural changes, look at the
experiment ideas table in the plan. The loop runs until the human interrupts you, period.

## Deployment check (optional, manual)

After a significant improvement, export for the real car:

```bash
python export_donkey.py   # produces autopilot.onnx
# copy to Raspberry Pi and run donkey drive --model autopilot.onnx
```
