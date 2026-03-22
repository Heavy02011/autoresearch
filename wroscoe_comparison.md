# Comparison: Our Solution vs. wroscoe's Results

> **Context**: On 14 March 2026, [wroscoe](https://github.com/wroscoe) (Will Roscoe,
> creator of DonkeyCar) set up Karpathy's autoresearcher on the sdsandbox
> simulator overnight and shared his findings.  This document compares his
> observed results with our current `train_donkey.py` implementation.

---

## wroscoe's Key Observations

| Detail             | Value / Note                                          |
|--------------------|-------------------------------------------------------|
| CTE (snapshot)     | 2.74 (point-in-time, at 7.3 s into eval)             |
| Steering           | 0.20                                                  |
| Throttle           | 0.33                                                  |
| Training tracks    | Multiple tracks                                       |
| Eval track         | Unseen track (not in the training set)                |
| Throttle behaviour | "interesting it ignored throttle adjustments"          |

> *"I set up kaparthy's autoresearcher on the sim last night and got a
> decent model, interesting it ignored throttle adjustments.  The model was
> trained on the other tracks and this is the eval on a track it's never
> seen."*

---

## Side-by-Side Comparison

| Aspect                    | Our current solution (`train_donkey.py`)         | wroscoe's run                            |
|---------------------------|--------------------------------------------------|------------------------------------------|
| **Model outputs**         | 2 (steering + throttle)                          | 2 (but throttle effectively ignored)     |
| **Throttle in training**  | Weighted MSE, `THROTTLE_WEIGHT = 0.5`            | Predicted but not meaningfully learned   |
| **Throttle in eval**      | Fixed `SIM_THROTTLE = 0.5`                       | ~0.33 (lower than our default)           |
| **Training tracks**       | Single track (`donkey-generated-track-v0`)       | Multiple tracks                          |
| **Eval track**            | Same track as training                           | Unseen track (generalization test)       |
| **Architecture**          | 5-conv + 3-FC (~27 k params)                    | Same autoresearch CNN baseline           |
| **Training budget**       | 5 minutes                                        | Overnight (~8–12 hours of iterations)    |
| **Metric**                | Mean |CTE| over 500 steps                        | Live CTE readout in simulator            |
| **Data source**           | PD controller tub (`Kp=0.8, Kd=0.3`)            | Scripted / PD controller tubs            |
| **Augmentation**          | Horizontal flip + brightness jitter              | Same (default autoresearch augmentation) |

---

## Key Differences & Takeaways

### 1. Throttle is wasted capacity

Our model predicts `[steering, throttle]` with `THROTTLE_WEIGHT = 0.5`, but
evaluation discards the throttle prediction entirely — `evaluate_sim()` sends
the fixed `SIM_THROTTLE = 0.5` at every step.  wroscoe confirmed this
empirically: the model "ignored throttle adjustments".

**Implication**: The network spends parameters and gradient signal learning a
throttle mapping that is never used.  Reducing `THROTTLE_WEIGHT` toward `0.0`
(or switching to a single-output steering-only head) would free model capacity
for steering prediction.

**Suggested experiments for the agent**:
- Set `THROTTLE_WEIGHT = 0.0` (ignore throttle in loss, keep 2-output head for
  compatibility).
- Replace the final `Linear(50, 2)` with `Linear(50, 1)` and predict steering
  only.  This requires a small change in `evaluate_sim()` to handle 1-output
  models (but `evaluate_sim` is fixed — so keep the 2-output head and zero the
  weight).

### 2. Multi-track training improves generalization

wroscoe trained across multiple tracks and evaluated on one the model had never
seen.  Our current setup generates a single tub on `donkey-generated-track-v0`
and evaluates on the same track — the model may overfit to that track's visual
appearance and layout.

**Implication**: Collecting tubs from multiple tracks (`donkey-warehouse-v0`,
`donkey-mountain-track-v0`, `donkey-roboracingleague-track-v0`) and training on
the combined dataset should improve robustness.

**Suggested approach**:
1. Run `prepare_donkey.py --generate` on each available track (different
   `SIM_ENV_ID`) to create diverse tubs.
2. Point `find_tub_paths()` at the parent directory containing all tubs.
3. Evaluate on the held-out track to measure true generalization.

### 3. Lower throttle may reduce CTE

wroscoe's eval shows `Throttle: 0.33`, significantly lower than our
`SIM_THROTTLE = 0.5`.  A slower car has more time to correct steering errors,
reducing cross-track error.

**Implication**: The fixed eval throttle in `prepare_donkey.py` is set to 0.5.
If the primary goal is minimizing CTE, a lower throttle would help — but
changing `SIM_THROTTLE` requires modifying the read-only `prepare_donkey.py`.
Within the 5-minute training budget, the agent should instead focus on
producing a model whose steering predictions are accurate enough at the fixed
0.5 throttle.

### 4. Overnight iteration vs. single 5-minute run

wroscoe ran the full autoresearch loop overnight, meaning dozens of
experimental iterations.  Our baseline comparison is a single 5-minute training
run.  The real power of the autoresearch framework is the autonomous iteration
loop — after many keep/discard cycles, the model converges toward a much
stronger solution.

**Implication**: The comparison metric should be our best `val_cte` after a
full autonomous overnight run (recorded in `results_donkey.tsv`), not the
baseline single-run result.

---

## Actionable Next Steps

| Priority | Action                                                       | File to change         |
|----------|--------------------------------------------------------------|------------------------|
| **P0**   | Set `THROTTLE_WEIGHT = 0.0` to stop wasting capacity         | `train_donkey.py`      |
| **P0**   | Run the full autonomous loop overnight and log results        | `results_donkey.tsv`   |
| **P1**   | Generate tubs from multiple tracks for training diversity     | `sim_setup.md` (docs)  |
| **P1**   | Add multi-track tub generation example to setup guide         | `sim_setup.md`         |
| **P2**   | Compare overnight `val_cte` progression against wroscoe's CTE | `analysis_donkey.ipynb` |
| **P2**   | Experiment with BatchNorm, dropout, cosine LR schedule        | `train_donkey.py`      |

---

## Reference: wroscoe's Screenshot

![wroscoe's sdsandbox eval](https://github.com/user-attachments/assets/6af6c3a1-33cc-4c37-ba04-0fc2b4eff26e)

*Lap: 0/2 · CTE: 2.74 · Steer: 0.20 · Throttle: 0.33 · Time: 7.3 s — eval on an unseen track.*
