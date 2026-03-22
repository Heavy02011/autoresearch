---
stepsCompleted:
  - step-01-validate-prerequisites
  - step-02-extract-requirements
  - step-03-create-epics
workflowType: epics_and_stories
projectName: autoresearch
projectType: brownfield
date: '2026-03-22'
inputDocuments:
  - /workspaces/autoresearch/_bmad-output/planning-artifacts/prd.md
  - /workspaces/autoresearch/_bmad-output/planning-artifacts/architecture.md
---

# Epics and Stories - autoresearch

**Project:** Autonomous DonkeyCar Steering Optimization  
**Status:** Implementation Complete (Core Framework) — Post-MVP Features Pending  
**Date Created:** 2026-03-22

---

## PROGRESS SUMMARY

### Current State ✅

**COMPLETED EPICS:**
1. ✅ **Epic 1: Framework Architecture & Core Modules** 
   - All 9 core modules implemented and validated
   - State machine, artifact management, configuration system complete
   - CLI interface operational
   - Integration testing framework ready

2. ✅ **Epic 2: Training Integration**
   - TrainingIterator class with time budgets
   - Lazy initialization of train.py globals
   - Full integration with existing training infrastructure
   - Graceful error handling and recovery

3. ✅ **Epic 3: Simulator Evaluation**
   - gym-donkeycar integration complete
   - Metric aggregation (lap_time, cte_mean, cte_max, success, timestamp)
   - Operability checks and error handling
   - Deterministic evaluation contract

### Remaining Work (Post-MVP)

4. ⏳ **Epic 4: Experiment Tracking & Observability** (NEW)
5. ⏳ **Epic 5: Deployment & Infrastructure** (NEW)
6. ⏳ **Epic 6: End-to-End Testing & Validation** (NEW)

---

## EPIC 4: EXPERIMENT TRACKING & OBSERVABILITY

**Epic Goal:** Enable comprehensive experiment tracking, visualization, and decision traceability for autonomous optimization loops.

**Status:** Not Started  
**Estimated Effort:** 8-12 story points  
**Priority:** High (blocks production use)

### Stories

#### Story 4.1: W&B Integration — Metrics Logging
**Description:** Integrate Weights & Biases for real-time metrics tracking and visualization.

**Acceptance Criteria:**
- W&B client initializes from config (wandb.use_wandb flag)
- Training losses logged to W&B at each step
- Evaluation metrics (lap_time, cte_mean, cte_max, success) logged per iteration
- Model metadata (git hash, config snapshot) attached to W&B run
- Promotion decisions (accept/reject) tracked with reasoning

**Files to Modify:**
- `src/autoresearch/orchestrate.py` — add W&B logging to autonomous loop
- `src/autoresearch/training.py` — log loss to W&B during training iterations
- `src/autoresearch/evaluate.py` — log metrics to W&B after evaluation
- `configs/environment.yaml` — add W&B project/entity configuration

**Implementation Detail:**
```python
# In Orchestrator.__init__:
if self.config.environment.use_wandb:
    wandb.init(project=self.config.environment.wandb_project, 
               name=run_id,
               config=OmegaConf.to_container(config))

# In _train_iteration:
wandb.log({"loss": smooth_loss, "step": iteration})

# In run_autonomous_loop (after evaluation):
wandb.log({"lap_time": metrics["lap_time"], "cte_mean": metrics["cte_mean"]})
```

**Testing:**
- Unit test: W&B client initialization with/without flag
- Integration test: metrics logged successfully in dry-run mode

**Estimated Effort:** 3 points

---

#### Story 4.2: W&B Integration — Decision Trace Logging
**Description:** Log promotion gate decisions, model changes, and experiment lineage to W&B for retrospective analysis.

**Acceptance Criteria:**
- Gate verdict (PASS/FAIL) logged with decision reasoning
- Accepted vs. rejected models tracked in W&B artifact lineage
- Model checkpoints linked to W&B with metadata
- Decision trace queryable by iteration, verdict, and metrics
- Hyperparameter changes logged when model file updated

**Files to Modify:**
- `src/autoresearch/promote.py` — log gate verdict and reasoning
- `src/autoresearch/orchestrate.py` — log accepted model metadata

**Estimated Effort:** 2 points

---

#### Story 4.3: Structured Logging Dashboard
**Description:** Create a local dashboard view (CLI JSON API or simple HTML) to inspect experiment progression without W&B.

**Acceptance Criteria:**
- `autoresearch status <run_id>` command shows live metrics progress
- JSON output option for programmatic querying: `autoresearch status <run_id> --format json`
- Running progress shows current iteration, best model, latest lap_time
- Full state history queryable via CLI
- Structured logs readable from `logs/run_ID.log`

**Files to Modify:**
- `src/autoresearch/cli.py` — enhance status command with JSON/html output

**Estimated Effort:** 2 points

---

#### Story 4.4: Experiment Comparison CLI
**Description:** Allow users to compare metrics across multiple runs for analysis.

**Acceptance Criteria:**
- `autoresearch compare <run_id1> <run_id2> ...` command shows side-by-side metrics
- Highlights best lap_time, improvement %, and decision trends
- CSV export for further analysis
- Diff of promotion gate thresholds used

**Files to Modify:**
- `src/autoresearch/cli.py` — add compare command

**Estimated Effort:** 2 points

---

**Epic 4 Total Effort:** 9 points

---

## EPIC 5: DEPLOYMENT & INFRASTRUCTURE

**Epic Goal:** Enable reproducible, automated deployment of the framework in production and containerized environments.

**Status:** Not Started  
**Estimated Effort:** 12-16 story points  
**Priority:** Medium (needed for overnight runs and cloud deployment)

### Stories

#### Story 5.1: Docker Containerization
**Description:** Create Docker image and docker-compose stack for reproducible environment.

**Acceptance Criteria:**
- Dockerfile builds complete environment with CUDA support (GPU-ready)
- docker-compose.yml orchestrates training service + simulator service
- All dependencies pinned (Python version, torch version, gym-donkeycar version)
- Simulator path configurable via environment variable
- Build includes all configs and documentation
- Image builds from clean checkout in <5 minutes

**Files to Create:**
- `docker/Dockerfile` — multi-stage build (dev/prod)
- `docker/docker-compose.yml` — training + simulator services
- `docker/.dockerignore` — exclude runs/, .git, etc.
- `docker/entrypoint.sh` — preflight validation + command routing

**Implementation Detail:**
```dockerfile
FROM nvidia/cuda:12.1-runtime-ubuntu22.04
WORKDIR /autoresearch
COPY pyproject.toml uv.lock ./
RUN pip install -e .
COPY . .
ENTRYPOINT ["./docker/entrypoint.sh"]
```

**Testing:**
- Build succeeds from clean repo
- Container runs dry-run test successfully
- Simulator connectivity tested (with env var for path)

**Estimated Effort:** 4 points

---

#### Story 5.2: GitHub Actions CI/CD Pipeline
**Description:** Automate testing, linting, and validation on every commit.

**Acceptance Criteria:**
- Workflow runs on every push and PR
- Tests pass: `pytest tests/ -v`
- Linting passes: `pylint src/`
- Type checking passes: `mypy src/`
- Dry-run test validates pipeline (no GPU required)
- Build artifact (Docker image) created on main branch merge
- Status badge shown in README

**Files to Create:**
- `.github/workflows/ci.yml` — GitHub Actions workflow
- `.github/workflows/docker-publish.yml` — publish image to registry

**Estimated Effort:** 3 points

---

#### Story 5.3: Preflight Validation Script
**Description:** Create setup validation to catch environment issues before run starts.

**Acceptance Criteria:**
- Script checks Python version (3.10+)
- Verifies all dependencies installed and importable
- Validates GPU availability (CUDA + torch)
- Tests simulator connectivity (if simulator path provided)
- Checks config file validity (Hydra + Pydantic validation)
- Provides actionable error messages
- Can be run standalone: `python -m autoresearch.preflight`

**Files to Create:**
- `src/autoresearch/preflight.py` — validation logic
- Update `src/autoresearch/cli.py` — add preflight command

**Implementation Detail:**
```python
def check_cuda():
    """Verify CUDA + torch availability."""
    try:
        import torch
        if not torch.cuda.is_available():
            warn("CUDA not available. Training will run on CPU (very slow).")
        else:
            print(f"✓ CUDA available ({torch.cuda.get_device_name()})")
    except ImportError:
        error("torch not installed")
```

**Estimated Effort:** 2 points

---

#### Story 5.4: Production Readiness Checklist
**Description:** Document prerequisites and setup steps for production overnight runs.

**Acceptance Criteria:**
- SETUP.md file created with step-by-step instructions
- Covers: Python env, CUDA setup, simulator setup, config customization
- Includes troubleshooting section for common failures
- Documents resource requirements (GPU memory, disk space, runtime)
- Provides example commands for dry-run, sample run (5 iterations), and full overnight run
- Includes rollback and resume procedures
- Includes example systemd service file

**Files to Create:**
- `SETUP.md` — comprehensive setup guide
- `examples/systemd/autoresearch.service` — example systemd unit file

**Estimated Effort:** 2 points

---

#### Story 5.5: Continuous Monitoring & Watchdog
**Description:** Add timeout, restart, and health-check mechanisms for overnight stability.

**Acceptance Criteria:**
- Watchdog monitors process health and restarts on failure
- Timeouts enforced per training iteration (configurable max_iteration_time)
- Evaluation phase has timeout (configurable max_eval_time)
- Graceful shutdown on timeout (save checkpoint, log error, exit cleanly)
- Resume from last checkpoint if crash + restart
- Health check logs to structured log every 5 minutes during long runs

**Files to Modify:**
- `src/autoresearch/orchestrate.py` — add timeout checks
- `src/autoresearch/training.py` — timeout in run_iteration
- Update `configs/experiment.yaml` — add timeout thresholds

**Estimated Effort:** 3 points

---

**Epic 5 Total Effort:** 14 points

---

## EPIC 6: END-TO-END TESTING & VALIDATION

**Epic Goal:** Establish comprehensive test coverage and validation procedures for production confidence.

**Status:** Not Started  
**Estimated Effort:** 10-14 story points  
**Priority:** High (critical for reliability)

### Stories

#### Story 6.1: End-to-End Integration Test
**Description:** Create full pipeline test covering train → evaluate → gate → promote cycle.

**Acceptance Criteria:**
- Test runs 2 dry-run iterations (no actual training/sim)
- Validates state transitions through all 5 states
- Checks artifact creation (checkpoint, metrics, state.json)
- Verifies gate verdicts (accept/reject logic)
- Confirms best model selection is correct
- Completes in <10 seconds
- Can be run via `pytest tests/test_e2e.py`

**Files to Create:**
- `tests/test_e2e.py` — full pipeline test

**Implementation Skeleton:**
```python
def test_full_autonomous_loop():
    """Test complete train-eval-promote cycle."""
    config = ExperimentConfig(dry_run=True, max_iterations=2)
    orchestrator = Orchestrator(config, run_id="test_run")
    orchestrator.run_autonomous_loop()
    
    # Verify state machine reached promoted state
    assert orchestrator.state_tracker.current_state == PromotionState.PROMOTED
    
    # Verify artifacts exist
    assert orchestrator.artifact_manager.best_model_path.exists()
    assert orchestrator.artifact_manager.metrics_dir.exists()
```

**Estimated Effort:** 3 points

---

#### Story 6.2: State Machine Edge Cases
**Description:** Test all legal and illegal state transitions.

**Acceptance Criteria:**
- Legal transitions (training → evaluating → promotion_gate → promoted) work
- Legal transition (promoted → training for next iteration) works
- Legal transition (promotion_gate → training if rejected) works
- Illegal transitions fail (e.g., training → promoted directly)
- Rollback from promoted → training works
- Rollback twice fails (only one backtrack allowed)
- State history immutable (no mutation after creation)

**Files to Create/Modify:**
- `tests/test_state_machine.py` — comprehensive state tests

**Estimated Effort:** 2 points

---

#### Story 6.3: Configuration Validation Tests
**Description:** Test Pydantic models and Hydra config loading.

**Acceptance Criteria:**
- Valid configs load without error
- Invalid configs raise clear Pydantic ValidationError
- Missing required fields detected at validation time
- Type coercion works (string "5" → int 5)
- Hydra overrides work: `--config-name=experiment.yaml`
- Environment variable overrides work: `AUTORESEARCH_MAX_ITERATIONS=10`
- Config snapshots are correct JSON

**Files to Create/Modify:**
- `tests/test_config.py` — enhance existing tests

**Estimated Effort:** 2 points

---

#### Story 6.4: Promotion Gate Logic Tests
**Description:** Test all promotion gate criteria and edge cases.

**Acceptance Criteria:**
- Gate accepts if metrics beat threshold AND model is operational
- Gate rejects if metrics miss threshold (even if operational)
- Gate rejects if operability check fails (even if metrics pass)
- Improvement % correctly calculated (vs. baseline)
- First iteration always promoted (no baseline to beat)
- Edge case: NaN metrics rejected
- Edge case: negative improvement rejected (regression)

**Files to Create/Modify:**
- `tests/test_promotion_gate.py` — comprehensive gate tests

**Estimated Effort:** 2 points

---

#### Story 6.5: Artifact Management Tests
**Description:** Test checkpoint saving, loading, and artifact hierarchy.

**Acceptance Criteria:**
- Checkpoints saved with correct naming convention
- Checkpoints loadable as torch models
- Metadata (git hash, timestamp) correctly recorded
- Config snapshots are valid JSON + loadable
- Best model symlink updated correctly
- Run manifest records all iterations
- Artifacts readable after 30-day gap (no session state required)

**Files to Create/Modify:**
- `tests/test_artifacts.py` — comprehensive artifact tests

**Estimated Effort:** 2 points

---

#### Story 6.6: Integration with Simulator & Training (Production Validation)
**Description:** Document and validate production readiness with real training + simulator.

**Acceptance Criteria:**
- Can run 5-iteration test with actual training (if GPU available)
- Can run gym-donkeycar evaluation (if simulator available)
- Metrics improve from iteration 1 → iteration 5 (or document why not)
- Model successfully exports to ONNX
- Framework handles training crashes gracefully
- Framework handles simulator timeouts gracefully

**Testing Strategy:**
- Dry-run test (passing, guaranteed)
- Production test (conditional, requires GPU + simulator)
- CI runs dry-run only; production tests run on release branches

**Estimated Effort:** 2 points

---

**Epic 6 Total Effort:** 11 points

---

## SUMMARY BY PRIORITY

### Immediate (Blocking Deployment)
- **Epic 6.1** — E2E integration test (3 points)
- **Epic 6.2** — State machine edge cases (2 points)
- **Story 5.1** — Docker containerization (4 points)
- **Story 5.3** — Preflight validation (2 points)

**Subtotal: 11 points** — Estimated 1-1.5 sprints

### High Priority (Needed for Overnight Runs)
- **Story 5.5** — Watchdog & monitoring (3 points)
- **Story 4.1** — W&B metrics logging (3 points)
- **Story 5.2** — GitHub Actions CI (3 points)

**Subtotal: 9 points** — Estimated 1 sprint

### Medium Priority (Nice-to-Have for Production)
- **Story 4.2** — Decision trace logging (2 points)
- **Story 4.3** — Local dashboard (2 points)
- **Story 5.4** — Setup documentation (2 points)
- **Epic 6.3-6** — Config/gate/artifact tests (6 points)

**Subtotal: 12 points** — Estimated 1.5 sprints

### Lower Priority (Enhancements)
- **Story 4.4** — Experiment comparison (2 points)

---

## TOTAL REMAINING EFFORT

- **Post-MVP Stories:** 34 story points
- **Estimated Delivery:** 4-5 sprints (2-2.5 months at 8 points/week)
- **High-Risk Items:** None (all stories are straightforward enhancements to working framework)

---

## NEXT STEPS

**Recommended Sequence for Implementation:**

1. **Sprint 1 (Immediate):** Stories 6.1, 6.2, 5.1, 5.3 — Get framework production-ready with tests + Docker
2. **Sprint 2:** Stories 5.5, 4.1, 5.2 — Add monitoring, W&B integration, CI/CD
3. **Sprint 3:** Stories 4.2, 4.3, 5.4, 6.3-6 — Polish and complete validation
4. **Optional:** Story 4.4 — Experiment comparison for data analysis

**Ready to proceed?** User can select:
- Option A: Start Sprint 1 (select a specific story to dev)
- Option B: Refine any epic/story scope
- Option C: Adjust priorities

