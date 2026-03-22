---
stepsCompleted:
  - step-01-init
  - step-02-discovery
  - step-02c-executive-summary
  - step-03-success
  - step-04-journeys
  - step-05-domain
  - step-06-innovation
  - step-07-project-type
  - step-08-scoping
  - step-09-functional
  - step-10-nonfunctional
  - step-11-polish
  - step-12-complete
inputDocuments:
  - README.md
  - program.md
projectName: autoresearch
projectType: brownfield
documentCounts:
  briefCount: 0
  researchCount: 0
  projectDocsCount: 2
classification:
  projectType: developer_tool
  domain: autonomous_vehicles_robotics
  complexity: medium-high
  projectContext: brownfield
  scope: "Working proof-of-concept: autoresearch framework optimizing DonkeyCar CNN steering models via sdsandbox simulator"
workflowType: prd
date: '2026-03-22'
---

# Product Requirements Document - autoresearch

**Author:** heavy
**Date:** 2026-03-22

## Overview

autoresearch is a framework for autonomous AI-driven research experimentation. The core concept enables an AI agent to autonomously conduct machine learning experiments over extended periods (e.g., overnight), iteratively modifying training code, executing 5-minute experiment runs, and evaluating results to keep or discard changes.

The system is built around a simplified single-GPU implementation of nanochat (a GPT-style language model trainer) with fixed time budgets and measurable outcomes, making it suitable for autonomous optimization loops.

### Core Use Cases

1. **Language Model Pretraining (Primary)** — GPT model training on text data with val_bpb metric
2. **Domain Adaptation Framework** — Extensible to other domains including robotics and control systems
3. **DonkeyCar Neural Autopilot Optimization (In Scope)** — Application of the autoresearch framework to train and optimize CNN-based steering models for RC vehicle control via the sdsandbox Unity simulator

## Executive Summary

Autonomous DonkeyCar steering optimization is technically feasible, but the capability is not currently reproducible from this repository. A public video demonstrates a working simulator run, yet no implementation code or operational documentation is available to recreate or extend that result. This creates a high-confidence but low-repeatability state: proof exists, but maintainers and contributors cannot reliably run the system.

This PRD defines a clean, reproducible implementation that adapts the existing autoresearch experimentation loop to DonkeyCar steering. The system will use fixed 5-minute training budgets, autonomous keep/discard iteration, and simulator-based evaluation to improve driving performance without manual hyperparameter search. The implementation will be built from the established technical blueprint in donkeycar_plan.md and validated through end-to-end simulator execution.

Primary users are DonkeyCar maintainers and advanced community contributors who need repeatable, agent-driven model tuning. The target outcome is a running simulator workflow where model quality improves over time, evidenced by decreasing lap times during training and iteration.

### What Makes This Special

The core differentiator is not speculative research but reproducible operationalization of an already validated concept. Instead of asking whether autonomous experimentation can work for DonkeyCar, this project focuses on making it repeatable, maintainable, and measurable from source control.

The adaptation reuses proven autoresearch mechanics and applies them to a robotics control task with minimal conceptual translation: code mutation loop, bounded experiment runtime, metric-based selection, and longitudinal optimization. DonkeyCar is a uniquely practical domain for this approach because it combines a real robotics community, simulator accessibility, and clear performance metrics that expose training progress.

## Project Classification

- **Project Type:** Developer Tool
- **Domain:** Autonomous Vehicles / Robotics
- **Complexity:** Medium-High
- **Project Context:** Brownfield proof-of-concept based on existing autoresearch architecture and documented DonkeyCar adaptation plan

## Success Criteria

### User Success

- A maintainer can run the DonkeyCar autoresearch pipeline end-to-end from a clean checkout by following project documentation only.
- A user can launch training and evaluation in simulator mode without undocumented manual intervention.
- The user can observe training progression through clear experiment logs and metric history.

### Business Success

- The project transitions from "demo-only proof" to a reproducible reference implementation suitable for maintainers and contributors.
- New contributors can onboard to the DonkeyCar optimization flow in less than one working session using repository documentation.
- The implementation is credible enough to share publicly as a repeatable benchmark workflow for DonkeyCar model optimization.

### Technical Success

- The autonomous experimentation loop executes reliably across multiple iterations: modify train file, run fixed-budget experiment, evaluate, keep/discard.
- Simulator-based evaluation is stable and produces consistent metrics under fixed evaluation settings.
- Model export path (ONNX) is operational and documented.
- Failures are diagnosable through structured logs (training, evaluation, and decision trace).

### Measurable Outcomes

- **Primary metric:** Median lap time over evaluation laps decreases during autonomous training iterations.
- **Secondary metric:** Cross-track error trend improves or remains stable while lap time improves.
- **Reproducibility metric:** At least one independent rerun from clean setup reproduces a lap-time-improvement trend.
- **Operational metric:** Overnight run completes planned experiment cycles without critical pipeline breakdown.

## Product Scope

### MVP - Minimum Viable Product

- Implement the DonkeyCar adaptation defined in donkeycar_plan.md.
- Run autonomous iterative training/evaluation in sdsandbox.
- Produce a working simulator-driving model and demonstrate lap-time improvement trend over training.
- Provide concise setup + run documentation sufficient for reproduction by another maintainer.

### Growth Features (Post-MVP)

- Improved experiment analytics dashboarding/visualization.
- Better policy safety constraints (e.g., stability guards during optimization).
- Extended track/condition evaluation for generalization checks.
- More robust automated regression checks before accepting model changes.

### Vision (Future)

- Standardized autonomous optimization workflow for DonkeyCar community usage.
- Continuous optimization loops that can target broader driving objectives beyond lap time.
- Generalized "autoresearch for robotics control" template reusable across similar simulator-based platforms.

## User Journeys

### Journey 1: Primary User - Success Path (Maintainer Runs First Full Optimization)

**Persona:** Alex, DonkeyCar Maintainer
**Opening Scene:** Alex has seen the proof video but cannot reproduce results from the repository. Confidence is high, trust is low.
**Rising Action:** Alex follows setup docs, starts simulator, launches autonomous training loop, and watches metrics/logs update across iterations.
**Climax:** After multiple fixed-budget cycles, median lap time improves versus baseline while run remains stable.
**Resolution:** Alex has a reproducible workflow, can rerun it, and can share exact commands/results with contributors.

**What could go wrong / recovery:**
- Environment mismatch or missing dependency.
- Recovery: clear preflight checks, actionable setup validation, fail-fast error messages.

**Information needed at each step:**
- Setup status, simulator connectivity, active experiment id, current best lap time, trend vs baseline.

**Emotional arc:**
- Skepticism -> cautious progress -> confidence -> ownership.

### Journey 2: Primary User - Edge Case (Performance Regresses During Training)

**Persona:** Alex, same maintainer under time pressure
**Opening Scene:** Training starts, but lap time worsens after early iterations.
**Rising Action:** Alex inspects logs/metrics, checks accepted vs rejected experiments, identifies unstable parameter changes.
**Climax:** The system rejects regressive candidates and returns to last known good model/config.
**Resolution:** Run continues without manual patching; reliability is preserved and the improvement trend resumes.

**What could go wrong / recovery:**
- Metric noise causes false "improvements".
- Recovery: median-over-multiple-laps evaluation and guardrails for acceptance.

**Information needed:**
- Acceptance decision rationale, rollback events, variance bands, confidence in improvement.

**Emotional arc:**
- Frustration -> diagnosis -> controlled recovery -> renewed trust.

### Journey 3: Admin/Operations - Environment and Runtime Stewardship

**Persona:** Sam, infra/ops contributor for project CI machines
**Opening Scene:** Sam needs the PoC to run overnight reliably on shared GPU resources.
**Rising Action:** Sam configures runtime profiles, monitors resource usage, validates simulator process health, and ensures run restarts are safe.
**Climax:** Overnight job completes planned cycle count without critical crash, and outputs complete artifact/log set.
**Resolution:** Sam can treat the workflow as operationally dependable and schedule periodic runs.

**What could go wrong / recovery:**
- GPU memory spikes, simulator hangs, long-tail process failures.
- Recovery: health checks, watchdog restart strategy, bounded timeouts, structured process exit codes.

**Information needed:**
- Resource telemetry, process heartbeat, run completion state, artifact completeness checks.

**Emotional arc:**
- Caution -> control -> operational confidence.

### Journey 4: Support/Troubleshooting - Investigating a Failed Reproduction Attempt

**Persona:** Priya, support-minded contributor helping new users
**Opening Scene:** A community member reports "it doesn't improve lap time" after following docs.
**Rising Action:** Priya requests run logs/artifacts, compares baseline configuration, reproduces run with same seed/settings.
**Climax:** Priya identifies root cause (e.g., wrong sim map/version, evaluation mismatch, or invalid data split).
**Resolution:** Priya provides a precise fix path and updates troubleshooting docs to prevent recurrence.

**What could go wrong / recovery:**
- Ambiguous logs prevent diagnosis.
- Recovery: standardized log schema, run manifest, deterministic config snapshot per run.

**Information needed:**
- Exact config hash, simulator version, data path, seed, accepted/rejected experiment timeline.

**Emotional arc:**
- Uncertainty -> evidence gathering -> clarity -> contributor enablement.

### Journey 5: API/Integration Consumer - Exporting and Reusing Best Model

**Persona:** Jordan, developer integrating best model into downstream validation tooling
**Opening Scene:** Jordan needs the best checkpoint/model in a stable export format.
**Rising Action:** Jordan consumes run outputs, exports ONNX, and executes integration validation scripts.
**Climax:** Exported model reproduces expected simulator behavior in downstream pipeline.
**Resolution:** Jordan can automate handoff from optimization run to deployment/benchmark workflows.

**What could go wrong / recovery:**
- Export incompatibility or missing metadata.
- Recovery: explicit export contract, model metadata manifest, compatibility check script.

**Information needed:**
- Best-model selection criteria, export command, model signature, validation pass/fail report.

**Emotional arc:**
- Dependency risk -> confirmation -> integration velocity.

### Journey Requirements Summary

These journeys reveal required capability areas:

- Reproducible setup and preflight validation.
- Deterministic, observable experiment orchestration with keep/discard logic.
- Metric pipeline with baseline comparison and lap-time trend reporting.
- Regression safety via rollback and acceptance guardrails.
- Operational resilience: health checks, watchdogs, bounded runtime behavior.
- Troubleshooting-grade observability: structured logs, run manifests, config snapshots.
- Artifact contract for downstream use: best-model traceability and ONNX export validation.

## Domain-Specific Requirements

### Compliance & Regulatory

- MVP is simulator-based and not subject to direct road-approval requirements, but robotics safety engineering principles are mandatory internal quality gates.
- For later real-car deployment, safety bounds (steering and throttle), test protocols, and promotion criteria must be explicitly documented.
- Every model promotion decision must be auditable with training configuration, dataset source, simulator setup, and evaluation evidence.

### Technical Constraints

- Deterministic evaluation conditions are required: fixed simulator version, map configuration, seed handling, and reset logic.
- Safety guardrails for autonomous iteration are mandatory:
  - Regressive models cannot be promoted to best model.
  - Automatic rollback to the latest stable model/config must be supported.
- Runtime and reliability constraints:
  - Overnight optimization runs must complete without manual intervention.
  - Per-iteration training and evaluation must stay within bounded runtime budgets.
- Observability constraints:
  - Structured logs for training, evaluation, keep/discard decisions, and failures.
  - Continuous lap-time trend and CTE trend visibility.
- Post-training operability gate:
  - The selected best model must run successfully in simulator execution after training.
  - Promotion requires a post-training simulator validation run with successful model load, inference loop stability, no critical runtime errors, and logged lap-time telemetry.

### Integration Requirements

- Stable integration with sdsandbox including startup, connectivity checks, and session recovery behavior.
- Reproducible data flow from DonkeyCar data format through training and evaluation pipeline.
- ONNX export must include consistent model interface and metadata for downstream validation.
- Per-run artifact contract must include:
  - best promoted model,
  - configuration snapshot,
  - metric history,
  - accepted/rejected experiment decision log.

### Risk Mitigations

- **Risk:** Lap-time gains result from overfitting to narrow simulator conditions.
  **Mitigation:** Multi-lap median evaluation and optional multi-track checks in post-MVP phases.
- **Risk:** Metric noise causes false keep decisions.
  **Mitigation:** Acceptance thresholds, variance-aware validation, and pre-promotion re-check.
- **Risk:** Simulator or infrastructure instability during long runs.
  **Mitigation:** Watchdog heartbeat, timeout/restart strategy, and resumable run state.
- **Risk:** Results are not reproducible across environments.
  **Mitigation:** Environment pinning, version locking, preflight validation, and complete run metadata.
- **Risk:** Trained/exported model improves metrics but fails to operate in simulator runtime.
  **Mitigation:** Mandatory post-training simulator execution validation before best-model promotion.

## Innovation & Novel Patterns

### Detected Innovation Areas

- **Autonomous maintainer loop for robotics training:** Instead of manual hyperparameter tuning sessions, the system runs autonomous experiment cycles with explicit keep/discard promotion decisions.
- **Cross-domain transfer as product pattern:** A proven experimentation paradigm from LLM research is adapted into a reproducible autonomous-driving simulator workflow.
- **Promotion by runtime operability, not metric only:** A model is promotable only if it improves metrics and also runs stably in simulator execution.

### Market Context & Competitive Landscape

- Typical DonkeyCar workflows focus on manual model iteration and isolated training runs.
- This PoC differentiates through:
  - automated repeatable iteration,
  - auditable model-promotion decisions,
  - overnight operational execution,
  - mandatory simulator operability validation before promotion.
- The differentiator is orchestration and reproducibility, not a novel neural architecture.

### Validation Approach

- Compare baseline against autonomous iterative runs under identical simulator conditions.
- Innovation is validated when all of the following hold:
  - lap-time trend improves,
  - CTE does not destabilize,
  - promoted best model passes post-training simulator runtime validation,
  - trend is reproducible in at least one independent rerun.

### Risk Mitigation

- **Risk:** "Autonomy" adds complexity without meaningful outcome gains.
  **Mitigation:** Hard outcome gates (lap-time improvement + simulator operability + reproducibility).
- **Risk:** Noisy metrics lead to false-positive promotions.
  **Mitigation:** Median-based evaluation, promotion thresholds, and revalidation before promotion.
- **Risk:** Orchestration fragility in long-running overnight operation.
  **Mitigation:** Watchdogs, resumable state, and structured failure paths.

## Developer Tool Specific Requirements

### Project-Type Overview

The product is a developer-facing research and optimization tool for DonkeyCar steering models in simulator workflows. Primary value comes from reproducible execution, deterministic evaluation, and auditable promotion decisions rather than end-user interface design.

### Technical Architecture Considerations

- The workflow must be scriptable and automation-friendly for local and overnight runs.
- Interfaces between prepare, train, evaluate, promote, and export stages must be explicit and stable.
- Runtime reliability is a first-order requirement: deterministic simulator configuration, structured logging, and resumable long runs.
- A model is only promotable when it both improves target metrics and passes simulator operability validation.

### Language Matrix

- **Primary language:** Python for orchestration, training, and evaluation.
- **Model portability:** ONNX export path for downstream validation and integration.
- **MVP scope decision:** single-language-first implementation (Python), no multi-language SDK in MVP.

### Installation Methods

- A single reproducible setup path from clean checkout to first successful run is required.
- Installation must include:
  - Python environment and dependencies,
  - simulator setup and connectivity checks,
  - preflight validation before long runs.
- Setup instructions must eliminate undocumented manual steps.

### API Surface

- MVP does not require a public network API.
- Stable tool surface is provided through:
  - script or CLI entrypoints,
  - configuration files and flags,
  - standardized artifact outputs.
- Orchestration contract across prepare/train/evaluate/promote/export must be versioned and testable.

### Code Examples

- Documentation must include copy-paste-ready examples for:
  - environment and simulator setup,
  - launching a full optimization run,
  - evaluating lap-time and CTE trends,
  - validating best-model simulator runtime,
  - exporting and validating ONNX output.

### Migration Guide

- Migration starts from a "video-proven but undocumented" baseline.
- Guide must define transition from ad-hoc/manual workflow to a reproducible and auditable runbook.
- Maintainer-facing migration notes must include:
  - old implicit steps vs. new explicit pipeline,
  - compatibility constraints (simulator/version/data assumptions),
  - troubleshooting paths for common migration failures.

### Implementation Considerations

- Priority 1: stable end-to-end run with measurable lap-time improvement.
- Priority 2: enforce post-training simulator operability gate before promotion.
- Priority 3: maximize developer observability via logs, manifests, and decision traces.
- Out of MVP scope: broad SDK ecosystem and UI-heavy product layers.

## Project Scoping & Phased Development

### MVP Strategy & Philosophy

**MVP Approach:** Problem-solving MVP focused on reproducible end-to-end optimization and measurable simulator driving improvement.
**Resource Requirements:** 1-2 technical maintainers (ML/training and simulator/infra), with optional documentation/QA support.

### MVP Feature Set (Phase 1)

**Core User Journeys Supported:**
- Primary user success path (full reproducible run)
- Primary user edge case (regressive iteration handling with guardrails)
- Operations baseline path (stable overnight run)
- Support baseline path (diagnosable failures via structured logs)

**Must-Have Capabilities:**
- Reproducible setup from clean checkout to first successful run.
- Deterministic simulator evaluation (fixed version/map/seeds/reset behavior).
- Autonomous iteration loop with keep/discard and rollback.
- Primary metrics: lap-time trend improvement with CTE stability guard.
- Post-training simulator operability gate before model promotion.
- ONNX export with minimum validation.
- Structured logs, run manifest, and promotion decision trace.
- Documentation for execution, troubleshooting, and reproducibility.

### Post-MVP Features

**Phase 2 (Post-MVP):**
- Enhanced analytics and visualization.
- Multi-track and multi-condition robustness evaluation.
- Stronger safety constraints and automatic regression checks.
- Improved integration surface for downstream workflows.

**Phase 3 (Expansion):**
- Generalized "autoresearch for robotics control" pattern.
- Expanded optimization targets beyond lap time.
- Community standardization and broader reuse.

### Risk Mitigation Strategy

**Technical Risks:**
- Highest risk: long-run stability and correct promotion under metric noise.
- Mitigation: strict promotion gates, median-based evaluation, watchdogs, resumable runs, mandatory simulator runtime validation.

**Market Risks:**
- Risk: convincing demos without practical reproducibility.
- Mitigation: reproducibility as hard acceptance criterion with independent rerun evidence.

**Resource Risks:**
- Risk: limited maintainer capacity for broad feature delivery.
- Mitigation: strict phase-1 scope freeze and defer non-essential work to phase 2/3.

## Functional Requirements

### Run Setup & Environment Readiness

- FR1: Maintainer can initialize a clean project environment using documented setup steps.
- FR2: Maintainer can validate simulator and dependency readiness before starting optimization.
- FR3: Maintainer can detect and receive actionable feedback for setup misconfiguration.
- FR4: Maintainer can run the full pipeline from a clean checkout without undocumented manual steps.

### Experiment Orchestration

- FR5: Maintainer can start an autonomous optimization run for DonkeyCar steering models.
- FR6: System can execute bounded iterative experiment cycles under a fixed runtime budget.
- FR7: System can evaluate each iteration and decide keep or discard based on configured criteria.
- FR8: System can persist decision history for each iteration in a traceable format.
- FR9: System can rollback to the latest stable model state after regressive outcomes.
- FR10: Maintainer can stop and safely resume long-running optimization workflows.

### Simulator Evaluation & Driving Performance

- FR11: System can evaluate candidate models in the simulator under deterministic conditions.
- FR12: Maintainer can view lap-time trend over optimization iterations.
- FR13: Maintainer can view cross-track-error trend alongside lap-time results.
- FR14: System can compare current iteration outcomes against a defined baseline.
- FR15: System can reject candidate models that fail defined stability or quality criteria.

### Model Promotion & Operability Gating

- FR16: System can promote a candidate model only after metric-based acceptance conditions are met.
- FR17: System can enforce a mandatory post-training simulator runtime validation before promotion.
- FR18: Maintainer can verify that promoted models load and execute inference successfully in simulator runtime.
- FR19: System can block promotion if simulator operability validation fails.
- FR20: Maintainer can identify the currently promoted best model and its promotion rationale.

### Model Export & Integration Artifacts

- FR21: Maintainer can export the promoted model in ONNX format.
- FR22: System can associate exported models with run metadata and decision trace.
- FR23: Integration user can retrieve best-model artifacts, metric history, and configuration snapshot from a run.
- FR24: Integration user can validate export usability through a documented verification flow.

### Observability, Logging & Auditability

- FR25: Maintainer can inspect structured logs for training, evaluation, promotion decisions, and failures.
- FR26: Maintainer can retrieve a run manifest containing configuration, simulator context, and key outputs.
- FR27: Support user can trace regressions to specific iterations and decision events.
- FR28: Maintainer can audit model promotion history with linked evidence.
- FR29: System can preserve sufficient run context for reproducibility checks across environments.

### Reliability, Recovery & Operations

- FR30: Operations user can monitor run health throughout overnight execution.
- FR31: System can detect run interruption conditions and surface failure state clearly.
- FR32: System can apply recovery behavior for long-running workflow interruptions.
- FR33: Operations user can confirm completion status and artifact completeness of scheduled runs.
- FR34: System can execute repeated runs using consistent evaluation context for reproducibility validation.

### Documentation & Maintainer Enablement

- FR35: Maintainer can follow documented commands to run setup, training, evaluation, promotion, and export.
- FR36: Maintainer can follow troubleshooting guidance for common runtime and reproducibility failures.
- FR37: New contributor can onboard to the optimization workflow within one working session.
- FR38: Maintainer can understand migration from ad-hoc workflow to the standardized reproducible workflow.
- FR39: Maintainer can identify which capabilities are Phase 1 versus post-MVP from documentation.

### Scope Governance

- FR40: Product team can distinguish Phase 1 must-have capabilities from Phase 2/3 deferred features.
- FR41: Product team can verify that every implemented feature maps to an approved functional requirement.
- FR42: Product team can identify and manage capability gaps before implementation planning.

## Non-Functional Requirements

### Performance

- NFR1: A full optimization iteration (train + eval + decision) must complete within the configured fixed runtime budget.
- NFR2: Post-training simulator operability validation must complete within a bounded validation window defined by run configuration.
- NFR3: Metric logging and decision trace generation must be available before the next iteration begins.
- NFR4: Overnight runs must sustain continuous iteration without unbounded slowdown across the planned cycle count.

### Reliability

- NFR5: The system must complete scheduled overnight runs with automatic recovery from transient simulator or process interruptions.
- NFR6: All run-critical stages (setup check, training, evaluation, promotion gate, export) must produce explicit success/failure states.
- NFR7: Promotion decisions must be deterministic under identical inputs, configuration, simulator version, and seed strategy.
- NFR8: The system must support resumable execution after interruption without losing prior accepted model state or decision history.
- NFR9: Reproducibility verification reruns must produce directionally consistent lap-time trend outcomes under fixed evaluation conditions.

### Security

- NFR10: Only authorized maintainers can trigger promotion of a new best model.
- NFR11: Run artifacts (models, manifests, decision logs) must be tamper-evident and traceable to a specific run context.
- NFR12: Configuration and artifact paths used in automated runs must be validated to prevent unsafe execution targets.
- NFR13: Model promotion evidence (metrics + operability gate result) must be immutable once recorded for auditability.

### Integration

- NFR14: Simulator integration must use a stable, version-pinned execution context for reproducible evaluations.
- NFR15: ONNX export outputs must conform to a documented interface contract consumable by downstream validation workflows.
- NFR16: Per-run artifact bundles must include model, config snapshot, metric history, and promotion decision trace in a consistent schema.
- NFR17: Integration failures (sim connection, export validation, artifact write) must surface actionable diagnostics for maintainers.
