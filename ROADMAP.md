# NSE Roadmap — The Trust Layer for Autonomous Code Change

> **Vision.** Evolve NSE from *"predict which patch passes"* into a **calibrated,
> cost-aware, self-improving verification & governance substrate**: a decision
> layer that allocates scarce verification budget under *provable* uncertainty
> bounds, knows the quality of its own test oracle, audits its own decisions, and
> gets cheaper, safer, and more autonomous every time it runs — while knowing its
> own limits.
>
> **Positioning.** Not "another coding agent." The trust/governance layer any
> agent (SWE-agent, Aider, Claude Code, …) plugs into.
>
> **One-line thesis.** *Ride the LLM frontier as a cheap, calibrated surrogate —
> never compete with it head-on.*

---

## Where we are (done)

The autonomous decision/learning loop is closed end-to-end and verified
(**69 tests pass**, ruff clean, Docker sandbox verified):

`memory graph → Planner → symbolic gate → Simulator → latent GNN ensemble →
Critic → Arbiter → hardened sandbox → SQLite logging → calibration → audit`.

- **Real neural core** — trained GNN on per-edited-function CPG-lite features;
  leave-one-file-out AUC **0.946** vs heuristic 0.889; supervised `r_long`.
- **Decision-level proof** — arbiter selection: trained executes 71% of tasks at
  **0.966 precision / 0.917 safe-abstain**; the heuristic never acts.
- **Self-monitoring** — calibration loop (ECE-triggered isotonic recalibration)
  + audit re-execution that catches false-negative prunes.

**The one big gap:** the LLM agents (Planner/Simulator/Critic) still run on the
canned mock, not a real model.

---

## The level-up: weakness → strength → mastery

| Hard truth (today) | The flip (strength) | Mastery |
|---|---|---|
| Lives or dies by test quality | Coverage-as-uncertainty: *know* when the oracle is weak; generate tests when the net is thin | Conformal EXECUTE gate with distribution-free coverage guarantees |
| Mutation ≠ real bugs | Mutation as *pretraining*; fine-tune on real defects | Report % resolved on SWE-bench / Defects4J |
| Value only when execution is costly | Reframe as a **verification-budget allocator** | Cost-aware acquisition (max info-gain per dollar) |
| A strong LLM may subsume the GNN | Cost-tiered **cascade**: GNN screens → LLM judges → sandbox verifies | Distill the LLM oracle into the GNN — improve *as the LLM improves* |
| `r_long` is a hand-wave | Measure it from git history (revert/rework within horizon) | Multi-horizon fragility / tech-debt prediction |
| Crowded field | Be the governance substrate, not an agent | Competence-aware autonomy (earns trust per segment) |

---

## North-star metrics

Track every phase against these — they replace "mutation AUC" as the scoreboard.

| Metric | Now | Target |
|---|---|---|
| % resolved (SWE-bench Verified) | n/a (mutation only) | establish, then climb |
| Execute precision (pick actually passes) | 0.966 (toy) | ≥ 0.95 on real defects |
| Safe-abstain (no-op when nothing works) | 0.917 | ≥ 0.95 |
| Audit false-negative rate | measure | drive ↓ over time |
| Cost per resolved patch (sandbox runs + LLM tokens) | unmeasured | baseline → minimize |
| Calibration guarantee (conformal coverage) | none | 95% guaranteed |
| Autonomy rate (auto-merged, no human) by segment | 0 | grow where calibrated |

---

## Phased plan

Each phase notes **Objective · Why · Key work · Effort · Risk · Depends on**.
Phases 7–8 are CPU-only and build directly on shipped code → **start here**.

### Phase 7 — Honest Oracle  ⭐ START HERE  *(mostly no-LLM)*
**Objective:** make NSE trust its oracle exactly as much as the oracle deserves,
and build a net when there isn't one.
**Why:** the #1 real-world blocker is poor test coverage; turning that into
"the system *knows* it's under-tested" is the sharpest differentiator.
- **7.1 Coverage-as-uncertainty** — ✅ **DONE**. `nse/tools/coverage_signal.py`
  measures pristine-repo line coverage (coverage.py, graceful no-op if absent),
  maps a patch to the *old-file* executable lines it touches, and converts low
  coverage → `coverage_u`. `arbiter.decide(coverage_u=)` routes under-tested
  changes to `INCREMENTAL_SANDBOX` (extends "never prune on uncertainty"). Only
  *executable* lines count (comments/blanks excluded). 82 tests pass.
- **7.2 Conformal EXECUTE gate** — ✅ **DONE**. `calibrate.fit_conformal_threshold`
  finds the smallest p_t threshold whose executed-set false rate is ≤ α with
  confidence 1−δ (Clopper-Pearson via scipy, Hoeffding fallback); `CONFORMAL_NEVER`
  when unachievable, `None` on sparse data. The calibration loop fits it on the
  *effective* (recalibrated) probs and persists `weights/conformal.json`;
  `arbiter.decide(conformal_threshold=)` routes sub-threshold branches to
  `INCREMENTAL_SANDBOX`. 90 tests pass.
- **7.3 Property-based + metamorphic oracle** — ✅ **DONE**.
  `nse/tools/property_oracle.py` differential-tests the patched function vs the
  pristine golden on fuzzed inputs; a *new crash* (patched raises where original
  succeeded) is unambiguous breakage. Runs **inside the sandbox** via a
  self-contained generated harness (patched code is untrusted). Orchestrator runs
  it for `INCREMENTAL_SANDBOX`-routed branches → new crash routes to PRUNE
  (`ORACLE_CRASH`). 102 tests pass.
- **7.4 LLM-proposed tests** — generate targeted tests for the changed function.
  *(depends on Phase 9)*

### Phase 8 — Self-Improving Flywheel  *(no-LLM)*  ✅ **DONE**
**Objective:** make the system get smarter where it's weakest, automatically.
- **8.1 Audit → priority retraining** — ✅ `nse/data/replay.py
  harvest_training_examples(db, audit_dir)` reconstructs LabeledExamples
  (features + per-function CPG + r_long) from executed outcomes **and audit-found
  false negatives**, up-weighted by prediction error (`example_weight`).
  `LabeledExample.weight` + weighted loss in `train()`; `train.py --replay`;
  `db.training_signals()`.
- **8.2 Richer sandbox signal** — ✅ `SandboxRun` now carries
  `n_passed`/`n_failed`/`failed_tests` (parsed via `-rfE`), capturing which test
  caught the regression.
- **8.3 Drift-triggered retrain** — ✅ `calibration_loop.detect_drift` flags when
  raw-model ECE exceeds `drift_ece_threshold` (recalibration can't keep up →
  retrain); orchestrator notes it + logs `full_retrain_recommended`.
  *(114 tests pass.)*

### Phase 9 — Real LLM Agents  ✅ **DONE** *(ollama / qwen2.5-coder:7b on RTX 4060)*
**Objective:** swap the mock for a real model. **Achieved + hardened for real
models:**
- `NSE_LLM_BASE_URL` / `NSE_LLM_MODEL` / `NSE_LLM_TIMEOUT_S` env overrides (mock
  stays the default so CI is offline). `nse/scripts/check_llm.py` validator.
- Agents return **schema-valid JSON** from the real model (Planner/Simulator/
  Critic verified); `tests/test_llm_integration.py` (gated, runs on a live host).
- **Token budget enforced**: `LLMClient.tokens_used` + orchestrator `check_budget`
  raises over `max_total_llm_tokens_per_task`.
- **Real-model hardening:** planner gets the **current file source** in context
  (so full rewrites are faithful); planner re-assigns fresh `branch_id`s (models
  echo the example UUID); `CriticReport.identified_failures` tolerates object/
  string shapes; orchestrator **degrades gracefully** if any agent emits
  unrepairable JSON (no task crash).
- **Honest finding:** end-to-end the pipeline runs, but real LLM patches get
  pruned — the latent model (trained on synthetic toy mutants) under-scores real
  patches (`p_t≈0.41`) and the arbiter prunes below `tau_prune`. The wiring is
  correct; this is the **toy→real model gap → motivates Phase 11**, not a bug.
- Prefix caching: ollama caches the shared prompt prefix itself; explicit vLLM
  prefix-cache verification deferred to a vLLM deployment.

### Phase 10 — The Cascade & Distillation  *(the future-proofing move)*  ✅ COMPLETE (10.1–10.3)
**Objective:** make the neural core durable against ever-better LLMs.
**Why:** the GNN's job becomes "cheap, calibrated filter that saves expensive
LLM/sandbox calls" — a role that survives every model upgrade.
- **10.1 Cost-tiered cascade** — ✅ **WIRED into `Orchestrator.run_task`**.
  Tier 1 GNN screen (`_screen_branch`, no LLM) routes with the *optimistic* sim
  prior (S(B) is monotone in p_t, so sim=1.0 prunes only the unrescuable — recall
  preserved); Tier 2 calls the Simulator/Critic **only on survivors**; Tier 3
  sandboxes the single finalist. Token budget now reflects only judged survivors;
  `TaskReport` carries `branches_screened/judged` + `cost_units`. Standalone spec +
  cost primitives in `nse/orchestrator/cascade.py` / `cost.py`.
- **10.2 Cost-aware acquisition** — ✅ **WIRED into `Orchestrator.run_task`**.
  When the cascade finds no EXECUTE finalist, it spends remaining sandbox budget
  on the most-informative uncertain (INCREMENTAL_SANDBOX) branches — ranked by
  `info_gain` per run (binary-entropy of p_t + epistemic u) — and the first that
  passes resolves the task, bounded by `max_sandbox_executions_per_task`.
  Primitives in `cost.py`; per-task spend recorded in the ledger (`cost_units`).
- **10.3 LLM→GNN distillation** — ✅ **DONE** (`nse/data/distill.py` +
  `train()` distillation MSE). The simulator's `p_t_sim` becomes a per-row
  `soft_label`; the GNN is trained to match it alongside the sandbox ground truth.
  **Real run (qwen2.5-coder:7b judged 125 mutants; held-out, distilled never saw
  these soft labels):** GNN↔teacher MAE 0.325→**0.300** (tracks the LLM closer)
  *and* ground-truth AUC 0.938→**0.963**, Brier 0.055→**0.047** — even a noisy
  teacher (qwen 76% accurate vs sandbox) net-improved the cheap GNN.

### Phase 11 — Real-World Grounding  *(parallel data track)*  ✅ **FOUNDATION DONE**
**Objective:** close the toy→real gap; make `r_long` a measured capability.
- **11.1 Real defect datasets** — ✅ **git PR-pair mining** shipped
  (`nse/data/git_mining.py`). SWE-bench-style protocol: find fix commits, rebuild
  the buggy tree by reverting only *source* files to the parent (fix-added tests
  stay, so the defect is observable), then **verify by execution** — keep a pair
  only if the buggy tree fails and the fixed tree passes in the sandbox. Each
  verified pair → a `real_fix` (label 1, up-weighted) + a `real_regression`
  (label 0), in the exact `LabeledExample` schema (no train/serve skew). Trees are
  materialised via `git archive` (read-only; working tree untouched). CLI:
  `python -m nse.data.git_mining --repo PATH --out real.jsonl`.
  *External corpora (SWE-bench/Defects4J/BugsInPy) still TODO — the miner is the
  general engine; point it at any clone (the sandbox image must carry that repo's
  deps, else `--force-local`).*
- **11.2 Curriculum** — ✅ `train.curriculum_train`: pretrain on the abundant
  synthetic mutants, then **fine-tune on real defects** (warm-started, lower lr,
  real examples up-weighted via `DEFAULT_REAL_WEIGHT`). CLI: `--real real.jsonl
  --curriculum`; reports a **real-defect held-out** comparison (the honest metric
  the synthetic held-out can't show).
- **11.3 Measured `r_long`** — ✅ `git_mining.measured_r_long`: temporal fragility
  = how often the fixed function is reworked in the commits that follow (replaces
  the structural `blast_radius` proxy for real examples; falls back to it when no
  function is attributed).
- **11.4 Report % resolved** — ✅ `nse/eval/real_defect_eval.py`: regroups mined
  examples into per-defect tasks and asks whether the *canonical* arbiter would
  EXECUTE the real fix. Reports **% resolved**, false-fix rate, abstain rate.
  Baseline on a toy 2-defect repo: heuristic resolves **0%** / abstains **100%** —
  the toy→real gap, made measurable.
- **Real corpus mined + evaluated, 2 libs / 33 verified pairs** (more-itertools 30
  + boltons 3; jmespath & toolz yielded 0 — their pre-py3.13 history isn't green,
  so the fixed tree never verified; *zero false pairs* throughout). **% resolved,
  leave-one-pair-out CV (leakage-free):** heuristic **0/33**, toy GNN
  (synthetic-only) **10/33 (0.30)**, curriculum GNN **18/33 (0.55)**; mean p_t on
  held-out real fixes 0.338→**0.674**. Curriculum grounding **~doubled % resolved**
  — the toy→real gap measured and substantially closed. measured r_long nonzero
  10/33 (max 1.0).
- **Lesson:** mining *old* repos needs per-commit Docker envs (as SWE-bench ships);
  a single modern interpreter only verifies modern, clean-history libs.
- **Next:** SWE-bench/Defects4J (or more modern libs) with dep-carrying sandbox
  images for trusted+larger numbers; then ground the shipped `latent.pt` on the
  combined real corpus (LOPO already justifies it).

### Phase 12 — Mastery of Uncertainty & Governance  🟢 12.1–12.3 DONE
**Objective:** principled, auditable autonomy.
- **12.1 Epistemic vs aleatoric** — ✅ ensemble predictive-variance decomposition
  (epistemic = head variance `u`; aleatoric = mean per-head p(1-p)) on
  Latent/BranchPrediction. `arbiter.decide(aleatoric_max=)` escalates a would-
  EXECUTE **near-coin-flip** (high aleatoric + |p_t-0.5| ≤ `coin_flip_band`) to
  HUMAN_REVIEW — more evidence can't fix irreducible noise. Composes with 10.2
  (epistemic → evidence runs; aleatoric → human).
- **12.2 Competence-aware autonomy** — ✅ `nse/orchestrator/autonomy.py`: per-
  segment (by edited file) ECE + execute-precision from logged history →
  AUTO_MERGE / ASSISTED / HUMAN_REVIEW. Autonomy is *earned*; unknown segments
  default to HUMAN_REVIEW. Advisory (never changes what executes).
- **12.3 Conformal-backed safety invariants** — ✅ `nse/orchestrator/invariants.py`:
  I1 never-prune-on-uncertainty, I2 symbolic-fail-prunes, I3 conformal-execute,
  asserted in the orchestrator + a 3000-iter randomized property sweep over
  `decide()`.
- **12.4 Optimal-stopping framing** — model run-another-sandbox vs execute vs
  abstain as sequential decision-making. *(H · H, research — deferred; 10.2
  acquisition is the greedy approximation.)*

### Phase 13 — Productization & Scale
- Governance-substrate API: pluggable layer over any agent.
- Cap audit-snapshot retention; Postgres backend; multi-repo; dashboards.
*(ongoing)*

---

## Critical path & recommended build order

```
        ┌─ Phase 11 (data track) ───────────────────────────┐  (runs in parallel)
        │                                                    ▼
Phase 7 ─► Phase 8 ─► Phase 9 (real LLM) ─► Phase 10 (cascade) ─► Phase 12 ─► Phase 13
(oracle)   (flywheel)                       (+ 7.4 LLM oracle)    (mastery)   (product)
```

- **Do first (CPU-only, highest leverage, builds on shipped code):**
  7.1 coverage-as-uncertainty → 7.2 conformal gate → 8.1 audit→retrain flywheel.
- **Then unlock the frontier:** Phase 9 (real LLM) → Phase 10 (cascade +
  distillation) + 7.4.
- **In parallel from day one:** Phase 11 data track (it gates real credibility).
- **Capstone:** Phase 12 (provable, competence-aware autonomy).

> Note: vLLM was previously slated as the immediate next task. The level-up
> analysis reprioritizes it *behind* Phases 7–8 (which are higher-leverage and
> need no GPU) but elevates its importance — it's now the unlock for the cascade.
> If GPU time is free now, Phase 9 can be pulled forward without blocking 7–8.

---

## The flywheel (why it compounds)

```
deploy → capture outcomes + coverage + flaky signals
       → calibrate (conformal) + audit (find worst mistakes)
       → retrain on highest-information examples + distill the LLM oracle
       → model improves where it was weakest
       → competence-aware autonomy expands
       → deploy more  ↺
```

**The more it runs, the cheaper, safer, and more autonomous it gets — and it
knows its own limits.** That self-improving, self-limiting loop is the moat no
single LLM provides, and ~70% of its machinery already exists.

---

## Honest caveats (kept visible on purpose)

- None of this matters on codebases without *some* test signal — Phase 7 mitigates
  but cannot fully manufacture a missing oracle.
- The GNN is a swappable component; if distillation shows an LLM dominates even on
  cost, the cascade lets us demote the GNN gracefully. That's by design, not a
  failure.
- `r_long` and "% resolved" claims stay marked *aspirational* until measured on
  real defects (Phase 11).
