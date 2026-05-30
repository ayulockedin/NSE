# Neural Shadow Executor (NSE)

A symbolic-first, uncertainty-aware, adversarially stressed, sandboxed
**anticipatory patch-selection engine**. NSE generates *k* candidate patches,
simulates and scores them with a hybrid simulator (LLM + latent ensemble),
applies a deterministic arbiter that enforces compile authority and safe
uncertainty routing, and executes only the highest-value branch in an isolated
sandbox — logging every outcome to continuously calibrate and audit itself.

This repository implements the **Day-1 foundation** from the build-ready
blueprint: a complete, runnable pipeline from task → Planner → symbolic gate →
Simulator → latent ensemble → Critic → Arbiter → sandbox → DB.

## Pipeline

```
task ─▶ Memory Graph (CPG-lite) ─▶ Planner (k branches)
     ─▶ per branch:  apply patch ─▶ symbolic gate (p_c) ─▶ Simulator (p_t_sim)
                     ─▶ Latent ensemble (p_t_latent, r_long, u) ─▶ Critic (r_critic)
     ─▶ Arbiter: score S(B) + route ─▶ sandbox-execute best ─▶ log outcome
```

### Decision math (canonical, in `nse/orchestrator/arbiter.py`)

```
p_t    = α·p_t_latent + (1-α)·p_t_sim
u_norm = u / 0.25
S(B)   = (p_c · p_t · c_planner) − λ1·r_critic − λ2·r_long − λ3·u_norm·(1−p_t)
```

Routing per branch: `p_c==0 → prune` · `u > u_max → incremental sandbox (never
prune on ignorance)` · `S < τ_prune → prune` · else `execute`.

Defaults (`nse/config.py`): `k=3, M=5, α=0.7, λ1=1.0, λ2=0.5, λ3=0.75,
τ_prune=0.4, u_max=0.15`.

## Layout

```
nse/
├─ orchestrator/   app.py · orchestrator.py · executor.py · patcher.py · arbiter.py · schemas.py
├─ memory_graph/   extractor.py (tree-sitter/ast) · graph_db.py (NetworkX) · context.py
├─ agents/         base_client.py · planner/simulator/critic_client.py · prompts.py
├─ models/         latent_model.py (GNN ensemble) · calibrate.py (ECE/Brier/temp/isotonic)
├─ data/           mutate.py (AST mutation engine) · dataset.py (labeled-example builder)
├─ eval/           harness.py (Brier/ECE/accuracy/AUC scoreboard)
├─ db/             schema.sql · db_client.py (SQLite)
├─ tools/          static_checks.py (ruff + mypy + ast gate)
├─ scripts/        vllm_mock.py · run_demo.py · run_eval.py · run_dev.ps1 · start_vllm.sh
└─ sandbox_repos/  toy_repo/ (smoke target) · mutation_seed/ (dataset seed)
```

## Setup

```powershell
python -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\venv\Scripts\python.exe -m pip install -e .
```

> **vLLM** has no native Windows wheels. On Windows the inference host is the
> mock (`nse/scripts/vllm_mock.py`); on Linux/WSL run `nse/scripts/start_vllm.sh`
> and point `LLMConfig.base_url` at it. The agent clients speak the OpenAI
> `/v1/chat/completions` protocol either way.

## Run the demo

```powershell
# terminal 1 — canned inference host
.\venv\Scripts\python.exe -m nse.scripts.vllm_mock

# terminal 2 — drive a task end-to-end against the toy repo
.\venv\Scripts\python.exe -m nse.scripts.run_demo
```

Expected: 3 branches generated, the simplest routed to **execute**, the rest
pruned, and the sandbox reporting tests passing.

Or run the orchestrator API: `.\nse\scripts\run_dev.ps1` then
`POST http://127.0.0.1:8000/tasks`.

## Tests

```powershell
.\venv\Scripts\python.exe -m pytest -q
```

## Security

Sandbox runs are Docker-isolated: `--network none`, all caps dropped,
`no-new-privileges`, non-root, read-only base FS, CPU/memory limits. Branches
touching sensitive paths (`.env`, credentials) are auto-pruned. A local
subprocess fallback exists **for development only** (when Docker is absent) and
must never run untrusted patches.

Because the sandbox runs with no network, test dependencies must be baked into
the image (you can't `pip install` at run time). Build it once:

```powershell
docker build -f nse/docker/Dockerfile.sandbox -t nse-sandbox:latest .
# override per-repo with the NSE_SANDBOX_IMAGE env var
```

Verify the hardened path (daemon, non-root, network-off, real test run):

```powershell
.\venv\Scripts\python.exe -m nse.scripts.verify_docker_sandbox
```

The Docker tests in `tests/test_sandbox_docker.py` auto-skip when Docker or the
image is absent.

## Evaluation (latent-model scoreboard)

The data engine mutates a known-good repo and labels each mutant by running it
through the sandbox (ground truth, never assumed), producing
`(features -> tests_passed)` examples. The eval harness then scores the latent
model's calibration (Brier/ECE) and discrimination (accuracy/AUC):

```powershell
.\venv\Scripts\python.exe -m nse.scripts.run_eval --force-local   # skip Docker
.\venv\Scripts\python.exe -m nse.scripts.run_eval                 # Docker = trusted labels
```

## Train the latent model

```powershell
.\venv\Scripts\python.exe -m nse.models.train --rebuild --force-local
```

This builds a labeled dataset from the seed modules, fits the M-head GNN
ensemble (per-head bootstrap → ensemble variance gives epistemic uncertainty),
and reports held-out metrics for the trained model vs the heuristic baseline.
Weights are saved to `nse/models/weights/latent.pt`; the orchestrator
auto-loads them when present and falls back to the heuristic otherwise.

Held-out result (66 mutants across three seed modules), **trained vs heuristic**:
AUC 0.78 → **0.85**, Brier 0.27 → **0.10**, ECE 0.22 → **0.08**,
accuracy 0.35 → **0.90**.

## Status (built) - Currently in Maintenance Phase
```
