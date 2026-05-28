"""FastAPI entrypoint — local API + CLI for submitting NSE tasks.

    uvicorn nse.orchestrator.app:app --port 8000

POST /tasks  {"task": "...", "repo_path": "...", "target_files": [...]}
"""

from __future__ import annotations

from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel

from nse.orchestrator.orchestrator import Orchestrator

app = FastAPI(title="nse-orchestrator")
_orch: Optional[Orchestrator] = None


class TaskRequest(BaseModel):
    task: str
    repo_path: str
    target_files: Optional[list[str]] = None
    force_local_sandbox: bool = False


def _get_orchestrator(force_local: bool) -> Orchestrator:
    global _orch
    if _orch is None or _orch.force_local_sandbox != force_local:
        _orch = Orchestrator(force_local_sandbox=force_local)
    return _orch


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/tasks")
def submit_task(req: TaskRequest) -> dict:
    orch = _get_orchestrator(req.force_local_sandbox)
    report = orch.run_task(req.task, req.repo_path, req.target_files)
    return report.__dict__ | {
        "predictions": [p.model_dump() for p in report.predictions]
    }
