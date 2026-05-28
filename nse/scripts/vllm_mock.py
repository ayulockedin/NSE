"""vLLM mock — OpenAI-compatible server returning canned agent JSON.

Lets the orchestrator be exercised end-to-end with no GPU and no real model.
It inspects the system prompt to decide whether to answer as PLANNER,
SIMULATOR, or CRITIC, and emits schema-valid JSON.

Run:  python -m nse.scripts.vllm_mock   (serves on 127.0.0.1:8080)
"""

from __future__ import annotations

import json
import uuid

from fastapi import FastAPI, Request

app = FastAPI(title="nse-vllm-mock")


def _planner_payload() -> list[dict]:
    base = [
        {
            "strategy": "Add a minimal review comment in the existing file.",
            "complexity": 0.05,
            "conf": 0.95,
        },
        {
            "strategy": "Extract a new validation module and call it.",
            "complexity": 0.5,
            "conf": 0.6,
        },
        {
            "strategy": "Use a third-party validation library.",
            "complexity": 0.4,
            "conf": 0.55,
        },
    ]
    out = []
    for b in base:
        bid = str(uuid.uuid4())
        out.append(
            {
                "branch_id": bid,
                "strategy": b["strategy"],
                "edited_files": ["calc.py"],
                "patch_preview": (
                    "--- a/calc.py\n+++ b/calc.py\n@@ -1,4 +1,5 @@\n"
                    ' """Tiny module under test for the NSE end-to-end smoke test."""\n'
                    "+# reviewed\n \n \n def add(a, b):\n"
                ),
                "full_file_rewrites": None,
                "assumptions": ["inputs are numeric"],
                "expected_complexity": b["complexity"],
                "planner_confidence": b["conf"],
            }
        )
    return out


def _simulator_payload(branch_ids: list[str]) -> list[dict]:
    # First (simplest) branch gets the strongest analytic prior.
    priors = [0.95, 0.65, 0.6]
    return [
        {
            "branch_id": bid,
            "p_t_sim": priors[i] if i < len(priors) else 0.6,
            "explanation": "Change is local and well-covered by tests.",
        }
        for i, bid in enumerate(branch_ids)
    ]


def _critic_payload(branch_ids: list[str]) -> list[dict]:
    risks = [0.02, 0.17, 0.14]
    return [
        {
            "branch_id": bid,
            "r_critic": risks[i] if i < len(risks) else 0.15,
            "identified_failures": [] if i == 0 else ["no handling for non-numeric input"],
            "attack_confidence": 0.4,
        }
        for i, bid in enumerate(branch_ids)
    ]


def _branch_ids_from_user(user_content: str) -> list[str]:
    try:
        start = user_content.find("{")
        data = json.loads(user_content[start : user_content.rfind("}") + 1])
        branches = data.get("branches", [])
        return [b["branch_id"] for b in branches]
    except Exception:
        return [str(uuid.uuid4()) for _ in range(3)]


@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> dict:
    body = await request.json()
    messages = body.get("messages", [])
    system = next((m["content"] for m in messages if m["role"] == "system"), "")
    user = next((m["content"] for m in messages if m["role"] == "user"), "")

    if system.startswith("You are PLANNER"):
        content = json.dumps(_planner_payload())
    elif system.startswith("You are SIMULATOR"):
        content = json.dumps(_simulator_payload(_branch_ids_from_user(user)))
    elif system.startswith("You are CRITIC"):
        content = json.dumps(_critic_payload(_branch_ids_from_user(user)))
    else:
        content = "[]"

    return {
        "id": f"mock-{uuid.uuid4()}",
        "object": "chat.completion",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}}],
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8080, log_level="warning")
