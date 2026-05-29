"""Shared LLM client: OpenAI-compatible HTTP call + JSON repair loop.

Talks to any OpenAI ``/v1/chat/completions``-compatible endpoint. By default
that is the local vLLM mock (``nse/scripts/vllm_mock.py``); point
``LLMConfig.base_url`` at a real vLLM host on Linux/WSL for production.
"""

from __future__ import annotations

import json
from typing import Any, Optional, Type, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from nse.agents.prompts import REPAIR_SYSTEM
from nse.config import SETTINGS

T = TypeVar("T", bound=BaseModel)


class AgentError(RuntimeError):
    """Raised when an agent output cannot be parsed even after repair."""


def llm_available(base_url: Optional[str] = None, timeout: float = 3.0) -> bool:
    """True if an OpenAI-compatible host is reachable at ``base_url`` (or the
    configured default). Used to gate real-LLM integration tests."""
    base = base_url or SETTINGS.llm.base_url
    try:
        resp = httpx.get(f"{base}/models", timeout=timeout)
        return resp.status_code < 500
    except httpx.HTTPError:
        return False


class LLMClient:
    def __init__(self, base_url: Optional[str] = None) -> None:
        self.cfg = SETTINGS.llm
        self.base_url = base_url or self.cfg.base_url
        self.tokens_used = 0  # cumulative total_tokens; reset per task

    def reset_tokens(self) -> None:
        self.tokens_used = 0

    def _chat(self, system: str, user: str, temperature: float) -> str:
        payload = {
            "model": self.cfg.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": self.cfg.max_output_tokens,
        }
        last_exc: Exception | None = None
        for _ in range(self.cfg.max_retries + 1):
            try:
                resp = httpx.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    timeout=self.cfg.request_timeout_s,
                )
                resp.raise_for_status()
                data = resp.json()
                usage = data.get("usage") or {}
                self.tokens_used += int(usage.get("total_tokens", 0) or 0)
                return data["choices"][0]["message"]["content"]
            except (httpx.HTTPError, KeyError, IndexError) as exc:
                last_exc = exc
        raise AgentError(f"LLM request failed: {last_exc}")

    @staticmethod
    def _extract_json(text: str) -> Any:
        """Tolerate code fences / leading prose around the JSON payload."""
        text = text.strip()
        if text.startswith("```"):
            text = text.strip("`")
            text = text[text.find("\n") + 1 :] if "\n" in text else text
        start = min(
            (i for i in (text.find("["), text.find("{")) if i != -1),
            default=0,
        )
        return json.loads(text[start:])

    def call_list(
        self,
        system: str,
        user: str,
        schema: Type[T],
        temperature: float,
    ) -> list[T]:
        """Call the LLM expecting a JSON array; validate each item.

        Performs one auto-repair attempt on parse/validation failure, per the
        blueprint parsing policy.
        """
        raw = self._chat(system, user, temperature)
        for attempt in range(2):  # original + one repair
            try:
                data = self._extract_json(raw)
                if isinstance(data, dict):
                    data = [data]
                return [schema.model_validate(item) for item in data]
            except (json.JSONDecodeError, ValidationError, TypeError) as exc:
                if attempt == 0:
                    raw = self._chat(
                        REPAIR_SYSTEM,
                        f"{user}\n\nPREVIOUS_OUTPUT:\n{raw}\n\nERROR:\n{exc}",
                        temperature=0.0,
                    )
                else:
                    raise AgentError(f"unrepairable agent output: {exc}")
        raise AgentError("unreachable")
