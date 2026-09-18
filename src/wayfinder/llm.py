"""Optional OpenAI-compatible LLM client (stdlib urllib only).

Defaults to OFF so the project deploys with zero keys. Point it at any
OpenAI-compatible endpoint — e.g. LM Studio's local server:

    LLM_BASE_URL=http://localhost:1234/v1
    LLM_API_KEY=lm-studio
    LLM_MODEL=<a loaded model>

Used for LLM-mode planning (thought + action selection). The heuristic
planner remains the fallback, so the agent never dead-ends.
"""

from __future__ import annotations

import json
import os
import urllib.request
from typing import List, Optional


class LLMError(Exception):
    pass


class LLM:
    def __init__(self) -> None:
        self.base_url = (os.environ.get("LLM_BASE_URL") or "").rstrip("/")
        self.api_key = os.environ.get("LLM_API_KEY") or ""
        self.model = os.environ.get("LLM_MODEL") or ""
        self.timeout = int(os.environ.get("LLM_TIMEOUT", "60"))

    @property
    def enabled(self) -> bool:
        return bool(self.base_url and self.model)

    def describe(self) -> str:
        return f"{self.model} @ {self.base_url}" if self.enabled else "disabled (heuristic planner)"

    def chat(self, messages: List[dict], temperature: float = 0.2, max_tokens: int = 700) -> str:
        if not self.enabled:
            raise LLMError("LLM not configured")
        body = json.dumps(
            {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=body,
            headers={
                "Content-Type": "application/json",
                **({"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}),
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            return payload["choices"][0]["message"]["content"] or ""
        except Exception as exc:  # noqa: BLE001 - surfaced to caller as LLMError
            raise LLMError(str(exc)) from exc

    def complete_json(self, system: str, user: str) -> Optional[dict]:
        """Ask the model for a strict JSON object; parse defensively."""
        raw = self.chat(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]
        )
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.strip("`")
            if raw.lower().startswith("json"):
                raw = raw[4:]
        start, end = raw.find("{"), raw.rfind("}")
        if start == -1 or end == -1:
            return None
        try:
            return json.loads(raw[start : end + 1])
        except json.JSONDecodeError:
            return None
