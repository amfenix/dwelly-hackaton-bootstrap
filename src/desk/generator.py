"""The generator: a system prompt and a prompt in, text out. For drafting replies and summaries."""

from __future__ import annotations

from typing import Any

from desk.calls import Recorder
from desk.classifier import claude_invoke, claude_text


class Generator:
    backend = "claude"

    def __init__(
        self, recorder: Recorder, model: str = "claude-sonnet-5-5", client: Any = None, max_tokens: int = 2000
    ) -> None:
        self.recorder, self.model, self.max_tokens = recorder, model, max_tokens
        self._client = client

    def client(self) -> Any:
        if self._client is None:  # pragma: no cover - needs the real SDK and a key
            import anthropic

            self._client = anthropic.Anthropic()
        return self._client

    def generate(self, system: str, prompt: str, *, model: str | None = None, max_tokens: int | None = None) -> str:
        request: dict[str, Any] = {
            "model": model or self.model,
            "max_tokens": max_tokens or self.max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": prompt}],
        }
        body = self.recorder.call(
            "generate", self.backend, request["model"], request, lambda: claude_invoke(self.client(), request)
        )
        return claude_text(body).strip()
