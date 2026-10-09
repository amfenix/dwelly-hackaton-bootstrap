"""Every model call goes through a `Recorder`: it is timed, counted, priced and kept.

The record is the trace a reader checks, and it is also what `--replay` runs from: a recorder
built with `replay=` answers each request with the response recorded for it, so a run can be
reproduced without keys or a network, through the same policy and the same functions.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

from desk.errors import ModelError

PRICES_PER_MTOK: dict[str, tuple[float, float]] = {
    # USD per million input / output tokens, from platform.claude.com/docs/en/about-claude/pricing
    # (read 2026-10-09). Cache reads and writes are not counted separately.
    "claude-fable-5-1": (10.0, 50.0),
    "claude-opus-5-5": (4.0, 20.0),
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-haiku-4-5-20251001": (1.0, 5.0),
}


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float | None:
    """What a call cost, or `None` for a model without a known price (Jev reports none)."""
    price = PRICES_PER_MTOK.get(model)
    if price is None:
        return None
    return (input_tokens * price[0] + output_tokens * price[1]) / 1_000_000


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class CallRecord:
    kind: str  # classify | generate | agent
    backend: str  # claude | jev
    model: str
    request: dict[str, Any]
    response: dict[str, Any] | None
    seconds: float
    input_tokens: int
    output_tokens: int
    cost_usd: float | None
    error: str | None = None
    replayed: bool = False
    digest: str = ""

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, raw: dict[str, Any]) -> CallRecord:
        return cls(**raw)


def digest(kind: str, request: dict[str, Any]) -> str:
    canonical = json.dumps({"kind": kind, "request": request}, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


Invoke = Callable[[], tuple[dict[str, Any], Usage, str]]
"""Makes the real call: returns the response as JSON, its usage, and the model that answered."""


@dataclass
class Recorder:
    replay: list[CallRecord] | None = None
    calls: list[CallRecord] = field(default_factory=list)
    clock: Callable[[], float] = time.perf_counter
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def call(self, kind: str, backend: str, model: str, request: dict[str, Any], invoke: Invoke) -> dict[str, Any]:
        key = digest(kind, request)
        if self.replay is not None:
            return self._replayed(kind, backend, model, request, key)
        started = self.clock()
        try:
            response, usage, answered_by = invoke()
        except ModelError as error:
            self._keep(
                CallRecord(
                    kind,
                    backend,
                    model,
                    request,
                    None,
                    self.clock() - started,
                    0,
                    0,
                    None,
                    error=str(error),
                    digest=key,
                )
            )
            raise
        record = CallRecord(
            kind,
            backend,
            answered_by or model,
            request,
            response,
            self.clock() - started,
            usage.input_tokens,
            usage.output_tokens,
            cost_usd(answered_by or model, usage.input_tokens, usage.output_tokens),
            digest=key,
        )
        self._keep(record)
        return response

    def _replayed(self, kind: str, backend: str, model: str, request: dict[str, Any], key: str) -> dict[str, Any]:
        assert self.replay is not None
        with self._lock:
            for index, record in enumerate(self.replay):
                if record.digest == key:
                    self.replay.pop(index)
                    break
            else:
                raise ModelError("replay", f"no recorded {kind} call matches this request ({key[:12]})")
        kept = CallRecord(**{**asdict(record), "replayed": True})
        self._keep(kept)
        if record.error is not None or record.response is None:
            raise ModelError("replay", f"the recorded call failed: {record.error}")
        return record.response

    def unused(self) -> int:
        """Recorded calls a replay has not made again: more than none means the run diverged."""
        with self._lock:
            return len(self.replay or [])

    def _keep(self, record: CallRecord) -> None:
        with self._lock:
            self.calls.append(record)

    def totals(self) -> dict[str, Any]:
        costs = [c.cost_usd for c in self.calls if c.cost_usd is not None]
        return {
            "calls": len(self.calls),
            "seconds": round(sum(c.seconds for c in self.calls), 3),
            "input_tokens": sum(c.input_tokens for c in self.calls),
            "output_tokens": sum(c.output_tokens for c in self.calls),
            "cost_usd": round(sum(costs), 6) if costs else None,
            "models": sorted({c.model for c in self.calls}),
        }
