"""What one case's run produced: the record the answer, the reasoning and the trace are written from."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from desk.calls import CallRecord
from desk.machine import Entry, Journal
from desk.policy import Policy

ERROR_OUTCOME = "escalate"
"""The outcome of a run that did not end in a final state: a person picks it up."""

ESCALATING = frozenset({"escalate", "escalated", "emergency", "handover"})
"""Outcomes that hand the case to a person."""


@dataclass(frozen=True)
class Failure:
    """Why a run did not finish the way the policy meant. Short: it goes into the published answer."""

    phase: str
    """Where it broke: adapt, start, agent, stall, replay, deadline or batch."""
    error: str
    trigger: str | None = None
    """The last rule that fired before it broke, when one had."""

    @classmethod
    def of(cls, phase: str, error: BaseException, journal: Journal | None = None) -> Failure:
        last = None
        if journal is not None:
            last = next((e.data["name"] for e in reversed(journal.entries) if e.kind == "trigger"), None)
        return cls(phase, f"{type(error).__name__}: {error}", last)

    def __str__(self) -> str:
        after = f" (after rule {self.trigger})" if self.trigger else ""
        return f"{self.phase}: {self.error}{after}"

    def to_json(self) -> dict[str, Any]:
        return {"phase": self.phase, "error": self.error, "trigger": self.trigger}


@dataclass
class RunResult:
    id: str
    input: dict[str, Any]
    state: str
    final: bool
    """Whether `state` is one of the policy's final states. When it is not, the outcome is ERROR_OUTCOME."""
    outcome: str
    fields: dict[str, Any]
    journal: list[Entry]
    failure: Failure | None = None
    calls: dict[str, Any] = field(default_factory=dict)
    agent_text: list[str] = field(default_factory=list)
    summary: str | None = None
    summary_error: str | None = None

    @classmethod
    def unworked(
        cls, policy: Policy, raw: dict[str, Any], failure: Failure, journal: Journal | None = None
    ) -> RunResult:
        """A case that never got a machine: its answer still goes out, escalated, with the reason."""
        journal = journal or Journal()
        if not journal.of("error"):
            journal.add("error", phase=failure.phase, error=failure.error)
        return cls(
            id=str(raw.get("id")),
            input=dict(raw),
            state=policy.initial,
            final=False,
            outcome=ERROR_OUTCOME,
            fields=dict.fromkeys(policy.fields),
            journal=list(journal.entries),
            failure=failure,
        )

    @property
    def actions(self) -> list[dict[str, Any]]:
        return [{"type": e.data["type"], "args": e.data["args"]} for e in self.journal if e.kind == "action"]

    @property
    def escalated(self) -> bool:
        return self.outcome in ESCALATING or self.failure is not None

    def plain_fields(self) -> dict[str, Any]:
        """Fields as JSON writes them: numbers as exact strings, never floats."""
        return {k: str(v) if isinstance(v, Decimal) else v for k, v in self.fields.items()}

    def to_line(self) -> dict[str, Any]:
        """One line of results.jsonl."""
        return {
            "id": self.id,
            "outcome": self.outcome,
            "state": self.state,
            "final": self.final,
            "fields": self.plain_fields(),
            "actions": self.actions,
            "error": str(self.failure) if self.failure else None,
        }

    def to_trace(self, calls: list[CallRecord]) -> dict[str, Any]:
        """traces/<id>.json: the case, the journal and every model call (the input for a replay)."""
        return {
            **self.to_line(),
            "input": self.input,
            "failure": self.failure.to_json() if self.failure else None,
            "summary": self.summary,
            "journal": [{"seq": e.seq, "at": e.at, "kind": e.kind, **e.data} for e in self.journal],
            "calls": [c.to_json() for c in calls],
        }
