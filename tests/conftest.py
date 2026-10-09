"""Fakes for the outside world: no test touches a network or needs a key."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from desk.classifier import Answer, Question
from desk.errors import ModelError
from desk.policy import Policy, parse_policy


class FakeResponse:
    def __init__(self, body: dict[str, Any]) -> None:
        self.body = body

    def model_dump(self, mode: str = "json") -> dict[str, Any]:
        return self.body


class _Scripted:
    def __init__(self, replies: list[Any]) -> None:
        self.replies = list(replies)
        self.requests: list[dict[str, Any]] = []

    def next(self, request: dict[str, Any]) -> FakeResponse:
        self.requests.append(request)
        reply = self.replies.pop(0)
        if isinstance(reply, BaseException):
            raise reply
        return FakeResponse(reply)


class FakeAnthropic:
    """`client.messages.create(**request)` answers from a script, in order."""

    def __init__(self, *replies: Any) -> None:
        self.script = _Scripted(list(replies))
        self.messages = self

    def create(self, **request: Any) -> FakeResponse:
        return self.script.next(request)

    @property
    def requests(self) -> list[dict[str, Any]]:
        return self.script.requests


class FakeTypeSafe:
    """`client.system_one(state, questions, model=, timeout=)` answers from a script, in order."""

    def __init__(self, *replies: Any) -> None:
        self.script = _Scripted(list(replies))

    def system_one(self, state: Any, questions: Any, **options: Any) -> FakeResponse:
        return self.script.next({"state": state, "questions": questions, **options})


def text_reply(text: str, *, model: str = "claude-sonnet-5-5", tokens: tuple[int, int] = (10, 5)) -> dict[str, Any]:
    return {
        "content": [{"type": "text", "text": text}],
        "model": model,
        "stop_reason": "end_turn",
        "usage": {"input_tokens": tokens[0], "output_tokens": tokens[1]},
    }


def tool_reply(
    *uses: tuple[str, dict[str, Any]], text: str | None = None, model: str = "claude-sonnet-5-5"
) -> dict[str, Any]:
    content: list[dict[str, Any]] = [{"type": "text", "text": text}] if text else []
    content += [
        {"type": "tool_use", "id": f"tu{i}", "name": name, "input": args} for i, (name, args) in enumerate(uses)
    ]
    return {
        "content": content,
        "model": model,
        "stop_reason": "tool_use",
        "usage": {"input_tokens": 20, "output_tokens": 8},
    }


class FakeClassifier:
    """Answers each question from `answers` (a value, or a list of values used one call at a time)."""

    backend = "fake"
    model = "fake-1"

    def __init__(self, answers: Mapping[str, Any] | None = None, fail: int = 0, error: str = "timeout") -> None:
        self.answers = dict(answers or {})
        self.fail, self.error = fail, error
        self.calls: list[tuple[str, dict[str, Question]]] = []

    def classify(self, text: str, questions: Mapping[str, Question]) -> dict[str, Answer]:
        self.calls.append((text, dict(questions)))
        if self.fail:
            self.fail -= 1
            raise ModelError(self.error, "scripted failure")
        return {name: Answer(self.answers[name], 0.9) for name in questions}


MINI = """
policy: mini
version: 2
description: A small desk for tests.
fields:
  known:    bool
  urgent:   bool
  topic:    enum [repair, question]
  level:    enum [low, mid, high]
  amount:   money
  limit:    money
  due:      date
  note:     text
  count:    number
  reason:   enum [FAILED, STALLED, DENIED, BY_AGENT]
states: [NEW, OPEN, WAIT, DONE, ESCALATED]
initial: NEW
final: { DONE: act, ESCALATED: escalate }
signals:
  pay:
    description: Pay an amount.
    args:
      amount: { type: money, description: how much }
      memo:   { type: text, optional: true }
  answer:
    description: Answer the question.
    args:
      text: text
      kind: enum [short, long]
  ping:
    description: A signal no rule accepts.
triggers:
  - name: look
    when: "state = NEW"
    call: lookup
    to: OPEN
  - name: classify
    when: "state = OPEN and known"
    ask:
      urgent: { noul: "Is it urgent?", above: 0.7, below: 0.3 }
      topic:  { choice: "What is it?", options: { repair: broken, question: asking } }
      level:  { score: "How bad?", levels: [low, mid, high] }
    to: WAIT
  - name: stranger
    when: "state = OPEN and not known"
    set: { reason: DENIED }
    to: ESCALATED
  - name: pay-within
    on: pay
    when: "state = WAIT and args.amount <= limit and topic = repair"
    call: settle
    to: DONE
  - name: pay-any
    on: pay
    when: "state = WAIT and urgent"
    set: { note: urgent pay }
    to: DONE
  - name: reply
    on: answer
    when: "state = WAIT and topic = question"
    set: { note: answered }
    to: DONE
on_error: { set: { reason: FAILED }, to: ESCALATED }
on_stall: { set: { reason: STALLED }, to: ESCALATED }
limits: { steps: 50 }
agent: { model: claude-haiku-5-5, max_turns: 3, max_tokens: 1000, instructions: Be brief. }
"""


@pytest.fixture
def mini() -> Policy:
    return parse_policy(MINI, "mini")
