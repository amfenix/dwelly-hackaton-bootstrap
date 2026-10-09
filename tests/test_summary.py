"""The summary is written from the record only, and a summary that fails never fails the case."""

from __future__ import annotations

import json
from typing import Any

from desk.calls import Recorder
from desk.generator import Generator
from desk.policy import Policy
from desk.runner import run_case
from desk.summary import SUMMARY_MAX_TOKENS, SUMMARY_SYSTEM, add_summary, summary_prompt
from tests.conftest import FakeAnthropic, text_reply
from tests.test_runner import full_result, setup


def generator(*replies: Any) -> tuple[Generator, FakeAnthropic]:
    client = FakeAnthropic(*replies)
    return Generator(Recorder(), client=client), client


def test_the_prompt_is_the_record() -> None:
    prompt = summary_prompt(full_result(fields={"note": "x", "empty": None}))
    assert prompt.startswith("<record>\n") and prompt.endswith("</record>")
    record = json.loads(prompt.removeprefix("<record>\n").split("\n\n")[0])
    assert record == {
        "outcome": "act",
        "state": "DONE",
        "final": True,
        "facts": {"note": "x"},
        "actions": [{"type": "book", "args": {"c": 1}}],
        "error": None,
    }
    assert "how the decision was reached" in prompt


def test_add_summary_sets_the_text() -> None:
    gen, client = generator(text_reply("  Booked the plumber.  "))
    result = full_result()
    add_summary(result, gen)
    assert result.summary == "Booked the plumber." and result.summary_error is None
    request = client.requests[0]
    assert request["system"] == SUMMARY_SYSTEM and request["max_tokens"] == SUMMARY_MAX_TOKENS


def test_a_failed_or_empty_summary_is_reported_not_raised() -> None:
    gen, _ = generator(TimeoutError("slow"))
    result = full_result()
    add_summary(result, gen)
    assert result.summary is None and (result.summary_error or "").startswith("timeout: TimeoutError")
    gen, _ = generator(text_reply("   "))
    empty = full_result()
    add_summary(empty, gen)
    assert empty.summary_error == "the generator returned no text"


def test_run_case_summarises_only_when_asked(mini: Policy) -> None:
    gen, client = generator(text_reply("Unknown sender, handed to a person."))
    asked = run_case(setup(mini, make_generator=lambda r: gen, summarise=True), {"id": "1", "known": False}, Recorder())
    assert asked.summary == "Unknown sender, handed to a person." and len(client.requests) == 1
    not_asked = run_case(setup(mini, make_generator=lambda r: gen), {"id": "2", "known": False}, Recorder())
    assert not_asked.summary is None and len(client.requests) == 1
    no_generator = run_case(setup(mini, summarise=True), {"id": "3", "known": False}, Recorder())
    assert no_generator.summary is None
