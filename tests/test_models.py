"""The recorder, the classifier on both backends, and the generator: all against scripted fakes."""

from __future__ import annotations

import json
from typing import Any

import pytest

from desk.calls import CallRecord, Recorder, Usage, cost_usd, digest
from desk.classifier import (
    UNKNOWN,
    Choice,
    ClaudeClassifier,
    JevClassifier,
    Noul,
    Score,
    make_classifier,
    with_unknown,
)
from desk.errors import ModelError
from desk.generator import Generator
from tests.conftest import FakeAnthropic, FakeTypeSafe, text_reply

QUESTIONS = {
    "urgent": Noul("Is it urgent?"),
    "topic": Choice("What is it?", {"repair": "broken", "question": ""}),
    "level": Score("How bad?", ("low", "mid", "high")),
}


def ticker(step: float = 0.5):
    now = [0.0]

    def clock() -> float:
        now[0] += step
        return now[0]

    return clock


# ------------------------------------------------------------------------------------ recorder


def test_cost_is_known_for_claude_and_absent_for_others() -> None:
    assert cost_usd("claude-haiku-5-5", 1_000_000, 1_000_000) == pytest.approx(0.6)
    assert cost_usd("jev-latest", 10, 10) is None


def test_recorder_times_counts_and_prices_a_call() -> None:
    recorder = Recorder(clock=ticker())
    body = recorder.call(
        "generate",
        "claude",
        "claude-sonnet-5-5",
        {"q": 1},
        lambda: ({"ok": True}, Usage(1000, 100), "claude-sonnet-5-5"),
    )
    assert body == {"ok": True}
    record = recorder.calls[0]
    assert record.seconds == 0.5 and record.input_tokens == 1000
    assert record.cost_usd == pytest.approx((1000 * 2 + 100 * 10) / 1_000_000)
    assert record.digest == digest("generate", {"q": 1})
    totals = recorder.totals()
    assert totals["calls"] == 1 and totals["models"] == ["claude-sonnet-5-5"]


def test_recorder_keeps_failed_calls_and_reraises() -> None:
    recorder = Recorder()

    def boom() -> tuple[dict[str, Any], Usage, str]:
        raise ModelError("timeout", "slow")

    with pytest.raises(ModelError):
        recorder.call("classify", "jev", "jev-latest", {}, boom)
    assert recorder.calls[0].error == "timeout: slow"
    assert recorder.totals()["cost_usd"] is None


def test_recorder_uses_the_requested_model_when_the_answer_names_none() -> None:
    recorder = Recorder()
    recorder.call("generate", "claude", "claude-opus-5-5", {}, lambda: ({}, Usage(), ""))
    assert recorder.calls[0].model == "claude-opus-5-5"


def test_replay_answers_from_the_record_without_calling() -> None:
    live = Recorder()
    live.call("classify", "claude", "m", {"a": 1}, lambda: ({"first": 1}, Usage(5, 5), "m"))
    live.call("classify", "claude", "m", {"a": 1}, lambda: ({"second": 2}, Usage(5, 5), "m"))
    saved = [CallRecord.from_json(json.loads(json.dumps(c.to_json()))) for c in live.calls]

    replay = Recorder(replay=saved)

    def never() -> tuple[dict[str, Any], Usage, str]:
        raise AssertionError("replay must not call the model")

    assert replay.call("classify", "claude", "m", {"a": 1}, never) == {"first": 1}
    assert replay.call("classify", "claude", "m", {"a": 1}, never) == {"second": 2}
    assert all(c.replayed for c in replay.calls)
    with pytest.raises(ModelError, match="replay: no recorded classify call"):
        replay.call("classify", "claude", "m", {"a": 1}, never)


def test_replay_of_a_failed_call_fails_the_same_way() -> None:
    failed = CallRecord(
        "classify", "jev", "jev", {}, None, 0.1, 0, 0, None, error="timeout: x", digest=digest("classify", {})
    )
    with pytest.raises(ModelError, match="the recorded call failed"):
        Recorder(replay=[failed]).call("classify", "jev", "jev", {}, lambda: ({}, Usage(), ""))


# ------------------------------------------------------------------------------------ classifier


def test_unknown_is_always_an_option() -> None:
    assert with_unknown({"a": "x"}) == {"a": "x", UNKNOWN: "The text does not say, or says too little to tell."}
    assert with_unknown({UNKNOWN: "mine"}) == {UNKNOWN: "mine"}


def jev_body(score: float = 1.6) -> dict[str, Any]:
    return {
        "model": "jev-1.13.0",
        "usage": {"input_tokens": 40, "output_tokens": None},
        "answers": {
            "urgent": {"type": "noul", "noul": 0.82},
            "topic": {"type": "choice", "choice": "repair", "confidence": 0.9, "probabilities": {}},
            "level": {"type": "score", "score": score, "confidence": None, "legend": {}, "probabilities": {}},
        },
    }


def test_jev_asks_every_question_in_one_call() -> None:
    client = FakeTypeSafe(jev_body())
    recorder = Recorder()
    answers = JevClassifier(recorder, "jev-latest", client).classify("the tap drips", QUESTIONS)
    request = client.script.requests[0]
    assert request["state"] == "the tap drips" and request["model"] == "jev-latest"
    assert request["questions"]["topic"]["criteria"][UNKNOWN]
    assert request["questions"]["level"]["criteria"] == ["low", "mid", "high"]
    assert request["questions"]["urgent"] == {"type": "noul", "instructions": "Is it urgent?"}
    assert answers["urgent"].value == pytest.approx(0.82)
    assert (answers["topic"].value, answers["topic"].confidence) == ("repair", 0.9)
    assert answers["level"].value == "high" and answers["level"].confidence is None
    assert recorder.calls[0].model == "jev-1.13.0" and recorder.calls[0].cost_usd is None


@pytest.mark.parametrize(("score", "level"), [(-3.0, "low"), (0.4, "low"), (1.0, "mid"), (9.0, "high")])
def test_jev_score_rounds_to_a_level(score: float, level: str) -> None:
    answers = JevClassifier(Recorder(), client=FakeTypeSafe(jev_body(score))).classify("x", QUESTIONS)
    assert answers["level"].value == level


class APITimeoutError(Exception):
    pass


@pytest.mark.parametrize(("error", "cause"), [(APITimeoutError("t"), "timeout"), (OSError("down"), "transport")])
def test_jev_failures_are_sorted(error: Exception, cause: str) -> None:
    with pytest.raises(ModelError) as caught:
        JevClassifier(Recorder(), client=FakeTypeSafe(error)).classify("x", QUESTIONS)
    assert caught.value.cause == cause


@pytest.mark.parametrize(
    ("body", "message"),
    [({"model": "j"}, "has no answers"), ({"answers": {"urgent": 1}}, "no answer for 'urgent'")],
)
def test_jev_malformed(body: dict[str, Any], message: str) -> None:
    with pytest.raises(ModelError, match=message):
        JevClassifier(Recorder(), client=FakeTypeSafe(body)).classify("x", {"urgent": Noul("q")})


@pytest.mark.parametrize(
    ("name", "answer", "message"),
    [
        ("urgent", {"type": "noul"}, "urgent.noul: None is not a number"),
        ("urgent", {"noul": "likely"}, "urgent.noul: 'likely' is not a number"),
        ("urgent", {"noul": True}, "urgent.noul: True is not a number"),
        ("urgent", {"noul": float("nan")}, "urgent.noul: nan is not a finite number"),
        ("topic", {"choice": 3}, "topic.choice: 3 is not an option"),
        ("topic", {"choice": "repair", "confidence": "high"}, "topic.confidence: 'high' is not a number"),
        ("level", {"score": None}, "level.score: None is not a number"),
        ("level", {"score": float("inf")}, "level.score: inf is not a finite number"),
    ],
)
def test_jev_malformed_answers_are_model_errors(name: str, answer: dict[str, Any], message: str) -> None:
    """A malformed answer is retried and then handed to on_error, like any other model failure."""
    body = jev_body()
    body["answers"][name] = answer
    with pytest.raises(ModelError, match=message) as caught:
        JevClassifier(Recorder(), client=FakeTypeSafe(body)).classify("x", QUESTIONS)
    assert caught.value.cause == "malformed"


def test_jev_probabilities_are_held_to_zero_and_one() -> None:
    body = jev_body()
    body["answers"]["urgent"] = {"noul": 1.4}
    body["answers"]["topic"]["confidence"] = -0.2
    answers = JevClassifier(Recorder(), client=FakeTypeSafe(body)).classify("x", QUESTIONS)
    assert answers["urgent"].value == 1.0 and answers["topic"].confidence == 0.0


def test_jev_response_without_usage_or_model() -> None:
    body = {"answers": {"urgent": {"type": "noul", "noul": 0.1}}}
    recorder = Recorder()
    JevClassifier(recorder, "jev-x", FakeTypeSafe(body)).classify("x", {"urgent": Noul("q")})
    assert recorder.calls[0].model == "jev-x" and recorder.calls[0].input_tokens == 0


def claude_answer(values: dict[str, Any]) -> dict[str, Any]:
    return text_reply(json.dumps(values), model="claude-haiku-5-5")


def test_claude_classifier_sends_a_schema_and_reads_the_json() -> None:
    client = FakeAnthropic(claude_answer({"urgent": 1.4, "topic": "REPAIR", "level": "mid"}))
    recorder = Recorder()
    answers = ClaudeClassifier(recorder, client=client).classify("the tap drips", QUESTIONS)
    request = client.requests[0]
    schema = request["output_config"]["format"]["schema"]
    assert schema["required"] == ["urgent", "topic", "level"] and schema["additionalProperties"] is False
    assert schema["properties"]["topic"]["enum"] == ["repair", "question", UNKNOWN]
    assert schema["properties"]["urgent"]["type"] == "number"
    prompt = request["messages"][0]["content"]
    assert "<state>\nthe tap drips\n</state>" in prompt
    assert "    repair: broken" in prompt and "    question\n" in prompt
    assert "lowest first: low, mid, high" in prompt
    assert answers["urgent"].value == 1.0  # clamped
    assert answers["topic"].value == "repair"  # case-insensitive
    assert answers["level"].value == "mid"
    assert recorder.calls[0].cost_usd == pytest.approx((10 * 0.10 + 5 * 0.50) / 1_000_000)


@pytest.mark.parametrize(
    ("reply", "message"),
    [
        (text_reply("not json"), "not JSON"),
        (text_reply("[1]"), "not an object"),
        (text_reply("{}"), "no answer for 'urgent'"),
        (text_reply('{"urgent": "very"}'), "is not a number"),
        (text_reply('{"urgent": 0.5, "topic": "other"}'), "is not one of"),
        ({"content": [], "stop_reason": "refusal"}, "no text in the response .stop: refusal"),
    ],
)
def test_claude_classifier_malformed(reply: dict[str, Any], message: str) -> None:
    with pytest.raises(ModelError, match=message):
        ClaudeClassifier(Recorder(), client=FakeAnthropic(reply)).classify(
            "x", {"urgent": Noul("q"), "topic": Choice("t", {"repair": ""})}
        )


@pytest.mark.parametrize(("error", "cause"), [(APITimeoutError("t"), "timeout"), (ConnectionError("x"), "transport")])
def test_claude_failures_are_sorted(error: Exception, cause: str) -> None:
    with pytest.raises(ModelError) as caught:
        ClaudeClassifier(Recorder(), client=FakeAnthropic(error)).classify("x", {"urgent": Noul("q")})
    assert caught.value.cause == cause


def test_claude_reply_without_usage_or_model() -> None:
    recorder = Recorder()
    reply = {"content": [{"type": "text", "text": '{"urgent": 0.2}'}]}
    ClaudeClassifier(recorder, "claude-haiku-5-5", FakeAnthropic(reply)).classify("x", {"urgent": Noul("q")})
    assert recorder.calls[0].model == "claude-haiku-5-5" and recorder.calls[0].output_tokens == 0


def test_make_classifier() -> None:
    recorder = Recorder()
    assert isinstance(make_classifier("jev", recorder), JevClassifier)
    assert make_classifier("jev", recorder, "jev-1.13.0").model == "jev-1.13.0"
    assert make_classifier("claude", recorder).model == "claude-haiku-5-5"
    with pytest.raises(ValueError, match="unknown classifier backend"):
        make_classifier("gpt", recorder)


# ------------------------------------------------------------------------------------ generator


def test_generator_sends_system_and_prompt_apart_and_returns_text() -> None:
    client = FakeAnthropic(text_reply("  Hello Sam.  "), text_reply("x"))
    generator = Generator(Recorder(), client=client)
    assert generator.generate("You write replies.", "Say hello to Sam.") == "Hello Sam."
    request = client.requests[0]
    assert request["system"] == "You write replies." and request["messages"][0]["content"] == "Say hello to Sam."
    assert request["model"] == "claude-sonnet-5-5" and request["max_tokens"] == 2000
    generator.generate("s", "p", model="claude-haiku-5-5", max_tokens=50)
    assert client.requests[1]["model"] == "claude-haiku-5-5" and client.requests[1]["max_tokens"] == 50


def test_replay_finds_the_matching_record_out_of_order() -> None:
    first = CallRecord(
        "generate", "claude", "m", {"n": 1}, {"one": 1}, 0.1, 1, 1, None, digest=digest("generate", {"n": 1})
    )
    second = CallRecord(
        "generate", "claude", "m", {"n": 2}, {"two": 2}, 0.1, 1, 1, None, digest=digest("generate", {"n": 2})
    )
    replay = Recorder(replay=[first, second])
    assert replay.call("generate", "claude", "m", {"n": 2}, lambda: ({}, Usage(), "")) == {"two": 2}
