from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from desk.errors import CaseError, ModelError, PolicyError
from desk.machine import Case, Context, Journal, Services, SignalResult, function, load_functions
from desk.policy import Policy, parse_policy
from tests.conftest import MINI, FakeAnthropic, FakeClassifier, text_reply

ANSWERS = {"urgent": 0.9, "topic": "repair", "level": "mid"}


def functions(**overrides: Any) -> dict[str, Any]:
    def lookup(ctx: Context) -> dict[str, Any]:
        return {"known": ctx.case.get("known", True), "limit": ctx.case.get("limit", "100")}

    def settle(ctx: Context) -> dict[str, Any]:
        ctx.act("pay", amount=str(ctx.args["amount"]), memo=ctx.args.get("memo"))
        ctx.note("paid")
        return {"amount": ctx.args["amount"]}

    found: dict[str, Any] = {"lookup": lookup, "settle": settle}
    found.update(overrides)
    return found


def make(policy: Policy, case: dict[str, Any] | None = None, answers: dict[str, Any] | None = None, **kw: Any) -> Case:
    classifier = kw.pop("classifier", FakeClassifier(answers or ANSWERS))
    return Case(
        policy,
        kw.pop("fns", functions()),
        case or {"id": "c1", "text": "the tap drips"},
        services=Services(classifier, kw.pop("generator", None)),
        **kw,
    )


def kinds(case: Case) -> list[str]:
    return [e.kind for e in case.journal.entries]


def test_start_runs_rules_until_none_fires(mini: Policy) -> None:
    case = make(mini)
    case.start()
    assert case.state == "WAIT" and not case.final and case.outcome is None
    assert case.fields["known"] is True and case.fields["limit"] == Decimal("100")
    assert (case.fields["urgent"], case.fields["topic"], case.fields["level"]) == (True, "repair", "mid")
    assert kinds(case)[:4] == ["start", "trigger", "call", "returned"]
    answers = case.journal.of("answer")
    assert [a.data["field"] for a in answers] == ["urgent", "topic", "level"]
    assert answers[0].data["raw"] == 0.9 and answers[0].data["value"] is True
    why = case.journal.of("trigger")[1].data["why"]
    assert why == [
        {"test": "state = OPEN", "result": True, "values": {"state": "OPEN"}},
        {"test": "known", "result": True, "values": {"known": True}},
    ]


def test_a_stranger_is_escalated_and_later_signals_meet_the_policy(mini: Policy) -> None:
    case = make(mini, {"id": "c", "text": "x", "known": False})
    case.start()
    assert case.final and case.outcome == "escalate" and case.fields["reason"] == "DENIED"
    assert case.journal.of("final")[0].data == {"state": "ESCALATED", "outcome": "escalate"}
    result = case.send("pay", {"amount": "5"})
    assert not result.accepted and "state = WAIT is false (state=ESCALATED)" in (result.reason or "")
    assert result.for_agent()["case_closed"] == "escalate"


MULTI = """
policy: multi
fields: { booked: bool, flagged: bool, note: text }
states: [OPEN, DONE, ESCALATED]
initial: OPEN
final: { DONE: act, ESCALATED: escalate }
signals:
  book: { description: book it }
  escalate: { description: hand over, args: { note: text } }
triggers:
  - { name: book, on: book, when: "state = OPEN", set: { booked: true }, to: DONE }
  - { name: hand-over, on: escalate, when: "state != ESCALATED", call: keep_note, to: ESCALATED }
  - { name: flag-after-handover, when: "state = ESCALATED and booked", set: { flagged: true } }
"""


def test_the_policy_decides_when_a_case_is_closed() -> None:
    policy = parse_policy(MULTI, "p")
    case = Case(policy, {"keep_note": lambda ctx: {"note": ctx.args["note"]}}, {"id": "m"})
    case.start()
    assert case.send("book").final and case.outcome == "act"
    second = case.send("escalate", {"note": "refund demand for a person"})
    assert second.accepted and case.outcome == "escalate"
    assert case.fields == {"booked": True, "flagged": True, "note": "refund demand for a person"}
    assert [e.data["state"] for e in case.journal.of("final")] == ["DONE", "ESCALATED"]
    assert not case.send("escalate", {"note": "again"}).accepted


@pytest.mark.parametrize(("probability", "value"), [(0.71, True), (0.5, None), (0.29, False), (0.3, False)])
def test_noul_thresholds(mini: Policy, probability: float, value: bool | None) -> None:
    case = make(mini, answers={**ANSWERS, "urgent": probability})
    case.start()
    assert case.fields["urgent"] is value


def test_a_choice_outside_the_values_becomes_unknown(mini: Policy) -> None:
    case = make(mini, answers={**ANSWERS, "topic": "weather"})
    case.start()
    assert case.fields["topic"] == "unknown"


def test_signal_accepted_runs_its_rule_then_settles(mini: Policy) -> None:
    case = make(mini)
    case.start()
    result = case.send("pay", {"amount": "40.00", "memo": "tap"})
    assert result.accepted and result.trigger == "pay-within" and result.final and result.outcome == "act"
    assert result.changed == {"amount": Decimal("40.00")}
    assert result.for_agent() == {
        "accepted": True,
        "state": "DONE",
        "now_known": {"amount": "40.00"},
        "case_closed": "act",
    }
    assert case.journal.of("action")[0].data["args"] == {"amount": "40.00", "memo": "tap"}
    assert case.journal.of("note")[0].data == {"text": "paid", "by": "pay-within"}


def test_the_next_rule_for_the_same_signal_is_tried(mini: Policy) -> None:
    case = make(mini)
    case.start()
    result = case.send("pay", {"amount": "400"})
    assert result.accepted and result.trigger == "pay-any" and case.fields["note"] == "urgent pay"


@pytest.mark.parametrize(
    ("signal", "args", "reason"),
    [
        ("fly", {}, "there is no signal 'fly'"),
        ("pay", {"amount": "1", "tip": "2"}, "pay takes no tip"),
        ("pay", {"memo": "x"}, "pay needs amount"),
        ("pay", {"amount": ""}, "pay needs amount"),
        ("pay", {"amount": "lots"}, "pay.amount must be a number"),
        ("pay", {"amount": ["1", ("2",)]}, "pay.amount must be a number"),
        ("answer", {"text": "hi", "kind": "medium"}, "must be one of short, long"),
        ("ping", {}, "no rule accepts this signal"),
    ],
)
def test_refusals_say_why(mini: Policy, signal: str, args: dict[str, Any], reason: str) -> None:
    case = make(mini)
    case.start()
    result = case.send(signal, args)
    assert not result.accepted and reason in (result.reason or "")
    assert case.journal.of("refused")[-1].data["reason"] == result.reason
    assert case.status()["refused"][-1]["signal"] == signal


def test_a_refusal_names_each_rule_and_the_facts_it_saw(mini: Policy) -> None:
    case = make(mini, answers={**ANSWERS, "urgent": 0.1, "topic": "question"})
    case.start()
    result = case.send("pay", {"amount": "500"})
    assert result.reason == (
        "pay-within: args.amount <= limit is false (args.amount=500, limit=100); "
        "topic = repair is false (topic=question); pay-any: urgent is false (urgent=False)"
    )


def test_an_unknown_fact_is_reported_as_not_known(mini: Policy) -> None:
    case = make(mini, answers={**ANSWERS, "urgent": 0.5, "topic": "question"})
    case.start()
    assert "urgent is not known (urgent=not known)" in (case.send("pay", {"amount": "500"}).reason or "")


def test_status_shows_known_facts_only(mini: Policy) -> None:
    case = make(mini)
    case.start()
    status = case.status()
    assert status["state"] == "WAIT" and status["final"] is False
    assert status["facts"]["limit"] == "100" and "note" not in status["facts"]


def test_classifier_failure_goes_to_on_error(mini: Policy) -> None:
    case = make(mini, classifier=FakeClassifier(ANSWERS, fail=1))
    case.start()
    assert case.state == "ESCALATED" and case.fields["reason"] == "FAILED"
    assert kinds(case).count("gave_up") == 1 and case.journal.of("failed")[0].data["trigger"] == "classify"


def test_retry_then_success(mini: Policy) -> None:
    policy = parse_policy(
        MINI.replace("      level:  { score:", "      level:  { score:").replace(
            "    to: WAIT\n", "    retry: 2\n    to: WAIT\n", 1
        ),
        "p",
    )
    case = make(policy, classifier=FakeClassifier(ANSWERS, fail=2))
    case.start()
    assert case.state == "WAIT" and kinds(case).count("retry") == 2
    assert [a.data["attempt"] for a in case.journal.of("answer")] == [3, 3, 3]


def test_a_rule_level_on_error_wins(mini: Policy) -> None:
    policy = parse_policy(
        MINI.replace(
            "    call: lookup\n    to: OPEN\n",
            "    call: lookup\n    to: OPEN\n    on_error: { set: { reason: DENIED }, to: ESCALATED }\n",
        ),
        "p",
    )

    def broken(ctx: Context) -> dict[str, Any]:
        raise KeyError("tenancy")

    case = make(policy, fns=functions(lookup=broken))
    case.start()
    assert case.fields["reason"] == "DENIED"
    assert "lookup raised KeyError" in case.journal.of("failed")[0].data["error"]


def test_no_on_error_anywhere_breaks_the_case(mini: Policy) -> None:
    policy = parse_policy(MINI.replace("on_error: { set: { reason: FAILED }, to: ESCALATED }\n", ""), "p")
    with pytest.raises(CaseError, match="classify failed and no on_error"):
        make(policy, classifier=FakeClassifier(ANSWERS, fail=1)).start()


def test_a_model_error_inside_a_function_is_a_failure(mini: Policy) -> None:
    def calls_a_model(ctx: Context) -> dict[str, Any]:
        raise ModelError("timeout", "the CRM's model")

    case = make(mini, fns=functions(lookup=calls_a_model))
    case.start()
    assert case.fields["reason"] == "FAILED"
    assert "lookup: timeout: the CRM's model" in case.journal.of("failed")[0].data["error"]


@pytest.mark.parametrize(
    ("returned", "message"),
    [
        (["known"], "must return a dict of fields, not list"),
        ({"ghost": 1}, "fields the policy does not declare: ghost"),
    ],
)
def test_a_function_that_breaks_the_contract_breaks_the_case(mini: Policy, returned: Any, message: str) -> None:
    case = make(mini, fns=functions(lookup=lambda ctx: returned))
    with pytest.raises(CaseError, match=message):
        case.start()


def test_a_function_may_return_nothing(mini: Policy) -> None:
    case = make(mini, fns=functions(lookup=lambda ctx: None))
    case.start()
    assert case.state == "OPEN"  # known is unknown: neither the classify nor the stranger rule fires


def test_a_bad_value_from_a_function_names_the_rule(mini: Policy) -> None:
    case = make(mini, fns=functions(lookup=lambda ctx: {"known": "yes"}))
    with pytest.raises(PolicyError, match="rule look: known must be true or false"):
        case.start()


def test_rules_that_loop_are_stopped() -> None:
    policy = parse_policy(
        """
policy: loop
fields: { n: number }
states: [A, B, END]
initial: A
final: { END: act }
signals: {}
triggers:
  - { name: ab, when: "state = A", to: B }
  - { name: ba, when: "state = B", to: A }
limits: { steps: 7 }
""",
        "p",
    )
    with pytest.raises(CaseError, match="more than 7 rule steps"):
        Case(policy, {}, {"id": "x"}).start()


def test_a_rule_without_a_move_fires_once_per_visit() -> None:
    policy = parse_policy(
        """
policy: once
fields: { n: number, seen: bool }
states: [A, END]
initial: A
final: { END: act }
signals: { go: { description: go } }
triggers:
  - { name: mark, when: "state = A", set: { seen: true } }
  - { name: finish, on: go, to: END }
""",
        "p",
    )
    case = Case(policy, {}, {"id": "x"})
    case.start()
    assert [e.data["name"] for e in case.journal.of("trigger")] == ["mark"]
    assert case.send("go").accepted and case.final


def test_stall_applies_on_stall(mini: Policy) -> None:
    case = make(mini)
    case.start()
    case.apply_stall("turn limit")
    assert case.state == "ESCALATED" and case.fields["reason"] == "STALLED"
    assert case.journal.of("stalled")[0].data == {"why": "turn limit", "state": "WAIT"}
    policy = parse_policy(MINI.replace("on_stall: { set: { reason: STALLED }, to: ESCALATED }\n", ""), "p")
    other = make(policy)
    other.start()
    with pytest.raises(CaseError, match="has no on_stall"):
        other.apply_stall("bored")


def test_asking_without_a_classifier_is_a_bug(mini: Policy) -> None:
    case = Case(mini, functions(), {"id": "x", "text": "t"})
    with pytest.raises(CaseError, match="asks the classifier, but this run has none"):
        case.start()


@pytest.mark.parametrize(
    ("case_input", "expected"),
    [
        ({"id": "1", "message": "from message"}, "from message"),
        ({"id": "1", "body": 3, "content": "from content"}, "from content"),
        ({"id": "1", "n": 2}, '{"id": "1", "n": 2}'),
    ],
)
def test_the_classifier_reads_the_case_text(mini: Policy, case_input: dict[str, Any], expected: str) -> None:
    classifier = FakeClassifier(ANSWERS)
    make(mini, case_input, classifier=classifier).start()
    assert classifier.calls[0][0] == expected


def test_input_template_reads_case_and_fields(mini: Policy) -> None:
    policy = parse_policy(
        MINI.replace(
            "    ask:\n      urgent:",
            '    input: "{case.mail.subject} / limit {fields.limit}"\n    ask:\n      urgent:',
        ),
        "p",
    )
    classifier = FakeClassifier(ANSWERS)
    make(policy, {"id": "1", "mail": {"subject": "Leak"}}, classifier=classifier).start()
    assert classifier.calls[0][0] == "Leak / limit 100"
    broken = make(policy, {"id": "1"}, classifier=FakeClassifier(ANSWERS))
    with pytest.raises(CaseError, match="names something the case lacks"):
        broken.start()


def test_context_gives_functions_what_they_need(mini: Policy) -> None:
    seen: dict[str, Any] = {}

    def lookup(ctx: Context) -> dict[str, Any]:
        seen.update(case=dict(ctx.case), state=ctx.state, key=ctx.key, attempt=ctx.attempt, world=ctx.world)
        seen["answer"] = ctx.classify("t", {})
        seen["text"] = ctx.generate("s", "p")
        with pytest.raises(TypeError):
            ctx.fields["known"] = True  # type: ignore[index]
        return {"known": False}

    from desk.calls import Recorder
    from desk.generator import Generator

    generator = Generator(Recorder(), client=FakeAnthropic(text_reply("drafted")))
    case = Case(
        mini,
        functions(lookup=lookup),
        {"id": "c9", "text": "t"},
        world={"w": 1},
        services=Services(FakeClassifier({}), generator),
    )
    case.start()
    assert seen["case"] == {"id": "c9", "text": "t"} and seen["state"] == "NEW" and seen["world"] == {"w": 1}
    assert seen["key"] == "c9:look:1" and seen["attempt"] == 1
    assert seen["answer"] == {} and seen["text"] == "drafted"


def test_context_without_services_fails_clearly(mini: Policy) -> None:
    def lookup(ctx: Context) -> dict[str, Any]:
        for use in (lambda: ctx.classify("t", {}), lambda: ctx.generate("s", "p")):
            with pytest.raises(CaseError, match="this run has no"):
                use()
        return {"known": False}

    Case(mini, functions(lookup=lookup), {"id": "x"}).start()


def test_the_idempotency_key_is_stable_across_retries() -> None:
    policy = parse_policy(
        """
policy: key
fields: { done: bool }
states: [A, END]
initial: A
final: { END: act }
signals: {}
triggers:
  - { name: work, when: "state = A", call: flaky, retry: 2, to: END }
""",
        "p",
    )
    keys: list[str] = []

    def flaky(ctx: Context) -> dict[str, Any]:
        keys.append(ctx.key)
        if ctx.attempt < 3:
            raise TimeoutError("slow")
        return {"done": True}

    case = Case(policy, {"flaky": flaky}, {"id": "k"})
    case.start()
    assert case.final and keys == ["k:work:1"] * 3


def test_case_id_and_journal_defaults(mini: Policy) -> None:
    assert Case(mini, {}, {}).id == "case"
    assert Case(mini, {}, {}, case_id=7).id == "7"
    journal = Journal(clock=iter([1.0, 1.5, 2.25]).__next__)
    journal.add("a", x=1)
    journal.add("b")
    assert [(e.seq, e.at, e.kind) for e in journal.entries] == [(1, 0.5, "a"), (2, 1.25, "b")]


def test_signal_result_without_extras() -> None:
    assert SignalResult(True, "s", "A", False, None).for_agent() == {"accepted": True, "state": "A"}


def test_load_functions_finds_only_marked_ones(tmp_path: Path) -> None:
    source = tmp_path / "fns.py"
    source.write_text(
        "from desk import function\n"
        "import json\n"
        "@function\ndef find(ctx):\n    return {}\n"
        "def helper():\n    return 1\n"
        "def adapt_case(raw):\n    return raw\n",
        encoding="utf-8",
    )
    found, module = load_functions(source)
    assert list(found) == ["find"] and module.adapt_case({"a": 1}) == {"a": 1}
    with pytest.raises(PolicyError, match="cannot load functions"):
        load_functions(tmp_path)


def test_function_marks() -> None:
    @function
    def f(ctx: Context) -> None:
        return None

    assert f.__desk_function__ is True  # type: ignore[attr-defined]


def test_each_firing_of_a_rule_gets_its_own_key() -> None:
    policy = parse_policy(
        """
policy: keys
fields: { done: bool }
states: [A, END]
initial: A
final: { END: act }
signals:
  log: { description: Log a line. }
triggers:
  - { name: write, on: log, when: "state = A", call: record }
""",
        "p",
    )
    keys: list[str] = []

    def record(ctx: Context) -> dict[str, Any]:
        keys.append(ctx.key)
        return {}

    case = Case(policy, {"record": record}, {"id": "k"})
    case.start()
    case.send("log")
    case.send("log")
    assert len(keys) == 2 and keys[0] != keys[1]  # the second log is a new call, not a retry of the first
