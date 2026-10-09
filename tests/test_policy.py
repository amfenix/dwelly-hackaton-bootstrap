from __future__ import annotations

import copy
from collections.abc import Callable
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
import yaml

from desk.errors import PolicyError
from desk.policy import UNKNOWN, Effect, Policy, coerce, load_policy, parse_policy, read_yaml
from tests.conftest import MINI

BASE: dict[str, Any] = read_yaml(MINI)


def build(change: Callable[[dict[str, Any]], None]) -> Policy:
    raw = copy.deepcopy(BASE)
    change(raw)
    return parse_policy(yaml.safe_dump(raw, sort_keys=False), "p")


def test_loads_everything(mini: Policy) -> None:
    assert mini.name == "mini" and mini.version == "2"
    assert mini.description == "A small desk for tests."
    assert mini.fields["topic"].values == ("repair", "question", UNKNOWN)
    assert mini.fields["topic"].kind.values[-1] == UNKNOWN
    assert mini.states[0] == "NEW" and mini.initial == "NEW"
    assert mini.final == {"DONE": "act", "ESCALATED": "escalate"}
    pay = mini.signals["pay"]
    assert [a.name for a in pay.args] == ["amount", "memo"]
    assert pay.args[1].optional and pay.args[0].kind.type == "money"
    classify = next(t for t in mini.triggers if t.name == "classify")
    assert [a.kind for a in classify.ask] == ["noul", "choice", "score"]
    assert classify.ask[0].above == 0.7 and classify.ask[0].below == 0.3
    assert classify.ask[1].options == (("repair", "broken"), ("question", "asking"))
    assert classify.ask[2].levels == ("low", "mid", "high")
    assert mini.on_error == Effect((("reason", "FAILED"),), "ESCALATED")
    assert mini.max_steps == 50
    assert mini.agent.model == "claude-haiku-5-5" and mini.agent.instructions == "Be brief."
    assert mini.functions_used() == {"lookup", "settle"}
    assert mini.warnings == ["signal 'ping' has no trigger, so it is always refused"]


def test_yaml_booleans_are_only_true_and_false() -> None:
    assert read_yaml("a: on\nb: yes\nc: true\nd: off\non: 1") == {"a": "on", "b": "yes", "c": True, "d": "off", "on": 1}


def test_load_from_file(tmp_path: Path) -> None:
    path = tmp_path / "p.yaml"
    path.write_text(MINI, encoding="utf-8")
    assert load_policy(path).name == "mini"


def test_functions_must_exist(mini: Policy) -> None:
    mini.check_functions({"lookup": 1, "settle": 2})
    with pytest.raises(PolicyError, match=r"do not exist: settle \(defined: lookup"):
        mini.check_functions({"lookup": 1})
    with pytest.raises(PolicyError, match="defined: none"):
        mini.check_functions({})


def test_briefing_counts_as_a_function() -> None:
    policy = build(lambda r: r.update(briefing="brief"))
    assert policy.briefing == "brief" and "brief" in policy.functions_used()


def test_many_signals_warn_about_strict_tools() -> None:
    def many(raw: dict[str, Any]) -> None:
        for i in range(21):
            raw["signals"][f"s{i}"] = {"description": "x"}

    assert any("without strict schemas" in w for w in build(many).warnings)


def test_coerce_values(mini: Policy) -> None:
    assert mini.coerce("amount", "12.50", "w") == Decimal("12.50")
    assert mini.coerce("note", 5, "w") == "5"
    assert mini.coerce("due", "2026-10-10", "w") == "2026-10-10"
    assert mini.coerce("known", None, "w") is None
    with pytest.raises(PolicyError, match="no field 'ghost'"):
        mini.coerce("ghost", 1, "w")
    with pytest.raises(PolicyError, match="true or false"):
        coerce("bool", (), "yes", "w")
    with pytest.raises(PolicyError, match="not a number"):
        coerce("money", (), "ten", "w")
    with pytest.raises(PolicyError, match="a date like"):
        coerce("date", (), "tomorrow", "w")
    with pytest.raises(PolicyError, match="must be one of"):
        coerce("enum", ("a",), "b", "w")
    import datetime

    assert coerce("date", (), datetime.date(2026, 10, 10), "w") == "2026-10-10"


def _set(path: str, value: Any) -> Callable[[dict[str, Any]], None]:
    def change(raw: dict[str, Any]) -> None:
        keys = path.split(".")
        node: Any = raw
        for key in keys[:-1]:
            node = node[int(key)] if isinstance(node, list) else node[key]
        last = keys[-1]
        if isinstance(node, list):
            node[int(last)] = value
        elif value is _DROP:
            node.pop(last, None)
        else:
            node[last] = value

    return change


_DROP = object()


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (_set("extra", 1), "unknown top-level keys: extra"),
        (_set("policy", " "), "'policy:' .a name. is required"),
        (_set("initial", "NOPE"), "initial: 'NOPE'"),
        (_set("limits", [1]), "limits: must be a mapping"),
        (_set("briefing", 5), "briefing: must be a function name"),
        (_set("fields.state", "text"), "'state' is not a usable name"),
        (_set("fields.Bad", "text"), "'Bad' is not a usable name"),
        (_set("fields.note", {"type": "text", "colour": "red"}), "fields.note: unknown keys colour"),
        (_set("fields.note", 5), "a type like 'text'"),
        (_set("fields.note", "string"), "unknown type 'string'"),
        (_set("fields.note", {"type": "enum"}), "an enum needs values"),
        (_set("fields.note", {"type": "text", "values": ["a"]}), "only an enum has values"),
        (_set("states", []), "states: a list of states is required"),
        (_set("states", ["NEW", "NEW"]), "'NEW' is listed twice"),
        (_set("states", ["NEW", "bad state"]), "'bad state' is not a usable state name"),
        (_set("final", {}), "final: map at least one state"),
        (_set("final", {"NOPE": "act"}), "final: 'NOPE' is not one of the states"),
        (_set("final", {"DONE": " "}), "final.DONE: the outcome must be a word"),
        (_set("signals.Bad", {"description": "x"}), "signals.Bad: a signal name is lower_snake_case"),
        (_set("signals.pay.colour", "red"), "signals.pay: unknown keys colour"),
        (_set("signals.pay.description", ""), "signals.pay: a description is required"),
        (_set("signals.pay.args.Bad", "text"), "'Bad' is not a usable name"),
        (_set("on_error", {"set": {}, "go": "x"}), "on_error: unknown keys go"),
        (_set("on_error", {"set": {"ghost": 1}}), "on_error.set: no field 'ghost'"),
        (_set("on_error", {"to": "NOPE"}), "on_error.to: 'NOPE'"),
        (_set("on_stall", {"set": {"reason": "NOPE"}}), "must be one of"),
        (_set("triggers", []), "triggers: a list of rules is required"),
        (_set("triggers.0.name", ""), "every trigger needs a name"),
        (_set("triggers.1.name", "look"), "trigger 'look': the name is used twice"),
        (_set("triggers.0.colour", "red"), "unknown keys colour"),
        (_set("triggers.3.on", "nothing"), "on: 'nothing' is not a signal"),
        (_set("triggers.0.ask", {"known": {"noul": "q"}}), "either asks the classifier or calls a function"),
        (_set("triggers.0.call", 5), "call: must be a function name"),
        (_set("triggers.0.input", "{case.text}"), "input: only applies to ask:"),
        (_set("triggers.0.retry", -1), "retry: must be a whole number"),
        (_set("triggers.0.retry", True), "retry: must be a whole number"),
        (_set("triggers.2.retry", 1), "retry: only applies to ask: or call:"),
        (_set("triggers.2.on_error", {"to": "DONE"}), "on_error: only applies to ask: or call:"),
        (_set("triggers.2", {"name": "idle", "when": "known"}), "must do something"),
        (_set("triggers.2.when", "state = LIMBO"), "'LIMBO' is not one of"),
        (_set("triggers.1.ask", {"ghost": {"noul": "q"}}), "no field 'ghost' to write the answer to"),
        (_set("triggers.1.ask", {"urgent": {"noul": "q", "choice": "r"}}), "exactly one of noul"),
        (_set("triggers.1.ask", {"urgent": {"noul": "q", "colour": 1}}), "unknown keys colour"),
        (_set("triggers.1.ask", {"note": {"noul": "q"}}), "note must be bool"),
        (_set("triggers.1.ask", {"urgent": {"noul": "q", "above": 0.2, "below": 0.5}}), "below <= above"),
        (_set("triggers.1.ask", {"urgent": {"choice": "q", "options": {"a": "b"}}}), "urgent must be an enum"),
        (_set("triggers.1.ask", {"topic": {"choice": "q"}}), "options: are required"),
        (_set("triggers.1.ask", {"topic": {"choice": "q", "options": {"x": "y"}}}), "option 'x' is not a value"),
        (_set("triggers.1.ask", {"urgent": {"score": "q", "levels": ["a", "b"]}}), "urgent must be an enum"),
        (_set("triggers.1.ask", {"level": {"score": "q", "levels": ["low"]}}), "levels: needs two or more"),
        (_set("triggers.1.ask", {"level": {"score": "q", "levels": ["low", "huge"]}}), "level 'huge' is not a value"),
        (_set("agent", {"temperature": 1}), "agent: unknown keys temperature"),
        (_set("agent", {"max_turns": 0}), "agent: max_turns: must be a whole number, 1 or more"),
        (_set("agent", {"max_tokens": "lots"}), "agent: max_tokens: must be a whole number"),
        (_set("limits", {"steps": True}), "limits: steps: must be a whole number"),
        (_set("limits", {"steps": 2.5}), "limits: steps: must be a whole number"),
        (_set("limits", {"steps": -1}), "limits: steps: must be a whole number"),
        (_set("limits", {"time": 9}), "limits: unknown keys time"),
        (_set("triggers.1.ask", {"urgent": {"noul": "q", "above": "high"}}), "above: must be a number from 0 to 1"),
        (_set("triggers.1.ask", {"urgent": {"noul": "q", "below": True}}), "below: must be a number from 0 to 1"),
    ],
)
def test_mistakes(change: Callable[[dict[str, Any]], None], message: str) -> None:
    with pytest.raises(PolicyError, match=message):
        build(change)


def test_not_yaml_and_not_a_mapping() -> None:
    with pytest.raises(PolicyError, match="not valid YAML"):
        parse_policy("a: [", "p")
    with pytest.raises(PolicyError, match="must be a mapping"):
        parse_policy("- 1", "p")
    with pytest.raises(PolicyError, match=r"signals\.pay\.args\.memo: a type like"):
        build(_set("signals.pay.args.memo", [1]))


def test_effects_without_set_or_to() -> None:
    policy = build(_set("on_stall", {"to": "ESCALATED"}))
    assert policy.on_stall == Effect((), "ESCALATED")
    assert build(_set("on_error", _DROP)).on_error is None


def test_noul_below_defaults_to_the_lower_of_half_and_above() -> None:
    policy = build(_set("triggers.1.ask", {"urgent": {"noul": "q", "above": 0.4}}))
    ask = next(t for t in policy.triggers if t.name == "classify").ask[0]
    assert (ask.above, ask.below) == (0.4, 0.4)


def test_a_bool_stays_and_an_explicit_unknown_is_not_doubled() -> None:
    assert coerce("bool", (), True, "w") is True
    policy = build(_set("fields.topic", "enum [repair, question, unknown]"))
    assert policy.fields["topic"].values == ("repair", "question", "unknown")
