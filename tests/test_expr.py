from __future__ import annotations

from decimal import Decimal
from typing import Any

import pytest

from desk.errors import PolicyError
from desk.expr import Kind, compile_condition, to_decimal

NAMES = {
    "state": Kind("enum", ("NEW", "DONE")),
    "urgent": Kind("bool"),
    "topic": Kind("enum", ("repair", "question", "unknown")),
    "fallback": Kind("enum", ("repair", "question", "unknown")),
    "amount": Kind("money"),
    "limit": Kind("money"),
    "count": Kind("number"),
    "due": Kind("date"),
    "note": Kind("text"),
    "args.memo": Kind("text"),
}


def env(**values: Any):
    return lambda name: values.get(name.replace("args.", "args_"))


def holds(source: str, **values: Any) -> bool | None:
    return compile_condition(source, NAMES, "test").holds(env(**values))


@pytest.mark.parametrize(
    ("source", "values", "expected"),
    [
        ("state = NEW", {"state": "NEW"}, True),
        ("state != NEW", {"state": "NEW"}, False),
        ("state == DONE", {"state": "DONE"}, True),
        ("urgent", {"urgent": True}, True),
        ("urgent", {}, None),
        ("not urgent", {"urgent": False}, True),
        ("not urgent", {}, None),
        ("urgent = true", {"urgent": True}, True),
        ("urgent = false", {"urgent": True}, False),
        ("amount <= limit", {"amount": Decimal("95"), "limit": Decimal("250")}, True),
        ("amount > limit", {"amount": Decimal("95"), "limit": Decimal("250")}, False),
        ("amount < 100.50", {"amount": Decimal("100.49")}, True),
        ("amount >= -1", {"amount": Decimal("0")}, True),
        ("count > amount", {"count": Decimal("3"), "amount": Decimal("2")}, True),
        ("amount <= limit", {"amount": Decimal("95")}, None),
        ('due < "2026-11-01"', {"due": "2026-10-10"}, True),
        ("topic in [repair, question]", {"topic": "repair"}, True),
        ("topic not in [repair]", {"topic": "question"}, True),
        ("topic in [repair]", {}, None),
        # a list with an unknown member: no match among the known ones is not a "no"
        ("topic in [fallback]", {"topic": "repair"}, None),
        ("topic in [question, fallback]", {"topic": "repair"}, None),
        ("topic in [repair, fallback]", {"topic": "repair"}, True),
        ("topic in [fallback]", {"topic": "repair", "fallback": "question"}, False),
        ("topic not in [fallback]", {"topic": "repair"}, None),
        ("topic not in [question, fallback]", {"topic": "repair", "fallback": "question"}, True),
        ("topic not in [repair, fallback]", {"topic": "repair"}, False),
        ('note = "needs a = sign"', {"note": "needs a = sign"}, True),
        ("note present", {"note": "x"}, True),
        ("note absent", {}, True),
        ("args.memo present", {"args_memo": "m"}, True),
        ("urgent and topic = repair", {"urgent": True, "topic": "repair"}, True),
        ("urgent and topic = repair", {"urgent": False}, False),
        ("urgent and topic = repair", {"urgent": True}, None),
        ("urgent or topic = repair", {"topic": "repair"}, True),
        ("urgent or topic = repair", {"urgent": False, "topic": "question"}, False),
        ("urgent or topic = repair", {"urgent": False}, None),
        ("not (urgent or note present)", {"urgent": False}, True),
        ("", {}, True),
        ("true", {}, True),
        ("false or urgent", {"urgent": True}, True),
    ],
)
def test_conditions(source: str, values: dict[str, Any], expected: bool | None) -> None:
    assert holds(source, **values) is expected


def test_none_condition_always_holds() -> None:
    condition = compile_condition(None, NAMES, "t")
    assert condition.holds(env()) is True
    assert condition.explain(env()) == []


@pytest.mark.parametrize(
    ("source", "message"),
    [
        ("state = ", "cannot read"),
        ("nothing", "unknown name 'nothing'"),
        ("note", "note is text, not bool"),
        ("1 + 2", "unsupported expression"),
        ("args.x.y present", "only args.<name> may use a dot"),
        ("foo.bar present", "only args.<name> may use a dot"),
        ("len(note)", "calls are not part of the language"),
        ("ghost present", "unknown name 'ghost'"),
        ("1 < amount < 3", "one comparison at a time"),
        ("note is note", "unsupported comparison"),
        ('"x" = note', "left side of a comparison"),
        ("topic in repair", "needs a list"),
        ("topic in []", "needs a list"),
        ("topic = broken", "'broken' is not one of"),
        ("topic = nowhere.thing", "only args.<name> may use a dot"),
        ("note = elsewhere", "unknown name 'elsewhere'"),
        ("amount = [1]", "unsupported value"),
        ("amount = 2j", "unsupported value"),
        ("amount = -note", "unsupported value"),
        ('note < "b"', "only number, money and date can be ordered"),
        ("amount = note", "cannot be compared with note"),
        ('amount = "ten"', "compare it with a number"),
        ('urgent = "yes"', "compare it with true or false"),
        ("note = 3", "compare it with a word"),
        ('topic = "broken"', "'broken' is not one of"),
        ('due < "banana"', "'banana' is not a date like 2026-10-10"),
        ('due = "2026-13-01"', "is not a date like"),
    ],
)
def test_mistakes_are_caught_at_load(source: str, message: str) -> None:
    with pytest.raises(PolicyError, match=message.replace("(", r"\(").replace(")", r"\)").replace("[", r"\[")):
        compile_condition(source, NAMES, "test")


def test_explain_lists_each_test_with_its_values() -> None:
    condition = compile_condition(
        "state = NEW and amount <= limit and not urgent and note absent and urgent and (topic in [repair] or false)",
        NAMES,
        "t",
    )
    atoms = condition.explain(env(state="NEW", amount=Decimal("300"), limit=Decimal("250"), urgent=False))
    shown = {a.text: (a.result, dict(a.values)) for a in atoms}
    assert shown["state = NEW"] == (True, {"state": "NEW"})
    assert shown["amount <= limit"] == (False, {"amount": Decimal("300"), "limit": Decimal("250")})
    assert shown["not urgent"] == (True, {"urgent": False})
    assert shown["note absent"] == (True, {"note": None})
    assert shown["urgent"] == (False, {"urgent": False})
    assert shown["topic in [repair]"] == (None, {"topic": None})
    assert shown["false"] == (False, {})
    assert condition.source.startswith("state = NEW")


def test_text_round_trips() -> None:
    condition = compile_condition("not (urgent and (note present or amount > 1)) or topic != repair", NAMES, "t")
    atoms = condition.explain(env())
    assert [a.text for a in atoms] == ["not (urgent and (note present or amount > 1))", "topic != repair"]


def test_to_decimal() -> None:
    assert to_decimal("1,200.50") == Decimal("1200.50")
    assert to_decimal(3) == Decimal(3)
    assert to_decimal(Decimal("2")) == Decimal("2")
    assert to_decimal(0.1) == Decimal("0.1")
    with pytest.raises(ValueError, match="not a number"):
        to_decimal("ten")
    with pytest.raises(ValueError, match="yes/no"):
        to_decimal(True)


@pytest.mark.parametrize("value", ["NaN", "inf", "-Infinity", Decimal("NaN"), Decimal("Infinity"), float("inf")])
def test_to_decimal_refuses_what_is_not_a_finite_number(value: Any) -> None:
    with pytest.raises(ValueError, match="not a finite number"):
        to_decimal(value)
