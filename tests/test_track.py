"""The example track's rules and functions: the agent's arguments are checked against the records."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from desk.machine import Case, Context, Services, load_functions
from desk.policy import load_policy
from desk.runner import load_data
from tests.conftest import FakeClassifier

TRACK = Path(__file__).resolve().parents[1] / "track"
POLICY = load_policy(TRACK / "policy.yaml")
FUNCTIONS, MODULE = load_functions(TRACK / "functions.py")
WORLD = load_data(TRACK / "world.json")


def triaged(category: str, trade: str = "plumbing", **case: Any) -> Case:
    answers = {"is_emergency": 0.05, "category": category, "trade": trade}
    machine = Case(
        POLICY,
        FUNCTIONS,
        {"id": "t", "from": "sam.reed@example.com", "text": "x", **case},
        WORLD,
        Services(FakeClassifier(answers)),
    )
    machine.start()
    assert machine.state == "TRIAGED"
    return machine


def test_a_booking_for_another_trade_than_the_classifier_found_is_refused() -> None:
    case = triaged("repair", trade="plumbing")
    refused = case.send("book_repair", {"trade": "electrical", "access": "weekdays"})
    assert not refused.accepted and refused.reason is not None
    assert "args.trade = trade is false (args.trade=electrical, trade=plumbing)" in refused.reason
    accepted = case.send("book_repair", {"trade": "plumbing", "access": "weekdays"})
    assert accepted.accepted and case.state == "DONE"


def test_a_reply_is_written_from_the_record_not_by_the_agent() -> None:
    expected = {
        "rent_due_date": "Your rent is due on day 1 of each month.",
        "rent_amount": "Your rent is £1200.00 a month.",
        "landlord": "Your landlord is Morgan Holt.",
        "property": "Our records have your home as Flat 4, 9 Ashgrove Road.",
    }
    for about, text in expected.items():
        case = triaged("question")
        assert case.send("reply", {"about": about}).accepted
        assert case.journal.of("action")[-1].data["args"] == {"to": "T-201", "text": text}
    assert not triaged("question").send("reply", {"about": "deposit"}).accepted  # no such topic to pick


def test_concurrent_retries_book_once() -> None:
    case = triaged("repair")
    case.fields.update(contractor="C-PIPE", quote=95, tenant="T-201")
    trigger = next(t for t in POLICY.triggers if t.name == "book")
    contexts = [Context(case, trigger, {}, attempt=1, key="race:book:1") for _ in range(8)]
    threads = [threading.Thread(target=FUNCTIONS["book_contractor"], args=(ctx,)) for ctx in contexts]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    booked = [e for e in case.journal.of("action") if e.data["type"] == "book_contractor"]
    assert len(booked) == 1
