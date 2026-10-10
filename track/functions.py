"""The example track's functions: lookups and actions against the world (world.json).

A function gets a `ctx` and returns the fields it found, as a dict. It never decides: the policy
reads what it returns. It records anything it does in the world with `ctx.act(...)`.

    ctx.case      the case as it came in            ctx.world   the records (world.json)
    ctx.fields    the facts known so far (read only) ctx.args    the agent's arguments, for a signal's rule
    ctx.key       the same on every retry: dedupe writes on it
    ctx.act(type, **args)       record an action taken in the world
    ctx.classify(text, {...})   ctx.generate(system, prompt)   when a check needs a model
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

from desk import Context, function

_BOOKED: set[str] = set()
"""Bookings already made, by idempotency key: a retry after a lost answer does not book twice.

It stands in for the booking system's own idempotency: a real integration sends `ctx.key` with
the request and lets that system dedupe, so it holds across restarts and across workers.
"""
_BOOKING = threading.Lock()

_ANSWERS: dict[str, Callable[[dict[str, Any], dict[str, Any]], str]] = {
    "rent_due_date": lambda t, w: f"Your rent is due on day {t['rent_due_day']} of each month.",
    "rent_amount": lambda t, w: f"Your rent is £{t['rent_gbp']} a month.",
    "landlord": lambda t, w: f"Your landlord is {w['landlords'][t['landlord']]['name']}.",
    "property": lambda t, w: f"Our records have your home as {t['property']}.",
}
"""What a reply can say, each read from the tenancy record: the agent picks the topic, not the words."""


def _tenancy(ctx: Context) -> dict[str, Any] | None:
    sender = str(ctx.case.get("from", "")).strip().lower()
    for tenancy_id, tenancy in ctx.world["tenancies"].items():
        contacts = [str(c).lower() for c in tenancy["contacts"]]
        if sender and sender in contacts:
            return {"id": tenancy_id, **tenancy}
    return None


@function
def find_sender(ctx: Context) -> dict[str, Any]:
    tenancy = _tenancy(ctx)
    if tenancy is None:
        ctx.note(f"no tenancy lists {ctx.case.get('from')!r} as a contact")
        return {}
    landlord = ctx.world["landlords"][tenancy["landlord"]]
    return {"tenant": tenancy["id"], "landlord": tenancy["landlord"], "limit": landlord["spend_limit_gbp"]}


@function
def get_quote(ctx: Context) -> dict[str, Any]:
    trade = ctx.fields["trade"]  # the classifier's, which the rule checked the agent's against
    approved = ctx.world["landlords"][ctx.fields["landlord"]]["approved"]
    for contractor_id in approved:
        contractor = ctx.world["contractors"][contractor_id]
        if trade in contractor["trades"]:
            return {"contractor": contractor_id, "quote": contractor["quotes_gbp"][trade]}
    ctx.note(f"no approved contractor of {ctx.fields['landlord']} does {trade}")
    return {}


@function
def book_contractor(ctx: Context) -> dict[str, Any]:
    failures = ctx.case.get("simulate", {}).get("booking_timeouts", 0)
    with _BOOKING:
        if ctx.key in _BOOKED:
            return {"booked": True}
        if ctx.attempt <= failures:
            raise TimeoutError("the contractor's booking system did not answer")
        ctx.act(
            "book_contractor",
            contractor=ctx.fields["contractor"],
            tenancy=ctx.fields["tenant"],
            quote_gbp=str(ctx.fields["quote"]),
        )
        ctx.act(
            "send_message",
            to=ctx.fields["tenant"],
            text=f"We've booked {ctx.world['contractors'][ctx.fields['contractor']]['name']}; "
            "they will contact you to confirm a time.",
        )
        _BOOKED.add(ctx.key)  # only once the booking is made: a failure before this is retried
    return {"booked": True}


@function
def send_reply(ctx: Context) -> dict[str, Any]:
    tenancy = ctx.world["tenancies"][ctx.fields["tenant"]]
    text = _ANSWERS[ctx.args["about"]](tenancy, ctx.world)
    ctx.act("send_message", to=ctx.fields["tenant"], text=text)
    return {}


@function
def brief(ctx: Any) -> str:
    """What the agent reads first: the message, and the records that matter to it."""
    lines = [
        "A message to the maintenance inbox:",
        "<case>",
        str(ctx.case.get("text", "")),
        "</case>",
        f"From: {ctx.case.get('from')} via {ctx.case.get('channel')}",
    ]
    facts = {k: str(v) for k, v in ctx.fields.items() if v is not None}
    lines.append(f"What the desk has established: {facts or 'nothing yet'}")
    lines.append(f"Case state: {ctx.state}")
    return "\n".join(lines)
