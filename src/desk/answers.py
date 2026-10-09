"""ANSWER.md and REASONING.md for one case, written from what actually happened.

Both are built from the record, not composed by a model: the answer is the case's final state,
facts and actions, and the reasoning is the journal read aloud, every rule with the facts it saw.
The one exception is the summary at the top of ANSWER.md, which is labelled as a model's retelling
(see `desk.summary`).
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from desk.machine import Entry

if TYPE_CHECKING:
    from desk.result import RunResult


def _show(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, default=str)
    return "not known" if value is None else str(value)


def _received(case_input: dict[str, Any]) -> str:
    key = next((k for k in ("text", "message", "body", "content") if isinstance(case_input.get(k), str)), None)
    rest = {k: v for k, v in case_input.items() if k != key}
    lines = []
    if key is not None:
        lines += ["> " + line for line in str(case_input[key]).splitlines() or [""]]
        lines.append("")
    if rest:
        lines += ["```json", json.dumps(rest, ensure_ascii=False, indent=2, default=str), "```"]
    return "\n".join(lines)


def answer_md(result: RunResult) -> str:
    entries = result.journal
    actions = [e for e in entries if e.kind == "action"]
    failures = [e for e in entries if e.kind in ("failed", "gave_up", "error")]
    refused = [e for e in entries if e.kind == "refused"]
    facts = {k: v for k, v in result.plain_fields().items() if v is not None}
    where = (
        f"final state `{result.state}`"
        if result.final
        else f"the run stopped in `{result.state}`, which is not a final state"
    )
    lines = [f"# Case {result.id}", "", f"**Outcome: {result.outcome}**  ({where})", ""]
    if result.failure:
        lines += [f"**The run failed:** {result.failure}", ""]
    lines += _summary(result)
    lines += ["## What we received", "", _received(result.input), "", "## Actions taken", ""]
    lines += [f"- `{a.data['type']}` {_show(a.data['args'])} (rule `{a.data['by']}`)" for a in actions] or ["- none"]
    lines += ["", "## Decision", ""]
    lines += [f"- {name}: **{_show(value)}**" for name, value in facts.items()] or ["- no facts were established"]
    lines += ["", "## Escalated", ""]
    if result.escalated:
        lines.append(f"Yes: the case ended as **{result.outcome}** in `{result.state}`.")
        lines.append(
            "A person picking this up has the facts above, the actions already taken, and the "
            "step-by-step record in REASONING.md."
        )
    else:
        lines.append("No.")
    lines += ["", "## What failed", ""]
    lines += [f"- {e.kind}: {_show(e.data.get('error'))}" for e in failures] or ["- nothing"]
    if refused:
        lines += ["", "Requests the policy refused:", ""]
        lines += [f"- `{e.data['signal']}`: {e.data['reason']}" for e in refused]
    lines += ["", "## Result", "", f"{result.outcome} ({result.state})", ""]
    return "\n".join(lines)


def _summary(result: RunResult) -> list[str]:
    if result.summary:
        note = "_Written by a model from the record below; the sections below are the record._"
        return ["## Summary", "", result.summary, "", note, ""]
    if result.summary_error:
        return ["## Summary", "", f"_No summary: {result.summary_error}._", ""]
    return []


def reasoning_md(result: RunResult) -> str:
    lines = [f"# Case {result.id}: how the decision was reached", ""]
    lines.append(
        "Every line below is a record of the run, in order: the agent's requests, the rules "
        "that fired with the facts they saw, what checks returned, and each change of state."
    )
    lines.append("")
    for entry in result.journal:
        lines.append(f"{entry.seq}. {_describe(entry)}")
    calls = result.calls
    if calls:
        lines += ["", "## Model calls", ""]
        lines.append(
            f"{calls.get('calls', 0)} calls, {calls.get('input_tokens', 0)} tokens in, "
            f"{calls.get('output_tokens', 0)} out, {calls.get('seconds', 0)} s, "
            f"cost {_money(calls.get('cost_usd'))}; "
            f"models: {', '.join(calls.get('models') or []) or 'none'}."
        )
    if result.agent_text:
        lines += ["", "## What the agent said", ""]
        lines += [f"> {line}" for text in result.agent_text for line in str(text).splitlines()]
    lines.append("")
    return "\n".join(lines)


def _describe(entry: Entry) -> str:
    d = entry.data
    kind = entry.kind
    if kind == "start":
        return f"Case opened in `{d['state']}` under policy {d['policy']}."
    if kind == "signal":
        return f"The agent asked for `{d['signal']}` {_show(d['args'])}."
    if kind == "refused":
        return f"Refused `{d['signal']}` in `{d['state']}`: {d['reason']}."
    if kind == "trigger":
        why = "; ".join(f"{a['test']} → {_verdict(a['result'])}{_values(a['values'])}" for a in d["why"])
        head = f"Rule `{d['name']}`" + (f" (on `{d['on']}`)" if d["on"] else "")
        return head + (f" fired: {why}." if why else " fired.") + (f" {d['note']}" if d.get("note") else "")
    if kind == "answer":
        conf = f", confidence {d['confidence']:.2f}" if d.get("confidence") is not None else ""
        who = f"{d['backend']} {d['model']}"
        return f"Classifier ({who}) on `{d['field']}`: {_show(d['raw'])}{conf} → {_show(d['value'])}."
    if kind == "call":
        return f"Called `{d['function']}` (attempt {d['attempt']}, key {d['key']})."
    if kind == "returned":
        return f"`{d['function']}` found {_show(d['values'])}."
    if kind == "action":
        return f"Action in the world: `{d['type']}` {_show(d['args'])}."
    if kind == "note":
        return f"Note from `{d['by']}`: {d['text']}"
    if kind == "set":
        return f"`{d['field']}`: {_show(d['old'])} → {_show(d['new'])} (by `{d['by']}`)."
    if kind == "state":
        return f"State `{d['frm']}` → `{d['to']}` (by `{d['by']}`)."
    if kind == "final":
        return f"The case closed in `{d['state']}`: **{d['outcome']}**."
    if kind in ("retry", "gave_up"):
        verb = "will retry" if kind == "retry" else "gave up"
        return f"`{d['trigger']}` attempt {d['attempt']} failed ({d['error']}); {verb}."
    if kind == "failed":
        return f"`{d['trigger']}` failed: {d['error']}. Its on_error applies."
    if kind == "stalled":
        return f"The agent stopped in `{d['state']}` ({d['why']}); the policy's on_stall applies."
    if kind == "error":
        phase = f" ({d['phase']})" if d.get("phase") else ""
        return f"The run broke{phase}: {d['error']}."
    return f"{kind}: {_show(d)}"


def _money(cost: Any) -> str:
    return "not priced" if cost is None else f"${cost:.4f}"


def _verdict(result: bool | None) -> str:
    return "not known" if result is None else ("true" if result else "false")


def _values(values: dict[str, Any]) -> str:
    if not values:
        return ""
    return " (" + ", ".join(f"{k}={_show(v)}" for k, v in values.items()) + ")"
