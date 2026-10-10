"""A batch always ends with an answer per case; answers and traces come from the journal."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from desk.agent import Agent
from desk.answers import answer_md, reasoning_md
from desk.calls import Recorder
from desk.classifier import ClaudeClassifier
from desk.errors import DeskError
from desk.machine import Context, Entry
from desk.policy import Policy, parse_policy
from desk.result import RunResult
from desk.runner import (
    Setup,
    _deadline,
    briefing,
    check_ids,
    folder_name,
    load_cases,
    load_data,
    recorded_calls,
    run_batch,
    run_case,
)
from tests.conftest import MINI, FakeAnthropic, FakeClassifier, text_reply, tool_reply
from tests.test_machine import ANSWERS, functions


def setup(policy: Policy, *, agent_replies: list[Any] | None = None, **kw: Any) -> Setup:
    return Setup(
        policy=policy,
        functions=kw.pop("fns", functions()),
        world={"w": 1},
        make_classifier=lambda recorder: FakeClassifier(ANSWERS),
        make_agent=None
        if agent_replies is None
        else (lambda p, recorder: Agent(p, recorder, FakeAnthropic(*agent_replies))),
        **kw,
    )


# ------------------------------------------------------------------------------------- loading


def test_load_data_reads_json_yaml_and_jsonl(tmp_path: Path) -> None:
    (tmp_path / "a.json").write_text('{"a": 1}', encoding="utf-8")
    (tmp_path / "b.yaml").write_text("b: on", encoding="utf-8")
    (tmp_path / "c.jsonl").write_text('{"c": 1}\n\n{"c": 2}\n', encoding="utf-8")
    assert load_data(tmp_path / "a.json") == {"a": 1}
    assert load_data(tmp_path / "b.yaml") == {"b": True}
    assert load_data(tmp_path / "c.jsonl") == [{"c": 1}, {"c": 2}]


def test_load_cases_accepts_lists_and_wrappers(tmp_path: Path) -> None:
    (tmp_path / "list.json").write_text(json.dumps([{"text": "a"}, "plain text", {"id": "x9"}]), encoding="utf-8")
    (tmp_path / "wrap.json").write_text(json.dumps({"cases": [{"id": 4}]}), encoding="utf-8")
    (tmp_path / "bad.json").write_text(json.dumps({"items": []}), encoding="utf-8")
    assert load_cases(tmp_path / "list.json") == [
        {"text": "a", "id": "1"},
        {"text": "plain text", "id": "2"},
        {"id": "x9"},
    ]
    assert load_cases(tmp_path / "wrap.json") == [{"id": 4}]
    with pytest.raises(DeskError, match="expected a list of cases"):
        load_cases(tmp_path / "bad.json")


def test_folder_names_are_safe() -> None:
    assert folder_name("PM-01") == "PM-01"
    assert folder_name("a/b c?") == "a_b_c"
    assert folder_name("..") == "case"


# ------------------------------------------------------------------------------------- one case


def test_the_agent_closes_the_case(mini: Policy) -> None:
    result = run_case(
        setup(mini, agent_replies=[tool_reply(("pay", {"amount": "40"}), text="ok"), text_reply("done")]),
        {"id": "1", "text": "x"},
        Recorder(),
    )
    assert result.outcome == "act" and result.state == "DONE" and result.final and result.failure is None
    assert result.actions == [{"type": "pay", "args": {"amount": "40", "memo": None}}]
    assert result.plain_fields()["amount"] == "40" and result.fields["known"] is True
    assert result.agent_text == ["ok", "done"] and result.calls["calls"] == 2


def test_a_failure_after_a_final_state_keeps_the_outcome(mini: Policy) -> None:
    replies = [tool_reply(("pay", {"amount": "40"})), RuntimeError("model down")]
    result = run_case(setup(mini, agent_replies=replies), {"id": "1", "text": "x"}, Recorder())
    # the payment was made: on_error does not relabel it as failed, the failure is recorded beside it
    assert result.outcome == "act" and result.state == "DONE" and result.fields["reason"] is None
    assert result.failure is not None and result.failure.phase == "agent" and result.escalated
    assert "The run failed:** agent: ModelError: transport: RuntimeError: model down" in answer_md(result)


def test_an_agent_that_stops_early_is_handed_on_stall(mini: Policy) -> None:
    result = run_case(setup(mini, agent_replies=[text_reply("no idea")]), {"id": "1", "text": "x"}, Recorder())
    assert result.outcome == "escalate" and result.fields["reason"] == "STALLED"


def test_without_an_agent_an_open_case_stalls(mini: Policy) -> None:
    result = run_case(setup(mini), {"id": "1", "text": "x"}, Recorder())
    assert result.fields["reason"] == "STALLED"
    stalled = [e for e in result.journal if e.kind == "stalled"]
    assert stalled[0].data["why"] == "no agent in this run"


def test_the_agent_is_asked_even_when_the_rules_closed_the_case(mini: Policy) -> None:
    replies = [text_reply("nothing to add")]
    result = run_case(setup(mini, agent_replies=replies), {"id": "1", "known": False}, Recorder())
    assert result.outcome == "escalate" and result.agent_text == ["nothing to add"]
    assert not [e for e in result.journal if e.kind == "stalled"]


def test_rules_alone_can_close_a_case(mini: Policy) -> None:
    result = run_case(setup(mini), {"id": "1", "known": False}, Recorder())
    assert result.outcome == "escalate" and result.fields["reason"] == "DENIED"


def test_a_crash_becomes_an_escalated_answer(mini: Policy) -> None:
    def broken(ctx: Context) -> dict[str, Any]:
        return {"ghost": 1}

    result = run_case(setup(mini, fns=functions(lookup=broken)), {"id": "1"}, Recorder())
    # on_error decided where the case ends: the outcome and the state agree
    assert result.outcome == "escalate" and result.state == "ESCALATED" and result.final
    assert result.fields["reason"] == "FAILED"
    assert result.failure is not None and result.failure.phase == "start" and result.failure.trigger == "look"
    assert "CaseError" in result.failure.error
    kinds = [e.kind for e in result.journal]
    assert kinds.index("error") < kinds.index("final")


def test_a_crash_with_no_on_error_is_reported_as_not_final(mini: Policy) -> None:
    policy = parse_policy(MINI.replace("on_error: { set: { reason: FAILED }, to: ESCALATED }\n", ""), "p")
    result = run_case(setup(policy, fns=functions(lookup=lambda ctx: {"ghost": 1})), {"id": "1"}, Recorder())
    assert result.outcome == "escalate" and result.state == "NEW" and not result.final
    assert "the run stopped in `NEW`, which is not a final state" in answer_md(result)


@pytest.mark.parametrize("where", ["adapt", "factory", "agent"])
def test_a_failure_anywhere_in_a_case_still_answers_it(mini: Policy, where: str) -> None:
    def boom(*_: Any) -> Any:
        raise RuntimeError(f"broken {where}")

    kw: dict[str, Any] = {"adapt": {"adapt_case": boom}, "factory": {"make_generator": boom}}.get(where, {})
    s = setup(mini, **kw)
    if where == "agent":
        s.make_agent = boom
    result = run_case(s, {"id": "1", "text": "x"}, Recorder())
    assert result.failure is not None and f"broken {where}" in result.failure.error
    assert result.outcome == "escalate" and result.id == "1"
    phase = {"adapt": "adapt", "factory": "start", "agent": "agent"}[where]
    assert result.failure.phase == phase


def test_adapt_case_reshapes_the_input(mini: Policy) -> None:
    def adapt(raw: dict[str, Any]) -> dict[str, Any]:
        return {"id": raw["ref"], "text": raw["body"], "known": False}

    result = run_case(setup(mini, adapt_case=adapt), {"ref": "Z-1", "body": "hello"}, Recorder())
    assert result.id == "Z-1" and result.input["text"] == "hello"


def test_a_generator_is_built_when_asked_for(mini: Policy) -> None:
    built: list[Recorder] = []

    def generator(recorder: Recorder) -> Any:
        built.append(recorder)
        return None

    run_case(setup(mini, make_generator=generator), {"id": "1", "known": False}, Recorder())
    assert len(built) == 1


# ------------------------------------------------------------------------------------- briefing


def test_default_briefing_wraps_the_case(mini: Policy) -> None:
    from desk.machine import Case

    case = Case(mini, {}, {"id": "1", "text": "hi"})
    text = briefing(setup(mini), case)
    assert text.startswith("A new case:\n<case>\n") and '"text": "hi"' in text


@pytest.mark.parametrize(("returned", "expected"), [("brief!", "brief!"), ({"briefing": "from dict"}, "from dict")])
def test_a_briefing_function(mini: Policy, returned: Any, expected: str) -> None:
    from desk.machine import Case

    policy = parse_policy(MINI + "briefing: brief\n", "p")
    seen: dict[str, Any] = {}

    def brief(ctx: Any) -> Any:
        seen.update(case=ctx.case, world=ctx.world, state=ctx.state, fields=ctx.fields)
        return returned

    s = setup(policy, fns=functions(brief=brief))
    case = Case(policy, s.functions, {"id": "1"}, world={"w": 2})
    assert briefing(s, case) == expected
    assert seen["world"] == {"w": 2} and seen["state"] == "NEW" and seen["fields"]["known"] is None


def test_a_briefing_must_be_text(mini: Policy) -> None:
    from desk.machine import Case

    policy = parse_policy(MINI + "briefing: brief\n", "p")
    s = setup(policy, fns=functions(brief=lambda ctx: 42))
    with pytest.raises(DeskError, match="must return text"):
        briefing(s, Case(policy, s.functions, {"id": "1"}))


# ------------------------------------------------------------------------------------- batches


def run(tmp_path: Path, s: Setup, cases: list[dict[str, Any]], **kw: Any) -> tuple[dict[str, int], list[str]]:
    lines: list[str] = []
    counts = run_batch(
        s,
        cases,
        tmp_path / "ANSWERS",
        tmp_path / "traces",
        tmp_path / "out" / "results.jsonl",
        progress=lines.append,
        **kw,
    )
    return counts, lines


def test_a_batch_writes_answers_traces_and_results(tmp_path: Path, mini: Policy) -> None:
    cases = [{"id": "1", "known": False}, {"id": "2", "text": "x"}]
    counts, lines = run(tmp_path, setup(mini), cases)
    assert counts == {"cases": 2, "skipped": 0, "worked": 2, "deadline": 0, "errors": 0, "unwritten": 0}
    assert lines == ["1: escalate (ESCALATED)", "2: escalate (ESCALATED)"]
    answer = (tmp_path / "ANSWERS" / "1" / "ANSWER.md").read_text(encoding="utf-8")
    assert "**Outcome: escalate**" in answer and "reason: **DENIED**" in answer
    assert (tmp_path / "ANSWERS" / "2" / "REASONING.md").exists()
    trace = json.loads((tmp_path / "traces" / "1.json").read_text(encoding="utf-8"))
    assert trace["outcome"] == "escalate" and trace["journal"][0]["kind"] == "start"
    results = [json.loads(line) for line in (tmp_path / "out" / "results.jsonl").read_text().splitlines()]
    assert [r["id"] for r in results] == ["1", "2"]


def test_a_rerun_skips_answered_cases_unless_told(tmp_path: Path, mini: Policy) -> None:
    run(tmp_path, setup(mini), [{"id": "1", "known": False}])
    counts, _ = run(tmp_path, setup(mini), [{"id": "1", "known": False}, {"id": "2", "known": False}])
    assert counts["skipped"] == 1 and counts["worked"] == 1
    counts, _ = run(tmp_path, setup(mini), [{"id": "1", "known": False}], resume=False)
    assert counts["worked"] == 1


def test_errors_are_counted_and_reported(tmp_path: Path, mini: Policy) -> None:
    s = setup(mini, fns=functions(lookup=lambda ctx: {"ghost": 1}))
    counts, lines = run(tmp_path, s, [{"id": "1"}])
    assert counts["errors"] == 1 and "CaseError" in lines[0]
    answer = (tmp_path / "ANSWERS" / "1" / "ANSWER.md").read_text(encoding="utf-8")
    assert "**The run failed:**" in answer


def test_the_deadline_still_answers_every_case(tmp_path: Path, mini: Policy) -> None:
    counts, lines = run(tmp_path, setup(mini), [{"id": "1"}, {"id": "2"}], deadline_seconds=-1)
    assert counts["deadline"] == 2 and counts["worked"] == 0
    assert "deadline: the batch deadline passed" in lines[0]
    assert "batch deadline passed" in (tmp_path / "ANSWERS" / "2" / "REASONING.md").read_text(encoding="utf-8")


def test_workers_run_cases_in_parallel(tmp_path: Path, mini: Policy) -> None:
    counts, lines = run(tmp_path, setup(mini), [{"id": str(i), "known": False} for i in range(6)], workers=3)
    assert counts["worked"] == 6 and sorted(lines) == sorted(f"{i}: escalate (ESCALATED)" for i in range(6))


def test_replay_reproduces_a_run_without_the_network(tmp_path: Path, mini: Policy) -> None:
    def live(recorder: Recorder) -> Any:
        return ClaudeClassifier(
            recorder,
            client=FakeAnthropic(
                text_reply(json.dumps({"urgent": 0.95, "topic": "repair", "level": "low"}), model="claude-haiku-5-5")
            ),
        )

    agent_reply = [tool_reply(("pay", {"amount": "40"})), text_reply("done")]
    first = Setup(
        mini, functions(), make_classifier=live, make_agent=lambda p, r: Agent(p, r, FakeAnthropic(*agent_reply))
    )
    run(tmp_path, first, [{"id": "1", "text": "the tap drips"}])
    before = (tmp_path / "ANSWERS" / "1" / "ANSWER.md").read_text(encoding="utf-8")

    offline = Setup(
        mini,
        functions(),
        make_classifier=lambda r: ClaudeClassifier(r, client=FakeAnthropic()),
        make_agent=lambda p, r: Agent(p, r, FakeAnthropic()),
    )
    again = tmp_path / "again"
    counts, _ = run(again, offline, [{"id": "1", "text": "the tap drips"}], replay_from=tmp_path / "traces")
    assert counts["worked"] == 1 and counts["errors"] == 0
    assert (again / "ANSWERS" / "1" / "ANSWER.md").read_text(encoding="utf-8") == before
    trace = json.loads((again / "traces" / "1.json").read_text(encoding="utf-8"))
    assert all(call["replayed"] for call in trace["calls"]) and len(trace["calls"]) == 3
    original = json.loads((tmp_path / "traces" / "1.json").read_text(encoding="utf-8"))
    assert not any(call["replayed"] for call in original["calls"])  # the evidence is untouched

    with pytest.raises(DeskError, match="somewhere other than"):
        run(tmp_path, offline, [{"id": "1"}], replay_from=tmp_path / "traces")


def test_a_replay_that_diverges_is_an_error(tmp_path: Path, mini: Policy) -> None:
    agent_reply = [tool_reply(("pay", {"amount": "40"})), text_reply("done")]
    first = Setup(mini, functions(), make_agent=lambda p, r: Agent(p, r, FakeAnthropic(*agent_reply)))
    run(tmp_path, first, [{"id": "1", "known": False}])
    rules_only = Setup(mini, functions())  # the agent's two recorded calls are never made
    counts, lines = run(tmp_path / "again", rules_only, [{"id": "1", "known": False}], replay_from=tmp_path / "traces")
    assert counts["errors"] == 1 and "2 recorded model call(s) were not made again" in lines[0]
    counts, lines = run(tmp_path / "third", rules_only, [{"id": "2"}], replay_from=tmp_path / "traces")
    assert counts["errors"] == 1 and "replay: DeskError: no trace for case 2" in lines[0]


def test_colliding_case_ids_are_refused_before_anything_runs(tmp_path: Path, mini: Policy) -> None:
    check_ids([{"id": "a"}, {"id": "b"}])
    for pair in (["a/b", "a_b"], ["X", "x"], ["7", "7"]):
        with pytest.raises(DeskError, match="would share the answer folder"):
            check_ids([{"id": pair[0]}, {"id": pair[1]}])
    with pytest.raises(DeskError):
        run(tmp_path, setup(mini), [{"id": "1"}, {"id": "1"}])
    assert not (tmp_path / "ANSWERS" / "1").exists()


def test_an_answer_that_cannot_be_written_is_counted(tmp_path: Path, mini: Policy) -> None:
    (tmp_path / "ANSWERS").mkdir()
    (tmp_path / "ANSWERS" / "1").write_text("a file where the folder goes", encoding="utf-8")
    counts, lines = run(tmp_path, setup(mini), [{"id": "1", "known": False}, {"id": "2", "known": False}])
    assert counts["unwritten"] == 1 and counts["worked"] == 1
    assert lines[0].startswith("1: the answer could not be written")


def test_a_failure_outside_run_case_still_answers(
    tmp_path: Path, mini: Policy, monkeypatch: pytest.MonkeyPatch
) -> None:
    import desk.runner

    def explode(*_: Any) -> Any:
        raise RuntimeError("unexpected")

    monkeypatch.setattr(desk.runner, "run_case", explode)
    counts, lines = run(tmp_path, setup(mini), [{"id": "1"}])
    assert counts["errors"] == 1 and "batch: RuntimeError: unexpected" in lines[0]
    assert (tmp_path / "ANSWERS" / "1" / "ANSWER.md").exists()


def test_replay_needs_a_trace(tmp_path: Path) -> None:
    with pytest.raises(DeskError, match="no trace for case 9"):
        recorded_calls(tmp_path, "9")


def test_deadline_result_shape(mini: Policy) -> None:
    result, calls = _deadline(setup(mini), {"id": 3, "text": "t"})
    assert result.id == "3" and result.state == "NEW" and result.outcome == "escalate" and not result.final
    assert result.failure is not None and result.failure.phase == "deadline" and calls == []


# ------------------------------------------------------------------------------------- answers


def entry(seq: int, kind: str, /, **data: Any) -> Entry:
    return Entry(seq, 0.0, kind, data)


def full_result(**overrides: Any) -> RunResult:
    journal = [
        entry(1, "start", state="NEW", case="7", policy="mini v2"),
        entry(
            2,
            "trigger",
            name="look",
            on=None,
            when="state = NEW",
            why=[
                {"test": "state = NEW", "result": True, "values": {"state": "NEW"}},
                {"test": "known", "result": None, "values": {"known": None}},
                {"test": "a = b", "result": False, "values": {}},
            ],
            note="A note.",
        ),
        entry(3, "trigger", name="go", on="pay", when="", why=[], note=""),
        entry(4, "call", function="lookup", attempt=1, key="7:look:0", args={}),
        entry(5, "returned", function="lookup", values={"known": True}),
        entry(
            6,
            "answer",
            field="urgent",
            kind="noul",
            raw=0.82,
            confidence=0.9,
            value=True,
            backend="jev",
            model="jev-1",
            attempt=1,
        ),
        entry(
            7,
            "answer",
            field="topic",
            kind="choice",
            raw="repair",
            confidence=None,
            value="repair",
            backend="claude",
            model="h",
            attempt=1,
        ),
        entry(8, "signal", signal="pay", args={"amount": "40"}),
        entry(9, "refused", signal="pay", reason="no", state="WAIT"),
        entry(10, "action", type="book", args={"c": 1}, by="go", key="k"),
        entry(11, "note", text="paid", by="go"),
        entry(12, "set", field="note", old=None, new="x", by="go"),
        entry(13, "state", frm="WAIT", to="DONE", by="go"),
        entry(14, "final", state="DONE", outcome="act"),
        entry(15, "retry", trigger="go", attempt=1, error="slow"),
        entry(16, "gave_up", trigger="go", attempt=2, error="slow"),
        entry(17, "failed", trigger="go", error="slow"),
        entry(18, "stalled", why="turn limit", state="WAIT"),
        entry(19, "error", error="boom", phase="agent"),
        entry(20, "mystery", x=1),
    ]
    result: dict[str, Any] = {
        "id": "7",
        "input": {"id": "7", "text": "line one\nline two", "from": "a@b"},
        "outcome": "act",
        "state": "DONE",
        "final": True,
        "fields": {"note": "x", "known": True, "extra": None, "list": [1]},
        "journal": journal,
        "calls": {
            "calls": 2,
            "input_tokens": 10,
            "output_tokens": 5,
            "seconds": 1.5,
            "cost_usd": 0.0012,
            "models": ["h"],
        },
        "agent_text": ["Done.\nBye."],
    }
    result.update(overrides)
    return RunResult(**result)


def test_answer_md_has_every_part() -> None:
    text = answer_md(full_result())
    for part in (
        "# Case 7",
        "**Outcome: act**",
        "> line one",
        '"from": "a@b"',
        '`book` {"c": 1} (rule `go`)',
        "- note: **x**",
        "- list: **[1]**",
        "No.",
        "- gave_up: slow",
        "- failed: slow",
        "- error: boom",
        "- `pay`: no",
        "act (DONE)",
    ):
        assert part in text, part
    assert "extra" not in text


def test_answer_md_for_an_escalated_case_with_nothing_done() -> None:
    text = answer_md(full_result(outcome="escalate", journal=[], fields={}, input={"id": "7"}))
    assert "Yes: the case ended as **escalate**" in text and "- none" in text
    assert "- no facts were established" in text and "- nothing" in text and "escalate (DONE)" in text
    assert "> " not in text and "## Summary" not in text


def test_answer_md_labels_the_summary_as_a_retelling() -> None:
    text = answer_md(full_result(summary="Handed over."))
    assert "## Summary\n\nHanded over.\n\n_Written by a model from the record below" in text
    assert text.index("## Summary") < text.index("## What we received")
    missing = answer_md(full_result(summary_error="timeout: slow"))
    assert "## Summary\n\n_No summary: timeout: slow._" in missing


def test_reasoning_md_reads_the_journal_aloud() -> None:
    text = reasoning_md(full_result())
    for part in (
        "1. Case opened in `NEW` under policy mini v2.",
        "2. Rule `look` fired: state = NEW → true (state=NEW); known → not known (known=not known); "
        "a = b → false. A note.",
        "3. Rule `go` (on `pay`) fired.",
        "4. Called `lookup` (attempt 1, key 7:look:0).",
        '5. `lookup` found {"known": true}.',
        "6. Classifier (jev jev-1) on `urgent`: 0.82, confidence 0.90 → True.",
        "7. Classifier (claude h) on `topic`: repair → repair.",
        '8. The agent asked for `pay` {"amount": "40"}.',
        "9. Refused `pay` in `WAIT`: no.",
        '10. Action in the world: `book` {"c": 1}.',
        "11. Note from `go`: paid",
        "12. `note`: not known → x (by `go`).",
        "13. State `WAIT` → `DONE` (by `go`).",
        "14. The case closed in `DONE`: **act**.",
        "15. `go` attempt 1 failed (slow); will retry.",
        "16. `go` attempt 2 failed (slow); gave up.",
        "17. `go` failed: slow. Its on_error applies.",
        "18. The agent stopped in `WAIT` (turn limit); the policy's on_stall applies.",
        "19. The run broke (agent): boom.",
        '20. mystery: {"x": 1}',
        "2 calls, 10 tokens in, 5 out, 1.5 s, cost $0.0012; models: h.",
        "> Done.\n> Bye.",
    ):
        assert part in text, part


def test_reasoning_md_without_calls_or_price() -> None:
    assert "## Model calls" not in reasoning_md(full_result(calls={}, agent_text=[]))
    unpriced = reasoning_md(full_result(calls={"calls": 1, "cost_usd": None, "models": []}))
    assert "cost not priced; models: none." in unpriced


def test_a_text_only_input_shows_no_json() -> None:
    text = answer_md(full_result(input={"text": ""}))
    assert "> \n" in text and "```json" not in text
