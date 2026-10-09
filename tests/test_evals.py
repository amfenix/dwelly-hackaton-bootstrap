"""Evals run on fakes here; on real models they run with `desk eval`."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from desk.agent import Agent
from desk.calls import Recorder
from desk.classifier import ClaudeClassifier, JevClassifier
from desk.errors import DeskError
from desk.evals import Factories, Trial, default_factories, load_spec, run_eval, save, summarise, table
from desk.generator import Generator
from tests.conftest import MINI, FakeAnthropic, FakeClassifier, text_reply, tool_reply
from tests.test_machine import functions

TRACK = Path(__file__).resolve().parents[1] / "track"


def write(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


CLASSIFY = """
eval: triage
kind: classify
models: [fake-a]
runs: 2
questions:
  urgent: { noul: "Urgent?", above: 0.7 }
  topic:  { choice: "What?", options: { repair: broken, question: ~ } }
  level:  { score: "How bad?", levels: [low, high] }
cases:
  - name: tap
    text: the tap drips
    expect: { urgent: false, topic: repair }
  - text: gas!
    expect: { urgent: true }
"""


def fakes(
    answers_by_model: dict[str, dict[str, Any]] | None = None,
    *,
    generate: str = "Hello Sam",
    agent: list[Any] | None = None,
) -> Factories:
    def classifier(model: str, recorder: Recorder) -> Any:
        answers = (answers_by_model or {}).get(model)
        if answers is None:
            raise RuntimeError(f"no such model {model}")
        return FakeClassifier(answers)

    return Factories(
        classifier=classifier,
        generator=lambda model, recorder: Generator(recorder, model, FakeAnthropic(text_reply(generate, model=model))),
        agent=lambda policy, model, recorder: Agent(policy, recorder, FakeAnthropic(*(agent or [])), model=model),
    )


def test_classify_eval_scores_each_model(tmp_path: Path) -> None:
    spec = load_spec(write(tmp_path, "e.yaml", CLASSIFY))
    assert spec.name == "triage" and spec.runs == 2 and [c["name"] for c in spec.cases] == ["tap", "2"]
    assert spec.questions["level"].levels == ("low", "high") and spec.questions["topic"].options[1] == ("question", "")
    trials = run_eval(
        spec,
        ["fake-a", "fake-b", "missing"],
        2,
        fakes(
            {
                "fake-a": {"urgent": 0.1, "topic": "repair", "level": "low"},
                "fake-b": {"urgent": 0.5, "topic": "question", "level": "high"},
            }
        ),
        workers=2,
    )
    rows = {r["model"]: r for r in summarise(trials)}
    assert rows["fake-a"]["passed"] == 2 and rows["fake-a"]["accuracy"] == 0.5
    assert rows["fake-a"]["failing"] == ["2"] and rows["fake-a"]["stability"] == 1.0
    assert rows["fake-b"]["passed"] == 0
    assert rows["missing"]["errors"] == 4 and rows["missing"]["cost_usd"] is None
    errored = next(t for t in trials if t.model == "missing")
    assert errored.error == "RuntimeError: no such model missing"


def test_classify_eval_from_a_policy_trigger(tmp_path: Path) -> None:
    (tmp_path / "track").mkdir()
    for name in ("policy.yaml",):
        (tmp_path / "track" / name).write_text((TRACK / name).read_text(encoding="utf-8"), encoding="utf-8")
    spec = load_spec(
        write(
            tmp_path,
            "e.yaml",
            "kind: classify\npolicy: track/policy.yaml\ntrigger: triage\n"
            "cases: [{text: x, expect: {category: repair}}]\n",
        )
    )
    assert list(spec.questions) == ["is_emergency", "category", "trade"] and spec.name == "e"


def test_generate_eval(tmp_path: Path) -> None:
    spec = load_spec(
        write(
            tmp_path,
            "g.yaml",
            """
kind: generate
system: "You reply to {who}."
prompt: "Tell {who} the plumber is booked."
cases:
  - { vars: { who: Sam }, expect_contains: [sam], expect_not_contains: [refund] }
  - { vars: { who: Jo }, expect_contains: [jo] }
""",
        )
    )
    trials = run_eval(spec, ["claude-haiku-5-5"], 1, fakes(generate="Hello Sam, booked."))
    assert [t.passed for t in trials] == [True, False]
    assert trials[0].answer == "Hello Sam, booked." and trials[0].cost_usd is not None


def test_agent_eval_checks_the_first_tool(tmp_path: Path) -> None:
    write(tmp_path, "p.yaml", MINI)
    fns = write(
        tmp_path,
        "f.py",
        "from desk import function\n@function\ndef lookup(ctx):\n    return {}\n"
        "@function\ndef settle(ctx):\n    return {}\n",
    )
    write(tmp_path, "w.json", '{"x": 1}')
    spec = load_spec(
        write(
            tmp_path,
            "a.yaml",
            f"""
kind: agent
policy: p.yaml
functions: {fns.name}
world: w.json
cases:
  - name: pays
    case: {{ id: "1", text: pay 40 }}
    given: {{ state: WAIT, fields: {{ topic: repair, limit: "100" }} }}
    expect_tool: pay
    expect_args: {{ amount: "40" }}
  - name: nothing
    expect_tool: pay
  - name: briefed
    briefing: Look at this.
    expect_tool: answer
""",
        )
    )
    assert spec.world == {"x": 1} and set(spec.functions) == {"lookup", "settle"}
    replies = [tool_reply(("case_status", {}), ("pay", {"amount": "40"}))]
    trials = run_eval(spec, ["claude-sonnet-5-5"], 1, fakes(agent=replies), workers=1)
    assert trials[0].passed and trials[0].answer == {"tool": "pay", "args": {"amount": "40"}}
    assert trials[1].passed  # the tool is what is judged, not whether the policy accepts it
    assert not trials[2].passed and trials[2].error is None


def test_agent_eval_reads_the_policy_briefing(tmp_path: Path) -> None:
    write(tmp_path, "p.yaml", MINI + "briefing: brief\n")
    spec = load_spec(write(tmp_path, "a.yaml", "kind: agent\npolicy: p.yaml\ncases: [{expect_tool: pay}]\n"))
    spec.functions = {**functions(), "brief": lambda ctx: f"Case {ctx.case.get('id')} in {ctx.state}"}
    client = FakeAnthropic(tool_reply(("pay", {"amount": "1"})))
    factories = Factories(
        classifier=lambda m, r: FakeClassifier({}),
        generator=lambda m, r: Generator(r, m, FakeAnthropic()),
        agent=lambda policy, model, recorder: Agent(policy, recorder, client, model=model),
    )
    run_eval(spec, ["m"], 1, factories)
    assert client.requests[0]["messages"][0]["content"] == "Case None in NEW"


def test_agent_eval_without_any_tool_call(tmp_path: Path) -> None:
    write(tmp_path, "p.yaml", MINI)
    spec = load_spec(write(tmp_path, "a.yaml", "kind: agent\npolicy: p.yaml\ncases: [{expect_tool: pay}]\n"))
    trials = run_eval(spec, ["m"], 1, fakes(agent=[text_reply("I would rather not.")]))
    assert trials[0].answer == {"tool": None, "args": {}} and not trials[0].passed


def test_agent_eval_with_only_a_status_call(tmp_path: Path) -> None:
    write(tmp_path, "p.yaml", MINI)
    spec = load_spec(write(tmp_path, "a.yaml", "kind: agent\npolicy: p.yaml\ncases: [{expect_tool: case_status}]\n"))
    spec.functions = functions()
    trials = run_eval(spec, ["m"], 1, fakes(agent=[tool_reply(("case_status", {}))] * 4))
    assert trials[0].passed


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("- 1", "an eval is a mapping"),
        ("kind: guess\ncases: [1]", "kind must be one of"),
        ("kind: classify", "cases: are required"),
        ("kind: generate\ncases: [{}]", "needs system: and prompt:"),
        ("kind: agent\ncases: [{}]", "needs policy:"),
        ("kind: classify\ncases: [{}]", "needs questions: or policy: \\+ trigger:"),
        ("kind: classify\ntrigger: t\ncases: [{}]", "trigger: needs policy:"),
        ("kind: classify\npolicy: p.yaml\ntrigger: nope\ncases: [{}]", "no trigger 'nope' with ask:"),
        ("kind: classify\nquestions: {q: {ask: x}}\ncases: [{}]", "needs noul:, choice: or score:"),
    ],
)
def test_spec_mistakes(tmp_path: Path, text: str, message: str) -> None:
    write(tmp_path, "p.yaml", MINI)
    with pytest.raises(DeskError, match=message):
        load_spec(write(tmp_path, "e.yaml", text))


def test_noul_below_defaults_and_score_without_levels(tmp_path: Path) -> None:
    spec = load_spec(
        write(
            tmp_path,
            "e.yaml",
            "kind: classify\nquestions:\n  a: {noul: q}\n  b: {score: s}\n  c: {choice: c}\ncases: [{text: x}]\n",
        )
    )
    assert (spec.questions["a"].above, spec.questions["a"].below) == (0.5, 0.5)
    assert spec.questions["b"].levels == () and spec.questions["c"].options == ()


def test_table_and_save(tmp_path: Path) -> None:
    trials = [
        Trial("m", "c1", 1, True, {"a": 1}, 0.2, 10, 2, 0.001, same_as="1"),
        Trial("m", "c1", 2, False, {"a": 2}, 0.4, 10, 2, 0.001, same_as="2"),
        Trial("m", "c2", 1, True, "x", 0.6, 10, 2, 0.002, same_as="x"),
    ]
    rows = summarise(trials)
    assert rows[0]["stability"] == 0.75 and rows[0]["cost_usd"] == 0.004 and rows[0]["seconds_p95"] == 0.6
    text = table(rows)
    assert "m" in text and "$0.0040" in text and "failing: c1" in text
    assert "n/a" in table([{**rows[0], "cost_usd": None, "failing": []}])
    spec = load_spec(write(tmp_path, "e.yaml", CLASSIFY))
    path = save(spec, rows, trials, tmp_path / "out", "20261009")
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert path.name == "triage-20261009.json" and saved["summary"] == rows and len(saved["trials"]) == 3


def test_stability_compares_what_matters_for_each_kind() -> None:
    from desk.evals import _same_as

    assert _same_as("agent", {"tool": "pay", "args": {"note": "free text"}}, True) == "pay"
    assert _same_as("agent", None, False) == "None"
    assert _same_as("generate", "any words", True) == "pass" and _same_as("generate", "x", False) == "fail"
    assert _same_as("classify", {"b": 1, "a": 2}, True) == '{"a": 2, "b": 1}'


def test_default_factories_pick_the_backend() -> None:
    f = default_factories()
    recorder = Recorder()
    assert isinstance(f.classifier("jev-latest", recorder), JevClassifier)
    assert isinstance(f.classifier("claude-haiku-5-5", recorder), ClaudeClassifier)
    assert f.generator("claude-sonnet-5-5", recorder).model == "claude-sonnet-5-5"
    from desk.policy import parse_policy

    assert f.agent(parse_policy(MINI, "p"), "claude-opus-5-5", recorder).model == "claude-opus-5-5"
