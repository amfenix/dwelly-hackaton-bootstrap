"""The command line, and the example track end to end: five cases in, five answers out, then a replay."""

from __future__ import annotations

import json
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from desk import cli
from desk.agent import Agent
from desk.calls import Recorder
from desk.classifier import Answer, Question
from desk.evals import Factories
from desk.generator import Generator
from tests.conftest import FakeResponse, text_reply, tool_reply

TRACK = Path(__file__).resolve().parents[1] / "track"


class KeywordClassifier:
    """Answers the triage questions from words in the text, as a model would on these cases."""

    backend, model = "fake", "keywords"

    def classify(self, text: str, questions: Mapping[str, Question]) -> dict[str, Answer]:
        low = text.lower()
        trade = "electrical" if "socket" in low else "plumbing"
        values = {"is_emergency": 0.95 if "gas" in low else 0.05, "category": "repair", "trade": trade}
        return {name: Answer(values[name]) for name in questions}


class KeywordModel:
    """A Messages client: books the trade the message needs, then stops when the case says so."""

    def __init__(self) -> None:
        self.messages = self

    def create(self, **request: Any) -> FakeResponse:
        brief = request["messages"][0]["content"]
        if len(request["messages"]) > 1:
            return FakeResponse(
                {
                    "content": [{"type": "text", "text": "Done."}],
                    "stop_reason": "end_turn",
                    "model": request["model"],
                    "usage": {"input_tokens": 5, "output_tokens": 2},
                }
            )
        trade = "electrical" if "socket" in brief.lower() else "plumbing"
        return FakeResponse(tool_reply(("book_repair", {"trade": trade, "access": "weekdays"}), model=request["model"]))


class SummaryModel:
    """A Messages client for the generator: one summary line naming the outcome it was shown."""

    def __init__(self) -> None:
        self.messages = self

    def create(self, **request: Any) -> FakeResponse:
        record = json.loads(request["messages"][0]["content"].split("\n\n")[0].removeprefix("<record>\n"))
        return FakeResponse(text_reply(f"The desk decided: {record['outcome']}.", model=request["model"]))


@pytest.fixture
def track(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    shutil.copytree(TRACK, tmp_path / "track")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "make_classifier", lambda backend, recorder, model: _recorded(recorder))
    monkeypatch.setattr(
        cli, "Agent", lambda policy, recorder, model=None: Agent(policy, recorder, KeywordModel(), model)
    )
    monkeypatch.setattr(cli, "Generator", lambda recorder, model: Generator(recorder, model, SummaryModel()))
    return tmp_path


def _recorded(recorder: Recorder) -> Any:
    """The keyword classifier, through the recorder, so a replay has something to answer from."""
    inner = KeywordClassifier()

    class Recorded:
        backend, model = "fake", "keywords"

        def classify(self, text: str, questions: Mapping[str, Question]) -> dict[str, Answer]:
            request = {"text": text, "questions": sorted(questions)}
            body = recorder.call(
                "classify",
                "fake",
                "keywords",
                request,
                lambda: ({k: a.value for k, a in inner.classify(text, questions).items()}, _usage(), "keywords"),
            )
            return {k: Answer(v) for k, v in body.items()}

    return Recorded()


def _usage() -> Any:
    from desk.calls import Usage

    return Usage(3, 1)


def lines_of(argv: list[str]) -> tuple[int, list[str]]:
    out: list[str] = []
    return cli.main(argv, out.append), out


def test_check_lists_the_tools(track: Path) -> None:
    code, out = lines_of(["check"])
    assert code == 0 and out[0].startswith("repairs-desk v1: 10 fields")
    assert out[1:] == [
        "  tool book_repair(trade, access)",
        "  tool reply(about)",
        "  tool escalate(note)",
        "  tool case_status()",
    ]
    code, out = lines_of(["check", "--tools"])
    assert json.loads(out[-1])[0]["name"] == "book_repair"


def test_check_prints_warnings(track: Path) -> None:
    policy = track / "track" / "policy.yaml"
    policy.write_text(
        policy.read_text(encoding="utf-8").replace(
            "signals:\n", "signals:\n  idle:\n    description: never accepted\n"
        ),
        encoding="utf-8",
    )
    code, out = lines_of(["check"])
    assert code == 0 and "warning: signal 'idle' has no trigger, so it is always refused" in out
    _, out = lines_of(["run", "--no-agent", "--cases", "track/cases.json"])
    assert out[0] == "warning: signal 'idle' has no trigger, so it is always refused"
    assert "1: escalate (ESCALATED)" in out  # with no agent, an open case goes to on_stall


def test_the_example_track_end_to_end(track: Path) -> None:
    code, out = lines_of(["run", "--workers", "2"])
    assert code == 0, out
    results = {r["id"]: r for r in map(json.loads, (track / "results.jsonl").read_text().splitlines())}
    assert {k: (r["outcome"], r["fields"]["reason"]) for k, r in results.items()} == {
        "1": ("act", None),
        "2": ("escalate", "EMERGENCY"),
        "3": ("escalate", "OVER_LIMIT"),
        "4": ("escalate", "UNKNOWN_SENDER"),
        "5": ("act", None),
    }
    assert results["1"]["actions"][0] == {
        "type": "book_contractor",
        "args": {"contractor": "C-PIPE", "tenancy": "T-201", "quote_gbp": "95.00"},
    }
    assert results["5"]["actions"][0]["type"] == "book_contractor"  # booked once, after one timeout
    assert len([a for a in results["5"]["actions"] if a["type"] == "book_contractor"]) == 1
    reasoning = (track / "ANSWERS" / "5" / "REASONING.md").read_text(encoding="utf-8")
    assert "attempt 1 failed (book_contractor raised TimeoutError" in reasoning
    assert "quote <= limit → true (quote=95.00, limit=250.00)" in reasoning
    over = (track / "ANSWERS" / "3" / "REASONING.md").read_text(encoding="utf-8")
    assert "quote > limit → true (quote=420.00, limit=250.00)" in over
    assert out[-1].startswith("5 cases: 5 worked, 0 already answered, 0 past the deadline, 0 with errors")
    answer = (track / "ANSWERS" / "3" / "ANSWER.md").read_text(encoding="utf-8")
    assert "## Summary\n\nThe desk decided: escalate.\n\n_Written by a model" in answer

    before = {p.parent.name: p.read_text(encoding="utf-8") for p in (track / "ANSWERS").glob("*/ANSWER.md")}
    shutil.copytree(track / "traces", track / "kept")
    code, out = lines_of(["run", "--replay", "--no-agent"])
    assert code == 0
    # --no-agent: the replay still needs the agent's recorded calls, so it is the agent that is missing
    assert "with errors" in out[-1]
    shutil.rmtree(track / "traces")
    shutil.copytree(track / "kept", track / "traces")
    code, out = lines_of(["run", "--replay"])
    after = {p.parent.name: p.read_text(encoding="utf-8") for p in (track / "ANSWERS").glob("*/ANSWER.md")}
    assert code == 0 and "0 with errors" in out[-1] and after == before


def test_no_summary_leaves_it_out(track: Path) -> None:
    code, _ = lines_of(["run", "--no-agent", "--no-summary"])
    assert code == 0
    assert "## Summary" not in (track / "ANSWERS" / "1" / "ANSWER.md").read_text(encoding="utf-8")


def test_errors_are_reported_not_raised(track: Path) -> None:
    code, out = lines_of(["check", "--policy", "nowhere.yaml"])
    assert code == 1 and out[0].startswith("error:")
    code, out = lines_of(["eval", str(track / "missing.yaml")])
    assert code == 1


def test_eval_command(track: Path) -> None:
    spec = track / "e.yaml"
    spec.write_text(
        "kind: classify\nquestions: {a: {noul: q}}\ncases: [{text: x, expect: {a: true}}]\n", encoding="utf-8"
    )
    code, out = lines_of(["eval", str(spec)])
    assert code == 1 and "no models" in out[0]

    class Yes:
        backend, model = "fake", "yes"

        def classify(self, text: str, questions: Mapping[str, Question]) -> dict[str, Answer]:
            return {"a": Answer(0.9)}

    factories = Factories(classifier=lambda m, r: Yes(), generator=None, agent=None)  # type: ignore[arg-type]
    out = []
    args = cli.parser().parse_args(["eval", str(spec), "--models", "yes", "--runs", "2", "--out", str(track / "r")])
    assert cli.evaluate(args, out.append, factories) == 0
    assert out[0] == "e (classify): 1 cases x 2 runs" and "2/2" in out[1] and out[2].startswith("saved ")
    assert list((track / "r").glob("e-*.json"))


def test_dot_env_is_read_without_overriding(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ALREADY_SET", "kept")
    for name in ("DESK_T1", "DESK_T2", "DESK_T3", "DESK_EMPTY"):
        monkeypatch.delenv(name, raising=False)
    (tmp_path / ".env").write_text(
        "# a comment\n\nDESK_T1=plain\nexport DESK_T2='single quoted'\nDESK_T3=\"double\"\n"
        "ALREADY_SET=replaced\nDESK_EMPTY=\nnot a pair\n=no name\n",
        encoding="utf-8",
    )
    assert cli.load_env(tmp_path / ".env") == ["DESK_T1", "DESK_T2", "DESK_T3", "DESK_EMPTY"]
    import os

    assert (os.environ["DESK_T1"], os.environ["DESK_T2"], os.environ["DESK_T3"]) == ("plain", "single quoted", "double")
    assert os.environ["ALREADY_SET"] == "kept" and os.environ["DESK_EMPTY"] == ""
    assert cli.load_env(tmp_path / "missing.env") == []


def test_main_reads_dot_env_and_prints_only_names(track: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DESK_SECRET", raising=False)
    (track / "keys.env").write_text("DESK_SECRET=s3cr3t-value\n", encoding="utf-8")
    code, out = lines_of(["--env", "keys.env", "check"])
    assert code == 0 and out[0] == "keys.env: set DESK_SECRET"
    assert not any("s3cr3t-value" in line for line in out)
