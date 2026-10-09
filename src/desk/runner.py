"""Runs a batch of cases, unattended: every case ends with an answer, whatever goes wrong.

    ANSWERS/<n>/ANSWER.md      the decision, the actions, what was escalated and what failed
    ANSWERS/<n>/REASONING.md   the journal, read aloud
    traces/<id>.json           the case, the journal and every model call (the input for a replay)
    results.jsonl              one line per case, for scoring

A crash, a broken function, a model that will not answer, or the deadline still writes an
answer: escalated, with the reason. A rerun skips cases that already have one (`resume`). A replay
reads the model calls from one trace folder and writes a new run elsewhere, so the evidence it
replays is never overwritten.
"""

from __future__ import annotations

import json
import re
import threading
import time
from collections import Counter
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from desk.agent import Agent
from desk.answers import answer_md, reasoning_md
from desk.calls import CallRecord, Recorder
from desk.classifier import Classifier
from desk.errors import DeskError
from desk.generator import Generator
from desk.machine import Case, Function, Journal, Services
from desk.policy import Policy
from desk.result import ERROR_OUTCOME, Failure, RunResult
from desk.summary import add_summary


@dataclass
class Setup:
    """Everything a run needs, built once. `make_*` build per-case model clients on one recorder."""

    policy: Policy
    functions: Mapping[str, Function]
    world: Any = None
    adapt_case: Callable[[dict[str, Any]], dict[str, Any]] | None = None
    make_classifier: Callable[[Recorder], Classifier] | None = None
    make_generator: Callable[[Recorder], Generator] | None = None
    make_agent: Callable[[Policy, Recorder], Agent] | None = None
    summarise: bool = False
    """Put a generated summary at the top of each ANSWER.md (needs `make_generator`)."""


# ------------------------------------------------------------------------------------- input


def load_data(path: str | Path) -> Any:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix in (".yaml", ".yml"):
        return yaml.safe_load(text)
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    return json.loads(text)


def load_cases(path: str | Path) -> list[dict[str, Any]]:
    """A list of cases, `{"cases": [...]}`, or JSON lines. A case without an `id` gets its position."""
    data = load_data(path)
    if isinstance(data, dict):
        data = data.get("cases")
    if not isinstance(data, list):
        raise DeskError(f"{path}: expected a list of cases or {{'cases': [...]}}")
    out = []
    for index, item in enumerate(data, start=1):
        case = dict(item) if isinstance(item, dict) else {"text": str(item)}
        case.setdefault("id", str(index))
        out.append(case)
    return out


def folder_name(case_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(case_id)).strip("._") or "case"


def check_ids(cases: list[dict[str, Any]]) -> None:
    """Two cases that would write to the same folder are refused before anything runs."""
    seen: dict[str, str] = {}
    for case in cases:
        case_id = str(case["id"])
        name = folder_name(case_id).lower()  # folders are case-insensitive on some systems
        if name in seen:
            raise DeskError(f"cases {seen[name]!r} and {case_id!r} would share the answer folder {name!r}")
        seen[name] = case_id


def recorded_calls(traces: Path, case_id: str) -> list[CallRecord]:
    path = traces / f"{folder_name(case_id)}.json"
    if not path.exists():
        raise DeskError(f"no trace for case {case_id} at {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [CallRecord.from_json(c) for c in raw.get("calls", [])]


# ------------------------------------------------------------------------------------- one case


def run_case(setup: Setup, raw: dict[str, Any], recorder: Recorder) -> RunResult:
    """One case to the end. Never raises: a failure becomes an escalated answer with the reason."""
    journal = Journal()
    services = Services()
    case: Case | None = None
    agent_text: list[str] = []
    failure: Failure | None = None
    phase = "adapt"
    try:
        case_input = setup.adapt_case(dict(raw)) if setup.adapt_case else dict(raw)
        phase = "start"
        services = _services(setup, recorder)
        case_id = str(case_input.get("id", raw.get("id")))
        case = Case(setup.policy, setup.functions, case_input, setup.world, services, case_id, journal)
        case.start()
        phase = "agent"
        stopped = "no agent in this run"
        if setup.make_agent is not None:
            run = setup.make_agent(setup.policy, recorder).run(case, briefing(setup, case))
            agent_text, stopped = run.text, run.stopped
        phase = "stall"
        if not case.final:
            case.apply_stall(stopped)
    except Exception as error:  # the batch must go on: the reason goes into the answer
        failure = Failure.of(phase, error, journal)
        journal.add("error", phase=phase, error=failure.error)
        if case is not None:
            case.fail()

    if case is None:
        assert failure is not None
        result = RunResult.unworked(setup.policy, raw, failure, journal)
    else:
        result = _result_of(case, failure, agent_text, recorder)
    if setup.summarise and services.generator is not None:
        add_summary(result, services.generator)
    _check_replay(result, recorder)
    return result


def _services(setup: Setup, recorder: Recorder) -> Services:
    return Services(
        classifier=setup.make_classifier(recorder) if setup.make_classifier else None,
        generator=setup.make_generator(recorder) if setup.make_generator else None,
    )


def _result_of(case: Case, failure: Failure | None, agent_text: list[str], recorder: Recorder) -> RunResult:
    return RunResult(
        id=case.id,
        input=case.input,
        state=case.state,
        final=case.final,
        outcome=case.outcome if case.outcome is not None else ERROR_OUTCOME,
        fields=dict(case.fields),
        journal=list(case.journal.entries),
        failure=failure,
        calls=recorder.totals(),
        agent_text=agent_text,
    )


def _check_replay(result: RunResult, recorder: Recorder) -> None:
    unused = recorder.unused()
    if unused and result.failure is None:
        result.failure = Failure("replay", f"{unused} recorded model call(s) were not made again: the run diverged")


def briefing(setup: Setup, case: Case) -> str:
    if setup.policy.briefing is not None:
        text = setup.functions[setup.policy.briefing](BriefingContext(case))  # type: ignore[arg-type]
        if isinstance(text, str):
            return text
        if isinstance(text, Mapping) and isinstance(text.get("briefing"), str):
            return str(text["briefing"])
        raise DeskError(f"briefing {setup.policy.briefing} must return text")
    return "A new case:\n<case>\n" + json.dumps(case.input, ensure_ascii=False, indent=2, default=str) + "\n</case>"


class BriefingContext:
    """What a `briefing:` function sees: the case, the world, the facts so far. Read only."""

    def __init__(self, case: Case) -> None:
        self.case = dict(case.input)
        self.world = case.world
        self.fields = dict(case.fields)
        self.state = case.state


# ------------------------------------------------------------------------------------- the batch


@dataclass
class Batch:
    """Where a run writes. One lock keeps a case's files and its results line together."""

    out: Path
    traces: Path
    results: Path
    lock: threading.Lock = field(default_factory=threading.Lock)

    def done(self, case_id: str) -> bool:
        return (self.out / folder_name(case_id) / "ANSWER.md").exists()

    def write(self, result: RunResult, calls: list[CallRecord]) -> None:
        name = folder_name(result.id)
        trace = json.dumps(result.to_trace(calls), ensure_ascii=False, indent=2, default=str)
        line = json.dumps(result.to_line(), ensure_ascii=False, default=str)
        with self.lock:
            folder = self.out / name
            folder.mkdir(parents=True, exist_ok=True)
            self.traces.mkdir(parents=True, exist_ok=True)
            (folder / "REASONING.md").write_text(reasoning_md(result), encoding="utf-8")
            (self.traces / f"{name}.json").write_text(trace, encoding="utf-8")
            with self.results.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
            # Last: its presence is what marks the case as answered for a resumed run.
            (folder / "ANSWER.md").write_text(answer_md(result), encoding="utf-8")


def run_batch(
    setup: Setup,
    cases: list[dict[str, Any]],
    out: Path,
    traces: Path,
    results: Path,
    *,
    workers: int = 1,
    deadline_seconds: float | None = None,
    resume: bool = True,
    replay_from: Path | None = None,
    progress: Callable[[str], None] = print,
) -> Counter[str]:
    """Works every case and writes its answer.

    Returns counts: cases, skipped, worked, deadline, errors, unwritten.
    """
    check_ids(cases)
    if replay_from is not None and replay_from.resolve() == traces.resolve():
        raise DeskError(f"replay: write the new traces somewhere other than {traces}, the ones being replayed")
    batch = Batch(out, traces, results)
    out.mkdir(parents=True, exist_ok=True)
    results.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    todo = [c for c in cases if not (resume and batch.done(str(c["id"])))]
    counts: Counter[str] = Counter(
        cases=len(cases), skipped=len(cases) - len(todo), worked=0, deadline=0, errors=0, unwritten=0
    )
    counts_lock = threading.Lock()

    def work(raw: dict[str, Any]) -> None:
        late = deadline_seconds is not None and time.monotonic() - started > deadline_seconds
        result, calls = _deadline(setup, raw) if late else _attempt(setup, raw, replay_from)
        try:
            batch.write(result, calls)
            line = _progress_line(result)
            tally = "deadline" if late else "worked"
        except OSError as error:
            line, tally = f"{result.id}: the answer could not be written: {error}", "unwritten"
        with counts_lock:
            counts[tally] += 1
            counts["errors"] += int(result.failure is not None and not late)
        progress(line)

    if workers <= 1:
        for raw in todo:
            work(raw)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(work, todo))  # work() does not raise; list() waits for every case
    return counts


def _progress_line(result: RunResult) -> str:
    line = f"{result.id}: {result.outcome} ({result.state})"
    return f"{line} — {result.failure}" if result.failure else line


def _attempt(setup: Setup, raw: dict[str, Any], replay_from: Path | None) -> tuple[RunResult, list[CallRecord]]:
    recorder = Recorder()
    phase = "replay"
    try:
        if replay_from is not None:
            recorder = Recorder(replay=recorded_calls(replay_from, str(raw["id"])))
        phase = "batch"
        return run_case(setup, raw, recorder), recorder.calls
    except Exception as error:  # last resort: run_case does not raise, but nothing may stop the batch
        return RunResult.unworked(setup.policy, raw, Failure.of(phase, error)), recorder.calls


def _deadline(setup: Setup, raw: dict[str, Any]) -> tuple[RunResult, list[CallRecord]]:
    failure = Failure("deadline", "the batch deadline passed before this case was worked")
    return RunResult.unworked(setup.policy, raw, failure), []


__all__ = ["Batch", "Setup", "check_ids", "load_cases", "load_data", "run_batch", "run_case"]
