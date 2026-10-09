"""Evals: the same situations run N times on each model, to choose wording and models by numbers.

    kind: classify   questions (inline, or a trigger's `ask:` from the policy) → expected answers
    kind: generate   a system prompt and a prompt → text that must / must not contain phrases
    kind: agent      a case in a given state → the tool the agent should call first, and its args

For each model: how often it is right, how stable it is (how often a case gets its most common
answer), time per call, tokens, and cost. Unlike the unit tests these call real models: run them
on purpose, `desk eval evals/<file>.yaml`.
"""

from __future__ import annotations

import json
import statistics
import time
from collections import Counter
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from desk.agent import Agent
from desk.calls import Recorder
from desk.classifier import Choice, Classifier, ClaudeClassifier, JevClassifier, Noul, Question, Score
from desk.errors import DeskError
from desk.generator import Generator
from desk.machine import Case, Function, Services, load_functions
from desk.policy import Ask, Policy, load_policy
from desk.runner import BriefingContext, load_data

KINDS = ("classify", "generate", "agent")


@dataclass
class Factories:
    classifier: Callable[[str, Recorder], Classifier]
    generator: Callable[[str, Recorder], Generator]
    agent: Callable[[Policy, str, Recorder], Agent]


def default_factories() -> Factories:
    def classifier(model: str, recorder: Recorder) -> Classifier:
        if model.startswith("jev"):
            return JevClassifier(recorder, model)
        return ClaudeClassifier(recorder, model)

    return Factories(
        classifier=classifier,
        generator=lambda model, recorder: Generator(recorder, model),
        agent=lambda policy, model, recorder: Agent(policy, recorder, model=model),
    )


@dataclass
class Trial:
    model: str
    case: str
    run: int
    passed: bool
    answer: Any
    seconds: float
    input_tokens: int
    output_tokens: int
    cost_usd: float | None
    error: str | None = None
    same_as: str = ""
    """What stability compares: the answers for a classifier, the tool for an agent, and for
    generated text (never the same twice) whether it passed."""


@dataclass
class Spec:
    name: str
    kind: str
    models: list[str]
    runs: int
    cases: list[dict[str, Any]]
    questions: dict[str, Ask] = field(default_factory=dict)
    system: str = ""
    prompt: str = ""
    policy: Policy | None = None
    functions: Mapping[str, Function] = field(default_factory=dict)
    world: Any = None


def load_spec(path: str | Path) -> Spec:
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise DeskError(f"{path}: an eval is a mapping")
    kind = raw.get("kind")
    if kind not in KINDS:
        raise DeskError(f"{path}: kind must be one of {', '.join(KINDS)}")
    cases = raw.get("cases")
    if not isinstance(cases, list) or not cases:
        raise DeskError(f"{path}: cases: are required")
    base = path.parent
    spec = Spec(
        name=str(raw.get("eval", path.stem)),
        kind=kind,
        models=[str(m) for m in raw.get("models") or []],
        runs=int(raw.get("runs", 3)),
        cases=[dict(c, name=str(c.get("name", i + 1))) for i, c in enumerate(cases)],
        system=str(raw.get("system", "")),
        prompt=str(raw.get("prompt", "")),
    )
    if raw.get("policy"):
        spec.policy = load_policy(base / str(raw["policy"]))
    if raw.get("functions"):
        spec.functions, _ = load_functions(base / str(raw["functions"]))
    if raw.get("world"):
        spec.world = load_data(base / str(raw["world"]))
    if kind == "classify":
        spec.questions = _questions(raw, spec.policy, path)
    if kind == "generate" and not (spec.system and spec.prompt):
        raise DeskError(f"{path}: a generate eval needs system: and prompt:")
    if kind == "agent" and spec.policy is None:
        raise DeskError(f"{path}: an agent eval needs policy:")
    return spec


def _questions(raw: dict[str, Any], policy: Policy | None, path: Path) -> dict[str, Ask]:
    trigger = raw.get("trigger")
    if trigger:
        if policy is None:
            raise DeskError(f"{path}: trigger: needs policy:")
        found = next((t for t in policy.triggers if t.name == trigger), None)
        if found is None or not found.ask:
            raise DeskError(f"{path}: the policy has no trigger {trigger!r} with ask:")
        return {a.field: a for a in found.ask}
    inline = raw.get("questions")
    if not isinstance(inline, dict) or not inline:
        raise DeskError(f"{path}: a classify eval needs questions: or policy: + trigger:")
    out: dict[str, Ask] = {}
    for name, spec in inline.items():
        if "noul" in spec:
            out[name] = Ask(
                name,
                "noul",
                str(spec["noul"]),
                above=float(spec.get("above", 0.5)),
                below=float(spec.get("below", min(float(spec.get("above", 0.5)), 0.5))),
            )
        elif "choice" in spec:
            out[name] = Ask(
                name,
                "choice",
                str(spec["choice"]),
                options=tuple((str(k), str(v or "")) for k, v in (spec.get("options") or {}).items()),
            )
        elif "score" in spec:
            out[name] = Ask(name, "score", str(spec["score"]), levels=tuple(str(v) for v in spec.get("levels") or []))
        else:
            raise DeskError(f"{path}: question {name!r} needs noul:, choice: or score:")
    return out


def _as_question(ask: Ask) -> Question:
    if ask.kind == "noul":
        return Noul(ask.instructions)
    if ask.kind == "choice":
        return Choice(ask.instructions, dict(ask.options))
    return Score(ask.instructions, ask.levels)


def _classify(spec: Spec, case: dict[str, Any], model: str, recorder: Recorder, f: Factories) -> tuple[Any, bool]:
    classifier = f.classifier(model, recorder)
    answers = classifier.classify(str(case["text"]), {n: _as_question(a) for n, a in spec.questions.items()})
    got: dict[str, Any] = {}
    for name, ask in spec.questions.items():
        value = answers[name].value
        if ask.kind == "noul":
            got[name] = True if value >= ask.above else False if value <= ask.below else None
        else:
            got[name] = value
    expect = case.get("expect") or {}
    return got, all(got.get(k) == v for k, v in expect.items())


def _generate(spec: Spec, case: dict[str, Any], model: str, recorder: Recorder, f: Factories) -> tuple[Any, bool]:
    variables = case.get("vars") or {}
    text = f.generator(model, recorder).generate(spec.system.format(**variables), spec.prompt.format(**variables))
    low = text.lower()
    passed = all(p.lower() in low for p in case.get("expect_contains") or []) and not any(
        p.lower() in low for p in case.get("expect_not_contains") or []
    )
    return text, passed


def _agent(spec: Spec, case: dict[str, Any], model: str, recorder: Recorder, f: Factories) -> tuple[Any, bool]:
    assert spec.policy is not None
    machine = Case(spec.policy, spec.functions, case.get("case") or {}, spec.world, Services(), case["name"])
    given = case.get("given") or {}
    if given.get("state"):
        machine.state = str(given["state"])
    for name, value in (given.get("fields") or {}).items():
        machine.fields[name] = spec.policy.coerce(name, value, "eval given")
    briefing = case.get("briefing") or _briefing(spec, machine)
    run = f.agent(spec.policy, model, recorder).run(machine, briefing, max_turns=4, until_signal=True)
    calls = [c for c in run.tool_calls if c["tool"] != "case_status"] or run.tool_calls
    first = calls[0] if calls else None
    got = {"tool": first["tool"], "args": first["input"]} if first else {"tool": None, "args": {}}
    want_args = case.get("expect_args") or {}
    passed = got["tool"] == case.get("expect_tool") and all(
        str(got["args"].get(k)).lower() == str(v).lower() for k, v in want_args.items()
    )
    return got, passed


def _briefing(spec: Spec, machine: Case) -> str:
    """What the agent would read in a real run: the policy's briefing function, if it has one."""
    assert spec.policy is not None
    if spec.policy.briefing and spec.policy.briefing in spec.functions:
        return str(spec.functions[spec.policy.briefing](BriefingContext(machine)))  # type: ignore[arg-type]
    return "A new case:\n" + json.dumps(machine.input, ensure_ascii=False, indent=2)


RUNNERS = {"classify": _classify, "generate": _generate, "agent": _agent}


def run_eval(
    spec: Spec,
    models: list[str],
    runs: int,
    factories: Factories | None = None,
    workers: int = 4,
    clock: Callable[[], float] = time.perf_counter,
) -> list[Trial]:
    f = factories or default_factories()
    jobs = [(model, case, n) for model in models for case in spec.cases for n in range(1, runs + 1)]

    def one(job: tuple[str, dict[str, Any], int]) -> Trial:
        model, case, n = job
        recorder = Recorder(clock=clock)
        started = clock()
        try:
            answer, passed = RUNNERS[spec.kind](spec, case, model, recorder, f)
            error = None
        except Exception as failure:  # an eval reports a failing model, it does not stop
            answer, passed, error = None, False, f"{type(failure).__name__}: {failure}"
        totals = recorder.totals()
        return Trial(
            model,
            case["name"],
            n,
            passed,
            answer,
            clock() - started,
            totals["input_tokens"],
            totals["output_tokens"],
            totals["cost_usd"],
            error,
            _same_as(spec.kind, answer, passed),
        )

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        return list(pool.map(one, jobs))


def _same_as(kind: str, answer: Any, passed: bool) -> str:
    if kind == "agent":
        return str((answer or {}).get("tool"))
    if kind == "generate":
        return "pass" if passed else "fail"
    return json.dumps(answer, sort_keys=True, default=str)


def summarise(trials: list[Trial]) -> list[dict[str, Any]]:
    out = []
    for model in dict.fromkeys(t.model for t in trials):
        mine = [t for t in trials if t.model == model]
        by_case: dict[str, list[Trial]] = {}
        for t in mine:
            by_case.setdefault(t.case, []).append(t)
        stability = [Counter(t.same_as for t in ts).most_common(1)[0][1] / len(ts) for ts in by_case.values()]
        seconds = sorted(t.seconds for t in mine)
        costs = [t.cost_usd for t in mine]
        out.append(
            {
                "model": model,
                "trials": len(mine),
                "passed": sum(t.passed for t in mine),
                "accuracy": round(sum(t.passed for t in mine) / len(mine), 3),
                "stability": round(statistics.mean(stability), 3),
                "errors": sum(t.error is not None for t in mine),
                "seconds_mean": round(statistics.mean(seconds), 3),
                "seconds_p95": round(seconds[min(len(seconds) - 1, int(0.95 * len(seconds)))], 3),
                "input_tokens": sum(t.input_tokens for t in mine),
                "output_tokens": sum(t.output_tokens for t in mine),
                "cost_usd": None if any(c is None for c in costs) else round(sum(c for c in costs if c is not None), 6),
                "failing": sorted({t.case for t in mine if not t.passed}),
            }
        )
    return out


def table(rows: list[dict[str, Any]]) -> str:
    head = (
        f"{'model':28} {'right':>9} {'stable':>7} {'errors':>6} {'s/call':>7} {'p95':>6} "
        f"{'tokens in/out':>15} {'cost':>9}"
    )
    lines = [head, "-" * len(head)]
    for r in rows:
        cost = "n/a" if r["cost_usd"] is None else f"${r['cost_usd']:.4f}"
        lines.append(
            f"{r['model']:28} {r['passed']:>4}/{r['trials']:<4} {r['stability']:>7.2f} {r['errors']:>6} "
            f"{r['seconds_mean']:>7.2f} {r['seconds_p95']:>6.2f} "
            f"{str(r['input_tokens']) + '/' + str(r['output_tokens']):>15} {cost:>9}"
        )
        if r["failing"]:
            lines.append(f"{'':28} failing: {', '.join(r['failing'])}")
    return "\n".join(lines)


def save(spec: Spec, rows: list[dict[str, Any]], trials: list[Trial], out_dir: Path, stamp: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{spec.name}-{stamp}.json"
    path.write_text(
        json.dumps(
            {"eval": spec.name, "kind": spec.kind, "summary": rows, "trials": [t.__dict__ for t in trials]},
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    return path
