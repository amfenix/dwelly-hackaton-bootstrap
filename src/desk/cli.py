"""The command line.

    desk check                       load the policy and the functions; print the agent's tools
    desk run                         every case → ANSWERS/, traces/, results.jsonl
    desk run --replay                the same again from traces/, into replay/: no keys, no network
    desk eval evals/triage.yaml      run an eval on real models; --models a,b --runs 5

Paths default to the track/ folder: track/policy.yaml, track/functions.py, track/world.json,
track/cases.json. Keys come from the environment: ANTHROPIC_API_KEY, TYPESAFE_API_KEY. A .env file
in the working folder is read first (`--env` to name another); a variable already set wins.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from desk.agent import Agent, tools_for
from desk.calls import Recorder
from desk.classifier import Classifier, make_classifier
from desk.errors import DeskError
from desk.evals import Factories, load_spec, run_eval, save, summarise, table
from desk.generator import Generator
from desk.machine import load_functions
from desk.policy import Policy, load_policy
from desk.runner import Setup, load_cases, load_data, run_batch

REPLAY_DIR = "replay"
"""Where a replay writes its run, unless --out, --traces and --results say otherwise."""


def _paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--policy", default="track/policy.yaml")
    parser.add_argument("--functions", default="track/functions.py")
    parser.add_argument("--world", default="track/world.json", help="ignored when the file does not exist")


def _load(args: argparse.Namespace) -> tuple[Policy, dict[str, Any], Any, Any]:
    policy = load_policy(args.policy)
    functions, module = load_functions(args.functions)
    policy.check_functions(functions)
    world = load_data(args.world) if Path(args.world).exists() else None
    return policy, functions, world, module


def check(args: argparse.Namespace, out: Callable[[str], None]) -> int:
    policy, functions, _, _ = _load(args)
    out(
        f"{policy.name} v{policy.version}: {len(policy.fields)} fields, {len(policy.states)} states, "
        f"{len(policy.signals)} signals, {len(policy.triggers)} rules; functions: {', '.join(sorted(functions))}"
    )
    for warning in policy.warnings:
        out(f"warning: {warning}")
    if args.tools:
        out(json.dumps(tools_for(policy), indent=2))
    else:
        for tool in tools_for(policy):
            out(f"  tool {tool['name']}({', '.join(tool['input_schema']['properties'])})")
    return 0


def run(args: argparse.Namespace, out: Callable[[str], None]) -> int:
    policy, functions, world, module = _load(args)
    for warning in policy.warnings:
        out(f"warning: {warning}")

    def classifier(recorder: Recorder) -> Classifier:
        return make_classifier(args.classifier, recorder, args.classifier_model)

    setup = Setup(
        policy=policy,
        functions=functions,
        world=world,
        adapt_case=getattr(module, "adapt_case", None),
        make_classifier=classifier,
        make_generator=lambda recorder: Generator(recorder, args.generator_model),
        make_agent=None if args.no_agent else (lambda p, recorder: Agent(p, recorder, model=args.agent_model)),
        summarise=not args.no_summary,
    )
    cases = load_cases(args.cases)
    # A replay writes a run of its own, so the traces it replays stay as they were.
    base = Path(REPLAY_DIR) if args.replay else Path()
    started = time.monotonic()
    counts = run_batch(
        setup,
        cases,
        Path(args.out) if args.out else base / "ANSWERS",
        Path(args.traces) if args.traces else base / "traces",
        Path(args.results) if args.results else base / "results.jsonl",
        workers=args.workers,
        deadline_seconds=args.deadline,
        resume=not args.no_resume,
        replay_from=Path(args.replay) if args.replay else None,
        progress=out,
    )
    out(
        f"{counts['cases']} cases: {counts['worked']} worked, {counts['skipped']} already answered, "
        f"{counts['deadline']} past the deadline, {counts['errors']} with errors"
        + (f", {counts['unwritten']} not written" if counts["unwritten"] else "")
        + f" ({time.monotonic() - started:.1f} s)"
    )
    return 1 if counts["unwritten"] else 0


def evaluate(args: argparse.Namespace, out: Callable[[str], None], factories: Factories | None = None) -> int:
    spec = load_spec(args.file)
    models = [m.strip() for m in args.models.split(",")] if args.models else spec.models
    if not models:
        raise DeskError("no models: list them in the eval (models:) or pass --models")
    runs = args.runs or spec.runs
    trials = run_eval(spec, models, runs, factories, workers=args.workers)
    rows = summarise(trials)
    out(f"{spec.name} ({spec.kind}): {len(spec.cases)} cases x {runs} runs")
    out(table(rows))
    path = save(spec, rows, trials, Path(args.out), time.strftime("%Y%m%d-%H%M%S"))
    out(f"saved {path}")
    return 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="desk", description="Cases in, decisions out, through a policy.")
    root.add_argument("--env", default=".env", help="a KEY=VALUE file read before the command, if it exists")
    sub = root.add_subparsers(dest="command", required=True)

    p = sub.add_parser("check", help="load the policy and functions, list the agent's tools")
    _paths(p)
    p.add_argument("--tools", action="store_true", help="print the tool definitions as JSON")

    p = sub.add_parser("run", help="work every case and write the answers")
    _paths(p)
    p.add_argument("--cases", default="track/cases.json")
    p.add_argument("--out", default=None, help="ANSWERS (replay/ANSWERS for a replay)")
    p.add_argument("--traces", default=None, help="traces (replay/traces for a replay)")
    p.add_argument("--results", default=None, help="results.jsonl (replay/results.jsonl for a replay)")
    p.add_argument("--classifier", choices=["claude", "jev"], default="claude")
    p.add_argument("--classifier-model", default=None)
    p.add_argument("--generator-model", default="claude-sonnet-5-5")
    p.add_argument("--agent-model", default=None, help="defaults to the policy's agent.model")
    p.add_argument("--no-agent", action="store_true", help="rules only; the agent is never asked")
    p.add_argument("--no-summary", action="store_true", help="no generated summary at the top of ANSWER.md")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--deadline", type=float, default=None, help="seconds; later cases are answered as not worked")
    p.add_argument("--no-resume", action="store_true", help="work cases that already have an answer again")
    p.add_argument(
        "--replay",
        nargs="?",
        const="traces",
        default=None,
        metavar="TRACES",
        help="answer every model call from these traces (default traces/): no network",
    )

    p = sub.add_parser("eval", help="run an eval on real models")
    p.add_argument("file")
    p.add_argument("--models", default=None, help="comma-separated; overrides the eval's models")
    p.add_argument("--runs", type=int, default=None)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out", default="evals/results")
    return root


COMMANDS = {"check": check, "run": run, "eval": evaluate}


def load_env(path: Path) -> list[str]:
    """`KEY=VALUE` lines into the environment; a variable already set is kept. Returns the names set."""
    if not path.is_file():
        return []
    loaded: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        name, value = name.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if name and name not in os.environ:
            os.environ[name] = value
            loaded.append(name)
    return loaded


def main(argv: Sequence[str] | None = None, out: Callable[[str], None] = print) -> int:
    args = parser().parse_args(argv)
    loaded = load_env(Path(args.env))
    if loaded:
        out(f"{args.env}: set {', '.join(loaded)}")  # names only, never values
    try:
        return COMMANDS[args.command](args, out)
    except (DeskError, OSError) as error:
        out(f"error: {error}")
        return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
