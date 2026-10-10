"""One case, one state machine.

    case = Case(policy, functions, case_input, world, services)
    case.start()                        # rules with no signal run until nothing more fires
    case.send("book_contractor", {...}) # the agent asks; a rule accepts it, or it is refused

Only the policy closes a case: a final state is the outcome the case has now, and a later rule may
still move it (book the repair, then hand the refund demand to a person). A rule (trigger) fires
when its condition is true: it asks the classifier or calls a function,
writes what came back to fields, applies its `set:`, and moves to its `to:`. Then rules with no
signal run again, in file order, the first true one each time, until none fires. A rule with no
signal fires at most once per visit to a state, so a rule that does not move the case cannot
loop. Everything that happens is written to the journal: that is the case's reasoning.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
import types
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any, TypeVar

from desk.classifier import Answer, Choice, Classifier, Noul, Question, Score
from desk.errors import CaseError, ModelError, PolicyError
from desk.expr import Atom, to_decimal
from desk.generator import Generator
from desk.policy import UNKNOWN, Arg, Effect, Policy, Signal, Trigger, coerce

Function = Callable[["Context"], Mapping[str, Any] | None]
"""A `call:` function: it reads the case and returns the fields it found."""

_Marked = TypeVar("_Marked", bound=Callable[..., Any])


def function(fn: _Marked) -> _Marked:
    """Marks a function the policy may name in `call:` (a `Function`) or `briefing:` (returns text)."""
    fn.__desk_function__ = True  # type: ignore[attr-defined]
    return fn


def load_functions(path: str | Path) -> tuple[dict[str, Function], types.ModuleType]:
    """The `@function`s of a Python file, by name, and the module (for `adapt_case`)."""
    path = Path(path)
    name = f"desk_functions_{path.stem}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise PolicyError(f"{path}: cannot load functions from it")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    found = {
        attr: value
        for attr, value in vars(module).items()
        if callable(value) and getattr(value, "__desk_function__", False)
    }
    return found, module


@dataclass
class Services:
    """What functions and `ask:` rules may use. Either may be absent in a run that needs none."""

    classifier: Classifier | None = None
    generator: Generator | None = None


@dataclass
class Entry:
    seq: int
    at: float
    kind: str
    data: dict[str, Any]


class Journal:
    def __init__(self, clock: Callable[[], float] = time.perf_counter) -> None:
        self.entries: list[Entry] = []
        self._clock = clock
        self._start = clock()

    def add(self, kind: str, /, **data: Any) -> None:
        self.entries.append(Entry(len(self.entries) + 1, round(self._clock() - self._start, 3), kind, data))

    def of(self, kind: str) -> list[Entry]:
        return [e for e in self.entries if e.kind == kind]


class Context:
    """What a function sees. It returns the fields it found; it may not touch the case otherwise."""

    def __init__(self, case: Case, trigger: Trigger, args: Mapping[str, Any], attempt: int, key: str) -> None:
        self._case, self._trigger = case, trigger
        self.case: Mapping[str, Any] = types.MappingProxyType(case.input)
        self.fields: Mapping[str, Any] = types.MappingProxyType(dict(case.fields))
        self.args: Mapping[str, Any] = types.MappingProxyType(dict(args))
        self.world: Any = case.world
        self.state: str = case.state
        self.attempt = attempt
        self.key = key
        """The same on every retry of this call, different for the next call: dedupe writes on it."""

    def act(self, type_: str, **args: Any) -> None:
        """Record something done in the world (a booking, a message sent). It goes into the answer."""
        self._case.journal.add("action", type=type_, args=args, by=self._trigger.name, key=self.key)

    def note(self, text: str) -> None:
        self._case.journal.add("note", text=str(text), by=self._trigger.name)

    def classify(self, text: str, questions: Mapping[str, Question]) -> dict[str, Answer]:
        if self._case.services.classifier is None:
            raise CaseError("this run has no classifier")
        return self._case.services.classifier.classify(text, questions)

    def generate(self, system: str, prompt: str, **options: Any) -> str:
        if self._case.services.generator is None:
            raise CaseError("this run has no generator")
        return self._case.services.generator.generate(system, prompt, **options)


@dataclass
class SignalResult:
    accepted: bool
    signal: str
    state: str
    final: bool
    outcome: str | None
    trigger: str | None = None
    reason: str | None = None
    changed: dict[str, Any] = field(default_factory=dict)

    def for_agent(self) -> dict[str, Any]:
        out: dict[str, Any] = {"accepted": self.accepted, "state": self.state}
        if self.reason:
            out["refused_because"] = self.reason
        if self.changed:
            out["now_known"] = {k: _plain(v) for k, v in self.changed.items()}
        if self.final:
            out["case_closed"] = self.outcome
        return out


def _plain(value: Any) -> Any:
    if isinstance(value, Decimal):
        return str(value)
    return value


class Case:
    def __init__(
        self,
        policy: Policy,
        functions: Mapping[str, Function],
        case_input: Mapping[str, Any],
        world: Any = None,
        services: Services | None = None,
        case_id: str | None = None,
        journal: Journal | None = None,
    ) -> None:
        self.policy, self.functions = policy, functions
        self.input: dict[str, Any] = dict(case_input)
        self.id = str(case_id if case_id is not None else self.input.get("id", "case"))
        self.world = world
        self.services = services or Services()
        self.journal = journal or Journal()
        self.state = policy.initial
        self.fields: dict[str, Any] = dict.fromkeys(policy.fields)
        self._fired: set[str] = set()
        self._steps = 0

    # ------------------------------------------------------------------------------ reading

    def lookup(self, args: Mapping[str, Any] | None = None) -> Callable[[str], Any]:
        def env(name: str) -> Any:
            if name == "state":
                return self.state
            if name.startswith("args."):
                return (args or {}).get(name[5:])
            return self.fields.get(name)

        return env

    @property
    def final(self) -> bool:
        return self.state in self.policy.final

    @property
    def outcome(self) -> str | None:
        return self.policy.final.get(self.state)

    def status(self) -> dict[str, Any]:
        """What the agent may read: where the case is and every fact known so far."""
        return {
            "state": self.state,
            "final": self.final,
            "outcome": self.outcome,
            "facts": {k: _plain(v) for k, v in self.fields.items() if v is not None},
            "refused": [e.data for e in self.journal.of("refused")][-3:],
        }

    # ------------------------------------------------------------------------------ running

    def start(self) -> None:
        self.journal.add("start", state=self.state, case=self.id, policy=f"{self.policy.name} v{self.policy.version}")
        self.settle()

    def settle(self) -> None:
        # A final state is an outcome, not a lock: the policy's own conditions say what may still happen.
        while True:
            trigger = self._next_internal()
            if trigger is None:
                return
            self._fire(trigger, {})

    def _next_internal(self) -> Trigger | None:
        env = self.lookup()
        for trigger in self.policy.triggers:
            if trigger.on is None and trigger.name not in self._fired and trigger.when.holds(env) is True:
                return trigger
        return None

    def send(self, signal: str, args: Mapping[str, Any] | None = None) -> SignalResult:
        raw = dict(args or {})
        self.journal.add("signal", signal=signal, args=_jsonable(raw))
        spec = self.policy.signals.get(signal)
        if spec is None:
            return self._refuse(signal, f"there is no signal {signal!r}")
        try:
            values = self._args(spec, raw)
        except PolicyError as error:
            return self._refuse(signal, str(error))
        env = self.lookup(values)
        candidates = [t for t in self.policy.triggers if t.on == signal]
        for trigger in candidates:
            if trigger.when.holds(env) is True:
                before = dict(self.fields)
                self._fire(trigger, values)
                self.settle()
                changed = {k: v for k, v in self.fields.items() if before.get(k) != v}
                return SignalResult(True, signal, self.state, self.final, self.outcome, trigger.name, None, changed)
        if not candidates:
            return self._refuse(signal, "no rule accepts this signal")
        reasons = [f"{t.name}: {_failed(t.when.explain(env))}" for t in candidates]
        return self._refuse(signal, "; ".join(reasons))

    def _refuse(self, signal: str, reason: str) -> SignalResult:
        self.journal.add("refused", signal=signal, reason=reason, state=self.state)
        return SignalResult(False, signal, self.state, self.final, self.outcome, None, reason)

    def _args(self, spec: Signal, raw: Mapping[str, Any]) -> dict[str, Any]:
        known = {arg.name: arg for arg in spec.args}
        extra = sorted(set(raw) - set(known))
        if extra:
            raise PolicyError(f"{spec.name} takes no {', '.join(extra)}")
        out: dict[str, Any] = {}
        for name, arg in known.items():
            value = raw.get(name)
            if value is None or value == "":
                if not arg.optional:
                    raise PolicyError(f"{spec.name} needs {name}")
                continue
            out[name] = _coerce_arg(arg, value, spec.name)
        return out

    def _fire(self, trigger: Trigger, args: Mapping[str, Any]) -> None:
        self._steps += 1
        if self._steps > self.policy.max_steps:
            raise CaseError(f"more than {self.policy.max_steps} rule steps: the rules loop")
        if trigger.on is None:
            self._fired.add(trigger.name)
        # One key per firing, kept across its retries: a second firing of the same rule is a new call.
        key = f"{self.id}:{trigger.name}:{self._steps}"
        env = self.lookup(args)
        self.journal.add(
            "trigger",
            name=trigger.name,
            on=trigger.on,
            when=trigger.when.source,
            why=[_atom(a) for a in trigger.when.explain(env)],
            note=trigger.note,
        )
        effect: Effect = trigger.effect
        updates: dict[str, Any] = {}
        if trigger.ask or trigger.call:
            try:
                updates = self._work(trigger, args, key)
            except (ModelError, _FunctionFailed) as error:
                fallback = trigger.on_error or self.policy.on_error
                self.journal.add("failed", trigger=trigger.name, error=str(error))
                if fallback is None:
                    raise CaseError(f"{trigger.name} failed and no on_error says what to do: {error}") from None
                effect = fallback
        for name, value in updates.items():
            self._write(name, value, trigger.name)
        self._apply(effect, trigger.name)

    def _apply(self, effect: Effect, by: str) -> None:
        for name, value in effect.set:
            self._write(name, value, by)
        if effect.to is not None and effect.to != self.state:
            self.journal.add("state", frm=self.state, to=effect.to, by=by)
            self.state = effect.to
            self._fired = set()
            if self.final:
                self.journal.add("final", state=self.state, outcome=self.outcome)

    def apply_stall(self, why: str) -> None:
        """The agent stopped before the case closed: the policy's `on_stall` decides what that means."""
        self.journal.add("stalled", why=why, state=self.state)
        if self.policy.on_stall is None:
            raise CaseError(f"the agent stopped in {self.state} ({why}) and the policy has no on_stall")
        self._apply(self.policy.on_stall, "on_stall")
        self.settle()

    def fail(self) -> None:
        """The run broke outside any rule: the policy's `on_error` says where the case ends.

        No rule runs after it: whatever broke the run could break it again. A final state the policy
        already reached stands: what was done was done, and the failure is recorded beside it.
        """
        if self.policy.on_error is not None and not self.final:
            self._apply(self.policy.on_error, "on_error")

    def _write(self, name: str, value: Any, by: str) -> None:
        value = self.policy.coerce(name, value, f"rule {by}")
        old = self.fields.get(name)
        if old != value:
            self.fields[name] = value
            self.journal.add("set", field=name, old=_plain(old), new=_plain(value), by=by)

    def _work(self, trigger: Trigger, args: Mapping[str, Any], key: str) -> dict[str, Any]:
        last: Exception | None = None
        for attempt in range(1, trigger.retry + 2):
            try:
                if trigger.ask:
                    return self._ask(trigger, attempt)
                return self._call(trigger, args, attempt, key)
            except (ModelError, _FunctionFailed) as error:
                last = error
                self.journal.add(
                    "retry" if attempt <= trigger.retry else "gave_up",
                    trigger=trigger.name,
                    attempt=attempt,
                    error=str(error),
                )
        assert last is not None
        raise last

    def _ask(self, trigger: Trigger, attempt: int) -> dict[str, Any]:
        classifier = self.services.classifier
        if classifier is None:
            raise CaseError(f"{trigger.name} asks the classifier, but this run has none")
        text = self._input_text(trigger)
        questions: dict[str, Question] = {}
        for ask in trigger.ask:
            if ask.kind == "noul":
                questions[ask.field] = Noul(ask.instructions)
            elif ask.kind == "choice":
                questions[ask.field] = Choice(ask.instructions, dict(ask.options))
            else:
                questions[ask.field] = Score(ask.instructions, ask.levels)
        answers = classifier.classify(text, questions)
        updates: dict[str, Any] = {}
        for ask in trigger.ask:
            answer = answers[ask.field]
            if ask.kind == "noul":
                probability = float(answer.value)
                value: Any = True if probability >= ask.above else False if probability <= ask.below else None
            else:
                value = answer.value if answer.value in self.policy.fields[ask.field].values else UNKNOWN
            updates[ask.field] = value
            self.journal.add(
                "answer",
                field=ask.field,
                kind=ask.kind,
                raw=answer.value,
                confidence=answer.confidence,
                value=_plain(value),
                backend=classifier.backend,
                model=classifier.model,
                attempt=attempt,
            )
        return updates

    def _input_text(self, trigger: Trigger) -> str:
        if trigger.input is None:
            for key in ("text", "message", "body", "content"):
                if isinstance(self.input.get(key), str):
                    return str(self.input[key])
            return json.dumps(self.input, ensure_ascii=False, default=str)
        source = trigger.input
        try:
            return source.format(case=_Dotted(self.input), fields=_Dotted(self.fields))
        except (KeyError, AttributeError, IndexError) as error:
            raise CaseError(f"{trigger.name}: input {source!r} names something the case lacks: {error}") from None

    def _call(self, trigger: Trigger, args: Mapping[str, Any], attempt: int, key: str) -> dict[str, Any]:
        assert trigger.call is not None
        fn = self.functions[trigger.call]
        ctx = Context(self, trigger, args, attempt, key)
        self.journal.add("call", function=trigger.call, attempt=attempt, key=ctx.key, args=_jsonable(dict(args)))
        try:
            result = fn(ctx)
        except (ModelError, CaseError) as error:
            raise _FunctionFailed(f"{trigger.call}: {error}") from error
        except Exception as error:
            raise _FunctionFailed(f"{trigger.call} raised {type(error).__name__}: {error}") from error
        if result is None:
            return {}
        if not isinstance(result, Mapping):
            raise CaseError(f"{trigger.call} must return a dict of fields, not {type(result).__name__}")
        unknown = sorted(set(result) - set(self.policy.fields))
        if unknown:
            raise CaseError(f"{trigger.call} returned fields the policy does not declare: {', '.join(unknown)}")
        self.journal.add("returned", function=trigger.call, values={k: _plain(v) for k, v in result.items()})
        return dict(result)


class _FunctionFailed(Exception):
    """A function raised: retried, then handed to on_error."""


class _Dotted:
    """Lets `input: "{case.text} {fields.category}"` read nested keys."""

    def __init__(self, data: Mapping[str, Any]) -> None:
        self._data = data

    def __getattr__(self, name: str) -> Any:
        value = self._data[name]
        return _Dotted(value) if isinstance(value, Mapping) else value


def _coerce_arg(arg: Arg, value: Any, signal: str) -> Any:
    try:
        return coerce(arg.type, arg.values, value, f"{signal}.{arg.name}")
    except PolicyError:
        if arg.type in ("number", "money"):
            raise PolicyError(f"{signal}.{arg.name} must be a number, not {value!r}") from None
        raise


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return _plain(value)


def _atom(atom: Atom) -> dict[str, Any]:
    return {"test": atom.text, "result": atom.result, "values": {k: _plain(v) for k, v in atom.values}}


def _failed(atoms: list[Atom]) -> str:
    bad = [a for a in atoms if a.result is not True]
    parts = []
    for atom in bad:
        shown = ", ".join(f"{k}={_plain(v) if v is not None else 'not known'}" for k, v in atom.values)
        verdict = "not known" if atom.result is None else "false"
        parts.append(f"{atom.text} is {verdict}" + (f" ({shown})" if shown else ""))
    return "; ".join(parts) or "its condition does not hold"


__all__ = [
    "Case",
    "Context",
    "Entry",
    "Function",
    "Journal",
    "Services",
    "SignalResult",
    "function",
    "load_functions",
    "to_decimal",
]
