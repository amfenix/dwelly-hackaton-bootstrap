"""The policy: a YAML file that says how a case moves and what the agent may ask for.

    fields:    facts about the case; only rules write them, never the agent
    states:    where a case can be; `initial` is where it starts, `final` maps an end to an outcome
    signals:   what the agent may ask for; each one is a tool the agent sees
    triggers:  rules, in order; the first that matches fires
    on_error:  what a failed function or classifier call does, when its trigger does not say
    on_stall:  what happens when the agent stops before the case reaches a final state

Everything is checked here, at load, so a mistake is a clear message before the first case runs,
not a surprise in the middle of a batch.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

import yaml

from desk.errors import PolicyError
from desk.expr import Condition, Kind, compile_condition, to_decimal

TYPES = ("bool", "text", "number", "money", "date", "enum")
UNKNOWN = "unknown"
"""Every enum allows it: the value a classifier gives when the text does not say."""

_NAME = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_STATE = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,63}$")
_RESERVED = {"state", "args", "true", "false", "and", "or", "not", "in", "present", "absent"}
_TOP = {
    "policy",
    "version",
    "description",
    "fields",
    "states",
    "initial",
    "final",
    "signals",
    "triggers",
    "on_error",
    "on_stall",
    "briefing",
    "agent",
    "limits",
}
_TRIGGER = {"name", "on", "when", "ask", "input", "call", "set", "to", "retry", "on_error", "note"}
STRICT_TOOL_LIMIT = 20
"""The API accepts at most this many strict tools in one request."""


@dataclass(frozen=True)
class Field:
    name: str
    type: str
    values: tuple[str, ...] = ()
    description: str = ""

    @property
    def kind(self) -> Kind:
        return Kind(self.type, self.values)


@dataclass(frozen=True)
class Arg:
    name: str
    type: str
    values: tuple[str, ...] = ()
    description: str = ""
    optional: bool = False

    @property
    def kind(self) -> Kind:
        return Kind(self.type, self.values)


@dataclass(frozen=True)
class Signal:
    name: str
    description: str
    args: tuple[Arg, ...]


@dataclass(frozen=True)
class Ask:
    """One classifier question whose answer is written to `field`."""

    field: str
    kind: str  # noul | choice | score
    instructions: str
    options: tuple[tuple[str, str], ...] = ()  # choice: (value, description)
    levels: tuple[str, ...] = ()  # score: values from lowest to highest
    above: float = 0.5  # noul: at or above → true
    below: float = 0.5  # noul: at or below → false; between the two → unknown


@dataclass(frozen=True)
class Effect:
    """What a rule, or a failure, writes: literal values, then a new state."""

    set: tuple[tuple[str, Any], ...] = ()
    to: str | None = None


@dataclass(frozen=True)
class Trigger:
    name: str
    on: str | None
    when: Condition
    ask: tuple[Ask, ...] = ()
    input: str | None = None
    call: str | None = None
    effect: Effect = Effect()
    retry: int = 0
    on_error: Effect | None = None
    note: str = ""


@dataclass(frozen=True)
class AgentSettings:
    model: str = "claude-sonnet-5-5"
    max_turns: int = 12
    max_tokens: int = 8000
    instructions: str = ""


@dataclass
class Policy:
    name: str
    version: str
    description: str
    fields: dict[str, Field]
    states: tuple[str, ...]
    initial: str
    final: dict[str, str]
    signals: dict[str, Signal]
    triggers: tuple[Trigger, ...]
    on_error: Effect | None
    on_stall: Effect | None
    briefing: str | None
    agent: AgentSettings
    max_steps: int
    warnings: list[str] = field(default_factory=list)

    def functions_used(self) -> set[str]:
        names = {t.call for t in self.triggers if t.call is not None}
        if self.briefing is not None:
            names.add(self.briefing)
        return names

    def check_functions(self, available: Mapping[str, Any]) -> None:
        missing = sorted(self.functions_used() - set(available))
        if missing:
            raise PolicyError(
                f"the policy calls functions that do not exist: {', '.join(missing)} "
                f"(defined: {', '.join(sorted(available)) or 'none'})"
            )

    def coerce(self, name: str, value: Any, where: str) -> Any:
        """A value for field `name`, in the field's type; `PolicyError` names the rule that wrote it."""
        spec = self.fields.get(name)
        if spec is None:
            raise PolicyError(f"{where}: no field {name!r}")
        return coerce(spec.type, spec.values, value, f"{where}: {name}")


def coerce(type_: str, values: tuple[str, ...], value: Any, where: str) -> Any:
    if value is None:
        return None
    if type_ == "bool":
        if isinstance(value, bool):
            return value
        raise PolicyError(f"{where} must be true or false, not {value!r}")
    if type_ in ("number", "money"):
        try:
            number: Decimal = to_decimal(value)
        except ValueError as error:
            raise PolicyError(f"{where}: {error}") from None
        return number
    if type_ == "date":
        text = value.isoformat() if isinstance(value, date) else str(value)
        try:
            date.fromisoformat(text)
        except ValueError:
            raise PolicyError(f"{where} must be a date like 2026-10-10, not {value!r}") from None
        return text
    if type_ == "enum":
        text = str(value)
        if text not in values:
            raise PolicyError(f"{where} must be one of {', '.join(values)}, not {value!r}")
        return text
    return value if isinstance(value, str) else str(value)


class YamlLoader(yaml.SafeLoader):
    """YAML with only true/false as booleans: `on:`, `yes`, `no` and `off` stay words (YAML 1.2)."""


YamlLoader.yaml_implicit_resolvers = {
    first: [r for r in resolvers if r[0] != "tag:yaml.org,2002:bool"]
    for first, resolvers in yaml.SafeLoader.yaml_implicit_resolvers.items()
}
YamlLoader.add_implicit_resolver(
    "tag:yaml.org,2002:bool", re.compile(r"^(?:true|True|TRUE|false|False|FALSE)$"), list("tTfF")
)


def read_yaml(text: str) -> Any:
    return yaml.load(text, Loader=YamlLoader)  # a SafeLoader subclass


def load_policy(path: str | Path) -> Policy:
    text = Path(path).read_text(encoding="utf-8")
    return parse_policy(text, str(path))


def parse_policy(text: str, where: str = "policy") -> Policy:
    try:
        raw = read_yaml(text)
    except yaml.YAMLError as error:
        raise PolicyError(f"{where}: not valid YAML: {error}") from None
    if not isinstance(raw, dict):
        raise PolicyError(f"{where}: the policy must be a mapping")
    return _Loader(raw, where).load()


class _Loader:
    def __init__(self, raw: dict[str, Any], where: str) -> None:
        self.raw, self.where = raw, where
        self.warnings: list[str] = []

    def fail(self, message: str) -> PolicyError:
        return PolicyError(f"{self.where}: {message}")

    def load(self) -> Policy:
        unknown = sorted(str(k) for k in (set(self.raw) - _TOP))
        if unknown:
            raise self.fail(f"unknown top-level keys: {', '.join(unknown)}")
        name = self.raw.get("policy")
        if not isinstance(name, str) or not name.strip():
            raise self.fail("'policy:' (a name) is required")
        fields = self.fields()
        states = self.states()
        initial = self.raw.get("initial")
        if initial not in states:
            raise self.fail(f"initial: {initial!r} is not one of the states")
        final = self.final(states)
        signals = self.signals()
        triggers = self.triggers(fields, states, signals)
        limits = self.mapping(self.raw.get("limits"), "limits")
        unknown_limits = sorted(str(k) for k in set(limits) - {"steps"})
        if unknown_limits:
            raise self.fail(f"limits: unknown keys {', '.join(unknown_limits)}")
        max_steps = self.positive_int(limits.get("steps", 100), "limits: steps")
        handled = {t.on for t in triggers if t.on is not None}
        for signal in signals:
            if signal not in handled:
                self.warnings.append(f"signal {signal!r} has no trigger, so it is always refused")
        if len(signals) > STRICT_TOOL_LIMIT:
            self.warnings.append(
                f"{len(signals)} signals: over {STRICT_TOOL_LIMIT}, tools are sent without strict schemas"
            )
        briefing = self.raw.get("briefing")
        if briefing is not None and not isinstance(briefing, str):
            raise self.fail("briefing: must be a function name")
        return Policy(
            name=name.strip(),
            version=str(self.raw.get("version", "1")),
            description=str(self.raw.get("description", "")).strip(),
            fields=fields,
            states=states,
            initial=str(initial),
            final=final,
            signals=signals,
            triggers=triggers,
            on_error=self.effect(self.raw.get("on_error"), "on_error", fields, states),
            on_stall=self.effect(self.raw.get("on_stall"), "on_stall", fields, states),
            briefing=briefing,
            agent=self.agent(),
            max_steps=max_steps,
            warnings=self.warnings,
        )

    def positive_int(self, value: Any, where: str) -> int:
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise self.fail(f"{where}: must be a whole number, 1 or more, not {value!r}")
        return value

    def fraction(self, value: Any, where: str) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise self.fail(f"{where}: must be a number from 0 to 1, not {value!r}")
        return float(value)

    def mapping(self, value: Any, where: str) -> dict[str, Any]:
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise self.fail(f"{where}: must be a mapping")
        return value

    def typed(self, name: str, spec: Any, where: str) -> tuple[str, tuple[str, ...], str, bool]:
        """`text`, `enum [a, b]`, or `{type, values, description, optional}`."""
        description, optional = "", False
        if isinstance(spec, dict):
            unknown = sorted(str(k) for k in (set(spec) - {"type", "values", "description", "optional"}))
            if unknown:
                raise self.fail(f"{where}: unknown keys {', '.join(unknown)}")
            type_ = str(spec.get("type", ""))
            values_raw = spec.get("values")
            description = str(spec.get("description", "")).strip()
            optional = bool(spec.get("optional", False))
        elif isinstance(spec, str):
            match = re.fullmatch(r"\s*enum\s*\[(.*)\]\s*", spec)
            if match:
                type_, values_raw = "enum", [v.strip() for v in match.group(1).split(",") if v.strip()]
            else:
                type_, values_raw = spec.strip(), None
        else:
            raise self.fail(f"{where}: a type like 'text' or 'enum [a, b]' is required")
        if type_ not in TYPES:
            raise self.fail(f"{where}: unknown type {type_!r} (use {', '.join(TYPES)})")
        values: tuple[str, ...] = ()
        if type_ == "enum":
            if not isinstance(values_raw, list) or not values_raw:
                raise self.fail(f"{where}: an enum needs values")
            values = tuple(str(v) for v in values_raw)
            if UNKNOWN not in values:
                values = (*values, UNKNOWN)
        elif values_raw is not None:
            raise self.fail(f"{where}: only an enum has values")
        return type_, values, description, optional

    def fields(self) -> dict[str, Field]:
        out: dict[str, Field] = {}
        for name, spec in self.mapping(self.raw.get("fields"), "fields").items():
            if not _NAME.match(str(name)) or name in _RESERVED:
                raise self.fail(f"fields: {name!r} is not a usable name (lower_snake_case, not reserved)")
            type_, values, description, _ = self.typed(name, spec, f"fields.{name}")
            out[name] = Field(name, type_, values, description)
        return out

    def states(self) -> tuple[str, ...]:
        states = self.raw.get("states")
        if not isinstance(states, list) or not states:
            raise self.fail("states: a list of states is required")
        seen: list[str] = []
        for state in states:
            if not isinstance(state, str) or not _STATE.match(state):
                raise self.fail(f"states: {state!r} is not a usable state name")
            if state in seen:
                raise self.fail(f"states: {state!r} is listed twice")
            seen.append(state)
        return tuple(seen)

    def final(self, states: tuple[str, ...]) -> dict[str, str]:
        final = self.mapping(self.raw.get("final"), "final")
        if not final:
            raise self.fail("final: map at least one state to its outcome, e.g. DONE: act")
        for state, outcome in final.items():
            if state not in states:
                raise self.fail(f"final: {state!r} is not one of the states")
            if not isinstance(outcome, str) or not outcome.strip():
                raise self.fail(f"final.{state}: the outcome must be a word, e.g. act or escalate")
        return {str(k): str(v).strip() for k, v in final.items()}

    def signals(self) -> dict[str, Signal]:
        out: dict[str, Signal] = {}
        for name, spec in self.mapping(self.raw.get("signals"), "signals").items():
            where = f"signals.{name}"
            if not _NAME.match(str(name)):
                raise self.fail(f"{where}: a signal name is lower_snake_case (it becomes a tool name)")
            spec = self.mapping(spec, where)
            unknown = sorted(str(k) for k in (set(spec) - {"description", "args"}))
            if unknown:
                raise self.fail(f"{where}: unknown keys {', '.join(unknown)}")
            description = str(spec.get("description", "")).strip()
            if not description:
                raise self.fail(f"{where}: a description is required; it is what the agent reads")
            args: list[Arg] = []
            for arg_name, arg_spec in self.mapping(spec.get("args"), f"{where}.args").items():
                if not _NAME.match(str(arg_name)):
                    raise self.fail(f"{where}.args: {arg_name!r} is not a usable name")
                type_, values, arg_description, optional = self.typed(arg_name, arg_spec, f"{where}.args.{arg_name}")
                args.append(Arg(arg_name, type_, values, arg_description, optional))
            out[name] = Signal(name, description, tuple(args))
        return out

    def effect(self, raw: Any, where: str, fields: dict[str, Field], states: tuple[str, ...]) -> Effect | None:
        if raw is None:
            return None
        raw = self.mapping(raw, where)
        unknown = sorted(str(k) for k in (set(raw) - {"set", "to"}))
        if unknown:
            raise self.fail(f"{where}: unknown keys {', '.join(unknown)}")
        return Effect(self.sets(raw.get("set"), where, fields), self.target(raw.get("to"), where, states))

    def sets(self, raw: Any, where: str, fields: dict[str, Field]) -> tuple[tuple[str, Any], ...]:
        out: list[tuple[str, Any]] = []
        for name, value in self.mapping(raw, f"{where}.set").items():
            spec = fields.get(name)
            if spec is None:
                raise self.fail(f"{where}.set: no field {name!r}")
            out.append((name, coerce(spec.type, spec.values, value, f"{self.where}: {where}.set.{name}")))
        return tuple(out)

    def target(self, raw: Any, where: str, states: tuple[str, ...]) -> str | None:
        if raw is None:
            return None
        if raw not in states:
            raise self.fail(f"{where}.to: {raw!r} is not one of the states")
        return str(raw)

    def triggers(
        self, fields: dict[str, Field], states: tuple[str, ...], signals: dict[str, Signal]
    ) -> tuple[Trigger, ...]:
        raw = self.raw.get("triggers")
        if not isinstance(raw, list) or not raw:
            raise self.fail("triggers: a list of rules is required")
        base: dict[str, Kind] = {name: spec.kind for name, spec in fields.items()}
        base["state"] = Kind("enum", states)
        out: list[Trigger] = []
        names: set[str] = set()
        for index, item in enumerate(raw):
            where = f"triggers[{index}]"
            item = self.mapping(item, where)
            name = item.get("name")
            if not isinstance(name, str) or not name.strip():
                raise self.fail(f"{where}: every trigger needs a name")
            where = f"trigger {name!r}"
            if name in names:
                raise self.fail(f"{where}: the name is used twice")
            names.add(name)
            unknown = sorted(str(k) for k in (set(item) - _TRIGGER))
            if unknown:
                raise self.fail(f"{where}: unknown keys {', '.join(unknown)}")
            on = item.get("on")
            scope = dict(base)
            if on is not None:
                if on not in signals:
                    raise self.fail(f"{where}: on: {on!r} is not a signal")
                for arg in signals[on].args:
                    scope[f"args.{arg.name}"] = arg.kind
            when = compile_condition(item.get("when"), scope, f"{self.where}: {where}")
            asks = self.asks(item.get("ask"), where, fields)
            call = item.get("call")
            if asks and call is not None:
                raise self.fail(f"{where}: a trigger either asks the classifier or calls a function, not both")
            if call is not None and not isinstance(call, str):
                raise self.fail(f"{where}: call: must be a function name")
            source = item.get("input")
            if source is not None and not asks:
                raise self.fail(f"{where}: input: only applies to ask:")
            retry = item.get("retry", 0)
            if not isinstance(retry, int) or isinstance(retry, bool) or retry < 0:
                raise self.fail(f"{where}: retry: must be a whole number, 0 or more")
            if retry and not (asks or call):
                raise self.fail(f"{where}: retry: only applies to ask: or call:")
            on_error = self.effect(item.get("on_error"), f"{where}.on_error", fields, states)
            if on_error is not None and not (asks or call):
                raise self.fail(f"{where}: on_error: only applies to ask: or call:")
            effect = Effect(self.sets(item.get("set"), where, fields), self.target(item.get("to"), where, states))
            if on is None and not (asks or call or effect.set or effect.to):
                raise self.fail(f"{where}: a rule with no signal must do something (ask, call, set or to)")
            out.append(
                Trigger(
                    name=name,
                    on=on,
                    when=when,
                    ask=asks,
                    input=None if source is None else str(source),
                    call=call,
                    effect=effect,
                    retry=retry,
                    on_error=on_error,
                    note=str(item.get("note", "")).strip(),
                )
            )
        return tuple(out)

    def asks(self, raw: Any, where: str, fields: dict[str, Field]) -> tuple[Ask, ...]:
        out: list[Ask] = []
        for field_name, spec in self.mapping(raw, f"{where}.ask").items():
            at = f"{where}.ask.{field_name}"
            target = fields.get(field_name)
            if target is None:
                raise self.fail(f"{at}: no field {field_name!r} to write the answer to")
            spec = self.mapping(spec, at)
            kinds = [k for k in ("noul", "choice", "score") if k in spec]
            if len(kinds) != 1:
                raise self.fail(f"{at}: say exactly one of noul:, choice: or score: (the question)")
            kind = kinds[0]
            instructions = str(spec[kind]).strip()
            allowed = {kind, "above", "below", "options", "levels"}
            unknown = sorted(str(k) for k in (set(spec) - allowed))
            if unknown:
                raise self.fail(f"{at}: unknown keys {', '.join(unknown)}")
            if kind == "noul":
                if target.type != "bool":
                    raise self.fail(f"{at}: a noul answers yes/no, so {field_name} must be bool")
                above = self.fraction(spec.get("above", 0.5), f"{at}: above")
                below = self.fraction(spec.get("below", min(above, 0.5)), f"{at}: below")
                if not 0 <= below <= above <= 1:
                    raise self.fail(f"{at}: need 0 <= below <= above <= 1")
                out.append(Ask(field_name, kind, instructions, above=above, below=below))
            elif kind == "choice":
                if target.type != "enum":
                    raise self.fail(f"{at}: a choice picks a value, so {field_name} must be an enum")
                options = self.mapping(spec.get("options"), f"{at}.options")
                if not options:
                    raise self.fail(f"{at}: options: are required")
                for value in options:
                    if value not in target.values:
                        raise self.fail(f"{at}: option {value!r} is not a value of {field_name}")
                out.append(
                    Ask(
                        field_name,
                        kind,
                        instructions,
                        options=tuple((str(k), str(v or "").strip()) for k, v in options.items()),
                    )
                )
            else:
                if target.type != "enum":
                    raise self.fail(f"{at}: a score picks a level, so {field_name} must be an enum")
                levels = spec.get("levels")
                if not isinstance(levels, list) or len(levels) < 2:
                    raise self.fail(f"{at}: levels: needs two or more values, lowest first")
                for value in levels:
                    if value not in target.values:
                        raise self.fail(f"{at}: level {value!r} is not a value of {field_name}")
                out.append(Ask(field_name, kind, instructions, levels=tuple(str(v) for v in levels)))
        return tuple(out)

    def agent(self) -> AgentSettings:
        raw = self.mapping(self.raw.get("agent"), "agent")
        unknown = sorted(str(k) for k in (set(raw) - {"model", "max_turns", "max_tokens", "instructions"}))
        if unknown:
            raise self.fail(f"agent: unknown keys {', '.join(unknown)}")
        defaults = AgentSettings()
        return AgentSettings(
            model=str(raw.get("model", defaults.model)),
            max_turns=self.positive_int(raw.get("max_turns", defaults.max_turns), "agent: max_turns"),
            max_tokens=self.positive_int(raw.get("max_tokens", defaults.max_tokens), "agent: max_tokens"),
            instructions=str(raw.get("instructions", "")).strip(),
        )
