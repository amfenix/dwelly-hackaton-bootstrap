"""Conditions: the `when:` language of a policy.

    state = TRIAGED and quote > limit
    category in [repair, leak] and not is_emergency
    tenant_id absent or args.contractor present
    quote <= 250.00 and note = "needs a visit"

Parsed with Python's `ast` from a small, fixed set of nodes; nothing is ever `eval`-ed. A bare
name is a field, `state`, or `args.<name>` of the signal being handled; any other bare name is an
enum value, and must be one of the values of what it is compared with.

Logic is three-valued. A field that is not set yet is *unknown*, and so is any comparison that
reads it: `unknown and x` is unknown, `unknown or true` is true, `not unknown` is unknown. A
trigger fires only on *true*, so an unknown fact never fires a rule by accident.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Any

from desk.errors import PolicyError

ORDERED = {"number", "money", "date"}
NUMERIC = {"number", "money"}

_OPS: dict[type[ast.cmpop], str] = {
    ast.Eq: "=",
    ast.NotEq: "!=",
    ast.Lt: "<",
    ast.LtE: "<=",
    ast.Gt: ">",
    ast.GtE: ">=",
    ast.In: "in",
    ast.NotIn: "not in",
}


@dataclass(frozen=True)
class Kind:
    """What a name holds: its type, and for an enum its values."""

    type: str
    values: tuple[str, ...] = ()


Env = Callable[[str], Any]
"""Looks a reference up: `state`, a field name, or `args.<name>`. `None` means not set."""


@dataclass(frozen=True)
class Atom:
    """One comparison or one test, as a reader would write it, and what it found."""

    text: str
    result: bool | None
    values: tuple[tuple[str, Any], ...]


class _Node:
    def eval(self, env: Env) -> Any:  # pragma: no cover - every subclass overrides it
        raise NotImplementedError

    def atoms(self, env: Env, out: list[Atom]) -> None:
        out.append(Atom(self.text(), _truth(self.eval(env)), ()))

    def text(self) -> str:  # pragma: no cover - every subclass overrides it
        raise NotImplementedError


class _Ref(_Node):
    def __init__(self, name: str, kind: Kind) -> None:
        self.name, self.kind = name, kind

    def eval(self, env: Env) -> Any:
        return env(self.name)

    def atoms(self, env: Env, out: list[Atom]) -> None:
        value = env(self.name)
        out.append(Atom(self.name, _truth(value), ((self.name, value),)))

    def text(self) -> str:
        return self.name


class _Const(_Node):
    def __init__(self, value: Any, shown: str) -> None:
        self.value, self.shown = value, shown

    def eval(self, env: Env) -> Any:
        return self.value

    def text(self) -> str:
        return self.shown


class _Compare(_Node):
    def __init__(self, left: _Node, op: str, right: _Node | list[_Node]) -> None:
        self.left, self.op, self.right = left, op, right

    def eval(self, env: Env) -> bool | None:
        left = self.left.eval(env)
        if left is None:
            return None
        if isinstance(self.right, list):
            items = [item.eval(env) for item in self.right]
            if left in items:
                found: bool | None = True
            elif None in items:
                found = None  # no match among the known items; an unknown one might still match
            else:
                found = False
            return found if self.op == "in" or found is None else not found
        right = self.right.eval(env)
        if right is None:
            return None
        if self.op == "=":
            return bool(left == right)
        if self.op == "!=":
            return bool(left != right)
        if self.op == "<":
            return bool(left < right)
        if self.op == "<=":
            return bool(left <= right)
        if self.op == ">":
            return bool(left > right)
        return bool(left >= right)

    def atoms(self, env: Env, out: list[Atom]) -> None:
        refs: list[tuple[str, Any]] = []
        for side in [self.left, *(self.right if isinstance(self.right, list) else [self.right])]:
            if isinstance(side, _Ref):
                refs.append((side.name, env(side.name)))
        out.append(Atom(self.text(), self.eval(env), tuple(refs)))

    def text(self) -> str:
        if isinstance(self.right, list):
            right = "[" + ", ".join(item.text() for item in self.right) + "]"
        else:
            right = self.right.text()
        return f"{self.left.text()} {self.op} {right}"


class _Present(_Node):
    def __init__(self, ref: _Ref, present: bool) -> None:
        self.ref, self.present = ref, present

    def eval(self, env: Env) -> bool:
        is_set = self.ref.eval(env) is not None
        return is_set if self.present else not is_set

    def atoms(self, env: Env, out: list[Atom]) -> None:
        out.append(Atom(self.text(), self.eval(env), ((self.ref.name, env(self.ref.name)),)))

    def text(self) -> str:
        return f"{self.ref.name} {'present' if self.present else 'absent'}"


class _Not(_Node):
    def __init__(self, inner: _Node) -> None:
        self.inner = inner

    def eval(self, env: Env) -> bool | None:
        value = _truth(self.inner.eval(env))
        return None if value is None else not value

    def atoms(self, env: Env, out: list[Atom]) -> None:
        inner: list[Atom] = []
        self.inner.atoms(env, inner)
        out.append(Atom(self.text(), self.eval(env), tuple(v for a in inner for v in a.values)))

    def text(self) -> str:
        inner = self.inner.text()
        return f"not {inner}" if isinstance(self.inner, (_Ref, _Present)) else f"not ({inner})"


class _Bool(_Node):
    def __init__(self, op: str, parts: list[_Node]) -> None:
        self.op, self.parts = op, parts

    def eval(self, env: Env) -> bool | None:
        values = [_truth(part.eval(env)) for part in self.parts]
        if self.op == "and":
            if False in values:
                return False
            return None if None in values else True
        if True in values:
            return True
        return None if None in values else False

    def atoms(self, env: Env, out: list[Atom]) -> None:
        for part in self.parts:
            part.atoms(env, out)

    def text(self) -> str:
        return f" {self.op} ".join(
            f"({part.text()})" if isinstance(part, _Bool) else part.text() for part in self.parts
        )


def _truth(value: Any) -> bool | None:
    if value is None:
        return None
    return bool(value)


class Condition:
    """A compiled `when:`. `holds(env)` is true, false or `None` (unknown)."""

    def __init__(self, source: str, root: _Node | None) -> None:
        self.source = source
        self._root = root

    def holds(self, env: Env) -> bool | None:
        return True if self._root is None else _truth(self._root.eval(env))

    def explain(self, env: Env) -> list[Atom]:
        """Every comparison in the condition with what it found, for the journal and refusals."""
        out: list[Atom] = []
        if self._root is not None:
            self._root.atoms(env, out)
        return out


_POSTFIX = re.compile(r"([A-Za-z_][\w.]*)\s+(present|absent)\b")
_SINGLE_EQ = re.compile(r"(?<![<>!=])=(?!=)")
_STRING = re.compile(r"\"[^\"]*\"|'[^']*'")


def _prepare(source: str) -> str:
    """Rewrite `=` to `==` and `x present` to a call, leaving quoted strings alone."""
    pieces: list[str] = []
    last = 0
    for match in _STRING.finditer(source):
        pieces.append(_rewrite(source[last : match.start()]))
        pieces.append(match.group(0))
        last = match.end()
    pieces.append(_rewrite(source[last:]))
    return "".join(pieces)


def _rewrite(text: str) -> str:
    text = _POSTFIX.sub(lambda m: f"__{m.group(2)}__({m.group(1)})", text)
    return _SINGLE_EQ.sub("==", text)


def compile_condition(
    source: str | None,
    names: Mapping[str, Kind],
    where: str,
) -> Condition:
    """Compile a `when:` against the names it may read. Every mistake is a `PolicyError` here, at load."""
    if source is None or str(source).strip() == "":
        return Condition("", None)
    text = str(source)
    prepared = _prepare(text)
    try:
        tree = ast.parse(prepared, mode="eval")
    except SyntaxError as error:
        raise PolicyError(f"{where}: cannot read the condition {text!r}: {error.msg}") from None
    return Condition(text, _Compiler(names, where, text, prepared).node(tree.body, boolean=True))


class _Compiler:
    def __init__(self, names: Mapping[str, Kind], where: str, source: str, prepared: str) -> None:
        self.names, self.where, self.source, self.prepared = names, where, source, prepared

    def fail(self, message: str) -> PolicyError:
        return PolicyError(f"{self.where}: {message} in {self.source!r}")

    def node(self, tree: ast.expr, *, boolean: bool) -> _Node:
        if isinstance(tree, ast.BoolOp):
            op = "and" if isinstance(tree.op, ast.And) else "or"
            return _Bool(op, [self.node(value, boolean=True) for value in tree.values])
        if isinstance(tree, ast.UnaryOp) and isinstance(tree.op, ast.Not):
            return _Not(self.node(tree.operand, boolean=True))
        if isinstance(tree, ast.Compare):
            return self.compare(tree)
        if isinstance(tree, ast.Call):
            return self.present(tree)
        if isinstance(tree, ast.Name) and tree.id in ("true", "false"):
            return _Const(tree.id == "true", tree.id)
        if isinstance(tree, (ast.Name, ast.Attribute)):
            ref = self.ref(tree)
            if ref is None:
                raise self.fail(f"unknown name {self.name_of(tree)!r}")
            if boolean and ref.kind.type != "bool":
                raise self.fail(f"{ref.name} is {ref.kind.type}, not bool; compare it, or test it with 'present'")
            return ref
        raise self.fail(f"unsupported expression {ast.get_source_segment(self.prepared, tree)!r}")

    def name_of(self, tree: ast.expr) -> str:
        if isinstance(tree, ast.Attribute) and isinstance(tree.value, ast.Name):
            return f"{tree.value.id}.{tree.attr}"
        if isinstance(tree, ast.Name):
            return tree.id
        return ast.get_source_segment(self.prepared, tree) or "?"

    def ref(self, tree: ast.expr) -> _Ref | None:
        name = self.name_of(tree)
        if isinstance(tree, ast.Name) and name in ("true", "false"):
            return None
        if isinstance(tree, ast.Attribute) and not (isinstance(tree.value, ast.Name) and tree.value.id == "args"):
            raise self.fail(f"only args.<name> may use a dot, not {name!r}")
        kind = self.names.get(name)
        return None if kind is None else _Ref(name, kind)

    def present(self, tree: ast.Call) -> _Node:
        func = tree.func
        if not (isinstance(func, ast.Name) and func.id in ("__present__", "__absent__")) or len(tree.args) != 1:
            raise self.fail("calls are not part of the language")
        ref = self.ref(tree.args[0])
        if ref is None:
            raise self.fail(f"unknown name {self.name_of(tree.args[0])!r}")
        return _Present(ref, func.id == "__present__")

    def compare(self, tree: ast.Compare) -> _Node:
        if len(tree.ops) != 1:
            raise self.fail("write one comparison at a time, joined with 'and'")
        op = _OPS.get(type(tree.ops[0]))
        if op is None:
            raise self.fail("unsupported comparison")
        left = self.operand(tree.left, None)
        if not isinstance(left, _Ref):
            raise self.fail("the left side of a comparison must be a field, state or args.<name>")
        right_tree = tree.comparators[0]
        if op in ("in", "not in"):
            if not isinstance(right_tree, (ast.List, ast.Tuple)) or not right_tree.elts:
                raise self.fail(f"'{op}' needs a list like [A, B]")
            items = [self.operand(item, left.kind) for item in right_tree.elts]
            for item in items:
                self.check_pair(left, op, item)
            return _Compare(left, op, items)
        right = self.operand(right_tree, left.kind)
        self.check_pair(left, op, right)
        return _Compare(left, op, right)

    def operand(self, tree: ast.expr, against: Kind | None) -> _Node:
        if isinstance(tree, (ast.Name, ast.Attribute)):
            ref = self.ref(tree)
            if ref is not None:
                return ref
            name = self.name_of(tree)
            if name in ("true", "false"):
                return _Const(name == "true", name)
            if against is not None and against.type == "enum" and isinstance(tree, ast.Name):
                if name not in against.values:
                    raise self.fail(f"{name!r} is not one of {', '.join(against.values)}")
                return _Const(name, name)
            raise self.fail(f"unknown name {name!r}")
        if isinstance(tree, ast.Constant) and not isinstance(tree.value, bool):
            if isinstance(tree.value, str):
                return _Const(tree.value, f'"{tree.value}"')
            if isinstance(tree.value, (int, float)):
                shown = ast.get_source_segment(self.prepared, tree) or str(tree.value)
                return _Const(Decimal(shown), shown)
        if isinstance(tree, ast.UnaryOp) and isinstance(tree.op, ast.USub):
            inner = self.operand(tree.operand, against)
            if isinstance(inner, _Const) and isinstance(inner.value, Decimal):
                return _Const(-inner.value, f"-{inner.shown}")
        raise self.fail(f"unsupported value {ast.get_source_segment(self.prepared, tree)!r}")

    def check_pair(self, left: _Ref, op: str, right: _Node) -> None:
        kind = left.kind
        if op in ("<", "<=", ">", ">=") and kind.type not in ORDERED:
            raise self.fail(f"{left.name} is {kind.type}; only number, money and date can be ordered")
        if isinstance(right, _Ref):
            other = right.kind
            same = other.type == kind.type or {other.type, kind.type} <= NUMERIC
            if not same:
                raise self.fail(f"{left.name} ({kind.type}) cannot be compared with {right.name} ({other.type})")
            return
        assert isinstance(right, _Const)
        value = right.value
        if kind.type in NUMERIC and not isinstance(value, Decimal):
            raise self.fail(f"{left.name} is {kind.type}; compare it with a number")
        if kind.type == "bool" and not isinstance(value, bool):
            raise self.fail(f"{left.name} is bool; compare it with true or false")
        if kind.type in ("text", "date", "enum") and not isinstance(value, str):
            raise self.fail(f"{left.name} is {kind.type}; compare it with a word or a quoted string")
        if kind.type == "enum" and value not in kind.values:
            raise self.fail(f"{value!r} is not one of {', '.join(kind.values)}")
        if kind.type == "date":
            try:
                date.fromisoformat(value)
            except ValueError:
                raise self.fail(f"{left.name} is a date; {value!r} is not a date like 2026-10-10") from None


def to_decimal(value: Any) -> Decimal:
    """A number from the outside world (YAML, JSON, a function, the agent) as a `Decimal`, never a float."""
    if isinstance(value, bool):
        raise ValueError("a yes/no is not a number")
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise ValueError(f"{value!r} is not a finite number")
        return value
    try:
        number = Decimal(str(value).strip().replace(",", ""))
    except InvalidOperation:
        raise ValueError(f"{value!r} is not a number") from None
    if not number.is_finite():
        raise ValueError(f"{value!r} is not a finite number")
    return number
