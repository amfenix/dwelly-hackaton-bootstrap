"""The docs stay true: the guide's example policy loads, and every flag the README names exists."""

from __future__ import annotations

import re
from pathlib import Path

from desk.policy import parse_policy

ROOT = Path(__file__).resolve().parents[1]


def test_the_guide_example_policy_loads() -> None:
    guide = (ROOT / "docs" / "policy.md").read_text(encoding="utf-8")
    example = re.search(r"```yaml\n(policy:.*?)```", guide, re.DOTALL)
    assert example is not None
    policy = parse_policy(example.group(1), "docs/policy.md")
    assert policy.name == "repairs-desk" and set(policy.signals) == {"book_repair"}
    assert policy.agent.instructions == "Be brief." and policy.max_steps == 100


def test_every_flag_in_the_readme_exists() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    cli = (ROOT / "src" / "desk" / "cli.py").read_text(encoding="utf-8")
    named = set(re.findall(r"`(--[a-z-]+)", readme))
    assert named
    for flag in named:
        assert f'"{flag}"' in cli, f"README names {flag}, which desk does not have"


def test_every_engine_module_in_the_readme_exists() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for module in re.findall(r"`(\w+\.py)`", readme):
        assert (ROOT / "src" / "desk" / module).exists() or (ROOT / "track" / module).exists(), module
