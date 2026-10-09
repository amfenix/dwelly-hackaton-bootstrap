"""The agent's tools come from the policy; its loop sends each tool call to the case."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from desk.agent import STATUS_TOOL, Agent, tools_for
from desk.calls import Recorder
from desk.machine import Case, Services, load_functions
from desk.policy import Policy, load_policy, parse_policy, read_yaml
from tests.conftest import MINI, FakeAnthropic, FakeClassifier, text_reply, tool_reply
from tests.test_machine import ANSWERS, functions

TRACK = Path(__file__).resolve().parents[1] / "track"


def test_one_tool_per_signal_then_case_status(mini: Policy) -> None:
    tools = tools_for(mini)
    assert [t["name"] for t in tools] == ["pay", "answer", "ping", STATUS_TOOL]
    pay = tools[0]
    assert pay["description"] == "Pay an amount." and pay["strict"] is True
    schema = pay["input_schema"]
    assert schema["required"] == ["amount", "memo"] and schema["additionalProperties"] is False
    assert schema["properties"]["amount"] == {"type": "string", "description": "how much a number, e.g. 120.50"}
    assert schema["properties"]["memo"] == {"anyOf": [{"type": "string"}, {"type": "null"}]}
    answer = tools[1]["input_schema"]["properties"]
    assert answer["kind"] == {"type": "string", "enum": ["short", "long"]}
    assert tools[2]["input_schema"]["properties"] == {}
    assert "strict" not in tools[3]


def test_every_type_has_a_schema() -> None:
    policy = parse_policy(
        """
policy: types
fields: {}
states: [A, B]
initial: A
final: { B: act }
signals:
  all:
    description: every type
    args:
      flag: bool
      when: { type: date, description: the day }
      size: number
      later: { type: date, optional: true, description: maybe }
triggers:
  - { name: go, on: all, to: B }
""",
        "p",
    )
    props = tools_for(policy)[0]["input_schema"]["properties"]
    assert props["flag"] == {"type": "boolean"}
    assert props["when"] == {"type": "string", "format": "date", "description": "the day"}
    assert props["size"]["type"] == "string"
    assert props["later"] == {
        "anyOf": [{"type": "string", "format": "date", "description": "maybe"}, {"type": "null"}],
        "description": "maybe",
    }


def test_over_twenty_signals_drop_strict(mini: Policy) -> None:
    raw = read_yaml(MINI)
    raw["signals"].update({f"s{i}": {"description": "x"} for i in range(21)})
    assert all("strict" not in t for t in tools_for(parse_policy(yaml.safe_dump(raw), "p")))


def test_tools_follow_whatever_the_policy_declares() -> None:
    policy = load_policy(TRACK / "policy.yaml")
    tools = tools_for(policy)
    assert [t["name"] for t in tools] == [*policy.signals, STATUS_TOOL]
    for tool, signal in zip(tools, policy.signals.values(), strict=False):
        assert tool["description"] == signal.description
        assert list(tool["input_schema"]["properties"]) == [a.name for a in signal.args]


def started(policy: Policy, **answers: object) -> Case:
    case = Case(
        policy, functions(), {"id": "a1", "text": "x"}, services=Services(FakeClassifier({**ANSWERS, **answers}))
    )
    case.start()
    return case


def test_a_tool_call_goes_to_the_case_and_the_result_comes_back(mini: Policy) -> None:
    client = FakeAnthropic(tool_reply(("pay", {"amount": "40", "memo": None}), text="Paying."), text_reply("Done."))
    case = started(mini)
    run = Agent(mini, Recorder(), client).run(case, "brief")
    assert case.final and run.stopped == "no tool call (end_turn)" and run.turns == 2
    assert run.text == ["Paying.", "Done."]
    assert run.tool_calls[0]["tool"] == "pay" and run.tool_calls[0]["result"]["accepted"] is True
    assert run.tool_calls[0]["result"]["case_closed"] == "act"
    request = client.requests[0]
    assert request["model"] == "claude-haiku-5-5" and request["max_tokens"] == 1000
    assert request["tools"] == tools_for(mini) and request["messages"][0] == {"role": "user", "content": "brief"}
    assert "A small desk for tests." in request["system"] and request["system"].endswith("Be brief.")


def test_a_refusal_is_returned_and_the_agent_tries_again(mini: Policy) -> None:
    client = FakeAnthropic(
        tool_reply(("answer", {"text": "hi", "kind": "short"}), (STATUS_TOOL, {})),
        tool_reply(("pay", {"amount": "10"})),
        text_reply("Paid."),
    )
    case = started(mini)
    run = Agent(mini, Recorder(), client, model="claude-opus-5-5").run(case, "brief")
    assert case.final and run.turns == 3
    second = client.requests[1]
    assert second["model"] == "claude-opus-5-5"
    results = second["messages"][2]["content"]
    refused = json.loads(results[0]["content"])
    assert refused["accepted"] is False and "topic = question is false" in refused["refused_because"]
    status = json.loads(results[1]["content"])
    assert status["state"] == "WAIT" and results[1]["tool_use_id"] == "tu1"


def test_until_signal_stops_after_the_first_signal_not_after_a_status_read(mini: Policy) -> None:
    client = FakeAnthropic(
        tool_reply((STATUS_TOOL, {})),
        tool_reply(("answer", {"text": "hi", "kind": "short"})),
        tool_reply(("pay", {"amount": "10"})),
    )
    run = Agent(mini, Recorder(), client).run(started(mini), "brief", until_signal=True)
    assert run.stopped == "signal sent" and run.turns == 2
    assert [c["tool"] for c in run.tool_calls] == [STATUS_TOOL, "answer"]


def test_no_tool_call_ends_the_run(mini: Policy) -> None:
    case = started(mini)
    run = Agent(mini, Recorder(), FakeAnthropic(text_reply("I am not sure."))).run(case, "brief")
    assert run.stopped == "no tool call (end_turn)" and run.text == ["I am not sure."] and not case.final


def test_the_turn_limit_ends_the_run(mini: Policy) -> None:
    replies = [tool_reply((STATUS_TOOL, {})) for _ in range(3)]
    run = Agent(mini, Recorder(), FakeAnthropic(*replies)).run(started(mini), "brief", max_turns=2)
    assert run.stopped == "turn limit (2)" and run.turns == 2


def test_the_agent_still_works_a_case_the_rules_closed(mini: Policy) -> None:
    case = Case(mini, functions(), {"id": "a", "known": False}, services=Services(FakeClassifier(ANSWERS)))
    case.start()
    assert case.final
    run = Agent(mini, Recorder(), FakeAnthropic(text_reply("Nothing to add."))).run(case, "brief")
    assert run.stopped == "no tool call (end_turn)" and run.turns == 1


def test_system_prompt_without_description_or_instructions() -> None:
    policy = parse_policy(
        """
policy: bare
fields: {}
states: [A, B]
initial: A
final: { B: act }
signals: { go: { description: go } }
triggers: [{ name: go, on: go, to: B }]
""",
        "p",
    )
    assert "\n\nAbout this desk" not in Agent(policy, Recorder(), FakeAnthropic()).system()


def test_the_example_track_loads_with_its_functions() -> None:
    policy = load_policy(TRACK / "policy.yaml")
    found, _ = load_functions(TRACK / "functions.py")
    policy.check_functions(found)
    assert policy.warnings == []
