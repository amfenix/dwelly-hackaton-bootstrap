# desk — cases in, decisions out, through a policy

An agent that works operational cases (messages, tickets, requests) to an outcome — act, ask, or
hand to a person — where **every action the agent asks for is accepted or refused by a written
policy**, and every fact a decision rests on comes from the records, not from the agent.

> The `track/` folder holds an example (a letting agency's repairs desk) until it is replaced by
> the event's track. Everything else is the system.

## 1. What we built

- **A policy file** (`track/policy.yaml`): the case's facts, its states, the actions the agent may
  ask for, and the rules — in order, the first true one fires. It is the whole decision logic, in
  a form a person can read and change without touching code.
- **A small engine** (`src/desk/`) that runs one case as a state machine under that policy.
  Rules call **functions** (`track/functions.py`: lookups and actions against the records) or a
  **classifier** (yes/no, choice and score questions about the text, answered in one call).
- **An agent** (Claude) whose tools are built from the policy: one tool per signal. It asks; the
  policy decides; a refusal comes back with the reason, so it can take another way or escalate.
- **A runner** that works a whole batch unattended and writes, for every case,
  `ANSWERS/<n>/ANSWER.md` and `REASONING.md` — built from the run's journal, not written by a
  model — plus a full trace of every model call. The one generated part is a two-to-three sentence
  summary at the top of ANSWER.md, written from that record and labelled as a retelling.

## 2. The operational problem

Operations desks get messages that look routine and are not: the sender is not on record, the
quote is over the owner's limit, the message hides an instruction, the booking system times out.
An agent that is simply told the rules will sometimes break them. Here it cannot: it has no tool
for what the policy does not allow, it cannot write a fact itself, and a step it is refused is
recorded with the reason. When the case is outside the rules, it ends with a person and with
everything that person needs.

## 3. Architecture

```
cases.json ──► runner ──► one case = one state machine (policy.yaml)
                              │
               rules with no signal fire by themselves, in order:
                 call: a function   ──► records, external APIs (track/functions.py)
                 ask:  the classifier ──► Jev (TypeSafe) or Claude, one call for all questions
                              │
               the agent (Claude) sends signals = its tools
                 accepted by a rule ──► the rule runs   ·   refused ──► the reason, back to the agent
                              │
               a final state = the outcome  (act / escalate …)
                              │
               ANSWERS/<n>/ANSWER.md · REASONING.md · traces/<n>.json · results.jsonl
```

- **Facts are written only by rules**: a function's return value, a classifier's answer, or a
  rule's `set:`. The agent's arguments are visible to the rule that handles its signal and never
  become facts by themselves.
- **Only the policy closes a case.** A final state is the outcome the case has *now*; a later rule
  may still move it — book the repair, then hand the refund demand in the same message to a person.
  The rules and the agent bring a case to a final state, or `on_stall` sets one; a policy that
  leaves no way out gets an answer that says the run stopped in a state that is not final.
- **Three-valued conditions**: a fact not known yet makes a condition *unknown*, and a rule fires
  only on *true* — a missing fact never triggers an action by accident.
- **Failures are part of the policy**: a function or classifier that fails is retried with the same
  idempotency key (`retry:`), then the rule's or the policy's `on_error:` decides. An agent that
  stops before the case is closed triggers `on_stall:`. A crash anywhere in a case (its input,
  its rules, the agent) applies `on_error:` and still writes an answer that says where it broke;
  one broken case never stops the batch.
- **The agent's words never become facts or actions unchecked**: a rule compares its arguments
  with the facts (`args.trade = trade`), and text that goes out is written by a function from the
  records — the agent picks *what* to answer, not the words.
- **Every model call is recorded** (request, response, time, tokens, cost). `desk run --replay`
  reproduces a run from the traces with no keys and no network, through the same policy and
  functions — the evidence that the answers were produced by the system. The replay writes to
  `replay/` and never over the traces it reads; a recorded call the replay does not make again
  is reported as a divergence.

## 4. How to run it

Prerequisites: Python 3.11+ and [uv](https://docs.astral.sh/uv/) (or pip).

```bash
uv sync                                   # or: pip install -e .
cp .env.example .env                      # then fill it in; or export the variables instead
#   ANTHROPIC_API_KEY  Claude: the agent, the generator, the Claude classifier
#   TYPESAFE_API_KEY   only for --classifier jev

uv run desk check                         # load the policy and functions, list the agent's tools
uv run desk run                           # every case in track/cases.json → ANSWERS/, traces/, results.jsonl
uv run desk run --classifier jev          # the same with TypeSafe's Jev as the classifier
uv run desk run --workers 4 --deadline 1500   # parallel; cases not reached in time are still answered
uv run desk run --replay                  # reproduce the run in traces/ into replay/, offline
diff -r ANSWERS replay/ANSWERS            # the same answers, from the recorded calls
uv run desk eval evals/triage.yaml --models claude-haiku-5-5,jev-latest --runs 3
```

`desk run` flags:

| Flag | Default | Meaning |
|---|---|---|
| `--policy`, `--functions`, `--world`, `--cases` | `track/…` | the track's files |
| `--out`, `--traces`, `--results` | `ANSWERS`, `traces`, `results.jsonl` | where the run writes (under `replay/` for a replay) |
| `--classifier` | `claude` | `claude` or `jev` |
| `--classifier-model` | `claude-haiku-5-5` / `jev-latest` | |
| `--generator-model` | `claude-sonnet-5-5` | the summary, and `ctx.generate` in functions |
| `--agent-model` | the policy's `agent.model` | |
| `--no-agent` | off | rules only; the agent is never asked |
| `--no-summary` | off | no generated summary at the top of ANSWER.md |
| `--workers N` | 1 | cases worked in parallel |
| `--deadline S` | none | after S seconds, cases not started are answered as not worked |
| `--no-resume` | off | re-work cases that already have an answer |
| `--replay [TRACES]` | off | answer every model call from these traces (default `traces`) |
| `desk --env FILE …` | `.env` | the key file read on start; a variable already set wins |

Tests and checks (no keys needed, no network):

```bash
uv run pytest            # 100% line and branch coverage is enforced
uv run ruff check src tests track && uv run ruff format --check src tests track && uv run mypy
```

## 5. Models, APIs and external services

| What | Used for | Configured by |
|---|---|---|
| Anthropic Claude API — `claude-sonnet-5-5` | the agent | `ANTHROPIC_API_KEY`; `agent.model` in the policy |
| Anthropic Claude API — `claude-haiku-5-5` | the classifier (default backend), evals | `--classifier claude` |
| Anthropic Claude API — `claude-sonnet-5-5` | the generator: the summary in ANSWER.md, `ctx.generate` | `--generator-model` |
| TypeSafe System One — Jev (`jev-latest`) | the classifier (alternative backend) | `TYPESAFE_API_KEY`; `--classifier jev` |
| Python packages | `anthropic`, `typesafe-sdk`, `pyyaml` | `pyproject.toml` |

No other service is called. Prices used for the cost figures are in `src/desk/calls.py`.

## 6. Assumptions

- A case is independent of the others; one case is one run of the machine.
- The records a decision needs (`world.json`) are available to the functions; anything not in them
  is treated as unknown, and the policy decides what unknown means.
- The classifier's yes/no answers are probabilities; each question sets its own thresholds, and an
  answer between them is unknown rather than a guess.

## 7. Known limitations

- The policy language is deliberately small: `and`/`or`/`not`, comparisons, `in`, `present`/
  `absent`. No arithmetic in conditions — a sum or a count across records is a function's job.
- Rules are checked when the policy loads (names, types, values), but there is no proof that every
  combination of facts is covered; an uncovered case ends in `on_stall`.
- One machine per case: a message with two requests is worked as one case with two signals.
- A function with side effects must dedupe on `ctx.key` itself. The example keeps its keys in
  memory; a real integration passes the key to the external system, which dedupes across restarts.
- `--replay` reproduces runs only when functions are deterministic given the case and the records.
- The records (`world`) are shared by every case of a run; functions read them and must not
  change them.
- Traces keep the cases, prompts and responses verbatim (they are the evidence asked for); they
  are not redacted. A failure is recorded as its phase, the last rule and the error, without a
  stack trace.
- A case's files are written under one lock, ANSWER.md last; a run killed mid-write can leave a
  REASONING.md without its ANSWER.md, which a resumed run works again.

## 8. What we would build next

- A proof step: show at load time that every state has a way out and no two rules overlap.
- Per-case cost and time budgets enforced by the runner.
- A reviewer view over `traces/` for a person picking up escalated cases.

## Repository layout

```
src/desk/        the engine: policy loader, conditions, state machine, agent, classifier, runner, evals
track/           the track: policy.yaml, functions.py, world.json, cases.json
evals/           evals on real models: classifier wording, the agent's first tool, generated text
tests/           unit and end-to-end tests on fakes (no network)
docs/policy.md   how to write the policy and the functions
```

## Working on the desk

The track is a policy, its functions, the records and the cases; the engine rarely needs to
change. `docs/policy.md` is the reference for both files.

| To … | change | and check with |
|---|---|---|
| add a fact | `fields:` in `track/policy.yaml` | `desk check` |
| let the agent do something new | a `signals:` entry and a rule with `on:` that accepts it | `desk check` lists the new tool |
| look something up or act in the world | a `@function` in `track/functions.py`, named by a rule's `call:` | a test in `tests/test_track.py` |
| ask a question about the text | a rule's `ask:` | an eval in `evals/` to tune the wording |
| change what the agent reads first | the `briefing:` function | `desk eval evals/agent.yaml` |

The loop while changing it:

```bash
uv run desk check                               # the policy and functions load; prints every problem
uv run pytest                                   # no keys, no network
uv run desk run --no-resume --out runs/try/ANSWERS --traces runs/try/traces --results runs/try/results.jsonl
                                                # a real run into runs/ (ignored by git); read REASONING.md
```

Engine code lives in `src/desk/`: `policy.py` loads and checks the YAML, `expr.py` the conditions,
`machine.py` runs one case, `agent.py` builds the tools and talks to Claude, `classifier.py` and
`generator.py` are the model calls, `calls.py` records them, `runner.py` works a batch,
`result.py`/`answers.py`/`summary.py` turn a run into its files. Every change keeps
`uv run pytest` at 100% coverage and ruff and mypy clean.

## Licence

Apache License 2.0 — see `LICENSE`.
