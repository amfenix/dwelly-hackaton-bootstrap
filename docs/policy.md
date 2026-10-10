# Writing the policy and the functions

The policy (`track/policy.yaml`) is the decision logic. The functions (`track/functions.py`) are
the hands: they look things up and do things. The agent only asks.

After every change: `uv run desk check` (loads both, says exactly what is wrong), then
`uv run pytest`. The example below is complete: it loads as it is (a test checks that).

## The policy file

```yaml
policy: repairs-desk            # a name
version: 1
description: >                  # the agent reads this
  What this desk does.

fields:                         # facts about the case; only rules write them
  tenant:       text
  limit:        money           # money and number are exact decimals, never floats
  quote:        money
  is_emergency: bool
  category:     enum [repair, question, payment]   # `unknown` is always added
  trade:        enum [plumbing, electrical]
  due:          date            # 2026-10-10
  reason:       { type: enum, values: [OVER_LIMIT, ACTION_FAILED, NOT_HANDLED], description: why it was escalated }

states: [NEW, TRIAGED, DONE, ESCALATED]
initial: NEW
final:                          # a final state is the case's answer
  DONE: act
  ESCALATED: escalate

signals:                        # the agent's tools: one tool per signal
  book_repair:
    description: What it does, when to use it, when not to. The agent reads every word.
    args:                       # what the agent passes; never a fact by itself
      trade: { type: enum, values: [plumbing, electrical], description: the kind of work }
      note:  { type: text, optional: true }

triggers:                       # rules, in order; the first true one fires
  - name: identify
    when: "state = NEW"
    call: find_sender           # a function in functions.py; what it returns becomes fields
    to: TRIAGED

  - name: over-limit
    when: "state = TRIAGED and quote > limit"
    note: Over the landlord's limit, so a person decides.   # a line in REASONING.md
    set: { reason: OVER_LIMIT }
    to: ESCALATED

  - name: book
    on: book_repair             # answers the agent's signal; args.trade is readable here
    # check what the agent says against the facts: a disagreement is refused, with both values
    when: "state = TRIAGED and args.trade = trade and quote <= limit"
    call: book_contractor
    retry: 2                    # same ctx.key on every attempt
    on_error: { set: { reason: ACTION_FAILED }, to: ESCALATED }
    to: DONE

on_error: { set: { reason: ACTION_FAILED }, to: ESCALATED }   # any failure without its own on_error
on_stall: { set: { reason: NOT_HANDLED }, to: ESCALATED }     # the agent stopped before a final state
briefing: brief                 # optional: a function that writes what the agent reads first
agent:                          # all optional
  model: claude-sonnet-5-5
  max_turns: 8                  # the agent's turns per case
  max_tokens: 2000              # per turn
  instructions: Be brief.       # added to the agent's system prompt
limits: { steps: 100 }          # rule firings per case before it counts as a loop
```

### How rules run

1. The case starts in `initial`. Rules **without** `on:` are checked top to bottom; the first whose
   `when:` is true fires. Then again from the top, until none fires.
2. A rule without `on:` fires **once per visit** to a state, so a rule that does not move the case
   cannot loop. Rules that bounce between states forever are stopped (`limits: { steps: 100 }`).
3. The agent sends a signal. The rules with that `on:` are tried in order; the first true one
   fires. None true → the signal is refused, and the agent is told which test failed and why.
4. A firing rule does, in this order: `ask:` or `call:` (one of them), then `set:`, then `to:`.
5. A signal returns only when the rules have settled: its rule, its function, and every rule that
   fires after it. The agent's tool result is the case as it is then — there is nothing to poll.
6. **Only the policy closes a case.** Reaching a `final:` state does not freeze it: the agent is
   still asked, and a rule may move the case on (`when: "state != ESCALATED"` lets a booked case
   still be handed to a person). Whatever state the case is in when the agent stops is its answer;
   if that state is not final, `on_stall:` sets one. Write conditions with that in mind.
7. When something breaks outside a rule's `call:` or `ask:` (bad input, a bug, the agent's model
   down), the policy's `on_error:` still decides where the case ends, no rule runs after it, and
   the answer says where it broke. If the case was already in a final state, that state stands:
   what was done was done, and the failure is recorded beside it. Without `on_error:` the answer
   says the run stopped in a state that is not final.

### Conditions

```
state = TRIAGED                     state != DONE
quote <= limit                      quote > 250.00          due < "2026-11-01"
category in [repair, question]      category not in [payment]
tenant present                      tenant absent           args.note present
is_emergency                        not is_emergency
a and b         a or b         not (a or b)
```

A fact not known yet is *unknown*; so is any test that reads it. `unknown and x` is unknown,
`unknown or true` is true, `not unknown` is unknown. In a list, `x in [a, other]` is true on a
match and unknown, not false, when `other` is not known yet. **A rule fires only when its condition
is true** — so to act on a missing fact, test it with `absent`.

### Asking the classifier (`ask:`)

One call answers every question of the rule, and the answers become fields:

```yaml
  - name: triage
    when: "state = IDENTIFIED"
    ask:
      is_emergency:              # a bool field
        noul: "Is anyone in immediate danger?"
        above: 0.7               # ≥ 0.7 → true
        below: 0.3               # ≤ 0.3 → false; in between → unknown
      category:                  # an enum field; `unknown` is always an option
        choice: "What is the message about?"
        options: { repair: something broken, question: a question about the tenancy }
      urgency:                   # an enum field, levels lowest first
        score: "How urgent is it?"
        levels: [low, medium, high]
    input: "{case.text}"         # optional: what the classifier reads (default: the case text)
    to: TRIAGED
```

Tune the wording with an eval (`evals/triage.yaml`): it runs the same questions on several models,
several times, and reports accuracy, stability, time and cost.

## The functions

```python
from desk import Context, function

@function
def find_sender(ctx: Context) -> dict:
    tenancy = lookup(ctx.world, ctx.case["from"])
    if tenancy is None:
        ctx.note("sender not on any tenancy")      # goes into REASONING.md
        return {}                                  # nothing found: tenant stays unknown
    return {"tenant": tenancy["id"], "limit": tenancy["limit"]}
```

| `ctx.` | type | |
|---|---|---|
| `case` | mapping, read only | the case as it came in (after `adapt_case`) |
| `world` | any | the records (`--world`, e.g. world.json); shared, do not change it |
| `fields` | mapping, read only | the facts known so far; a fact not known yet is `None` |
| `args` | mapping, read only | the agent's arguments, in a rule with `on:`; empty otherwise |
| `state` | str | the current state |
| `key` | str | one per firing of the rule, the same on its retries: pass it to anything that writes |
| `attempt` | int | 1 on the first try, 2 on the first retry, … |
| `act(type, **args)` | → None | record an action taken in the world; it becomes an action in ANSWER.md |
| `note(text)` | → None | a line in REASONING.md |
| `classify(text, questions)` | → `{name: Answer}` | ask the classifier from code (below) |
| `generate(system, prompt, model=, max_tokens=)` | → str | draft text with Claude (`--generator-model`) |

Asking the classifier from code, when a rule's `ask:` is not enough:

```python
from desk.classifier import Choice, Noul, Score

answers = ctx.classify(ctx.case["text"], {
    "threat": Noul("Does the message threaten anyone?"),
    "topic": Choice("What is it about?", {"rent": "rent or payments", "repair": "something broken"}),
    "urgency": Score("How urgent is it?", ("low", "medium", "high")),
})
answers["threat"].value     # a probability 0..1 for a Noul, an option for a Choice, a level for a Score
answers["topic"].confidence # may be None
```

`classify` and `generate` raise when the model fails; the rule's `retry:` and `on_error:` handle it,
as for any other error.

Rules for a function:

- **Return the fields you found** — only fields the policy declares. Return `{}` or `None` for
  "found nothing". Do not decide: return the fact, let the policy decide.
- **Raise** when the lookup or the action could not be done (timeout, error). The rule's `retry:`
  and `on_error:` handle it. "Not found" is not an error: return nothing.
- **Anything that writes** (a booking, a payment, a message) must dedupe on `ctx.key`, so a retry
  after a lost answer does not do it twice. Pass the key to the external system when it takes one.
- **Do not send the agent's words.** Take what the agent chose from `ctx.args` (a topic, a trade)
  and build what goes out from `ctx.world` and `ctx.fields`; the example's `send_reply` does this.
- **Do not change `ctx.world`.** It is shared by every case of a run.
- Functions are plain synchronous Python. A call that can hang (an HTTP request, a model) needs its own timeout.

`adapt_case(raw) -> dict`, if defined in functions.py, reshapes each incoming case before it runs.

The `briefing:` function gets `ctx.case`, `ctx.world`, `ctx.fields` and `ctx.state` (read only)
and returns the text the agent reads first. Put the case text inside `<case>…</case>`: the agent
is told that text is data, not instructions.

## Checking a change

- `tests/test_track.py` runs the track's rules and functions with a scripted classifier: add a
  test there for every rule you add (a case that fires it, a case that does not).
- `evals/` measures what only a model can get wrong: the classifier's answers on sample texts
  (`triage.yaml`), the agent's first tool (`agent.yaml`), generated text (`reply.yaml`).
  `uv run desk eval evals/triage.yaml --models claude-haiku-5-5,jev-latest --runs 3`.
- A real run's `REASONING.md` shows every rule that fired with the facts it saw: read it for the
  cases you changed.
