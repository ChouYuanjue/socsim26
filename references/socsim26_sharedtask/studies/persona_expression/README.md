# persona_expression — where agent-population diversity comes from

## What happens

A population of 30 LLM agents each writes one post in response to the same
neutral, open-ended prompt ("share what is on your mind"). The agents differ
only in how they are described and how they are instructed. The study measures
how much the 30 posts differ from each other (inter-agent diversity) and which
factor drives it: how richly each agent is described, how the posting
instruction is worded, which model runs, and whether the agents can see each
other.

Each run produces 30 posts, one per agent per step.

## The persona-richness ladder

The personas come from NVIDIA Nemotron-Personas-USA joined by `uuid` to
Salesforce SCOPE. The same 30 individuals appear at every rung. Only the amount
of detail an agent is told about itself changes:

| rung | what the agent is told about itself |
|---|---|
| `generic` | nothing; a neutral "User NN" name and one generic line |
| `demographic` | structured fields only: age, sex, education, occupation, location |
| `full-bio` | the above plus the free-text biography |
| `sociopsychological` | the above plus SCOPE values, Big-Five traits, identity narrative |

Style and goal are held constant across rungs; only the persona text changes.

## What we vary

**Signal variables** (the hypotheses are about these):

- **persona_richness** — `generic`, `demographic`, `full-bio`, or
  `sociopsychological`. The ladder above.
- **action_prompt** — `deliberate`, `reactive`, or `engagement-max`. Three
  wordings of the posting instruction.
- **interaction_pathway** — `closed` or `open`. Whether agents see each other's
  posts.
- **model** — the four local models (Qwen3.5 4B / 9B / 27B and Gemma-4 31B).

**Design variations** (also swept; whether they matter is for you to test):

- **persona_paraphrase** — `original` or `reworded`. The biographies reworded.
  Swept at the full-bio rung and primary model.
- **persona_order** — `original` or `reversed`. The roster order reversed. Swept
  at the full-bio rung and primary model.

## Hypotheses

Full statements are in `design.yaml`. In short:

- **h1** — richer persona grounding raises diversity, but with diminishing and
  unreliable returns; moving from generic to demographic moves it less than the
  added content would suggest.
- **h2** — changing one sentence of the action prompt shifts diversity by an
  amount comparable to changing persona richness or the model.
- **h3** — diversity does not increase with model scale within the Qwen family.
- **h4** — opening the pathway changes diversity relative to the closed arm;
  whether it amplifies or collapses diversity is open.
- **h5** — the richness and prompt orderings are stable under persona paraphrase,
  persona reordering, and reseeding.

We hold back our own measures and predictions until after the deadline.
Choosing how to measure these hypotheses is the task. You may also pose and
test your own questions grounded in this scenario and its data.

## Reading the logs

Each run records one post per agent per step in `action_events.jsonl`. The full
prompt and response for each post is in `prompts_and_responses.jsonl`, the audit
trail: in closed runs it shows the agent saw no peer content; in open runs it
shows the timeline the agent saw. Per-persona metadata (sex, age, education,
occupation) rides along in the roster JSONs for analysis but is not put into
prompts beyond the rung context. The richness, prompt, pathway, and model for a
run are in `manifest.jsonl`.

## Personas / regenerating

The rosters are committed under `scenario/assets/personas/` (`roster.json`,
`roster_paraphrase.json`, `roster_order_b.json`): 30 census-stratified
individuals from the Nemotron × SCOPE uuid join. To rebuild, download both
datasets from Hugging Face into the gitignored `data/`, run `get_personas.py` to
join on uuid and draw the sample, then run
`tools/gen_persona_expression_rosters.py` to build the rung context strings and
the paraphrase and reversed variants. SCOPE is CC BY-NC, research-only, so the
sociopsychological rung must not be used for commercial work.

## Sweep size

48 signal-grid cells (4 richness × 3 prompt × 4 model) × 8 seeds = 384 grid runs,
plus 144 variation runs (96 open-arm runs at the primary model across richness ×
prompt, 24 paraphrase, 24 reorder), for **528 runs** total.
