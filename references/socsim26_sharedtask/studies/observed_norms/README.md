# observed_norms — public observation and cross-cultural value matching

## What happens

Agents drawn from four countries answer a short World Values Survey battery.
First they answer privately, in isolation (round 1, turn `t0`). Then, in the
observed condition, they answer again after seeing everyone's round-1 answers
plus a framing sentence (round 2, turn `t1`). Each agent is asked for one
integer per item. There is no free text, so there are no parse failures.

| item | scale | role |
|---|---|---|
| `q1_family_importance` | 1–4 (not at all → very important) | near-universal control |
| `q182_homosexuality` | 1–10 (never → always justifiable) | high cross-country variance |
| `q188_euthanasia` | 1–10 (never → always justifiable) | high cross-country variance |

In round 1 each elicitation is a fresh prompt holding the persona and the item
only, so the agent has no memory of having answered. In round 2 the prompt also
carries the other agents' round-1 answers and the framing sentence. The peer
view lives in the prompt, not in agent memory.

## What we vary

**Signal variables** (the hypotheses are about these):

- **population** — `us`, `netherlands`, `jordan`, `japan`, or `mixed`. The
  `mixed` roster is country-balanced, six agents from each of the four
  countries, and is the only roster where peers are from a different country.
- **condition** — `private` or `observed`. Private runs round 1 only; observed
  runs round 1, then round 2.
- **model** — the four local models (Qwen3.5 27B / 9B / 4B and Gemma-4 31B).
  Every model runs the full population × condition grid.

**Design variations** (also swept; whether they matter is for you to test):

- **scale_labels** — `justifiable` or the `acceptable` paraphrase of the anchor
  wording, on the private condition only.
- **framing** — the single sentence shown above peers' answers in the observed
  round: `neutral`, `same-culture`, or `different-culture`. Observed condition
  only.

## Hypotheses

Full statements are in `design.yaml`. In short (there is no h4):

- **h1** — in the private condition, populations reproduce the known
  cross-country differences on the high-variance items, and the control item
  shows no spurious cross-population variation.
- **h2** — even privately, the simulation compresses between-population
  differences relative to the human anchor.
- **h3** — public observation moves populations away from their per-country
  anchor relative to the private baseline, and the shift is larger under mixed
  observation than homogeneous; the control item stays flat.
- **h5** — per-population fidelity grows with model scale while
  interaction-induced compression shrinks; the smaller model conforms more.

We hold back our own measures and predictions until after the deadline.
Choosing how to measure these hypotheses is the task. You may also pose and
test your own questions grounded in this scenario and its data.

## The anchor

The WVS Wave 7 human distributions are not shipped. The license forbids
redistribution. Build them yourself: register at
[worldvaluessurvey.org](https://www.worldvaluessurvey.org/wvs.jsp), download
Wave 7 in CSV format (`WVS_Cross-National_Wave_7_csv_v6_0.csv`) into the
gitignored `studies/observed_norms/data/`, then run
`studies/observed_norms/get_anchor.py` to produce the matched per-country
distributions for Q1, Q182, and Q188.

## Reading the logs

`probe_events.jsonl` is the primary observable: one record per agent per item
per round. Each record carries `probe_return` (the parsed integer), `turn`
(`t0` or `t1`), `condition`, `item`, and the answering agent. The private
baseline is the `t0` records. The interaction effect is `t1 − t0` within the
observed condition. `prompts_and_responses.jsonl` is the audit trail: round-1
prompts contain no peer answers, round-2 prompts contain peers plus the framing
sentence. Per-persona demographics ride along in the roster JSONs and are never
mapped to attitudes by us.

## Sweep size

10 grid cells (5 populations × 2 conditions) × 6 seeds (6001–6006) = 60 grid
runs. Plus 45 variation conditions × 6 seeds = 270 variation runs (model: the 3
non-primary models × the 10-cell grid = 30; scale_labels: 5; framing: 10, the
latter two at the primary model only), for **330 runs** total (about 36,720
probe elicitations).
