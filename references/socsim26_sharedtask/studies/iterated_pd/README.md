# iterated_pd — a ten-round two-player prisoner's dilemma

## What happens

Two players play a prisoner's dilemma for 10 rounds. Each round they choose to
cooperate or defect, and both see the full history of past rounds before
choosing. The payoffs at base are T=5, R=3, P=1, S=0. Each run is one full
ten-round game between two players.

## What we vary

**Signal variables** (the hypotheses are about these):

- **instruction_framing** — `canonical`, `moralized`, or `risk`. A framing note
  appended to the rules. The moralized note calls mutual benefit honest and
  betrayal questionable; the risk note stresses self-protection.
- **payoff_scale** — `lambda-1`, `lambda-0.1`, or `lambda-10`. The payoff
  magnitude scaled by 0.1, 1, or 10 with the structure fixed.
- **vignette_frame** — `abstract`, `business`, or `fictional`. The same game
  wrapped as a bare abstract game, a business-partners cover story, or an
  explicitly imaginary story.
- **model** — the four local models (Qwen3.5 4B / 9B / 27B and Gemma-4 31B).
- **persona_stance** — `neutral`, `cooperative`, `competitive`, or `reciprocal`.
  A game-playing disposition given to both players, stated in the persona ahead
  of the observations. Cooperative and competitive use the TRAILS
  (arXiv:2605.18890) personas; reciprocal is a tit-for-tat archetype; neutral
  adds nothing.

**Design variations** (also swept; whether they matter is for you to test):

- **persona_format** — `plain`, `descriptive`, or `tabular`.
- **prompt_wording** — two paraphrases plus a version that lists DEFECT before
  COOPERATE.
- **choice_labels** — `COOPERATE`/`DEFECT`, `GREEN`/`BLUE`, or
  `ACTION_A`/`ACTION_B`.
- **history_format** — `lines`, `table`, or `summary_counts`.

## Hypotheses

Full statements are in `design.yaml`. In short:

- **h1** — moralized framing raises the cooperation rate and risk framing lowers
  it, relative to canonical instructions.
- **h2** — attenuating the payoff magnitude increases defection; amplifying it
  does not reduce cooperation.
- **h3** — the explicitly imaginary story reduces cooperation relative to the
  business cover story.
- **h4** — the cooperation level differs across models, while the direction of
  the h1 framing effect holds for every model.
- **h5** — cooperation declines in the final rounds.
- **h6** — a cooperative stance raises cooperation and a competitive stance
  lowers it relative to neutral, while a reciprocal stance conditions on the
  other player's last move.

We hold back our own measures and predictions until after the deadline.
Choosing how to measure these hypotheses is the task. You may also pose and
test your own questions grounded in this scenario and its data.

## Reading the logs

Each run records its choices in `action_events.jsonl`: rows with
`action_type: choose_pd_action` carry the choice in canonical form (COOPERATE or
DEFECT) even when the condition relabels the actions, with the round index in
`data.round`. A `round_payoff` row logs the joint choices and payoffs for each
of the 10 rounds. The schedule includes one extra no-action finalization step so
the last round's payoffs land in the log. `prompts_and_responses.jsonl` shows
exactly what each player saw, including the labels, framing note, and history
rendering. The framing, scale, frame, and model for a run are in
`manifest.jsonl`.

## Sweep size

3 framings × 3 scales × 3 frames × 4 models × 4 persona stances = 432 grid
cells × 5 seeds = 2,160 grid runs, plus 1,305 design-variation runs (5 seeds,
mostly the primary model, at the neutral stance), for **3,465 runs** total
(about 69.3k model calls, 20 per run).
