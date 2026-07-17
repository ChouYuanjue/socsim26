# beauty_contest — the 11-20 money request game

## What happens

Two players each request a whole number of points from 11 to 20. Each player
receives the number they request. A player who requests exactly one less than
the other player gets a 20-point bonus. The two requests are made at the same
time, and the game is played once. This is the game from Arad & Rubinstein
(2012). Requesting 20 is the safe, no-reasoning choice; each step down to 19,
18, 17 is one more round of "but if they expect me to do that, I should..."

Each run is one play of the game, so it produces two choices.

## What we vary

**Signal variables** (the hypotheses are about these):

- **game_variant** — `basic`, `cycle`, or `costless`. The three payoff rules
  from Arad & Rubinstein, each with a published human distribution.
- **goal_framing** — `none` or `strategic`. The strategic framing adds a
  sentence telling the player the opponent is also reasoning about them.
- **model** — the four local models (Qwen3.5 4B / 9B / 27B and Gemma-4 31B).
- **persona** — `neutral`, `intuitive`, `cautious`, or `competitive`. A player
  disposition stated in the persona ahead of the observations: going with
  instinct, securing a high guaranteed amount, or undercutting to win the bonus.
  The neutral level adds no disposition. Separate from `goal_framing`.

**Design variations** (also swept; whether they matter is for you to test):

- **instruction_wording** — four rewordings of the rules that keep the same
  meaning (two paraphrases, a descending 20-to-11 range, numbers as words).
- **response_format** — the `CHOOSE_NUMBER: <n>` format or a bare number.
- **temperature** — the sampling temperature.
- **persona_format** — `plain`, `descriptive`, or `tabular`. The same persona
  content rendered three ways (terse, elaborated prose, pipe-delimited table).

## Human anchors

The three human choice distributions from Arad & Rubinstein ship in
`design.yaml` under `anchors`. They are the only ground truth in this study:

| choice | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 |
|---|---|---|---|---|---|---|---|---|---|---|
| basic (n=108) | .04 | .00 | .03 | .06 | .01 | .06 | **.32** | .30 | .12 | .06 |
| cycle (n=72) | .01 | .01 | .00 | .01 | .00 | .04 | .10 | .22 | **.47** | .13 |
| costless (n=53) | .00 | .04 | .00 | .04 | .04 | .04 | .09 | .21 | **.40** | .15 |

## Hypotheses

Full statements are in `design.yaml`. In short:

- **h1** — in the basic game with neutral framing, choices concentrate in the
  human 17-19 region rather than at 19-20.
- **h2** — the cyclic bonus shifts choices upward.
- **h3** — removing the cost of undercutting does not deepen reasoning past
  about three steps.
- **h4** — the strategic framing shifts choices downward.
- **h5** — apparent reasoning depth does not decrease with model scale in the
  Qwen family, and the h2 shift holds for all four models.
- **h6** — a cautious disposition shifts choices upward (toward the secure
  anchor) and a competitive disposition shifts them downward (more
  undercutting), relative to neutral.

We hold back our own measures and predictions until after the deadline.
Choosing how to measure these hypotheses is the task. You may also pose and
test your own questions grounded in this scenario and its data.

## Reading the logs

Each run records two choices in `action_events.jsonl`: rows with
`action_type: choose_number` carry the choice in `data.choice`. A
`round_payoff` row logs the two choices and the payoffs. The reasoning text for
each choice is in `prompts_and_responses.jsonl`. The variant, framing, and model
for a run are in `manifest.jsonl`. Estimate distributions per condition over its
5 seeds.

## Sweep size

3 variants × 2 framings × 4 models × 4 personas = 96 signal-grid cells × 5 seeds
= 480 runs, plus 38 design-variation arms × 5 seeds = 190 (at the neutral
persona), for **670 runs** total (about 1,340 model calls).
