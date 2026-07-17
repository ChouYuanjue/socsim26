# Final Simulated Cold Review — Cross-Study Version

## Overall simulated decision

Likely range: **Accept / strong Weak Accept** for the Shared Task, subject to the reviewer valuing execution-fidelity auditing as a conceptual contribution.

Estimated score: **8.7/10**.

This is not an independent external review. It is an internal hostile-read simulation by the same assistant that helped produce the artifact.

## Why the revised version is stronger

The old Beauty-only version was credible but still vulnerable to the objection that the core execution anomaly might be simulator- or scenario-specific. The cross-study version removes that weakness by auditing all five released sweeps and showing that the execution layer has multiple distinct regimes:

1. free-text game parsers greedily select early legal values or labels;
2. multi-tool posting can expand one model decision into multiple measured events;
3. Polarization can execute a generated action batch above the advertised per-turn cap;
4. strict integer probes provide a clean negative control.

The result is no longer just “Beauty Contest has a parser issue.” The contribution is now a general evaluation principle: social-simulation claims require an execution gate before metric interpretation.

## Strengths

- Covers all five official sweeps: 5,213 released runs.
- Separates claim identifiability, design robustness, and execution fidelity.
- Source-confirms the key parser/runtime mechanisms rather than relying only on log symptoms.
- Includes a negative control: Observed Norms has zero strict-integer discrepancies.
- Avoids overclaiming WVS fidelity where licensed anchors are unavailable.
- Reports both supported and contradicted claims, instead of forcing a single robustness story.
- Provides a claim-evidence ledger and one-command reproduction script.

## Likely positive reviewer summary

The submission makes a clear conceptual contribution by showing that robustness audits of LLM social simulations must be preceded by an execution-fidelity audit. Its strongest feature is the cross-study taxonomy: the same released benchmark contains greedy free-text parsers, multi-call measurement expansion, cap overshoot in open-ended social simulations, and a clean strict-integer control. The claim-identifiable audit is appropriately conservative and distinguishes support, contradiction, mixed evidence, and non-identifiability.

## Main residual risks

1. The paper is dense. Fitting all five studies into two pages makes some design details terse.
2. Some execution discrepancies are not necessarily “bugs”; Persona Expression multi-call behavior is intentionally allowed by runtime configuration, though it conflicts with the one-post measurement description.
3. Polarization cap overshoot is a serious data-quality finding, but the paper does not rerun counterfactual truncated simulations. This is the right conservative choice, but some reviewers may want repaired trajectories.
4. The official submission still needs a permanent anonymized code/result URL.

## Score estimate

| Dimension | Score |
|---|---:|
| Conceptual contribution | 9.0 |
| Metric/evaluation quality | 8.2 |
| Correctness | 8.7 |
| Reproducibility | 9.0 |
| Clarity under 2-page limit | 8.0 |
| Shared-task fit | 9.1 |

Weighted estimate: **8.7/10**.

## Final recommendation

Keep the cross-study framing. Do not revert to the Beauty-only paper. The cross-study version is more distinctive, harder to dismiss as a local anomaly, and much closer to what the Shared Task wants: an evaluation methodology rather than a leaderboard-style metric.
