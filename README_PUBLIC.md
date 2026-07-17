# COLM 2026 Many Worlds Shared Task — Public Artifact

This artifact contains an anonymized, public-only submission package for the COLM 2026 Many Worlds Shared Task on Evaluating LLM Social Simulations.

Suggested anonymized repository name: `socsim26-claim-execution-audit`.

## Paper

- `paper/main_final.pdf` — final two-page submission.
- `paper/main_final.tex` — LaTeX source for the submission.
- `paper/supplement_final.pdf` — unlimited appendix with all atomic claims, execution projections, source provenance, and data hashes.
- `paper/supplement_final.tex` — LaTeX source for the appendix.

## Method

The submission implements a cross-study claim-identifiable audit with an execution check. The core rule is that a social-simulation verdict is only meaningful when:

1. the atomic claim is identifiable from matched released sweep cells;
2. the result is checked under available design variations without imputing missing cells;
3. the simulator's executed event stream preserves the intended action or measurement unit.

## Released-sweep coverage

The final audit covers all five official released sweeps:

- Beauty Contest: 670 runs.
- Iterated PD: 3,465 runs.
- Persona Expression: 528 runs.
- Observed Norms: 330 runs.
- Polarization: 220 runs.

Total: 5,213 released runs.

## Main execution findings

- Beauty Contest: 230/1,340 free-text integer actions diverge from the final explicit number field; the released parser selects the first legal integer.
- Iterated PD: 58/69,300 free-text categorical actions diverge from the final explicit choice field; the released parser selects the earliest displayed label.
- Persona Expression: multi-tool configuration expands 961 decisions into 2,289 extra post events relative to the one-post measurement description.
- Observed Norms: strict integer probe control has zero execution discrepancy across 36,720 probes.
- Polarization: 23,327 agent-turns execute actions above the advertised three-action cap, totaling 60,634 above-cap events.

## Important scope limits

The package does not redistribute official raw sweep tarballs. Download them from the official Hugging Face dataset using the instructions in `REPRODUCIBILITY_FINAL.md`.

World Values Survey anchors for Observed Norms are not included because the official task notes that WVS redistribution is license-restricted. Human-anchor fidelity claims for that study are therefore marked unavailable unless the evaluator obtains WVS data independently.

No permanent repository URL is included in this package. The scientific checks pass, but the official submission remains `NOT_SUBMISSION_READY` until an anonymized permanent code/result repository or organizer-approved supplementary upload location is configured.

## GitHub-size note

The repository omits three regenerable high-granularity CSV intermediates that exceed GitHub's recommended 50 MB file size. They are produced by `python analysis/reproduce_final.py` from the official released sweep tarballs:

- `artifacts_ipd_final/action_projection.csv`
- `artifacts_persona_expression_final/decision_level.csv`
- `artifacts_polarization_final/decision_level.csv`

All paper-facing tables, validation JSON files, the claim evidence ledger, the main PDF, and the supplement PDF remain included.
