# Execution Before Evaluation in LLM Social Simulations

This repository contains the public artifact for a COLM 2026 Many Worlds Shared Task submission on evaluating LLM social simulations. The paper audits all five released sweeps and asks a simple question before interpreting any social-scientific result: whether the target claim is identifiable from matched sweep cells, and whether the recorded event stream preserves the action or measurement unit that the claim relies on.

Current repository URL: `git@github.com:ChouYuanjue/socsim26.git`.

Blind-review note: this GitHub repository is not anonymous because it is hosted under a personal account. For a blind submission, mirror the same contents to an anonymous repository or use an organizer-approved supplementary upload location.

## Repository status

| Check | Status | Where to verify |
|---|---:|---|
| Main paper included | yes | `paper/main_final.pdf`, `paper/main_final.tex` |
| Appendix included | yes | `paper/supplement_final.pdf`, `paper/supplement_final.tex` |
| All five official sweeps covered | yes | `artifacts_cross_study_final/summary.json` |
| Paper number validation | PASS | `artifacts_cross_study_final/PAPER_NUMBER_VALIDATION.json` |
| Manuscript preflight | PASS | `artifacts_cross_study_final/MANUSCRIPT_PREFLIGHT.json` |
| Claim evidence ledger | 43/43 verified | `CLAIM_EVIDENCE_LEDGER.csv` |
| PDF render check | nonblank pages | `paper/PDF_RENDER_CHECK.json` |
| Official raw sweep tarballs included | no | excluded by design |
| Extracted run logs included | no | excluded by design |
| Tracked files over GitHub's 50 MB warning threshold | no | checked before push |
| Secret scan before push | clean | checked before push |

## Main results in the repository

The cross-study audit covers 5,213 released runs:

| Study | Runs | Main interface audit result | Main files |
|---|---:|---|---|
| Beauty Contest | 670 | 230/1,340 game actions differ between the recorded parser output and the model's later explicit `number:` parameter line; target verdicts change for H2 and H6b under this sensitivity projection. | `artifacts_final/summary.json`, `artifacts_final/hypothesis_audit.csv`, `artifacts_final/claim_execution_sensitivity.csv`, `artifacts_final/parser_source_audit.json` |
| Iterated PD | 3,465 | 58/69,300 categorical game actions differ between the recorded parser output and the model's later explicit `choice:` parameter line; target verdicts do not change. | `artifacts_ipd_final/summary.json`, `artifacts_ipd_final/hypothesis_audit.csv`, `artifacts_ipd_final/claim_execution_sensitivity.csv`, `artifacts_ipd_final/parser_source_audit.json` |
| Persona Expression | 528 | Multi-tool configuration expands 961 decisions into 2,289 extra post events relative to the one-post measurement description. | `artifacts_persona_expression_final/summary.json`, `artifacts_persona_expression_final/hypothesis_audit.csv`, `artifacts_persona_expression_final/measurement_unit_source_audit.json` |
| Observed Norms | 330 | Strict integer probes are a clean execution control: 0 discrepancies across 36,720 probes. WVS human-anchor claims remain unavailable without licensed WVS anchors. | `artifacts_observed_norms_final/summary.json`, `artifacts_observed_norms_final/identifiable_hypothesis_audit.csv`, `artifacts_observed_norms_final/nonidentifiable_official_claims.csv` |
| Polarization | 220 | 23,327 agent-turns execute actions above the advertised three-action cap, totaling 60,634 above-cap social events. | `artifacts_polarization_final/summary.json`, `artifacts_polarization_final/hypothesis_audit.csv`, `artifacts_polarization_final/action_cap_source_audit.json` |

## Quick start

Install the Python dependencies:

```bash
pip install -r requirements-final.txt
```

Fast validation of the checked-in artifacts:

```bash
python -m unittest -v analysis/test_cross_study_audits.py
python analysis/build_cross_study_summary.py
python analysis/build_cross_study_evidence.py --skip-source-integrity --validation-output artifacts_cross_study_final/PAPER_NUMBER_VALIDATION_LOCAL.json
python analysis/manuscript_preflight.py
```

Full reproduction, after downloading the official sweep tarballs as described in `REPRODUCIBILITY_FINAL.md`:

```bash
python analysis/reproduce_final.py
```

The full command verifies official data hashes, runs the unit tests, regenerates all study-level artifacts, rebuilds cross-study tables, regenerates the appendix, compiles the main paper and appendix, and validates every cited paper number.

## Data policy

This repository intentionally does not include the official raw sweep tarballs or extracted run logs. Put the official files under:

```text
references/socsim26_sharedtask/socsim26_data/
```

Expected SHA256 hashes for the five tarballs and their extracted manifests are listed in `REPRODUCIBILITY_FINAL.md` and rechecked by `analysis/reproduce_final.py`.

Three large, regenerable intermediate CSVs are also omitted from GitHub to keep the repository cloneable and below GitHub's large-file warning threshold:

```text
artifacts_ipd_final/action_projection.csv
artifacts_persona_expression_final/decision_level.csv
artifacts_polarization_final/decision_level.csv
```

They are regenerated by `python analysis/reproduce_final.py`. Paper-facing tables, summaries, evidence ledgers, and validation JSON files remain included.

## Directory and file map

### Root files

| File | Role |
|---|---|
| `README.md` | Main repository guide and file map. |
| `README_PUBLIC.md` | Short public-artifact summary. Kept for package compatibility. |
| `REPRODUCIBILITY_FINAL.md` | Official data download layout, tarball and manifest hashes, and reproduction commands. |
| `SUBMISSION_CHECKLIST_FINAL.md` | Scientific and logistics checklist for final submission. |
| `FINAL_REVIEW.md` | Internal-style assessment of strengths, weaknesses, and expected review outcome. |
| `CLAIM_EVIDENCE_LEDGER.csv` | Paper-number ledger. Each main-text number is mapped to a result file, selector, and validation status. |
| `PACKAGE_CONTENTS_MANIFEST.json` | Package policy: included files, excluded raw data, and repository naming note. |
| `requirements-final.txt` | Minimal Python package versions used for the final analysis. |
| `.gitignore` | Excludes caches, LaTeX build files, official raw data, extracted run logs, and large regenerable intermediates. |

### `paper/`

| File | Role |
|---|---|
| `paper/main_final.pdf` | Final two-page paper. |
| `paper/main_final.tex` | LaTeX source for the main paper. |
| `paper/supplement_final.pdf` | Appendix with complete claim tables, execution projections, source provenance, and data hashes. |
| `paper/supplement_final.tex` | Generated LaTeX source for the appendix. |
| `paper/PDF_RENDER_CHECK.json` | Render check for main and supplement PDFs. Records page count, text extraction length, image dimensions, and blank-page checks. |

### `analysis/`

| File | Produces or verifies |
|---|---|
| `analysis/reproduce_final.py` | Full one-command reproduction pipeline. Verifies data hashes, runs tests, regenerates artifacts, compiles PDFs, and runs validation. |
| `analysis/final_cira_analysis.py` | Beauty Contest claim audit, parser replay, design-cell audit, and execution-sensitivity tables. |
| `analysis/ipd_cira_analysis.py` | Iterated PD claim audit, parser replay, cooperation contrasts, and execution-sensitivity tables. |
| `analysis/observed_norms_audit.py` | Observed Norms strict-integer probe audit and identifiable within-simulation norm-dispersion claim. |
| `analysis/persona_expression_execution_audit.py` | Persona Expression multi-tool measurement-unit audit, diversity metrics, and claim tables. |
| `analysis/polarization_cira_analysis.py` | Polarization claim audit, exposure metrics, and action-cap execution audit. |
| `analysis/build_cross_study_summary.py` | Cross-study execution table, selected-claim table, and `artifacts_cross_study_final/summary.json`. |
| `analysis/build_cross_study_evidence.py` | Main-text number validation and `CLAIM_EVIDENCE_LEDGER.csv`. |
| `analysis/build_supplement.py` | Generates `paper/supplement_final.tex` from artifact CSV/JSON files. |
| `analysis/manuscript_preflight.py` | Checks common manuscript problems: stale numbers, raw paths, overused coined acronyms, abstract formatting, and run totals. |
| `analysis/test_cross_study_audits.py` | Unit tests for parser behavior, strict integer parsing, multi-tool parsing, and edge-alignment metric. |
| `analysis/test_final_cira_analysis.py` | Unit tests for Beauty Contest analysis helpers. |
| `analysis/bootstrap_uncertainty.py` | Shared bootstrap utilities used by study analyses. |
| `analysis/beauty_metrics.py` | Earlier Beauty Contest metric helper retained for reproducibility of related outputs. |
| `analysis/make_figure.py` | Figure helper for checked-in Beauty Contest plots. |
| `analysis/build_submission_evidence.py` | Earlier single-study evidence builder retained for provenance of earlier result checks. |
| `analysis/build_final_packages.py` | Local packaging script used to create the public artifact zip. Not required for reviewing the repository contents. |

### `artifacts_cross_study_final/`

| File | Result |
|---|---|
| `summary.json` | Cross-study totals, execution taxonomy, interface audit table, selected claim list, and scope statement. |
| `execution_interface_audit.csv` | Table 1 source data: one row per study with runs, interface type, discrepancy counts, affected models, and scientific impact. |
| `selected_claim_audit.csv` | Table 2 source data: selected atomic claims, signed effect estimates, confidence intervals, target verdicts, and design verdicts. |
| `execution_table.tex` | LaTeX-ready version of the execution-interface table. |
| `PAPER_NUMBER_VALIDATION.json` | Evidence gate for the main paper. Confirms page count, 43/43 ledger rows, required phrases, and official data hashes. |
| `MANUSCRIPT_PREFLIGHT.json` | Manuscript-quality gate for common AI-draft problems and number drift. |

### `artifacts_final/` for Beauty Contest

| File | Result |
|---|---|
| `summary.json` | Study-level summary for Beauty Contest. |
| `hypothesis_audit.csv` | Main atomic-claim audit under recorded actions. |
| `explicit_parameter_hypothesis_audit.csv` | Sensitivity audit using model-emitted explicit `number:` parameters. |
| `claim_execution_sensitivity.csv` | Claim-level comparison between recorded actions and explicit-parameter projection. |
| `hypothesis_effect_rows.csv` | Run/context-level effect rows used to aggregate each claim. |
| `explicit_parameter_hypothesis_effect_rows.csv` | Effect rows for the explicit-parameter projection. |
| `design_identifiability.csv` | Which non-default design cells identify each claim contrast. |
| `explicit_parameter_design_identifiability.csv` | Design-cell audit under explicit-parameter projection. |
| `matched_design_effects.csv` | Matched non-default versus default design contrasts. |
| `explicit_parameter_matched_design_effects.csv` | Matched design contrasts under explicit-parameter projection. |
| `agent_choices.csv` | Parsed action-level choices for the released sweep. |
| `condition_summary.csv` | Condition-level recorded-action summaries. |
| `explicit_parameter_condition_summary.csv` | Condition-level explicit-parameter summaries. |
| `cycle_effect_by_model.csv` | Model-specific cycle-effect summaries. |
| `explicit_parameter_cycle_effect_by_model.csv` | Model-specific cycle-effect summaries under explicit-parameter projection. |
| `execution_fidelity_summary.json` | Counts of parser/final-parameter divergences and affected claims. |
| `execution_fidelity_by_condition.csv` | Divergence rates by sweep condition. |
| `execution_fidelity_by_model.csv` | Divergence rates by model. |
| `executor_divergent_actions.csv` | Action-level rows where recorded action and explicit parameter disagree. |
| `executor_mismatch_patterns.csv` | Pattern summary of divergence types. |
| `parser_source_audit.json` | Source hash and source-location evidence for the Beauty parser. |
| `data_integrity.json` | Input data checks for the Beauty sweep. |
| `permutation_tests.json` | Permutation-test results for selected Beauty checks. |
| `anchor_sanity_checks.csv` | Sanity checks for the earlier anchor-distance issue. |
| `mechanism_condition_summary.csv` | Condition-level mechanism summaries. |
| `mechanism_summary.json` | Mechanism-level summary statistics. |
| `mechanism_transition_matrix.csv` | Transition matrix for mechanism analysis. |
| `mechanism_transitions.csv` | Transition-level mechanism rows. |
| `mechanism_table.tex` | LaTeX-ready mechanism table. |
| `h5c_model_direction.json` | Model-direction check for H5c. |
| `explicit_parameter_h5c_model_direction.json` | H5c model-direction check under explicit-parameter projection. |
| `prompt_feature_summary.csv` | Prompt feature summary used in exploratory checks. |
| `run_level.csv` | Beauty run-level metrics used by audits. |
| `claim_table.tex` | LaTeX-ready Beauty claim table. |
| `claim_effects.pdf`, `claim_effects.png` | Beauty claim-effect figure. |
| `identifiability_matrix.pdf`, `identifiability_matrix.png` | Beauty identifiability matrix figure. |
| `ARTIFACT_MANIFEST.json` | Input/output manifest for this artifact directory. |

### `artifacts_ipd_final/` for Iterated PD

| File | Result |
|---|---|
| `summary.json` | Study-level summary for Iterated PD. |
| `hypothesis_audit.csv` | Main atomic-claim audit under recorded game actions. |
| `explicit_parameter_hypothesis_audit.csv` | Sensitivity audit using model-emitted explicit `choice:` parameters. |
| `claim_execution_sensitivity.csv` | Claim-level recorded-versus-explicit sensitivity. |
| `hypothesis_effect_rows.csv` | Effect rows for recorded-action IPD claims. |
| `explicit_parameter_hypothesis_effect_rows.csv` | Effect rows for explicit-parameter IPD claims. |
| `design_identifiability.csv` | Identifiable non-default design cells for IPD claims. |
| `explicit_parameter_design_identifiability.csv` | Design-cell audit under explicit-parameter projection. |
| `matched_design_effects.csv` | Matched design effects under recorded actions. |
| `explicit_parameter_matched_design_effects.csv` | Matched design effects under explicit-parameter projection. |
| `divergent_actions.csv` | Action-level rows where recorded choice and explicit choice differ. |
| `execution_fidelity_by_model.csv` | Divergence rates by model. |
| `execution_fidelity_by_round.csv` | Divergence rates by repeated-game round. |
| `execution_fidelity_by_signal_context.csv` | Divergence rates by signal/context cell. |
| `framing_effect_by_model.csv` | Model-specific framing-effect summaries. |
| `model_cooperation_summary.csv` | Cooperation rates by model. |
| `parser_source_audit.json` | Source hash and source-location evidence for the IPD parser. |
| `synthetic_parser_checks.csv` | Synthetic parser sanity checks. |
| `run_level.csv` | Run-level cooperation and condition metrics. |
| `data_integrity.json` | Input data checks for the IPD sweep. |
| `ARTIFACT_MANIFEST.json` | Input/output manifest for this artifact directory. |

Omitted but regenerable: `action_projection.csv`, a high-granularity action-level intermediate over GitHub's recommended file-size threshold.

### `artifacts_observed_norms_final/` for Observed Norms

| File | Result |
|---|---|
| `summary.json` | Study-level summary for Observed Norms. |
| `identifiable_hypothesis_audit.csv` | Auditable within-simulation claim results. |
| `nonidentifiable_official_claims.csv` | Official claims marked unavailable because WVS anchors are not redistributed. |
| `dispersion_compression_effect_rows.csv` | Effect rows for the observation-compresses-dispersion claim. |
| `probe_level.csv` | Strict integer probe-level rows. |
| `run_level.csv` | Run-level norm-response metrics. |
| `country_rank_stability.csv` | Country-rank stability details. |
| `country_rank_stability_summary.csv` | Summary of country-rank stability. |
| `parser_source_audit.json` | Source hash and source-location evidence for strict integer parsing. |
| `data_integrity.json` | Input data checks for the Observed Norms sweep. |
| `ARTIFACT_MANIFEST.json` | Input/output manifest for this artifact directory. |

### `artifacts_persona_expression_final/` for Persona Expression

| File | Result |
|---|---|
| `summary.json` | Study-level summary for Persona Expression. |
| `hypothesis_audit.csv` | Main atomic-claim audit under executed post events. |
| `strict_one_action_hypothesis_audit.csv` | Sensitivity audit under one-decision/one-post projection. |
| `claim_execution_sensitivity.csv` | Claim-level comparison between executed posts and one-post projection. |
| `hypothesis_effect_rows.csv` | Effect rows for executed-post claims. |
| `strict_one_action_hypothesis_effect_rows.csv` | Effect rows for one-post projection claims. |
| `design_identifiability.csv` | Identifiable non-default design cells under executed events. |
| `strict_one_action_design_identifiability.csv` | Design-cell audit under one-post projection. |
| `actions_per_decision_distribution.csv` | Distribution of tool-call counts per model decision. |
| `multi_action_outputs.csv` | Rows where one model decision produced multiple post events. |
| `execution_fidelity_by_model.csv` | Multi-action expansion rates by model. |
| `factor_ranges.csv` | Range of effects by persona/model factors. |
| `interaction_effect.csv` | Interaction-effect summary for selected factors. |
| `measurement_unit_source_audit.json` | Source evidence for one-post measurement description and multi-call execution behavior. |
| `run_level.csv` | Run-level diversity and condition metrics. |
| `data_integrity.json` | Input data checks for the Persona Expression sweep. |
| `ARTIFACT_MANIFEST.json` | Input/output manifest for this artifact directory. |

Omitted but regenerable: `decision_level.csv`, a high-granularity decision-level intermediate over GitHub's recommended file-size threshold.

### `artifacts_polarization_final/` for Polarization

| File | Result |
|---|---|
| `summary.json` | Study-level summary for Polarization. |
| `hypothesis_audit.csv` | Main atomic-claim audit for polarization metrics. |
| `hypothesis_effect_rows.csv` | Effect rows for polarization claims. |
| `design_identifiability.csv` | Identifiable non-default design cells for polarization claims. |
| `action_cap_overshoots.csv` | Agent-turns where executed events exceed the advertised three-action cap. |
| `action_cap_source_audit.json` | Source evidence for batch resolution before action-cap checking. |
| `execution_fidelity_by_model.csv` | Overshoot and execution-fidelity summaries by model. |
| `exposure_effect_by_model.csv` | Exposure-effect summaries by model. |
| `exposure_effect_by_topic.csv` | Exposure-effect summaries by topic. |
| `run_level.csv` | Run-level network and polarization metrics. |
| `data_integrity.json` | Input data checks for the Polarization sweep. |
| `ARTIFACT_MANIFEST.json` | Input/output manifest for this artifact directory. |

Omitted but regenerable: `decision_level.csv`, a high-granularity decision-level intermediate over GitHub's recommended file-size threshold.

### `references/`

These are lightweight source files needed for parser/runtime audit reproducibility. They are not raw sweep data.

| Path | Role |
|---|---|
| `references/socsim26_sharedtask/task_components/game.py` | Released parser source for Beauty Contest and Iterated PD game actions. |
| `references/socsim26_sharedtask/studies/*/README.md` | Official study descriptions used to verify measurement intent and hypotheses. |
| `references/socsim26_sharedtask/studies/*/design.yaml` | Official design grids used to interpret signal/design variables. |
| `references/socsim26_sharedtask/studies/persona_expression/scenario/sim.yaml` | Persona Expression simulation configuration. |
| `references/socsim26_sharedtask/studies/persona_expression/scenario/world/default.yaml` | Persona Expression world definition used for one-post measurement interpretation. |
| `references/socsim26_sharedtask/studies/polarization/scenario/sim.yaml` | Polarization simulation configuration, including action-cap settings. |
| `references/silisocs_0_2_0/extracted/silisocs/runtime/prompts/action_prompts.py` | Pinned runtime prompt code for generated action batches. |
| `references/silisocs_0_2_0/extracted/silisocs/environments/gm/components/resolve.py` | Pinned runtime resolver used to explain multi-call execution. |
| `references/silisocs_0_2_0/extracted/silisocs/simulation_engines/policies/turns.py` | Pinned runtime turn policy used to explain action-cap timing. |
| `references/silisocs_0_2_0/silisocs-0.2.0-py3-none-any.whl` | Pinned SiliSocS wheel for source provenance. |

## What to cite or link in a submission

Use the permanent repository link for code/results, plus these files as entry points:

```text
paper/main_final.pdf
paper/supplement_final.pdf
artifacts_cross_study_final/PAPER_NUMBER_VALIDATION.json
CLAIM_EVIDENCE_LEDGER.csv
REPRODUCIBILITY_FINAL.md
```

For blind review, use an anonymous mirror containing the same commit contents rather than this personal-account URL.
