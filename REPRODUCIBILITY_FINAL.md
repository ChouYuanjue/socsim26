# Reproducibility — Final Cross-Study Submission

## Environment used

- Python 3.13.5 on Windows.
- pandas 2.2.3, numpy 2.2.6, matplotlib 3.10.0.
- MiKTeX pdfLaTeX.
- Official `socsim26_sharedtask` repository cloned under `references/socsim26_sharedtask`.
- Pinned SiliSocS runtime source extracted under `references/silisocs_0_2_0/` for source-confirmed execution audits.

## Official data required

Download the official released sweep tarballs into:

`references/socsim26_sharedtask/socsim26_data/`

Expected tarball SHA256 values:

| Study | Expected SHA256 |
|---|---|
| beauty_contest | `62e80580beaa63af652942ecd1d2953640c34e7606539a91a6fc455ee4a34f95` |
| iterated_pd | `8a5590c95b8ace0f4d6552fa9885d0ac7dbbc918a485cb1667d91da779f69365` |
| observed_norms | `686e324ae66bbc6f7a5f22aea4766819d13b9dc00c3fd866992782983f18e2bf` |
| persona_expression | `6e8d28894d7deb73fa6c8ccd92c2270a78111f92a86e1487c6fea9ab5c0c16f8` |
| polarization | `b506d9e84d765fd1e66bb0d19c01a1ef572e52b85c085616508c5c2b7d5d96e2` |

Expected manifest SHA256 values after extraction:

| Study | Expected manifest SHA256 |
|---|---|
| beauty_contest | `7370f6dde54f4497ff3a63d81fc196c28d2cb25b06a29e386be5d45255e0d26c` |
| iterated_pd | `d9d51bc54136583c05761853d69cea023b77614a2816b473c6626658a6cba335` |
| observed_norms | `e6b01c59a754b0eedbd2a47df162b53e61de3b4d798c3d5a134fabd78fe5a6d4` |
| persona_expression | `d197bf081ce2663c1587529eb1020b82e939c1ae832b644b255ec5a80f07ef58` |
| polarization | `0b11c3df2a161d870e78112e2874baf62361e34991c2fa963573f461ff8647fa` |

## One-command reproduction

From the project root:

```bash
python analysis/reproduce_final.py
```

This command verifies the five official tarballs and extracted manifests, runs unit tests, regenerates all study-level audit artifacts with 4,000 bootstrap replicates, rebuilds the cross-study summary, generates the unlimited appendix, compiles both the two-page PDF and appendix twice, and validates every cited paper number against result files.

## Faster validation of the existing artifacts

When the artifact CSV/JSON files are already generated, this faster gate validates the paper and source hashes:

```bash
python -m unittest -v test_cross_study_audits.py
python analysis/build_cross_study_summary.py
python analysis/build_cross_study_evidence.py --skip-source-integrity --validation-output artifacts_cross_study_final/PAPER_NUMBER_VALIDATION_LOCAL.json
```

This checks the paper numbers against the checked-in result tables without requiring raw data. Full source-integrity validation is performed by `python analysis/reproduce_final.py` after the official tarballs are downloaded.

Current validation result:

- `artifacts_cross_study_final/PAPER_NUMBER_VALIDATION.json`: `PASS`.
- PDF pages: 2.
- Claim-evidence ledger rows: 43/43 verified.
- Five tarballs and five manifests match expected SHA256 values.
- Render check: `paper/PDF_RENDER_CHECK.json`, two nonblank rendered pages.

## Generated result directories

- `artifacts_final/` — Beauty Contest claim audit and parser audit.
- `artifacts_ipd_final/` — Iterated PD claim audit and parser audit.
- `artifacts_persona_expression_final/` — Persona Expression measurement-unit audit.
- `artifacts_observed_norms_final/` — Observed Norms strict-integer audit and identifiable within-simulation claim.
- `artifacts_polarization_final/` — Polarization claim audit and action-cap audit.
- `artifacts_cross_study_final/` — final cross-study tables, summary, and evidence validation.

## Non-reproducible by design

WVS human anchors are not redistributed in the official released package. Claims requiring WVS anchor distributions remain marked unavailable unless the evaluator independently obtains the licensed WVS Wave 7 data and rebuilds the anchors.
