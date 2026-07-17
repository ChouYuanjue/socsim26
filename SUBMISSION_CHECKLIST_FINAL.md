# Final Submission Checklist

## Scientific status

`PASS` for scientific/reproducibility checks.

`NOT_SUBMISSION_READY` for official submission logistics until a permanent anonymized repository URL or organizer-approved supplementary upload is configured.

## Paper

- [x] Final PDF: `paper/main_final.pdf`.
- [x] Exactly 2 pages.
- [x] Anonymous author line.
- [x] No unsupported IPD-era claims retained from the older draft.
- [x] Covers all five released sweeps.
- [x] Main thesis shifted from single-study Beauty-only claim audit to cross-study execution-before-evaluation audit.

## Data and source integrity

- [x] Beauty Contest tarball and manifest SHA256 verified.
- [x] Iterated PD tarball and manifest SHA256 verified.
- [x] Observed Norms tarball and manifest SHA256 verified.
- [x] Persona Expression tarball and manifest SHA256 verified.
- [x] Polarization tarball and manifest SHA256 verified.
- [x] Pinned source excerpts recorded for parser/runtime claims.

## Validation

- [x] Cross-study unit tests pass: 7/7.
- [x] `analysis/build_cross_study_summary.py` regenerates the main cross-study tables.
- [x] `analysis/build_cross_study_evidence.py` reports `PASS`.
- [x] `CLAIM_EVIDENCE_LEDGER.csv` has 43 verified rows.
- [x] `paper/PDF_RENDER_CHECK.json` confirms two nonblank rendered pages.

## Public package constraints

- [x] Public package excludes official raw sweep tarballs.
- [x] Public package includes paper, code, result artifacts, ledger, README, and reproducibility instructions.
- [x] No permanent personal identity details intentionally included in the public package.
- [x] Secret scan run before final links.

## Remaining official-submission blocker

A permanent anonymized code/result location is still required. Temporary MCP download links are useful for handoff but should not be used as the final paper/reviewer link.
