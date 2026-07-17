from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_ZIP = ROOT / "COLM2026_ManyWorlds_claim_execution_crossstudy_submission_public.zip"
AUDIT_ZIP = ROOT / "COLM2026_ManyWorlds_claim_execution_crossstudy_audit_handoff.zip"
FINAL_MANIFEST = ROOT / "FINAL_DELIVERY_MANIFEST.json"
PACKAGE_MANIFEST = ROOT / "PACKAGE_CONTENTS_MANIFEST.json"

PUBLIC_DIRS = [
    "analysis",
    "artifacts_final",
    "artifacts_ipd_final",
    "artifacts_observed_norms_final",
    "artifacts_persona_expression_final",
    "artifacts_polarization_final",
    "artifacts_cross_study_final",
]
PUBLIC_ROOT_FILES = [
    "README_PUBLIC.md",
    "REPRODUCIBILITY_FINAL.md",
    "SUBMISSION_CHECKLIST_FINAL.md",
    "requirements-final.txt",
    "CLAIM_EVIDENCE_LEDGER.csv",
    "FINAL_REVIEW.md",
    "PACKAGE_CONTENTS_MANIFEST.json",
]
PAPER_FILES = [
    "paper/main_final.tex",
    "paper/main_final.pdf",
    "paper/supplement_final.tex",
    "paper/supplement_final.pdf",
    "paper/PDF_RENDER_CHECK.json",
]
REFERENCE_FILES = [
    "references/socsim26_sharedtask/task_components/game.py",
    "references/socsim26_sharedtask/studies/beauty_contest/README.md",
    "references/socsim26_sharedtask/studies/beauty_contest/design.yaml",
    "references/socsim26_sharedtask/studies/iterated_pd/README.md",
    "references/socsim26_sharedtask/studies/iterated_pd/design.yaml",
    "references/socsim26_sharedtask/studies/observed_norms/README.md",
    "references/socsim26_sharedtask/studies/observed_norms/design.yaml",
    "references/socsim26_sharedtask/studies/persona_expression/README.md",
    "references/socsim26_sharedtask/studies/persona_expression/design.yaml",
    "references/socsim26_sharedtask/studies/persona_expression/scenario/world/default.yaml",
    "references/socsim26_sharedtask/studies/persona_expression/scenario/sim.yaml",
    "references/socsim26_sharedtask/studies/polarization/README.md",
    "references/socsim26_sharedtask/studies/polarization/design.yaml",
    "references/socsim26_sharedtask/studies/polarization/scenario/sim.yaml",
    "references/silisocs_0_2_0/silisocs-0.2.0-py3-none-any.whl",
    "references/silisocs_0_2_0/extracted/silisocs/runtime/prompts/action_prompts.py",
    "references/silisocs_0_2_0/extracted/silisocs/environments/gm/components/resolve.py",
    "references/silisocs_0_2_0/extracted/silisocs/simulation_engines/policies/turns.py",
]
REFERENCE_DIRS = []
AUDIT_EXTRA_FILES = [
    "COMPETITION_REQUIREMENTS.md",
    "PRE_REVIEW.md",
    "RESEARCH_REDESIGN.md",
    "EXPERIMENT_AUDIT.md",
    "PAPER_CLAIM_AUDIT.md",
    "CITATION_AUDIT.md",
    "KILL_ARGUMENT.md",
    "SELF_REVIEW_LIMITATION.md",
]

EXCLUDE_PARTS = {
    "__pycache__",
    ".git",
}
EXCLUDE_SUFFIXES = {
    ".pyc", ".aux", ".log", ".out", ".toc", ".fls", ".fdb_latexmk",
}
RAW_DATA_MARKERS = [
    "socsim26_data",
    "beauty_contest_sweep.tar.gz",
    "iterated_pd_sweep.tar.gz",
    "observed_norms_sweep.tar.gz",
    "persona_expression_sweep.tar.gz",
    "polarization_sweep.tar.gz",
]

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest().upper()

def allowed(path: Path) -> bool:
    rel = path.relative_to(ROOT).as_posix()
    if any(marker in rel for marker in RAW_DATA_MARKERS):
        return False
    if any(part in EXCLUDE_PARTS for part in path.parts):
        return False
    if path.suffix in EXCLUDE_SUFFIXES:
        return False
    return path.is_file()

def add_file(zf: zipfile.ZipFile, path: Path, entries: list[str]) -> None:
    if not path.exists() or not allowed(path):
        return
    rel = path.relative_to(ROOT).as_posix()
    zf.write(path, rel)
    entries.append(rel)

def add_dir(zf: zipfile.ZipFile, rel_dir: str, entries: list[str]) -> None:
    root = ROOT / rel_dir
    if not root.exists():
        return
    for path in root.rglob("*"):
        add_file(zf, path, entries)

def build_zip(path: Path, include_audit: bool = False) -> dict:
    if path.exists():
        path.unlink()
    entries: list[str] = []
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for f in PUBLIC_ROOT_FILES + PAPER_FILES + REFERENCE_FILES:
            add_file(zf, ROOT / f, entries)
        for d in PUBLIC_DIRS + REFERENCE_DIRS:
            add_dir(zf, d, entries)
        if include_audit:
            for f in AUDIT_EXTRA_FILES:
                add_file(zf, ROOT / f, entries)
    bad = [e for e in entries if any(marker in e for marker in RAW_DATA_MARKERS)]
    if bad:
        raise RuntimeError(f"raw data leaked into package: {bad[:5]}")
    return {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256(path), "entries": len(entries)}

def main() -> None:
    PACKAGE_MANIFEST.write_text(json.dumps({
        "public_package_policy": "Includes final paper, supplement, analysis scripts, generated result tables, and lightweight official/pinned source files needed for parser/runtime source audits. Excludes official raw sweep tarballs and extracted run logs.",
        "suggested_repository_name": "socsim26-claim-execution-audit",
        "raw_data_excluded": True,
    }, indent=2), encoding="utf-8")
    public = build_zip(PUBLIC_ZIP, include_audit=False)
    # Include the package manifest itself in audit package by writing a temporary premanifest first.
    PACKAGE_MANIFEST.write_text(json.dumps({
        "public_package_policy": "Includes final paper, supplement, analysis scripts, generated result tables, and lightweight official/pinned source files needed for parser/runtime source audits. Excludes official raw sweep tarballs and extracted run logs.",
        "suggested_repository_name": "socsim26-claim-execution-audit",
        "raw_data_excluded": True,
        "public_package": public,
    }, indent=2), encoding="utf-8")
    audit = build_zip(AUDIT_ZIP, include_audit=True)
    pdf = ROOT / "paper" / "main_final.pdf"
    supp = ROOT / "paper" / "supplement_final.pdf"
    final = {
        "status": "NOT_SUBMISSION_READY",
        "scientific_validation": "PASS",
        "official_submission_blocker": "Permanent anonymized code/result URL or organizer-approved supplementary upload location not configured.",
        "suggested_repository_name": "socsim26-claim-execution-audit",
        "public_package": public,
        "audit_handoff_package": audit,
        "paper_pdf": {"path": "paper/main_final.pdf", "bytes": pdf.stat().st_size, "sha256": sha256(pdf)},
        "supplement_pdf": {"path": "paper/supplement_final.pdf", "bytes": supp.stat().st_size, "sha256": sha256(supp)},
        "validation_file": "artifacts_cross_study_final/PAPER_NUMBER_VALIDATION.json",
        "ledger": "CLAIM_EVIDENCE_LEDGER.csv",
    }
    FINAL_MANIFEST.write_text(json.dumps(final, indent=2, ensure_ascii=False), encoding="utf-8")
    PACKAGE_MANIFEST.write_text(json.dumps({
        "public_package_policy": "Includes final paper, supplement, analysis scripts, generated result tables, and lightweight official/pinned source files needed for parser/runtime source audits. Excludes official raw sweep tarballs and extracted run logs.",
        "suggested_repository_name": "socsim26-claim-execution-audit",
        "raw_data_excluded": True,
        "public_package": public,
        "audit_handoff_package": audit,
    }, indent=2), encoding="utf-8")
    print(json.dumps(final, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
