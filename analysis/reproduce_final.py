from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "references" / "socsim26_sharedtask" / "socsim26_data"
EXPECTED_TAR_SHA256 = {
    "beauty_contest": "62e80580beaa63af652942ecd1d2953640c34e7606539a91a6fc455ee4a34f95",
    "iterated_pd": "8a5590c95b8ace0f4d6552fa9885d0ac7dbbc918a485cb1667d91da779f69365",
    "observed_norms": "686e324ae66bbc6f7a5f22aea4766819d13b9dc00c3fd866992782983f18e2bf",
    "persona_expression": "6e8d28894d7deb73fa6c8ccd92c2270a78111f92a86e1487c6fea9ab5c0c16f8",
    "polarization": "b506d9e84d765fd1e66bb0d19c01a1ef572e52b85c085616508c5c2b7d5d96e2",
}
EXPECTED_MANIFEST_SHA256 = {
    "beauty_contest": "7370f6dde54f4497ff3a63d81fc196c28d2cb25b06a29e386be5d45255e0d26c",
    "iterated_pd": "d9d51bc54136583c05761853d69cea023b77614a2816b473c6626658a6cba335",
    "observed_norms": "e6b01c59a754b0eedbd2a47df162b53e61de3b4d798c3d5a134fabd78fe5a6d4",
    "persona_expression": "d197bf081ce2663c1587529eb1020b82e939c1ae832b644b255ec5a80f07ef58",
    "polarization": "0b11c3df2a161d870e78112e2874baf62361e34991c2fa963573f461ff8647fa",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def run(command: list[str], cwd: Path = ROOT) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def verify_data() -> None:
    for study, expected in EXPECTED_TAR_SHA256.items():
        tar = DATA / f"{study}_sweep.tar.gz"
        manifest = DATA / study / "sweeps" / "manifest.csv"
        if not tar.exists() or not manifest.exists():
            raise SystemExit(f"Missing released data for {study}; see REPRODUCIBILITY_FINAL.md")
        actual_tar = sha256(tar)
        actual_manifest = sha256(manifest)
        if actual_tar != expected:
            raise SystemExit(f"{study} tarball checksum mismatch: {actual_tar}")
        if actual_manifest != EXPECTED_MANIFEST_SHA256[study]:
            raise SystemExit(f"{study} manifest checksum mismatch: {actual_manifest}")


def main() -> None:
    verify_data()

    run([sys.executable, "-m", "unittest", "-v", "test_final_cira_analysis.py", "test_cross_study_audits.py"], ROOT / "analysis")
    run([sys.executable, str(ROOT / "analysis" / "final_cira_analysis.py"), "--bootstrap-replicates", "4000", "--permutations", "20000", "--seed", "20260716"])
    run([sys.executable, str(ROOT / "analysis" / "ipd_cira_analysis.py"), "--bootstrap-replicates", "4000", "--seed", "20260717"])
    run([sys.executable, str(ROOT / "analysis" / "observed_norms_audit.py"), "--bootstrap-replicates", "4000", "--seed", "20260718"])
    run([sys.executable, str(ROOT / "analysis" / "persona_expression_execution_audit.py"), "--bootstrap-replicates", "4000", "--seed", "20260718"])
    run([sys.executable, str(ROOT / "analysis" / "polarization_cira_analysis.py"), "--bootstrap-replicates", "4000", "--seed", "20260720"])
    run([sys.executable, str(ROOT / "analysis" / "build_cross_study_summary.py")])
    run([sys.executable, str(ROOT / "analysis" / "build_supplement.py")])

    pdflatex = shutil.which("pdflatex")
    if pdflatex is None:
        raise SystemExit("pdflatex is required to reproduce paper/main_final.pdf")
    run([pdflatex, "-interaction=nonstopmode", "-halt-on-error", "main_final.tex"], ROOT / "paper")
    run([pdflatex, "-interaction=nonstopmode", "-halt-on-error", "main_final.tex"], ROOT / "paper")
    run([pdflatex, "-interaction=nonstopmode", "-halt-on-error", "supplement_final.tex"], ROOT / "paper")
    run([pdflatex, "-interaction=nonstopmode", "-halt-on-error", "supplement_final.tex"], ROOT / "paper")
    run([sys.executable, str(ROOT / "analysis" / "build_cross_study_evidence.py")])

    validation = json.loads((ROOT / "artifacts_cross_study_final" / "PAPER_NUMBER_VALIDATION.json").read_text(encoding="utf-8"))
    if validation["status"] != "PASS" or validation["pdf_pages"] != 2:
        raise SystemExit(f"Submission evidence gate failed: {validation}")
    print(json.dumps({
        "status": "PASS",
        "studies": validation["cross_study_totals"]["studies"],
        "runs": validation["cross_study_totals"]["runs"],
        "paper_pages": validation["pdf_pages"],
        "supplement_pdf": str((ROOT / "paper" / "supplement_final.pdf").relative_to(ROOT)),
        "ledger_rows": validation["ledger_rows"],
    }, indent=2))


if __name__ == "__main__":
    main()
