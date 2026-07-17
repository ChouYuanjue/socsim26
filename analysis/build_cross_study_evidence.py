from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper" / "main_final.tex"
PDF = ROOT / "paper" / "main_final.pdf"
LEDGER = ROOT / "CLAIM_EVIDENCE_LEDGER.csv"
VALIDATION = ROOT / "artifacts_cross_study_final" / "PAPER_NUMBER_VALIDATION.json"

TARBALLS = {
    "beauty_contest": "62e80580beaa63af652942ecd1d2953640c34e7606539a91a6fc455ee4a34f95",
    "iterated_pd": "8a5590c95b8ace0f4d6552fa9885d0ac7dbbc918a485cb1667d91da779f69365",
    "observed_norms": "686e324ae66bbc6f7a5f22aea4766819d13b9dc00c3fd866992782983f18e2bf",
    "persona_expression": "6e8d28894d7deb73fa6c8ccd92c2270a78111f92a86e1487c6fea9ab5c0c16f8",
    "polarization": "b506d9e84d765fd1e66bb0d19c01a1ef572e52b85c085616508c5c2b7d5d96e2",
}
MANIFESTS = {
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


def pdf_pages(path: Path) -> int:
    pdfinfo = shutil.which("pdfinfo")
    if pdfinfo:
        out = subprocess.run([pdfinfo, str(path)], capture_output=True, text=True, check=True).stdout
        match = re.search(r"^Pages:\s+(\d+)", out, flags=re.MULTILINE)
        if match:
            return int(match.group(1))
    try:
        import fitz  # type: ignore
        with fitz.open(str(path)) as doc:
            return doc.page_count
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"Could not determine PDF page count: {exc}") from exc


def add(rows: list[dict[str, str]], claim_id: str, location: str, description: str, value: object,
        displayed: str, source_file: str, selector: str, token: str | None = None,
        rounding: str = "as displayed") -> None:
    rows.append({
        "claim_id": claim_id,
        "paper_location": location,
        "description": description,
        "unrounded_value": str(value),
        "displayed_value": displayed,
        "generating_script": "analysis/build_cross_study_summary.py plus per-study audit scripts",
        "raw_input": "official socsim26 released sweep tarballs; WVS anchors not redistributed",
        "result_file": source_file,
        "exact_selector": selector,
        "rounding_rule": rounding,
        "paper_token": token or displayed,
        "validation_status": "pending",
    })


def main(skip_source_integrity: bool = False, validation_output: Path | None = None) -> None:
    tex = PAPER.read_text(encoding="utf-8")
    summary = json.loads((ROOT / "artifacts_cross_study_final" / "summary.json").read_text(encoding="utf-8"))
    interfaces = pd.read_csv(ROOT / "artifacts_cross_study_final" / "execution_interface_audit.csv")
    selected = pd.read_csv(ROOT / "artifacts_cross_study_final" / "selected_claim_audit.csv")
    rows: list[dict[str, str]] = []

    totals = summary["totals"]
    add(rows, "TOTAL-STUDIES", "Abstract", "released sweeps audited", totals["studies"], "five", "artifacts_cross_study_final/summary.json", "totals.studies", "five")
    add(rows, "TOTAL-RUNS", "Abstract", "released runs audited", totals["runs"], "5,213", "artifacts_cross_study_final/summary.json", "totals.runs", "5,213")
    add(rows, "GAME-DIFF", "Abstract", "free-text game action divergences", totals["free_text_action_field_divergences"], "288", "artifacts_cross_study_final/summary.json", "totals.free_text_action_field_divergences", "288")
    add(rows, "PERSONA-EXTRA", "Abstract/Table 1", "extra Persona Expression post events", totals["persona_extra_post_events"], "2,289", "artifacts_cross_study_final/summary.json", "totals.persona_extra_post_events", "2,289")
    add(rows, "POLAR-ABOVE-CAP", "Abstract/Table 1", "Polarization executed events above advertised cap", totals["polarization_executed_events_above_cap"], "60,634", "artifacts_cross_study_final/summary.json", "totals.polarization_executed_events_above_cap", "60,634")
    add(rows, "STRICT-PROBE", "Abstract/Table 1", "Observed Norms strict integer discrepancies", 0, "0", "artifacts_observed_norms_final/summary.json", "strict_integer_fidelity == 1.0", "0")

    interface_tokens = {
        "Beauty Contest": ("670", "230/1,340", "H2 and H6b"),
        "Iterated PD": ("3,465", "58/69,300", "target verdicts do not"),
        "Persona Expression": ("528", "+2,289", ".078"),
        "Observed Norms": ("330", "0/36,720", "WVS anchors"),
        "Polarization": ("220", "+60,634", "686-action"),
    }
    for _, r in interfaces.iterrows():
        study = r["study"]
        run_token, diff_token, consequence_token = interface_tokens[study]
        add(rows, f"{study}-RUNS", "Table 1", f"{study} runs", int(r["runs"]), run_token, "artifacts_cross_study_final/execution_interface_audit.csv", f"study={study},runs", run_token)
        add(rows, f"{study}-DIFF", "Table 1", f"{study} execution-interface difference", r["execution_difference"], diff_token, "artifacts_cross_study_final/execution_interface_audit.csv", f"study={study},execution_difference", diff_token)
        add(rows, f"{study}-CONSEQ", "Table 1", f"{study} consequence phrase", r["scientific_impact"], consequence_token, "artifacts_cross_study_final/execution_interface_audit.csv", f"study={study},scientific_impact", consequence_token)

    claim_tokens = {
        ("Beauty Contest", "H2"): ("1.55", "[.38,2.79]", "mixed"),
        ("Iterated PD", "P1b"): (".252", "[.181,.324]", "positive in 9/9 cells"),
        ("Persona Expression", "E1a"): (".081", "[.050,.112]", "\\Upt"),
        ("Observed Norms", "N3"): (".572", "[.224,.965]", "--"),
        ("Polarization", "Z1"): ("-.032", "[-.053,-.012]", "nonpositive in 5/5 cells"),
        ("Polarization", "Z4"): (".170", "[.114,.232]", "positive in 5/5 cells"),
    }
    for (study, claim), (effect_token, ci_token, design_token) in claim_tokens.items():
        r = selected.loc[(selected["study"] == study) & (selected["claim_id"] == claim)].iloc[0]
        low, high = ci_token.strip("[]").split(",")
        add(rows, f"{claim}-E", "Table 2", f"{study} {claim} effect", r["point_estimate"], effect_token, r["source_file"], f"{r['source_row_key']},point_estimate", effect_token)
        add(rows, f"{claim}-L", "Table 2", f"{study} {claim} CI lower", r["ci_low"], low, r["source_file"], f"{r['source_row_key']},ci_low", low)
        add(rows, f"{claim}-U", "Table 2", f"{study} {claim} CI upper", r["ci_high"], high, r["source_file"], f"{r['source_row_key']},ci_high", high)
        if design_token not in {"\\Upt", "--"}:
            add(rows, f"{claim}-D", "Table 2", f"{study} {claim} design token", r["design_verdict"], design_token, r["source_file"], f"{r['source_row_key']},design_verdict", design_token)

    required_phrases = [
        "Execution Before Evaluation",
        "signed claim contrasts",
        "5,213 runs",
        "288 game actions",
        "2,289 extra post events",
        "60,634 social events",
        "Observed Norms",
        "Polarization",
        "WVS-fidelity claims are",
        "do not infer counterfactual societies",
    ]
    missing_phrases = [p for p in required_phrases if p not in tex]

    missing_tokens: list[dict[str, str]] = []
    for row in rows:
        token = row["paper_token"]
        if token not in tex:
            missing_tokens.append({"claim_id": row["claim_id"], "paper_token": token})
            row["validation_status"] = "missing_from_tex"
        else:
            row["validation_status"] = "verified"

    pages = pdf_pages(PDF)
    source_integrity: dict[str, dict[str, object]] = {}
    data_root = ROOT / "references" / "socsim26_sharedtask" / "socsim26_data"
    for study, expected in TARBALLS.items():
        tar = data_root / f"{study}_sweep.tar.gz"
        manifest = data_root / study / "sweeps" / "manifest.csv"
        if skip_source_integrity:
            source_integrity[study] = {
                "mode": "skipped",
                "reason": "official raw sweep tarballs are intentionally not redistributed in the public repository",
                "tarball_sha256_expected": expected,
                "manifest_sha256_expected": MANIFESTS[study],
            }
        else:
            source_integrity[study] = {
                "mode": "checked",
                "tarball_exists": tar.exists(),
                "tarball_sha256": sha256(tar) if tar.exists() else None,
                "tarball_sha256_expected": expected,
                "tarball_ok": tar.exists() and sha256(tar) == expected,
                "manifest_exists": manifest.exists(),
                "manifest_sha256": sha256(manifest) if manifest.exists() else None,
                "manifest_sha256_expected": MANIFESTS[study],
                "manifest_ok": manifest.exists() and sha256(manifest) == MANIFESTS[study],
            }

    status = "PASS"
    source_ok = skip_source_integrity or all(v["tarball_ok"] and v["manifest_ok"] for v in source_integrity.values())
    if pages != 2 or missing_tokens or missing_phrases or not source_ok:
        status = "FAIL"

    with LEDGER.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    validation = {
        "status": status,
        "paper": str(PAPER.relative_to(ROOT)),
        "pdf": str(PDF.relative_to(ROOT)),
        "pdf_pages": pages,
        "ledger_rows": len(rows),
        "verified_rows": sum(row["validation_status"] == "verified" for row in rows),
        "missing_tokens": missing_tokens,
        "missing_required_phrases": missing_phrases,
        "source_integrity": source_integrity,
        "cross_study_totals": totals,
        "scope": summary["scope"],
    }
    output_path = validation_output or VALIDATION
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(validation, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(validation, indent=2, ensure_ascii=False))
    if status != "PASS":
        raise SystemExit("cross-study evidence validation failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate main-paper numbers against checked-in result tables.")
    parser.add_argument(
        "--skip-source-integrity",
        action="store_true",
        help="Skip official raw tarball/manifest checks. Use this for a clean public GitHub clone before downloading the official data.",
    )
    parser.add_argument(
        "--validation-output",
        type=Path,
        default=None,
        help="Optional path for the validation JSON. Defaults to artifacts_cross_study_final/PAPER_NUMBER_VALIDATION.json.",
    )
    args = parser.parse_args()
    main(skip_source_integrity=args.skip_source_integrity, validation_output=args.validation_output)
