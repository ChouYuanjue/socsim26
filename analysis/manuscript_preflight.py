from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper" / "main_final.tex"
SUPP = ROOT / "paper" / "supplement_final.tex"
OUT = ROOT / "artifacts_cross_study_final" / "MANUSCRIPT_PREFLIGHT.json"


def contains_any(text: str, needles: list[str]) -> list[str]:
    return [needle for needle in needles if needle in text]


def norm_number(x: float, digits: int) -> str:
    s = f"{x:.{digits}f}"
    if s.startswith("0."):
        s = s[1:]
    if s.startswith("-0."):
        s = "-." + s[3:]
    return s


def main() -> None:
    main_tex = PAPER.read_text(encoding="utf-8")
    supp_tex = SUPP.read_text(encoding="utf-8") if SUPP.exists() else ""
    cross = json.loads((ROOT / "artifacts_cross_study_final/summary.json").read_text(encoding="utf-8"))
    selected = pd.read_csv(ROOT / "artifacts_cross_study_final/selected_claim_audit.csv")
    interfaces = pd.read_csv(ROOT / "artifacts_cross_study_final/execution_interface_audit.csv")

    findings: list[dict[str, object]] = []

    # Style checks aimed at common AI-draft artifacts.
    if "—" in main_tex or "–" in main_tex:
        findings.append({"severity": "fail", "issue": "unicode_dash_in_main"})
    if main_tex.count("---") > 0:
        findings.append({"severity": "warn", "issue": "latex_em_dash_in_main", "count": main_tex.count("---")})
    if "**" in main_tex:
        findings.append({"severity": "fail", "issue": "markdown_bold_in_latex"})
    if main_tex.count("\\textbf{") > 1:
        findings.append({"severity": "warn", "issue": "excessive_bold_commands", "count": main_tex.count("\\textbf{")})
    title_match = re.search(r"\\title\{(.+?)\}", main_tex, flags=re.S)
    if title_match and ":" in title_match.group(1):
        findings.append({"severity": "warn", "issue": "colon_in_title"})
    abstract = re.search(r"\\begin\{abstract\}(.+?)\\end\{abstract\}", main_tex, flags=re.S)
    if not abstract:
        findings.append({"severity": "fail", "issue": "missing_abstract"})
    elif "\\paragraph" in abstract.group(1) or "\\begin{itemize}" in abstract.group(1):
        findings.append({"severity": "fail", "issue": "segmented_abstract"})
    if "\\cira" in main_tex:
        findings.append({"severity": "fail", "issue": "self_coined_acronym_in_main", "token": "\\cira"})
    if re.search(r"\bCIRA\b", main_tex):
        findings.append({"severity": "fail", "issue": "self_coined_acronym_in_main", "token": "CIRA"})
    if contains_any(main_tex, ["ground truth", "privileged intent", "oracle"]):
        findings.append({"severity": "fail", "issue": "ground_truth_or_oracle_language"})
    if re.search(r"[A-Za-z]:\\|/home/|artifacts_[A-Za-z0-9_/-]+|references/[A-Za-z0-9_/-]+", main_tex):
        findings.append({"severity": "fail", "issue": "raw_path_in_main_text"})

    # Numeric consistency checks against generated artifacts.
    totals = cross["totals"]
    expected_tokens = {
        "5,213": totals["runs"],
        "288": totals["free_text_action_field_divergences"],
        "2,289": totals["persona_extra_post_events"],
        "60,634": totals["polarization_executed_events_above_cap"],
        "36,720": int(interfaces.loc[interfaces["study"] == "Observed Norms", "model_decisions_or_probes"].iloc[0]),
        "23,327": totals["polarization_overshoot_turns"],
    }
    for token in expected_tokens:
        if token not in main_tex:
            findings.append({"severity": "fail", "issue": "missing_main_number", "token": token})

    if totals["runs"] != int(interfaces["runs"].sum()):
        findings.append({"severity": "fail", "issue": "run_total_mismatch", "total": totals["runs"], "sum": int(interfaces["runs"].sum())})
    if totals["free_text_action_field_divergences"] != int(interfaces.loc[interfaces["study"].isin(["Beauty Contest", "Iterated PD"]), "execution_difference"].sum()):
        findings.append({"severity": "fail", "issue": "free_text_divergence_total_mismatch"})

    claim_checks = {
        ("Beauty Contest", "H2"): (2, "1.55", ".38", "2.79"),
        ("Iterated PD", "P1b"): (3, ".252", ".181", ".324"),
        ("Persona Expression", "E1a"): (3, ".081", ".050", ".112"),
        ("Observed Norms", "N3"): (3, ".572", ".224", ".965"),
        ("Polarization", "Z1"): (3, "-.032", "-.053", "-.012"),
        ("Polarization", "Z4"): (3, ".170", ".114", ".232"),
    }
    for (study, claim), (digits, point_token, low_token, high_token) in claim_checks.items():
        row = selected.loc[(selected["study"] == study) & (selected["claim_id"] == claim)].iloc[0]
        computed = (norm_number(float(row["point_estimate"]), digits), norm_number(float(row["ci_low"]), digits), norm_number(float(row["ci_high"]), digits))
        expected = (point_token, low_token, high_token)
        if computed != expected:
            findings.append({"severity": "fail", "issue": "claim_rounding_mismatch", "study": study, "claim": claim, "computed": computed, "expected": expected})
        for token in expected:
            if token not in main_tex:
                findings.append({"severity": "fail", "issue": "claim_token_missing_main", "study": study, "claim": claim, "token": token})
            # The supplement may report more decimal places than the two-page main paper.
            # Require the claim row/source to be present, not the main-paper rounded token.
            if supp_tex and claim not in supp_tex:
                findings.append({"severity": "fail", "issue": "claim_missing_supplement", "study": study, "claim": claim})

    # Reporting completeness checks.
    for phrase in [
        "without treating missing design cells as evidence",
        "recorded events remain authoritative",
        "not claims about latent model intent",
        "WVS anchors",
        "do not infer counterfactual societies",
    ]:
        joined = (main_tex + "\n" + supp_tex).lower()
        if phrase.lower() not in joined:
            findings.append({"severity": "warn", "issue": "missing_scope_phrase", "phrase": phrase})

    status = "PASS" if not any(f["severity"] == "fail" for f in findings) else "FAIL"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "status": status,
        "checks": {
            "unicode_dash_absent_main": "—" not in main_tex and "–" not in main_tex,
            "self_coined_acronym_absent_main": "\\cira" not in main_tex and not re.search(r"\bCIRA\b", main_tex),
            "abstract_single_block": bool(abstract and "\\paragraph" not in abstract.group(1) and "\\begin{itemize}" not in abstract.group(1)),
            "raw_paths_absent_main": not bool(re.search(r"[A-Za-z]:\\|/home/|artifacts_[A-Za-z0-9_/-]+|references/[A-Za-z0-9_/-]+", main_tex)),
            "run_total_matches_interface_table": totals["runs"] == int(interfaces["runs"].sum()),
            "free_text_divergence_total_matches": totals["free_text_action_field_divergences"] == int(interfaces.loc[interfaces["study"].isin(["Beauty Contest", "Iterated PD"]), "execution_difference"].sum()),
        },
        "findings": findings,
    }
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if status != "PASS":
        raise SystemExit("manuscript preflight failed")


if __name__ == "__main__":
    main()
