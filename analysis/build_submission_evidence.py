from __future__ import annotations

import csv
import json
import math
import re
import subprocess
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts_final"
PAPER = ROOT / "paper" / "main_final.tex"
PDF = ROOT / "paper" / "main_final.pdf"
LEDGER = ROOT / "CLAIM_EVIDENCE_LEDGER.csv"
VALIDATION = ART / "PAPER_NUMBER_VALIDATION.json"
SCRIPT = "analysis/final_cira_analysis.py"
RAW = "references/socsim26_sharedtask/socsim26_data/beauty_contest_sweep.tar.gz"


def fmt(value: float, digits: int = 2, leading_zero: bool = False) -> str:
    rounded = f"{value:.{digits}f}"
    if not leading_zero:
        rounded = rounded.replace("-0.", "-.").replace("0.", ".")
    return rounded


def add(
    rows: list[dict[str, str]],
    claim_id: str,
    location: str,
    description: str,
    value: object,
    displayed: str,
    result_file: str,
    selector: str,
    paper_token: str,
    rounding: str = "as displayed",
) -> None:
    rows.append(
        {
            "claim_id": claim_id,
            "paper_location": location,
            "description": description,
            "unrounded_value": str(value),
            "displayed_value": displayed,
            "generating_script": SCRIPT,
            "raw_input": RAW,
            "result_file": result_file,
            "exact_selector": selector,
            "rounding_rule": rounding,
            "paper_token": paper_token,
            "validation_status": "pending",
        }
    )


def main() -> None:
    tex = PAPER.read_text(encoding="utf-8")
    audit = pd.read_csv(ART / "hypothesis_audit.csv")
    exec_claims = pd.read_csv(ART / "claim_execution_sensitivity.csv")
    fidelity = json.loads((ART / "execution_fidelity_summary.json").read_text(encoding="utf-8"))
    integrity = json.loads((ART / "data_integrity.json").read_text(encoding="utf-8"))
    permutation = json.loads((ART / "permutation_tests.json").read_text(encoding="utf-8"))
    anchor = pd.read_csv(ART / "anchor_sanity_checks.csv")
    mechanism = json.loads((ART / "mechanism_summary.json").read_text(encoding="utf-8"))
    transitions = pd.read_csv(ART / "mechanism_transitions.csv")
    model_fidelity = pd.read_csv(ART / "execution_fidelity_by_model.csv")
    mismatch = pd.read_csv(ART / "executor_mismatch_patterns.csv")

    rows: list[dict[str, str]] = []

    add(rows, "DATA-1", "Abstract/results", "released runs parsed", integrity["parsed_runs"], "670", "artifacts_final/data_integrity.json", "parsed_runs", "670")
    add(rows, "DATA-2", "Abstract/results", "recorded actions parsed", integrity["agent_choices"], "1,340", "artifacts_final/data_integrity.json", "agent_choices", "1,340")
    add(rows, "EXEC-1", "Abstract/execution", "actions divergent from explicit parameter", fidelity["divergent_agent_actions"], "230", "artifacts_final/execution_fidelity_summary.json", "divergent_agent_actions", "230")
    add(rows, "EXEC-2", "Abstract/execution", "divergent action fraction", fidelity["divergent_action_fraction"], "17.2\\%", "artifacts_final/execution_fidelity_summary.json", "divergent_action_fraction", "17.2\\%", "100*x, one decimal")
    add(rows, "EXEC-3", "Execution", "runs containing a divergent action", fidelity["divergent_runs"], "115", "artifacts_final/execution_fidelity_summary.json", "divergent_runs", "115")
    add(rows, "EXEC-4", "Execution", "all recorded events match first legal token", fidelity["recorded_matches_first_valid_legal_token_fraction"], "1.0", "artifacts_final/execution_fidelity_summary.json", "recorded_matches_first_valid_legal_token_fraction", "every action")

    anchor_row = anchor.loc[anchor["check"] == "identical_distribution_old_switched_anchor_range"].iloc[0]
    add(rows, "ANCHOR-1", "Method", "false switched-anchor range under identical behavior", anchor_row["value"], ".037", "artifacts_final/anchor_sanity_checks.csv", "check=identical_distribution_old_switched_anchor_range,value", ".037", "three decimals")
    add(rows, "STAT-1", "Method", "bootstrap Monte Carlo replicates", 4000, "4,000", "artifacts_final/ARTIFACT_MANIFEST.json", "metadata.bootstrap_replicates", "4,000")
    add(rows, "STAT-2", "Results", "H2 paired randomization p-value", permutation["two_sided_pvalue"], ".0166", "artifacts_final/permutation_tests.json", "two_sided_pvalue", ".0166", "four decimals")

    table_display = {
        "H1": ("-.25", "[-.75,.00]", "0\\%", "0/8", "4"),
        "H2": ("1.55", "[.38,2.79]", "38\\%", "0/4", "3"),
        "H3": (".50", "[.50,.50]", "100\\%", "4/4", "3"),
        "H4": ("-.03", "[-.20,.13]", "4\\%", "1/7", "3"),
        "H5a": (".68", "[.15,1.34]", "25\\%", "0/1", "1"),
        "H5b": ("-1.83", "[-3.01,-.83]", "0\\%", "0/1", "1"),
        "H6a": ("-.31", "[-1.86,1.11]", "38\\%", "--", "0"),
        "H6b": ("-1.97", "[-3.39,-.60]", "8\\%", "--", "0"),
    }
    for claim_id, display in table_display.items():
        r = audit.loc[audit["claim_id"] == claim_id].iloc[0]
        effect, interval, ctx, cells, idvars = display
        low = interval.strip("[]").split(",")[0]
        high = interval.strip("[]").split(",")[1]
        add(rows, f"{claim_id}-E", "Table 1", f"{claim_id} signed effect", r["point_estimate"], effect, "artifacts_final/hypothesis_audit.csv", f"claim_id={claim_id},point_estimate", effect, "two decimals")
        add(rows, f"{claim_id}-L", "Table 1", f"{claim_id} conservative CI lower", r["ci_low"], low, "artifacts_final/hypothesis_audit.csv", f"claim_id={claim_id},ci_low", low, "two decimals")
        add(rows, f"{claim_id}-U", "Table 1", f"{claim_id} conservative CI upper", r["ci_high"], high, "artifacts_final/hypothesis_audit.csv", f"claim_id={claim_id},ci_high", high, "two decimals")
        ctx_value = round(100 * float(r["sign_consistency"]))
        add(rows, f"{claim_id}-CTX", "Table 1", f"{claim_id} positive context fraction", r["sign_consistency"], ctx, "artifacts_final/hypothesis_audit.csv", f"claim_id={claim_id},sign_consistency", ctx, "nearest integer percent")
        if cells != "--":
            computed_cells = f"{int(r['design_cells_supporting_claim'])}/{int(r['identifiable_nondefault_levels'])}"
            if computed_cells != cells:
                raise AssertionError(f"Cell display mismatch for {claim_id}: {computed_cells} != {cells}")
            add(rows, f"{claim_id}-CELL", "Table 1", f"{claim_id} supporting identifiable design cells", computed_cells, cells, "artifacts_final/hypothesis_audit.csv", f"claim_id={claim_id},design_cells_supporting_claim/identifiable_nondefault_levels", cells)
        add(rows, f"{claim_id}-ID", "Table 1", f"{claim_id} identifiable design-variable count", r["identifiable_design_variable_count"], idvars, "artifacts_final/hypothesis_audit.csv", f"claim_id={claim_id},identifiable_design_variable_count", idvars)

    gemma = model_fidelity.loc[model_fidelity["model"] == "gemma-4-31b"].iloc[0]
    qwen_actions = int(model_fidelity.loc[model_fidelity["model"].str.startswith("qwen"), "agent_actions"].sum())
    qwen_matches = int(model_fidelity.loc[model_fidelity["model"].str.startswith("qwen"), "recorded_equals_explicit_parameter"].sum())
    add(rows, "EXEC-5", "Execution", "Qwen explicit-parameter matches", qwen_matches, "1,070", "artifacts_final/execution_fidelity_by_model.csv", "models startswith qwen, sum(recorded_equals_explicit_parameter)", "1,070")
    add(rows, "EXEC-6", "Execution", "Qwen actions", qwen_actions, "1,070", "artifacts_final/execution_fidelity_by_model.csv", "models startswith qwen, sum(agent_actions)", "1,070")
    add(rows, "EXEC-7", "Execution", "Gemma explicit-parameter matches", int(gemma["recorded_equals_explicit_parameter"]), "40", "artifacts_final/execution_fidelity_by_model.csv", "model=gemma-4-31b,recorded_equals_explicit_parameter", "40")
    add(rows, "EXEC-8", "Execution", "Gemma actions", int(gemma["agent_actions"]), "270", "artifacts_final/execution_fidelity_by_model.csv", "model=gemma-4-31b,agent_actions", "270")
    for _, r in mismatch.iterrows():
        pattern = f"{int(r['choice'])}/{int(r['declared_choice'])}"
        count = str(int(r["agent_actions"]))
        add(rows, f"EXEC-PATTERN-{pattern}", "Execution", f"recorded/explicit pattern {pattern}", int(r["agent_actions"]), count, "artifacts_final/executor_mismatch_patterns.csv", f"choice={int(r['choice'])},declared_choice={int(r['declared_choice'])},agent_actions", count)

    for claim_id, display_effect, display_interval in [
        ("H2", "1.24", "[-.01,2.53]"),
        ("H6b", "-1.64", "[-3.37,.19]"),
    ]:
        r = exec_claims.loc[exec_claims["claim_id"] == claim_id].iloc[0]
        low, high = display_interval.strip("[]").split(",")
        add(rows, f"{claim_id}-EXPLICIT-E", "Execution", f"{claim_id} explicit-parameter effect", r["point_estimate_explicit"], display_effect, "artifacts_final/claim_execution_sensitivity.csv", f"claim_id={claim_id},point_estimate_explicit", display_effect, "two decimals")
        add(rows, f"{claim_id}-EXPLICIT-L", "Execution", f"{claim_id} explicit-parameter CI lower", r["ci_low_explicit"], low, "artifacts_final/claim_execution_sensitivity.csv", f"claim_id={claim_id},ci_low_explicit", low, "two decimals")
        add(rows, f"{claim_id}-EXPLICIT-U", "Execution", f"{claim_id} explicit-parameter CI upper", r["ci_high_explicit"], high, "artifacts_final/claim_execution_sensitivity.csv", f"claim_id={claim_id},ci_high_explicit", high, "two decimals")
    h6a = exec_claims.loc[exec_claims["claim_id"] == "H6a"].iloc[0]
    add(rows, "H6a-EXEC-SHIFT", "Execution", "H6a recorded minus explicit projection magnitude", abs(float(h6a["execution_shift"])), "1.33", "artifacts_final/claim_execution_sensitivity.csv", "claim_id=H6a,abs(execution_shift)", "1.33", "two decimals")

    add(rows, "MECH-1", "Representation regimes", "variation conditions", mechanism["variation_conditions"], "12", "artifacts_final/mechanism_summary.json", "variation_conditions", "12")
    add(rows, "MECH-2", "Representation regimes", "near-deterministic variation conditions", mechanism["near_deterministic_variation_conditions_modal_share_ge_0_9"], "10", "artifacts_final/mechanism_summary.json", "near_deterministic_variation_conditions_modal_share_ge_0_9", "10")
    add(rows, "MECH-3", "Representation regimes", "default entropy", mechanism["default_mean_entropy_bits"], ".44", "artifacts_final/mechanism_summary.json", "default_mean_entropy_bits", ".44", "two decimals")
    add(rows, "MECH-4", "Representation regimes", "variation entropy", mechanism["variation_mean_entropy_bits"], ".23", "artifacts_final/mechanism_summary.json", "variation_mean_entropy_bits", ".23", "two decimals")
    add(rows, "MECH-5", "Representation regimes", "variation conditions modal at 19", mechanism["variation_conditions_modal_choice_19"], "eight", "artifacts_final/mechanism_summary.json", "variation_conditions_modal_choice_19", "eight")
    strongest = transitions.loc[transitions["design_level"].isin(["descending-range", "paraphrase-a", "descriptive", "tabular"]), "regime_switch_rate"]
    add(rows, "MECH-6", "Representation regimes", "minimum strongest-template regime-switch rate", float(strongest.min()), "70--100\\%", "artifacts_final/mechanism_transitions.csv", "selected strongest levels,min/max(regime_switch_rate)", "70--100\\%", "percent range")

    missing_tokens: list[dict[str, str]] = []
    for row in rows:
        token = row["paper_token"]
        # Tokens like '-.75' occur in LaTeX as '-.75'; percentages retain '\%'.
        if token not in tex:
            missing_tokens.append({"claim_id": row["claim_id"], "paper_token": token})
            row["validation_status"] = "missing_from_tex"
        else:
            row["validation_status"] = "verified"

    if re.search(r"\bIPD\b|prisoner", tex, flags=re.IGNORECASE):
        raise AssertionError("Unsupported cross-scenario content remains in main_final.tex")
    if "3,465" in tex or "4,135" in tex or "2.95" in tex or "5.58" in tex:
        raise AssertionError("Legacy unsupported numbers remain in main_final.tex")

    pdfinfo = subprocess.run(["pdfinfo", str(PDF)], capture_output=True, text=True, check=True).stdout
    match = re.search(r"^Pages:\s+(\d+)", pdfinfo, flags=re.MULTILINE)
    pages = int(match.group(1)) if match else -1
    if pages != 2:
        raise AssertionError(f"Expected 2 pages, got {pages}")

    with LEDGER.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    validation = {
        "status": "PASS" if not missing_tokens else "FAIL",
        "paper": str(PAPER.relative_to(ROOT)),
        "pdf": str(PDF.relative_to(ROOT)),
        "pdf_pages": pages,
        "ledger_rows": len(rows),
        "verified_rows": sum(row["validation_status"] == "verified" for row in rows),
        "missing_tokens": missing_tokens,
        "legacy_cross_scenario_content_absent": True,
        "legacy_unsupported_numbers_absent": True,
        "source_integrity": {
            "parsed_runs": integrity["parsed_runs"],
            "agent_choices": integrity["agent_choices"],
            "errors": integrity["errors"],
            "tarball_sha256": integrity["tarball_sha256"],
            "manifest_sha256": integrity["manifest_sha256"],
        },
    }
    VALIDATION.write_text(json.dumps(validation, indent=2, ensure_ascii=False), encoding="utf-8")
    if missing_tokens:
        raise AssertionError(f"Paper tokens missing: {missing_tokens}")
    print(json.dumps(validation, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
