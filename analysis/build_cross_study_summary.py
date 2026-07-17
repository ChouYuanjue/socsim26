from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts_cross_study_final"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def claim_rows(path: Path, study: str, selected: list[str] | None = None) -> list[dict]:
    frame = pd.read_csv(path)
    if selected is not None:
        frame = frame[frame["claim_id"].isin(selected)]
    rows = []
    for _, row in frame.iterrows():
        cross = row.get("cross_cell_claim_status", row.get("cross_design_status", row.get("design_status", "")))
        rows.append({
            "study": study,
            "claim_id": row["claim_id"],
            "short_label": row["short_label"],
            "point_estimate": float(row["point_estimate"]),
            "ci_low": float(row["ci_low"]),
            "ci_high": float(row["ci_high"]),
            "target_verdict": row["target_verdict"],
            "design_verdict": cross,
            "source_file": path.relative_to(ROOT).as_posix(),
            "source_row_key": f"claim_id={row['claim_id']}",
        })
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    beauty = read_json(ROOT / "artifacts_final/summary.json")
    beauty_exec = read_json(ROOT / "artifacts_final/execution_fidelity_summary.json")
    ipd = read_json(ROOT / "artifacts_ipd_final/summary.json")
    persona = read_json(ROOT / "artifacts_persona_expression_final/summary.json")
    norms = read_json(ROOT / "artifacts_observed_norms_final/summary.json")
    polar = read_json(ROOT / "artifacts_polarization_final/summary.json")
    polar_runs = pd.read_csv(ROOT / "artifacts_polarization_final/run_level.csv")

    persona_model = pd.read_csv(ROOT / "artifacts_persona_expression_final/execution_fidelity_by_model.csv")
    polar_model = pd.read_csv(ROOT / "artifacts_polarization_final/execution_fidelity_by_model.csv")
    interface_rows = [
        {
            "study": "Beauty Contest",
            "runs": beauty["runs"],
            "interface": "free-text integer action",
            "model_decisions_or_probes": beauty_exec["agent_actions"],
            "executed_events": beauty_exec["agent_actions"],
            "execution_difference": beauty_exec["divergent_agent_actions"],
            "difference_rate": beauty_exec["divergent_action_fraction"],
            "difference_type": "released parser executes first legal integer rather than final number field",
            "affected_models": ";".join(beauty_exec["divergent_models"]),
            "scientific_impact": "target verdict changes: " + ";".join(beauty_exec["claims_with_target_verdict_change"]),
            "source_confirmed": True,
            "source_file": "artifacts_final/parser_source_audit.json",
        },
        {
            "study": "Iterated PD",
            "runs": ipd["runs"],
            "interface": "free-text categorical action",
            "model_decisions_or_probes": ipd["action_calls"],
            "executed_events": ipd["action_calls"],
            "execution_difference": ipd["divergent_action_fields"],
            "difference_rate": ipd["divergent_action_fields"] / ipd["action_calls"],
            "difference_type": "released parser executes earliest displayed label rather than final choice field",
            "affected_models": ";".join(ipd["divergent_models"]),
            "scientific_impact": "no target-verdict change; effect sensitivity retained",
            "source_confirmed": True,
            "source_file": "artifacts_ipd_final/parser_source_audit.json",
        },
        {
            "study": "Persona Expression",
            "runs": persona["runs"],
            "interface": "multi-tool social post",
            "model_decisions_or_probes": persona["model_decisions"],
            "executed_events": persona["executed_posts"],
            "execution_difference": persona["extra_posts"],
            "difference_rate": persona["extra_posts"] / persona["model_decisions"],
            "difference_type": "one model decision expands to multiple post events despite one-post measurement description",
            "affected_models": ";".join(persona_model.loc[persona_model["extra_posts"] > 0, "model"].tolist()),
            "scientific_impact": f"max per-run diversity shift {persona['maximum_absolute_run_diversity_shift']:.4f}; no selected target-verdict change",
            "source_confirmed": True,
            "source_file": "artifacts_persona_expression_final/measurement_unit_source_audit.json",
        },
        {
            "study": "Observed Norms",
            "runs": norms["runs"],
            "interface": "strict integer probe",
            "model_decisions_or_probes": norms["probe_calls"],
            "executed_events": norms["probe_calls"],
            "execution_difference": 0,
            "difference_rate": 0.0,
            "difference_type": "clean negative control: every response is exactly one in-range integer",
            "affected_models": "none",
            "scientific_impact": "execution-clean; three human-anchor claims remain data-unidentifiable",
            "source_confirmed": True,
            "source_file": "artifacts_observed_norms_final/parser_source_audit.json",
        },
        {
            "study": "Polarization",
            "runs": polar["runs"],
            "interface": "open-ended multi-tool turn plus integer probe",
            "model_decisions_or_probes": polar["tool_call_decisions"],
            "executed_events": int(polar_runs["executed_nonterminal_actions"].sum()),
            "execution_difference": polar["executed_events_above_cap"],
            "difference_rate": polar["executed_events_above_cap"] / max(1, int(polar_runs["executed_nonterminal_actions"].sum())),
            "difference_type": "batched calls execute before the advertised three-action cap is checked",
            "affected_models": ";".join(polar_model.loc[polar_model["executed_events_above_cap"] > 0, "model"].tolist()),
            "scientific_impact": (
                f"{polar['executed_overshoot_turns']:,} turns exceed cap; "
                f"{polar['unexecuted_generated_calls']} generated calls rejected; "
                f"{polar['missing_probe_outputs']} probes failed"
            ),
            "source_confirmed": True,
            "source_file": "artifacts_polarization_final/action_cap_source_audit.json",
        },
    ]
    interfaces = pd.DataFrame(interface_rows)
    interfaces.to_csv(OUT / "execution_interface_audit.csv", index=False)

    selected = []
    selected += claim_rows(ROOT / "artifacts_final/hypothesis_audit.csv", "Beauty Contest", ["H2", "H3", "H5b"])
    selected += claim_rows(ROOT / "artifacts_ipd_final/hypothesis_audit.csv", "Iterated PD", ["P1b", "P2b", "P6a", "P6b"])
    selected += claim_rows(ROOT / "artifacts_persona_expression_final/hypothesis_audit.csv", "Persona Expression", ["E1a", "E1b", "E1c", "E3b"])
    selected += claim_rows(ROOT / "artifacts_observed_norms_final/identifiable_hypothesis_audit.csv", "Observed Norms", ["N3"])
    selected += claim_rows(ROOT / "artifacts_polarization_final/hypothesis_audit.csv", "Polarization", None)
    claims = pd.DataFrame(selected)
    claims.to_csv(OUT / "selected_claim_audit.csv", index=False)

    totals = {
        "studies": 5,
        "runs": int(interfaces["runs"].sum()),
        "structured_probe_calls": int(norms["probe_calls"] + polar["probe_calls"]),
        "game_action_calls": int(beauty_exec["agent_actions"] + ipd["action_calls"]),
        "social_tool_decisions": int(persona["model_decisions"] + polar["tool_call_decisions"]),
        "free_text_action_field_divergences": int(beauty_exec["divergent_agent_actions"] + ipd["divergent_action_fields"]),
        "persona_extra_post_events": int(persona["extra_posts"]),
        "polarization_executed_events_above_cap": int(polar["executed_events_above_cap"]),
        "polarization_overshoot_turns": int(polar["executed_overshoot_turns"]),
        "polarization_unexecuted_generated_calls": int(polar["unexecuted_generated_calls"]),
        "polarization_missing_probes": int(polar["missing_probe_outputs"]),
    }
    summary = {
        "method": "Cross-study Claim-Identifiable and Execution-Fidelity Audit",
        "totals": totals,
        "execution_taxonomy": [
            "greedy value substitution in free-text game actions",
            "decision-to-many-event expansion in multi-tool posting",
            "post-resolution open-ended cap overshoot and rejected generated calls",
            "clean strict-integer probe control",
        ],
        "interface_table": interface_rows,
        "selected_claims": selected,
        "scope": "All findings are confined to the released sweeps and pinned runtime/source versions. Counterfactual social trajectories are not imputed when execution truncation would require rerunning the simulation.",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    tex_rows = []
    for _, row in interfaces.iterrows():
        difference = f"{int(row['execution_difference']):,}" if int(row["execution_difference"]) else "0"
        tex_rows.append(
            f"{row['study']} & {int(row['runs']):,} & {row['interface']} & {difference} & {row['affected_models'].replace(';', ', ')} \\\\"
        )
    tex = "\n".join([
        r"\begin{tabular}{lrlrl}",
        r"\toprule",
        r"Study & runs & interface & exec. diff. & affected models \\",
        r"\midrule",
        *tex_rows,
        r"\bottomrule",
        r"\end{tabular}",
    ])
    (OUT / "execution_table.tex").write_text(tex, encoding="utf-8")
    print(json.dumps(totals, indent=2))


if __name__ == "__main__":
    main()
