from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
OUT = PAPER / "supplement_final.tex"


def esc(value: object) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


def fmt(value: object, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "--"
    number = float(value)
    text = f"{number:.{digits}f}"
    if text.startswith("0."):
        text = text[1:]
    elif text.startswith("-0."):
        text = "-." + text[3:]
    return text


def verdict(value: object) -> str:
    mapping = {
        "supported": "S",
        "contradicted": "C",
        "mixed_or_inconclusive": "M",
        "not_identifiable": "U",
    }
    return mapping.get(str(value), esc(value))


def claim_table(path: Path, study: str, design_column: str) -> str:
    df = pd.read_csv(path)
    rows = []
    for _, row in df.iterrows():
        design = row.get(design_column, "")
        levels = row.get("identifiable_nondefault_levels", "")
        if pd.isna(design):
            design = ""
        design_text = str(design).replace("_", " ")
        if levels != "" and not pd.isna(levels):
            design_text += f" ({int(levels)} levels)"
        rows.append(
            f"{esc(study)} & {esc(row['claim_id'])} & {esc(row['short_label'])} & "
            f"{fmt(row['point_estimate'])} & [{fmt(row['ci_low'])}, {fmt(row['ci_high'])}] & "
            f"{verdict(row['target_verdict'])} & {esc(design_text)} \\\\"
        )
    return "\n".join(rows)


def execution_projection_table(path: Path, study: str, left: str, right: str) -> str:
    df = pd.read_csv(path)
    rows = []
    for _, row in df.iterrows():
        left_point = row[f"point_estimate_{left}"]
        right_point = row[f"point_estimate_{right}"]
        left_verdict = row[f"target_verdict_{left}"]
        right_verdict = row[f"target_verdict_{right}"]
        changed = "yes" if str(left_verdict) != str(right_verdict) else "no"
        rows.append(
            f"{esc(study)} & {esc(row['claim_id'])} & {fmt(left_point)} & {verdict(left_verdict)} & "
            f"{fmt(right_point)} & {verdict(right_verdict)} & {changed} \\\\"
        )
    return "\n".join(rows)


def source_rows() -> str:
    beauty = json.loads((ROOT / "artifacts_final/parser_source_audit.json").read_text(encoding="utf-8"))
    ipd = json.loads((ROOT / "artifacts_ipd_final/parser_source_audit.json").read_text(encoding="utf-8"))
    norms = json.loads((ROOT / "artifacts_observed_norms_final/parser_source_audit.json").read_text(encoding="utf-8"))
    persona = json.loads((ROOT / "artifacts_persona_expression_final/measurement_unit_source_audit.json").read_text(encoding="utf-8"))
    polar = json.loads((ROOT / "artifacts_polarization_final/action_cap_source_audit.json").read_text(encoding="utf-8"))

    def sha_for(data: dict, suffix: str) -> str:
        item = next(item for item in data["files"] if item["path"].endswith(suffix))
        return item["sha256"][:12] + r"\ldots"

    rows = [
        ("Beauty Contest", f"game.py:{beauty['start_line']}--{beauty['end_line']}", beauty["sha256"][:12] + r"\ldots", "returns the first legal integer in the entire response"),
        ("Iterated PD", f"game.py:{ipd['functions']['parse_pd_choice']['start_line']}--{ipd['functions']['parse_pd_choice']['end_line']}", ipd["sha256"][:12] + r"\ldots", "returns the earliest displayed cooperation/defection label"),
        ("Observed Norms", f"norms.py:{norms['start_line']}--{norms['end_line']}", norms["sha256"][:12] + r"\ldots", "parses the first in-range integer; all released outputs are exact integers"),
        ("Persona Expression", "README + world + sim", sha_for(persona, "scenario/sim.yaml"), "measurement description says one post per agent-step while tool_calling.mode is multi"),
        ("Persona Expression", "action_prompts.py + resolve.py", sha_for(persona, "resolve.py"), "prompt permits batches and resolver executes every normalized call"),
        ("Persona Expression", "turns.py + SiliSocS 0.2.0", sha_for(persona, "silisocs-0.2.0-py3-none-any.whl"), "pins the one-agent-step policy and exact runtime wheel"),
        ("Polarization", "README + sim", sha_for(polar, "scenario/sim.yaml"), "advertises up to three calls and configures max_actions: 3"),
        ("Polarization", "action_prompts.py + resolve.py", sha_for(polar, "resolve.py"), "an entire multi-call batch is resolved before control returns"),
        ("Polarization", "turns.py + SiliSocS 0.2.0", sha_for(polar, "silisocs-0.2.0-py3-none-any.whl"), "the open-ended counter is updated after batch resolution"),
    ]
    return "\n".join(
        f"{esc(study)} & {esc(component)} & \texttt{{{sha}}} & {esc(implication)} " + r"\\"
        for study, component, sha, implication in rows
    )


def build() -> str:
    cross = json.loads((ROOT / "artifacts_cross_study_final/summary.json").read_text(encoding="utf-8"))
    interfaces = pd.read_csv(ROOT / "artifacts_cross_study_final/execution_interface_audit.csv")
    interface_rows = []
    for _, row in interfaces.iterrows():
        interface_rows.append(
            f"{esc(row['study'])} & {int(row['runs']):,} & {esc(row['interface'])} & "
            f"{int(row['model_decisions_or_probes']):,} & {int(row['executed_events']):,} & "
            f"{int(row['execution_difference']):,} & {esc(row['scientific_impact'])} \\\\"
        )

    beauty_claims = claim_table(ROOT / "artifacts_final/hypothesis_audit.csv", "Beauty", "cross_cell_claim_status")
    ipd_claims = claim_table(ROOT / "artifacts_ipd_final/hypothesis_audit.csv", "IPD", "cross_cell_claim_status")
    persona_claims = claim_table(ROOT / "artifacts_persona_expression_final/hypothesis_audit.csv", "Persona", "cross_design_status")
    norms_claims = claim_table(ROOT / "artifacts_observed_norms_final/identifiable_hypothesis_audit.csv", "Norms", "")
    polar_claims = claim_table(ROOT / "artifacts_polarization_final/hypothesis_audit.csv", "Polarization", "cross_design_status")

    beauty_projection = execution_projection_table(
        ROOT / "artifacts_final/claim_execution_sensitivity.csv", "Beauty", "recorded", "explicit"
    )
    ipd_projection = execution_projection_table(
        ROOT / "artifacts_ipd_final/claim_execution_sensitivity.csv", "IPD", "recorded", "explicit"
    )
    persona_projection = execution_projection_table(
        ROOT / "artifacts_persona_expression_final/claim_execution_sensitivity.csv", "Persona", "executed", "strict"
    )

    nonident = pd.read_csv(ROOT / "artifacts_observed_norms_final/nonidentifiable_official_claims.csv")
    nonident_rows = "\n".join(
        f"Observed Norms & {esc(str(row['official_parent']).replace('_', ' '))} & U & {esc(row['reason'])} \\\\"
        for _, row in nonident.iterrows()
    )

    data_hash_rows = []
    for study, artifact in [
        ("Beauty Contest", ROOT / "artifacts_final/ARTIFACT_MANIFEST.json"),
        ("Iterated PD", ROOT / "artifacts_ipd_final/ARTIFACT_MANIFEST.json"),
        ("Persona Expression", ROOT / "artifacts_persona_expression_final/ARTIFACT_MANIFEST.json"),
        ("Observed Norms", ROOT / "artifacts_observed_norms_final/ARTIFACT_MANIFEST.json"),
        ("Polarization", ROOT / "artifacts_polarization_final/ARTIFACT_MANIFEST.json"),
    ]:
        data = json.loads(artifact.read_text(encoding="utf-8"))
        tar_input = next(item for item in data["inputs"] if item["path"].endswith(".tar.gz"))
        data_hash_rows.append(
            f"{esc(study)} & \texttt{{{esc(Path(tar_input['path']).name)}}} & {int(tar_input['bytes']):,} & "
            f"\texttt{{{tar_input['sha256'][:12]}\\ldots}} \\\\"
        )

    persona_summary = json.loads((ROOT / "artifacts_persona_expression_final/summary.json").read_text(encoding="utf-8"))
    polar_summary = json.loads((ROOT / "artifacts_polarization_final/summary.json").read_text(encoding="utf-8"))
    beauty_exec = json.loads((ROOT / "artifacts_final/execution_fidelity_summary.json").read_text(encoding="utf-8"))
    ipd_summary = json.loads((ROOT / "artifacts_ipd_final/summary.json").read_text(encoding="utf-8"))
    norms_summary = json.loads((ROOT / "artifacts_observed_norms_final/summary.json").read_text(encoding="utf-8"))

    return rf"""\documentclass[10pt]{{article}}
\usepackage[letterpaper,margin=0.72in]{{geometry}}
\usepackage{{amsmath,amssymb,booktabs,longtable,microtype,array,enumitem,xcolor,pdflscape}}
\usepackage[hidelinks]{{hyperref}}
\setlength{{\parskip}}{{0.28em}}
\setlength{{\parindent}}{{0em}}
\renewcommand{{\arraystretch}}{{1.05}}
\title{{\vspace{{-1.2em}}Supplement: Execution Before Evaluation in LLM Social Simulations}}
\author{{Anonymous Shared-Task Submission}}
\date{{}}
\begin{{document}}
\maketitle
\vspace{{-1.7em}}

\section{{Scope and contribution}}
The main paper reports a five-study claim-identifiable audit over {cross['totals']['runs']:,} released runs. This supplement contains the complete atomic-claim tables, execution projections, identifiability decisions, source provenance, and reproduction contract. A positive signed effect supports the registered direction. Target verdicts are S (supported), C (contradicted), M (mixed/inconclusive), and U (unidentified). Design verdicts concern only non-default cells in which treatment and reference arms can be matched; absence of such cells is not interpreted as stability.

\section{{Method}}
For a paired claim $h$, treatment $T_h$ and reference $R_h$ are matched at the same seed and every remaining signal variable. Its signed run effect is
\[
\tau_h = s_h\,[m(T_h)-m(R_h)], \qquad s_h\in\{{-1,+1\}},
\]
where the sign is chosen so that $\tau_h>0$ supports the registered hypothesis. One-arm hypotheses use $\tau_h=m-\theta_h$. Let $c$ index matched context strata. The reported point estimate is $|C|^{{-1}}\sum_c \bar\tau_{{h,c}}$, so contexts with denser grids do not dominate. The interval is the envelope of two 95\% sensitivity intervals: resample contexts then seeds within context, and resample the released seed labels. Bootstrap draws are computational replicates, not independent observations.

A design level $d$ is identifiable for $h$ only if both claim arms exist under $d$, every other design variable is at default, and the matching keys above remain available. We separately report: (i) the hypothesis direction in the non-default cell, (ii) the matched-default direction on the same context subset, and (iii) the incremental design shift. No global max/max ratio is used.

The execution gate has four interface-specific forms: exact replay of released free-text parsers; generated tool-call to event replay; source inspection of the pinned runtime's batching and turn policy; and strict-format probes as negative controls. Recorded events remain authoritative. Final-field or one-decision projections are sensitivity analyses, not claims about latent model intent.

\section{{Released evidence}}
\begin{{landscape}}
\small
\begin{{longtable}}{{@{{}}l r p{{3.2cm}} r r r p{{6.4cm}}@{{}}}}
\caption{{Cross-study execution evidence. Difference has the interface-specific meaning in the final column.}}\\
\toprule
Study & runs & interface & decisions/probes & events & difference & scientific impact \\
\midrule
\endfirsthead
\toprule
Study & runs & interface & decisions/probes & events & difference & scientific impact \\
\midrule
\endhead
{chr(10).join(interface_rows)}
\bottomrule
\end{{longtable}}
\end{{landscape}}

The free-text game discrepancies total {cross['totals']['free_text_action_field_divergences']:,} actions. Beauty contributes {beauty_exec['divergent_agent_actions']:,}/{beauty_exec['agent_actions']:,}, changing target verdicts for {', '.join(beauty_exec['claims_with_target_verdict_change'])}; IPD contributes {ipd_summary['divergent_action_fields']:,}/{ipd_summary['action_calls']:,} with no target-verdict change. Persona Expression executes {persona_summary['extra_posts']:,} posts beyond its {persona_summary['model_decisions']:,} model decisions; the one-decision/one-post projection changes per-run Jaccard diversity by at most {persona_summary['maximum_absolute_run_diversity_shift']:.4f}. In Polarization, {polar_summary['executed_overshoot_turns']:,} agent-turns exceed the three-action cap by {polar_summary['executed_events_above_cap']:,} actual events; the largest executed turn contains {polar_summary['maximum_executed_events_in_agent_episode']:,} events. The same sweep has {polar_summary['unexecuted_generated_calls']} generated calls without events and {polar_summary['missing_probe_outputs']} failed probes. Observed Norms is the clean control: all {norms_summary['probe_calls']:,} responses are exactly one in-range integer.

\section{{All atomic claims}}
\begin{{landscape}}
\small
\begin{{longtable}}{{@{{}}llp{{6.0cm}}rclp{{5.2cm}}@{{}}}}
\caption{{All auditable atomic hypotheses. Intervals are conditional on released seeds.}}\\
\toprule
Study & ID & claim & effect & 95\% CI & target & identifiable-design verdict \\
\midrule
\endfirsthead
\toprule
Study & ID & claim & effect & 95\% CI & target & identifiable-design verdict \\
\midrule
\endhead
{beauty_claims}
{ipd_claims}
{persona_claims}
{norms_claims}
{polar_claims}
\bottomrule
\end{{longtable}}
\end{{landscape}}

\subsection{{Unavailable registered claims}}
\small
\begin{{tabular}}{{@{{}}lp{{3.7cm}}lp{{8.6cm}}@{{}}}}
\toprule
Study & parent claim & verdict & reason \\
\midrule
{nonident_rows}
\bottomrule
\end{{tabular}}

\section{{Execution-projection sensitivity}}
\begin{{landscape}}
\small
\begin{{longtable}}{{@{{}}llrrrrl@{{}}}}
\caption{{Target sensitivity to the alternative observable projection. For games, ``explicit'' is the prompt-specified final action field. For Persona, ``strict'' retains the first post from each model decision.}}\\
\toprule
Study & claim & recorded/executed & verdict & alternative & verdict & changed? \\
\midrule
\endfirsthead
\toprule
Study & claim & recorded/executed & verdict & alternative & verdict & changed? \\
\midrule
\endhead
{beauty_projection}
{ipd_projection}
{persona_projection}
\bottomrule
\end{{longtable}}
\end{{landscape}}

Only the Beauty H2 and H6b target verdicts change. This distinction matters: an execution discrepancy can be real without overturning a particular scientific conclusion. Conversely, agreement of a target verdict does not validate the interface; it only bounds the consequence for that claim and released sweep.

\section{{Source-backed mechanisms}}
\begin{{landscape}}
\small
\begin{{longtable}}{{@{{}}lp{{5.0cm}}p{{3.2cm}}p{{10.0cm}}@{{}}}}
\caption{{Source and runtime provenance. Hashes are abbreviated here and complete in artifact manifests.}}\\
\toprule
Study & file & SHA-256 & audit use \\
\midrule
\endfirsthead
\toprule
Study & file & SHA-256 & audit use \\
\midrule
\endhead
{source_rows()}
\bottomrule
\end{{longtable}}
\end{{landscape}}

The released Beauty parser scans the full response and returns the first standalone integer in $[11,20]$. The released IPD parser searches both displayed labels and returns whichever occurs earlier. Persona Expression combines a one-post-per-agent-step measurement description with multi-tool prompting and a resolver that executes every normalized call. Polarization's open-ended policy updates its action counter only after an entire batch has been resolved; therefore a single response can cross the cap before the stopping condition is reevaluated. These are source-replayed statements for the pinned files above, not inferred latent mechanisms.

\section{{Input provenance}}
\small
\begin{{tabular}}{{@{{}}lp{{5.5cm}}rp{{3.5cm}}@{{}}}}
\toprule
Study & official tarball & bytes & SHA-256 \\
\midrule
{chr(10).join(data_hash_rows)}
\bottomrule
\end{{tabular}}

All analyses read compressed JSONL members directly from the official tarballs. This avoids Windows long-path truncation and records the exact tar member in run- or action-level tables. Manifests, tarballs, parser files, runtime wheel, generated artifacts, and paper files are hashed in machine-readable manifests.

\section{{Interpretive limits}}
\begin{{enumerate}}[leftmargin=1.4em,itemsep=0.15em]
\item Intervals are conditional on the released seeds and sweep support. They do not quantify uncertainty over providers, future model versions, or an open-ended population of prompts.
\item An execution discrepancy is not automatically a changed scientific verdict. The projection tables report when it matters for the registered contrast.
\item We do not rerun counterfactual societies after replacing parsers, truncating batches, or restoring rejected calls. Such interventions would change subsequent histories and require new simulations.
\item Persona lexical diversity is a transparent token-set Jaccard metric used for execution sensitivity, not a universal measure of persona expression quality.
\item Polarization edge alignment is a direct network-neighbor stance-similarity statistic derived from the released graph and opinion probes. It is not a causal estimate of exposure policy.
\item Human-anchor claims in Observed Norms remain U because the licensed WVS reference data are not distributed. No surrogate anchor is invented.
\end{{enumerate}}

\section{{Reproduction contract}}
The public entry point runs unit tests, regenerates all five study artifact directories, rebuilds the cross-study summary, compiles the two-page paper and this supplement, validates the claim--evidence ledger, and checks page limits. Each study script fails on missing tar members, parser replay failure, malformed structured outputs, or unmatched claim arms rather than silently skipping them. Polarization is the exception only in the scientifically appropriate sense: failed probes and rejected calls are retained as explicit data-quality fields, while affected claim contrasts drop only rows whose required metric is undefined.

\begin{{thebibliography}}{{9}}
\small
\bibitem{{ju2024}} D.~Ju, A.~Williams, B.~Karrer, and M.~Nickel. Sense and sensitivity: Evaluating the simulation of social dynamics via LLMs. arXiv:2412.05093, 2024.
\bibitem{{linde2026}} M.~Linde, J.~Sun, P.~Balluff, D.~Radovanovi\'c, and C.-h.~Chan. Making uncertainty visible: Multiverse analysis for robust computational social science. arXiv:2605.19745, 2026.
\bibitem{{sarangi2026}} S.~Sarangi, M.~P.~Touzel, A.~B\"uck-Kaeffer, Z.~Yang, J.-F.~Godbout, and R.~Rabbany. EASE configuration facilitates a reproducible science of LLM social simulations. arXiv:2605.30258, 2026.
\bibitem{{ye2026}} J.~Ye, L.~Cao, D.~Chen, and E.~Ferrara. Stop drawing scientific claims from LLM social simulations without robustness audits. arXiv:2605.18890, 2026.
\bibitem{{manyworlds2026}} Many Worlds organizers. COLM 2026 Shared Task on Evaluating LLM Social Simulations. Official task page and repository, 2026.
\end{{thebibliography}}
\end{{document}}
"""


if __name__ == "__main__":
    PAPER.mkdir(parents=True, exist_ok=True)
    OUT.write_text(build(), encoding="utf-8")
    print(OUT)
