"""Render Paper 68 CSV evidence into LaTeX assets and audit summaries."""

from __future__ import annotations

import csv
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
PAPER = ROOT / "paper"
GENERATED = PAPER / "generated"
DOCS = ROOT / "docs"

PROPOSED = "ebm_transformer_compositional_v5"
OLD = "ebm_transformer_compositional"

METHOD_ORDER = [
    "random_grasp",
    "antipodal_geometry",
    "force_closure_score",
    "collision_aware_force_closure",
    "cem_grasp_search",
    "robust_perturbation_ranker",
    "mlp_energy_model",
    "gradient_boosted_energy_model",
    "transformer_policy_ranker",
    "ensemble_uncertainty_ranker",
    "calibrated_stacked_ranker",
    OLD,
    PROPOSED,
    "oracle_mujoco_grid",
]

SELECTED_METHODS = [
    "force_closure_score",
    "collision_aware_force_closure",
    "cem_grasp_search",
    "robust_perturbation_ranker",
    "mlp_energy_model",
    "calibrated_stacked_ranker",
    OLD,
    PROPOSED,
    "oracle_mujoco_grid",
]

HOSTILE_SPLITS = [
    "combined_composition_shift",
    "long_bar_high_aspect",
    "thin_handle_shift",
    "adversarial_clutter_gap",
    "material_shift_proxy",
]

LABELS = {
    "random_grasp": "Random",
    "antipodal_geometry": "Antipodal",
    "force_closure_score": "Force closure",
    "collision_aware_force_closure": "Collision-aware FC",
    "cem_grasp_search": "CEM analytic",
    "robust_perturbation_ranker": "Robust perturb.",
    "mlp_energy_model": "MLP energy",
    "gradient_boosted_energy_model": "HGB energy",
    "transformer_policy_ranker": "Transformer",
    "ensemble_uncertainty_ranker": "Ensemble",
    "calibrated_stacked_ranker": "Calibrated stack",
    OLD: "Old EBM",
    PROPOSED: "EBM v5",
    "oracle_mujoco_grid": "Oracle grid",
    "full_ebm_transformer_compositional_v5": "Full v5",
    "no_object_energy": "No object",
    "no_contact_energy": "No contact",
    "no_task_energy": "No task",
    "no_collision_energy": "No collision",
    "no_feasibility_energy": "No feasibility",
    "no_transformer_context": "No transformer",
    "no_robust_perturbation": "No robust",
    "no_calibrated_stack": "No stack",
    "no_cem_fallback": "No fallback",
    "no_hard_collision_filter": "No hard filter",
    "old_v4_compositional": "Old v4",
    "monolithic_scalar_energy_only": "Scalar only",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def count_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        return sum(1 for _ in reader)


def f3(value: object) -> str:
    return f"{float(value):.3f}"


def esc(text: object) -> str:
    out = str(text)
    return (
        out.replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("$", "\\$")
        .replace("#", "\\#")
        .replace("_", "\\_")
        .replace("{", "\\{")
        .replace("}", "\\}")
        .replace("~", "\\textasciitilde{}")
        .replace("^", "\\textasciicircum{}")
    )


def label(name: str) -> str:
    return LABELS.get(name, name.replace("_", " "))


def method_rank(name: str) -> int:
    return METHOD_ORDER.index(name) if name in METHOD_ORDER else len(METHOD_ORDER)


def write(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")


def parse_summary() -> dict[str, str]:
    text = (RESULTS / "summary.txt").read_text(encoding="utf-8")
    out: dict[str, str] = {}
    decision = re.search(r"Terminal decision:\s*(.+)", text)
    reason = re.search(r"Terminal reason:\s*(.+)", text)
    counts = re.search(r"Main rows:\s*(\d+);\s*ablation rows:\s*(\d+);\s*stress rows:\s*(\d+)", text)
    training = re.search(r"Training rows:\s*(\d+);", text)
    out["decision"] = decision.group(1).strip() if decision else "UNKNOWN"
    out["reason"] = reason.group(1).strip() if reason else "missing"
    out["training_rows"] = training.group(1) if training else str(count_rows(RESULTS / "training_rollouts.csv"))
    if counts:
        out["main_rows"], out["ablation_rows"], out["stress_rows"] = counts.groups()
    else:
        out["main_rows"] = str(count_rows(RESULTS / "ebm_grasping_raw.csv"))
        out["ablation_rows"] = str(count_rows(RESULTS / "ebm_grasping_ablation_raw.csv"))
        out["stress_rows"] = str(count_rows(RESULTS / "stress_sweep.csv"))
    return out


def render_macros() -> str:
    summary = parse_summary()
    return "\n".join(
        [
            f"\\newcommand{{\\PaperDecision}}{{{esc(summary['decision'])}}}",
            f"\\newcommand{{\\PaperDecisionReason}}{{{esc(summary['reason'])}}}",
            f"\\newcommand{{\\TrainingRows}}{{{summary['training_rows']}}}",
            f"\\newcommand{{\\MainRows}}{{{summary['main_rows']}}}",
            f"\\newcommand{{\\AblationRows}}{{{summary['ablation_rows']}}}",
            f"\\newcommand{{\\StressRows}}{{{summary['stress_rows']}}}",
            f"\\newcommand{{\\SeedMetricRows}}{{{count_rows(RESULTS / 'raw_seed_metrics.csv')}}}",
            "",
        ]
    )


def render_aggregate(rows: list[dict[str, str]]) -> str:
    body = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Aggregate frozen results across all grasping shifts. Success is higher-is-better; drop, collision, unsafe force, and energy are lower-is-better.}",
        "\\label{tab:aggregate-main}",
        "\\scriptsize",
        "\\begin{tabular}{lrrrrrrr}",
        "\\toprule",
        "Method & Episodes & Success & Slip & Drop & Collision & Unsafe & Energy \\\\",
        "\\midrule",
    ]
    for row in sorted(rows, key=lambda r: method_rank(r["method"])):
        body.append(
            f"{esc(label(row['method']))} & {row['episodes']} & {f3(row['mean_success'])} & "
            f"{f3(row['mean_slip'])} & {f3(row['drop_rate'])} & {f3(row['collision_rate'])} & "
            f"{f3(row['unsafe_force_rate'])} & {f3(row['mean_energy'])} \\\\"
        )
    body += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(body)


def render_selected_splits(metrics: list[dict[str, str]]) -> str:
    by_key = {(row["split"], row["method"]): row for row in metrics}
    body = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Hostile split success rates. These are the splits most likely to falsify compositional grasping claims.}",
        "\\label{tab:selected-splits}",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{2.0pt}",
        "\\begin{tabular}{lrrrrrrrrr}",
        "\\toprule",
        "Split & FC & Coll-FC & CEM & Robust & MLP & Stack & Old & v5 & Oracle \\\\",
        "\\midrule",
    ]
    for split in HOSTILE_SPLITS:
        vals = [f3(by_key[(split, method)]["mean_success"]) for method in SELECTED_METHODS]
        body.append(f"{esc(split)} & " + " & ".join(vals) + " \\\\")
    body += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(body)


def render_ablation(rows: list[dict[str, str]]) -> str:
    body = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Aggregate hostile ablations over the three pre-registered hostile ablation splits.}",
        "\\label{tab:ablation-aggregate}",
        "\\scriptsize",
        "\\begin{tabular}{lrrrrr}",
        "\\toprule",
        "Variant & Episodes & Success & Drop & Collision & Energy \\\\",
        "\\midrule",
    ]
    ordered = sorted(rows, key=lambda r: (-float(r["mean_success"]), label(r["method"])))
    for row in ordered:
        body.append(
            f"{esc(label(row['method']))} & {row['episodes']} & {f3(row['mean_success'])} & "
            f"{f3(row['drop_rate'])} & {f3(row['collision_rate'])} & {f3(row['mean_energy'])} \\\\"
        )
    body += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(body)


def render_stress(rows: list[dict[str, str]]) -> str:
    body = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Stress sweep success rates. Stress jointly changes friction, clutter clearance, aspect ratio, sensor noise, and task constraint strength.}",
        "\\label{tab:stress}",
        "\\scriptsize",
        "\\begin{tabular}{lrrrrrr}",
        "\\toprule",
        "Method & 0.0 & 0.2 & 0.4 & 0.6 & 0.8 & 1.0 \\\\",
        "\\midrule",
    ]
    by_key = {(row["method"], f"{float(row['stress_level']):.1f}"): row for row in rows}
    methods = [
        "cem_grasp_search",
        "collision_aware_force_closure",
        "robust_perturbation_ranker",
        "mlp_energy_model",
        "gradient_boosted_energy_model",
        "calibrated_stacked_ranker",
        PROPOSED,
        "oracle_mujoco_grid",
    ]
    for method in methods:
        vals = []
        for level in ["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"]:
            vals.append(f3(by_key[(method, level)]["mean_success"]))
        body.append(f"{esc(label(method))} & " + " & ".join(vals) + " \\\\")
    body += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(body)


def render_negative_cases(rows: list[dict[str, str]]) -> str:
    body = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Pre-specified negative cases beyond the local benchmark scope.}",
        "\\label{tab:negative-cases}",
        "\\scriptsize",
        "\\begin{tabular}{p{0.20\\linewidth}p{0.34\\linewidth}p{0.34\\linewidth}}",
        "\\toprule",
        "Case & Observed failure mode & Submission implication \\\\",
        "\\midrule",
    ]
    for row in rows:
        body.append(
            f"{esc(row['case'])} & {esc(row['observed_failure_mode'])} & {esc(row['submission_implication'])} \\\\"
        )
    body += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(body)


def longtable_metrics(rows: list[dict[str, str]], caption: str, label_name: str) -> str:
    body = [
        "{\\scriptsize",
        "\\setlength{\\tabcolsep}{2.5pt}",
        f"\\begin{{longtable}}{{llrrrrrr}}",
        f"\\caption{{{esc(caption)}}}\\label{{{label_name}}}\\\\",
        "\\toprule",
        "Split & Method & Episodes & Success & Slip & Drop & Collision & Energy \\\\",
        "\\midrule",
        "\\endfirsthead",
        "\\toprule",
        "Split & Method & Episodes & Success & Slip & Drop & Collision & Energy \\\\",
        "\\midrule",
        "\\endhead",
    ]
    for row in sorted(rows, key=lambda r: (r.get("split", ""), method_rank(r["method"]))):
        body.append(
            f"{esc(row.get('split', 'all'))} & {esc(label(row['method']))} & {row['episodes']} & "
            f"{f3(row['mean_success'])} & {f3(row['mean_slip'])} & {f3(row['drop_rate'])} & "
            f"{f3(row['collision_rate'])} & {f3(row['mean_energy'])} \\\\"
        )
    body += ["\\bottomrule", "\\end{longtable}", "}", ""]
    return "\n".join(body)


def longtable_pairwise(rows: list[dict[str, str]]) -> str:
    body = [
        "{\\scriptsize",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\begin{longtable}{llrrr}",
        "\\caption{All split-level paired seed comparisons versus EBM v5. Positive values favor EBM v5.}\\label{tab:full-pairwise}\\\\",
        "\\toprule",
        "Split & Baseline & Diff & t approx. & p approx. \\\\",
        "\\midrule",
        "\\endfirsthead",
        "\\toprule",
        "Split & Baseline & Diff & t approx. & p approx. \\\\",
        "\\midrule",
        "\\endhead",
    ]
    for row in rows:
        body.append(
            f"{esc(row['split'])} & {esc(label(row['baseline']))} & {f3(row['mean_success_diff_vs_ebm'])} & "
            f"{f3(row['paired_t_approx'])} & {f3(row['normal_approx_p'])} \\\\"
        )
    body += ["\\bottomrule", "\\end{longtable}", "}", ""]
    return "\n".join(body)


def write_terminal_doc() -> None:
    summary = parse_summary()
    text = "\n".join(
        [
            "# Paper 68 Expanded Terminal Decision",
            "",
            f"Decision: {summary['decision']}",
            "",
            f"Reason: {summary['reason']}",
            "",
            f"Training rows: {summary['training_rows']}",
            f"Main rows: {summary['main_rows']}",
            f"Ablation rows: {summary['ablation_rows']}",
            f"Stress rows: {summary['stress_rows']}",
            "",
            "This decision is generated from frozen CSV artifacts, not hand-transcribed table values.",
            "",
        ]
    )
    write(DOCS / "paper68_expanded_terminal_decision_20260620.md", text)


def main() -> None:
    GENERATED.mkdir(parents=True, exist_ok=True)
    metrics = read_csv(RESULTS / "ebm_grasping_metrics.csv")
    aggregate = read_csv(RESULTS / "aggregate_metrics.csv")
    ablation = read_csv(RESULTS / "ablation_aggregate_metrics.csv")
    ablation_full = read_csv(RESULTS / "ebm_grasping_ablation.csv")
    stress = read_csv(RESULTS / "stress_sweep.csv")
    pairwise = read_csv(RESULTS / "pairwise_stats.csv")
    negatives = read_csv(RESULTS / "negative_cases.csv")
    seed_rows = read_csv(RESULTS / "raw_seed_metrics.csv")

    write(GENERATED / "result_macros.tex", render_macros())
    write(GENERATED / "aggregate_metrics_table.tex", render_aggregate(aggregate))
    write(GENERATED / "selected_split_table.tex", render_selected_splits(metrics))
    write(GENERATED / "ablation_table.tex", render_ablation(ablation))
    write(GENERATED / "stress_table.tex", render_stress(stress))
    write(GENERATED / "negative_cases_table.tex", render_negative_cases(negatives))
    write(GENERATED / "full_metrics_longtable.tex", longtable_metrics(metrics, "Full split-level main metrics.", "tab:full-metrics"))
    write(GENERATED / "full_ablation_longtable.tex", longtable_metrics(ablation_full, "Full hostile split ablation metrics.", "tab:full-ablation"))
    write(GENERATED / "full_stress_longtable.tex", longtable_metrics(stress, "Full stress-sweep metrics.", "tab:full-stress"))
    write(GENERATED / "full_pairwise_longtable.tex", longtable_pairwise(pairwise))
    write(
        GENERATED / "seed_metrics_selected_longtable.tex",
        longtable_metrics(
            [row for row in seed_rows if row["method"] in {"cem_grasp_search", "robust_perturbation_ranker", PROPOSED, "oracle_mujoco_grid"}],
            "Selected seed-level metrics for reproducibility.",
            "tab:seed-metrics",
        ),
    )
    write(
        GENERATED / "all_seed_metrics_longtable.tex",
        longtable_metrics(
            seed_rows,
            "All method/split/seed metrics from the frozen run.",
            "tab:all-seed-metrics",
        ),
    )
    write_terminal_doc()
    print(f"Rendered Paper 68 generated assets in {GENERATED}")


if __name__ == "__main__":
    main()
