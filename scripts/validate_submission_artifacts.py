"""Validate Paper 68 expanded-standard artifacts."""

from __future__ import annotations

import csv
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
PAPER = ROOT / "paper"
DOWNLOADS = Path.home() / "Downloads"
DESKTOP = Path.home() / "Desktop"

EXPECTED_COUNTS = {
    "training_rollouts.csv": 1152,
    "training_summary.csv": 1,
    "ebm_grasping_raw.csv": 18480,
    "raw_seed_metrics.csv": 1232,
    "ebm_grasping_metrics.csv": 154,
    "aggregate_metrics.csv": 14,
    "ebm_grasping_pairwise.csv": 143,
    "ebm_grasping_ablation_raw.csv": 4680,
    "ebm_grasping_ablation.csv": 39,
    "ablation_aggregate_metrics.csv": 13,
    "stress_sweep.csv": 54,
    "negative_cases.csv": 4,
}

EXPECTED_FIGURES = [
    "ebm_grasping_success_by_split.png",
    "ebm_grasping_safety_failures.png",
    "ebm_grasping_ablation_success.png",
    "ebm_grasping_stress_sweep.png",
    "stress_curve_data.csv",
]


def fail(message: str) -> None:
    raise SystemExit(f"VALIDATION FAILED: {message}")


def count_rows(path: Path) -> int:
    if not path.exists():
        fail(f"missing {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        return sum(1 for _ in reader)


def validate_counts() -> None:
    for name, expected in EXPECTED_COUNTS.items():
        actual = count_rows(RESULTS / name)
        if actual != expected:
            fail(f"{name} row count {actual} != {expected}")
    body = (RESULTS / "summary.txt").read_text(encoding="utf-8")
    if "Main rows: 18480; ablation rows: 4680; stress rows: 4320" not in body:
        fail("summary.txt does not record the frozen row counts")


def validate_figures() -> None:
    for name in EXPECTED_FIGURES:
        path = FIGURES / name
        if not path.exists() or path.stat().st_size < 1000:
            fail(f"missing or tiny figure/data artifact {path}")


def pdf_page_count(path: Path) -> int:
    try:
        from pypdf import PdfReader  # type: ignore

        return len(PdfReader(str(path)).pages)
    except Exception:
        pass
    try:
        from PyPDF2 import PdfReader  # type: ignore

        return len(PdfReader(str(path)).pages)
    except Exception:
        pass
    data = path.read_bytes()
    return len(re.findall(rb"/Type\s*/Page\b", data))


def validate_pdf() -> None:
    pdf = DOWNLOADS / "68.pdf"
    if not pdf.exists() or pdf.stat().st_size < 100000:
        fail(f"missing or tiny final PDF {pdf}")
    pages = pdf_page_count(pdf)
    if pages < 25:
        fail(f"final PDF has {pages} pages, expected at least 25")
    desktop_pdf = DESKTOP / "68.pdf"
    if desktop_pdf.exists():
        fail(f"Desktop PDF is forbidden: {desktop_pdf}")


def validate_tex() -> None:
    tex = PAPER / "main.tex"
    if not tex.exists():
        fail("paper/main.tex missing")
    body = tex.read_text(encoding="utf-8")
    required = [
        "citebordercolor",
        "pdfborder",
        "\\bibliography{references}",
        "\\input{generated/result_macros.tex}",
        "\\input{generated/full_metrics_longtable.tex}",
    ]
    missing = [item for item in required if item not in body]
    if missing:
        fail("main.tex missing citation/link hardening or generated tables: " + ", ".join(missing))


def validate_compile() -> None:
    subprocess.run([sys.executable, "-m", "py_compile", str(ROOT / "src" / "run_experiment.py")], check=True)
    subprocess.run([sys.executable, "-m", "py_compile", str(ROOT / "scripts" / "render_submission_assets.py")], check=True)


def main() -> None:
    validate_compile()
    validate_counts()
    validate_figures()
    validate_tex()
    validate_pdf()
    print("Paper 68 validation passed: counts, figures, TeX links, Downloads PDF, and Desktop hygiene are OK.")


if __name__ == "__main__":
    main()
