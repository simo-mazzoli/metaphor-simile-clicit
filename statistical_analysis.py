#!/usr/bin/env python3
"""Run regression analysis on LLM-derived metrics and compare trends across models.

This script expects one or more CSV files generated from `compute_likelihoods.py`.
It reads independent variables from a TXT file (one variable per line), estimates
linear effects on selected dependent variables, and creates plots across LLMs.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEPENDENT_VARIABLES_DEFAULT = [
    "Metaphor_log_likelihood_norm",
    "Simile_log_likelihood_norm",
    "Simile_come_cloze_surprisal",
]


def parse_independent_vars(path: Path) -> list[str]:
    vars_: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        vars_.append(line)

    if not vars_:
        raise ValueError(f"No independent variables found in: {path}")

    return vars_


def fit_linear_model(df: pd.DataFrame, y_col: str, x_cols: list[str]) -> pd.DataFrame:
    usable = df[[y_col, *x_cols]].copy()
    usable = usable.apply(pd.to_numeric, errors="coerce").dropna()

    if usable.empty:
        return pd.DataFrame(
            [{"term": "(no_data)", "coef": np.nan, "std_err": np.nan, "n": 0}]
        )

    y = usable[y_col].to_numpy(dtype=float)
    x = usable[x_cols].to_numpy(dtype=float)

    x_design = np.column_stack([np.ones(len(x)), x])
    coef, *_ = np.linalg.lstsq(x_design, y, rcond=None)

    y_hat = x_design @ coef
    residuals = y - y_hat
    n, p = x_design.shape
    dof = max(n - p, 1)
    sigma2 = float((residuals @ residuals) / dof)
    xtx_inv = np.linalg.pinv(x_design.T @ x_design)
    se = np.sqrt(np.diag(sigma2 * xtx_inv))

    terms = ["Intercept", *x_cols]
    rows = []
    for term, value, err in zip(terms, coef, se):
        rows.append({"term": term, "coef": float(value), "std_err": float(err), "n": int(n)})
    return pd.DataFrame(rows)


def load_inputs(paths: Iterable[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        df = pd.read_csv(path)
        if "source_llm" not in df.columns:
            inferred = path.stem
            df["source_llm"] = inferred
        frames.append(df)
    if not frames:
        raise ValueError("No input files provided.")
    return pd.concat(frames, ignore_index=True)


def plot_model_trends(df: pd.DataFrame, dependent_vars: list[str], outdir: Path) -> list[Path]:
    out_paths: list[Path] = []
    for dep in dependent_vars:
        if dep not in df.columns:
            continue

        grouped = (
            pd.to_numeric(df[dep], errors="coerce")
            .groupby(df["source_llm"])
            .mean()
            .sort_values()
        )
        if grouped.empty:
            continue

        fig, ax = plt.subplots(figsize=(10, 5))
        grouped.plot(kind="bar", ax=ax, color="#4C72B0")
        ax.set_title(f"Model trend for {dep}")
        ax.set_xlabel("LLM")
        ax.set_ylabel(f"Mean {dep}")
        ax.tick_params(axis="x", rotation=45, labelsize=8)
        fig.tight_layout()

        out_path = outdir / f"trend_{dep}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        out_paths.append(out_path)
    return out_paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Estimate variable effects and compare trends across LLM models."
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        type=Path,
        required=True,
        help="One or more CSV files produced by compute_likelihoods.py",
    )
    parser.add_argument(
        "--independent-vars",
        type=Path,
        default=Path("analysis_config.txt"),
        help="TXT file with independent variables (one per line)",
    )
    parser.add_argument(
        "--dependent-vars",
        nargs="*",
        default=DEPENDENT_VARIABLES_DEFAULT,
        help="Dependent variables to analyze",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis_output"),
        help="Directory where analysis files are written",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    independent_vars = parse_independent_vars(args.independent_vars)
    all_data = load_inputs(args.inputs)

    missing_iv = [col for col in independent_vars if col not in all_data.columns]
    if missing_iv:
        raise ValueError(f"Independent variable columns missing in data: {missing_iv}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    combined_results = []
    for model_name, model_df in all_data.groupby("source_llm"):
        for dep in args.dependent_vars:
            if dep not in model_df.columns:
                continue
            fit_df = fit_linear_model(model_df, dep, independent_vars)
            fit_df.insert(0, "dependent_var", dep)
            fit_df.insert(0, "source_llm", model_name)
            combined_results.append(fit_df)

    if not combined_results:
        raise ValueError("No model fits were produced. Check variable names and input files.")

    results_df = pd.concat(combined_results, ignore_index=True)
    coeff_out = args.output_dir / "regression_coefficients_by_llm.csv"
    results_df.to_csv(coeff_out, index=False)

    plot_paths = plot_model_trends(all_data, args.dependent_vars, args.output_dir)

    summary_out = args.output_dir / "analysis_summary.txt"
    with summary_out.open("w", encoding="utf-8") as f:
        f.write("Statistical analysis summary\n")
        f.write("==========================\n")
        f.write(f"Input files: {[str(p) for p in args.inputs]}\n")
        f.write(f"Independent variables: {independent_vars}\n")
        f.write(f"Dependent variables: {args.dependent_vars}\n")
        f.write(f"Models found: {sorted(all_data['source_llm'].dropna().unique().tolist())}\n")
        f.write(f"Regression output: {coeff_out}\n")
        f.write("Trend plots:\n")
        for path in plot_paths:
            f.write(f"- {path}\n")

    print(f"Saved regression coefficients to: {coeff_out}")
    print(f"Saved summary to: {summary_out}")
    for p in plot_paths:
        print(f"Saved plot: {p}")


if __name__ == "__main__":
    main()
