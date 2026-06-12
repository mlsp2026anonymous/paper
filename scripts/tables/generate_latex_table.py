#!/usr/bin/env python3
"""
Generate a LaTeX results table from training logs (reads results2.csv).

Identical to generate_latex_table.py except:
  - reads results2.csv instead of results.csv
  - the mean row shows mean ± std, where std = std of per-run means across seeds

Results are expected at:
    {root}/{backbone}/{dataset}/{method}_{ext}/results2.csv

Usage:
    python generate_latex_table2.py \
        --root     logs/run_sf_unida        \
        --metric   H-score                  \
        --backbone CNN                      \
        --dataset  HAR                      \
        --methods  bGAT LEAD GLC GLCpp UMAD \
        --method-names bGAT LEAD GLC GLC++ UMAD \
        --output   tables/table_HAR_CNN.tex
"""

import argparse
import math
import os
from pathlib import Path

import pandas as pd
from tabulate import tabulate


# ─── auto-recompute ───────────────────────────────────────────────────────────

def _scenario_sort_key(sc: str) -> tuple:
    parts = sc.split("_to_")
    try:
        return (int(parts[0]), int(parts[-1]))
    except ValueError:
        return (0, 0)


def _recompute_one(temp_path: Path) -> None:
    """Write results2.csv next to results_temp.csv."""
    df = pd.read_csv(temp_path, index_col=0)
    metrics = [c for c in df.columns if c not in ("scenario", "run")]
    rows = []
    for sc in sorted(df["scenario"].unique(), key=_scenario_sort_key):
        sc_df = df[df["scenario"] == sc]
        row = {"scenario": sc}
        for m in metrics:
            vals = pd.to_numeric(sc_df[m], errors="coerce")
            row[m] = vals.mean()
            row[f"{m}_std"] = vals.std(ddof=1)
        rows.append(row)
    run_means = df.groupby("run")[metrics].mean()
    mean_row = {"scenario": "mean"}
    for m in metrics:
        mean_row[m] = run_means[m].mean()
        mean_row[f"{m}_std"] = run_means[m].std(ddof=1)
    rows.append(mean_row)
    sc_agg = pd.DataFrame([r for r in rows if r["scenario"] not in ("mean", "std")])
    std_row = {"scenario": "std"}
    for m in metrics:
        std_row[m] = sc_agg[m].std(ddof=1)
        std_row[f"{m}_std"] = float("nan")
    rows.append(std_row)
    cols = ["scenario"] + metrics + [f"{m}_std" for m in metrics]
    pd.DataFrame(rows, columns=cols).to_csv(temp_path.parent / "results2.csv")


def _ensure_results2(results2_path: str) -> None:
    """If results2.csv is missing but results_temp.csv exists, recompute it."""
    p = Path(results2_path)
    if p.exists():
        return
    temp = p.parent / "results_temp.csv"
    if temp.exists():
        print(f"  [auto-recompute] {p.parent.name}/results2.csv")
        _recompute_one(temp)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _results_path(root: str, backbone: str, dataset: str, method: str, ext: str) -> str:
    folder = f"{method}_{ext}" if ext else method
    return os.path.join(root, backbone, dataset, folder, "results2.csv")


def _load_results(path: str) -> pd.DataFrame | None:
    _ensure_results2(path)
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
        for col in df.columns:
            if col != "scenario":
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df
    except Exception as e:
        print(f"[warn] Could not read {path}: {e}")
        return None


def _second_largest(values: list[float]) -> float | None:
    unique = sorted(set(v for v in values if not math.isnan(v)))
    return unique[-2] if len(unique) >= 2 else None


def _fmt_pct(val_01: float, std_01: float | None = None, decimals: int = 1) -> str:
    vp = val_01 * 100
    mm = f"{vp:.{decimals}f}"
    if std_01 is None or round(vp, decimals) == 0.0:
        return mm
    return f"{mm} $\\pm$ {std_01 * 100:.{decimals}f}"


def _fmt_terminal(val_01: float, std_01: float | None = None, decimals: int = 1) -> str:
    vp = val_01 * 100
    mm = f"{vp:.{decimals}f}"
    if std_01 is None or round(vp, decimals) == 0.0:
        return mm
    return f"{mm} ± {std_01 * 100:.{decimals}f}"


def _scenario_sort_key(sc: str) -> tuple:
    parts = sc.split("_to_")
    try:
        return (int(parts[0]), int(parts[-1]))
    except ValueError:
        return (0, 0)


def _bold_latex(s: str) -> str:
    return f"\\textbf{{{s}}}"


def _underline_latex(s: str) -> str:
    return f"\\underline{{{s}}}"


# ─── core ─────────────────────────────────────────────────────────────────────

def generate_table(
    root: str,
    metric: str,
    backbone: str,
    dataset: str,
    methods: list[str],
    method_names: list[str],
    ext: str = "",
    caption: str | None = None,
    label: str | None = None,
    output: str | None = None,
) -> str:
    std_col = f"{metric}_std"

    method_df: dict[str, pd.DataFrame | None] = {}
    for method in methods:
        path = _results_path(root, backbone, dataset, method, ext)
        df = _load_results(path)
        if df is None:
            print(f"[warn] No results at {path}")
        method_df[method] = df

    seen: set[str] = set()
    scenarios: list[str] = []
    for df in method_df.values():
        if df is None:
            continue
        for sc in df["scenario"]:
            if sc not in ("mean", "std") and sc not in seen:
                seen.add(sc)
                scenarios.append(sc)
    scenarios.sort(key=_scenario_sort_key)

    if not scenarios:
        raise RuntimeError("No scenario data found for any method.")

    def _get_val_std(method: str, sc: str) -> tuple[float, float]:
        df = method_df[method]
        if df is not None and sc in df["scenario"].values:
            row = df[df["scenario"] == sc].iloc[0]
            v = float(row[metric]) if metric in row.index else float("nan")
            s = float(row[std_col]) if std_col in row.index else 0.0
            return v, s
        return float("nan"), 0.0

    def _get_mean(method: str) -> float:
        df = method_df[method]
        if df is None:
            return float("nan")
        if "mean" in df["scenario"].values:
            row = df[df["scenario"] == "mean"].iloc[0]
            return float(row[metric]) if metric in row.index else float("nan")
        sc_rows = df[~df["scenario"].isin(["mean", "std"])]
        v_series = pd.to_numeric(sc_rows[metric], errors="coerce").dropna()
        return float(v_series.mean()) if not v_series.empty else float("nan")

    def _get_mean_std(method: str) -> float:
        """Std of per-run means, read from the 'mean' row's _std column."""
        df = method_df[method]
        if df is None:
            return float("nan")
        if "mean" in df["scenario"].values:
            row = df[df["scenario"] == "mean"].iloc[0]
            return float(row[std_col]) if std_col in row.index else float("nan")
        return float("nan")

    if caption is None:
        caption = f"  {metric} (\\%) for {dataset} ({backbone})"
    if label is None:
        safe_metric = metric.lower().replace("-", "_")
        label = f"tab:{safe_metric}_{dataset}_{backbone}"

    n = len(methods)
    col_spec = "l " + " ".join(["c"] * n)
    header_cells = " & ".join(f"\\textbf{{{name}}}" for name in method_names)

    latex_lines: list[str] = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\begin{{tabular}}{{ {col_spec}}}",
        "\\toprule",
        f"\\textbf{{Scenario}} & {header_cells}\\\\",
        "\\midrule ",
    ]
    terminal_rows: list[list[str]] = []

    def _build_row(vals: list[float], stds: list[float],
                   sc_latex: str, sc_term: str, show_std: bool) -> None:
        valid = [v for v in vals if not math.isnan(v)]
        best = max(valid) if valid else None
        second = _second_largest(valid)

        latex_cells, term_cells = [], []
        for v, s in zip(vals, stds):
            if math.isnan(v):
                latex_cells.append("--")
                term_cells.append("--")
                continue
            is_best = best is not None and round(v, 4) == round(best, 4)
            is_2nd  = second is not None and round(v, 4) == round(second, 4)

            s_arg = s if show_std else None
            latex_fmt = _fmt_pct(v, s_arg)
            term_fmt  = _fmt_terminal(v, s_arg)

            if is_best:
                latex_cells.append(_bold_latex(latex_fmt))
                term_cells.append(f"*{term_fmt}*")
            elif is_2nd:
                latex_cells.append(_underline_latex(latex_fmt))
                term_cells.append(f"_{term_fmt}_")
            else:
                latex_cells.append(latex_fmt)
                term_cells.append(term_fmt)

        latex_lines.append(f" {sc_latex} & {' & '.join(latex_cells)} \\\\")
        terminal_rows.append([sc_term] + term_cells)

    for sc in scenarios:
        parts = sc.split("_to_")
        sc_latex = f"${parts[0]} \\rightarrow {parts[-1]}$"
        sc_term = f"{parts[0]} → {parts[-1]}"
        vals, stds = zip(*[_get_val_std(m, sc) for m in methods])
        _build_row(list(vals), list(stds), sc_latex, sc_term, show_std=True)

    # Mean row: show ± std of per-run means
    mean_vals = [_get_mean(m) for m in methods]
    mean_stds = [_get_mean_std(m) for m in methods]
    _build_row(mean_vals, mean_stds, "\\hdashline\n mean", "mean", show_std=True)

    latex_lines.append("")
    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabular}")
    latex_lines.append(f"\\label{{{label}}}")
    latex_lines.append("\\end{table}")
    latex_lines.append("")

    table_str = "\n".join(latex_lines)

    sep_idx = len(terminal_rows) - 1
    display_rows = terminal_rows[:sep_idx] + [["─" * 12] + ["─" * 10] * n] + terminal_rows[sep_idx:]
    print(f"\n{'═' * 60}")
    print(f"  {metric} (%)  ·  {dataset}  ·  {backbone}")
    print(f"  (* = best,  _ = 2nd best,  mean row shows ± std across seeds)")
    print(f"{'═' * 60}")
    print(tabulate(
        display_rows,
        headers=["Scenario"] + method_names,
        tablefmt="simple",
        stralign="right",
        numalign="right",
    ))
    print()

    if output:
        os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
        with open(output, "w") as f:
            f.write(table_str)
        print(f"LaTeX table written to {output}")

    return table_str


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate a LaTeX results table (mean row shows ± std across seeds).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--root", default="logs/run_sf_unida")
    parser.add_argument("--metric", required=True)
    parser.add_argument("--backbone", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--methods", nargs="+", required=True)
    parser.add_argument("--method-names", nargs="+", default=None)
    parser.add_argument("--ext", default="")
    parser.add_argument("--caption", default=None)
    parser.add_argument("--label", default=None)
    parser.add_argument("--output", "-o", default=None)
    args = parser.parse_args()

    method_names = args.method_names or args.methods
    if len(method_names) != len(args.methods):
        parser.error(
            f"--method-names has {len(method_names)} entries "
            f"but --methods has {len(args.methods)}"
        )

    generate_table(
        root=args.root,
        metric=args.metric,
        backbone=args.backbone,
        dataset=args.dataset,
        methods=args.methods,
        method_names=method_names,
        ext=args.ext,
        caption=args.caption,
        label=args.label,
        output=args.output,
    )


if __name__ == "__main__":
    main()
