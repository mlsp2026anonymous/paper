#!/usr/bin/env python3
"""
Generate a LaTeX table comparing auto-thresholding methods across backbones.

Table layout:
  - Rows    : thresholding methods (yen, otsu, li, triangle …)
  - Columns : backbones (CNN, FNO, Mantis …)
  - Cells   : mean H-score (or any metric) in %
  - Bold    : best thresholding method per backbone column
  - Underline: 2nd-best thresholding method per backbone column

Results are read from:
    {root}/{thr_method}/{backbone}/{dataset}/{method}_{exp_name}/results.csv

Usage:
    python generate_threshold_table.py \\
        --root        logs_test/threshold_comparison \\
        --method      GLC                            \\
        --dataset     HAR                            \\
        --backbones   CNN FNO Mantis                 \\
        --thr-methods yen otsu li triangle           \\
        --exp-name    THR_EXP1                       \\
        --metric      H-score                        \\
        --output      tables/threshold_GLC_HAR.tex
"""

import argparse
import math
import os

import pandas as pd
from tabulate import tabulate


# ─── helpers ──────────────────────────────────────────────────────────────────

def _results_path(root, thr_method, backbone, dataset, method, exp_name):
    folder = f"{method}_{exp_name}" if exp_name else method
    return os.path.join(root, thr_method, backbone, dataset, folder, "results.csv")


def _load_mean(path, metric):
    if not os.path.exists(path):
        return float("nan")
    try:
        df = pd.read_csv(path)
        for col in df.columns:
            if col != "scenario":
                df[col] = pd.to_numeric(df[col], errors="coerce")
        if metric not in df.columns:
            return float("nan")
        if "mean" in df["scenario"].values:
            return float(df[df["scenario"] == "mean"].iloc[0][metric])
        mask = ~df["scenario"].isin(["mean", "std"])
        vals = df.loc[mask, metric].dropna()
        return float(vals.mean()) if not vals.empty else float("nan")
    except Exception as e:
        print(f"[warn] {path}: {e}")
        return float("nan")


def _second_largest(values):
    unique = sorted(set(v for v in values if not math.isnan(v)))
    return unique[-2] if len(unique) >= 2 else None


def _fmt_pct(v, decimals=1):
    if math.isnan(v):
        return "--"
    return f"{v * 100:.{decimals}f}"


# ─── core ─────────────────────────────────────────────────────────────────────

def generate_threshold_table(
    root,
    method,
    dataset,
    backbones,
    backbone_names,
    thr_methods,
    thr_names,
    exp_name,
    metric="H-score",
    caption=None,
    label=None,
    output=None,
    decimals=1,
):
    nb = len(backbones)
    nt = len(thr_methods)

    # data[thr][backbone] = mean value
    data = {}
    for thr in thr_methods:
        data[thr] = {}
        for bb in backbones:
            path = _results_path(root, thr, bb, dataset, method, exp_name)
            data[thr][bb] = _load_mean(path, metric)
            if math.isnan(data[thr][bb]):
                print(f"[warn] missing: {path}")

    # best/2nd-best per backbone column
    best_per_bb   = {}
    second_per_bb = {}
    for bb in backbones:
        col_vals = [data[thr][bb] for thr in thr_methods]
        valid = [v for v in col_vals if not math.isnan(v)]
        best_per_bb[bb]   = max(valid) if valid else None
        second_per_bb[bb] = _second_largest(valid)

    # ── terminal ──────────────────────────────────────────────────────────────
    term_rows = []
    for thr, tname in zip(thr_methods, thr_names):
        cells = []
        for bb in backbones:
            v = data[thr][bb]
            if math.isnan(v):
                cells.append("--")
                continue
            s = _fmt_pct(v, decimals)
            if best_per_bb[bb] is not None and round(v, 5) == round(best_per_bb[bb], 5):
                cells.append(f"*{s}*")
            elif second_per_bb[bb] is not None and round(v, 5) == round(second_per_bb[bb], 5):
                cells.append(f"_{s}_")
            else:
                cells.append(s)
        term_rows.append([tname] + cells)

    sep = "═" * (20 + 10 * nb)
    print(f"\n{sep}")
    print(f"  {metric} (%)  —  {method} on {dataset}  (threshold comparison)")
    print(f"  (* = best thr method per backbone,  _ = 2nd best)")
    print(sep)
    print(tabulate(
        term_rows,
        headers=["Threshold"] + backbone_names,
        tablefmt="simple",
        stralign="right",
        numalign="right",
    ))
    print()

    # ── LaTeX ─────────────────────────────────────────────────────────────────
    if caption is None:
        caption = (f"{metric} (\\%) for \\textbf{{{method}}} on \\textbf{{{dataset}}} "
                   f"across backbones and thresholding methods")
    if label is None:
        label = f"tab:thr_{method.lower()}_{dataset.lower()}"

    col_spec = "l " + " ".join(["c"] * nb)
    cmidrule_end = 1 + nb

    lines = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{{caption}}}",
        "\\resizebox{\\linewidth}{!}{",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\toprule",
        f"\\textbf{{Threshold}} & "
        f"\\multicolumn{{{nb}}}{{c}}{{\\textbf{{Backbones}}}} \\\\",
        f"\\cmidrule(lr){{2-{cmidrule_end}}}",
        "& " + " & ".join(f"\\textbf{{{bn}}}" for bn in backbone_names) + " \\\\",
        "\\midrule",
    ]

    for thr, tname in zip(thr_methods, thr_names):
        cells = []
        for bb in backbones:
            v = data[thr][bb]
            if math.isnan(v):
                cells.append("--")
                continue
            s = _fmt_pct(v, decimals)
            if best_per_bb[bb] is not None and round(v, 5) == round(best_per_bb[bb], 5):
                cells.append(f"\\textbf{{{s}}}")
            elif second_per_bb[bb] is not None and round(v, 5) == round(second_per_bb[bb], 5):
                cells.append(f"\\underline{{{s}}}")
            else:
                cells.append(s)
        lines.append(f"\\textbf{{{tname}}} & {' & '.join(cells)} \\\\")

    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "}",
        f"\\label{{{label}}}",
        "\\end{table}",
        "",
    ]

    table_str = "\n".join(lines)

    if output:
        os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
        with open(output, "w") as f:
            f.write(table_str)
        print(f"LaTeX table written to {output}")

    return table_str


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate a LaTeX threshold-comparison table (thr_methods × backbones).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--root", required=True,
                        help="Root log directory, e.g. logs_test/threshold_comparison")
    parser.add_argument("--method", required=True,
                        help="DA method, e.g. GLC")
    parser.add_argument("--dataset", required=True,
                        help="Dataset, e.g. HAR")
    parser.add_argument("--backbones", nargs="+", required=True,
                        help="Backbone folder names, e.g. CNN FNO Mantis")
    parser.add_argument("--backbone-names", nargs="+", default=None,
                        help="Display names for backbones (defaults to --backbones)")
    parser.add_argument("--thr-methods", nargs="+",
                        default=["yen", "otsu", "li", "triangle"],
                        help="Thresholding method names (default: yen otsu li triangle)")
    parser.add_argument("--thr-names", nargs="+", default=None,
                        help="Display names for thr methods (defaults to --thr-methods)")
    parser.add_argument("--exp-name", default="THR_EXP1",
                        help="Experiment suffix, e.g. THR_EXP1 (default: THR_EXP1)")
    parser.add_argument("--metric", default="H-score",
                        help="Metric column name (default: H-score)")
    parser.add_argument("--decimals", type=int, default=1,
                        help="Decimal places for percentage values (default: 1)")
    parser.add_argument("--caption", default=None)
    parser.add_argument("--label", default=None)
    parser.add_argument("--output", "-o", default=None,
                        help="Output .tex file (prints to stdout if omitted)")
    args = parser.parse_args()

    backbone_names = args.backbone_names or args.backbones
    thr_names      = args.thr_names      or args.thr_methods

    for flag, given, expected in [
        ("--backbone-names", backbone_names, args.backbones),
        ("--thr-names",      thr_names,      args.thr_methods),
    ]:
        if len(given) != len(expected):
            parser.error(f"{flag} has {len(given)} entries but its counterpart has {len(expected)}")

    generate_threshold_table(
        root=args.root,
        method=args.method,
        dataset=args.dataset,
        backbones=args.backbones,
        backbone_names=backbone_names,
        thr_methods=args.thr_methods,
        thr_names=thr_names,
        exp_name=args.exp_name,
        metric=args.metric,
        caption=args.caption,
        label=args.label,
        output=args.output,
        decimals=args.decimals,
    )


if __name__ == "__main__":
    main()
