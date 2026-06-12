#!/usr/bin/env python3
"""
Generate a LaTeX summary table (datasets × methods × backbones) with ± std.
Reads results2.csv. Identical to generate_summary_table.py except:
  - reads results2.csv instead of results.csv
  - every cell shows mean ± std (std = std of per-run means across seeds)

Results are read from:
    {root}/{backbone}/{dataset}/{method}_{ext}/results2.csv

Usage:
    python generate_summary_table2.py \\
        --root       logs/run_sf_unida                     \\
        --metric     H-score                               \\
        --datasets   HAR HHAR EEG                          \\
        --methods    bGAT GLC GLCpp LEAD                   \\
        --method-names bGAT GLC GLC++ LEAD                 \\
        --backbones  CNN FNO TSLANet S3Layer                \\
        --backbone-names CNN FNO TSLANet S3                \\
        --ext        EXP1                                  \\
        --output     tables/summary_sf_unida_hscore.tex
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

def _results_path(root, backbone, dataset, method, ext):
    folder = f"{method}_{ext}" if ext else method
    return os.path.join(root, backbone, dataset, folder, "results2.csv")


def _load_mean_std(path, metric):
    """Return (mean, std) from the 'mean' row of results2.csv, or (NaN, NaN)."""
    _ensure_results2(path)
    if not os.path.exists(path):
        return float("nan"), float("nan")
    try:
        df = pd.read_csv(path)
        for col in df.columns:
            if col != "scenario":
                df[col] = pd.to_numeric(df[col], errors="coerce")
        std_col = f"{metric}_std"
        if metric not in df.columns:
            return float("nan"), float("nan")
        if "mean" in df["scenario"].values:
            row = df[df["scenario"] == "mean"].iloc[0]
            v = float(row[metric])
            s = float(row[std_col]) if std_col in df.columns else float("nan")
            return v, s
        mask = ~df["scenario"].isin(["mean", "std"])
        vals = df.loc[mask, metric].dropna()
        return (float(vals.mean()) if not vals.empty else float("nan")), float("nan")
    except Exception as e:
        print(f"[warn] {path}: {e}")
        return float("nan"), float("nan")


def _second_largest(values):
    unique = sorted(set(v for v in values if not math.isnan(v)))
    return unique[-2] if len(unique) >= 2 else None


def _fmt_pct(v, decimals=1):
    if math.isnan(v):
        return "--"
    return f"{v * 100:.{decimals}f}"


def _fmt_pct_std(v, s, decimals=1):
    if math.isnan(v):
        return "--"
    mean_str = f"{v * 100:.{decimals}f}"
    if math.isnan(s):
        return mean_str
    std_str = f"{s * 100:.{decimals}f}"
    return f"{mean_str} $\\pm$ {std_str}"


def _fmt_pct_std_terminal(v, s, decimals=1):
    if math.isnan(v):
        return "--"
    mean_str = f"{v * 100:.{decimals}f}"
    if math.isnan(s):
        return mean_str
    std_str = f"{s * 100:.{decimals}f}"
    return f"{mean_str} ± {std_str}"


# ─── core ─────────────────────────────────────────────────────────────────────

def generate_summary_table(
    root,
    metric,
    datasets,
    methods,
    method_names,
    backbones,
    backbone_names,
    ext="",
    caption=None,
    label=None,
    output=None,
    decimals=1,
):
    nb = len(backbones)
    nm = len(methods)

    # Collect mean and std: data[dataset][method][backbone] = (mean, std)
    data = {}
    for ds in datasets:
        data[ds] = {}
        for method in methods:
            data[ds][method] = {}
            for bb in backbones:
                path = _results_path(root, bb, ds, method, ext)
                v, s = _load_mean_std(path, metric)
                data[ds][method][bb] = (v, s)
                if math.isnan(v):
                    print(f"[warn] missing: {path}")

    # Terminal pretty-print
    term_rows = []
    for ds in datasets:
        first_method = True
        for method, mname in zip(methods, method_names):
            vals = [data[ds][method][bb][0] for bb in backbones]
            valid = [v for v in vals if not math.isnan(v)]
            best   = max(valid) if valid else None
            second = _second_largest(valid)

            cells = []
            for bb in backbones:
                v, s = data[ds][method][bb]
                if math.isnan(v):
                    cells.append("--")
                    continue
                cell = _fmt_pct_std_terminal(v, s, decimals)
                is_best = best is not None and round(v, 5) == round(best, 5)
                is_2nd  = second is not None and round(v, 5) == round(second, 5)
                if is_best:
                    cells.append(f"*{cell}*")
                elif is_2nd:
                    cells.append(f"_{cell}_")
                else:
                    cells.append(cell)

            ds_label = ds if first_method else ""
            term_rows.append([ds_label, mname] + cells)
            first_method = False

        term_rows.append([""] * (2 + nb))

    if term_rows and all(c == "" for c in term_rows[-1]):
        term_rows.pop()

    print(f"\n{'═' * (40 + 14 * nb)}")
    print(f"  {metric} (%)  —  Summary table  (mean ± std across seeds)")
    print(f"  (* = best backbone,  _ = 2nd best backbone,  per row)")
    print(f"{'═' * (40 + 14 * nb)}")
    print(tabulate(
        term_rows,
        headers=["Dataset", "Method"] + backbone_names,
        tablefmt="simple",
        stralign="right",
        numalign="right",
    ))
    print()

    # LaTeX
    if caption is None:
        ds_str = ", ".join(datasets)
        caption = f"{metric} (\\%) for {ds_str}"
    if label is None:
        safe = metric.lower().replace("-", "_")
        label = f"tab:{safe}_summary"

    col_spec = "l l " + " ".join(["c"] * nb)
    cmidrule_end = 2 + nb

    lines = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{{caption}}}",
        "\\resizebox{\\linewidth}{!}{",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\toprule",
        f"\\multirow{{3}}{{*}}{{\\textbf{{Datasets}}}} & "
        f"\\multirow{{3}}{{*}}{{\\textbf{{Methods}}}} & "
        f"\\multicolumn{{{nb}}}{{c}}{{\\textbf{{Backbones}}}} \\\\",
        f"\\cmidrule(lr){{3-{cmidrule_end}}}",
        "& & " + " & ".join(f"\\textbf{{{bn}}}" for bn in backbone_names) + "\\\\",
        "\\midrule ",
    ]

    for ds_idx, ds in enumerate(datasets):
        for m_idx, (method, mname) in enumerate(zip(methods, method_names)):
            vals = [data[ds][method][bb][0] for bb in backbones]
            valid = [v for v in vals if not math.isnan(v)]
            best   = max(valid) if valid else None
            second = _second_largest(valid)

            cells = []
            for bb in backbones:
                v, s = data[ds][method][bb]
                if math.isnan(v):
                    cells.append("--")
                    continue
                cell = _fmt_pct_std(v, s, decimals)
                is_best = best is not None and round(v, 5) == round(best, 5)
                is_2nd  = second is not None and round(v, 5) == round(second, 5)
                if is_best:
                    cells.append(f"\\textbf{{{cell}}}")
                elif is_2nd:
                    cells.append(f"\\underline{{{cell}}}")
                else:
                    cells.append(cell)

            if m_idx == 0:
                ds_cell = f"\\multirow{{{nm}}}{{*}}{{\\textbf{{{ds}}}}}"
            else:
                ds_cell = ""

            row = f"{ds_cell} & \\textbf{{{mname}}} & {' & '.join(cells)} \\\\"
            lines.append(row)

        if ds_idx < len(datasets) - 1:
            lines.append("\\midrule")

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
        description="Generate a LaTeX summary table with mean ± std across seeds.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--root", required=True)
    parser.add_argument("--metric", required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--methods", nargs="+", required=True)
    parser.add_argument("--method-names", nargs="+", default=None)
    parser.add_argument("--backbones", nargs="+", required=True)
    parser.add_argument("--backbone-names", nargs="+", default=None)
    parser.add_argument("--ext", default="")
    parser.add_argument("--decimals", type=int, default=1)
    parser.add_argument("--caption", default=None)
    parser.add_argument("--label", default=None)
    parser.add_argument("--output", "-o", default=None)
    args = parser.parse_args()

    method_names   = args.method_names   or args.methods
    backbone_names = args.backbone_names or args.backbones

    for flag, given, expected in [
        ("--method-names",   method_names,   args.methods),
        ("--backbone-names", backbone_names, args.backbones),
    ]:
        if len(given) != len(expected):
            parser.error(f"{flag} has {len(given)} entries but its counterpart has {len(expected)}")

    generate_summary_table(
        root=args.root,
        metric=args.metric,
        datasets=args.datasets,
        methods=args.methods,
        method_names=method_names,
        backbones=args.backbones,
        backbone_names=backbone_names,
        ext=args.ext,
        caption=args.caption,
        label=args.label,
        output=args.output,
        decimals=args.decimals,
    )


if __name__ == "__main__":
    main()
