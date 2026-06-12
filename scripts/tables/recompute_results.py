#!/usr/bin/env python3
"""
Recompute results2.csv from results_temp.csv for all experiments under a root.
results.csv is never touched.

Per-scenario rows : mean and std over runs (recomputed from scratch).
Mean row          : global mean + std of per-run means (variability across seeds).
Std row           : std of per-scenario means (variability across scenarios).

Usage:
    python recompute_results.py --root logs/run_sf_unida
    python recompute_results.py --root logs/run_sf_unida --dry-run
"""

import argparse
from pathlib import Path

import pandas as pd


def _scenario_sort_key(sc: str) -> tuple:
    parts = sc.split("_to_")
    try:
        return (int(parts[0]), int(parts[-1]))
    except ValueError:
        return (0, 0)


def recompute_one(temp_path: Path, dry_run: bool) -> None:
    df = pd.read_csv(temp_path, index_col=0)
    metrics = [c for c in df.columns if c not in ("scenario", "run")]

    rows = []
    scenarios = sorted(df["scenario"].unique(), key=_scenario_sort_key)

    # Per-scenario rows: mean and std over runs
    for sc in scenarios:
        sc_df = df[df["scenario"] == sc]
        row = {"scenario": sc}
        for m in metrics:
            vals = pd.to_numeric(sc_df[m], errors="coerce")
            row[m] = vals.mean()
            row[f"{m}_std"] = vals.std(ddof=1)
        rows.append(row)

    # Per-run means: for each run, average over all scenarios
    run_means = df.groupby("run")[metrics].mean()

    # Mean row: global mean + std of per-run means (std across seeds)
    mean_row = {"scenario": "mean"}
    for m in metrics:
        mean_row[m] = run_means[m].mean()
        mean_row[f"{m}_std"] = run_means[m].std(ddof=1)
    rows.append(mean_row)

    # Std row: std of per-scenario means (variation across scenarios)
    sc_df_agg = pd.DataFrame([r for r in rows if r["scenario"] not in ("mean", "std")])
    std_row = {"scenario": "std"}
    for m in metrics:
        std_row[m] = sc_df_agg[m].std(ddof=1)
        std_row[f"{m}_std"] = float("nan")
    rows.append(std_row)

    cols = ["scenario"] + metrics + [f"{m}_std" for m in metrics]
    result = pd.DataFrame(rows, columns=cols)
    out_path = temp_path.parent / "results2.csv"

    if dry_run:
        print(f"  [dry_run] would write → {out_path}")
        show = ["scenario"] + [c for c in ["H-score", "H-score_std"] if c in result.columns]
        print(result[show].to_string(index=False))
    else:
        result.to_csv(out_path)
        print(f"  ok  → {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Recompute results2.csv from results_temp.csv. "
                    "Mean-row std = std of per-run means (not mean of per-scenario stds). "
                    "results.csv is never modified."
    )
    parser.add_argument("--root", required=True, help="Root log directory, e.g. logs/run_sf_unida")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be written without writing.")
    args = parser.parse_args()

    temp_files = sorted(Path(args.root).rglob("results_temp.csv"))
    if not temp_files:
        print(f"No results_temp.csv found under {args.root}")
        return

    print(f"Found {len(temp_files)} results_temp.csv file(s) under {args.root}")
    ok = 0
    for p in temp_files:
        rel = p.parent.relative_to(args.root)
        print(f"\n[{rel}]")
        try:
            recompute_one(p, args.dry_run)
            ok += 1
        except Exception as e:
            print(f"  ERROR: {e}")

    print(f"\nDone: {ok}/{len(temp_files)} processed.")


if __name__ == "__main__":
    main()
