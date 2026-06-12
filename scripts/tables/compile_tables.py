#!/usr/bin/env python3
"""Compile LaTeX table .tex files in ./tables/ into PNG images in ./tables/images/."""

import argparse
import subprocess
import sys
import tempfile
import shutil
from pathlib import Path

LATEX_TEMPLATE = r"""
\documentclass[preview,border=4pt]{{standalone}}
\usepackage{{booktabs}}
\usepackage{{arydshln}}
\usepackage{{amsmath}}
\usepackage{{array}}
\begin{{document}}
{body}
\end{{document}}
"""


def compile_table(tex_path: Path, out_dir: Path, force: bool) -> bool:
    out_path = out_dir / (tex_path.stem + ".png")
    if out_path.exists() and not force:
        print(f"  skip  {out_path.name}  (use --force to overwrite)")
        return True

    body = tex_path.read_text()
    doc = LATEX_TEMPLATE.format(body=body)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        tex_file = tmp / "table.tex"
        tex_file.write_text(doc)

        result = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "table.tex"],
            cwd=tmp,
            capture_output=True,
            text=True,
        )
        pdf_file = tmp / "table.pdf"
        if not pdf_file.exists():
            print(f"  ERROR compiling {tex_path.name}")
            print(result.stdout[-2000:])
            return False

        png_file = tmp / "table.png"

        # pdftoppm outputs table-1.png (one file per page)
        result = subprocess.run(
            ["pdftoppm", "-r", "300", "-png", "-singlefile", str(pdf_file), str(tmp / "table")],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"  ERROR converting {tex_path.name} to PNG")
            print(result.stderr)
            return False

        # pdftoppm with -singlefile writes table.png
        shutil.copy(tmp / "table.png", out_path)
        print(f"  ok    {out_path.name}")
        return True


def main():
    parser = argparse.ArgumentParser(description="Compile LaTeX tables to PNG images.")
    parser.add_argument(
        "--tables-dir",
        default="tables",
        help="Directory containing .tex table files (default: tables/)",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory for images (default: <tables-dir>/images/)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing images.",
    )
    parser.add_argument(
        "files",
        nargs="*",
        help="Specific .tex files to compile (default: all in tables-dir).",
    )
    args = parser.parse_args()

    tables_dir = Path(args.tables_dir)
    if not tables_dir.is_dir():
        print(f"Error: tables directory '{tables_dir}' not found.")
        sys.exit(1)

    out_dir = Path(args.out_dir) if args.out_dir else tables_dir / "images"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.files:
        tex_files = [Path(f) for f in args.files]
    else:
        tex_files = sorted(tables_dir.glob("*.tex"))

    if not tex_files:
        print("No .tex files found.")
        sys.exit(0)

    print(f"Compiling {len(tex_files)} table(s) → {out_dir}/")
    ok = 0
    for tex in tex_files:
        if compile_table(tex, out_dir, args.force):
            ok += 1

    print(f"\nDone: {ok}/{len(tex_files)} succeeded.")
    if ok < len(tex_files):
        sys.exit(1)


if __name__ == "__main__":
    main()
