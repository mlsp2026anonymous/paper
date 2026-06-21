#!/usr/bin/env python3
"""
Create a LaTeX table that keeps the values from the first table and replaces
``$\\pm$ ...`` uncertainty terms with deltas against a second table.

Usage:
    python3 compare_latex_tables.py table_a.tex table_b.tex
    python3 compare_latex_tables.py table_a.tex table_b.tex --table-name my_table.tex

The output is written next to table_a as ``<table_a_stem>_delta.tex``.
Use ``--table-name`` to choose a different output filename.
Second-best underlined values are converted to italic by default. Use
``--underline-second-best`` to keep underline formatting.
If table_b has ``--`` where table_a has a number, the output cell is marked
with ``(--)`` because no delta can be computed.
Displayed dataset name ``EEG`` is normalized to ``EDF``.
"""

from __future__ import annotations

import argparse
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path


NUMBER_RE = re.compile(r"(?<![A-Za-z])[-+]?(?:\d+(?:\.\d*)?|\.\d+)")
PM_RE = re.compile(
    r"\s*\$\\pm\$\s*[-+]?(?:\d+(?:\.\d*)?|\.\d+)", re.ASCII
)
ROW_END_RE = re.compile(r"(?P<body>.*?)(?P<ending>\s*\\\\[^\r\n]*(?:\r?\n)?)$", re.DOTALL)
WRAPPER_RE = re.compile(r"^\s*\\(?:textbf|textit|underline)\{(?P<body>.*)\}\s*$", re.DOTALL)
UNIDA_METHOD_COLOR = "gray!65!black"
STAR_BACKBONE_NAMES = frozenset({"CNN", "FNO", "S3", "S3Layer", "TSLANet"})


def split_latex_row(line: str) -> list[str]:
    """Split a tabular row on unescaped ampersands."""
    cells: list[str] = []
    start = 0
    escaped = False
    for idx, char in enumerate(line):
        if char == "\\" and not escaped:
            escaped = True
            continue
        if char == "&" and not escaped:
            cells.append(line[start:idx])
            start = idx + 1
        escaped = False
    cells.append(line[start:])
    return cells


def strip_row_ending(cell: str) -> tuple[str, str]:
    match = ROW_END_RE.match(cell)
    if match is None:
        return cell, ""
    return match.group("body"), match.group("ending")


def display_body(cell: str) -> str:
    """Return the part of a cell that should be treated as displayed content."""
    body, _ = strip_row_ending(cell)
    body = body.strip()
    while True:
        match = WRAPPER_RE.match(body)
        if match is None:
            return body
        body = match.group("body").strip()


def extract_value(cell: str) -> Decimal | None:
    """Return the first displayed numeric value in a cell, if any."""
    body = display_body(cell)
    match = NUMBER_RE.match(body)
    if match is None:
        return None
    try:
        return Decimal(match.group(0))
    except InvalidOperation:
        return None


def decimal_places(cell: str) -> int:
    match = NUMBER_RE.match(display_body(cell))
    if match is None:
        return 1
    text = match.group(0)
    if "." not in text:
        return 0
    return len(text.rsplit(".", 1)[1])


def format_decimal(value: Decimal, places: int) -> str:
    quantizer = Decimal(1).scaleb(-places)
    return f"{value.quantize(quantizer):f}"


def remove_uncertainty(cell: str) -> str:
    return PM_RE.sub("", cell, count=1)


def replace_cell(cell_a: str, cell_b: str, color: bool = False, color_arrow: bool = False, add_star: bool = False) -> str:
    value_a = extract_value(cell_a)
    value_b = extract_value(cell_b)

    if value_a is None and value_b is None:
        return cell_a
    if value_a is None:
        return cell_a

    body, ending = strip_row_ending(cell_a)
    cleaned = remove_uncertainty(body)

    if add_star:
        cleaned = cleaned.rstrip() + "$^*$"

    if value_b is None:
        return append_annotation(cleaned, "(--)") + ending

    places = max(decimal_places(cell_a), decimal_places(cell_b))

    if value_a == value_b:
        annotation = "(=)"
    elif value_a > value_b:
        diff = format_decimal(abs(value_a - value_b), places)
        if color:
            annotation = f"\\textcolor{{green!60!black}}{{($\\uparrow$ {diff})}}"
        elif color_arrow:
            annotation = f"(\\textcolor{{green!60!black}}{{$\\boldsymbol{{\\uparrow}}$}} {diff})"
        else:
            annotation = f"($\\uparrow$ {diff})"
    else:
        diff = format_decimal(abs(value_a - value_b), places)
        if color:
            annotation = f"\\textcolor{{red}}{{($\\downarrow$ {diff})}}"
        elif color_arrow:
            annotation = f"(\\textcolor{{red}}{{$\\boldsymbol{{\\downarrow}}$}} {diff})"
        else:
            annotation = f"($\\downarrow$ {diff})"

    return append_annotation(cleaned, annotation) + ending


def append_annotation(cell: str, annotation: str) -> str:
    """Append the annotation inside common one-argument LaTeX wrappers."""
    stripped_right = cell.rstrip()
    trailing = cell[len(stripped_right) :]

    if stripped_right.endswith("}"):
        brace_depth = 0
        for idx in range(len(stripped_right) - 1, -1, -1):
            char = stripped_right[idx]
            if char == "}":
                brace_depth += 1
            elif char == "{":
                brace_depth -= 1
                if brace_depth == 0:
                    prefix = stripped_right[:idx]
                    if prefix.endswith("\\textbf") or prefix.endswith("\\underline"):
                        inner = stripped_right[idx + 1 : -1].rstrip()
                        inner_trailing = stripped_right[idx + 1 : -1][len(inner) :]
                        return (
                            f"{prefix}{{{inner} {annotation}{inner_trailing}}}"
                            f"{trailing}"
                        )
                    break

    return f"{stripped_right} {annotation}{trailing}"


def color_cell(cell: str, color_name: str) -> str:
    """Wrap the visible content of a cell in a LaTeX text color."""
    body, ending = strip_row_ending(cell)
    leading = body[: len(body) - len(body.lstrip())]
    stripped_left = body[len(leading) :]
    trailing = stripped_left[len(stripped_left.rstrip()) :]
    content = stripped_left[: len(stripped_left) - len(trailing)]

    if content == "":
        return cell
    return f"{leading}\\textcolor{{{color_name}}}{{{content}}}{trailing}{ending}"


def normalize_dataset_names(table: str) -> str:
    return table.replace("EEG", "EDF")


def normalize_second_best_style(table: str, underline_second_best: bool = False) -> str:
    if underline_second_best:
        return table
    return table.replace(r"\underline", r"\textit")


def find_star_col_indices(lines: list[str]) -> set[int]:
    """Return column indices whose header text matches a star-eligible backbone name."""
    for line in lines:
        if "&" not in line:
            continue
        cells = split_latex_row(line)
        cols = {
            col_idx
            for col_idx, cell in enumerate(cells)
            if display_body(cell).strip() in STAR_BACKBONE_NAMES
        }
        if cols:
            return cols
    return set()


def convert_table(
    table_a: str,
    table_b: str,
    color: bool = False,
    color_arrow: bool = False,
    underline_second_best: bool = False,
    star_unijdot: bool = False,
) -> str:
    lines_a = table_a.splitlines(keepends=True)
    lines_b = table_b.splitlines(keepends=True)

    if len(lines_a) != len(lines_b):
        raise ValueError(
            f"tables have different line counts: {len(lines_a)} and {len(lines_b)}"
        )

    star_cols: set[int] = find_star_col_indices(lines_a) if star_unijdot else set()

    output: list[str] = []
    numeric_cells = 0

    for line_number, (line_a, line_b) in enumerate(zip(lines_a, lines_b), start=1):
        if "&" not in line_a and "&" not in line_b:
            output.append(line_a)
            continue

        cells_a = split_latex_row(line_a)
        cells_b = split_latex_row(line_b)
        if len(cells_a) != len(cells_b):
            raise ValueError(
                f"line {line_number}: different column counts "
                f"({len(cells_a)} vs {len(cells_b)})"
            )

        converted_cells = []
        is_unida_method_row = "UniJDOT" in line_a or "UniJDOT" in line_b
        for col_idx, (cell_a, cell_b) in enumerate(zip(cells_a, cells_b)):
            add_star = is_unida_method_row and col_idx in star_cols
            converted = replace_cell(cell_a, cell_b, color=color, color_arrow=color_arrow, add_star=add_star)
            if extract_value(cell_a) is not None and extract_value(cell_b) is not None:
                numeric_cells += 1
            converted_cells.append(converted)
        if is_unida_method_row:
            converted_cells = [
                converted_cells[0],
                *[
                    color_cell(cell, UNIDA_METHOD_COLOR)
                    for cell in converted_cells[1:]
                ],
            ]
        output.append("&".join(converted_cells))

    if numeric_cells == 0:
        raise ValueError("no comparable numeric cells were found")

    table = "".join(output)
    table = normalize_second_best_style(table, underline_second_best)
    return normalize_dataset_names(table)


def output_path_for(first_table_path: Path, table_name: str | None = None) -> Path:
    if table_name is None:
        return first_table_path.with_name(f"{first_table_path.stem}_delta.tex")

    output_path = Path(table_name)
    if output_path.suffix == "":
        output_path = output_path.with_suffix(".tex")
    if output_path.is_absolute() or output_path.parent != Path("."):
        return output_path
    return first_table_path.with_name(output_path.name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare two LaTeX tables with the same shape and write a new table "
            "showing values from the first table plus up/down/equal deltas."
        )
    )
    parser.add_argument("first_table", type=Path)
    parser.add_argument("second_table", type=Path)
    parser.add_argument(
        "--table-name",
        default=None,
        help=(
            "Output table filename or path. If only a filename is given, it is "
            "written next to first_table. Adds .tex when no suffix is provided."
        ),
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--color", action="store_true",
                       help="Color the full annotation (parentheses included) green/red.")
    group.add_argument("--color-arrow", action="store_true",
                       help="Color only the arrow symbol green/red.")
    parser.add_argument(
        "--underline-second-best",
        action="store_true",
        help="Keep underline formatting for second-best values instead of using italic.",
    )
    parser.add_argument(
        "--star-unijdot",
        action="store_true",
        help=(
            "Append $^*$ after UniJDOT results for CNN/FNO/S3(Layer)/TSLANet backbone "
            "columns (detected from the table header)."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    table_a = args.first_table.read_text()
    table_b = args.second_table.read_text()
    output_path = output_path_for(args.first_table, args.table_name)
    output_path.write_text(
        convert_table(
            table_a,
            table_b,
            color=args.color,
            color_arrow=args.color_arrow,
            underline_second_best=args.underline_second_best,
            star_unijdot=args.star_unijdot,
        )
    )
    print(output_path)


if __name__ == "__main__":
    main()
