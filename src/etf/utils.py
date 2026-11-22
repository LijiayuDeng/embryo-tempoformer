"""Small I/O helpers shared by the analysis modules."""

from __future__ import annotations

import csv
import json
import math
import os
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np


def parse_list(text: str) -> list[str]:
    """Split a comma-separated string into stripped, non-empty items."""
    return [x.strip() for x in str(text).split(",") if x.strip()]


def parse_int_list(text: str) -> list[int]:
    """Parse a comma-separated list of integers into a sorted, de-duplicated list."""
    out = sorted({int(x) for x in parse_list(text)})
    if not out:
        raise ValueError(f"empty integer list: {text!r}")
    return out


def read_json(path: str | Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(obj: Any, path: str | Path) -> None:
    """Write UTF-8 JSON (indent=2) atomically, creating parent directories.

    The file is written next to its destination and renamed into place, so an
    interrupted run never leaves a truncated file that a resumed run would skip.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: str | Path, rows: Iterable[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def read_column(path: str | Path, col: str) -> np.ndarray:
    """Read the finite numeric values of one CSV column as float64."""
    vals: list[float] = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or col not in reader.fieldnames:
            raise ValueError(f"Column '{col}' not found in {path}. fields={reader.fieldnames}")
        for row in reader:
            try:
                v = float(row[col])
            except (TypeError, ValueError):
                continue
            if math.isfinite(v):
                vals.append(v)
    if not vals:
        raise ValueError(f"No finite values for column '{col}' in {path}")
    return np.asarray(vals, dtype=np.float64)


def fmt(value: Any, digits: int = 3) -> str:
    """Format a number for tables; empty for None/'', passthrough for non-numeric text."""
    if value is None or value == "":
        return ""
    try:
        x = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{x:.{digits}f}"


def markdown_table(rows: Iterable[Mapping[str, Any]], headers: Sequence[str]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(h, "")) for h in headers) + " |")
    return "\n".join(lines) + "\n"
