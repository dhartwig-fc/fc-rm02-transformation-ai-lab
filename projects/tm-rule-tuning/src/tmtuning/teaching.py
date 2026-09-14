"""Small presentation helpers so the weekly exercise scripts stay readable.

Nothing here is part of the tuning method -- it is formatting, kept out of the
analysis modules so those stay importable into real work without dragging
print-and-banner code along.
"""

from __future__ import annotations

import textwrap

import pandas as pd

__all__ = ["banner", "show", "answer", "OUTPUT_DIR"]

#: Where week scripts write charts and generated papers.
OUTPUT_DIR = "outputs"


def banner(title: str, char: str = "=") -> None:
    """Print a section heading."""
    print(f"\n{char * 78}\n{title}\n{char * 78}")


def show(df: pd.DataFrame, title: str | None = None, decimals: int = 4, index: bool = False) -> None:
    """Print a DataFrame without pandas truncating it to a width nobody can read."""
    if title:
        print(f"\n{title}")
    with pd.option_context("display.width", 220, "display.max_columns", 60):
        print(df.round(decimals).to_string(index=index))


def answer(question: str, response: str) -> None:
    """Print a written answer to an exercise's success criteria.

    The weeks use this for the "explain why" parts. Those are the parts that
    matter: a threshold sweep anyone can run, but the judgement about what the
    numbers license you to claim is the actual skill being built.
    """
    print(f"\nQ: {question}")
    # dedent, not per-line strip: the sub-points in these answers are indented
    # relative to each other, and a per-line strip flattens that structure away.
    for line in textwrap.dedent(response).strip("\n").splitlines():
        print(f"   {line}" if line.strip() else "")
