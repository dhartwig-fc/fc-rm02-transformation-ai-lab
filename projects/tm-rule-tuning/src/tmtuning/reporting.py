"""Generate a validation-ready tuning paper from analysis outputs.

The sections follow what a model risk / independent validation function
expects to be handed with a threshold change: what the rule is, what evidence
was gathered, what the evidence shows, what is being recommended, what the
residual risk is, and what had to be assumed. A tuning result that cannot be
written up in this shape is not finished.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

__all__ = ["markdown_table", "tuning_paper", "save_paper"]


def _fmt(value, spec: str | None = None) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "n/a"
    if isinstance(value, float) and np.isinf(value):
        return "none"
    if spec:
        return format(value, spec)
    if isinstance(value, (int, np.integer)):
        return f"{value:,}"
    if isinstance(value, (float, np.floating)):
        return f"{value:,.4f}"
    return str(value)


def markdown_table(
    df: pd.DataFrame,
    columns: Sequence[str] | None = None,
    formats: Mapping[str, str] | None = None,
    headers: Mapping[str, str] | None = None,
) -> str:
    """Render a DataFrame as a GitHub-flavoured markdown table.

    Hand-rolled rather than ``DataFrame.to_markdown`` so the package does not
    need ``tabulate`` just to emit a report.
    """
    columns = list(columns or df.columns)
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(f"columns not in frame: {missing}")

    formats = formats or {}
    headers = headers or {}

    head = "| " + " | ".join(headers.get(c, c) for c in columns) + " |"
    rule = "| " + " | ".join("---" for _ in columns) + " |"

    # Column-wise, not `iterrows()`. A row Series drawn from a mixed-dtype frame
    # is upcast to one common dtype, which silently turns integer alert counts
    # into floats and renders them as "10,000.0".
    cells = {c: df[c].tolist() for c in columns}
    body = [
        "| " + " | ".join(_fmt(cells[c][i], formats.get(c)) for c in columns) + " |"
        for i in range(len(df))
    ]
    return "\n".join([head, rule, *body])


def _recommendation_block(current: pd.Series | None, proposed: pd.Series) -> str:
    if current is None:
        return (
            f"Recommended threshold: **£{proposed['threshold']:,.0f}**, producing "
            f"{int(proposed['alerts']):,} alerts at {proposed['precision']:.1%} precision "
            f"and {proposed['recall']:.1%} recall."
        )

    d_alerts = int(proposed["alerts"] - current["alerts"])
    d_tp = int(proposed["tp"] - current["tp"])
    d_recall = proposed["recall"] - current["recall"]
    d_precision = proposed["precision"] - current["precision"]

    direction = "increase" if d_alerts > 0 else "reduction" if d_alerts < 0 else "no change"
    return "\n".join([
        f"Move the threshold from **£{current['threshold']:,.0f}** to "
        f"**£{proposed['threshold']:,.0f}**.",
        "",
        f"- Alert volume: {int(current['alerts']):,} -> {int(proposed['alerts']):,} "
        f"({d_alerts:+,}, a {abs(d_alerts)/max(current['alerts'],1):.1%} {direction})",
        f"- True cases detected: {int(current['tp']):,} -> {int(proposed['tp']):,} ({d_tp:+,})",
        # Deltas between two percentages are percentage POINTS. Writing "+0.34%"
        # for a move from 7.53% to 7.87% reads as a 0.34% relative change, which
        # is a different and much smaller claim.
        f"- Precision: {current['precision']:.2%} -> {proposed['precision']:.2%} ({d_precision * 100:+.2f}pp)",
        f"- Recall: {current['recall']:.2%} -> {proposed['recall']:.2%} ({d_recall * 100:+.2f}pp)",
        f"- Investigator effort per true case: {current['alerts_per_true_positive']:.1f} -> "
        f"{proposed['alerts_per_true_positive']:.1f} alerts",
    ])


def tuning_paper(
    rule_name: str,
    rule_logic: str,
    population: pd.DataFrame,
    sweep: pd.DataFrame,
    proposed: pd.Series,
    current: pd.Series | None = None,
    btl: Mapping | None = None,
    segment_comparison: pd.DataFrame | None = None,
    stability: pd.DataFrame | None = None,
    challenger: pd.DataFrame | None = None,
    overlap: pd.DataFrame | None = None,
    author: str = "",
    paper_date: date | None = None,
    label_basis: str = "Confirmed SAR/STR submission within 90 days of the alert period",
    limitations: Sequence[str] | None = None,
) -> str:
    """Assemble a full tuning paper in markdown.

    Only ``rule_name``, ``rule_logic``, ``population``, ``sweep`` and
    ``proposed`` are required; every other section is included when its evidence
    is supplied and omitted otherwise, with the omission recorded in the
    limitations section so a reviewer can see what was *not* done.
    """
    paper_date = paper_date or date.today()
    parts: list[str] = []
    skipped: list[str] = []

    parts.append(f"# Tuning Paper: {rule_name}\n")
    parts.append(f"**Date:** {paper_date.isoformat()}  ")
    parts.append(f"**Author:** {author or '_to be completed_'}  ")
    parts.append(f"**Status:** Draft for independent validation\n")

    parts.append("## 1. Recommendation\n")
    parts.append(_recommendation_block(current, proposed) + "\n")

    parts.append("## 2. Rule under review\n")
    parts.append(f"```\n{rule_logic}\n```\n")

    parts.append("## 3. Data and sample\n")
    prevalence = float(population["case"].mean()) if "case" in population else float("nan")
    parts.append(
        f"- Observations: {len(population):,} scored records\n"
        f"- True cases in sample: {int(population['case'].sum()):,} "
        f"({prevalence:.2%} base rate)\n"
        f"- Label basis: {label_basis}\n"
    )
    if "period" in population:
        periods = sorted(population["period"].unique())
        parts.append(f"- Period covered: {periods[0]} to {periods[-1]} ({len(periods)} periods)\n")

    parts.append("## 4. Methodology\n")
    parts.append(
        "Retrospective backtest over the sample above. The rule was re-executed at "
        "each candidate threshold and scored against the label set, producing the "
        "sweep in section 5. Threshold selection was constrained by operational "
        "alert capacity rather than chosen to maximise a single composite metric, "
        "so that the detection/effort trade-off is made explicitly rather than "
        "implied by a scoring function.\n"
    )

    parts.append("## 5. Threshold sweep\n")
    show = sweep.copy()
    if len(show) > 20:
        show = show.iloc[:: max(len(show) // 15, 1)]
    parts.append(markdown_table(
        show,
        columns=["threshold", "alerts", "tp", "fn", "precision", "recall", "alerts_per_true_positive"],
        formats={"threshold": ",.0f", "alerts": ",", "tp": ",", "fn": ",",
                 "precision": ".2%", "recall": ".2%", "alerts_per_true_positive": ".1f"},
        headers={"threshold": "Threshold (£)", "alerts": "Alerts", "tp": "True cases found",
                 "fn": "Missed", "precision": "Precision", "recall": "Recall",
                 "alerts_per_true_positive": "Alerts per case"},
    ) + "\n")

    section = 6
    if btl is not None:
        parts.append(f"## {section}. Below-the-line testing\n")
        parts.append(
            f"A simple random sample of {btl['sampled']:,} records was drawn from the "
            f"{btl['below_the_line_population']:,} records the rule does not alert on. "
            f"{btl['productive_in_sample']:,} were true cases.\n\n"
            f"- Observed below-the-line productive rate: **{btl['observed_rate']:.2%}**\n"
            f"- {btl['confidence']:.0%} Wilson confidence interval: "
            f"{btl['rate_lower']:.2%} to {btl['rate_upper']:.2%}\n"
            f"- Estimated cases missed across the full below-the-line population: "
            f"**{btl['estimated_missed_cases']:,.0f}** "
            f"(upper bound {btl['estimated_missed_cases_upper']:,.0f})\n\n"
            "The upper bound is the figure to test against risk appetite: it is the "
            "worst case the sample is consistent with, not the best guess.\n"
        )
        section += 1
    else:
        skipped.append("No below-the-line test was performed, so missed risk below the threshold is unquantified.")

    if segment_comparison is not None:
        parts.append(f"## {section}. Segment calibration\n")
        parts.append(markdown_table(
            segment_comparison,
            columns=["segment", "threshold_uniform", "alerts_uniform", "tp_uniform",
                     "threshold_segmented", "alerts_segmented", "tp_segmented", "uplift_tp"],
            formats={"threshold_uniform": ",.0f", "threshold_segmented": ",.0f",
                     "alerts_uniform": ",", "alerts_segmented": ",",
                     "tp_uniform": ",", "tp_segmented": ",", "uplift_tp": "+,"},
            headers={"segment": "Segment", "threshold_uniform": "Global £", "alerts_uniform": "Alerts",
                     "tp_uniform": "Cases", "threshold_segmented": "Calibrated £",
                     "alerts_segmented": "Alerts", "tp_segmented": "Cases", "uplift_tp": "Uplift"},
        ) + "\n\nBoth options are costed at the same total alert budget, so the uplift "
            "column is additional detection for no additional investigator effort.\n")
        section += 1
    else:
        skipped.append("Segment-level calibration was not assessed; a single global threshold is assumed appropriate across all customer segments.")

    if stability is not None:
        parts.append(f"## {section}. Stability and drift\n")
        breaches = int(stability["any_breach"].sum()) if "any_breach" in stability else 0
        parts.append(
            f"Performance was recomputed independently in each period. "
            f"{breaches} of {len(stability)} periods breached at least one control limit.\n\n"
        )
        parts.append(markdown_table(
            stability,
            columns=["period", "alerts", "tp", "precision", "recall", "psi", "band", "any_breach"],
            formats={"alerts": ",", "tp": ",", "precision": ".2%", "recall": ".2%", "psi": ".3f"},
            headers={"period": "Period", "alerts": "Alerts", "tp": "Cases", "precision": "Precision",
                     "recall": "Recall", "psi": "PSI", "band": "Stability", "any_breach": "Breach"},
        ) + "\n")
        section += 1
    else:
        skipped.append("Period-by-period stability was not assessed; the result is a pooled figure that may conceal drift.")

    if challenger is not None:
        parts.append(f"## {section}. Champion vs challenger\n")
        cols = [c for c in ["alerts", "tp", "fn", "precision", "recall"] if c in challenger.columns]
        parts.append(markdown_table(
            challenger.reset_index(),
            columns=["rule", *cols],
            formats={"alerts": ",", "tp": ",", "fn": ",", "precision": ".2%", "recall": ".2%"},
        ) + "\n")
        if overlap is not None:
            parts.append(
                "\nNetted metrics conceal reallocation, so the overlap is broken out below. "
                "The 'Champion only' row is risk the challenger would newly miss.\n\n"
            )
            parts.append(markdown_table(
                overlap,
                columns=["cell", "rows", "cases", "share_of_cases", "precision"],
                formats={"rows": ",", "cases": ",", "share_of_cases": ".2%", "precision": ".2%"},
                headers={"cell": "Cell", "rows": "Records", "cases": "True cases",
                         "share_of_cases": "Share of all cases", "precision": "Precision"},
            ) + "\n")
        section += 1
    else:
        skipped.append("No challenger rule was tested; the recommendation is a threshold move on the incumbent logic only.")

    parts.append(f"## {section}. Limitations and assumptions\n")
    standing = [
        "Labels are a proxy for true financial crime. A record with no SAR/STR is "
        "not proven clean -- it may be undetected risk, which biases measured "
        "recall upward. Every figure in this paper is conditional on the label set.",
        "Backtesting assumes historical behaviour is representative of the forward "
        "period. Any known upcoming change in product, customer mix or typology "
        "invalidates that assumption and should be raised before implementation.",
    ]
    for item in [*standing, *skipped, *(limitations or [])]:
        parts.append(f"- {item}\n")
    section += 1

    parts.append(f"\n## {section}. Approval\n")
    parts.append(
        "| Role | Name | Date | Decision |\n| --- | --- | --- | --- |\n"
        "| Tuning analyst | | | |\n| Financial crime risk owner | | | |\n"
        "| Independent model validation | | | |\n"
    )

    return "\n".join(parts)


def save_paper(path: str | Path, content: str) -> Path:
    """Write a generated paper to disk, creating parent directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path
