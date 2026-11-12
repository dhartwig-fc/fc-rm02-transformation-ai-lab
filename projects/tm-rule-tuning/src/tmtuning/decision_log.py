"""The Tuning Decision Log.

The habit the course recommends, made into a tool: record every threshold
tested, the metric changes observed, the operational implications, the
assumptions made and the final rationale.

The point is not tidiness. A tuning paper reports the option that won; the
decision log reports the options that lost and why, which is the part a
validator, a compliance reviewer or an auditor actually needs three years later
when the author has moved on and the question is "why £15,000?".

Keeping it as you go costs nothing. Reconstructing it afterwards is guesswork
dressed as evidence -- and it is obvious to a reader which of the two happened.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import date
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

__all__ = ["DecisionEntry", "TuningDecisionLog"]

#: What happened to an option once it was tested.
OUTCOMES = ("adopted", "rejected", "carried forward", "for information")


@dataclass
class DecisionEntry:
    """One tested option and everything a reviewer needs to re-derive it."""

    step: str
    option: str
    alerts: int | None = None
    precision: float | None = None
    recall: float | None = None
    fpr: float | None = None
    operational_implication: str = ""
    assumptions: str = ""
    outcome: str = "for information"
    rationale: str = ""

    def __post_init__(self) -> None:
        if self.outcome not in OUTCOMES:
            raise ValueError(f"outcome must be one of {OUTCOMES}, got {self.outcome!r}")

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class TuningDecisionLog:
    """An ordered record of every option considered during a tuning exercise.

    >>> log = TuningDecisionLog(rule="TM-021", author="A. Analyst")
    >>> _ = log.record("Baseline", "cash > £10k", alerts=26_897, precision=0.0771,
    ...                outcome="for information", rationale="Incumbent, as running.")
    >>> len(log)
    1
    """

    rule: str
    author: str = ""
    log_date: date = field(default_factory=date.today)
    entries: list[DecisionEntry] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.entries)

    def record(
        self,
        step: str,
        option: str,
        metrics: Mapping | None = None,
        *,
        alerts: int | None = None,
        precision: float | None = None,
        recall: float | None = None,
        fpr: float | None = None,
        operational_implication: str = "",
        assumptions: str = "",
        outcome: str = "for information",
        rationale: str = "",
    ) -> DecisionEntry:
        """Add one tested option.

        Pass ``metrics`` (the dict from
        :func:`~tmtuning.metrics.classification_metrics`) to fill the numbers
        automatically, or give them individually. Explicit arguments win, so a
        segmented option whose alert count is a sum across segments can override
        the metric dict.
        """
        if metrics is not None:
            alerts = metrics.get("alerts") if alerts is None else alerts
            precision = metrics.get("precision") if precision is None else precision
            recall = metrics.get("recall") if recall is None else recall
            fpr = metrics.get("fpr") if fpr is None else fpr

        entry = DecisionEntry(
            step=step, option=option,
            alerts=None if alerts is None else int(alerts),
            precision=precision, recall=recall, fpr=fpr,
            operational_implication=operational_implication,
            assumptions=assumptions, outcome=outcome, rationale=rationale,
        )
        self.entries.append(entry)
        return entry

    def to_frame(self) -> pd.DataFrame:
        """The log as a DataFrame, in the order options were tested."""
        if not self.entries:
            return pd.DataFrame(columns=[f.name for f in DecisionEntry.__dataclass_fields__.values()])
        return pd.DataFrame([e.as_dict() for e in self.entries])

    def adopted(self) -> pd.DataFrame:
        """Only the options that were adopted -- what the paper recommends."""
        frame = self.to_frame()
        return frame[frame["outcome"] == "adopted"] if len(frame) else frame

    def to_markdown(self) -> str:
        """Render the log for inclusion in a tuning paper or evidence pack."""
        from .reporting import markdown_table

        lines = [
            f"# Tuning Decision Log: {self.rule}\n",
            f"**Date:** {self.log_date.isoformat()}  ",
            f"**Author:** {self.author or '_to be completed_'}  ",
            f"**Options tested:** {len(self.entries)}\n",
            "Every option considered during this exercise, including those rejected.",
            "Recorded as the work was done, not reconstructed afterwards.\n",
        ]

        frame = self.to_frame()
        if frame.empty:
            lines.append("_No options recorded._\n")
            return "\n".join(lines)

        for step in frame["step"].unique():
            part = frame[frame["step"] == step]
            lines.append(f"\n## {step}\n")
            lines.append(markdown_table(
                part,
                columns=["option", "alerts", "precision", "recall", "fpr", "outcome"],
                formats={"alerts": ",", "precision": ".2%", "recall": ".2%", "fpr": ".2%"},
                headers={"option": "Option tested", "alerts": "Alerts", "precision": "Precision",
                         "recall": "Recall", "fpr": "FPR", "outcome": "Outcome"},
            ) + "\n")
            for _, row in part.iterrows():
                details = [
                    ("Operational implication", row["operational_implication"]),
                    ("Assumptions", row["assumptions"]),
                    ("Rationale", row["rationale"]),
                ]
                written = [f"  - _{label}:_ {text}" for label, text in details if text]
                if written:
                    lines.append(f"\n- **{row['option']}**")
                    lines.extend(written)
            lines.append("")

        return "\n".join(lines)

    def save(self, path: str | Path) -> Path:
        """Write the log to disk as markdown."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_markdown(), encoding="utf-8")
        return path
