"""Charts for tuning analysis.

Colours are taken unchanged from the validated reference palette (categorical
slots in fixed order, status colours reserved for breach states). The slot order
is the colour-vision-deficiency safety mechanism, so series are assigned slots
in order and never cycled.

These render to static PNG for tuning papers, so there is no hover layer. The
compensating rule applies throughout: every chart with two or more series
carries a legend *and* direct labels, so identity is never colour-alone, and
each function returns the underlying frame so a table view is always available.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Headless: these are written to file, never shown interactively.

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

__all__ = ["PALETTE", "STATUS", "INK", "save", "plot_alert_volume_curve", "plot_risk_yield_curve",
           "plot_precision_recall_tradeoff", "plot_segment_curves", "plot_stability_chart"]

#: Categorical slots, light mode, in validated order. Assign in order; never cycle.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
           "#e87ba4", "#008300", "#4a3aa7", "#e34948"]

#: Reserved status colours -- never reused as a series colour.
STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}

#: Text and surface tokens. All chart text uses these, never a series colour.
INK = {"primary": "#0b0b0b", "secondary": "#52514e", "muted": "#8a8880",
       "surface": "#fcfcfb", "grid": "#e5e4e0"}

_LINE_WIDTH = 2.0
_MARKER_SIZE = 6.5


def _style(ax: plt.Axes, title: str, xlabel: str, ylabel: str, subtitle: str | None = None) -> None:
    """Apply the recessive-chrome house style: quiet grid, no box, ink text."""
    ax.set_facecolor(INK["surface"])
    ax.figure.set_facecolor(INK["surface"])
    ax.grid(True, color=INK["grid"], linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK["grid"])
    ax.tick_params(colors=INK["secondary"], labelsize=9, length=0)
    ax.set_xlabel(xlabel, color=INK["secondary"], fontsize=10)
    ax.set_ylabel(ylabel, color=INK["secondary"], fontsize=10)
    if subtitle:
        ax.set_title(title, color=INK["primary"], fontsize=13, fontweight="bold", loc="left", pad=22)
        ax.text(0, 1.02, subtitle, transform=ax.transAxes, color=INK["secondary"], fontsize=10, va="bottom")
    else:
        ax.set_title(title, color=INK["primary"], fontsize=13, fontweight="bold", loc="left", pad=10)


def _gbp(ax: plt.Axes) -> None:
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda v, _: f"£{v/1_000_000:.1f}m" if v >= 1_000_000 else f"£{v/1_000:.0f}k"))


def _pct(ax: plt.Axes) -> None:
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0%}"))


def _legend(ax: plt.Axes) -> None:
    leg = ax.legend(frameon=False, fontsize=9, loc="best")
    for text in leg.get_texts():
        text.set_color(INK["secondary"])


def _place_end_labels(ax: plt.Axes, entries: list[tuple[float, float, str]],
                      min_gap_frac: float = 0.055) -> None:
    """Draw direct labels at line ends, nudged apart so they cannot overlap.

    Two series ending at similar values would otherwise print their labels on
    top of each other. Labels are pushed up in order of their true value, so the
    vertical ordering still matches the data; the coloured marker stays on the
    real end point and carries the identity.
    """
    low, high = ax.get_ylim()
    gap = (high - low) * min_gap_frac

    placed: list[float] = []
    for _, y, _ in sorted(entries, key=lambda e: e[1]):
        y = max(y, placed[-1] + gap) if placed else y
        placed.append(y)

    for (x, _, text), y in zip(sorted(entries, key=lambda e: e[1]), placed):
        ax.annotate(text, xy=(x, y), textcoords="offset points", xytext=(9, 0),
                    color=INK["primary"], fontsize=9, va="center", annotation_clip=False)


def save(fig: plt.Figure, path: str | Path) -> Path:
    """Write a figure to PNG, creating the parent directory if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=INK["surface"])
    plt.close(fig)
    return path


def plot_alert_volume_curve(sweep: pd.DataFrame, capacity: int | None = None,
                            title: str = "Alert volume by threshold") -> plt.Figure:
    """Alert volume against threshold, with an optional operational capacity line.

    One series, so no legend -- the title names it. The capacity line is drawn in
    the reserved ``warning`` status colour with a label, because it marks a
    constraint state rather than another series.

    When the curve spans more than an order of magnitude above the capacity
    line, the y-axis switches to log. On a linear axis a peak of 240,000 alerts
    crushes a 6,300 capacity line onto the baseline, so the one thing the chart
    exists to show -- where the curve crosses capacity -- becomes unreadable.
    The subtitle says which scale is in use, because a log axis that is not
    announced is its own way of misleading.
    """
    fig, ax = plt.subplots(figsize=(8, 4.6))
    alerts = sweep["alerts"].clip(lower=0)
    ax.plot(sweep["threshold"], alerts, color=PALETTE[0], linewidth=_LINE_WIDTH, zorder=3)

    log_scale = capacity is not None and capacity > 0 and alerts.max() > 10 * capacity
    if log_scale:
        ax.set_yscale("log")
        ax.set_ylim(bottom=max(alerts[alerts > 0].min(), capacity / 20))

    if capacity is not None:
        ax.axhline(capacity, color=STATUS["warning"], linewidth=_LINE_WIDTH, linestyle="--", zorder=2)
        ax.text(sweep["threshold"].max(), capacity, f"  Capacity {capacity:,}",
                color=INK["secondary"], fontsize=9, va="bottom", ha="right")

    subtitle = ("Volume falls steeply at first, then flattens - past the knee, "
                "tightening further buys little relief")
    if log_scale:
        subtitle = ("Log scale: the curve spans two orders of magnitude, so a linear "
                    "axis would hide the capacity crossing")

    _style(ax, title, "Threshold", "Alerts" + (" (log scale)" if log_scale else ""), subtitle=subtitle)
    _gbp(ax)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    fig.tight_layout()
    return fig


def plot_risk_yield_curve(sweep: pd.DataFrame, title: str = "Risk yield by threshold") -> plt.Figure:
    """Precision and recall against threshold on a single shared axis.

    Both series are proportions, so they share one y-axis honestly. This chart
    is never drawn with two y-scales: a dual axis lets the two curves be slid
    against each other until they tell whichever story is wanted.
    """
    fig, ax = plt.subplots(figsize=(8, 4.6))

    series = [("Precision (alert yield)", "precision", PALETTE[0]),
              ("Recall (detection rate)", "recall", PALETTE[1])]

    end_labels: list[tuple[float, float, str]] = []
    for label, col, colour in series:
        ax.plot(sweep["threshold"], sweep[col], color=colour, linewidth=_LINE_WIDTH, label=label, zorder=3)
        last = sweep.dropna(subset=[col]).iloc[-1]
        ax.plot(last["threshold"], last[col], "o", color=colour, markersize=_MARKER_SIZE, zorder=4)
        end_labels.append((float(last["threshold"]), float(last[col]), label.split(" (")[0]))

    _style(ax, title, "Threshold", "Rate",
           subtitle="Tightening the threshold trades detection away for yield - "
                    "the tuning decision is where on this trade to sit")
    _gbp(ax)
    _pct(ax)
    ax.set_xlim(right=sweep["threshold"].max() * 1.22)
    _place_end_labels(ax, end_labels)
    _legend(ax)
    fig.tight_layout()
    return fig


def plot_precision_recall_tradeoff(sweep: pd.DataFrame, annotate_every: int = 6,
                                   title: str = "Precision-recall trade-off") -> plt.Figure:
    """The trade-off frontier, with selected thresholds labelled on the curve.

    Labels are placed on every *n*-th point, never on all of them -- a number on
    every point turns the curve into a wall of text and hides the shape that the
    chart exists to show.
    """
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    clean = sweep.dropna(subset=["precision", "recall"])

    ax.plot(clean["recall"], clean["precision"], color=PALETTE[0], linewidth=_LINE_WIDTH, zorder=3)
    ax.plot(clean["recall"], clean["precision"], "o", color=PALETTE[0],
            markersize=4, zorder=4, markeredgecolor=INK["surface"], markeredgewidth=1)

    base_rate = clean["prevalence"].iloc[0] if "prevalence" in clean else None
    if base_rate is not None and not np.isnan(base_rate):
        ax.axhline(base_rate, color=INK["muted"], linewidth=1.2, linestyle=":", zorder=2)
        ax.text(clean["recall"].max(), base_rate, f"  Base rate {base_rate:.1%}",
                color=INK["secondary"], fontsize=9, va="bottom", ha="right")

    for _, row in clean.iloc[::annotate_every].iterrows():
        ax.annotate(f"£{row['threshold']/1000:,.0f}k", (row["recall"], row["precision"]),
                    textcoords="offset points", xytext=(6, 6), color=INK["secondary"], fontsize=8)

    _style(ax, title, "Recall (detection rate)", "Precision (alert yield)",
           subtitle="Each point is one threshold. A curve hugging the base rate "
                    "means the variable carries no risk signal")
    _pct(ax)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0%}"))
    fig.tight_layout()
    return fig


def plot_segment_curves(sweeps: dict[str, pd.DataFrame], x: str = "alert_rate", y: str = "precision",
                        title: str = "Yield curves by segment") -> plt.Figure:
    """One curve per segment, direct-labelled at the line end.

    Capped at the first eight palette slots. A ninth segment is not given a
    generated colour -- fold the smallest segments into "Other" or facet the
    chart instead.
    """
    if len(sweeps) > len(PALETTE):
        raise ValueError(
            f"{len(sweeps)} segments exceeds the {len(PALETTE)} validated palette slots. "
            "Fold the smallest into 'Other', or draw small multiples."
        )

    fig, ax = plt.subplots(figsize=(8.2, 5))
    end_labels: list[tuple[float, float, str]] = []
    for slot, (name, sweep) in enumerate(sorted(sweeps.items())):
        colour = PALETTE[slot]
        clean = sweep.dropna(subset=[x, y]).sort_values(x)
        ax.plot(clean[x], clean[y], color=colour, linewidth=_LINE_WIDTH, label=name, zorder=3)
        last = clean.iloc[-1]
        ax.plot(last[x], last[y], "o", color=colour, markersize=_MARKER_SIZE, zorder=4)
        end_labels.append((float(last[x]), float(last[y]), name))

    _style(ax, title, "Alert rate within segment", "Precision (alert yield)",
           subtitle="Separated curves mean one global threshold cannot serve every segment")
    _pct(ax)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_xlim(right=max(s[x].max() for s in sweeps.values()) * 1.25)
    _place_end_labels(ax, end_labels)
    _legend(ax)
    fig.tight_layout()
    return fig


def plot_stability_chart(report: pd.DataFrame, metric: str = "alerts", period_col: str = "period",
                         limits_key: str = "volume_limits", title: str | None = None) -> plt.Figure:
    """Control chart for one monitoring metric across periods.

    The baseline band is drawn from limits anchored to the tuning window (see
    :func:`~tmtuning.stability.stability_report`). Breaching points are marked in
    the reserved ``critical`` status colour *and* given a distinct marker and a
    legend entry, so the breach state never depends on colour alone.
    """
    limits = report.attrs.get(limits_key)
    if limits is None:
        raise KeyError(
            f"report has no {limits_key!r} in .attrs -- pass the frame returned by "
            "stability_report(), and note that slicing a DataFrame drops .attrs."
        )

    breach_col = {"alerts": "alert_volume_breach", "precision": "precision_breach"}.get(metric)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    x = np.arange(len(report))

    ax.fill_between(x, limits["lower"], limits["upper"], color=PALETTE[0], alpha=0.10, zorder=1,
                    label=f"Baseline control band ({limits['k']:g}$\\sigma$)")
    ax.axhline(limits["centre"], color=INK["muted"], linewidth=1.2, linestyle="--", zorder=2)
    ax.plot(x, report[metric], color=PALETTE[0], linewidth=_LINE_WIDTH, zorder=3, label=metric.title())

    if breach_col and breach_col in report:
        breached = report[breach_col].to_numpy(dtype=bool)
        if breached.any():
            ax.plot(x[breached], report[metric].to_numpy()[breached], "D", color=STATUS["critical"],
                    markersize=_MARKER_SIZE, markeredgecolor=INK["surface"], markeredgewidth=1.2,
                    zorder=5, linestyle="none", label="Outside control limits")

    ax.set_xticks(x)
    ax.set_xticklabels(report[period_col], rotation=45, ha="right")
    _style(ax, title or f"{metric.title()} stability", "Period", metric.title(),
           subtitle="Limits are anchored to the tuning baseline, not to the whole series - "
                    "otherwise drift moves the limits with it")
    if metric == "precision":
        _pct(ax)
    else:
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    _legend(ax)
    fig.tight_layout()
    return fig
