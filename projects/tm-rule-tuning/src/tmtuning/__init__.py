"""tmtuning -- a small toolkit for transaction monitoring rule tuning.

Built for the 10-week Advanced Transaction Monitoring Rule Tuning curriculum in
``../LEARNING_PLAN.md``. Each module maps to the weeks that use it:

===========  =====================================================
Module       Weeks
===========  =====================================================
metrics      1 (confusion matrix, precision/recall, why not accuracy)
data         2, 10 (synthetic populations)
thresholds   2, 3, 4 (sweeps, capacity-constrained selection)
segments     5 (segment calibration)
challenger   6, 8 (ATL/BTL testing, champion vs challenger)
stability    7 (PSI, drift, control charts)
plots        4, 5, 7 (curves and control charts)
reporting    9, 10 (validation-ready tuning papers)
===========  =====================================================
"""

from .metrics import (
    ConfusionCounts, classification_metrics, confusion_counts, confusion_frame, metrics_frame, safe_divide,
)
from .data import SEGMENTS, generate_population, spec_population, split_by_period
from .thresholds import (
    apply_rule, capacity_frontier, default_thresholds, marginal_yield, optimise_threshold, threshold_sweep,
)
from .segments import (
    calibrate_segments, compare_uniform_vs_segmented, segment_summary, segment_sweeps, uniform_threshold_at_capacity,
)
from .stability import control_limits, period_performance, psi, psi_by_period, stability_report
from .challenger import (
    atl_btl_sample, btl_test, compare_rules, required_sample_size, rule_overlap, wilson_interval,
)
from .reporting import markdown_table, save_paper, tuning_paper
from .teaching import answer, banner, show

__version__ = "0.1.0"

__all__ = [
    "ConfusionCounts", "classification_metrics", "confusion_counts", "confusion_frame",
    "metrics_frame", "safe_divide",
    "SEGMENTS", "generate_population", "spec_population", "split_by_period",
    "apply_rule", "capacity_frontier", "default_thresholds", "marginal_yield",
    "optimise_threshold", "threshold_sweep",
    "calibrate_segments", "compare_uniform_vs_segmented", "segment_summary",
    "segment_sweeps", "uniform_threshold_at_capacity",
    "control_limits", "period_performance", "psi", "psi_by_period", "stability_report",
    "atl_btl_sample", "btl_test", "compare_rules", "required_sample_size",
    "rule_overlap", "wilson_interval",
    "markdown_table", "save_paper", "tuning_paper",
    "answer", "banner", "show",
]
