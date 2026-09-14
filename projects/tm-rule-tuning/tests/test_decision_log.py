"""The Tuning Decision Log (Week 10's recommended habit)."""

import pytest

from tmtuning.decision_log import DecisionEntry, TuningDecisionLog


@pytest.fixture
def log():
    entry_log = TuningDecisionLog(rule="TM-021", author="A. Analyst")
    entry_log.record("1. Baseline", "cash > £10k", alerts=26_897, precision=0.0771,
                     recall=0.6993, fpr=0.2558, outcome="for information",
                     rationale="Incumbent as running.")
    entry_log.record("2. Sweep", "cash > £20k", alerts=13_523, precision=0.0994,
                     recall=0.4531, outcome="adopted", rationale="Fits capacity.")
    entry_log.record("2. Sweep", "cash > £30k", alerts=7_251, precision=0.1185,
                     recall=0.2896, outcome="rejected", rationale="Gives up too much recall.")
    return entry_log


def test_records_every_option_in_order(log):
    assert len(log) == 3
    assert log.to_frame()["option"].tolist() == ["cash > £10k", "cash > £20k", "cash > £30k"]


def test_metrics_dict_populates_the_entry():
    entry_log = TuningDecisionLog(rule="TM-021")
    metrics = {"alerts": 100, "precision": 0.2, "recall": 0.5, "fpr": 0.01, "tp": 20}
    entry = entry_log.record("step", "option", metrics)
    assert (entry.alerts, entry.precision, entry.recall, entry.fpr) == (100, 0.2, 0.5, 0.01)


def test_explicit_arguments_override_the_metrics_dict():
    """A segmented option's alert count is a sum across segments, not the dict's."""
    entry_log = TuningDecisionLog(rule="TM-021")
    entry = entry_log.record("step", "option", {"alerts": 100, "precision": 0.2}, alerts=250)
    assert entry.alerts == 250
    assert entry.precision == 0.2


def test_rejects_an_unknown_outcome():
    with pytest.raises(ValueError, match="outcome must be one of"):
        DecisionEntry(step="s", option="o", outcome="maybe")


def test_adopted_returns_only_adopted_options(log):
    adopted = log.adopted()
    assert len(adopted) == 1
    assert adopted.iloc[0]["option"] == "cash > £20k"


def test_markdown_keeps_the_rejected_options(log):
    """The whole point: the log records what lost, not just what won."""
    markdown = log.to_markdown()
    assert "cash > £30k" in markdown
    assert "rejected" in markdown
    assert "Gives up too much recall." in markdown


def test_markdown_groups_by_step(log):
    markdown = log.to_markdown()
    assert "## 1. Baseline" in markdown
    assert "## 2. Sweep" in markdown
    assert markdown.index("## 1. Baseline") < markdown.index("## 2. Sweep")


def test_empty_log_renders_without_error():
    markdown = TuningDecisionLog(rule="TM-021").to_markdown()
    assert "No options recorded" in markdown


def test_empty_log_frame_has_the_expected_columns():
    frame = TuningDecisionLog(rule="TM-021").to_frame()
    assert "option" in frame.columns and "rationale" in frame.columns
    assert frame.empty


def test_save_writes_markdown(tmp_path, log):
    path = log.save(tmp_path / "nested" / "log.md")
    assert path.exists()
    assert "Tuning Decision Log: TM-021" in path.read_text()
