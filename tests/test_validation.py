"""Tests for the validation harness."""

from __future__ import annotations

from pathlib import Path

from llm_apk_scanner.validation import (
    LabelRecord,
    LabelStore,
    Metrics,
    compute_metrics,
    render_confusion_matrix,
)


def test_label_store_append_and_iterate(tmp_path: Path) -> None:
    store = LabelStore(tmp_path / "labels.jsonl")
    store.append(
        LabelRecord(
            apk_sha256="abc",
            apk_path="/tmp/x.apk",
            is_llm_integrated=True,
            notes="OpenAI key visible in classes.dex",
            labelled_at="2026-04-30T10:00:00Z",
        )
    )
    store.append(
        LabelRecord(
            apk_sha256="def",
            apk_path="/tmp/y.apk",
            is_llm_integrated=False,
        )
    )
    records = list(store)
    assert len(records) == 2
    assert records[0].apk_sha256 == "abc"
    assert records[1].is_llm_integrated is False


def test_label_store_uses_latest_label(tmp_path: Path) -> None:
    store = LabelStore(tmp_path / "labels.jsonl")
    store.append(LabelRecord(apk_sha256="x", apk_path="a", is_llm_integrated=False))
    store.append(LabelRecord(apk_sha256="x", apk_path="a", is_llm_integrated=True))
    latest = store.latest_for("x")
    assert latest is not None
    assert latest.is_llm_integrated is True


def test_compute_metrics_basic() -> None:
    predicted = {"a": True, "b": True, "c": False, "d": False}
    actual = {"a": True, "b": False, "c": True, "d": False}
    m = compute_metrics(predicted, actual)
    assert m.tp == 1  # a
    assert m.fp == 1  # b
    assert m.fn == 1  # c
    assert m.tn == 1  # d
    assert m.precision == 0.5
    assert m.recall == 0.5
    assert m.f1 == 0.5


def test_metrics_uncertain_excluded() -> None:
    predicted = {"a": True, "b": True}
    actual = {"a": True, "b": None}  # b uncertain
    m = compute_metrics(predicted, actual)
    assert m.tp == 1
    assert m.fp == 0
    assert m.tp + m.fp + m.fn + m.tn == 1


def test_metrics_zero_division_safe() -> None:
    m = Metrics(tp=0, fp=0, fn=0, tn=10)
    assert m.precision == 0.0
    assert m.recall == 0.0
    assert m.f1 == 0.0


def test_render_confusion_matrix_contains_counts() -> None:
    m = Metrics(tp=15, fp=3, fn=2, tn=80)
    rendered = render_confusion_matrix(m)
    assert "TP = 15" in rendered
    assert "FP = 3" in rendered
    assert "FN = 2" in rendered
    assert "TN = 80" in rendered
    assert "Precision" in rendered
    assert "Recall" in rendered
