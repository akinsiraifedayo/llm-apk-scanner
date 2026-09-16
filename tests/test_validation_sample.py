"""Tests for validation sub-sample drawing.

The sample exists to measure the filter against independent ground truth, so
two properties matter beyond "it returns rows": the draw must be reproducible,
and the coder's worksheet must not leak the prediction it is meant to check.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from llm_apk_scanner.validation.sample import (
    draw_validation_sample,
    read_findings,
    read_key,
    write_sample,
)


def _rows(n_flagged: int, n_unflagged: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for i in range(n_flagged):
        rows.append({
            "sha256": f"F{i:063d}", "pkg_name": f"com.flagged{i}", "version": "1.0",
            "apk_size": 1000, "llm_integrated": True, "providers": ["openai"],
            "n_findings": 3,
        })
    for i in range(n_unflagged):
        rows.append({
            "sha256": f"U{i:063d}", "pkg_name": f"com.plain{i}", "version": "1.0",
            "apk_size": 1000, "llm_integrated": False, "providers": [],
            "n_findings": 0,
        })
    return rows


def test_draws_equal_strata():
    sample = draw_validation_sample(_rows(300, 300), per_stratum=100, seed=1)
    assert len(sample.flagged) == 100
    assert len(sample.unflagged) == 100
    assert sample.size == 200
    assert not sample.is_short


def test_draw_is_reproducible_for_a_given_seed():
    first = draw_validation_sample(_rows(300, 300), per_stratum=50, seed=7)
    second = draw_validation_sample(_rows(300, 300), per_stratum=50, seed=7)
    assert [r["sha256"] for r in first.all_rows()] == [
        r["sha256"] for r in second.all_rows()
    ]


def test_different_seeds_draw_differently():
    first = draw_validation_sample(_rows(300, 300), per_stratum=50, seed=1)
    second = draw_validation_sample(_rows(300, 300), per_stratum=50, seed=2)
    assert [r["sha256"] for r in first.all_rows()] != [
        r["sha256"] for r in second.all_rows()
    ]


def test_draw_does_not_depend_on_input_order():
    rows = _rows(200, 200)
    forward = draw_validation_sample(rows, per_stratum=40, seed=3)
    backward = draw_validation_sample(list(reversed(rows)), per_stratum=40, seed=3)
    assert [r["sha256"] for r in forward.all_rows()] == [
        r["sha256"] for r in backward.all_rows()
    ]


def test_short_pool_is_reported_not_silently_accepted():
    """A thin sample gives wide intervals; that must be visible."""
    sample = draw_validation_sample(_rows(3, 44), per_stratum=100, seed=1)
    assert sample.is_short
    assert len(sample.flagged) == 3
    assert sample.pool_flagged == 3
    assert sample.pool_unflagged == 44


def test_full_pool_is_not_flagged_short():
    sample = draw_validation_sample(_rows(100, 100), per_stratum=100, seed=1)
    assert not sample.is_short


class TestBlindWorksheet:
    """The coder must not see what the scanner predicted."""

    def _write(self, tmp_path: Path):
        sample = draw_validation_sample(_rows(10, 10), per_stratum=5, seed=1)
        label_path = tmp_path / "sample_labels.csv"
        key_path = tmp_path / "sample_key.csv"
        write_sample(sample, label_path=label_path, key_path=key_path)
        return label_path, key_path

    def test_worksheet_omits_the_prediction(self, tmp_path: Path):
        label_path, _ = self._write(tmp_path)
        text = label_path.read_text()
        assert "llm_integrated" not in text
        assert "providers" not in text
        assert "n_findings" not in text

    def test_worksheet_omits_prediction_bearing_values(self, tmp_path: Path):
        """Not just the column names — no 'openai' leaking through either."""
        label_path, _ = self._write(tmp_path)
        assert "openai" not in label_path.read_text().lower()

    def test_worksheet_still_identifies_each_apk(self, tmp_path: Path):
        label_path, _ = self._write(tmp_path)
        rows = list(csv.DictReader(label_path.open()))
        assert len(rows) == 10
        assert all(row["sha256"] and row["pkg_name"] for row in rows)

    def test_key_retains_the_prediction(self, tmp_path: Path):
        _, key_path = self._write(tmp_path)
        predictions = read_key(key_path)
        assert len(predictions) == 10
        assert sum(predictions.values()) == 5

    def test_worksheet_and_key_cover_the_same_apks(self, tmp_path: Path):
        label_path, key_path = self._write(tmp_path)
        worksheet = {r["sha256"] for r in csv.DictReader(label_path.open())}
        assert worksheet == set(read_key(key_path))

    def test_worksheet_interleaves_strata(self, tmp_path: Path):
        """Ordered by SHA-256, so position does not reveal the stratum.

        If every flagged APK came first the coder could infer the answer from
        where they were in the list rather than from the evidence.
        """
        label_path, key_path = self._write(tmp_path)
        predictions = read_key(key_path)
        order = [predictions[r["sha256"]] for r in csv.DictReader(label_path.open())]
        assert order != sorted(order, reverse=True), "flagged APKs are front-loaded"


def test_read_findings_skips_blank_lines(tmp_path: Path):
    path = tmp_path / "findings.jsonl"
    path.write_text(
        json.dumps({"sha256": "A", "llm_integrated": True}) + "\n\n"
        + json.dumps({"sha256": "B", "llm_integrated": False}) + "\n"
    )
    assert len(read_findings(path)) == 2
