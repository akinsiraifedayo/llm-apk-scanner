"""Tests for the durable corpus ledger."""

from __future__ import annotations

from llm_apk_scanner.corpus.ledger import (
    DOWNLOAD_FAILED,
    PENDING,
    SCANNED,
    Ledger,
    ScanOutcome,
)
from llm_apk_scanner.corpus.model import ApkRecord


def _record(sha: str, pkg: str = "com.example.ai", year: int = 2025) -> ApkRecord:
    return ApkRecord(
        sha256=sha,
        pkg_name=pkg,
        markets="play.google.com",
        dex_date=f"{year}-06-01 00:00:00",
        apk_size=1024,
    )


def test_enroll_is_idempotent(tmp_path):
    with Ledger(tmp_path / "l.db") as ledger:
        records = [_record("A" * 64), _record("B" * 64)]
        assert ledger.enroll(records) == 2
        # Re-enrolling the same records adds nothing.
        assert ledger.enroll(records) == 0
        assert ledger.counts() == {PENDING: 2}


def test_enroll_preserves_state_of_existing_rows(tmp_path):
    """Extending the manifest must not reset work already done."""
    with Ledger(tmp_path / "l.db") as ledger:
        ledger.enroll([_record("A" * 64)])
        ledger.record_scan(ScanOutcome(sha256="A" * 64, pkg_name="com.done"))
        ledger.enroll([_record("A" * 64), _record("B" * 64)])
        assert ledger.counts() == {SCANNED: 1, PENDING: 1}


def test_pending_excludes_scanned_but_includes_downloaded(tmp_path):
    """A run interrupted after download resumes rather than skipping."""
    with Ledger(tmp_path / "l.db") as ledger:
        ledger.enroll([_record("A" * 64), _record("B" * 64)])
        ledger.mark_downloaded("A" * 64)
        ledger.record_scan(ScanOutcome(sha256="B" * 64))
        shas = {r.sha256 for r in ledger.pending()}
        assert shas == {"A" * 64}


def test_pending_holds_back_repeatedly_failing_rows(tmp_path):
    with Ledger(tmp_path / "l.db") as ledger:
        ledger.enroll([_record("A" * 64)])
        for _ in range(3):
            ledger.mark_download_failed("A" * 64, "boom")
        assert ledger.pending(max_attempts=3) == []
        assert ledger.counts() == {DOWNLOAD_FAILED: 1}


def test_record_scan_persists_outcome_and_stats(tmp_path):
    with Ledger(tmp_path / "l.db") as ledger:
        ledger.enroll([_record("A" * 64), _record("B" * 64)])
        ledger.record_scan(
            ScanOutcome(
                sha256="A" * 64,
                pkg_name="com.openai.chatgpt",
                llm_integrated=True,
                providers=["openai", "anthropic"],
                n_findings=7,
                scan_seconds=1.5,
            )
        )
        ledger.record_scan(
            ScanOutcome(sha256="B" * 64, pkg_name="com.other", providers=["openai"])
        )
        stats = ledger.stats()
        assert stats.scanned == 2
        assert stats.llm_integrated == 1
        assert stats.findings == 7
        assert stats.llm_rate == 0.5
        assert ledger.provider_counts() == {"openai": 2, "anthropic": 1}


def test_failed_scan_is_recorded_with_error(tmp_path):
    with Ledger(tmp_path / "l.db") as ledger:
        ledger.enroll([_record("A" * 64)])
        ledger.record_scan(ScanOutcome(sha256="A" * 64, error="ZipError: bad EOCD"))
        rows = ledger.export_rows()
        assert rows[0]["state"] == "scan_failed"
        assert "bad EOCD" in str(rows[0]["error"])
        assert rows[0]["attempts"] == 1


def test_record_round_trips_through_ledger(tmp_path):
    """store_stratum must survive enroll -> pending, not decay to 'other'."""
    with Ledger(tmp_path / "l.db") as ledger:
        original = _record("A" * 64, pkg="ai.replika.app", year=2024)
        ledger.enroll([original])
        restored = ledger.pending()[0]
        assert restored.sha256 == original.sha256
        assert restored.pkg_name == original.pkg_name
        assert restored.year == original.year
        assert restored.store_stratum == original.store_stratum == "google_play"


def test_state_survives_reopen(tmp_path):
    """The point of the ledger: state outlives the process."""
    path = tmp_path / "l.db"
    with Ledger(path) as ledger:
        ledger.enroll([_record("A" * 64), _record("B" * 64)])
        ledger.record_scan(ScanOutcome(sha256="A" * 64, pkg_name="com.done"))
    with Ledger(path) as reopened:
        assert reopened.counts() == {SCANNED: 1, PENDING: 1}
        assert [r.sha256 for r in reopened.pending()] == ["B" * 64]
