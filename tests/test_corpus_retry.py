"""Tests for download retry and corpus refetch.

A 10,000-APK run makes ten thousand requests over a day or more, so transient
failures are certain. What matters is that they are retried, that permanent
failures are *not* retried (they only burn quota), and that nothing ends up
stranded with no way back into the queue.
"""

from __future__ import annotations

import hashlib
import urllib.error

import pytest

from llm_apk_scanner.corpus.androzoo import AndroZooClient, AndroZooError
from llm_apk_scanner.corpus.ledger import (
    DOWNLOAD_FAILED,
    PENDING,
    SCAN_FAILED,
    SCANNED,
    Ledger,
    ScanOutcome,
)
from llm_apk_scanner.corpus.model import ApkRecord

_BODY = b"apk-bytes"
_SHA = hashlib.sha256(_BODY).hexdigest()


def _http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError("http://x", code, "err", {}, None)  # type: ignore[arg-type]


class _FlakyFetcher:
    """Fails a set number of times, then succeeds."""

    def __init__(self, failures: int, exc: BaseException) -> None:
        self.remaining = failures
        self.exc = exc
        self.calls = 0

    def __call__(self, url: str) -> bytes:
        self.calls += 1
        if self.remaining > 0:
            self.remaining -= 1
            raise self.exc
        return _BODY


def _client(fetcher, **kwargs) -> AndroZooClient:
    slept: list[float] = []
    client = AndroZooClient(
        api_key="k", fetcher=fetcher, sleeper=slept.append, **kwargs
    )
    client.slept = slept  # type: ignore[attr-defined]
    return client


class TestRetry:
    def test_transient_failure_is_retried_then_succeeds(self, tmp_path):
        fetcher = _FlakyFetcher(2, _http_error(429))
        client = _client(fetcher)

        client.download(_SHA, tmp_path / "a.apk")

        assert fetcher.calls == 3
        assert (tmp_path / "a.apk").read_bytes() == _BODY

    def test_backoff_is_exponential(self, tmp_path):
        fetcher = _FlakyFetcher(3, _http_error(503))
        client = _client(fetcher, backoff=1.0)

        client.download(_SHA, tmp_path / "a.apk")

        assert client.slept == [1.0, 2.0, 4.0]

    def test_permanent_failure_is_not_retried(self, tmp_path):
        """A 404 will never succeed; retrying it only wastes the quota."""
        fetcher = _FlakyFetcher(99, _http_error(404))
        client = _client(fetcher)

        with pytest.raises(AndroZooError):
            client.download(_SHA, tmp_path / "a.apk")

        assert fetcher.calls == 1

    def test_bad_api_key_is_not_retried(self, tmp_path):
        fetcher = _FlakyFetcher(99, _http_error(403))
        client = _client(fetcher)

        with pytest.raises(AndroZooError):
            client.download(_SHA, tmp_path / "a.apk")

        assert fetcher.calls == 1

    def test_timeout_is_retried(self, tmp_path):
        fetcher = _FlakyFetcher(1, TimeoutError("timed out"))
        client = _client(fetcher)

        client.download(_SHA, tmp_path / "a.apk")

        assert fetcher.calls == 2

    def test_gives_up_after_max_attempts(self, tmp_path):
        fetcher = _FlakyFetcher(99, _http_error(503))
        client = _client(fetcher, max_attempts=3)

        with pytest.raises(AndroZooError, match="after 3 attempt"):
            client.download(_SHA, tmp_path / "a.apk")

        assert fetcher.calls == 3

    def test_retry_can_be_disabled(self, tmp_path):
        fetcher = _FlakyFetcher(99, _http_error(503))
        client = _client(fetcher, max_attempts=1)

        with pytest.raises(AndroZooError):
            client.download(_SHA, tmp_path / "a.apk")

        assert fetcher.calls == 1


def _record(sha: str) -> ApkRecord:
    return ApkRecord(sha256=sha, pkg_name="com.example.ai", markets="play.google.com",
                     dex_date="2025-01-01 00:00:00")


class TestReset:
    def test_reset_returns_stranded_rows_to_the_queue(self, tmp_path):
        """Rows that exhausted their attempts must be recoverable."""
        with Ledger(tmp_path / "l.db") as ledger:
            ledger.enroll([_record("A" * 64), _record("B" * 64)])
            for _ in range(3):
                ledger.mark_download_failed("A" * 64, "boom")  # exhausted
            ledger.record_scan(ScanOutcome(sha256="B" * 64, error="bad zip"))  # 1 attempt

            # B is retried automatically; only A has run out of attempts.
            assert {r.sha256 for r in ledger.pending()} == {"B" * 64}

            assert ledger.reset() == 2
            assert {r.sha256 for r in ledger.pending()} == {"A" * 64, "B" * 64}

    def test_reset_clears_attempt_count(self, tmp_path):
        """Otherwise a reset row is held back again immediately."""
        with Ledger(tmp_path / "l.db") as ledger:
            ledger.enroll([_record("A" * 64)])
            for _ in range(3):
                ledger.mark_download_failed("A" * 64, "boom")
            ledger.reset()
            rows = ledger.export_rows()
            assert rows[0]["attempts"] == 0
            assert rows[0]["state"] == PENDING
            assert rows[0]["error"] is None

    def test_reset_leaves_scanned_rows_alone_by_default(self, tmp_path):
        with Ledger(tmp_path / "l.db") as ledger:
            ledger.enroll([_record("A" * 64), _record("B" * 64)])
            ledger.record_scan(ScanOutcome(sha256="A" * 64, pkg_name="com.ok"))
            ledger.mark_download_failed("B" * 64, "boom")

            assert ledger.reset() == 1
            assert ledger.counts() == {SCANNED: 1, PENDING: 1}

    def test_rescan_resets_scanned_rows(self, tmp_path):
        """After a scanner rule change, scanned rows must be re-runnable."""
        with Ledger(tmp_path / "l.db") as ledger:
            ledger.enroll([_record("A" * 64), _record("B" * 64)])
            ledger.record_scan(ScanOutcome(sha256="A" * 64, pkg_name="com.ok"))
            ledger.record_scan(ScanOutcome(sha256="B" * 64, pkg_name="com.ok2"))

            assert ledger.reset([SCANNED]) == 2
            assert ledger.counts() == {PENDING: 2}

    def test_reset_of_empty_state_is_a_no_op(self, tmp_path):
        with Ledger(tmp_path / "l.db") as ledger:
            ledger.enroll([_record("A" * 64)])
            assert ledger.reset([DOWNLOAD_FAILED, SCAN_FAILED]) == 0
            assert ledger.counts() == {PENDING: 1}


class TestDownloadDeadline:
    """Regression: a stalled connection must not wedge the pipeline.

    Observed live against AndroZoo — three downloader threads blocked in
    ``poll()`` on ESTABLISHED sockets with an empty receive queue, raising
    nothing for >15 minutes. Every downloader was stuck, so the run
    deadlocked, and retry never engaged because there was no exception.
    A per-socket timeout does not catch this; only a total deadline does.
    """

    def test_trickling_server_hits_the_total_deadline(self):
        """Bytes arriving forever, none fast enough to finish."""
        from llm_apk_scanner.corpus import androzoo

        clock = {"t": 0.0}

        class _Trickle:
            """Yields one byte per read, advancing the clock each time.

            Never idle long enough to trip a per-operation socket timeout —
            exactly the shape that hung the live run.
            """

            def read(self, _n: int) -> bytes:
                clock["t"] += 1.0
                return b"x"

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        monkey_url = "http://example/never-ends"
        real_urlopen = androzoo.urllib.request.urlopen
        real_monotonic = androzoo.time.monotonic
        androzoo.urllib.request.urlopen = lambda *a, **k: _Trickle()  # type: ignore[assignment]
        androzoo.time.monotonic = lambda: clock["t"]  # type: ignore[assignment]
        try:
            with pytest.raises(TimeoutError, match="exceeded"):
                androzoo._urllib_fetch(monkey_url, total_timeout=10.0)
        finally:
            androzoo.urllib.request.urlopen = real_urlopen  # type: ignore[assignment]
            androzoo.time.monotonic = real_monotonic  # type: ignore[assignment]

    def test_deadline_timeout_is_classified_transient(self):
        """So the retry layer engages instead of failing the APK outright."""
        from llm_apk_scanner.corpus.androzoo import _is_transient

        assert _is_transient(TimeoutError("download exceeded 300s"))

    def test_stalled_download_is_retried_then_succeeds(self, tmp_path):
        """End to end: a stall becomes a retry, not a wedged thread."""
        fetcher = _FlakyFetcher(2, TimeoutError("download exceeded 300s"))
        client = _client(fetcher)

        client.download(_SHA, tmp_path / "a.apk")

        assert fetcher.calls == 3
        assert (tmp_path / "a.apk").read_bytes() == _BODY

    def test_normal_download_is_unaffected(self):
        """The deadline must not break ordinary reads."""
        from llm_apk_scanner.corpus import androzoo

        class _Normal:
            def __init__(self):
                self.sent = False

            def read(self, _n):
                if self.sent:
                    return b""
                self.sent = True
                return b"payload"

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        real = androzoo.urllib.request.urlopen
        androzoo.urllib.request.urlopen = lambda *a, **k: _Normal()  # type: ignore[assignment]
        try:
            assert androzoo._urllib_fetch("http://example/ok") == b"payload"
        finally:
            androzoo.urllib.request.urlopen = real  # type: ignore[assignment]
