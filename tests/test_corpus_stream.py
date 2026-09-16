"""Tests for the streaming download-scan-discard pipeline.

The property that matters most here is **bounded disk**: the corpus is
164GB but the volume has far less free, so the pipeline is only safe if
staging is capped no matter how fast downloads outrun scans.

Scan workers are injected (``scan_fn``) so these tests exercise real
orchestration — real threads, a real process pool, a real ledger — without
needing real APK binaries. Worker functions are module-level so they pickle
to the pool.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from llm_apk_scanner.corpus.androzoo import AndroZooClient
from llm_apk_scanner.corpus.ledger import Ledger
from llm_apk_scanner.corpus.model import ApkRecord
from llm_apk_scanner.corpus.stream import run_stream

# Payload every fake APK contains; its SHA-256 is what the fake index serves.
_APK_BODY = b"PK\x03\x04fake-apk-body"


def _fake_result(pkg: str, *, llm: bool = False, findings: int = 0) -> dict:
    return {
        "apk": {"package_name": pkg, "version_name": "1.0"},
        "is_llm_integrated": llm,
        "detected_providers": ["openai"] if llm else [],
        "findings": [{"kind": "secret", "severity": "high"} for _ in range(findings)],
        "scan_duration_seconds": 0.01,
        "summary": {"high": findings},
    }


def ok_worker(apk_path: str) -> dict:
    """Succeeds for every APK; one in three looks LLM-integrated."""
    sha = Path(apk_path).stem.upper()
    llm = sha.startswith(("0", "1", "2", "3", "4"))
    return {
        "sha256": sha,
        "ok": True,
        "elapsed": 0.01,
        "result": _fake_result(f"com.example.{sha[:4].lower()}", llm=llm, findings=2 if llm else 0),
    }


def failing_worker(apk_path: str) -> dict:
    return {
        "sha256": Path(apk_path).stem.upper(),
        "ok": False,
        "elapsed": 0.0,
        "error": "ZipError: EOCD signature not found",
    }


def slow_worker(apk_path: str) -> dict:
    """Deliberately slower than the downloader, to make staging back up."""
    time.sleep(0.05)
    return ok_worker(apk_path)


def raising_worker(apk_path: str) -> dict:
    raise RuntimeError("worker exploded")


class _FakeAndroZoo:
    """Serves a fixed body for any SHA-256, recording what was requested."""

    def __init__(self, bodies: dict[str, bytes]) -> None:
        self.bodies = bodies
        self.requested: list[str] = []

    def __call__(self, url: str) -> bytes:
        sha = url.split("sha256=")[1].split("&")[0]
        self.requested.append(sha)
        if sha not in self.bodies:
            raise OSError(f"HTTP 404 for {sha}")
        return self.bodies[sha]


def _make_corpus(n: int) -> tuple[list[ApkRecord], _FakeAndroZoo]:
    """Build n records whose SHA-256s genuinely match their payloads."""
    import hashlib

    records, bodies = [], {}
    for i in range(n):
        body = _APK_BODY + str(i).encode()
        sha = hashlib.sha256(body).hexdigest().upper()
        bodies[sha] = body
        records.append(
            ApkRecord(
                sha256=sha,
                pkg_name=f"com.example.app{i}",
                markets="play.google.com",
                dex_date="2025-01-01 00:00:00",
                apk_size=len(body),
            )
        )
    return records, _FakeAndroZoo(bodies)


@pytest.fixture
def env(tmp_path: Path):
    """A ledger plus the four paths a run needs."""
    ledger = Ledger(tmp_path / "ledger.db")
    yield {
        "ledger": ledger,
        "staging_dir": tmp_path / "staging",
        "findings_path": tmp_path / "findings.jsonl",
        "detail_dir": tmp_path / "detail",
    }
    ledger.close()


def _run(env, records, fetcher, **kwargs):
    client = AndroZooClient(api_key="test-key", fetcher=fetcher)
    return run_stream(
        records,
        ledger=env["ledger"],
        client=client,
        staging_dir=env["staging_dir"],
        findings_path=env["findings_path"],
        detail_dir=env["detail_dir"],
        delay=0,
        workers=2,
        **kwargs,
    )


def test_apks_are_discarded_after_scanning(env):
    """The core of the streaming design: nothing is left on disk."""
    records, fetcher = _make_corpus(12)
    env["ledger"].enroll(records)

    report = _run(env, records, fetcher, scan_fn=ok_worker)

    assert report.scanned == 12
    leftover = list(env["staging_dir"].glob("*.apk"))
    assert leftover == [], f"{len(leftover)} APKs left on disk"


def test_keep_apks_retains_binaries(env):
    records, fetcher = _make_corpus(5)
    env["ledger"].enroll(records)

    _run(env, records, fetcher, scan_fn=ok_worker, keep_apks=True)

    assert len(list(env["staging_dir"].glob("*.apk"))) == 5


def test_staging_never_exceeds_the_cap(env):
    """Disk safety: slow scans must not let downloads pile up unbounded."""
    records, fetcher = _make_corpus(40)
    env["ledger"].enroll(records)
    peak = 0

    def watch(event, payload):
        nonlocal peak
        peak = max(peak, len(list(env["staging_dir"].glob("*.apk"))))

    report = _run(
        env, records, fetcher, scan_fn=slow_worker, max_staged=4,
        download_workers=4, on_event=watch,
    )

    assert report.scanned == 40
    assert peak <= 4, f"staging peaked at {peak}, cap was 4"
    # Guard against a vacuous pass: if staging were never observed occupied,
    # the cap assertion above would hold for the wrong reason.
    assert peak > 0, "staging never observed in use; the cap was not exercised"


def test_ledger_records_every_outcome(env):
    records, fetcher = _make_corpus(10)
    env["ledger"].enroll(records)

    _run(env, records, fetcher, scan_fn=ok_worker)

    stats = env["ledger"].stats()
    assert stats.scanned == 10
    assert env["ledger"].pending() == []


def test_findings_jsonl_has_one_line_per_apk(env):
    records, fetcher = _make_corpus(8)
    env["ledger"].enroll(records)

    _run(env, records, fetcher, scan_fn=ok_worker)

    lines = env["findings_path"].read_text().strip().splitlines()
    assert len(lines) == 8
    rows = [json.loads(line) for line in lines]
    assert {r["sha256"] for r in rows} == {r.sha256 for r in records}
    assert all(r["store"] == "google_play" and r["year"] == 2025 for r in rows)


def test_detail_written_only_for_interesting_apks(env):
    """Full detail for LLM-integrated apps; nothing for uneventful negatives."""
    records, fetcher = _make_corpus(20)
    env["ledger"].enroll(records)

    _run(env, records, fetcher, scan_fn=ok_worker)

    n_llm = env["ledger"].stats().llm_integrated
    assert 0 < n_llm < 20, "fixture should produce a mix"
    assert len(list(env["detail_dir"].glob("*.json.gz"))) == n_llm


def test_download_failures_are_recorded_and_do_not_stall(env):
    """A 404 must cost one row, not the run."""
    records, fetcher = _make_corpus(10)
    missing = records[3].sha256
    del fetcher.bodies[missing]
    env["ledger"].enroll(records)

    report = _run(env, records, fetcher, scan_fn=ok_worker)

    assert report.scanned == 9
    assert report.download_failed == 1
    assert env["ledger"].counts().get("download_failed") == 1
    assert list(env["staging_dir"].glob("*.apk")) == []


def test_scan_failures_are_recorded_and_apk_still_discarded(env):
    records, fetcher = _make_corpus(6)
    env["ledger"].enroll(records)

    report = _run(env, records, fetcher, scan_fn=failing_worker)

    assert report.scan_failed == 6
    assert report.scanned == 0
    assert list(env["staging_dir"].glob("*.apk")) == []
    assert env["ledger"].counts().get("scan_failed") == 6


def test_corrupt_download_is_rejected_by_checksum(env):
    """A body that does not hash to its SHA-256 must never be scanned."""
    records, fetcher = _make_corpus(4)
    fetcher.bodies[records[0].sha256] = b"tampered-content"
    env["ledger"].enroll(records)

    report = _run(env, records, fetcher, scan_fn=ok_worker)

    assert report.scanned == 3
    assert report.download_failed == 1
    assert list(env["staging_dir"].glob("*.apk")) == []


def test_run_resumes_without_repeating_work(env):
    """The reason the ledger exists: stop and restart mid-corpus."""
    records, fetcher = _make_corpus(10)
    env["ledger"].enroll(records)

    _run(env, records[:6], fetcher, scan_fn=ok_worker)
    assert env["ledger"].stats().scanned == 6

    remaining = env["ledger"].pending()
    assert len(remaining) == 4

    fetcher.requested.clear()
    _run(env, remaining, fetcher, scan_fn=ok_worker)

    assert env["ledger"].stats().scanned == 10
    assert env["ledger"].pending() == []
    # The resumed run re-downloaded only what was outstanding.
    assert len(fetcher.requested) == 4


def test_findings_are_appended_across_runs(env):
    records, fetcher = _make_corpus(6)
    env["ledger"].enroll(records)

    _run(env, records[:3], fetcher, scan_fn=ok_worker)
    _run(env, records[3:], fetcher, scan_fn=ok_worker)

    lines = env["findings_path"].read_text().strip().splitlines()
    assert len(lines) == 6, "second run must append, not overwrite"


@pytest.mark.skipif(os.name == "nt", reason="pool teardown differs on Windows")
def test_worker_that_raises_is_recorded_not_fatal(env):
    """A worker dying outright must cost one APK, not the run.

    ``scan_one`` catches its own errors, but a worker can still be lost to
    an OOM kill or a broken pool. That has to degrade into a scan failure.
    """
    records, fetcher = _make_corpus(4)
    env["ledger"].enroll(records)

    report = _run(env, records, fetcher, scan_fn=raising_worker)

    assert report.scan_failed == 4
    assert env["ledger"].counts().get("scan_failed") == 4
    assert list(env["staging_dir"].glob("*.apk")) == []
    errors = [str(r["error"]) for r in env["ledger"].export_rows()]
    assert all("worker lost" in e for e in errors)
