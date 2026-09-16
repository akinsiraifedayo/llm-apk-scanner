"""Streaming download-scan-discard pipeline.

A 10,000-APK corpus is ~164GB at the measured mean of 16.4MB/APK, which does
not fit the project volume, and ``docs/DATA_HANDLING.md`` forbids staging it
on external drives. So the corpus is never materialised: each APK is
downloaded, scanned, reduced to a compact evidence record, and deleted.

The invariant that makes this safe is a **bounded staging area**. A semaphore
caps how many APKs may exist on disk at once, and a slot is released only
after the APK is deleted, so the downloader can never outrun the scanners and
fill the volume. Peak disk is ``max_staged x APK size`` regardless of corpus
size: at the default of 16 slots that is ~260MB for mean-sized APKs and ~1.1GB
against the largest APK seen in the pilot (72MB). Both are far inside budget,
and neither grows as the corpus does.

Two pools, matched to two different bottlenecks:

* **Downloads** are I/O-bound and remote-rate-limited, so a small thread
  pool with a politeness delay.
* **Scans** are CPU-bound (0.32 s/MB measured), so a process pool that
  actually uses the machine's cores.

Everything is resumable. The ledger is the source of truth, findings are
appended a line at a time, and neither is rewritten wholesale — so an
interrupted run resumes by asking the ledger what is still pending.
"""

from __future__ import annotations

import gzip
import json
import logging
import os
import queue
import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from llm_apk_scanner.corpus.androzoo import AndroZooClient
from llm_apk_scanner.corpus.ledger import Ledger, ScanOutcome
from llm_apk_scanner.corpus.model import ApkRecord

#: Default cap on APKs resident on disk simultaneously.
DEFAULT_MAX_STAGED = 16


def _silence_third_party_logging() -> None:
    """Mute androguard's per-DEX-item debug logging.

    Androguard logs several thousand DEBUG lines *per APK*. Across 10,000
    APKs that is tens of millions of lines: it dominates runtime, floods any
    log file, and makes real progress output unreadable.
    """
    try:  # androguard >=4 routes through loguru
        from loguru import logger

        logger.remove()
    except Exception:  # pragma: no cover - loguru absent or already removed
        pass
    for name in ("androguard", "androguard.core", "androguard.core.api_specific_resources"):
        logging.getLogger(name).setLevel(logging.ERROR)


def scan_one(apk_path: str) -> dict[str, object]:
    """Scan a single APK in a worker process; never raises.

    Returns a plain dict so the payload pickles cleanly back to the parent.
    A failure here must not kill the pool — one malformed APK in ten thousand
    should cost one row, not the run.
    """
    _silence_third_party_logging()
    sha256 = Path(apk_path).stem.upper()
    started = time.monotonic()
    try:
        from llm_apk_scanner.scan import scan_apk

        result = scan_apk(apk_path)
        return {
            "sha256": sha256,
            "ok": True,
            "elapsed": time.monotonic() - started,
            "result": result.to_dict(),
        }
    except Exception as exc:  # noqa: BLE001 - deliberate catch-all at the boundary
        return {
            "sha256": sha256,
            "ok": False,
            "elapsed": time.monotonic() - started,
            "error": f"{type(exc).__name__}: {exc}",
        }


@dataclass
class StreamReport:
    """Totals for one streaming run."""

    scanned: int = 0
    llm_integrated: int = 0
    download_failed: int = 0
    scan_failed: int = 0
    bytes_downloaded: int = 0
    findings: int = 0
    #: APKs kept on disk under the selective-retention policy.
    retained: int = 0
    bytes_retained: int = 0
    #: APKs that qualified for retention but were discarded because the
    #: retention budget was exhausted. Non-zero means the retention set is
    #: incomplete, which matters when it is later used as a re-scan corpus.
    retention_skipped: int = 0
    #: Zero-finding APKs kept as a false-negative audit sample.
    negatives_retained: int = 0
    started_at: float = field(default_factory=time.monotonic)

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.started_at


class _FindingsWriter:
    """Append-only JSONL sink, plus gzipped detail for the interesting apps.

    One line per APK keeps the summary stream greppable and appendable; a
    crash costs at most the line in flight rather than the whole file. Full
    finding detail is written only for APKs that are LLM-integrated or that
    produced findings, since carrying complete detail for every negative
    would dominate the output for no analytic gain.
    """

    def __init__(self, jsonl_path: Path, detail_dir: Path) -> None:
        self.jsonl_path = Path(jsonl_path)
        self.detail_dir = Path(detail_dir)
        self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        self.detail_dir.mkdir(parents=True, exist_ok=True)
        self._handle = self.jsonl_path.open("a", encoding="utf-8")
        self._lock = threading.Lock()

    def write(self, sha256: str, record: ApkRecord, result: dict[str, object]) -> int:
        # ``result`` crosses a process boundary as a plain dict, so its values
        # are untyped on arrival; narrow them rather than trusting the shape.
        raw_apk = result.get("apk")
        apk: dict[str, object] = raw_apk if isinstance(raw_apk, dict) else {}
        raw_findings = result.get("findings")
        findings: list[object] = raw_findings if isinstance(raw_findings, list) else []
        summary = result.get("summary") or {}
        row = {
            "sha256": sha256,
            "pkg_name": apk.get("package_name") or record.pkg_name,
            "version": apk.get("version_name"),
            "year": record.year,
            "store": record.store_stratum,
            "apk_size": record.apk_size,
            "llm_integrated": result.get("is_llm_integrated", False),
            "providers": result.get("detected_providers", []),
            "n_findings": len(findings),
            "by_severity": summary,
            "scan_seconds": result.get("scan_duration_seconds"),
        }
        with self._lock:
            self._handle.write(json.dumps(row) + "\n")
            self._handle.flush()

        if result.get("is_llm_integrated") or findings:
            detail = self.detail_dir / f"{sha256}.json.gz"
            with gzip.open(detail, "wt", encoding="utf-8") as fh:
                json.dump(result, fh)
        return len(findings)

    def close(self) -> None:
        with self._lock:
            self._handle.close()


def run_stream(
    records: Sequence[ApkRecord],
    *,
    ledger: Ledger,
    client: AndroZooClient,
    staging_dir: Path,
    findings_path: Path,
    detail_dir: Path,
    workers: int | None = None,
    download_workers: int = 3,
    max_staged: int = DEFAULT_MAX_STAGED,
    delay: float = 0.5,
    keep_apks: bool = False,
    keep_with_findings: bool = False,
    keep_negatives: int = 0,
    keep_budget_bytes: int = 0,
    scan_fn: Callable[[str], dict[str, object]] = scan_one,
    on_event: Callable[[str, dict[str, object]], None] | None = None,
) -> StreamReport:
    """Download, scan and discard each record, updating the ledger as it goes.

    Args:
        records: APKs still needing work (typically ``ledger.pending()``).
        ledger: state store; the parent process is its only writer.
        client: AndroZoo client (verifies SHA-256 on download).
        staging_dir: transient APK area, emptied as the run proceeds.
        findings_path: append-only JSONL summary, one line per APK.
        detail_dir: gzipped full findings for interesting APKs.
        workers: scan processes (default: CPU count, capped at 8).
        download_workers: concurrent downloads; keep low to stay polite.
        max_staged: hard cap on APKs on disk at once — the disk-safety knob.
        delay: seconds between download starts, per downloader thread.
        keep_apks: retain *every* APK instead of deleting after scan. Only
            viable for small runs; the default streams and discards.
        keep_with_findings: retain only APKs that produced a finding or were
            confirmed LLM-integrated, discarding the rest. This is the
            research-relevant subset: re-scanning after a detector fix only
            needs the APKs a detector could fire on, and manual precision
            review only ever inspects APKs with findings. Measured on the
            2024-2026 frame, retaining everything would need ~52.6GB at 1,500
            APKs (mean 35.9MB); the finding-carrying subset is a fraction of
            that. Ignored when ``keep_apks`` is set.
        keep_negatives: also retain up to this many APKs that produced *no*
            findings. Retaining only positives makes false negatives
            unauditable, which is the wrong way round: a missed detection is
            invisible by definition, so the only way to find one is to read an
            APK the scanner called empty. Defect 14 (React Native bundles
            never read) was found in an app that happened to carry an
            unrelated Google key; had it scored zero it would have been
            discarded and the blind spot would have survived the study.
            The manifest is ordered by SHA-256, which is uniform, so taking
            the first N encountered is an unbiased sample of negatives.
        keep_budget_bytes: cap on bytes retained by ``keep_with_findings``.
            Once exceeded, later APKs are discarded even if they carry
            findings, and the run continues. A long run that fills the volume
            loses everything, so a partial retention set is strictly better
            than a failed run. ``0`` means no cap.
        scan_fn: the worker callable, injected so orchestration can be
            tested without real APKs. Must be a module-level function so it
            pickles to the pool.
        on_event: ``(event, payload)`` progress callback.

    Returns:
        A ``StreamReport`` of totals for this run.
    """
    staging_dir = Path(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)
    workers = workers or min(8, os.cpu_count() or 4)
    report = StreamReport()
    writer = _FindingsWriter(findings_path, detail_dir)

    def emit(event: str, **payload: object) -> None:
        if on_event is not None:
            on_event(event, payload)

    # Caps APKs on disk. Acquired before a download starts, released only
    # once the file is gone (or deliberately kept), so disk use is bounded
    # by construction rather than by hoping scans keep up.
    slots = threading.Semaphore(max_staged)

    retained_lock = threading.Lock()

    # Seed the budget from what previous runs already retained.
    #
    # Without this the cap is per-run rather than cumulative, and this
    # pipeline is explicitly built to be interrupted and resumed: a 23-hour
    # run stopped and restarted three times would grant itself three full
    # budgets and quietly overrun the volume — the exact failure the cap
    # exists to prevent.
    if keep_with_findings and keep_budget_bytes:
        for existing in staging_dir.glob("*.apk"):
            try:
                report.bytes_retained += existing.stat().st_size
                report.retained += 1
            except OSError:
                continue

    def _reserve_negative(apk_path: Path) -> bool:
        """Whether to retain a zero-finding APK as a false-negative sample.

        Charged against the same byte budget as positives, but capped
        separately by count so a long run of empty scans cannot crowd out the
        findings-bearing set.
        """
        with retained_lock:
            if report.negatives_retained >= keep_negatives:
                return False
        if not _reserve_retention(apk_path):
            return False
        with retained_lock:
            report.negatives_retained += 1
        return True

    def _reserve_retention(apk_path: Path) -> bool:
        """Whether this APK may be retained, charging it to the budget.

        Returns ``False`` once the budget is exhausted, so retention degrades
        to discarding rather than to filling the volume.
        """
        try:
            size = apk_path.stat().st_size
        except OSError:
            return False
        with retained_lock:
            if keep_budget_bytes and report.bytes_retained + size > keep_budget_bytes:
                report.retention_skipped += 1
                return False
            report.bytes_retained += size
            report.retained += 1
            return True
    # Downloaded-but-not-yet-submitted APKs. Bounded so a fast network cannot
    # queue work faster than the semaphore permits anyway.
    ready: queue.Queue[tuple[ApkRecord, Path] | None] = queue.Queue(maxsize=max_staged)
    pending = list(records)
    source = iter(pending)
    source_lock = threading.Lock()
    stop = threading.Event()

    # Every record must reach exactly one terminal outcome: scanned, scan
    # failed, or download failed. Counting them explicitly is what lets the
    # main loop know it is finished, rather than inferring it from thread
    # liveness (which races when a downloader dies between checks).
    done_lock = threading.Lock()
    done = 0
    total = len(pending)

    def account() -> None:
        nonlocal done
        with done_lock:
            done += 1

    def next_record() -> ApkRecord | None:
        with source_lock:
            return next(source, None)

    def download_one(record: ApkRecord) -> None:
        dest = staging_dir / f"{record.sha256.upper()}.apk"
        try:
            client.download(record.sha256, dest)
        except Exception as exc:  # noqa: BLE001 - see download_loop
            ledger.mark_download_failed(record.sha256, str(exc))
            report.download_failed += 1
            emit("download_failed", sha256=record.sha256, error=str(exc))
            dest.unlink(missing_ok=True)
            account()
            slots.release()
        else:
            report.bytes_downloaded += dest.stat().st_size
            ledger.mark_downloaded(record.sha256)
            emit("downloaded", sha256=record.sha256, pkg=record.pkg_name)
            ready.put((record, dest))

    def download_loop() -> None:
        # Every exit path from this loop must either hand the record on or
        # account for it. A downloader that dies with a record unaccounted
        # for leaves the main loop waiting on a count that can never be
        # reached — so the catch here is deliberately total, not just
        # AndroZooError/OSError. An unexpected exception must fail one APK,
        # never wedge a multi-day run.
        while not stop.is_set():
            record = next_record()
            if record is None:
                return
            slots.acquire()
            if stop.is_set():
                slots.release()
                return
            try:
                download_one(record)
            except Exception as exc:  # noqa: BLE001
                report.download_failed += 1
                emit("download_failed", sha256=record.sha256, error=repr(exc))
                account()
                slots.release()
            if delay:
                time.sleep(delay)

    threads = [
        threading.Thread(target=download_loop, name=f"dl-{i}", daemon=True)
        for i in range(max(1, download_workers))
    ]
    for thread in threads:
        thread.start()

    def finish(record: ApkRecord, apk_path: Path, payload: dict[str, object]) -> None:
        """Persist one scan result and free its staging slot."""
        sha = str(payload["sha256"])
        # Default to discarding. A scan that fails, or crashes the worker,
        # yields no evidence that this APK is worth keeping — so the failure
        # path must not accidentally retain, which is how a "keep the
        # interesting ones" policy quietly becomes "keep everything".
        keep_this = keep_apks
        try:
            if payload.get("ok"):
                result = payload["result"]
                assert isinstance(result, dict)
                n_findings = writer.write(sha, record, result)
                raw_apk = result.get("apk")
                apk_info: dict[str, object] = raw_apk if isinstance(raw_apk, dict) else {}
                raw_providers = result.get("detected_providers")
                elapsed = payload.get("elapsed")
                outcome = ScanOutcome(
                    sha256=sha,
                    pkg_name=str(apk_info.get("package_name") or record.pkg_name),
                    llm_integrated=bool(result.get("is_llm_integrated")),
                    providers=[str(p) for p in raw_providers or []]
                    if isinstance(raw_providers, list)
                    else [],
                    n_findings=n_findings,
                    scan_seconds=float(elapsed) if isinstance(elapsed, (int, float)) else 0.0,
                )
                ledger.record_scan(outcome)
                report.scanned += 1
                report.findings += n_findings
                if outcome.llm_integrated:
                    report.llm_integrated += 1
                # LLM-integrated apps are kept even with zero findings: they
                # are the RQ1 numerator's population, and an integrated app
                # that exposes nothing is itself a result worth re-examining.
                if keep_with_findings and (n_findings > 0 or outcome.llm_integrated):
                    keep_this = _reserve_retention(apk_path)
                elif keep_negatives and n_findings == 0:
                    keep_this = _reserve_negative(apk_path)
                emit("scanned", sha256=sha, pkg=outcome.pkg_name,
                     llm=outcome.llm_integrated, findings=n_findings)
            else:
                error = str(payload.get("error"))
                ledger.record_scan(ScanOutcome(sha256=sha, error=error))
                report.scan_failed += 1
                emit("scan_failed", sha256=sha, error=error)
        finally:
            if not keep_this:
                apk_path.unlink(missing_ok=True)
            account()
            slots.release()

    inflight: dict[Future[dict[str, object]], tuple[ApkRecord, Path]] = {}

    def payload_of(future: Future[dict[str, object]], record: ApkRecord) -> dict[str, object]:
        """A worker's result, or a synthetic failure if it died outright.

        ``scan_one`` catches its own exceptions, but the worker can still be
        lost to something it cannot catch — an OOM kill, or a
        ``BrokenProcessPool``. Converting that into a scan failure keeps the
        APK accounted for and its staging slot recoverable.
        """
        try:
            return future.result()
        except Exception as exc:  # noqa: BLE001
            return {
                "sha256": record.sha256.upper(),
                "ok": False,
                "elapsed": 0.0,
                "error": f"worker lost: {type(exc).__name__}: {exc}",
            }

    try:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            while done < total:
                # Reap finished scans first so staging slots free up promptly.
                for future in [f for f in inflight if f.done()]:
                    record, apk_path = inflight.pop(future)
                    finish(record, apk_path, payload_of(future, record))

                # Keep the pool fed but not swamped; beyond ~2x workers the
                # extra futures only pin APKs on disk.
                if len(inflight) >= workers * 2:
                    time.sleep(0.05)
                    continue

                try:
                    item = ready.get(timeout=0.2)
                except queue.Empty:
                    continue
                if item is None:
                    continue
                record, apk_path = item
                future = pool.submit(scan_fn, str(apk_path))
                inflight[future] = (record, apk_path)

            for future, (record, apk_path) in list(inflight.items()):
                finish(record, apk_path, payload_of(future, record))
                inflight.pop(future, None)
    finally:
        stop.set()
        # Unblock any downloader parked on the semaphore so threads exit.
        for _ in threads:
            slots.release()
        for thread in threads:
            thread.join(timeout=5)
        writer.close()
        # Drain anything downloaded but never scanned, so an interrupted run
        # leaves no orphaned APKs occupying the volume. This covers both the
        # queue and any scan still in flight when the run was torn down.
        #
        # Note this checks ``keep_apks`` only, not ``keep_with_findings``: an
        # APK that was never scanned has no findings to justify keeping it,
        # and treating "unknown" as "interesting" would retain exactly the
        # APKs we know least about.
        if not keep_apks:
            for _, staged in inflight.values():
                staged.unlink(missing_ok=True)
            while True:
                try:
                    leftover = ready.get_nowait()
                except queue.Empty:
                    break
                if leftover is not None:
                    leftover[1].unlink(missing_ok=True)

    return report
