"""Durable per-APK state for a corpus run.

A 10,000-APK run spans days and will be interrupted — by a dropped
connection, an AndroZoo rate limit, or a laptop lid. The ledger is the
single source of truth for *what has happened to each SHA-256*, so any
run can be stopped and resumed without re-downloading or re-scanning work
that is already done.

It replaces the previous ``ai_download_progress.json``, which held every
completed hash in one list and rewrote the whole file on each checkpoint:
that is O(n) per update and O(n^2) over a run, and a crash mid-write
truncates the only copy. SQLite gives O(1) updates, atomic commits, and a
crash-safe WAL.

Concurrency model: **one process, many threads.** Only the orchestrating
process touches the ledger — scan workers are pure functions that return
results to the parent — but within that process the downloader threads write
too, so every method serialises on an instance lock. Keeping the lock inside
the ledger rather than at the call site means a caller cannot forget it; the
failure mode when they do is a thread dying silently mid-run.

State machine::

    PENDING ──download──> DOWNLOADED ──scan──> SCANNED
       │                      │
       └──> DOWNLOAD_FAILED   └──> SCAN_FAILED
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Iterable, Sequence
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from llm_apk_scanner.corpus.model import ApkRecord

#: Terminal and intermediate states an APK can occupy.
PENDING = "pending"
DOWNLOADED = "downloaded"
SCANNED = "scanned"
DOWNLOAD_FAILED = "download_failed"
SCAN_FAILED = "scan_failed"

#: States that mean "no further work needed on this APK".
TERMINAL = (SCANNED,)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS apk (
    sha256          TEXT PRIMARY KEY,
    pkg_name        TEXT NOT NULL DEFAULT '',
    year            INTEGER,
    -- ``markets`` is AndroZoo's raw field, kept verbatim so an ApkRecord
    -- round-trips exactly; ``store`` is the stratum derived from it, stored
    -- separately only so it can be indexed and grouped by.
    markets         TEXT NOT NULL DEFAULT '',
    store           TEXT NOT NULL DEFAULT '',
    apk_size        INTEGER,
    state           TEXT NOT NULL DEFAULT 'pending',
    attempts        INTEGER NOT NULL DEFAULT 0,
    error           TEXT,
    llm_integrated  INTEGER,
    providers       TEXT,
    n_findings      INTEGER,
    scan_seconds    REAL,
    scanned_at      TEXT,
    updated_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_apk_state ON apk(state);
CREATE INDEX IF NOT EXISTS idx_apk_llm ON apk(llm_integrated);

CREATE TABLE IF NOT EXISTS run (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at  TEXT NOT NULL,
    finished_at TEXT,
    note        TEXT
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass(frozen=True)
class CorpusStats:
    """Headline progress numbers for a corpus run.

    A dataclass rather than a dict so callers get real types: the CLI reads
    these straight into a status table and should not have to cast every
    field back from ``object``.
    """

    total: int
    by_state: dict[str, int]
    scanned: int
    llm_integrated: int
    findings: int
    scan_seconds: float

    @property
    def llm_rate(self) -> float:
        """Share of scanned APKs that integrate an LLM, 0.0 if none scanned."""
        return self.llm_integrated / self.scanned if self.scanned else 0.0


@dataclass(frozen=True)
class ScanOutcome:
    """What a worker learned about one APK. Returned by value to the parent."""

    sha256: str
    pkg_name: str = ""
    llm_integrated: bool = False
    providers: Sequence[str] = ()
    n_findings: int = 0
    scan_seconds: float = 0.0
    error: str | None = None


class Ledger:
    """Crash-safe per-APK state for a corpus run.

    Usage::

        with Ledger(Path("corpus/ledger.db")) as ledger:
            ledger.enroll(records)
            for rec in ledger.pending(limit=50):
                ...
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # ``check_same_thread=False`` because the streaming pipeline writes
        # from its downloader threads as well as the main thread. SQLite
        # objects are otherwise pinned to their creating thread, and a
        # violation raises inside the worker thread — killing it silently and
        # hanging the run. Access is serialised by ``self._lock`` instead, so
        # the ledger is thread-safe for every caller rather than relying on
        # each one to bring its own lock.
        self._conn = sqlite3.connect(self.path, timeout=30.0, check_same_thread=False)
        self._lock = threading.RLock()
        self._conn.row_factory = sqlite3.Row
        # WAL survives a crash mid-write and lets readers work during writes.
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def __enter__(self) -> Ledger:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ---------------------------------------------------------------- enroll

    def enroll(self, records: Iterable[ApkRecord]) -> int:
        """Register manifest records as ``pending``. Idempotent.

        Re-enrolling an existing SHA-256 leaves its state untouched, so
        re-running against an extended manifest only adds the new rows.
        Returns the number of rows actually inserted.
        """
        rows = [
            (
                r.sha256.upper(),
                r.pkg_name,
                r.year,
                r.markets,
                r.store_stratum,
                r.apk_size,
                PENDING,
                _now(),
            )
            for r in records
        ]
        with self._lock, self._conn:
            before = self._count_all()
            self._conn.executemany(
                "INSERT OR IGNORE INTO apk "
                "(sha256, pkg_name, year, markets, store, apk_size, state, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                rows,
            )
            return self._count_all() - before

    def _count_all(self) -> int:
        with self._lock:
            cur = self._conn.execute("SELECT COUNT(*) FROM apk")
            return int(cur.fetchone()[0])

    # ----------------------------------------------------------------- query

    def pending(self, limit: int | None = None, *, max_attempts: int = 3) -> list[ApkRecord]:
        """Records still needing work, oldest-enrolled first.

        Anything not yet ``scanned`` is eligible, so a run interrupted after
        download but before scan resumes correctly. Rows that have failed
        ``max_attempts`` times are held back so one poison APK cannot spin
        the pipeline forever.
        """
        sql = (
            "SELECT sha256, pkg_name, year, markets, apk_size FROM apk "
            "WHERE state != ? AND attempts < ? ORDER BY sha256"
        )
        params: list[object] = [SCANNED, max_attempts]
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        with self._lock:
            cur = self._conn.execute(sql, params)
            return [_record_from_row(row) for row in cur.fetchall()]

    def counts(self) -> dict[str, int]:
        """Histogram of states, e.g. ``{"pending": 9875, "scanned": 125}``."""
        with self._lock:
            cur = self._conn.execute("SELECT state, COUNT(*) FROM apk GROUP BY state")
            return {str(state): int(n) for state, n in cur.fetchall()}

    def stats(self) -> CorpusStats:
        """Headline numbers for progress reporting and the corpus chapter."""
        counts = self.counts()
        with self._lock:
            cur = self._conn.execute(
                "SELECT COUNT(*), SUM(llm_integrated), SUM(n_findings), SUM(scan_seconds) "
                "FROM apk WHERE state = ?",
                (SCANNED,),
            )
            n_scanned, n_llm, n_findings, seconds = cur.fetchone()
        return CorpusStats(
            total=self._count_all(),
            by_state=counts,
            scanned=int(n_scanned or 0),
            llm_integrated=int(n_llm or 0),
            findings=int(n_findings or 0),
            scan_seconds=float(seconds or 0.0),
        )

    def provider_counts(self) -> dict[str, int]:
        """How many scanned APKs referenced each inference provider."""
        with self._lock:
            cur = self._conn.execute(
                "SELECT providers FROM apk WHERE state = ? AND providers IS NOT NULL",
                (SCANNED,),
            )
            rows = cur.fetchall()
        tally: dict[str, int] = {}
        for (raw,) in rows:
            for provider in json.loads(raw):
                tally[provider] = tally.get(provider, 0) + 1
        return dict(sorted(tally.items(), key=lambda kv: -kv[1]))

    # ---------------------------------------------------------------- mutate

    def mark_downloaded(self, sha256: str) -> None:
        self._set_state(sha256, DOWNLOADED)

    def mark_download_failed(self, sha256: str, error: str) -> None:
        self._set_state(sha256, DOWNLOAD_FAILED, error=error, bump_attempts=True)

    def record_scan(self, outcome: ScanOutcome) -> None:
        """Persist a worker's result, moving the row to its terminal state."""
        if outcome.error:
            self._set_state(
                outcome.sha256, SCAN_FAILED, error=outcome.error, bump_attempts=True
            )
            return
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE apk SET state = ?, pkg_name = ?, llm_integrated = ?, "
                "providers = ?, n_findings = ?, scan_seconds = ?, "
                "scanned_at = ?, error = NULL, updated_at = ? "
                "WHERE sha256 = ?",
                (
                    SCANNED,
                    outcome.pkg_name,
                    int(outcome.llm_integrated),
                    json.dumps(list(outcome.providers)),
                    outcome.n_findings,
                    outcome.scan_seconds,
                    _now(),
                    _now(),
                    outcome.sha256.upper(),
                ),
            )

    def _set_state(
        self,
        sha256: str,
        state: str,
        *,
        error: str | None = None,
        bump_attempts: bool = False,
    ) -> None:
        attempts = "attempts + 1" if bump_attempts else "attempts"
        with self._lock, self._conn:
            self._conn.execute(
                f"UPDATE apk SET state = ?, error = ?, attempts = {attempts}, "  # noqa: S608
                "updated_at = ? WHERE sha256 = ?",
                (state, error, _now(), sha256.upper()),
            )

    def reset(self, states: Sequence[str] | None = None) -> int:
        """Return rows to ``pending`` so they are picked up again.

        Two things make this necessary. Rows that exhausted their attempts are
        held out of ``pending`` forever, so a transient outage that burned
        three attempts would silently shrink the corpus with no way back. And
        a scanner rule change means already-``scanned`` rows need re-running —
        under the streaming design that implies re-downloading, since the APK
        is long gone.

        Args:
            states: which states to reset. Defaults to the two failure states;
                pass ``[SCANNED]`` to force a re-scan, or ``None``-equivalent
                explicit lists for anything else.

        Returns:
            Number of rows reset.
        """
        targets = list(states) if states is not None else [DOWNLOAD_FAILED, SCAN_FAILED]
        if not targets:
            return 0
        placeholders = ",".join("?" for _ in targets)
        with self._lock, self._conn:
            cur = self._conn.execute(
                f"UPDATE apk SET state = ?, attempts = 0, error = NULL, updated_at = ? "  # noqa: S608
                f"WHERE state IN ({placeholders})",
                [PENDING, _now(), *targets],
            )
            return int(cur.rowcount or 0)

    # ------------------------------------------------------------------ runs

    def start_run(self, note: str = "") -> int:
        with self._lock, self._conn:
            cur = self._conn.execute(
                "INSERT INTO run (started_at, note) VALUES (?, ?)", (_now(), note)
            )
            return int(cur.lastrowid or 0)

    def finish_run(self, run_id: int) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE run SET finished_at = ? WHERE id = ?", (_now(), run_id)
            )

    # ---------------------------------------------------------------- export

    def export_rows(self) -> list[dict[str, object]]:
        """Every row as a plain dict, for CSV/JSON export and analysis."""
        with self._lock, closing(
            self._conn.execute("SELECT * FROM apk ORDER BY sha256")
        ) as cur:
            return [dict(row) for row in cur.fetchall()]


def _record_from_row(row: sqlite3.Row) -> ApkRecord:
    """Rebuild an ``ApkRecord`` from a ledger row.

    ``markets`` is restored verbatim so ``store_stratum`` re-derives to the
    same value it was enrolled with. ``dex_date`` is synthesised from the
    stored year so ``ApkRecord.year`` round-trips, mirroring
    ``manifest.read_manifest``.
    """
    year = row["year"]
    return ApkRecord(
        sha256=row["sha256"],
        pkg_name=row["pkg_name"] or "",
        markets=row["markets"] or "",
        dex_date=f"{year}-01-01 00:00:00" if year else None,
        apk_size=row["apk_size"],
    )
