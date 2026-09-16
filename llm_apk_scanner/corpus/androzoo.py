"""AndroZoo API client: metadata parsing and APK download.

The client talks to the AndroZoo REST API (https://androzoo.uni.lu/api_doc).
The API key is supplied explicitly or read from ``ANDROZOO_API_KEY``; it is
never hard-coded and is excluded from ``repr`` so it does not leak into logs
or tracebacks (see ``docs/DATA_HANDLING.md`` §1).

HTTP access is injected via the ``fetcher`` callable so the client is fully
testable without network access. The default fetcher uses the standard
library only — the scanner deliberately carries no HTTP dependency.
"""

from __future__ import annotations

import csv
import hashlib
import os
import time
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

from llm_apk_scanner.corpus.model import ApkRecord

#: A fetcher maps a URL to the raw bytes at that URL.
Fetcher = Callable[[str], bytes]

DEFAULT_BASE_URL = "https://androzoo.uni.lu/api"
ENV_API_KEY = "ANDROZOO_API_KEY"

# Columns of the AndroZoo ``latest.csv`` metadata index that we consume.
_COL_SHA256 = "sha256"
_COL_PKG = "pkg_name"
_COL_VERCODE = "vercode"
_COL_MARKETS = "markets"
_COL_DEX_DATE = "dex_date"
_COL_VT_SCAN_DATE = "vt_scan_date"
#: Not present in the public ``latest.csv.gz`` index — its columns are
#: sha256, sha1, md5, dex_date, apk_size, pkg_name, vercode, vt_detection,
#: vt_scan_date, dex_size, markets. Parsed anyway so that private or extended
#: AndroZoo exports carrying it are handled, but ``vt_scan_date`` is the
#: fallback that actually fires. See ``ApkRecord.year``.
_COL_ADDED = "added"
_COL_APK_SIZE = "apk_size"


class AndroZooError(RuntimeError):
    """Raised for AndroZoo configuration, download, or verification errors."""


#: Seconds a single socket operation may block before erroring.
DEFAULT_SOCKET_TIMEOUT = 60.0
#: Seconds one whole download may take, however slowly it trickles.
#:
#: Sized against observed behaviour rather than generosity: AndroZoo stalls
#: connections intermittently, and every stall costs this much wall-clock
#: before the retry can start. 180s still allows the largest APK seen in the
#: pilot (72MB) to arrive at 0.4MB/s, while recovering from a stall in a third
#: of the time a 300s budget took.
DEFAULT_TOTAL_TIMEOUT = 180.0
#: Read granularity; also how often the deadline is checked.
_READ_CHUNK = 1 << 16


def _urllib_fetch(
    url: str,
    *,
    timeout: float = DEFAULT_SOCKET_TIMEOUT,
    total_timeout: float = DEFAULT_TOTAL_TIMEOUT,
) -> bytes:
    """Default network fetcher, standard library only.

    Enforces **two** limits, because a socket timeout alone is not enough.
    ``urlopen(timeout=...)`` bounds each individual socket operation, so a
    server that dribbles a few bytes just often enough resets the clock
    forever and the read never returns. Observed live against AndroZoo: three
    downloader threads sat in ``poll()`` on ESTABLISHED connections with an
    empty receive queue for over fifteen minutes, raising nothing. With every
    downloader wedged, the run deadlocked — and because no exception was
    raised, retry logic never engaged.

    Reading in chunks against a wall-clock deadline turns that silent hang
    into a ``TimeoutError``, which the retry layer treats as transient and
    retries against (most likely) a healthier connection.
    """
    deadline = time.monotonic() + total_timeout
    chunks: list[bytes] = []
    with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310
        while True:
            if time.monotonic() > deadline:
                got = sum(len(c) for c in chunks)
                raise TimeoutError(
                    f"download exceeded {total_timeout:.0f}s (got {got} bytes)"
                )
            chunk = response.read(_READ_CHUNK)
            if not chunk:
                break
            chunks.append(chunk)
    return b"".join(chunks)


def parse_metadata(lines: Iterable[str]) -> Iterator[ApkRecord]:
    """Yield ``ApkRecord`` objects from AndroZoo metadata CSV lines.

    Accepts any iterable of lines (a file handle, ``str.splitlines()``,
    or a streaming download). Rows missing a SHA-256 are skipped.

    Also handles frame.csv format (has "year" column instead of "dex_date").
    """
    reader = csv.DictReader(line for line in lines if line.strip())
    for row in reader:
        sha256 = (row.get(_COL_SHA256) or "").strip()
        if not sha256:
            continue
        # Handle both AndroZoo format (dex_date) and frame.csv format (year)
        dex_date = (row.get(_COL_DEX_DATE) or "").strip() or None
        if not dex_date and row.get("year"):
            # Convert year column to fake dex_date for ApkRecord.year property
            year_val = row.get("year", "").strip()
            if year_val.isdigit():
                dex_date = f"{year_val}-01-01 00:00:00"
        yield ApkRecord(
            sha256=sha256,
            pkg_name=(row.get(_COL_PKG) or "").strip(),
            markets=(row.get(_COL_MARKETS) or "").strip(),
            dex_date=dex_date,
            vt_scan_date=(row.get(_COL_VT_SCAN_DATE) or "").strip() or None,
            added=(row.get(_COL_ADDED) or "").strip() or None,
            vercode=(row.get(_COL_VERCODE) or "").strip() or None,
            apk_size=_maybe_int(row.get(_COL_APK_SIZE)),
        )


def _maybe_int(value: str | None) -> int | None:
    if value is None:
        return None
    value = value.strip()
    if not value.isdigit():
        return None
    return int(value)


#: HTTP statuses worth retrying: rate limiting and server-side faults. A 404
#: (not in AndroZoo) or 401/403 (bad key) will never succeed on retry.
TRANSIENT_STATUSES = frozenset({408, 429, 500, 502, 503, 504})


def _is_transient(exc: BaseException) -> bool:
    """Whether a failed fetch is worth retrying.

    Over a run of thousands of downloads spanning days, rate limits, timeouts
    and dropped connections are routine and self-correcting; a missing SHA-256
    or a rejected API key is not. Retrying the latter just burns the quota.
    """
    status = getattr(exc, "code", None)
    if status is not None:
        return int(status) in TRANSIENT_STATUSES
    # No HTTP status: a socket timeout, reset connection, or DNS blip.
    return isinstance(exc, (TimeoutError, ConnectionError, OSError))


def make_fetcher(
    *,
    timeout: float = DEFAULT_SOCKET_TIMEOUT,
    total_timeout: float = DEFAULT_TOTAL_TIMEOUT,
) -> Fetcher:
    """Build a fetcher with custom timeouts.

    The public way to tune download deadlines without reaching for the
    module-private default.
    """

    def fetcher(url: str) -> bytes:
        return _urllib_fetch(url, timeout=timeout, total_timeout=total_timeout)

    return fetcher


@dataclass
class AndroZooClient:
    """Minimal AndroZoo API client for metadata and APK download."""

    api_key: str = field(repr=False)
    base_url: str = DEFAULT_BASE_URL
    fetcher: Fetcher = field(default=_urllib_fetch, repr=False)
    #: Attempts per download, including the first. 1 disables retrying.
    max_attempts: int = 4
    #: Base seconds for exponential backoff (1s, 2s, 4s, ...).
    backoff: float = 1.0
    #: Sleep hook, injected so tests do not actually wait.
    sleeper: Callable[[float], None] = field(default=time.sleep, repr=False)

    @classmethod
    def from_env(cls, **kwargs: object) -> AndroZooClient:
        """Construct a client from the ``ANDROZOO_API_KEY`` environment variable."""
        api_key = os.environ.get(ENV_API_KEY)
        if not api_key:
            raise AndroZooError(
                f"{ENV_API_KEY} is not set. Export your AndroZoo API key "
                "(store it in a password manager, never in the repository)."
            )
        return cls(api_key=api_key, **kwargs)  # type: ignore[arg-type]

    def download_url(self, sha256: str) -> str:
        """Build the AndroZoo download URL for a given SHA-256."""
        query = urllib.parse.urlencode({"apikey": self.api_key, "sha256": sha256})
        return f"{self.base_url}/download?{query}"

    def download(self, sha256: str, dest: Path) -> Path:
        """Download one APK to ``dest``, verifying its SHA-256.

        Idempotent: if ``dest`` already exists and matches ``sha256`` it is
        left untouched and no request is made. On checksum mismatch nothing
        is written and ``AndroZooError`` is raised.
        """
        dest = Path(dest)
        if dest.exists() and _sha256_of_file(dest).lower() == sha256.lower():
            return dest

        data = self._fetch_with_retry(sha256)
        digest = hashlib.sha256(data).hexdigest()
        if digest.lower() != sha256.lower():
            raise AndroZooError(
                f"SHA-256 mismatch for {sha256}: server returned {digest}"
            )
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        return dest

    def _fetch_with_retry(self, sha256: str) -> bytes:
        """Fetch one APK, retrying transient failures with exponential backoff.

        A 10,000-APK run makes ten thousand requests over a day or more. Some
        will hit a rate limit or a dropped connection; without retrying, each
        one costs an APK and silently biases the corpus toward whatever was
        being served when the network was healthy.
        """
        url = self.download_url(sha256)
        last: BaseException | None = None
        for attempt in range(1, max(1, self.max_attempts) + 1):
            try:
                return self.fetcher(url)
            except Exception as exc:  # noqa: BLE001 - classified just below
                last = exc
                if not _is_transient(exc) or attempt == self.max_attempts:
                    break
                self.sleeper(self.backoff * (2 ** (attempt - 1)))
        raise AndroZooError(
            f"download failed for {sha256} after {self.max_attempts} attempt(s): {last}"
        ) from last


def _sha256_of_file(path: Path, *, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()
