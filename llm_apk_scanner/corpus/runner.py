"""Orchestration behind the ``corpus`` CLI commands.

Kept separate from ``cli.py`` so the sample-and-download logic is unit-tested
without click or network. ``run_sample`` builds the manifest from an AndroZoo
metadata slice; ``run_download`` fetches each APK, tolerating and recording
per-APK failures so a multi-thousand-APK run is never derailed by one bad file.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from llm_apk_scanner.corpus.androzoo import AndroZooClient, AndroZooError, parse_metadata
from llm_apk_scanner.corpus.manifest import write_manifest
from llm_apk_scanner.corpus.model import ApkRecord
from llm_apk_scanner.corpus.sample import DEFAULT_YEARS, SampleResult, stratified_sample


def run_sample(
    metadata_lines: Iterable[str],
    *,
    total: int,
    seed: int,
    out_path: Path,
    years: Sequence[int] = DEFAULT_YEARS,
) -> SampleResult:
    """Parse a metadata slice, draw the stratified sample, write the manifest."""
    records = parse_metadata(metadata_lines)
    result = stratified_sample(records, total=total, seed=seed, years=years)
    write_manifest(result, Path(out_path))
    return result


@dataclass
class DownloadReport:
    """Outcome of a batch download."""

    ok: list[str] = field(default_factory=list)
    failed: list[tuple[str, str]] = field(default_factory=list)

    @property
    def n_ok(self) -> int:
        return len(self.ok)

    @property
    def n_failed(self) -> int:
        return len(self.failed)


def run_download(
    records: Sequence[ApkRecord],
    *,
    dest_dir: Path,
    client: AndroZooClient,
    limit: int | None = None,
    delay: float = 1.0,
    on_progress: Callable[[str, bool], None] | None = None,
) -> DownloadReport:
    """Download each record's APK into ``dest_dir``.

    Failures (network, HTTP, or checksum) are captured per-APK and do not
    abort the run. ``on_progress(sha256, ok)`` is invoked after each attempt.

    Args:
        records: APKs to download.
        dest_dir: Directory to write ``<sha256>.apk`` files into.
        client: AndroZoo API client.
        limit: If set, only download the first N records.
        delay: Seconds to sleep between downloads (rate-limiting, default 1s).
               Set to 0 for no delay (pilots only).
        on_progress: Callback ``(sha256, ok)`` after each download attempt.
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    report = DownloadReport()

    selected = records if limit is None else list(records)[:limit]
    for i, record in enumerate(selected):
        sha = record.sha256
        dest = dest_dir / f"{sha}.apk"
        try:
            client.download(sha, dest)
        except (AndroZooError, OSError) as exc:
            report.failed.append((sha, str(exc)))
            ok = False
        else:
            report.ok.append(sha)
            ok = True
        if on_progress is not None:
            on_progress(sha, ok)
        # Rate-limit: sleep between downloads (not after the last one)
        if delay > 0 and i < len(selected) - 1:
            time.sleep(delay)
    return report
