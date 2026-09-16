"""Read and write the corpus manifest.

The manifest is a CSV of the sampled APKs (SHA-256 plus strata). It is
written *before* any download so the sample frame is fixed and publishable,
and it is the artefact a reader uses to recover the exact corpus.
"""

from __future__ import annotations

import csv
from collections.abc import Iterable
from pathlib import Path

from llm_apk_scanner.corpus.model import ApkRecord
from llm_apk_scanner.corpus.sample import SampleResult

_FIELDS = ["sha256", "pkg_name", "year", "store", "vercode", "markets", "apk_size"]


def write_manifest(sample: SampleResult, path: Path) -> Path:
    """Write a sample's records to ``path`` as a CSV manifest."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_FIELDS)
        writer.writeheader()
        for record in sample.records:
            writer.writerow(_row(record))
    return path


def _row(record: ApkRecord) -> dict[str, object]:
    return {
        "sha256": record.sha256,
        "pkg_name": record.pkg_name,
        "year": record.year if record.year is not None else "",
        "store": record.store_stratum,
        "vercode": record.vercode or "",
        "markets": record.markets,
        "apk_size": record.apk_size if record.apk_size is not None else "",
    }


def read_manifest(path: Path) -> list[ApkRecord]:
    """Read a manifest CSV back into ``ApkRecord`` objects.

    The ``dex_date`` is reconstructed as ``<year>-01-01`` so the derived
    ``year`` round-trips; the store stratum is re-derived from ``markets``.
    """
    records: list[ApkRecord] = []
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            year = row.get("year") or ""
            size_raw = row.get("apk_size") or ""
            records.append(
                ApkRecord(
                    sha256=row["sha256"],
                    pkg_name=row.get("pkg_name", ""),
                    markets=row.get("markets", ""),
                    dex_date=f"{year}-01-01 00:00:00" if year else None,
                    vercode=row.get("vercode") or None,
                    apk_size=int(size_raw) if size_raw.isdigit() else None,
                )
            )
    return records


def manifest_shas(records: Iterable[ApkRecord]) -> list[str]:
    """Convenience: the SHA-256 list for a set of records."""
    return [r.sha256 for r in records]
