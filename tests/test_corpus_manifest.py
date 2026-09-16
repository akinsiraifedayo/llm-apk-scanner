"""Tests for the corpus manifest (the reproducibility artefact).

The manifest is the SHA-256 list published so that any AndroZoo-licensed
reader can recover the exact corpus (``docs/METHODOLOGY.md`` §Reproducibility).
"""

from __future__ import annotations

from pathlib import Path

from llm_apk_scanner.corpus.manifest import read_manifest, write_manifest
from llm_apk_scanner.corpus.model import ApkRecord
from llm_apk_scanner.corpus.sample import stratified_sample


def _record(i: int, *, year: int, market: str) -> ApkRecord:
    return ApkRecord(
        sha256=f"{i:064x}",
        pkg_name=f"com.example.app{i}",
        markets=market,
        dex_date=f"{year}-01-01 00:00:00",
        vercode=str(i),
        apk_size=1000 + i,
    )


def test_manifest_round_trip(tmp_path: Path) -> None:
    pop = [_record(i, year=2025, market="play.google.com") for i in range(20)]
    result = stratified_sample(pop, total=10, seed=1)

    path = tmp_path / "manifest.csv"
    write_manifest(result, path)
    assert path.exists()

    loaded = read_manifest(path)
    assert [r.sha256 for r in loaded] == [r.sha256 for r in result.records]
    assert loaded[0].store_stratum == "google_play"
    assert loaded[0].year == 2025


def test_manifest_has_header_and_row_count(tmp_path: Path) -> None:
    pop = [_record(i, year=2024, market="f-droid.org") for i in range(5)]
    result = stratified_sample(pop, total=5, seed=2)

    path = tmp_path / "manifest.csv"
    write_manifest(result, path)
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert lines[0].startswith("sha256,")
    assert len(lines) == 1 + 5  # header + one row per record
