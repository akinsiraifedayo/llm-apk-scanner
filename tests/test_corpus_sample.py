"""Tests for the AndroZoo two-stage stratified sampler.

These are pure, network-free tests: the sampler operates on in-memory
``ApkRecord`` objects. They pin down the sampling behaviour that the
dissertation's methodology (§2.3) relies on: year+store strata,
proportional allocation, and reproducibility under a fixed seed.
"""

from __future__ import annotations

from llm_apk_scanner.corpus.model import ApkRecord
from llm_apk_scanner.corpus.sample import stratified_sample


def _record(i: int, *, year: int, market: str, pkg: str = "com.example.app") -> ApkRecord:
    return ApkRecord(
        sha256=f"{i:064x}",
        pkg_name=f"{pkg}{i}",
        markets=market,
        dex_date=f"{year}-01-01 00:00:00",
        added=f"{year}-06-01 00:00:00",
        vercode=str(i),
        apk_size=1000 + i,
    )


def _population(n_play: int, n_fdroid: int, *, year: int) -> list[ApkRecord]:
    recs: list[ApkRecord] = []
    idx = 0
    for _ in range(n_play):
        recs.append(_record(idx, year=year, market="play.google.com"))
        idx += 1
    for _ in range(n_fdroid):
        recs.append(_record(idx, year=year, market="f-droid.org"))
        idx += 1
    return recs


def test_record_year_and_store_derivation() -> None:
    play = _record(1, year=2025, market="play.google.com")
    fdroid = _record(2, year=2024, market="appchina|f-droid.org")
    other = _record(3, year=2026, market="appchina")
    assert play.year == 2025
    assert play.store_stratum == "google_play"
    assert fdroid.store_stratum == "f_droid"
    assert other.store_stratum == "other"


def test_sample_size_matches_requested_total() -> None:
    pop = _population(800, 200, year=2025)
    result = stratified_sample(pop, total=100, seed=42)
    assert len(result.records) == 100


def test_records_outside_target_years_excluded() -> None:
    pop = _population(50, 50, year=2019)  # out of 2024-2026 window
    result = stratified_sample(pop, total=10, seed=1)
    assert result.records == []
    assert result.allocation == {}


def test_proportional_allocation_across_strata() -> None:
    # 80% Google Play, 20% F-Droid → sample should mirror that split.
    pop = _population(800, 200, year=2025)
    result = stratified_sample(pop, total=100, seed=7)
    play = sum(1 for r in result.records if r.store_stratum == "google_play")
    fdroid = sum(1 for r in result.records if r.store_stratum == "f_droid")
    assert play == 80
    assert fdroid == 20


def test_deterministic_under_fixed_seed() -> None:
    pop = _population(500, 500, year=2024)
    a = stratified_sample(pop, total=50, seed=99)
    b = stratified_sample(pop, total=50, seed=99)
    assert [r.sha256 for r in a.records] == [r.sha256 for r in b.records]


def test_different_seed_changes_selection() -> None:
    pop = _population(500, 500, year=2024)
    a = stratified_sample(pop, total=50, seed=1)
    b = stratified_sample(pop, total=50, seed=2)
    assert {r.sha256 for r in a.records} != {r.sha256 for r in b.records}


def test_allocation_capped_by_stratum_population() -> None:
    # Only 5 F-Droid apps exist but proportional demand is higher;
    # the sampler must not over-draw, and must not duplicate.
    pop = _population(995, 5, year=2026)
    result = stratified_sample(pop, total=100, seed=3)
    fdroid = [r for r in result.records if r.store_stratum == "f_droid"]
    assert len(fdroid) <= 5
    assert len({r.sha256 for r in result.records}) == len(result.records)


def test_sample_truncates_when_population_smaller_than_total() -> None:
    pop = _population(30, 10, year=2025)
    result = stratified_sample(pop, total=100, seed=5)
    assert len(result.records) == 40  # cannot exceed the population
