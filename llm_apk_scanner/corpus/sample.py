"""Two-stage stratified sampling over the AndroZoo frame.

Implements the sampling strategy in ``docs/METHODOLOGY.md`` §2.3:

1. Restrict the frame to APKs published in the target years (2024-2026).
2. Partition into strata of (year x store).
3. Allocate the requested sample size across strata **proportionally** to
   each stratum's population (largest-remainder rounding, capped by the
   stratum's size), then draw without replacement within each stratum.

The draw is reproducible: given the same records, ``total`` and ``seed`` the
selected SHA-256 set is identical. This is what lets the dissertation publish
a SHA-256 manifest that any AndroZoo-licensed reader can recover.
"""

from __future__ import annotations

import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from llm_apk_scanner.corpus.model import ApkRecord

Stratum = tuple[int, str]

DEFAULT_YEARS: tuple[int, ...] = (2024, 2025, 2026)


@dataclass
class SampleResult:
    """The outcome of a stratified draw."""

    records: list[ApkRecord]
    allocation: dict[Stratum, int] = field(default_factory=dict)
    requested: int = 0

    @property
    def size(self) -> int:
        return len(self.records)


def _allocate(sizes: dict[Stratum, int], total: int) -> dict[Stratum, int]:
    """Proportional allocation of ``total`` across strata, capped by size.

    Uses the largest-remainder method so the allocation sums to
    ``min(total, sum(sizes))`` exactly, never exceeding any stratum's
    available population.
    """
    population = sum(sizes.values())
    target = min(total, population)
    alloc: dict[Stratum, int] = {k: 0 for k in sizes}
    if population == 0 or target == 0:
        return alloc

    remainders: dict[Stratum, float] = {}
    for key, size in sizes.items():
        ideal = size / population * target
        base = min(int(ideal), size)
        alloc[key] = base
        remainders[key] = ideal - base

    leftover = target - sum(alloc.values())
    # Deterministic tie-break: larger fractional remainder first, then key.
    order = sorted(sizes, key=lambda k: (-remainders[k], k))
    while leftover > 0:
        progressed = False
        for key in order:
            if leftover == 0:
                break
            if alloc[key] < sizes[key]:
                alloc[key] += 1
                leftover -= 1
                progressed = True
        if not progressed:  # every stratum at capacity
            break
    return alloc


def stratified_sample(
    records: Iterable[ApkRecord],
    *,
    total: int,
    seed: int,
    years: Sequence[int] = DEFAULT_YEARS,
) -> SampleResult:
    """Draw a reproducible two-stage stratified sample.

    Args:
        records: the candidate frame (an AndroZoo metadata slice).
        total: desired sample size; the result is capped at the eligible
            population when the frame is smaller.
        seed: RNG seed making the draw reproducible.
        years: publication years that define eligibility.

    Returns:
        A ``SampleResult`` whose ``records`` are sorted by SHA-256 for a
        stable, manifest-friendly ordering.
    """
    year_set = set(years)
    buckets: dict[Stratum, list[ApkRecord]] = {}
    for record in records:
        if record.year not in year_set:
            continue
        key: Stratum = (record.year, record.store_stratum)
        buckets.setdefault(key, []).append(record)

    # Stable within-bucket order so the seeded draw is fully deterministic.
    sizes: dict[Stratum, int] = {}
    for key in sorted(buckets):
        buckets[key].sort(key=lambda r: r.sha256)
        sizes[key] = len(buckets[key])

    allocation = _allocate(sizes, total)

    rng = random.Random(seed)
    selected: list[ApkRecord] = []
    for key in sorted(allocation):
        take = allocation[key]
        if take <= 0:
            continue
        selected.extend(rng.sample(buckets[key], take))

    selected.sort(key=lambda r: r.sha256)
    allocation = {k: v for k, v in allocation.items() if v > 0}
    return SampleResult(records=selected, allocation=allocation, requested=total)
