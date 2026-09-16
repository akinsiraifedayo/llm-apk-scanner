"""
Statistics and aggregation utilities.

Used by Chapter 6 of the dissertation to translate per-APK
ScanResult objects into population-level prevalence figures with
honest uncertainty.

The headline computation is the **Wilson score interval** for a
proportion, which is the standard small-sample-friendly
confidence interval used in security-measurement literature.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from llm_apk_scanner.models import (
    FindingKind,
    Provider,
    ScanResult,
    Severity,
)


# ---------------------------------------------------------------------------
# Wilson score interval
# ---------------------------------------------------------------------------


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """
    Wilson score 95% confidence interval for a binomial proportion.

    Args:
        successes: number of positive outcomes (k).
        n: number of trials.
        z: z-score (default 1.96 for 95% CI).

    Returns:
        (lower, upper) bounds on the proportion.

    Reference: Wilson, E. B. (1927). Probable inference, the law of
    succession, and statistical inference. JASA 22, 209–212.
    """
    if n == 0:
        return (0.0, 0.0)
    p_hat = successes / n
    denom = 1 + z * z / n
    centre = p_hat + z * z / (2 * n)
    half_width = z * math.sqrt(p_hat * (1 - p_hat) / n + z * z / (4 * n * n))
    lower = (centre - half_width) / denom
    upper = (centre + half_width) / denom
    return (max(0.0, lower), min(1.0, upper))


# ---------------------------------------------------------------------------
# Aggregate result type
# ---------------------------------------------------------------------------


@dataclass
class CategoryStat:
    """Per-category prevalence figure with CI."""

    category: str
    successes: int
    n: int
    proportion: float
    ci_lower: float
    ci_upper: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CorpusAggregate:
    """Top-level aggregate across all scanned APKs."""

    n_apks_total: int
    n_apks_llm_integrated: int
    by_severity: list[CategoryStat] = field(default_factory=list)
    by_kind: list[CategoryStat] = field(default_factory=list)
    by_provider: list[CategoryStat] = field(default_factory=list)
    by_provider_and_kind: list[CategoryStat] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_apks_total": self.n_apks_total,
            "n_apks_llm_integrated": self.n_apks_llm_integrated,
            "by_severity": [s.to_dict() for s in self.by_severity],
            "by_kind": [s.to_dict() for s in self.by_kind],
            "by_provider": [s.to_dict() for s in self.by_provider],
            "by_provider_and_kind": [
                s.to_dict() for s in self.by_provider_and_kind
            ],
        }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _stat(category: str, successes: int, n: int) -> CategoryStat:
    proportion = successes / n if n else 0.0
    lo, hi = wilson_interval(successes, n)
    return CategoryStat(
        category=category,
        successes=successes,
        n=n,
        proportion=proportion,
        ci_lower=lo,
        ci_upper=hi,
    )


def aggregate(
    results: Iterable[ScanResult],
    high_confidence_threshold: float = 0.85,
) -> CorpusAggregate:
    """
    Aggregate a stream of ScanResults into population-level stats.

    "Successes" for prevalence statistics are *APKs containing at
    least one high-confidence finding of the relevant kind*. We do
    not count individual findings, only APKs, so per-APK rates are
    bounded in [0, 1].
    """
    results_list = list(results)
    n_total = len(results_list)
    n_llm = sum(1 for r in results_list if r.is_llm_integrated)

    by_severity: list[CategoryStat] = []
    for sev in Severity:
        successes = sum(
            1
            for r in results_list
            if r.is_llm_integrated
            and any(
                f.severity == sev and f.confidence >= high_confidence_threshold
                for f in r.findings
            )
        )
        by_severity.append(_stat(sev.value, successes, n_llm))

    by_kind: list[CategoryStat] = []
    for kind in FindingKind:
        successes = sum(
            1
            for r in results_list
            if r.is_llm_integrated
            and any(
                f.kind == kind and f.confidence >= high_confidence_threshold
                for f in r.findings
            )
        )
        by_kind.append(_stat(kind.value, successes, n_llm))

    by_provider: list[CategoryStat] = []
    for provider in Provider:
        n_with_provider = sum(
            1 for r in results_list if provider in r.detected_providers
        )
        successes = sum(
            1
            for r in results_list
            if provider in r.detected_providers
            and any(
                f.provider == provider
                and f.kind == FindingKind.SECRET
                and f.confidence >= high_confidence_threshold
                for f in r.findings
            )
        )
        if n_with_provider:
            by_provider.append(_stat(provider.value, successes, n_with_provider))

    by_provider_and_kind: list[CategoryStat] = []
    for provider in Provider:
        for kind in (
            FindingKind.SECRET,
            FindingKind.SYSTEM_PROMPT,
            FindingKind.TOOL_SCHEMA,
            FindingKind.RAG_ARTEFACT,
        ):
            n_with_provider = sum(
                1 for r in results_list if provider in r.detected_providers
            )
            successes = sum(
                1
                for r in results_list
                if provider in r.detected_providers
                and any(
                    f.kind == kind
                    and f.confidence >= high_confidence_threshold
                    for f in r.findings
                )
            )
            if n_with_provider:
                by_provider_and_kind.append(
                    _stat(f"{provider.value}/{kind.value}", successes, n_with_provider)
                )

    return CorpusAggregate(
        n_apks_total=n_total,
        n_apks_llm_integrated=n_llm,
        by_severity=by_severity,
        by_kind=by_kind,
        by_provider=by_provider,
        by_provider_and_kind=by_provider_and_kind,
    )


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def load_results_dir(path: str | Path) -> list[ScanResult]:
    """Load a directory of per-APK JSON reports written by `pipeline`."""
    path = Path(path)
    out: list[ScanResult] = []
    for jf in sorted(path.glob("*.json")):
        if jf.name.startswith("_"):
            continue  # skip _aggregate.json and friends
        with jf.open(encoding="utf-8") as fh:
            d = json.load(fh)
        out.append(_dict_to_scan_result(d))
    return out


def _dict_to_scan_result(d: dict[str, Any]) -> ScanResult:
    """Reconstruct a ScanResult from its serialised dict form."""
    from llm_apk_scanner.models import ApkInfo, Finding, SourceLocation

    apk = ApkInfo(**d["apk"])
    providers = [Provider(p) for p in d["detected_providers"]]
    findings: list[Finding] = []
    for f in d["findings"]:
        loc = SourceLocation(**f["location"])
        findings.append(
            Finding(
                kind=FindingKind(f["kind"]),
                severity=Severity(f["severity"]),
                title=f["title"],
                description=f["description"],
                location=loc,
                evidence=f["evidence"],
                confidence=f["confidence"],
                provider=Provider(f["provider"]) if f["provider"] else None,
                metadata=f.get("metadata", {}),
            )
        )
    return ScanResult(
        apk=apk,
        is_llm_integrated=d["is_llm_integrated"],
        detected_providers=providers,
        findings=findings,
        scan_duration_seconds=d["scan_duration_seconds"],
        scanner_version=d["scanner_version"],
    )


def write_csv(aggregate: CorpusAggregate, path: str | Path) -> None:
    """Emit a CSV ready for inclusion in the dissertation."""
    path = Path(path)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "category_kind",
                "category",
                "successes",
                "n",
                "proportion",
                "ci_lower",
                "ci_upper",
            ]
        )
        for label, stats in (
            ("severity", aggregate.by_severity),
            ("finding_kind", aggregate.by_kind),
            ("provider_secrets", aggregate.by_provider),
            ("provider_and_kind", aggregate.by_provider_and_kind),
        ):
            for s in stats:
                w.writerow(
                    [
                        label,
                        s.category,
                        s.successes,
                        s.n,
                        f"{s.proportion:.4f}",
                        f"{s.ci_lower:.4f}",
                        f"{s.ci_upper:.4f}",
                    ]
                )


def write_json(aggregate: CorpusAggregate, path: str | Path) -> None:
    Path(path).write_text(
        json.dumps(aggregate.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
