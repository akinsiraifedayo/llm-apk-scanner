"""Tests for statistics and aggregation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_apk_scanner.models import (
    ApkInfo,
    Finding,
    FindingKind,
    Provider,
    ScanResult,
    Severity,
    SourceLocation,
)
from llm_apk_scanner.stats import (
    aggregate,
    load_results_dir,
    wilson_interval,
    write_csv,
    write_json,
)


def _result(
    *,
    is_llm: bool,
    providers: list[Provider],
    findings: list[Finding],
    name: str = "x",
) -> ScanResult:
    return ScanResult(
        apk=ApkInfo(
            package_name=f"com.example.{name}",
            version_name="1.0",
            version_code=1,
            min_sdk=21,
            target_sdk=34,
            file_path=f"/tmp/{name}.apk",
            sha256="0" * 64,
        ),
        is_llm_integrated=is_llm,
        detected_providers=providers,
        findings=findings,
        scan_duration_seconds=0.01,
        scanner_version="0.1.0",
    )


def _finding(
    *, kind: FindingKind, severity: Severity, confidence: float = 0.9, provider=None
) -> Finding:
    return Finding(
        kind=kind,
        severity=severity,
        title="t",
        description="d",
        location=SourceLocation(file_path="classes.dex"),
        evidence="x",
        confidence=confidence,
        provider=provider,
    )


def test_wilson_interval_zero_n() -> None:
    assert wilson_interval(0, 0) == (0.0, 0.0)


def test_wilson_interval_known_values() -> None:
    # k=10, n=100 → p_hat 0.10. Expected Wilson 95% ~ (0.055, 0.175).
    lo, hi = wilson_interval(10, 100)
    assert 0.04 < lo < 0.07
    assert 0.16 < hi < 0.19


def test_wilson_interval_zero_successes() -> None:
    lo, hi = wilson_interval(0, 100)
    assert lo == 0.0
    assert 0.02 < hi < 0.05


def test_wilson_interval_full_successes() -> None:
    lo, hi = wilson_interval(100, 100)
    assert hi == pytest.approx(1.0)
    assert 0.95 < lo < 0.97


def test_aggregate_counts_apks_with_finding() -> None:
    results = [
        _result(
            is_llm=True,
            providers=[Provider.OPENAI],
            findings=[
                _finding(
                    kind=FindingKind.SECRET,
                    severity=Severity.CRITICAL,
                    provider=Provider.OPENAI,
                )
            ],
            name="a",
        ),
        _result(
            is_llm=True,
            providers=[Provider.OPENAI],
            findings=[],
            name="b",
        ),
        _result(
            is_llm=False,
            providers=[],
            findings=[],
            name="c",
        ),
    ]
    agg = aggregate(results)
    assert agg.n_apks_total == 3
    assert agg.n_apks_llm_integrated == 2

    secret_kind = next(s for s in agg.by_kind if s.category == "secret")
    assert secret_kind.successes == 1
    assert secret_kind.n == 2
    assert 0.4 < secret_kind.proportion < 0.6
    assert secret_kind.ci_lower < secret_kind.proportion < secret_kind.ci_upper


def test_aggregate_low_confidence_excluded() -> None:
    """Findings below 0.85 confidence don't count toward prevalence."""
    results = [
        _result(
            is_llm=True,
            providers=[Provider.OPENAI],
            findings=[
                _finding(
                    kind=FindingKind.SECRET,
                    severity=Severity.MEDIUM,
                    confidence=0.4,
                    provider=Provider.OPENAI,
                )
            ],
        )
    ]
    agg = aggregate(results)
    secret_kind = next(s for s in agg.by_kind if s.category == "secret")
    assert secret_kind.successes == 0


def test_csv_and_json_round_trip(tmp_path: Path) -> None:
    results = [
        _result(
            is_llm=True,
            providers=[Provider.OPENAI],
            findings=[
                _finding(
                    kind=FindingKind.SECRET,
                    severity=Severity.CRITICAL,
                    provider=Provider.OPENAI,
                )
            ],
        )
    ]
    agg = aggregate(results)

    csv_path = tmp_path / "agg.csv"
    json_path = tmp_path / "agg.json"
    write_csv(agg, csv_path)
    write_json(agg, json_path)

    assert csv_path.exists()
    assert json_path.exists()
    payload = json.loads(json_path.read_text())
    assert payload["n_apks_llm_integrated"] == 1


def test_load_results_dir(tmp_path: Path) -> None:
    """Round-trip via the per-APK JSON files written by the pipeline."""
    result = _result(
        is_llm=True,
        providers=[Provider.OPENAI],
        findings=[
            _finding(
                kind=FindingKind.SECRET,
                severity=Severity.CRITICAL,
                provider=Provider.OPENAI,
            )
        ],
    )
    p = tmp_path / "com.example.x.json"
    p.write_text(json.dumps(result.to_dict()), encoding="utf-8")
    loaded = load_results_dir(tmp_path)
    assert len(loaded) == 1
    assert loaded[0].apk.package_name == "com.example.x"
    assert loaded[0].is_llm_integrated
    assert len(loaded[0].findings) == 1
    assert loaded[0].findings[0].kind == FindingKind.SECRET
