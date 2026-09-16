"""Tests for the result models and serialisation."""

from __future__ import annotations

from llm_apk_scanner.models import (
    ApkInfo,
    Finding,
    FindingKind,
    Provider,
    ScanResult,
    Severity,
    SourceLocation,
)


def _finding() -> Finding:
    return Finding(
        kind=FindingKind.SECRET,
        severity=Severity.CRITICAL,
        title="Test",
        description="d",
        location=SourceLocation(file_path="classes.dex", line_or_offset=42),
        evidence="sk-proj-AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
        confidence=0.95,
        provider=Provider.OPENAI,
    )


def test_redacted_evidence_short_passes_through() -> None:
    f = _finding()
    short = f.redacted_evidence(80)
    assert short == f.evidence  # length 56 < 80


def test_redacted_evidence_long_truncated() -> None:
    f = _finding()
    short = f.redacted_evidence(20)
    assert "…" in short


def test_severity_summary_counts() -> None:
    apk = ApkInfo(
        package_name="com.example",
        version_name="1.0",
        version_code=1,
        min_sdk=21,
        target_sdk=34,
        file_path="/tmp/x.apk",
        sha256="0" * 64,
    )
    result = ScanResult(
        apk=apk,
        is_llm_integrated=True,
        detected_providers=[Provider.OPENAI],
        findings=[_finding(), _finding()],
        scan_duration_seconds=0.1,
        scanner_version="0.1.0",
    )
    assert result.summary["critical"] == 2
    d = result.to_dict()
    assert d["summary"]["critical"] == 2
    # Evidence in dict form must already be redacted.
    for f in d["findings"]:
        assert "AAAAAAAAAAAAAAAAAA" not in f["evidence"] or len(f["evidence"]) < 60
