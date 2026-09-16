"""Tests for the HTML reporter."""

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
from llm_apk_scanner.reporters.html_reporter import HtmlReporter


def _result_with_findings() -> ScanResult:
    apk = ApkInfo(
        package_name="com.example.test",
        version_name="1.0",
        version_code=1,
        min_sdk=21,
        target_sdk=34,
        file_path="/tmp/x.apk",
        sha256="0" * 64,
    )
    findings = [
        Finding(
            kind=FindingKind.SECRET,
            severity=Severity.CRITICAL,
            title="OpenAI key",
            description="A leaked OpenAI key was found.",
            location=SourceLocation(file_path="classes.dex", line_or_offset=42),
            evidence="sk-proj-AAAA",
            confidence=0.95,
            provider=Provider.OPENAI,
        ),
        Finding(
            kind=FindingKind.SYSTEM_PROMPT,
            severity=Severity.MEDIUM,
            title="System prompt",
            description="Embedded system prompt.",
            location=SourceLocation(file_path="classes.dex"),
            evidence="You are a helpful assistant.",
            confidence=0.7,
        ),
    ]
    return ScanResult(
        apk=apk,
        is_llm_integrated=True,
        detected_providers=[Provider.OPENAI],
        findings=findings,
        scan_duration_seconds=0.123,
        scanner_version="0.1.0",
    )


def test_html_includes_package_name() -> None:
    html = HtmlReporter().render(_result_with_findings())
    assert "com.example.test" in html


def test_html_renders_severity_badges() -> None:
    html = HtmlReporter().render(_result_with_findings())
    assert "critical" in html
    assert "medium" in html
    # Each finding renders as a <details> block.
    assert html.count("<details>") == 2


def test_html_includes_summary_table() -> None:
    html = HtmlReporter().render(_result_with_findings())
    assert "Findings summary" in html
    assert "table" in html.lower()


def test_html_no_findings_renders_message() -> None:
    apk = ApkInfo(
        package_name="com.empty",
        version_name="1",
        version_code=1,
        min_sdk=21,
        target_sdk=34,
        file_path="/tmp/y.apk",
        sha256="0" * 64,
    )
    result = ScanResult(
        apk=apk,
        is_llm_integrated=False,
        detected_providers=[],
        findings=[],
        scan_duration_seconds=0.01,
        scanner_version="0.1.0",
    )
    html = HtmlReporter().render(result)
    assert "No findings" in html


def test_html_escapes_evidence() -> None:
    """HTML special characters in evidence must be escaped."""
    apk = ApkInfo(
        package_name="com.example",
        version_name="1",
        version_code=1,
        min_sdk=21,
        target_sdk=34,
        file_path="/tmp/x.apk",
        sha256="0" * 64,
    )
    findings = [
        Finding(
            kind=FindingKind.SYSTEM_PROMPT,
            severity=Severity.LOW,
            title="<script>alert('xss')</script>",
            description="Test",
            location=SourceLocation(file_path="x.dex"),
            evidence="<script>alert('xss')</script>",
            confidence=0.5,
        )
    ]
    result = ScanResult(
        apk=apk,
        is_llm_integrated=True,
        detected_providers=[],
        findings=findings,
        scan_duration_seconds=0.01,
        scanner_version="0.1.0",
    )
    html = HtmlReporter().render(result)
    assert "<script>" not in html  # raw tag
    assert "&lt;script&gt;" in html  # escaped form
