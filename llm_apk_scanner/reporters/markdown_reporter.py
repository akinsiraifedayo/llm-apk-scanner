"""Markdown report generator suitable for inclusion in dissertation appendices."""

from __future__ import annotations

from llm_apk_scanner.models import ScanResult, Severity


SEVERITY_BADGES = {
    Severity.CRITICAL: "[CRITICAL]",
    Severity.HIGH: "[HIGH]",
    Severity.MEDIUM: "[MEDIUM]",
    Severity.LOW: "[LOW]",
    Severity.INFO: "[INFO]",
}


class MarkdownReporter:
    def render(self, result: ScanResult) -> str:
        lines: list[str] = []
        lines.append(f"# Scan report: {result.apk.package_name}")
        lines.append("")
        lines.append("## Application")
        lines.append("")
        lines.append(f"- Package: `{result.apk.package_name}`")
        lines.append(
            f"- Version: {result.apk.version_name} "
            f"(code {result.apk.version_code})"
        )
        lines.append(
            f"- min/target SDK: {result.apk.min_sdk} / {result.apk.target_sdk}"
        )
        lines.append(f"- SHA256: `{result.apk.sha256}`")
        lines.append(f"- File: `{result.apk.file_path}`")
        lines.append("")

        lines.append("## LLM integration")
        lines.append("")
        lines.append(
            f"- Confirmed LLM-integrated: **{'yes' if result.is_llm_integrated else 'no'}**"
        )
        if result.detected_providers:
            providers = ", ".join(p.value for p in result.detected_providers)
            lines.append(f"- Detected providers: {providers}")
        lines.append("")

        lines.append("## Findings summary")
        lines.append("")
        lines.append("| Severity | Count |")
        lines.append("|----------|-------|")
        for sev in (
            Severity.CRITICAL,
            Severity.HIGH,
            Severity.MEDIUM,
            Severity.LOW,
            Severity.INFO,
        ):
            lines.append(f"| {sev.value} | {result.summary[sev.value]} |")
        lines.append("")

        if not result.findings:
            lines.append("_No findings._")
            return "\n".join(lines)

        lines.append("## Findings")
        lines.append("")
        for i, f in enumerate(result.findings, 1):
            badge = SEVERITY_BADGES[f.severity]
            lines.append(f"### {i}. {badge} {f.title}")
            lines.append("")
            lines.append(f"- Kind: `{f.kind.value}`")
            if f.provider:
                lines.append(f"- Provider: `{f.provider.value}`")
            lines.append(f"- Confidence: {f.confidence:.2f}")
            loc = f.location
            location_line = f"- Location: `{loc.file_path}`"
            if loc.line_or_offset is not None:
                location_line += f" @ offset {loc.line_or_offset}"
            lines.append(location_line)
            lines.append("")
            lines.append(f"{f.description}")
            lines.append("")
            lines.append("```")
            lines.append(f.redacted_evidence(80))
            lines.append("```")
            lines.append("")

        return "\n".join(lines)
