"""HTML reporter — produces a self-contained, single-file HTML report."""

from __future__ import annotations

import html
from llm_apk_scanner.models import ScanResult, Severity


SEVERITY_COLORS = {
    Severity.CRITICAL: "#b91c1c",
    Severity.HIGH: "#ea580c",
    Severity.MEDIUM: "#ca8a04",
    Severity.LOW: "#0284c7",
    Severity.INFO: "#475569",
}

CSS = """
body{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;
 margin:2em auto;max-width:980px;color:#0f172a;background:#fafafa;line-height:1.5}
h1{margin-bottom:.2em}
.subtitle{color:#64748b;font-size:1.1em;margin-bottom:2em}
.meta{background:#fff;border:1px solid #e2e8f0;border-radius:6px;padding:1em 1.4em;margin:1em 0}
.meta dt{font-weight:600;display:inline-block;min-width:160px;color:#475569}
.meta dd{display:inline;margin:0}
table.summary{border-collapse:collapse;margin:1.5em 0;font-size:.95em}
table.summary th, table.summary td{border:1px solid #e2e8f0;padding:.5em 1em;text-align:left}
table.summary th{background:#f1f5f9}
.badge{display:inline-block;padding:1px 9px;border-radius:99px;color:#fff;font-size:.75em;
 font-weight:600;letter-spacing:.04em;text-transform:uppercase}
details{background:#fff;border:1px solid #e2e8f0;border-radius:6px;padding:.6em 1em;margin:.6em 0}
details[open]{box-shadow:0 1px 4px rgba(0,0,0,.04)}
summary{cursor:pointer;font-weight:600}
summary .title{margin-left:.5em}
.location{font-family:'SF Mono',Monaco,monospace;color:#64748b;font-size:.85em;margin-top:.3em}
.evidence{font-family:'SF Mono',Monaco,monospace;background:#f8fafc;border:1px solid #e2e8f0;
 border-radius:4px;padding:.6em .9em;margin-top:.5em;white-space:pre-wrap;word-break:break-all;
 color:#334155;font-size:.85em}
.confidence-bar{display:inline-block;width:120px;height:6px;background:#e2e8f0;border-radius:3px;
 overflow:hidden;vertical-align:middle;margin-left:.5em}
.confidence-bar .fill{height:100%;background:linear-gradient(to right,#0284c7,#0369a1)}
.no-findings{padding:1.5em;text-align:center;color:#64748b}
"""


class HtmlReporter:
    def render(self, result: ScanResult) -> str:
        info = result.apk
        rows: list[str] = []
        rows.append("<!DOCTYPE html><html lang='en'><head>")
        rows.append("<meta charset='UTF-8'>")
        rows.append(
            f"<title>Scan report — {html.escape(info.package_name)}</title>"
        )
        rows.append(f"<style>{CSS}</style></head><body>")

        rows.append("<h1>Scan report</h1>")
        rows.append(
            f"<div class='subtitle'>{html.escape(info.package_name)} "
            f"v{html.escape(info.version_name or '?')}</div>"
        )

        rows.append("<dl class='meta'>")
        for label, value in [
            ("Package", info.package_name),
            ("Version", f"{info.version_name} (code {info.version_code})"),
            ("Min/Target SDK", f"{info.min_sdk} / {info.target_sdk}"),
            ("SHA-256", info.sha256),
            ("Path", info.file_path),
            ("Scanner version", result.scanner_version),
            ("Scan duration", f"{result.scan_duration_seconds}s"),
            (
                "LLM-integrated",
                "Yes" if result.is_llm_integrated else "No",
            ),
            (
                "Detected providers",
                ", ".join(p.value for p in result.detected_providers) or "—",
            ),
        ]:
            rows.append(
                f"<div><dt>{html.escape(label)}</dt> "
                f"<dd>{html.escape(str(value))}</dd></div>"
            )
        rows.append("</dl>")

        rows.append("<h2>Findings summary</h2>")
        rows.append("<table class='summary'>")
        rows.append("<tr><th>Severity</th><th>Count</th></tr>")
        for sev in (
            Severity.CRITICAL,
            Severity.HIGH,
            Severity.MEDIUM,
            Severity.LOW,
            Severity.INFO,
        ):
            count = result.summary[sev.value]
            color = SEVERITY_COLORS[sev]
            rows.append(
                f"<tr><td><span class='badge' style='background:{color}'>"
                f"{sev.value}</span></td><td>{count}</td></tr>"
            )
        rows.append("</table>")

        if not result.findings:
            rows.append("<div class='no-findings'>No findings.</div>")
        else:
            rows.append("<h2>Findings</h2>")
            for i, f in enumerate(result.findings, 1):
                color = SEVERITY_COLORS[f.severity]
                title = html.escape(f.title)
                rows.append("<details>")
                rows.append(
                    f"<summary>"
                    f"<span class='badge' style='background:{color}'>"
                    f"{f.severity.value}</span>"
                    f"<span class='title'>#{i} {title}</span>"
                    f"<span class='confidence-bar'>"
                    f"<span class='fill' style='width:{int(f.confidence*100)}%'></span>"
                    f"</span>"
                    f"</summary>"
                )
                rows.append(f"<p>{html.escape(f.description)}</p>")
                loc = f.location
                location_text = html.escape(loc.file_path)
                if loc.line_or_offset is not None:
                    location_text += f" @ offset {loc.line_or_offset}"
                rows.append(f"<div class='location'>{location_text}</div>")
                rows.append(f"<div>Confidence: {f.confidence:.2f}</div>")
                if f.provider:
                    rows.append(
                        f"<div>Provider: <code>{html.escape(f.provider.value)}</code></div>"
                    )
                rows.append(
                    f"<div class='evidence'>"
                    f"{html.escape(f.redacted_evidence(120))}"
                    f"</div>"
                )
                rows.append("</details>")

        rows.append("</body></html>")
        return "\n".join(rows)
