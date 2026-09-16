"""
Result schemas for the scanner.

These classes are deliberately framework-free (no pydantic, no attrs)
so that the scanner has zero non-essential dependencies in its hot path.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class Severity(str, Enum):
    """Finding severity, aligned loosely to OWASP risk ratings."""

    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class FindingKind(str, Enum):
    """Top-level taxonomy of static findings."""

    LLM_ENDPOINT = "llm_endpoint"
    LLM_SDK_REFERENCE = "llm_sdk_reference"
    SECRET = "secret"
    SYSTEM_PROMPT = "system_prompt"
    TOOL_SCHEMA = "tool_schema"
    RAG_ARTEFACT = "rag_artefact"


class Provider(str, Enum):
    """LLM inference providers detected by lexical signatures."""

    OPENAI = "openai"
    AZURE_OPENAI = "azure_openai"
    ANTHROPIC = "anthropic"
    GOOGLE_GEMINI = "google_gemini"
    AWS_BEDROCK = "aws_bedrock"
    MISTRAL = "mistral"
    COHERE = "cohere"
    HUGGINGFACE = "huggingface"
    REPLICATE = "replicate"
    TOGETHER = "together"
    GROQ = "groq"
    PERPLEXITY = "perplexity"
    DEEPSEEK = "deepseek"
    XAI = "xai"
    OPENROUTER = "openrouter"
    FIREWORKS = "fireworks"
    # Chinese and other regional providers. AndroZoo's frame explicitly
    # includes AppChina, Anzhi and other regional stores
    # (METHODOLOGY.md §2.2), so a Western-only provider list under-counts
    # exactly the apps those markets contribute — a systematic bias, not a
    # random one.
    MOONSHOT = "moonshot"
    ZHIPU = "zhipu"
    ALIBABA = "alibaba"
    BAIDU = "baidu"
    BYTEDANCE = "bytedance"
    MINIMAX = "minimax"
    IFLYTEK = "iflytek"
    ZEROONE = "zeroone"
    OLLAMA = "ollama"
    GENERIC = "generic"


@dataclass
class SourceLocation:
    """Where in the APK a finding was discovered."""

    file_path: str
    line_or_offset: int | None = None
    container: str | None = None  # e.g. "classes.dex", "AndroidManifest.xml"


@dataclass
class Finding:
    """A single security finding from a scanner module."""

    kind: FindingKind
    severity: Severity
    title: str
    description: str
    location: SourceLocation
    evidence: str
    confidence: float  # 0.0 to 1.0
    provider: Provider | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def redacted_evidence(self, max_chars: int = 60) -> str:
        """Truncated evidence for safe display.

        Reports never contain the full secret string, only enough
        to identify the finding pattern.
        """
        evidence = self.evidence
        if len(evidence) <= max_chars:
            return evidence
        head = evidence[: max_chars // 2]
        tail = evidence[-(max_chars // 4) :]
        return f"{head}…{tail}"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["kind"] = self.kind.value
        d["severity"] = self.severity.value
        d["provider"] = self.provider.value if self.provider else None
        d["evidence"] = self.redacted_evidence()
        return d


@dataclass
class ApkInfo:
    """Static metadata about an analysed APK."""

    package_name: str
    version_name: str | None
    version_code: int | None
    min_sdk: int | None
    target_sdk: int | None
    file_path: str
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ScanResult:
    """Top-level result returned by `scan_apk`."""

    apk: ApkInfo
    is_llm_integrated: bool
    detected_providers: list[Provider]
    findings: list[Finding]
    scan_duration_seconds: float
    scanner_version: str

    @property
    def summary(self) -> dict[str, int]:
        """Count of findings by severity."""
        counts: dict[str, int] = {s.value: 0 for s in Severity}
        for f in self.findings:
            counts[f.severity.value] += 1
        return counts

    def to_dict(self) -> dict[str, Any]:
        return {
            "apk": self.apk.to_dict(),
            "is_llm_integrated": self.is_llm_integrated,
            "detected_providers": [p.value for p in self.detected_providers],
            "findings": [f.to_dict() for f in self.findings],
            "scan_duration_seconds": self.scan_duration_seconds,
            "scanner_version": self.scanner_version,
            "summary": self.summary,
        }
