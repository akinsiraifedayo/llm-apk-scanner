"""
Secret scanner for LLM-provider API keys and bearer tokens.

Detection combines:
 1. Provider-specific structural regexes (high precision).
 2. Generic high-entropy-string heuristics (high recall, lower precision).

Validity is *never* confirmed by exercising the credential. If a key
is identified, the validators module has a separate, explicit pathway
for non-revocable introspection (e.g. OpenAI's `list models` endpoint
which is rate-limited and idempotent).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from llm_apk_scanner.models import (
    Finding,
    FindingKind,
    Provider,
    Severity,
    SourceLocation,
)
from llm_apk_scanner.scanners.base import BaseScanner


@dataclass(frozen=True)
class SecretRule:
    name: str
    provider: Provider
    pattern: re.Pattern[str]
    severity: Severity
    base_confidence: float


# Rules listed by *specificity* — narrower patterns first.
# Reference: published key-format schemas as of April 2026.
SECRET_RULES: list[SecretRule] = [
    SecretRule(
        name="OpenAI project key",
        provider=Provider.OPENAI,
        pattern=re.compile(r"\bsk-proj-[A-Za-z0-9_-]{32,}\b"),
        severity=Severity.CRITICAL,
        base_confidence=0.99,
    ),
    SecretRule(
        name="OpenAI service account key",
        provider=Provider.OPENAI,
        pattern=re.compile(r"\bsk-svcacct-[A-Za-z0-9_-]{32,}\b"),
        severity=Severity.CRITICAL,
        base_confidence=0.99,
    ),
    SecretRule(
        name="OpenAI service key (T3BlbkFJ)",
        provider=Provider.OPENAI,
        pattern=re.compile(r"\bsk-[A-Za-z0-9]{20}T3BlbkFJ[A-Za-z0-9]{20}\b"),
        severity=Severity.CRITICAL,
        base_confidence=0.99,
    ),
    SecretRule(
        name="OpenAI legacy user key",
        provider=Provider.OPENAI,
        pattern=re.compile(r"\bsk-[A-Za-z0-9]{48}\b"),
        severity=Severity.CRITICAL,
        base_confidence=0.85,
    ),
    SecretRule(
        name="Anthropic API key",
        provider=Provider.ANTHROPIC,
        pattern=re.compile(r"\bsk-ant-api\d{2}-[A-Za-z0-9_-]{80,}\b"),
        severity=Severity.CRITICAL,
        base_confidence=0.99,
    ),
    SecretRule(
        name="Anthropic session key",
        provider=Provider.ANTHROPIC,
        pattern=re.compile(r"\bsk-ant-sid\d{2}-[A-Za-z0-9_-]{80,}\b"),
        severity=Severity.CRITICAL,
        base_confidence=0.99,
    ),
    SecretRule(
        name="Cohere production key",
        provider=Provider.COHERE,
        pattern=re.compile(r"\bco1_[A-Za-z0-9]{32,}\b"),
        severity=Severity.HIGH,
        base_confidence=0.95,
    ),
    SecretRule(
        name="Perplexity API key",
        provider=Provider.PERPLEXITY,
        pattern=re.compile(r"\bpplx-[A-Za-z0-9]{48,}\b"),
        severity=Severity.HIGH,
        base_confidence=0.95,
    ),
    SecretRule(
        name="OpenRouter API key",
        provider=Provider.OPENROUTER,
        pattern=re.compile(r"\bsk-or-v1-[A-Za-z0-9]{64}\b"),
        severity=Severity.HIGH,
        base_confidence=0.99,
    ),
    SecretRule(
        name="Fireworks AI key",
        provider=Provider.FIREWORKS,
        pattern=re.compile(r"\bfw_[A-Za-z0-9]{32,}\b"),
        severity=Severity.HIGH,
        base_confidence=0.95,
    ),
    SecretRule(
        name="DeepSeek API key",
        provider=Provider.DEEPSEEK,
        pattern=re.compile(r"\bsk-[A-Za-z0-9]{48,}\b"),
        severity=Severity.HIGH,
        base_confidence=0.80,
    ),
    # `AIza…` is the *generic* Google API key format, shared by Maps,
    # Firebase, YouTube, Places, Drive and Gemini alike — the string itself
    # carries no indication of which service it authorises. Most instances in
    # Android apps are Firebase or Maps keys, which Google's own
    # documentation requires to be embedded in the client and which are
    # restricted by package name and signing certificate; those are not
    # credential exposure in any meaningful sense.
    #
    # It is therefore attributed to GENERIC, not GOOGLE_GEMINI, and
    # `scan.py` promotes it back to GOOGLE_GEMINI/HIGH only when the same
    # APK also references a Gemini or Vertex inference endpoint. Attributing
    # every AIza key to an inference provider inflated the pilot's
    # "credential exposure" rate by roughly an order of magnitude — 48 of
    # 125 apps carried a secret finding, but only 3 carried an actual
    # inference-provider key.
    SecretRule(
        name="Google API key (service unknown)",
        provider=Provider.GENERIC,
        pattern=re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b"),
        severity=Severity.MEDIUM,
        base_confidence=0.85,
    ),
    SecretRule(
        name="AWS access key id",
        provider=Provider.AWS_BEDROCK,
        pattern=re.compile(r"\b(AKIA|ASIA)[0-9A-Z]{16}\b"),
        severity=Severity.HIGH,
        base_confidence=0.95,
    ),
    SecretRule(
        name="Hugging Face token",
        provider=Provider.HUGGINGFACE,
        pattern=re.compile(r"\bhf_[A-Za-z0-9]{34,}\b"),
        severity=Severity.HIGH,
        base_confidence=0.95,
    ),
    SecretRule(
        name="Replicate API token",
        provider=Provider.REPLICATE,
        pattern=re.compile(r"\br8_[A-Za-z0-9]{37,}\b"),
        severity=Severity.HIGH,
        base_confidence=0.95,
    ),
    SecretRule(
        name="Groq API key",
        provider=Provider.GROQ,
        pattern=re.compile(r"\bgsk_[A-Za-z0-9]{40,}\b"),
        severity=Severity.HIGH,
        base_confidence=0.95,
    ),
    SecretRule(
        name="xAI API key",
        provider=Provider.XAI,
        pattern=re.compile(r"\bxai-[A-Za-z0-9]{40,}\b"),
        severity=Severity.HIGH,
        base_confidence=0.95,
    ),
    SecretRule(
        name="Bearer token literal",
        provider=Provider.GENERIC,
        # Token strings of >=32 chars stamped directly into a
        # `Bearer <...>` Authorization header in code.
        pattern=re.compile(
            r"\bBearer\s+([A-Za-z0-9_\-\.=]{32,})\b",
        ),
        severity=Severity.MEDIUM,
        base_confidence=0.55,
    ),
    # Non-LLM tokens that are commonly leaked alongside LLM integrations.
    # These are bonus coverage, not core to the research questions.
    SecretRule(
        name="Slack bot token",
        provider=Provider.GENERIC,
        pattern=re.compile(r"\bxoxb-[0-9]+-[0-9A-Za-z]+-[A-Za-z0-9]+\b"),
        severity=Severity.HIGH,
        base_confidence=0.95,
    ),
    SecretRule(
        name="Slack user token",
        provider=Provider.GENERIC,
        pattern=re.compile(r"\bxoxp-[0-9]+-[0-9]+-[0-9]+-[A-Za-z0-9]+\b"),
        severity=Severity.HIGH,
        base_confidence=0.95,
    ),
    SecretRule(
        name="Slack app token",
        provider=Provider.GENERIC,
        pattern=re.compile(r"\bxapp-[0-9]+-[A-Za-z0-9]+-[0-9]+-[A-Za-z0-9]+\b"),
        severity=Severity.HIGH,
        base_confidence=0.95,
    ),
    SecretRule(
        name="GitHub token",
        provider=Provider.GENERIC,
        pattern=re.compile(r"\b(ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{36,}\b"),
        severity=Severity.HIGH,
        base_confidence=0.95,
    ),
    SecretRule(
        name="Firebase/Google service account key",
        provider=Provider.GOOGLE_GEMINI,
        pattern=re.compile(r"\b[0-9]+-[a-z0-9]+@[a-z\-]+\.iam\.gserviceaccount\.com\b"),
        severity=Severity.HIGH,
        base_confidence=0.90,
    ),
]


# Strings with these substrings are almost always test fixtures or
# placeholders — used to demote confidence.
PLACEHOLDER_SUBSTRINGS = (
    "your-api-key",
    "YOUR_API_KEY",
    "<your_key>",
    "INSERT_KEY_HERE",
    "REPLACE_ME",
    "xxxxxxxx",
    "0000000000",
    "example-key",
    "test_key_",
)


def shannon_entropy(text: str) -> float:
    """Standard Shannon entropy of a string, base 2."""
    if not text:
        return 0.0
    freq: dict[str, int] = {}
    for ch in text:
        freq[ch] = freq.get(ch, 0) + 1
    n = len(text)
    return -sum((c / n) * math.log2(c / n) for c in freq.values())


class SecretScanner(BaseScanner):
    """
    Detect leaked LLM-provider credentials in APK strings.

    The scanner produces one Finding per (rule, location) match. The
    confidence score is adjusted downward for placeholders and upward
    for high-entropy matches.
    """

    name = "secrets"

    # Minimum entropy at which a generic match is considered worth keeping
    # without a specific structural rule. Empirical threshold; tuned
    # against the seeded fixture corpus.
    GENERIC_ENTROPY_FLOOR = 4.5

    def __init__(self, rules: list[SecretRule] | None = None) -> None:
        self.rules = rules or SECRET_RULES

    def scan(self, sources: list[tuple[str, str]]) -> list[Finding]:
        findings: list[Finding] = []
        for source_file, content in sources:
            findings.extend(self._scan_one(source_file, content))
        return findings

    def _scan_one(self, source_file: str, content: str) -> list[Finding]:
        findings: list[Finding] = []
        for rule in self.rules:
            for m in rule.pattern.finditer(content):
                evidence = m.group(0)
                confidence = self._confidence_for(rule, evidence)
                if confidence < 0.3:
                    continue
                findings.append(
                    Finding(
                        kind=FindingKind.SECRET,
                        severity=rule.severity,
                        title=f"Possible {rule.name}",
                        description=(
                            f"Static analysis matched the structural "
                            f"pattern for a {rule.name}. The provider "
                            f"is {rule.provider.value}. Validity has not "
                            "been exercised; confirm via non-revocable "
                            "introspection only."
                        ),
                        location=SourceLocation(
                            file_path=source_file,
                            line_or_offset=m.start(),
                        ),
                        evidence=evidence,
                        confidence=confidence,
                        provider=rule.provider,
                        metadata={
                            "rule": rule.name,
                            "entropy": round(shannon_entropy(evidence), 3),
                        },
                    )
                )
        return findings

    def _confidence_for(self, rule: SecretRule, evidence: str) -> float:
        confidence = rule.base_confidence
        lowered = evidence.lower()
        if any(p.lower() in lowered for p in PLACEHOLDER_SUBSTRINGS):
            confidence *= 0.1
        # Reward high-entropy evidence beyond what the regex alone proves.
        entropy = shannon_entropy(evidence)
        if entropy >= self.GENERIC_ENTROPY_FLOOR:
            confidence = min(1.0, confidence + 0.05)
        else:
            confidence *= 0.7
        return round(confidence, 3)
