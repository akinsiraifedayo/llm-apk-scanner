"""
Stage-1 lexical filter for LLM-integrated apps.

Searches the decoded APK text corpus (manifest, string tables, classes.dex
strings, native-library strings) for known LLM provider endpoints and
SDK package prefixes.

The filter intentionally has *high recall, lower precision*; the
structural filter narrows down false positives in stage 2.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

from llm_apk_scanner.models import Provider

# Provider endpoint patterns. Order matters only for tie-breaking;
# every pattern is tested against every input string.
LLM_ENDPOINT_PATTERNS: dict[Provider, list[re.Pattern[str]]] = {
    Provider.OPENAI: [
        re.compile(r"\bapi\.openai\.com\b", re.IGNORECASE),
        re.compile(r"\bapi\.openai\.azure\.com\b", re.IGNORECASE),
    ],
    Provider.AZURE_OPENAI: [
        re.compile(r"\b[a-z0-9-]+\.openai\.azure\.com\b", re.IGNORECASE),
    ],
    Provider.ANTHROPIC: [
        re.compile(r"\bapi\.anthropic\.com\b", re.IGNORECASE),
        re.compile(r"\bclaude\.ai/api\b", re.IGNORECASE),
    ],
    Provider.GOOGLE_GEMINI: [
        re.compile(r"\bgenerativelanguage\.googleapis\.com\b", re.IGNORECASE),
        re.compile(r"\baiplatform\.googleapis\.com\b", re.IGNORECASE),
    ],
    Provider.AWS_BEDROCK: [
        re.compile(r"\bbedrock-runtime\.[a-z0-9-]+\.amazonaws\.com\b", re.IGNORECASE),
        re.compile(r"\bbedrock\.[a-z0-9-]+\.amazonaws\.com\b", re.IGNORECASE),
    ],
    Provider.MISTRAL: [re.compile(r"\bapi\.mistral\.ai\b", re.IGNORECASE)],
    Provider.COHERE: [re.compile(r"\bapi\.cohere\.(?:com|ai)\b", re.IGNORECASE)],
    Provider.HUGGINGFACE: [
        re.compile(r"\bapi-inference\.huggingface\.co\b", re.IGNORECASE),
        re.compile(r"\bhuggingface\.co/api\b", re.IGNORECASE),
    ],
    Provider.REPLICATE: [re.compile(r"\bapi\.replicate\.com\b", re.IGNORECASE)],
    Provider.TOGETHER: [re.compile(r"\bapi\.together\.(?:xyz|ai)\b", re.IGNORECASE)],
    Provider.GROQ: [re.compile(r"\bapi\.groq\.com\b", re.IGNORECASE)],
    Provider.PERPLEXITY: [re.compile(r"\bapi\.perplexity\.ai\b", re.IGNORECASE)],
    Provider.DEEPSEEK: [re.compile(r"\bapi\.deepseek\.com\b", re.IGNORECASE)],
    Provider.XAI: [
        re.compile(r"\bapi\.x\.ai\b", re.IGNORECASE),
        # The Grok app reaches its models via versioned hosts and an asset
        # CDN rather than api.x.ai; validation found `grok-v2.x.ai` in
        # ai.x.grok, which the api.x.ai pattern alone did not catch.
        re.compile(r"\bgrok(?:-v?\d+)?\.x\.ai\b", re.IGNORECASE),
        re.compile(r"\bapi\.grok\.com\b", re.IGNORECASE),
    ],
    # Declared in the Provider enum but previously undetectable: no pattern
    # existed, so these providers could never be reported.
    Provider.OPENROUTER: [re.compile(r"\bopenrouter\.ai\b", re.IGNORECASE)],
    Provider.FIREWORKS: [re.compile(r"\bapi\.fireworks\.ai\b", re.IGNORECASE)],
    # Regional providers. See the Provider enum for why these matter here.
    Provider.MOONSHOT: [re.compile(r"\bapi\.moonshot\.(?:cn|ai)\b", re.IGNORECASE)],
    Provider.ZHIPU: [
        re.compile(r"\bopen\.bigmodel\.cn\b", re.IGNORECASE),
        re.compile(r"\bapi\.z\.ai\b", re.IGNORECASE),
    ],
    Provider.ALIBABA: [
        re.compile(r"\bdashscope(?:-intl)?\.aliyuncs\.com\b", re.IGNORECASE),
    ],
    Provider.BAIDU: [re.compile(r"\baip\.baidubce\.com\b", re.IGNORECASE)],
    Provider.BYTEDANCE: [
        re.compile(r"\bark\.[a-z0-9-]+\.volces\.com\b", re.IGNORECASE),
    ],
    Provider.MINIMAX: [re.compile(r"\bapi\.minimax(?:i)?\.(?:chat|com)\b", re.IGNORECASE)],
    Provider.IFLYTEK: [re.compile(r"\bspark-api(?:-open)?\.xf-yun\.com\b", re.IGNORECASE)],
    Provider.ZEROONE: [re.compile(r"\bapi\.lingyiwanwu\.com\b", re.IGNORECASE)],
    Provider.OLLAMA: [
        # On-device Ollama is a different threat surface but we capture it.
        re.compile(r"\b(?:127\.0\.0\.1|localhost):11434\b"),
        re.compile(r"\b/api/generate\b"),
    ],
}


# Java/Kotlin SDK package prefixes that imply LLM client usage.
# These are namespaces, not endpoints.
LLM_SDK_PACKAGES: dict[Provider, list[str]] = {
    Provider.OPENAI: [
        "com/openai",
        "com/aallam/openai",  # Kotlin OpenAI SDK
        "com/theokanning/openai",
    ],
    Provider.ANTHROPIC: [
        "com/anthropic",
    ],
    Provider.GOOGLE_GEMINI: [
        "com/google/ai/client/generativeai",
        "com/google/cloud/vertexai",
    ],
    Provider.AWS_BEDROCK: [
        "software/amazon/awssdk/services/bedrock",
        "software/amazon/awssdk/services/bedrockruntime",
    ],
    Provider.MISTRAL: ["com/mistralai"],
    Provider.COHERE: ["co/cohere", "com/cohere"],
    Provider.HUGGINGFACE: ["co/huggingface"],
    Provider.GENERIC: [
        "ai/langchain",
        "io/langchain",
        "com/llamaindex",
    ],
}


@dataclass
class LexicalMatch:
    """A single lexical hit during filtering."""

    provider: Provider
    pattern: str
    matched_text: str
    source_file: str
    is_endpoint: bool  # True for endpoint, False for SDK package

    def to_dict(self) -> dict[str, str | bool]:
        return {
            "provider": self.provider.value,
            "pattern": self.pattern,
            "matched_text": self.matched_text,
            "source_file": self.source_file,
            "is_endpoint": self.is_endpoint,
        }


class LexicalFilter:
    """
    Stage-1 LLM-app identification.

    The filter consumes an iterable of (source_file, content) tuples
    and emits matches. It does not itself decide whether the APK is
    LLM-integrated; the caller aggregates matches and applies a
    threshold.
    """

    def __init__(
        self,
        endpoint_patterns: dict[Provider, list[re.Pattern[str]]] | None = None,
        sdk_packages: dict[Provider, list[str]] | None = None,
    ) -> None:
        self.endpoint_patterns = endpoint_patterns or LLM_ENDPOINT_PATTERNS
        self.sdk_packages = sdk_packages or LLM_SDK_PACKAGES

    def scan_text(self, source_file: str, content: str) -> list[LexicalMatch]:
        """Find all lexical matches in a single text blob."""
        matches: list[LexicalMatch] = []

        for provider, patterns in self.endpoint_patterns.items():
            for pattern in patterns:
                for m in pattern.finditer(content):
                    matches.append(
                        LexicalMatch(
                            provider=provider,
                            pattern=pattern.pattern,
                            matched_text=m.group(0),
                            source_file=source_file,
                            is_endpoint=True,
                        )
                    )

        for provider, packages in self.sdk_packages.items():
            for package in packages:
                # SDK packages appear as path-style strings in classes.dex.
                # Case-sensitive, but the match must end on a package
                # boundary: a bare substring test reports `com/anthropic`
                # inside `Lcom/anthropics/zyler_swipe/MainActivity;`, which
                # is an unrelated app's own namespace (Anthropics Technology,
                # a photo-editing vendor). Validation caught exactly that
                # false positive.
                idx = 0
                while True:
                    idx = content.find(package, idx)
                    if idx == -1:
                        break
                    end = idx + len(package)
                    following = content[end] if end < len(content) else ""
                    if not following or not (following.isalnum() or following == "_"):
                        matches.append(
                            LexicalMatch(
                                provider=provider,
                                pattern=package,
                                matched_text=package,
                                source_file=source_file,
                                is_endpoint=False,
                            )
                        )
                    idx = end

        return matches

    def scan_corpus(
        self, sources: Iterable[tuple[str, str]]
    ) -> list[LexicalMatch]:
        """Scan an iterable of (source_file, content) tuples."""
        all_matches: list[LexicalMatch] = []
        for source_file, content in sources:
            all_matches.extend(self.scan_text(source_file, content))
        return all_matches

    @staticmethod
    def is_llm_integrated(
        matches: list[LexicalMatch],
        min_unique_signals: int = 2,
    ) -> bool:
        """
        Lexical-only integration test. **Not the pipeline's decision.**

        .. warning::
           ``scan.py`` decides integration with
           :meth:`StructuralFilter.is_confirmed`, not this method. The two
           rules disagreed for years and nothing reconciled them: on the
           125-APK pilot this method recognised 2 apps and ``is_confirmed``
           recognised 8, and the figures published in ``PROGRESS.md`` came
           from this one via the now-retired ``scripts/legacy/batch_scan.py``.

           Retained because it is the stage-1 rule described in
           METHODOLOGY.md §3.1 and is useful for measuring the lexical stage
           in isolation. Anything reporting corpus prevalence must call
           ``is_confirmed`` instead.

        Decide whether matches indicate genuine LLM integration.

        A single *SDK-name* match is weak evidence: many apps ship
        third-party libraries that reference vendor names incidentally,
        so those still require `min_unique_signals` distinct hits.

        A single *inference-endpoint* match is not weak. A host like
        `api.openai.com` does not appear in an APK by accident — it is
        there because something calls it. Requiring a second signal
        alongside it discards conclusive evidence.

        Validation on the 125-APK pilot (`validation/FINDINGS.md`) measured
        the cost of the stricter rule: 9 APKs carried an inference
        endpoint, only 2 also carried an SDK, and none carried two
        distinct endpoints — so the old quorum recognised 2 of 9 and
        recall was 22%. Accepting a lone endpoint recovers all 9 with no
        loss of precision, because no APK in that corpus referenced an
        inference endpoint without genuinely integrating an LLM.
        """
        if not matches:
            return False

        endpoint_hits = {m.matched_text.lower() for m in matches if m.is_endpoint}
        sdk_hits = {m.matched_text for m in matches if not m.is_endpoint}

        if endpoint_hits:
            return True
        return len(sdk_hits) >= min_unique_signals

    @staticmethod
    def detected_providers(matches: list[LexicalMatch]) -> list[Provider]:
        """Unique providers across matches, in deterministic order."""
        seen: dict[Provider, None] = {}
        for m in matches:
            seen[m.provider] = None
        return list(seen.keys())
