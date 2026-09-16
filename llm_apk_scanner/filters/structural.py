"""
Stage-2 structural filter for LLM-integrated apps.

Where the lexical filter looks for surface tokens, the structural
filter validates that those tokens correspond to actual LLM-client
behaviour: invocation of HTTP requests against the endpoint, presence
of LLM-shaped JSON request bodies, declared TLS pinning targeting the
provider domain, etc.

This is the second stage of the proposal §7.2 pipeline. Its purpose
is to reduce the false-positive rate of the lexical filter, e.g. apps
that merely *mention* "openai" in copy-pasted licence text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from llm_apk_scanner.filters.lexical import LexicalMatch
from llm_apk_scanner.models import Provider

# Method-signature shapes that imply runtime LLM invocation. These are
# string patterns one expects to see in classes.dex when an SDK is
# actually exercised, not just imported.
LLM_INVOCATION_HINTS: dict[Provider, list[re.Pattern[str]]] = {
    Provider.OPENAI: [
        re.compile(r"createChatCompletion|chat/completions|ChatCompletionRequest"),
        re.compile(r"createCompletion|/completions"),
        re.compile(r"createEmbedding|/embeddings"),
        re.compile(r"createResponse|/responses"),
    ],
    Provider.ANTHROPIC: [
        re.compile(r"/v1/messages|MessagesRequest|MessageRequestBuilder"),
    ],
    Provider.GOOGLE_GEMINI: [
        re.compile(r":generateContent|GenerativeModel|GenerativeAiClient"),
    ],
    Provider.AWS_BEDROCK: [
        re.compile(r"InvokeModel|invokeModel|BedrockRuntimeClient"),
    ],
    Provider.HUGGINGFACE: [re.compile(r"InferenceApi|/pipeline/")],
    Provider.GENERIC: [
        re.compile(r"ChatPromptTemplate|RetrievalQA|VectorStore|LLMChain"),
    ],
}


# JSON-body shape hints. We look for substrings characteristic of
# real LLM request payloads.
JSON_BODY_HINTS: list[re.Pattern[str]] = [
    re.compile(r'"messages"\s*:\s*\[\s*\{\s*"role"\s*:\s*"(system|user|assistant)"'),
    re.compile(r'"role"\s*:\s*"(system|user|assistant|tool)"\s*,\s*"content"'),
    re.compile(r'"model"\s*:\s*"(gpt-|claude-|gemini-|mistral-|command-|llama-)'),
    re.compile(r'"max_tokens"|"temperature"|"top_p"|"top_k"'),
    re.compile(r'"tools"\s*:\s*\[\s*\{\s*"type"\s*:\s*"function"'),
]


@dataclass
class StructuralEvidence:
    """A single piece of structural evidence supporting LLM-integration."""

    kind: str  # "invocation_signature" | "json_body_shape" | "tls_pinning"
    detail: str
    source_file: str


class StructuralFilter:
    """Validates lexical matches with deeper structural checks."""

    def __init__(
        self,
        invocation_hints: dict[Provider, list[re.Pattern[str]]] | None = None,
        json_hints: list[re.Pattern[str]] | None = None,
    ) -> None:
        self.invocation_hints = invocation_hints or LLM_INVOCATION_HINTS
        self.json_hints = json_hints or JSON_BODY_HINTS

    def scan(
        self,
        sources: list[tuple[str, str]],
        candidate_providers: list[Provider],
    ) -> list[StructuralEvidence]:
        """
        Look for invocation-shape and JSON-body evidence.

        Args:
            sources: list of (source_file, content) blobs.
            candidate_providers: providers flagged by the lexical filter.
        """
        evidence: list[StructuralEvidence] = []

        for source_file, content in sources:
            for provider in candidate_providers:
                for pattern in self.invocation_hints.get(provider, []):
                    if pattern.search(content):
                        evidence.append(
                            StructuralEvidence(
                                kind="invocation_signature",
                                detail=f"{provider.value}: {pattern.pattern}",
                                source_file=source_file,
                            )
                        )
            for pattern in self.json_hints:
                if pattern.search(content):
                    evidence.append(
                        StructuralEvidence(
                            kind="json_body_shape",
                            detail=pattern.pattern,
                            source_file=source_file,
                        )
                    )

        return evidence

    @staticmethod
    def is_confirmed(
        lexical_matches: list[LexicalMatch],
        structural_evidence: list[StructuralEvidence],
    ) -> bool:
        """
        Final two-stage decision.

        Confirmed when either:

        * an **inference-endpoint** hostname is present — conclusive on its
          own, because a host like ``api.anthropic.com`` does not appear in
          an APK unless something calls it; or
        * a weaker lexical signal (an SDK namespace) is *corroborated* by
          structural evidence — an invocation signature or a request-body
          shape.

        The original rule demanded structural corroboration for everything.
        Validation on the pilot corpus (`validation/FINDINGS.md`) showed the
        cost: one pilot application carries ``api.anthropic.com``
        in both its DEX strings and its ``network_security_config.xml``,
        plus a live Anthropic API key, yet was rejected because the
        structural stage found nothing to corroborate. Requiring a second
        opinion on conclusive evidence discards true positives while
        protecting against a risk — incidental vendor-name mentions — that
        only applies to the weaker signal class.
        """
        if any(m.is_endpoint for m in lexical_matches):
            return True
        return bool(lexical_matches) and bool(structural_evidence)
