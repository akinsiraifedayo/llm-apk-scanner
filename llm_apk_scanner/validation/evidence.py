"""Neutral evidence extraction for blind hand-labelling.

To label an APK's ground truth a coder has to see what is actually inside it.
The previous labeller showed only a file path, which leaves two bad options:
guess, or go and look at the scanner's answer — and a ground truth that agrees
with the system by construction measures nothing.

This module extracts *raw facts*, never verdicts:

* every network host referenced,
* strings matching a deliberately broad AI vocabulary, in context,
* class and asset paths that look like a vendor SDK,
* the manifest basics (package, version, permissions).

The vocabulary here is **wider than the scanner's detection rules on purpose.**
The point of validation is to find what the scanner *missed*, so the coder must
be shown evidence the scanner does not itself act on — an app calling
``api.cohere.ai`` should be visible to the coder whether or not any rule fired.
Narrowing this to the scanner's own signals would make false negatives
structurally invisible and the measured recall meaningless.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

#: Hosts, from URLs or bare domain-like strings. The final label must be
#: alphabetic, which alone rejects the decimals that pervade a DEX string
#: table (``0.0``, ``2.761``, ``1.41409301758``) — without it those dominate
#: the extracted "hosts" and bury the real endpoints.
_HOST_RE = re.compile(
    r"(?:https?://)?((?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z]{2,})", re.IGNORECASE
)

#: Public suffixes accepted as a real host's final label.
#:
#: An allowlist rather than a pattern, because Android class names are
#: reverse-domain and otherwise indistinguishable from hosts:
#: ``androidx.credentials.type`` ends in a plausible-looking label but is a
#: package, not an endpoint. Requiring the *last* label to be a real TLD keeps
#: ``api.openai.com`` and drops ``com.google.android.gms`` — the reversal is
#: exactly what separates them.
_TLDS = frozenset(
    """com org net io ai co dev app cloud me us uk de fr nl eu ru cn jp in br
    xyz tech online site info biz gg sh fm tv cc ly to id pro live one work
    space store site studio digital network systems services solutions""".split()
)

#: Hosts that are plainly not inference endpoints; dropped to keep the
#: worksheet readable. Everything else is shown, including unknowns.
_BORING_HOST_SUFFIXES = (
    "w3.org", "apache.org", "schemas.android.com", "googleapis.com/auth",
    "xml.org", "java.sun.com", "github.io", "bit.ly", "goo.gl",
    "facebook.com", "fbcdn.net", "twitter.com", "instagram.com",
    "play.google.com", "gstatic.com", "googletagmanager.com",
    "google-analytics.com", "crashlytics.com", "unity3d.com",
)

#: Broad AI/LLM vocabulary — wider than the scanner's rules, see module docstring.
_AI_TERMS = (
    "openai", "anthropic", "claude", "gpt", "chatgpt", "gemini", "bard",
    "palm", "llama", "mistral", "cohere", "huggingface", "replicate",
    "perplexity", "deepseek", "groq", "together.ai", "fireworks",
    "stability", "midjourney", "dall-e", "dalle", "diffusion",
    "system prompt", "systemprompt", "you are a", "you are an",
    "chat/completions", "completions", "embeddings", "inference",
    "temperature", "max_tokens", "top_p", "assistant", "prompt",
    "function_call", "tool_call", "tools", "rag", "vector", "pinecone",
    "chroma", "weaviate", "qdrant", "milvus", "faiss", "langchain",
    "llamaindex", "transformers", "onnx", "tflite", "gguf", "ggml",
)

#: Path fragments suggesting a bundled vendor SDK.
_SDK_HINTS = (
    "openai", "anthropic", "cohere", "huggingface", "langchain", "llamaindex",
    "generativeai", "vertexai", "mlkit", "tensorflow", "pytorch", "onnxruntime",
)

_MAX_PER_SECTION = 40
_CONTEXT = 60


@dataclass
class Evidence:
    """Raw, verdict-free facts about one APK."""

    sha256: str = ""
    package: str = ""
    version: str = ""
    permissions: list[str] = field(default_factory=list)
    hosts: list[tuple[str, int]] = field(default_factory=list)
    ai_strings: list[tuple[str, str]] = field(default_factory=list)
    sdk_paths: list[str] = field(default_factory=list)
    error: str = ""

    @property
    def is_empty(self) -> bool:
        return not (self.hosts or self.ai_strings or self.sdk_paths)


def _interesting_host(host: str) -> bool:
    """Whether a regex match is a real host worth showing the coder."""
    host = host.lower()
    if host.rsplit(".", 1)[-1] not in _TLDS:
        return False
    if any(host.endswith(suffix) for suffix in _BORING_HOST_SUFFIXES):
        return False
    # Reverse-domain package roots: `com.google.android...`, `io.reactivex...`.
    return host.split(".", 1)[0] not in {"com", "org", "net", "io", "co", "androidx"}


def extract_evidence(apk_path: str | Path) -> Evidence:
    """Pull neutral evidence from one APK. Never raises."""
    from llm_apk_scanner.apk_loader import load_apk

    path = Path(apk_path)
    evidence = Evidence(sha256=path.stem.upper())
    try:
        apk = load_apk(str(path))
    except Exception as exc:  # noqa: BLE001 - a bad APK still needs a worksheet row
        evidence.error = f"{type(exc).__name__}: {exc}"
        return evidence

    evidence.package = apk.info.package_name or ""
    evidence.version = apk.info.version_name or ""

    hosts: Counter[str] = Counter()
    ai_hits: list[tuple[str, str]] = []
    sdk_paths: set[str] = set()

    for source_name, content in apk.sources:
        lowered = content.lower()

        for match in _HOST_RE.finditer(content):
            host = match.group(1).lower()
            if _interesting_host(host):
                hosts[host] += 1

        for hint in _SDK_HINTS:
            if hint in lowered:
                for line in content.splitlines():
                    low = line.lower()
                    if hint in low and ("/" in line or "." in line) and len(line) < 200:
                        sdk_paths.add(line.strip())
                        if len(sdk_paths) > _MAX_PER_SECTION:
                            break

        if len(ai_hits) < _MAX_PER_SECTION:
            for term in _AI_TERMS:
                index = lowered.find(term)
                if index == -1:
                    continue
                start = max(0, index - _CONTEXT)
                end = min(len(content), index + len(term) + _CONTEXT)
                snippet = content[start:end].replace("\n", " ").strip()
                ai_hits.append((source_name, snippet))
                if len(ai_hits) >= _MAX_PER_SECTION:
                    break

    evidence.hosts = hosts.most_common(_MAX_PER_SECTION)
    evidence.ai_strings = ai_hits
    evidence.sdk_paths = sorted(sdk_paths)[:_MAX_PER_SECTION]
    return evidence


def render_evidence(evidence: Evidence) -> str:
    """Format evidence as a plain-text worksheet entry for the coder."""
    lines: list[str] = []
    lines.append("=" * 78)
    lines.append(f"PACKAGE : {evidence.package or '(unknown)'}")
    lines.append(f"VERSION : {evidence.version or '(unknown)'}")
    lines.append(f"SHA-256 : {evidence.sha256[:32]}…")
    if evidence.error:
        lines.append(f"ERROR   : {evidence.error}")
        return "\n".join(lines)

    lines.append("")
    lines.append(f"-- Network hosts referenced ({len(evidence.hosts)}) " + "-" * 30)
    if evidence.hosts:
        for host, count in evidence.hosts:
            lines.append(f"   {count:5}x  {host}")
    else:
        lines.append("   (none found)")

    lines.append("")
    lines.append(f"-- SDK-like paths ({len(evidence.sdk_paths)}) " + "-" * 38)
    if evidence.sdk_paths:
        for item in evidence.sdk_paths[:15]:
            lines.append(f"   {item[:88]}")
    else:
        lines.append("   (none found)")

    lines.append("")
    lines.append(f"-- AI-vocabulary strings in context ({len(evidence.ai_strings)}) " + "-" * 16)
    if evidence.ai_strings:
        for source, snippet in evidence.ai_strings[:20]:
            lines.append(f"   [{source[:22]}] {snippet[:100]}")
    else:
        lines.append("   (none found)")

    return "\n".join(lines)
