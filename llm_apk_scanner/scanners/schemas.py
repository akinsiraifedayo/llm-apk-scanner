"""
Tool-call schema scanner.

Detects JSON definitions that match the OpenAI / Anthropic
tool-definition shapes embedded in client code. A leaked tool
schema gives an attacker the precise interface for tool abuse and
goal hijacking attacks.

We avoid full JSON parsing because tool definitions are often
multi-line and split across string concatenation in compiled
bytecode. A structural regex pass over the raw string table is more
robust.
"""

from __future__ import annotations

import json
import re
from typing import Any

from llm_apk_scanner.models import (
    Finding,
    FindingKind,
    Severity,
    SourceLocation,
)
from llm_apk_scanner.scanners.base import BaseScanner

# OpenAI Chat Completions tool format and Anthropic Messages tool format
# both share the substring `"name"` and `"description"` and a parameters
# block typed as JSON Schema. We anchor on `"type":"function"` (OpenAI)
# and `"input_schema"` (Anthropic).
OPENAI_FN_PATTERN = re.compile(
    r'"type"\s*:\s*"function"\s*,\s*"function"\s*:\s*\{'
    r'\s*"name"\s*:\s*"[^"]{1,100}"',
    re.DOTALL,
)
OPENAI_FN_SHORT = re.compile(
    r'"type"\s*:\s*"function"',
)
ANTHROPIC_TOOL_PATTERN = re.compile(
    r'"input_schema"\s*:\s*\{\s*"type"\s*:\s*"object"',
    re.DOTALL,
)

# A minimal hand-rolled JSON-object extractor that walks brace depth.
# Used to extract the schema payload around a regex hit so we can try
# to parse it for fidelity.
def _extract_object_around(text: str, start_index: int) -> str | None:
    # Find the nearest enclosing `{`.
    open_idx = text.rfind("{", 0, start_index)
    if open_idx == -1:
        return None
    depth = 0
    for i in range(open_idx, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[open_idx : i + 1]
    return None


def _is_real_tool_schema(obj: dict[str, Any]) -> bool:
    """Validate that a parsed object looks like a real tool definition."""
    # OpenAI shape
    if obj.get("type") == "function" and isinstance(obj.get("function"), dict):
        fn = obj["function"]
        if isinstance(fn.get("name"), str) and "parameters" in fn:
            return True
    # Anthropic shape
    if (
        isinstance(obj.get("name"), str)
        and isinstance(obj.get("input_schema"), dict)
        and obj["input_schema"].get("type") == "object"
    ):
        return True
    return False


class ToolSchemaScanner(BaseScanner):
    """Detects embedded LLM tool/function-call schemas."""

    name = "tool_schemas"

    def scan(self, sources: list[tuple[str, str]]) -> list[Finding]:
        findings: list[Finding] = []
        for source_file, content in sources:
            seen_objects: set[str] = set()
            for pattern in (OPENAI_FN_PATTERN, ANTHROPIC_TOOL_PATTERN):
                for m in pattern.finditer(content):
                    obj_text = _extract_object_around(content, m.start())
                    if obj_text is None or obj_text in seen_objects:
                        continue
                    seen_objects.add(obj_text)
                    confidence = 0.7
                    parsed: dict[str, Any] | None = None
                    try:
                        parsed = json.loads(obj_text)
                    except json.JSONDecodeError:
                        # Many string-concat artefacts will not parse;
                        # the structural regex is still informative.
                        confidence = 0.5
                    if isinstance(parsed, dict) and _is_real_tool_schema(parsed):
                        confidence = 0.95
                    findings.append(
                        Finding(
                            kind=FindingKind.TOOL_SCHEMA,
                            severity=Severity.MEDIUM,
                            title="Embedded tool/function schema",
                            description=(
                                "A JSON object matching OpenAI or "
                                "Anthropic tool-definition shape was "
                                "embedded in the APK. Tool schemas "
                                "expose the agent's capability "
                                "interface and are used for tool-abuse "
                                "and goal-hijacking attacks."
                            ),
                            location=SourceLocation(
                                file_path=source_file,
                                line_or_offset=m.start(),
                            ),
                            evidence=obj_text[:300],
                            confidence=confidence,
                            metadata={
                                "parsed_ok": parsed is not None,
                                "tool_name": (
                                    parsed.get("function", {}).get("name")
                                    if parsed and parsed.get("type") == "function"
                                    else parsed.get("name")
                                    if parsed
                                    else None
                                ),
                            },
                        )
                    )
            # Also catch the OpenAI short shape that didn't match the
            # full pattern, to keep recall high (lower confidence).
            for m in OPENAI_FN_SHORT.finditer(content):
                if any(m.start() == f.location.line_or_offset for f in findings):
                    continue
                findings.append(
                    Finding(
                        kind=FindingKind.TOOL_SCHEMA,
                        severity=Severity.LOW,
                        title="Possible tool schema fragment",
                        description=(
                            'A "type":"function" fragment matching the '
                            "OpenAI tool shape was found, but the "
                            "surrounding object could not be extracted."
                        ),
                        location=SourceLocation(
                            file_path=source_file, line_or_offset=m.start()
                        ),
                        evidence=content[m.start() : m.start() + 80],
                        confidence=0.4,
                    )
                )
        return findings
