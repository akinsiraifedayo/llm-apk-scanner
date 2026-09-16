"""Tests for the tool-schema scanner."""

from __future__ import annotations

from llm_apk_scanner.scanners.schemas import ToolSchemaScanner


def test_detects_openai_function_schema(positive_tool_schemas) -> None:
    scanner = ToolSchemaScanner()
    findings = scanner.scan([("test", positive_tool_schemas[0])])
    assert findings, "Should detect OpenAI function schema"
    assert findings[0].confidence >= 0.7


def test_detects_anthropic_input_schema(positive_tool_schemas) -> None:
    scanner = ToolSchemaScanner()
    findings = scanner.scan([("test", positive_tool_schemas[1])])
    assert findings, "Should detect Anthropic input_schema"


def test_rejects_random_json() -> None:
    scanner = ToolSchemaScanner()
    findings = scanner.scan(
        [("test", '{"unrelated": true, "data": [1, 2, 3]}')]
    )
    assert not findings


def test_high_confidence_when_parseable() -> None:
    scanner = ToolSchemaScanner()
    schema = (
        '{"type":"function","function":{"name":"do_thing",'
        '"description":"Does a thing.","parameters":{"type":"object",'
        '"properties":{"x":{"type":"string"}},"required":["x"]}}}'
    )
    findings = scanner.scan([("test", schema)])
    assert findings
    assert findings[0].confidence >= 0.9
    assert findings[0].metadata["tool_name"] == "do_thing"
