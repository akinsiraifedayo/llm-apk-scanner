"""Tests for the lexical and structural filters."""

from __future__ import annotations

from llm_apk_scanner.filters.lexical import LexicalFilter
from llm_apk_scanner.filters.structural import StructuralFilter
from llm_apk_scanner.models import Provider


def test_lexical_detects_openai_endpoint() -> None:
    f = LexicalFilter()
    matches = f.scan_text("classes.dex", "url=https://api.openai.com/v1/chat")
    assert any(m.provider == Provider.OPENAI for m in matches)


def test_lexical_detects_anthropic_endpoint() -> None:
    f = LexicalFilter()
    matches = f.scan_text("classes.dex", "https://api.anthropic.com/v1/messages")
    assert any(m.provider == Provider.ANTHROPIC for m in matches)


def test_lexical_detects_sdk_package() -> None:
    f = LexicalFilter()
    matches = f.scan_text(
        "classes.dex", "Lcom/google/ai/client/generativeai/Foo;"
    )
    assert any(m.provider == Provider.GOOGLE_GEMINI for m in matches)


def test_single_inference_endpoint_is_sufficient() -> None:
    """One inference host confirms integration on its own.

    This reverses the original two-signal rule, on measured evidence rather
    than taste. In the 125-APK pilot (`validation/FINDINGS.md`) 9 APKs
    carried an inference endpoint, only 2 also carried an SDK, and none
    carried two distinct endpoints — so requiring a quorum recognised 2 of 9
    and recall was 22% against a ≥85% criterion. An endpoint like
    `api.openai.com` is not incidental: it is present because something
    calls it.
    """
    f = LexicalFilter()
    one = f.scan_text("classes.dex", "https://api.openai.com")
    assert f.is_llm_integrated(one)


def test_single_sdk_name_is_still_insufficient() -> None:
    """The quorum still applies to SDK names, which *are* incidental.

    Third-party libraries reference vendor names without integrating
    anything, so the original reasoning holds for this class of evidence.
    """
    f = LexicalFilter()
    one = f.scan_text("classes.dex", "Lcom/aallam/openai/Foo;")
    assert not f.is_llm_integrated(one)


def test_two_distinct_sdk_names_are_sufficient() -> None:
    f = LexicalFilter()
    matches = f.scan_text(
        "classes.dex",
        "Lcom/aallam/openai/Foo; Lcom/google/ai/client/generativeai/Bar;",
    )
    assert f.is_llm_integrated(matches)


def test_endpoint_and_sdk_together_are_sufficient() -> None:
    f = LexicalFilter()
    both = f.scan_text(
        "classes.dex",
        "https://api.openai.com/v1/chat ; com/aallam/openai/Foo",
    )
    assert f.is_llm_integrated(both)


def test_no_signal_is_not_integrated() -> None:
    f = LexicalFilter()
    assert not f.is_llm_integrated(f.scan_text("classes.dex", "hello world"))


def test_structural_invocation_signatures() -> None:
    s = StructuralFilter()
    evidence = s.scan(
        [("classes.dex", "Lcom/openai/Foo->createChatCompletion()")],
        [Provider.OPENAI],
    )
    assert any(e.kind == "invocation_signature" for e in evidence)


def test_structural_json_body_shape() -> None:
    s = StructuralFilter()
    evidence = s.scan(
        [
            (
                "classes.dex",
                '{"messages":[{"role":"system","content":"hi"}],'
                '"model":"gpt-4o"}',
            )
        ],
        [Provider.OPENAI],
    )
    assert any(e.kind == "json_body_shape" for e in evidence)


def test_two_stage_combined() -> None:
    lex = LexicalFilter()
    struct = StructuralFilter()
    sources = [
        (
            "classes.dex",
            'https://api.openai.com/v1/chat com/aallam/openai/Bar '
            'createChatCompletion {"messages":[{"role":"user","content":"x"}]}'
        )
    ]
    lex_matches = lex.scan_corpus(sources)
    providers = lex.detected_providers(lex_matches)
    struct_evidence = struct.scan(sources, providers)
    assert StructuralFilter.is_confirmed(lex_matches, struct_evidence)
