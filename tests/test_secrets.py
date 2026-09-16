"""Tests for the secret scanner."""

from __future__ import annotations

import pytest

from llm_apk_scanner.models import Provider, Severity
from llm_apk_scanner.scan import _attribute_google_keys
from llm_apk_scanner.scanners.secrets import (
    SecretScanner,
    shannon_entropy,
)


def test_shannon_entropy_empty() -> None:
    assert shannon_entropy("") == 0.0


def test_shannon_entropy_uniform() -> None:
    # All same char => zero entropy.
    assert shannon_entropy("aaaaa") == pytest.approx(0.0)


def test_shannon_entropy_diverse() -> None:
    # Diverse string => high entropy.
    assert shannon_entropy("aB3dEf6gHi9jKl2m") > 3.0


def test_detects_all_positive_secrets(positive_secrets) -> None:
    scanner = SecretScanner()
    detected_labels = []
    for label, value in positive_secrets:
        findings = scanner.scan([(f"test/{label}", f"key = \"{value}\";")])
        if findings:
            detected_labels.append(label)
    recall = len(detected_labels) / len(positive_secrets)
    # Proposal §9 success criterion: >=85% recall on seeded ground truth.
    assert recall >= 0.85, f"Recall {recall:.2%} below threshold; "\
        f"missed {set(l for l, _ in positive_secrets) - set(detected_labels)}"


def test_rejects_negative_secrets(negative_secrets) -> None:
    scanner = SecretScanner()
    high_confidence_false_positives: list[tuple[str, float]] = []
    for label, value in negative_secrets:
        findings = scanner.scan([(f"test/{label}", value)])
        for f in findings:
            if f.confidence >= 0.8:
                high_confidence_false_positives.append(
                    (label, f.confidence)
                )
    assert not high_confidence_false_positives, (
        f"High-confidence FPs: {high_confidence_false_positives}"
    )


def test_placeholder_demotes_confidence() -> None:
    scanner = SecretScanner()
    real = scanner.scan([("test", "sk-proj-" + "A" * 64)])
    placeholder = scanner.scan(
        [("test", "sk-proj-YOUR_API_KEY_HERE_PLEASE_REPLACE_aaaaaaaaaaa")]
    )
    assert real, "expected detection of real-shaped key"
    if placeholder:
        assert placeholder[0].confidence < real[0].confidence


def test_finding_evidence_is_redacted() -> None:
    scanner = SecretScanner()
    findings = scanner.scan(
        [("test", "sk-proj-aB3dEf6gHi9jKl2mNo5pQr8sTu1vWx4yZa7bCd0eFg3hIj6kLm9nOp")]
    )
    assert findings
    redacted = findings[0].redacted_evidence(20)
    # Redacted form must not equal the full evidence.
    assert "…" in redacted or len(redacted) <= 20


def test_metadata_includes_entropy() -> None:
    scanner = SecretScanner()
    findings = scanner.scan(
        [("test", "sk-proj-aB3dEf6gHi9jKl2mNo5pQr8sTu1vWx4yZa7bCd0eFg3hIj")]
    )
    assert findings
    assert "entropy" in findings[0].metadata
    assert findings[0].metadata["entropy"] > 3.0


class TestGoogleKeyAttribution:
    """`AIza…` says which *company*, never which *service*.

    Maps, Firebase, YouTube, Places and Gemini all share the format.
    Firebase keys are required by Google's own documentation to ship in the
    client and are restricted by package name and signing certificate, so
    treating every match as inference-credential exposure inflated the
    pilot's rate by roughly an order of magnitude: 48 of 125 apps carried a
    secret finding, but only 3 carried an actual provider key.
    """

    KEY = "AIzaSyEXAMPLEKEYnotREALdoNOTuseTHIS0000"

    def _findings(self):
        return SecretScanner().scan([("classes.dex", self.KEY)])

    def test_key_is_still_detected(self):
        assert len(self._findings()) == 1

    def test_uncorroborated_key_is_generic_and_medium(self):
        out = _attribute_google_keys(self._findings(), [])
        assert out[0].provider == Provider.GENERIC
        assert out[0].severity == Severity.MEDIUM

    def test_key_is_promoted_when_gemini_endpoint_present(self):
        """Context makes the attribution defensible rather than a guess."""
        out = _attribute_google_keys(self._findings(), [Provider.GOOGLE_GEMINI])
        assert out[0].provider == Provider.GOOGLE_GEMINI
        assert out[0].severity == Severity.HIGH

    def test_unrelated_provider_context_does_not_promote(self):
        out = _attribute_google_keys(self._findings(), [Provider.OPENAI])
        assert out[0].provider == Provider.GENERIC

    def test_real_provider_keys_are_untouched(self):
        # In pieces for the reason given in tests/fixtures/seeded_strings.py.
        key = "sk-" + "AbCdEfGhIjKlMnOpQrSt" + "T3BlbkFJ" + "AbCdEfGhIjKlMnOpQrSt"
        openai = SecretScanner().scan([("classes.dex", key)])
        out = _attribute_google_keys(openai, [Provider.GOOGLE_GEMINI])
        assert out[0].provider == Provider.OPENAI
        assert out[0].severity == openai[0].severity
