"""Tests for neutral evidence extraction.

The coder judges ground truth from this output, so it has to surface real
inference endpoints rather than bury them under noise. The regression cases
here come from running the first version against the ChatGPT APK, where the
extracted "hosts" were dominated by decimals out of the DEX string table.
"""

from __future__ import annotations

from llm_apk_scanner.validation.evidence import (
    Evidence,
    _interesting_host,
    render_evidence,
)


class TestHostFiltering:
    def test_accepts_real_inference_endpoints(self):
        for host in (
            "api.openai.com",
            "api.anthropic.com",
            "generativelanguage.googleapis.com",
            "api.cohere.ai",
            "chatgpt.com",
            "livekit.io",
        ):
            assert _interesting_host(host), host

    def test_rejects_decimals_from_the_dex_string_table(self):
        """These flooded the first version and buried the real endpoints."""
        for noise in ("0.0", "2.761", "1.41409301758", "14.0", "0.160784313725"):
            assert not _interesting_host(noise), noise

    def test_rejects_version_like_strings(self):
        assert not _interesting_host("license-2.0")
        assert not _interesting_host("apache-2.0")

    def test_rejects_reverse_domain_package_names(self):
        """Android classes are reverse-domain and look just like hosts."""
        for pkg in (
            "androidx.credentials.type",
            "com.google.android.gms",
            "io.reactivex.rxjava3",
            "org.apache.commons",
        ):
            assert not _interesting_host(pkg), pkg

    def test_rejects_boring_infrastructure(self):
        assert not _interesting_host("play.google.com")
        assert not _interesting_host("crashlytics.com")
        assert not _interesting_host("google-analytics.com")

    def test_rejects_unknown_tlds(self):
        assert not _interesting_host("some.thing.zzzz")


class TestRendering:
    def test_renders_all_sections(self):
        evidence = Evidence(
            sha256="A" * 64,
            package="com.example.ai",
            version="1.0",
            hosts=[("api.openai.com", 4)],
            ai_strings=[("classes.dex", "you are a helpful assistant")],
            sdk_paths=["com/openai/client/Chat.java"],
        )
        rendered = render_evidence(evidence)
        assert "com.example.ai" in rendered
        assert "api.openai.com" in rendered
        assert "you are a helpful assistant" in rendered
        assert "com/openai/client/Chat.java" in rendered

    def test_renders_empty_sections_without_crashing(self):
        rendered = render_evidence(Evidence(sha256="B" * 64, package="com.plain"))
        assert "(none found)" in rendered

    def test_reports_load_errors_instead_of_raising(self):
        """A corrupt APK still needs a worksheet row the coder can act on."""
        rendered = render_evidence(
            Evidence(sha256="C" * 64, error="BadZipFile: not a zip")
        )
        assert "BadZipFile" in rendered

    def test_never_states_a_verdict(self):
        """Evidence only — showing the prediction would anchor the coder."""
        evidence = Evidence(
            sha256="A" * 64, package="com.example.ai",
            hosts=[("api.openai.com", 4)],
        )
        rendered = render_evidence(evidence).lower()
        for verdict_word in ("llm_integrated", "predicted", "scanner says", "finding"):
            assert verdict_word not in rendered


def test_is_empty_reports_absence_of_signal():
    assert Evidence(sha256="A" * 64).is_empty
    assert not Evidence(sha256="A" * 64, hosts=[("api.openai.com", 1)]).is_empty
