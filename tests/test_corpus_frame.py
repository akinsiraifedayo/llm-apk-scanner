"""Tests for AI-candidate frame construction.

The regression cases here are drawn from the pilot corpus: the earlier
substring filter admitted a card game, a live wallpaper and a pill manager,
and rejected genuine AI apps whose names contained ``tail``/``mail``.
"""

from __future__ import annotations

from pathlib import Path

from llm_apk_scanner.corpus.frame import (
    ai_signals,
    deduplicate_by_package,
    filter_frame,
    is_ai_candidate,
    read_frame,
    tokenize,
    write_frame,
)
from llm_apk_scanner.corpus.model import ApkRecord


def test_tokenize_splits_separators_and_camel_case():
    assert tokenize("com.myApp.aiChat_v2") == ["com", "my", "app", "ai", "chat", "v", "2"]


class TestAcceptsRealAiApps:
    """Packages that must stay in the frame."""

    def test_vendor_packages(self):
        assert is_ai_candidate("com.openai.chatgpt")
        assert is_ai_candidate("ai.replika.app")

    def test_ai_as_standalone_segment(self):
        assert is_ai_candidate("com.google.android.apps.ai.sandbox")
        assert is_ai_candidate("com.ai.flashcontact")

    def test_concatenated_segments(self):
        """Package segments run words together with no separator."""
        assert is_ai_candidate("com.myguru.aichatbot")
        assert is_ai_candidate("chatgptassistant.app")

    def test_inflected_vendor_stems(self):
        assert is_ai_candidate("com.anthropics.zyler_swipe")

    def test_not_rejected_by_incidental_substrings(self):
        """These were wrongly excluded by the old 'tail'/'mail' blacklist."""
        assert is_ai_candidate("com.aitranslate.detailview")
        assert is_ai_candidate("com.gpt.retailassistant")
        assert is_ai_candidate("com.chatbot.email.helper")


class TestRejectsNonAiApps:
    """Packages the old substring filter wrongly pulled into the corpus."""

    def test_short_signals_do_not_match_inside_words(self):
        assert not is_ai_candidate("com.ninefold.lombardos")  # 'bard'
        assert not is_ai_candidate("com.lexonuk.pillmanager")  # 'llm'
        assert not is_ai_candidate("com.llmobile")  # 'llm'
        assert not is_ai_candidate("com.subsplashconsulting.s_4GPTVX")  # 'gpt'
        assert not is_ai_candidate("com.bardi.smarthome.tv")  # 'bard'

    def test_ai_does_not_match_inside_ordinary_words(self):
        for pkg in (
            "com.gmail.client",
            "com.retailstore.app",
            "com.certain.software",
            "com.detailing.car",
            "com.airport.flighttracker",
            "com.waiting.room",
            "com.maintain.tool",
        ):
            assert not is_ai_candidate(pkg), pkg

    def test_unrelated_apps(self):
        assert not is_ai_candidate("com.facebook.katana")
        assert not is_ai_candidate("com.spotify.music")

    def test_vetoed_genres(self):
        """Genre veto overrides a real signal, including inside a token."""
        assert ai_signals("com.chatbot.wallpaper.hd") == []
        assert ai_signals("com.gpt.keyboard") == []
        # The genre word is concatenated, exactly as signal words are.
        assert ai_signals("com.aiwallpapermaker.hd") == []
        assert ai_signals("com.chatgptkeyboardpro.app") == []


def test_signals_are_reported_for_audit():
    """The frame records *why* a row was admitted."""
    signals = ai_signals("com.openai.chatgpt")
    assert "openai" in signals
    assert "chatgpt" in signals
    assert ai_signals("com.spotify.music") == []


def test_frame_round_trips_through_csv(tmp_path: Path):
    records = [
        ApkRecord(
            sha256="A" * 64,
            pkg_name="com.openai.chatgpt",
            markets="play.google.com",
            dex_date="2025-03-01 00:00:00",
            vercode="17",
            apk_size=4096,
        ),
        ApkRecord(sha256="B" * 64, pkg_name="com.spotify.music", markets="play.google.com"),
    ]
    path = tmp_path / "frame.csv"
    count = write_frame(filter_frame(records), path)

    assert count == 1  # only the AI candidate is admitted
    restored = read_frame(path)
    assert len(restored) == 1
    assert restored[0].sha256 == "A" * 64
    assert restored[0].pkg_name == "com.openai.chatgpt"
    assert restored[0].year == 2025
    assert restored[0].apk_size == 4096
    assert restored[0].store_stratum == "google_play"


class TestDeduplicateByPackage:
    """AndroZoo indexes every version, so one app appears many times.

    Sampling over those rows would make the unit of analysis the APK rather
    than the app, and RQ1 asks about apps.
    """

    @staticmethod
    def _rec(sha: str, pkg: str, vercode: str | None, year: int = 2025) -> ApkRecord:
        return ApkRecord(
            sha256=sha * 64,
            pkg_name=pkg,
            markets="play.google.com",
            dex_date=f"{year}-01-01 00:00:00",
            vercode=vercode,
        )

    def _dedupe(self, records):
        rows = [(r, ["gpt"]) for r in records]
        return list(deduplicate_by_package(rows))

    def test_keeps_highest_version_code(self):
        kept = self._dedupe([
            self._rec("A", "com.x.gpt", "3"),
            self._rec("B", "com.x.gpt", "17"),
            self._rec("C", "com.x.gpt", "9"),
        ])
        assert len(kept) == 1
        assert kept[0][0].sha256 == "B" * 64

    def test_distinct_packages_all_survive(self):
        kept = self._dedupe([
            self._rec("A", "com.one.gpt", "1"),
            self._rec("B", "com.two.gpt", "1"),
        ])
        assert {r.pkg_name for r, _ in kept} == {"com.one.gpt", "com.two.gpt"}

    def test_non_numeric_vercode_loses_to_numeric(self):
        kept = self._dedupe([
            self._rec("A", "com.x.gpt", "not-a-number"),
            self._rec("B", "com.x.gpt", "2"),
        ])
        assert kept[0][0].sha256 == "B" * 64

    def test_missing_vercode_falls_back_to_date_then_sha(self):
        """The ordering must be total, or the frame is not reproducible."""
        kept = self._dedupe([
            self._rec("A", "com.x.gpt", None, year=2024),
            self._rec("B", "com.x.gpt", None, year=2026),
        ])
        assert kept[0][0].sha256 == "B" * 64

    def test_output_is_deterministic(self):
        records = [
            self._rec("C", "com.b.gpt", "1"),
            self._rec("A", "com.a.gpt", "1"),
            self._rec("B", "com.c.gpt", "1"),
        ]
        first = [r.sha256 for r, _ in self._dedupe(records)]
        second = [r.sha256 for r, _ in self._dedupe(list(reversed(records)))]
        assert first == second == sorted(first)

    def test_signals_travel_with_the_kept_record(self):
        rows = [
            (self._rec("A", "com.x.gpt", "1"), ["gpt"]),
            (self._rec("B", "com.x.gpt", "2"), ["gpt", "chatbot"]),
        ]
        kept = list(deduplicate_by_package(rows))
        assert kept[0][1] == ["gpt", "chatbot"]
