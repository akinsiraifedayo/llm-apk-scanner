"""Regression tests for the frame defects found on 2026-08-13.

Both changed the study population, so both are pinned here:
  * MIT App Inventor namespace collision (`ai_` means *App Inventor*)
  * corrupt `dex_date`, and the `vt_scan_date` fallback that repairs it

See `docs/METHODOLOGY.md` §2.3.
"""

from __future__ import annotations

from llm_apk_scanner.corpus.frame import ai_signals
from llm_apk_scanner.corpus.model import MIN_PLAUSIBLE_YEAR, ApkRecord


class TestAppInventorVeto:
    """`appinventor.ai_<user>.<App>` is not an AI signal.

    56,597 of 72,960 rows in the 2026-08-12 frame (77.6%) were App Inventor
    apps admitted on the `ai` token alone.
    """

    def test_app_inventor_package_yields_no_signals(self):
        assert ai_signals("appinventor.ai_bhagattv.PremiumNotepad") == []

    def test_app_inventor_with_real_ai_word_still_vetoed(self):
        """The veto is a namespace exclusion, so it outranks other signals.

        An App Inventor app *called* "chatbot" is still a block-built
        hobbyist app, not evidence of LLM integration.
        """
        assert ai_signals("appinventor.ai_someuser.MyChatbot") == []

    def test_legitimate_ai_prefix_is_not_vetoed(self):
        """`ai.` is a real vendor namespace and must survive.

        This is the case the veto could most plausibly break: it is a prefix
        test precisely so that `ai.` does not collide with `appinventor.ai_`.
        """
        assert ai_signals("ai.pic.solve.answer.photo.math.mcq.homework")
        assert ai_signals("ai.seaart.app.global")

    def test_other_appinventor_namespaces_unaffected(self):
        """Only the `ai_` namespace is vetoed, not the vendor's own apps."""
        assert ai_signals("edu.mit.appinventor.aicompanion3")


class TestDateFallback:
    """`dex_date` is absent or zeroed on 70.3% of index rows.

    Those rows surface as 1980/1981. Treating them as publication years makes
    a year-stratified sample a sample of the minority whose build tooling
    happened to preserve a timestamp.
    """

    def _rec(self, **kw) -> ApkRecord:
        return ApkRecord(sha256="A" * 64, pkg_name="com.example.app", markets="", **kw)

    def test_valid_dex_date_is_preferred(self):
        r = self._rec(dex_date="2025-03-04 11:00:00", vt_scan_date="2026-01-01 00:00:00")
        assert r.year == 2025
        assert r.date_source == "dex_date"

    def test_zeroed_dex_date_falls_back_to_vt_scan_date(self):
        r = self._rec(dex_date="1980-01-01 00:00:00", vt_scan_date="2025-09-04 08:26:10")
        assert r.year == 2025
        assert r.date_source == "vt_scan_date"

    def test_1981_artefact_falls_back(self):
        r = self._rec(dex_date="1981-01-01 01:01:02", vt_scan_date="2024-06-01 00:00:00")
        assert r.year == 2024
        assert r.date_source == "vt_scan_date"

    def test_missing_dex_date_falls_back(self):
        r = self._rec(dex_date=None, vt_scan_date="2026-02-02 00:00:00")
        assert r.year == 2026

    def test_no_usable_source_yields_none(self):
        r = self._rec(dex_date="1980-01-01 00:00:00", vt_scan_date="1980-01-01 00:00:00")
        assert r.year is None
        assert r.date_source == "none"

    def test_boundary_year_is_accepted(self):
        """2008 is the first Android release and must not be rejected."""
        r = self._rec(dex_date=f"{MIN_PLAUSIBLE_YEAR}-09-23 00:00:00")
        assert r.year == MIN_PLAUSIBLE_YEAR
        assert r.date_source == "dex_date"

    def test_year_before_boundary_is_rejected(self):
        r = self._rec(dex_date=f"{MIN_PLAUSIBLE_YEAR - 1}-01-01 00:00:00")
        assert r.year is None

    def test_added_still_works_as_last_resort(self):
        """The public index has no `added` column, but private exports may."""
        r = self._rec(dex_date="1980-01-01 00:00:00", added="2023-05-05 00:00:00")
        assert r.year == 2023
        assert r.date_source == "added"
