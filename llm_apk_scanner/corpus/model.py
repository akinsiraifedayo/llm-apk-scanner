"""The AndroZoo metadata record and its derived strata.

An ``ApkRecord`` is one row of the AndroZoo metadata index (``latest.csv``).
The properties derive the two stratification dimensions used by the sampler:
publication *year* and *store*.
"""

from __future__ import annotations

from dataclasses import dataclass

# Store-stratum classification. AndroZoo's ``markets`` field is a
# separator-delimited list of the app stores a given APK was observed on.
_GOOGLE_PLAY_MARKERS = ("play.google.com", "google play", "googleplay")
_FDROID_MARKERS = ("f-droid", "fdroid")

#: Earliest year an Android APK can plausibly carry.
#:
#: The first Android device shipped in September 2008, so any earlier date is
#: an artefact rather than a fact. AndroZoo's ``dex_date`` derives from the DEX
#: file's embedded timestamp, which build tooling routinely zeroes or omits;
#: those rows surface as 1980-01-01 or 1981-01-01. Measured on the 2026-08-13
#: index snapshot, **70.3% of rows carry such a date**, so treating them as
#: genuine publication years does not merely add noise — it makes any
#: year-stratified sample a sample of the minority whose tooling happened to
#: preserve a timestamp.
MIN_PLAUSIBLE_YEAR = 2008


@dataclass(frozen=True)
class ApkRecord:
    """One APK as described by the AndroZoo metadata index."""

    sha256: str
    pkg_name: str
    markets: str
    dex_date: str | None = None
    vt_scan_date: str | None = None
    added: str | None = None
    vercode: str | None = None
    apk_size: int | None = None

    @property
    def year(self) -> int | None:
        """Publication year, with a fallback for corrupt DEX timestamps.

        Sources are tried in order of directness: ``dex_date`` (when the
        build), then ``vt_scan_date`` (when AndroZoo's VirusTotal scan
        observed it), then ``added``. A source is used only if it yields a
        year at or after :data:`MIN_PLAUSIBLE_YEAR`; otherwise the next is
        tried. Returns ``None`` when no source is usable.

        ``vt_scan_date`` dates *observation in circulation* rather than
        construction, which is a weaker claim and should be reported as one.
        For this study it is also arguably the more relevant one: an
        application observed in circulation during the study window is an
        application a user could have installed then, whatever its DEX
        timestamp says. The fallback is recorded per row via
        :attr:`date_source` so the composition is auditable.

        ``added`` is retained last and is normally absent: the public
        ``latest.csv.gz`` index carries no such column.
        """
        for source in (self.dex_date, self.vt_scan_date, self.added):
            if source and len(source) >= 4 and source[:4].isdigit():
                candidate = int(source[:4])
                if candidate >= MIN_PLAUSIBLE_YEAR:
                    return candidate
        return None

    @property
    def date_source(self) -> str:
        """Which field supplied :attr:`year` — ``dex_date``, ``vt_scan_date``,
        ``added``, or ``none``.

        Carried into the frame so the Methodology chapter can report how many
        rows were dated by each field rather than asserting that the dating is
        uniform.
        """
        for name, source in (
            ("dex_date", self.dex_date),
            ("vt_scan_date", self.vt_scan_date),
            ("added", self.added),
        ):
            if source and len(source) >= 4 and source[:4].isdigit():
                if int(source[:4]) >= MIN_PLAUSIBLE_YEAR:
                    return name
        return "none"

    @property
    def store_stratum(self) -> str:
        """Coarse store bucket: ``google_play``, ``f_droid`` or ``other``."""
        markets = self.markets.lower()
        if any(marker in markets for marker in _GOOGLE_PLAY_MARKERS):
            return "google_play"
        if any(marker in markets for marker in _FDROID_MARKERS):
            return "f_droid"
        return "other"

    @property
    def stratum(self) -> tuple[int | None, str]:
        """The (year, store) key this record belongs to."""
        return (self.year, self.store_stratum)
