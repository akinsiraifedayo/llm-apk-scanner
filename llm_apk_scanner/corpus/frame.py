"""Construct the AI-candidate sample frame from the AndroZoo index.

``docs/METHODOLOGY.md`` §2.2-2.3 defines the study population as Android
apps that *plausibly* integrate an LLM, drawn from the AndroZoo index. This
module builds that frame: it streams AndroZoo's ``latest.csv.gz`` and keeps
rows whose package name carries an AI signal. The result is the population
from which ``sample.stratified_sample`` draws the 10,000-APK manifest.

Matching is **token-based, not substring-based.** A package name is split on
``.``, ``_``, ``-``, digits, and camelCase boundaries, and signals are tested
against those tokens. This matters: matching ``"ai"`` as a raw substring
flags ``mail``, ``retail``, ``certain``, ``waiting`` and ``airport``, which
the earlier implementation then tried to undo with a blacklist of fragments
(``"tain"``, ``"tail"``, ``"rait"``, ...) that also silently discarded real
AI apps whose names happened to contain them. Tokenising removes both the
false positives and the need for the blacklist.

The frame is a *filter*, not a classifier: it decides what is worth
downloading, and the scanner decides what actually integrates an LLM. It is
deliberately recall-oriented — a package with no AI signal in its name can
still integrate an LLM, and §6 of the methodology records this as a known
frame-coverage limitation.
"""

from __future__ import annotations

import csv
import gzip
import re
import urllib.request
from collections.abc import Iterable, Iterator
from io import TextIOWrapper
from pathlib import Path

from llm_apk_scanner.corpus.androzoo import parse_metadata
from llm_apk_scanner.corpus.model import ApkRecord

#: AndroZoo's public index of every APK it holds (~7GB gzipped).
METADATA_URL = "https://androzoo.uni.lu/static/lists/latest.csv.gz"

#: Short or ambiguous signals, matched only as a **whole token**. These are
#: too short to match inside a word without generating nonsense: ``gpt``
#: inside ``s_4GPTVX``, ``bard`` inside ``lombardos``, ``llm`` inside
#: ``pillmanager``, ``ai`` inside ``mail`` — every one of which the earlier
#: substring implementation pulled into the corpus.
SIGNAL_TOKENS: frozenset[str] = frozenset(
    {
        "ai", "ml", "llm", "genai", "aigc",
        "gpt", "gpt3", "gpt4", "gpt5", "gpt4o",
        "bard", "palm", "chai", "sora", "dalle", "grok", "qwen", "claude",
        "llama", "cohere", "aibot", "chatai", "aiart",
    }
)

#: Long, unambiguous signals, matched **anywhere inside a token**. Package
#: segments routinely concatenate words with no separator
#: (``aichatbot``, ``chatgptassistant``), so these would be missed
#: by whole-token matching. Every entry is >=6 characters and specific enough
#: that an incidental substring hit is implausible.
SIGNAL_SUBSTRINGS: tuple[str, ...] = (
    # Vendors and model families (stems, so inflections are caught:
    # ``anthropics``, ``geminiai``, ``llamachat``).
    "chatgpt", "openai", "anthropic", "gemini", "mistral", "deepseek",
    "copilot", "perplexity", "huggingface", "midjourney", "replika",
    "stablediffusion",
    # Capability phrases.
    "chatbot", "aichat", "aiassist", "aicompanion", "aiwriter", "aiavatar",
    "aiimage", "aivideo", "aigener", "aisummar", "aiparaphras", "aitranslat",
    "aienhance", "texttoimage", "imagegener", "artgener", "texttospeech",
    "speechtotext", "machinelearn", "deeplearn", "faceswap", "deepfake",
    "neural", "transcrib", "summariz", "paraphras",
)

#: Package prefixes that are AI apps by construction.
SIGNAL_PACKAGE_PREFIXES: tuple[str, ...] = (
    "com.openai.", "com.anthropic.", "ai.", "com.google.android.apps.bard",
)

#: Genres that veto a match. These adopt AI vocabulary for marketing
#: ("AI wallpaper", "AI keyboard") but are out of scope for LLM integration.
#: Matched inside tokens, because genre words concatenate exactly as signals
#: do (``aiwallpaper``, ``gptkeyboard``). Kept deliberately short: this is a
#: scope boundary, not the fragment blacklist the old substring matcher
#: needed to undo its own false positives.
VETO_SUBSTRINGS: tuple[str, ...] = (
    "wallpaper", "ringtone", "sticker", "launcher", "keyboard", "emoji",
)

#: Package namespaces whose ``ai`` token does not mean artificial intelligence.
#:
#: MIT App Inventor, a block-based educational app builder, emits every
#: application it produces under ``appinventor.ai_<user>.<AppName>`` — where
#: ``ai_`` abbreviates *App Inventor*. Tokenising splits that into
#: ``["appinventor", "ai", ...]``, and ``ai`` is in :data:`SIGNAL_TOKENS`, so
#: the entire App Inventor corpus matched.
#:
#: Measured on the 2026-08-12 frame: **56,597 of 72,960 rows (77.6%)** were
#: App Inventor apps — calculators, radio-code lookups, notepads, puzzle aids.
#: In the sample drawn from that frame the share was 505 of 677 (74.6%).
#:
#: This is kept separate from :data:`VETO_SUBSTRINGS` deliberately. That list
#: is a *scope* boundary — genres that use AI vocabulary for marketing but are
#: out of scope. This one is a *namespace* exclusion: the token is not an AI
#: signal at all in this context. Conflating the two would lose the
#: distinction that makes each defensible.
#:
#: Note this is a prefix test, not a substring test, so the legitimate ``ai.``
#: package prefix in :data:`SIGNAL_PACKAGE_PREFIXES` is unaffected.
VETO_PACKAGE_PREFIXES: tuple[str, ...] = ("appinventor.ai_",)

_TOKEN_SPLIT = re.compile(r"[._\-]+|(?<=[a-z])(?=[A-Z])|(?<=[A-Za-z])(?=\d)|(?<=\d)(?=[A-Za-z])")


def tokenize(pkg_name: str) -> list[str]:
    """Split a package name into lowercase tokens.

    ``com.myApp.aiChat_v2`` -> ``["com", "my", "app", "ai", "chat", "v", "2"]``
    """
    parts = _TOKEN_SPLIT.split(pkg_name or "")
    return [p.lower() for p in parts if p]


def ai_signals(pkg_name: str) -> list[str]:
    """Return the AI signals present in a package name (empty if none).

    Returning the matched signals rather than a bare bool lets the frame
    record *why* each row was included, which is what makes the frame
    auditable in the corpus chapter.
    """
    pkg_lower = (pkg_name or "").lower()
    tokens = tokenize(pkg_name)
    token_set = set(tokens)

    # Namespace veto first: no signal in an excluded namespace is a signal.
    if pkg_lower.startswith(VETO_PACKAGE_PREFIXES):
        return []

    if any(veto in token for token in tokens for veto in VETO_SUBSTRINGS):
        return []

    signals: list[str] = []
    for prefix in SIGNAL_PACKAGE_PREFIXES:
        if pkg_lower.startswith(prefix):
            signals.append(f"pkg:{prefix}")

    signals.extend(sorted(token_set & SIGNAL_TOKENS))

    for token in tokens:
        for needle in SIGNAL_SUBSTRINGS:
            if needle in token:
                signals.append(needle)

    # Preserve order of first appearance while removing duplicates.
    seen: set[str] = set()
    unique: list[str] = []
    for signal in signals:
        if signal not in seen:
            seen.add(signal)
            unique.append(signal)
    return unique


def is_ai_candidate(pkg_name: str) -> bool:
    """Whether a package name carries any AI signal."""
    return bool(ai_signals(pkg_name))


def filter_frame(records: Iterable[ApkRecord]) -> Iterator[tuple[ApkRecord, list[str]]]:
    """Yield ``(record, signals)`` for records with an AI signal."""
    for record in records:
        signals = ai_signals(record.pkg_name)
        if signals:
            yield record, signals


def _version_key(record: ApkRecord) -> tuple[int, str, str]:
    """Sort key picking the newest release of an app.

    Highest ``vercode`` wins. Ties fall back to the later ``dex_date``/``added``
    and finally to the SHA-256, so the choice is total and deterministic — the
    frame must be reproducible from the same inputs.
    """
    vercode = record.vercode or ""
    numeric = int(vercode) if vercode.isdigit() else -1
    return (numeric, record.dex_date or record.added or "", record.sha256)


def deduplicate_by_package(
    rows: Iterable[tuple[ApkRecord, list[str]]],
) -> Iterator[tuple[ApkRecord, list[str]]]:
    """Collapse each package to its newest APK.

    AndroZoo indexes every *version* it has seen, so one app can appear many
    times under different SHA-256s. Sampling over those rows makes the unit of
    analysis the APK, not the app: a widely-versioned app gets many chances to
    be drawn, and a prevalence figure phrased as "X% of apps" is then counting
    some apps repeatedly.

    RQ1 asks about applications, so the frame is reduced to one row per
    ``pkg_name`` — the highest version code — before the sample is drawn.

    Output is sorted by SHA-256 so the frame is byte-identical across runs.
    """
    best: dict[str, tuple[ApkRecord, list[str]]] = {}
    for record, signals in rows:
        key = record.pkg_name
        current = best.get(key)
        if current is None or _version_key(record) > _version_key(current[0]):
            best[key] = (record, signals)
    yield from sorted(best.values(), key=lambda item: item[0].sha256)


def stream_androzoo_index(
    url: str = METADATA_URL, *, timeout: float = 600.0
) -> Iterator[ApkRecord]:
    """Stream and parse the AndroZoo index without materialising it on disk.

    The index is ~7GB gzipped; it is decompressed on the fly and parsed row
    by row so peak memory stays flat regardless of index size.
    """
    request = urllib.request.Request(url, headers={"User-Agent": "UEL-Dissertation-Research/1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        with gzip.GzipFile(fileobj=response) as gz:
            text = TextIOWrapper(gz, encoding="utf-8", errors="replace")
            yield from parse_metadata(text)


_FRAME_FIELDS = [
    "sha256", "pkg_name", "year", "date_source", "store",
    "vercode", "markets", "apk_size", "signals",
]


def write_frame(rows: Iterable[tuple[ApkRecord, list[str]]], path: Path) -> int:
    """Write the candidate frame to CSV, returning the row count.

    Written incrementally so an interrupted index stream still leaves a
    usable partial frame on disk.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_FRAME_FIELDS)
        writer.writeheader()
        for record, signals in rows:
            writer.writerow(
                {
                    "sha256": record.sha256,
                    "pkg_name": record.pkg_name,
                    "year": record.year if record.year is not None else "",
                    "date_source": record.date_source,
                    "store": record.store_stratum,
                    "vercode": record.vercode or "",
                    "markets": record.markets,
                    "apk_size": record.apk_size if record.apk_size is not None else "",
                    "signals": "|".join(signals),
                }
            )
            count += 1
            if count % 1000 == 0:
                handle.flush()
    return count


def read_frame(path: Path) -> list[ApkRecord]:
    """Read a frame CSV back into records, ready for stratified sampling."""
    records: list[ApkRecord] = []
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            year = (row.get("year") or "").strip()
            size = (row.get("apk_size") or "").strip()
            records.append(
                ApkRecord(
                    sha256=row["sha256"],
                    pkg_name=row.get("pkg_name", ""),
                    markets=row.get("markets", ""),
                    dex_date=f"{year}-01-01 00:00:00" if year else None,
                    vercode=row.get("vercode") or None,
                    apk_size=int(size) if size.isdigit() else None,
                )
            )
    return records
