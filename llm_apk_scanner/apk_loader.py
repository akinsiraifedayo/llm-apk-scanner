"""
APK loading and string-corpus extraction.

This module is the boundary between Androguard's object model and the
plain (source_file, content) tuples that the scanners and filters
consume. Keeping that boundary thin means we can swap in alternative
backends (e.g. apktool + plain file walks, or our own DEX parser)
without touching the detectors.

The loader is import-time tolerant: if Androguard is not installed,
imports of this module still succeed; only `load_apk()` raises. This
is so unit tests can run against the pure-Python detectors without
needing the full Android tool-chain.
"""

from __future__ import annotations

import hashlib
import io
import logging
import os
from collections.abc import Iterator
from dataclasses import dataclass

from llm_apk_scanner.models import ApkInfo
from llm_apk_scanner.native_strings import extract_strings

logger = logging.getLogger(__name__)

#: Skip native libraries larger than this. Beyond ~64MB the extracted
#: strings are dominated by symbol tables and padding, and the cost stops
#: buying coverage.
_MAX_NATIVE_LIB_BYTES = 64 * 1024 * 1024


@dataclass
class ApkCorpus:
    """The flattened text corpus of an APK ready for scanner input."""

    info: ApkInfo
    sources: list[tuple[str, str]]

    def __iter__(self) -> Iterator[tuple[str, str]]:
        return iter(self.sources)


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


#: Largest asset read into memory. Bounds worst-case cost per APK; assets
#: above this are almost always media, which carry no signal anyway.
MAX_ASSET_BYTES = 64 * 1024 * 1024

#: Fraction of a sample that must be printable for the asset to be treated as
#: text rather than run through binary string extraction.
_TEXT_RATIO = 0.85
_SNIFF_BYTES = 4096

#: Media, fonts and archives: never worth reading, and expensive at scale.
#: This is a *denylist*, deliberately. An allowlist of "textual" extensions is
#: what caused Defect 14: React Native ships its entire application logic in
#: ``assets/index.android.bundle``, which is not on any plausible textual
#: allowlist, and the compiled resource table ``resources.arsc`` holds string
#: pools that routinely carry API keys. Both were silently skipped. The
#: failure mode of an allowlist is invisible under-reading; the failure mode of
#: a denylist is wasted cycles on a file with no signal, which is recoverable.
_SKIP_EXTENSIONS = (
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".ico", ".svg",
    ".mp3", ".mp4", ".wav", ".ogg", ".m4a", ".aac", ".webm", ".avi", ".mov",
    ".ttf", ".otf", ".woff", ".woff2", ".eot",
    ".zip", ".gz", ".br", ".7z", ".rar", ".jar", ".aar",
)


def _is_skippable_media(asset_path: str) -> bool:
    return asset_path.lower().endswith(_SKIP_EXTENSIONS)


def _decode_asset(blob: bytes) -> str:
    """Return searchable text for an asset, whatever its container format.

    Text decodes directly. Anything else is run through the same ASCII and
    UTF-16-LE string extraction used for native libraries, because binary
    containers still carry readable string pools: ``resources.arsc`` stores
    resource strings, and a React Native bundle compiled to Hermes bytecode
    (magic ``0xC61FBC03``) stores every string literal the application uses,
    including endpoints and credentials.
    """
    sample = blob[:_SNIFF_BYTES]
    if sample:
        printable = sum(1 for b in sample if 32 <= b < 127 or b in (9, 10, 13))
        if printable / len(sample) >= _TEXT_RATIO:
            return blob.decode("utf-8", errors="replace")
    return "\n".join(extract_strings(io.BytesIO(blob)))


def load_apk(path: str) -> ApkCorpus:
    """
    Load an APK and return its (source_file, content) corpus.

    Sources extracted (matching METHODOLOGY.md §3.1):
      - AndroidManifest.xml (decoded XML).
      - All entries from the dex string tables, attributed to
        `classes.dex` (or classesN.dex).
      - Native-library strings from every bundled `.so`, ASCII and
        UTF-16-LE. Flutter, React Native and NDK apps can hold their
        endpoints and prompts here and nowhere else.
      - All asset and resource files of plausibly textual MIME (txt,
        json, xml, jsonl, csv, properties, conf, yml, yaml, html).
      - Asset paths themselves (so file-name-based detectors fire).

    Raises:
      ImportError if Androguard is not installed.
      FileNotFoundError if `path` does not exist.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    try:
        from androguard.core.apk import APK  # type: ignore
        from androguard.core.dex import DEX  # type: ignore
    except ImportError as e:
        raise ImportError(
            "androguard is required to load APKs. Install with "
            "`pip install androguard` or run scanners directly "
            "against text corpora using the test fixtures."
        ) from e

    apk = APK(path)
    info = ApkInfo(
        package_name=apk.get_package() or "<unknown>",
        version_name=apk.get_androidversion_name(),
        version_code=int(apk.get_androidversion_code() or 0) or None,
        min_sdk=int(apk.get_min_sdk_version() or 0) or None,
        target_sdk=int(apk.get_target_sdk_version() or 0) or None,
        file_path=os.path.abspath(path),
        sha256=_sha256(path),
    )

    sources: list[tuple[str, str]] = []

    # 1. Manifest
    try:
        manifest_xml = apk.get_android_manifest_axml().get_xml().decode(
            "utf-8", errors="replace"
        )
        sources.append(("AndroidManifest.xml", manifest_xml))
    except Exception as e:  # noqa: BLE001 — Androguard surfaces many shapes
        logger.warning("Failed to extract AndroidManifest.xml: %s", e)

    # 2. DEX string tables
    for dex_name in apk.get_dex_names():
        try:
            dex_bytes = apk.get_file(dex_name)
            dex = DEX(dex_bytes)
            strings = "\n".join(s for s in dex.get_strings() if isinstance(s, str))
            sources.append((dex_name, strings))
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to parse %s: %s", dex_name, e)

    # 3. Native libraries.
    #
    # METHODOLOGY.md §3.1 states native-library strings are searched, and
    # native_strings.py implements the extraction, but nothing called it —
    # it was reachable only from its own unit test. Apps built with the NDK,
    # Flutter (which compiles Dart into libapp.so) or React Native can keep
    # their endpoints and prompts entirely outside the DEX string table, so
    # this closes a documented-but-absent capability rather than adding a
    # new one.
    #
    # Extraction is capped: a large .so yields megabytes of strings, and
    # beyond a point the additional strings are padding and symbol noise
    # rather than program data.
    for native_path in apk.get_files():
        if not native_path.endswith(".so"):
            continue
        try:
            blob = apk.get_file(native_path)
            if not blob or len(blob) > _MAX_NATIVE_LIB_BYTES:
                continue
            extracted = "\n".join(
                extract_strings(io.BytesIO(blob), min_length=6)
            )
            if extracted:
                sources.append((native_path, extracted))
        except Exception as e:  # noqa: BLE001
            logger.debug("Skipping native library %s: %s", native_path, e)

    # 4. Assets and resources.
    for asset_path in apk.get_files():
        # Always emit the asset *path* so file-name-based detectors
        # (RAG file rules) fire.
        sources.append((asset_path, asset_path))
        if _is_skippable_media(asset_path):
            continue
        try:
            blob = apk.get_file(asset_path)
        except Exception as e:  # noqa: BLE001
            logger.debug("Skipping asset %s: %s", asset_path, e)
            continue
        if not blob or len(blob) > MAX_ASSET_BYTES:
            continue
        content = _decode_asset(blob)
        if content:
            sources.append((asset_path, content))

    return ApkCorpus(info=info, sources=sources)
