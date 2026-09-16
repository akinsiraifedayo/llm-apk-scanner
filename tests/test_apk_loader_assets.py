"""Regression tests for Defect 14: assets read by extension allowlist.

`apk_loader` filtered assets through a list of "textual" extensions. React
Native ships its entire application logic in `assets/index.android.bundle`
(Hermes bytecode, magic 0xC61FBC03) and Android stores resource strings in
`resources.arsc`. Neither is on any plausible textual allowlist, so both were
silently skipped.

Measured consequence on one React Native application: the scanner
reported "not LLM-integrated, one generic Google key". The bundle actually
carries `api.openai.com`, an OpenAI-format credential and `gpt-3.5-turbo`,
and `resources.arsc` carries a second Google key.
"""

from __future__ import annotations

import io

from llm_apk_scanner.apk_loader import (
    MAX_ASSET_BYTES,
    _decode_asset,
    _is_skippable_media,
)

HERMES_MAGIC = b"\xc6\x1f\xbc\x03"


class TestAssetDecoding:
    def test_plain_text_decodes_directly(self):
        blob = b'{"model": "gpt-4", "endpoint": "https://api.openai.com/v1"}'
        out = _decode_asset(blob)
        assert "api.openai.com" in out
        assert "gpt-4" in out

    def test_hermes_bytecode_yields_its_string_pool(self):
        """A React Native bundle is binary but its literals are readable."""
        blob = HERMES_MAGIC + bytes(64) + b"api.openai.com" + bytes(32) \
            + b"sk-proj-AAAAAAAAAAAAAAAAAAAAAAAA" + bytes(16)
        out = _decode_asset(blob)
        assert "api.openai.com" in out
        assert "sk-proj-AAAAAAAAAAAAAAAAAAAAAAAA" in out

    def test_binary_resource_table_yields_strings(self):
        """`resources.arsc` is binary and routinely carries API keys."""
        blob = b"\x02\x00\x0c\x00" + bytes(48) \
            + b"AIzaSyEXAMPLEarscKEYnotREALdoNOTuse0000" + bytes(24)
        out = _decode_asset(blob)
        assert "AIzaSyEXAMPLEarscKEYnotREALdoNOTuse0000" in out

    def test_utf16_strings_are_recovered(self):
        blob = bytes(32) + "api.anthropic.com".encode("utf-16-le") + bytes(32)
        assert "api.anthropic.com" in _decode_asset(blob)

    def test_empty_asset_is_safe(self):
        assert _decode_asset(b"") == ""

    def test_mostly_text_with_a_few_control_bytes_still_decodes(self):
        blob = b"You are a helpful assistant.\x00 Always answer in English." * 4
        assert "helpful assistant" in _decode_asset(blob)


class TestMediaSkipping:
    def test_media_is_skipped(self):
        for p in ("res/drawable/icon.png", "assets/a.MP4", "res/font/x.ttf",
                  "assets/bundle.zip", "res/raw/clip.webm"):
            assert _is_skippable_media(p), p

    def test_payload_bearing_assets_are_not_skipped(self):
        """The regression: none of these may ever be skipped again."""
        for p in ("assets/index.android.bundle", "resources.arsc",
                  "assets/main.jsbundle", "assets/flutter_assets/AssetManifest",
                  "assets/config", "assets/data.pak", "assets/model.bin"):
            assert not _is_skippable_media(p), p

    def test_size_cap_is_sane(self):
        assert MAX_ASSET_BYTES >= 16 * 1024 * 1024
