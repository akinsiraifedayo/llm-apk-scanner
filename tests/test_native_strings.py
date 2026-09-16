"""Tests for the native-library string extractor."""

from __future__ import annotations

import io

from llm_apk_scanner.native_strings import extract_strings


def _stream(b: bytes) -> io.BytesIO:
    return io.BytesIO(b)


def test_extracts_ascii_run() -> None:
    data = b"\x00\x00api.openai.com\x00\x00"
    found = list(extract_strings(_stream(data)))
    assert "api.openai.com" in found


def test_respects_min_length() -> None:
    data = b"abc\x00\x00\x00abcdef\x00"
    found = list(extract_strings(_stream(data), min_length=4))
    # 'abc' is len 3 — below threshold.
    assert "abcdef" in found
    assert "abc" not in found


def test_extracts_utf16_le_run() -> None:
    # b'h\x00e\x00l\x00l\x00o\x00' → "hello" UTF-16-LE.
    data = b"\xff\xff" + b"h\x00e\x00l\x00l\x00o\x00 \x00w\x00o\x00r\x00l\x00d\x00\xff\xff"
    found = list(extract_strings(_stream(data)))
    assert "hello world" in found


def test_handles_mixed_garbage_and_strings() -> None:
    junk = b"\x01\x02\x03\xff\xfe"
    s = b"https://api.openai.com/v1/chat"
    data = junk + s + junk + b"sk-anth-abcdef" + junk
    found = list(extract_strings(_stream(data)))
    assert "https://api.openai.com/v1/chat" in found
    assert "sk-anth-abcdef" in found


def test_empty_stream() -> None:
    assert list(extract_strings(_stream(b""))) == []


def test_only_one_byte() -> None:
    assert list(extract_strings(_stream(b"a"))) == []
