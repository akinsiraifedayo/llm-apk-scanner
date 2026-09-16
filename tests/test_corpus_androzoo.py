"""Tests for the AndroZoo API client (metadata parsing + download).

No network is used: an in-memory ``fetcher`` is injected so the client's
URL construction, checksum verification, and error handling are exercised
against known bytes.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from llm_apk_scanner.corpus.androzoo import (
    AndroZooClient,
    AndroZooError,
    parse_metadata,
)

# A minimal slice of the AndroZoo ``latest.csv`` schema.
_HEADER = (
    "sha256,sha1,md5,dex_date,apk_size,pkg_name,vercode,"
    "vt_detection,vt_scan_date,dex_size,markets,added"
)
_ROW_PLAY = (
    "AAAA1111,sha1,md5,2025-03-01 10:00:00,123456,com.example.play,42,"
    "0,,50000,play.google.com,2025-03-02 09:00:00"
)
_ROW_FDROID = (
    "BBBB2222,sha1,md5,2024-11-01 10:00:00,222222,org.example.fdroid,7,"
    "0,,60000,f-droid.org,2024-11-02 09:00:00"
)


def test_parse_metadata_maps_columns() -> None:
    csv_text = "\n".join([_HEADER, _ROW_PLAY, _ROW_FDROID])
    records = list(parse_metadata(csv_text.splitlines()))
    assert len(records) == 2

    play = records[0]
    assert play.sha256 == "AAAA1111"
    assert play.pkg_name == "com.example.play"
    assert play.vercode == "42"
    assert play.apk_size == 123456
    assert play.year == 2025
    assert play.store_stratum == "google_play"

    assert records[1].store_stratum == "f_droid"
    assert records[1].year == 2024


def test_parse_metadata_skips_blank_lines() -> None:
    csv_text = "\n".join([_HEADER, _ROW_PLAY, "", _ROW_FDROID, ""])
    records = list(parse_metadata(csv_text.splitlines()))
    assert len(records) == 2


def test_download_url_contains_key_and_sha() -> None:
    client = AndroZooClient(api_key="SECRETKEY")
    url = client.download_url("ABC123")
    assert "apikey=SECRETKEY" in url
    assert "sha256=ABC123" in url
    assert url.startswith("https://androzoo.uni.lu/api/download")


def test_download_writes_and_verifies_checksum(tmp_path: Path) -> None:
    payload = b"fake apk bytes"
    digest = hashlib.sha256(payload).hexdigest()
    client = AndroZooClient(api_key="k", fetcher=lambda url: payload)

    dest = tmp_path / f"{digest}.apk"
    result = client.download(digest, dest)

    assert result == dest
    assert dest.read_bytes() == payload


def test_download_rejects_checksum_mismatch(tmp_path: Path) -> None:
    client = AndroZooClient(api_key="k", fetcher=lambda url: b"corrupted")
    # Ask for a sha256 that does not match the returned bytes.
    wrong_sha = "0" * 64
    dest = tmp_path / "out.apk"

    with pytest.raises(AndroZooError):
        client.download(wrong_sha, dest)
    assert not dest.exists()  # no partial/wrong file left behind


def test_download_is_idempotent_skips_existing(tmp_path: Path) -> None:
    payload = b"already here"
    digest = hashlib.sha256(payload).hexdigest()
    dest = tmp_path / f"{digest}.apk"
    dest.write_bytes(payload)

    calls: list[str] = []

    def _fetcher(url: str) -> bytes:
        calls.append(url)
        return payload

    client = AndroZooClient(api_key="k", fetcher=_fetcher)
    client.download(digest, dest)
    assert calls == []  # verified existing file; no re-fetch


def test_from_env_reads_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANDROZOO_API_KEY", "env-key-123")
    client = AndroZooClient.from_env()
    assert client.api_key == "env-key-123"


def test_from_env_missing_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANDROZOO_API_KEY", raising=False)
    with pytest.raises(AndroZooError):
        AndroZooClient.from_env()


def test_repr_does_not_leak_api_key() -> None:
    client = AndroZooClient(api_key="TOPSECRET")
    assert "TOPSECRET" not in repr(client)
