"""Tests for corpus orchestration (sample-from-metadata, batch download).

These exercise the logic behind the ``corpus`` CLI commands with an injected
fake AndroZoo client, so no network or API key is required.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from llm_apk_scanner.corpus.androzoo import AndroZooClient
from llm_apk_scanner.corpus.manifest import read_manifest
from llm_apk_scanner.corpus.runner import run_download, run_sample

_HEADER = "sha256,dex_date,apk_size,pkg_name,vercode,markets,added"


def _metadata_lines(n: int, *, year: int, market: str) -> list[str]:
    lines = [_HEADER]
    for i in range(n):
        lines.append(
            f"{i:064x},{year}-01-01 00:00:00,{1000 + i},com.example.a{i},{i},{market},"
            f"{year}-02-01 00:00:00"
        )
    return lines


def test_run_sample_writes_manifest(tmp_path: Path) -> None:
    lines = _metadata_lines(40, year=2025, market="play.google.com")
    out = tmp_path / "manifest.csv"
    result = run_sample(lines, total=10, seed=1, out_path=out)

    assert result.size == 10
    assert out.exists()
    assert len(read_manifest(out)) == 10


def _fake_client(store: dict[str, bytes]) -> AndroZooClient:
    def _fetch(url: str) -> bytes:
        # Extract sha256 from the query string and return its known bytes.
        sha = url.split("sha256=")[1]
        return store[sha]

    return AndroZooClient(api_key="k", fetcher=_fetch)


def test_run_download_fetches_all_records(tmp_path: Path) -> None:
    payloads = {f"{i:064x}": f"apk-{i}".encode() for i in range(3)}
    store = {hashlib.sha256(v).hexdigest(): v for v in payloads.values()}
    # The manifest SHA must equal the content SHA for verification to pass.
    lines = ["sha256,dex_date,apk_size,pkg_name,vercode,markets,added"]
    for sha in store:
        lines.append(f"{sha},2025-01-01 00:00:00,10,com.example,1,play.google.com,")
    manifest = tmp_path / "m.csv"
    run_sample(lines, total=3, seed=1, out_path=manifest)

    report = run_download(
        read_manifest(manifest),
        dest_dir=tmp_path / "corpus",
        client=_fake_client(store),
    )
    assert len(report.ok) == 3
    assert report.failed == []
    for sha in store:
        assert (tmp_path / "corpus" / f"{sha}.apk").exists()


def test_run_download_records_failures_and_continues(tmp_path: Path) -> None:
    good = b"good-apk"
    good_sha = hashlib.sha256(good).hexdigest()
    bad_sha = "f" * 64  # nothing will hash to this

    def _fetch(url: str) -> bytes:
        if f"sha256={good_sha}" in url:
            return good
        return b"junk-that-wont-match"

    client = AndroZooClient(api_key="k", fetcher=_fetch)
    records = read_manifest_from_shas(tmp_path, [good_sha, bad_sha])

    report = run_download(records, dest_dir=tmp_path / "c", client=client)
    assert report.ok == [good_sha]
    assert len(report.failed) == 1
    assert report.failed[0][0] == bad_sha


def test_run_download_respects_limit(tmp_path: Path) -> None:
    payloads = {f"{i:064x}": f"x-{i}".encode() for i in range(5)}
    store = {hashlib.sha256(v).hexdigest(): v for v in payloads.values()}
    records = read_manifest_from_shas(tmp_path, list(store.keys()))

    report = run_download(
        records, dest_dir=tmp_path / "c", client=_fake_client(store), limit=2
    )
    assert len(report.ok) == 2


def read_manifest_from_shas(tmp_path: Path, shas: list[str]):
    """Helper: build a manifest from raw SHA strings and read it back."""
    lines = ["sha256,dex_date,apk_size,pkg_name,vercode,markets,added"]
    for sha in shas:
        lines.append(f"{sha},2025-01-01 00:00:00,10,com.example,1,play.google.com,")
    manifest = tmp_path / f"m_{len(shas)}_{shas[0][:6]}.csv"
    run_sample(lines, total=len(shas), seed=1, out_path=manifest)
    return read_manifest(manifest)
