"""Run credential liveness validation over all provider-attributed credentials.

Executes live introspection (with user authorization) to determine:
  found / live / revoked / unverifiable

The evidence field in detail files is truncated to 60 chars. Per HANDOFF.md,
validation must use full strings extracted from the APKs, never truncated
evidence. This script re-extracts credentials using the secrets scanner.

Output goes to validation/liveness_results.json for Chapter 6 reporting.

Usage:
    python scripts/run_liveness_validation.py [--dry-run]
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
import re
import sqlite3
from pathlib import Path

from llm_apk_scanner.apk_loader import load_apk
from llm_apk_scanner.models import Provider
from llm_apk_scanner.scanners.secrets import SecretScanner
from llm_apk_scanner.validators.runner import (
    CredentialRecord,
    ValidationSummary,
    render_report,
    unvalidatable_providers,
    validate_credentials,
)

LEDGER = "corpus/ledger_v2.db"
DETAIL = "results/detail_v2"
STAGING = "corpus/staging"
OUTPUT = "validation/liveness_results.json"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't make real network calls, just report what would be checked",
    )
    args = ap.parse_args()

    db = sqlite3.connect(LEDGER)
    llm = {
        r[0]
        for r in db.execute(
            "select sha256 from apk where state='scanned' and llm_integrated=1"
        )
    }
    pkgs = dict(db.execute("select sha256, pkg_name from apk"))
    providers_by_sha = dict(db.execute("select sha256, providers from apk where llm_integrated=1"))

    # Map provider string -> Provider enum
    provider_map = {p.value: p for p in Provider}

    # Identify which apps have secret findings we need to re-extract
    apps_with_secrets: list[tuple[str, set[str]]] = []
    for path in sorted(glob.glob(f"{DETAIL}/*.json.gz")):
        sha = Path(path).name[: -len(".json.gz")]
        if sha not in llm:
            continue

        detail = json.load(gzip.open(path, "rt"))
        secret_providers = set()
        for finding in detail["findings"]:
            if finding["kind"] != "secret":
                continue
            prov_str = finding.get("provider", "generic")
            if prov_str != "generic":
                secret_providers.add(prov_str)
        if secret_providers:
            apps_with_secrets.append((sha, secret_providers))

    print(f"Apps with provider-attributed secrets: {len(apps_with_secrets)}")

    # Re-extract credentials from APKs using the secrets scanner
    scanner = SecretScanner()
    records: list[CredentialRecord] = []
    seen_keys: set[str] = set()

    for i, (sha, expected_providers) in enumerate(apps_with_secrets, 1):
        apk_path = f"{STAGING}/{sha}.apk"
        if not Path(apk_path).exists():
            print(f"  [{i}/{len(apps_with_secrets)}] {pkgs.get(sha)}: APK NOT FOUND")
            continue

        corpus = load_apk(apk_path)
        sources = list(corpus)
        findings = scanner.scan(sources)

        for finding in findings:
            provider = finding.provider
            if provider == Provider.GENERIC:
                continue

            key = finding.evidence
            if key in seen_keys:
                continue
            seen_keys.add(key)

            records.append(
                CredentialRecord(
                    provider=provider,
                    key=key,
                    sha256=sha,
                    package=pkgs.get(sha, ""),
                )
            )

        print(f"  [{i}/{len(apps_with_secrets)}] {pkgs.get(sha, sha[:12])}: {len(findings)} secrets, {len([f for f in findings if f.provider != Provider.GENERIC])} attributed")

    print(f"Found {len(records)} unique provider-attributed credentials in {len(llm)} LLM apps")

    # Show which providers cannot be validated
    unval = unvalidatable_providers(records)
    if unval:
        print(f"Providers without introspection endpoint (will abstain): {[p.value for p in unval]}")

    # Group by provider for reporting
    by_prov: dict[str, int] = {}
    for r in records:
        by_prov[r.provider.value] = by_prov.get(r.provider.value, 0) + 1
    print("By provider:", by_prov)

    if args.dry_run:
        print("\n[DRY RUN MODE - no network calls]\n")

    # Run validation
    summary = validate_credentials(records, live=not args.dry_run, timeout_seconds=30.0)

    # Print report
    print("\n" + render_report(summary))

    # Build output for Chapter 6
    output = {
        "run_mode": "dry_run" if args.dry_run else "live",
        "total_credentials": summary.total,
        "live": summary.live,
        "revoked": summary.revoked,
        "unknown": summary.unknown,
        "live_rate": summary.live_rate if (summary.live + summary.revoked) else None,
        "by_provider": {},
        "outcomes": [],
    }

    for outcome in summary.outcomes:
        prov = outcome.record.provider.value
        if prov not in output["by_provider"]:
            output["by_provider"][prov] = {"found": 0, "live": 0, "revoked": 0, "unknown": 0}
        output["by_provider"][prov]["found"] += 1
        if outcome.valid is True:
            output["by_provider"][prov]["live"] += 1
        elif outcome.valid is False:
            output["by_provider"][prov]["revoked"] += 1
        else:
            output["by_provider"][prov]["unknown"] += 1

        output["outcomes"].append({
            "provider": prov,
            "fingerprint": outcome.record.fingerprint,
            "package": outcome.record.package,
            "state": outcome.state,
            "http_status": outcome.http_status,
            "note": outcome.note,
            "checked_at": outcome.checked_at,
        })

    # Identify credentials needing disclosure
    to_disclose = summary.to_disclose()
    output["to_disclose_count"] = len(to_disclose)
    output["to_disclose"] = [
        {
            "provider": o.record.provider.value,
            "fingerprint": o.record.fingerprint,
            "package": o.record.package,
            "state": o.state,
        }
        for o in to_disclose
    ]

    Path(OUTPUT).parent.mkdir(parents=True, exist_ok=True)
    Path(OUTPUT).write_text(json.dumps(output, indent=2))
    print(f"\nWrote {OUTPUT}")

    # Summary for Chapter 6
    print("\n=== Chapter 6 Table 6.9 data ===")
    print(f"Total credentials found: {summary.total}")
    print(f"  Live: {summary.live}")
    print(f"  Revoked: {summary.revoked}")
    print(f"  Unverifiable: {summary.unknown}")
    if summary.live + summary.revoked > 0:
        print(f"  Live rate: {100 * summary.live_rate:.1f}% (denominator: live + revoked)")
    print(f"\nCredentials warranting disclosure: {len(to_disclose)}")


if __name__ == "__main__":
    main()
