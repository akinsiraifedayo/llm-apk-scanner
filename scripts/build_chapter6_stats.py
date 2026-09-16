"""Build every statistic Chapter 6 reports, from the authoritative sources.

Authoritative sources, and why:

- ``corpus/ledger_v2.db``  — one row per APK, upserted, so it is the only
  non-duplicated record of what was scanned. 1,495 scanned, 5 download-failed.
- ``results/detail_v2/``   — one file per APK keyed by SHA-256, so a re-scan
  overwrites rather than appends. Holds the *last* scan of every application.
- ``results/findings_v2.jsonl`` is deliberately NOT used for counts. It is
  append-only and the interrupted-and-resumed runs left 38 duplicate rows in
  it (1,533 rows for 1,495 applications), four of which disagree with each
  other. Deduplicating it last-row-wins reproduces ``detail_v2`` exactly; that
  equivalence is asserted by ``verify_sources()`` below rather than assumed.
- ``corpus/frame_20260813.csv`` — the sampling frame, for the funnel and
  dating counts of §6.1.

Output: ``validation/chapter6_stats.json``.
"""

from __future__ import annotations

import csv
import glob
import gzip
import json
import math
import re
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

LEDGER = "corpus/ledger_v2.db"
DETAIL = "results/detail_v2"
FINDINGS_JSONL = "results/findings_v2.jsonl"
FRAME = "corpus/frame_20260813.csv"
AUDIT = "audit_full.txt"
PROMPT_CLASSES = "validation/prompt_classes.json"
LIVENESS = "validation/liveness_results.json"
OUTPUT = "validation/chapter6_stats.json"

#: Artefact classes that answer RQ1. ``llm_endpoint`` and ``llm_sdk_reference``
#: are population-identification evidence, not exposed artefacts, so they are
#: counted and reported but never folded into the RQ1 numerator.
RQ1_KINDS = ("secret", "system_prompt", "tool_schema", "rag_artefact")

DEX_RE = re.compile(r"^classes\d*\.dex$")

#: Totals printed by ``corpus frame`` over the frozen index (Appendix A.1). The
#: frame keeps neither, so they are recorded here rather than recomputed from
#: the 3.5 GB snapshot on every run.
INDEX_ROWS = 27_589_444
CANDIDATE_ROWS = 68_419

#: The sampling window, as ``llm_apk_scanner.corpus.sample.DEFAULT_YEARS``.
ELIGIBLE_YEARS = ("2024", "2025", "2026")


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval. Preferred over the normal approximation because
    several proportions here sit near 0 and would otherwise take a negative
    lower bound (Brown et al., 2001)."""
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = z * math.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def stat(k: int, n: int) -> dict:
    lo, hi = wilson(k, n)
    return {
        "apps": k,
        "rate": round(k / n, 4) if n else 0.0,
        "wilson_lo": round(lo, 4),
        "wilson_hi": round(hi, 4),
        "ci_width": round(hi - lo, 4),
    }


def load_details(llm_sha: set[str]) -> dict[str, dict]:
    """Return {sha256: detail} for the LLM-integrated applications."""
    out = {}
    for path in sorted(glob.glob(f"{DETAIL}/*.json.gz")):
        sha = Path(path).name[: -len(".json.gz")].lower()
        if sha in llm_sha:
            out[sha] = json.load(gzip.open(path, "rt"))
    return out


def verify_sources(db: sqlite3.Connection, details: dict[str, dict]) -> dict:
    """Assert that detail_v2 and the deduplicated JSONL tell the same story.

    This is the check that establishes detail_v2 as authoritative. If it ever
    fails, the two result sets have diverged and no number below is safe.
    """
    rows = [json.loads(line) for line in open(FINDINGS_JSONL)]
    dedup: dict[str, dict] = {}
    for r in rows:
        dedup[r["sha256"].lower()] = r  # last scan wins

    ledger_scanned = db.execute(
        "select count(*) from apk where state='scanned'"
    ).fetchone()[0]
    jsonl_llm = {s for s, r in dedup.items() if r.get("llm_integrated")}
    jsonl_sev: Counter = Counter()
    for sha in jsonl_llm:
        jsonl_sev.update(dedup[sha].get("by_severity") or {})

    detail_sev: Counter = Counter()
    for d in details.values():
        detail_sev.update(f["severity"] for f in d["findings"])

    agree = dict(jsonl_sev) == dict(detail_sev)
    assert len(dedup) == ledger_scanned, "JSONL unique rows != ledger scanned"
    assert jsonl_llm == set(details), "JSONL and detail_v2 disagree on which apps are LLM"
    assert agree, f"severity mismatch: jsonl={dict(jsonl_sev)} detail={dict(detail_sev)}"

    return {
        "jsonl_rows_raw": len(rows),
        "jsonl_rows_unique": len(dedup),
        "jsonl_duplicate_rows": len(rows) - len(dedup),
        "ledger_scanned": ledger_scanned,
        "detail_files_for_llm_apps": len(details),
        "severity_agrees": agree,
    }


def frame_summary() -> dict:
    """The acquisition funnel and dating composition of §6.1."""
    with open(FRAME, encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    eligible = [r for r in rows if r["year"] in ELIGIBLE_YEARS]
    return {
        "index_rows": INDEX_ROWS,
        "candidate_rows": CANDIDATE_ROWS,
        "unique_applications": len(rows),
        "dated_by": dict(Counter(r["date_source"] for r in rows)),
        "eligible": len(eligible),
        "eligible_dated_by": dict(Counter(r["date_source"] for r in eligible)),
    }


def filter_missed(db: sqlite3.Connection) -> tuple[set[str], set[str]]:
    """Applications the confirmation rule rejected that ship a credential (§6.3.1).

    A provider credential is not confirming evidence under §4.3, so these sit
    outside the population. Returns the applications with any
    provider-attributed credential and, among them, those whose only such
    credential is an ``AKIA…`` identifier, which names AWS rather than an
    inference service and is excluded for the reason §4.4 excludes an
    uncorroborated ``AIza…`` key.
    """
    rejected = {
        r[0].lower()
        for r in db.execute(
            "select sha256 from apk where state='scanned' and llm_integrated=0"
        )
    }
    attributed: set[str] = set()
    aws_only: set[str] = set()
    for sha, d in load_details(rejected).items():
        providers = {
            f.get("provider")
            for f in d["findings"]
            if f["kind"] == "secret" and f.get("provider", "generic") != "generic"
        }
        if providers:
            attributed.add(sha)
            if providers == {"aws_bedrock"}:
                aws_only.add(sha)
    return attributed, aws_only


def parse_audit() -> dict:
    """Tally the adversarial-audit transcript.

    Each application contributes one verdict line. The three verdicts are
    mutually exclusive and exhaustive, so they must sum to the applications
    audited; the assertion below enforces that rather than trusting the grep.
    """
    if not Path(AUDIT).exists():
        return {}
    text = Path(AUDIT).read_text(errors="ignore")
    header = re.compile(
        r"^(?P<pkg>\S+)\s+\[scanner: (?P<n>\d+) findings, llm=(?P<llm>[01])\]\s+(?P<verdict>.*)$",
        re.M,
    )
    audited = agree = weak = empty_signal = llm_audited = 0
    for m in header.finditer(text):
        audited += 1
        llm_audited += int(m.group("llm"))
        v = m.group("verdict")
        if "SCANNER SAID EMPTY" in v:
            empty_signal += 1
        elif "AGREE" in v:
            agree += 1
        else:
            weak += 1
    assert agree + weak + empty_signal == audited, "audit verdicts do not partition"
    return {
        "applications_audited": audited,
        "llm_integrated_audited": llm_audited,
        "agreements": agree,
        "scanner_found_audit_weak": weak,
        "scanner_empty_audit_signal": empty_signal,
    }


def main() -> None:
    db = sqlite3.connect(LEDGER)
    scanned = db.execute("select count(*) from apk where state='scanned'").fetchone()[0]
    failed = db.execute(
        "select count(*) from apk where state='download_failed'"
    ).fetchone()[0]
    llm_sha = {
        r[0].lower()
        for r in db.execute(
            "select sha256 from apk where state='scanned' and llm_integrated=1"
        )
    }
    n = len(llm_sha)

    details = load_details(llm_sha)
    sources = verify_sources(db, details)

    # ---- findings ---------------------------------------------------------
    severity: Counter = Counter()
    kinds: Counter = Counter()
    confidence: Counter = Counter()
    apps_by_kind: dict[str, set] = defaultdict(set)
    apps_by_provider: dict[str, set] = defaultdict(set)
    cred_apps: set[str] = set()
    other_secret_apps: set[str] = set()
    apps_with_nondex_finding: set[str] = set()
    findings_by_container: Counter = Counter()

    for sha, d in details.items():
        for p in d.get("detected_providers", []):
            apps_by_provider[p].add(sha)
        for f in d["findings"]:
            severity[f["severity"]] += 1
            kinds[f["kind"]] += 1
            apps_by_kind[f["kind"]].add(sha)
            c = f.get("confidence")
            if c is not None:
                confidence["high" if c >= 0.9 else "medium" if c >= 0.7 else "low"] += 1
            if f["kind"] == "secret":
                (cred_apps if f.get("provider", "generic") != "generic" else other_secret_apps).add(sha)

            fp = (f.get("location") or {}).get("file_path") or ""
            base = fp.split("/")[-1]
            top = fp.split("/")[0] if "/" in fp else fp
            findings_by_container[top] += 1
            if fp and not DEX_RE.match(base):
                apps_with_nondex_finding.add(sha)

    total_findings = sum(severity.values())

    # ---- prompt classification -------------------------------------------
    pc = {
        a["sha256"].lower(): a
        for a in json.load(open(PROMPT_CLASSES))
        if "error" not in a
    }
    sysprompt_apps = {s for s, v in pc.items() if v.get("has_system_prompt")}
    preset_only_apps = {
        s for s, v in pc.items() if v.get("has_preset") and not v.get("has_system_prompt")
    }

    # ---- RQ1 ---------------------------------------------------------------
    rq1_apps = (
        cred_apps
        | sysprompt_apps
        | apps_by_kind.get("tool_schema", set())
        | apps_by_kind.get("rag_artefact", set())
    )
    # Applications confirmed LLM-integrated that expose no client-side RQ1
    # artefact: the server-proxied population bounded in Ch. 6 §6.6.
    server_proxied = llm_sha - rq1_apps

    prevalence = {
        "denominator": n,
        "inference_credential": stat(len(cred_apps), n),
        "other_secret": stat(len(other_secret_apps), n),
        "system_prompt": stat(len(sysprompt_apps), n),
        "preset_only": stat(len(preset_only_apps), n),
        "tool_schema": stat(len(apps_by_kind.get("tool_schema", set())), n),
        "rag_artefact": stat(len(apps_by_kind.get("rag_artefact", set())), n),
        "llm_endpoint": stat(len(apps_by_kind.get("llm_endpoint", set())), n),
        "any_rq1_artefact": stat(len(rq1_apps), n),
        "no_rq1_artefact_server_proxied": stat(len(server_proxied), n),
        "finding_outside_dex": stat(len(apps_with_nondex_finding), n),
    }

    cooccur = cred_apps & sysprompt_apps
    play_llm = db.execute(
        "select count(*) from apk where state='scanned' and llm_integrated=1 and store='google_play'"
    ).fetchone()[0]

    # ---- §6.3.1: the confirmation rule's blind spot -----------------------
    attributed, aws_only = filter_missed(db)
    missed = len(attributed - aws_only)
    widened = n + missed
    sensitivity = {
        "rejected_with_provider_credential": len(attributed),
        "aws_identifier_only": len(aws_only),
        "unambiguous_credential": missed,
        "denominator": widened,
        # Every missed application ships a credential, so it joins both
        # numerators below as well as the denominator.
        "integration": stat(widened, scanned),
        "inference_credential": stat(len(cred_apps) + missed, widened),
        "any_rq1_artefact": stat(len(rq1_apps) + missed, widened),
    }

    liveness = json.load(open(LIVENESS)) if Path(LIVENESS).exists() else None

    stats = {
        "generated": "2026-09-16",
        "sources": sources,
        "frame": frame_summary(),
        "corpus": {
            "total_sampled": scanned + failed,
            "scanned": scanned,
            "download_failed": failed,
            "download_failure_rate": round(failed / (scanned + failed), 4),
            "llm_integrated": n,
            "non_llm": scanned - n,
            "llm_rate": round(n / scanned, 4),
        },
        "by_stratum": {
            f"{y}_{s}": c
            for y, s, c in db.execute(
                "select year, store, count(*) from apk where state='scanned' "
                "group by year, store order by year, store"
            )
        },
        "integration_by_year": {
            str(y): {"scanned": t, **stat(k, t)}
            for y, t, k in db.execute(
                "select year, count(*), sum(llm_integrated=1) from apk "
                "where state='scanned' group by year order by year"
            )
        },
        "findings": {
            "total": total_findings,
            "per_app_mean": round(total_findings / n, 2),
            "by_kind": dict(kinds),
            "apps_by_kind": {k: len(v) for k, v in apps_by_kind.items()},
            "by_severity": dict(severity),
            "by_confidence_tier": dict(confidence),
            "by_top_level_container": dict(findings_by_container.most_common()),
        },
        "prevalence": prevalence,
        "sensitivity": sensitivity,
        "provider_distribution": {
            p: {"apps": len(a), "share": round(len(a) / n, 4)}
            for p, a in sorted(apps_by_provider.items(), key=lambda x: -len(x[1]))
        },
        "store_breakdown": {"google_play_llm": play_llm, "other_llm": n - play_llm},
        "prompt_classification": {
            "apps_with_prompt_finding": len(apps_by_kind.get("system_prompt", set())),
            "apps_with_system_prompt": len(sysprompt_apps),
            "apps_with_preset_only": len(preset_only_apps),
            "apps_with_confidentiality_marker": sum(
                1 for v in pc.values() if v.get("any_confidentiality_marker")
            ),
            "apps_with_role_binding": sum(
                1 for v in pc.values() if v.get("any_role_binding")
            ),
            "apps_with_white_labelling": sum(
                1 for v in pc.values() if v.get("any_white_labelling")
            ),
        },
        "co_occurrence": {
            "credential_and_system_prompt": len(cooccur),
            "share_of_credential_apps": round(len(cooccur) / len(cred_apps), 4),
        },
        "adversarial_audit": parse_audit(),
        "liveness": liveness,
        "liveness_applications": {
            "with_live_credential": len(
                {o["package"] for o in liveness["outcomes"] if o["state"] == "live"}
            ),
            "with_disclosable_credential": len(
                {o["package"] for o in liveness["to_disclose"]}
            ),
        } if liveness else None,
    }

    Path(OUTPUT).write_text(json.dumps(stats, indent=2))
    print(f"Wrote {OUTPUT}\n")
    print("SOURCE VERIFICATION")
    for k, v in sources.items():
        print(f"  {k}: {v}")
    print(f"\nCORPUS  {scanned} scanned, {failed} failed, {n} LLM-integrated "
          f"({stats['corpus']['llm_rate']*100:.1f}%)")
    print(f"FINDINGS in LLM apps: {total_findings}  "
          f"({stats['findings']['per_app_mean']}/app)")
    print(f"  severity: {dict(severity)}")
    print("\nPREVALENCE")
    for k, v in prevalence.items():
        if k == "denominator":
            continue
        print(f"  {k:34} {v['apps']:>3}/{n} = {v['apps']/n*100:5.1f}% "
              f"[{v['wilson_lo']*100:.1f}, {v['wilson_hi']*100:.1f}]")
    print(f"\nSENSITIVITY  {len(attributed)} rejected apps ship a provider credential, "
          f"{len(aws_only)} only an AWS identifier; n = {n} -> {widened}, "
          f"any RQ1 artefact {sensitivity['any_rq1_artefact']['rate']*100:.1f}%")
    print(f"\nAUDIT  {stats['adversarial_audit']}")
    print(f"CO-OCCURRENCE  {len(cooccur)} of {len(cred_apps)} "
          f"({len(cooccur)/len(cred_apps)*100:.1f}%)")


if __name__ == "__main__":
    main()
