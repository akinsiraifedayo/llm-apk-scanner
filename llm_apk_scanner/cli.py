"""
Command-line interface.

Usage:
    llm-apk-scanner scan APK [-o OUTPUT] [--format json|md]
    llm-apk-scanner filter DIR [--min-signals N]
    llm-apk-scanner pipeline DIR -o OUTPUT_DIR

The CLI is deliberately simple. All heavy lifting is in
`scan.py` so that the same code path is exercised by tests.
"""

from __future__ import annotations

import gzip
import csv
import io
import json
import os
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from llm_apk_scanner import __version__
from llm_apk_scanner.corpus.androzoo import (
    AndroZooClient,
    AndroZooError,
    make_fetcher,
    parse_metadata,
)
from llm_apk_scanner.corpus.frame import (
    ai_signals,
    deduplicate_by_package,
    stream_androzoo_index,
    write_frame,
)
from llm_apk_scanner.corpus.ledger import (
    DOWNLOAD_FAILED,
    SCAN_FAILED,
    SCANNED,
    Ledger,
)
from llm_apk_scanner.corpus.manifest import read_manifest
from llm_apk_scanner.corpus.runner import run_download, run_sample
from llm_apk_scanner.corpus.stream import run_stream
from llm_apk_scanner.models import Provider
from llm_apk_scanner.reporters import HtmlReporter, JsonReporter, MarkdownReporter
from llm_apk_scanner.scan import scan_apk
from llm_apk_scanner.stats import (
    aggregate as _aggregate,
)
from llm_apk_scanner.stats import (
    load_results_dir,
    write_csv,
    write_json,
)
from llm_apk_scanner.validation import LabelStore, compute_metrics
from llm_apk_scanner.validation.sample import (
    draw_validation_sample,
    read_findings,
    read_key,
    write_sample,
)
from llm_apk_scanner.validators import (
    CredentialRecord,
    render_report,
    unvalidatable_providers,
    validate_credentials,
)

#: A corpus run lasting days is normally piped into a log file, and Python
#: block-buffers a non-TTY stdout — progress then appears only when the process
#: ends, which is useless for watching a long run. Writing through a
#: write-through wrapper keeps `tee` and `>` live. Rich still auto-detects
#: whether the underlying stream is a terminal for styling purposes.
console = Console(file=io.TextIOWrapper(sys.stdout.buffer, write_through=True))


@click.group()
@click.version_option(__version__, prog_name="llm-apk-scanner")
def main() -> None:
    """Empirical security analysis of LLM-integrated Android applications."""


@main.command()
@click.argument("apk_path", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "-o", "--output", type=click.Path(), default=None,
    help="Write report to file instead of stdout.",
)
@click.option(
    "--format", "fmt", type=click.Choice(["json", "md", "html"]),
    default="md", show_default=True,
)
def scan(apk_path: str, output: str | None, fmt: str) -> None:
    """Scan a single APK and emit a report."""
    try:
        result = scan_apk(apk_path)
    except ImportError as e:
        console.print(f"[red]Cannot scan APK: {e}[/red]")
        sys.exit(2)

    if fmt == "json":
        rendered = JsonReporter().render(result)
    elif fmt == "html":
        rendered = HtmlReporter().render(result)
    else:
        rendered = MarkdownReporter().render(result)
    if output:
        Path(output).write_text(rendered, encoding="utf-8")
        console.print(f"[green]Wrote report:[/green] {output}")
    else:
        click.echo(rendered)

    _print_summary(result)


@main.command()
@click.argument("apk_dir", type=click.Path(exists=True, file_okay=False))
@click.option(
    "--min-signals", type=int, default=2, show_default=True,
    help="Minimum unique LLM signals required to flag an app.",
)
def filter(apk_dir: str, min_signals: int) -> None:
    """List APKs in DIR that appear to integrate LLM features."""
    paths = sorted(_walk_apks(apk_dir))
    table = Table(title=f"LLM-integrated APKs (of {len(paths)} scanned)")
    table.add_column("Package")
    table.add_column("Providers")
    table.add_column("Path", overflow="fold")
    integrated = 0
    for p in paths:
        try:
            result = scan_apk(str(p))
        except Exception as e:
            console.print(f"[yellow]Skip {p}: {e}[/yellow]")
            continue
        if result.is_llm_integrated:
            integrated += 1
            providers = ",".join(pr.value for pr in result.detected_providers)
            table.add_row(result.apk.package_name, providers or "-", str(p))
    console.print(table)
    console.print(
        f"[bold]{integrated}/{len(paths)}[/bold] APKs flagged as LLM-integrated"
    )


@main.command()
@click.argument("apk_dir", type=click.Path(exists=True, file_okay=False))
@click.option(
    "-o", "--output-dir", type=click.Path(), required=True,
    help="Directory for per-APK JSON and Markdown reports.",
)
def pipeline(apk_dir: str, output_dir: str) -> None:
    """Filter and scan a corpus, writing per-APK reports."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    aggregate: list[dict] = []
    for p in sorted(_walk_apks(apk_dir)):
        try:
            result = scan_apk(str(p))
        except Exception as e:
            console.print(f"[yellow]Skip {p}: {e}[/yellow]")
            continue
        if not result.is_llm_integrated:
            continue
        slug = result.apk.package_name or p.stem
        (out / f"{slug}.md").write_text(
            MarkdownReporter().render(result), encoding="utf-8"
        )
        (out / f"{slug}.json").write_text(
            JsonReporter().render(result), encoding="utf-8"
        )
        aggregate.append(result.to_dict())
    (out / "_aggregate.json").write_text(
        json.dumps(aggregate, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    console.print(
        f"[green]Pipeline complete:[/green] {len(aggregate)} reports in {out}"
    )


@main.command(name="aggregate")
@click.argument(
    "results_dir", type=click.Path(exists=True, file_okay=False)
)
@click.option(
    "-o", "--output-dir", type=click.Path(), required=True,
    help="Directory to write aggregate.json and aggregate.csv.",
)
def aggregate_cmd(results_dir: str, output_dir: str) -> None:
    """Aggregate per-APK results into corpus-level statistics."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    results = load_results_dir(results_dir)
    agg = _aggregate(results)
    write_json(agg, out / "aggregate.json")
    write_csv(agg, out / "aggregate.csv")
    console.print(
        f"[green]Aggregate complete:[/green] "
        f"{agg.n_apks_llm_integrated}/{agg.n_apks_total} LLM-integrated APKs. "
        f"Wrote {out}/aggregate.json and aggregate.csv"
    )


@main.group()
def corpus() -> None:
    """Build the AndroZoo sample frame (draw a manifest, download APKs)."""


@corpus.command(name="sample")
@click.option(
    "--metadata", "metadata_path", type=click.Path(exists=True, dir_okay=False),
    required=True, help="AndroZoo metadata index (e.g. latest.csv).",
)
@click.option("--total", type=int, default=10000, show_default=True,
              help="Target sample size.")
@click.option("--seed", type=int, default=20260714, show_default=True,
              help="RNG seed for a reproducible draw.")
@click.option("--years", type=str, default="2020-2026", show_default=True,
              help="Year range for eligible apps (e.g. '2020-2026' or '2019,2020,2021').")
@click.option("-o", "--output", type=click.Path(), default="corpus/manifest.csv",
              show_default=True, help="Where to write the manifest CSV.")
def corpus_sample(metadata_path: str, total: int, seed: int, years: str, output: str) -> None:
    """Draw a two-stage stratified sample and write the SHA-256 manifest."""
    # Parse years: "2020-2026" or "2019,2020,2021"
    if "-" in years:
        start, end = years.split("-")
        year_list = tuple(range(int(start), int(end) + 1))
    else:
        year_list = tuple(int(y.strip()) for y in years.split(","))

    out = Path(output)
    with open(metadata_path, encoding="utf-8") as handle:
        result = run_sample(handle, total=total, seed=seed, years=year_list, out_path=out)
    table = Table(title="Sample allocation (year, store)")
    table.add_column("Stratum")
    table.add_column("APKs", justify="right")
    for stratum in sorted(result.allocation):
        table.add_row(str(stratum), str(result.allocation[stratum]))
    console.print(table)
    console.print(
        f"[green]Sampled[/green] {result.size} APKs "
        f"(requested {total}); manifest → {out}"
    )


@corpus.command(name="download")
@click.option(
    "--manifest", "manifest_path", type=click.Path(exists=True, dir_okay=False),
    default="corpus/manifest.csv", show_default=True,
    help="Manifest CSV produced by 'corpus sample'.",
)
@click.option("-o", "--output-dir", type=click.Path(), default="corpus/apks",
              show_default=True, help="Directory to download APKs into.")
@click.option("--limit", type=int, default=None,
              help="Only download the first N APKs (for pilots).")
@click.option("--delay", type=float, default=1.0, show_default=True,
              help="Seconds to wait between downloads (rate-limiting).")
def corpus_download(
    manifest_path: str, output_dir: str, limit: int | None, delay: float
) -> None:
    """Download the sampled APKs from AndroZoo, verifying each checksum."""
    try:
        client = AndroZooClient.from_env()
    except AndroZooError as exc:
        console.print(f"[red]{exc}[/red]")
        sys.exit(2)

    records = read_manifest(Path(manifest_path))
    n_to_get = len(records) if limit is None else min(limit, len(records))
    console.print(f"Downloading {n_to_get} APKs (delay={delay}s between)…")

    def _progress(sha: str, ok: bool) -> None:
        mark = "[green]ok[/green]" if ok else "[red]FAIL[/red]"
        console.print(f"  {mark} {sha[:16]}…")

    report = run_download(
        records, dest_dir=Path(output_dir), client=client,
        limit=limit, delay=delay, on_progress=_progress,
    )
    console.print(
        f"[bold]Done:[/bold] {report.n_ok} downloaded, {report.n_failed} failed."
    )
    if report.failed:
        console.print("[yellow]Failures logged (sha, error):[/yellow]")
        for sha, err in report.failed:
            console.print(f"  {sha[:16]}… {err}")


@corpus.command(name="frame")
@click.option(
    "--index", "index_path", type=click.Path(exists=True, dir_okay=False), default=None,
    help="Local AndroZoo index CSV. Omit to stream latest.csv.gz from AndroZoo.",
)
@click.option("-o", "--output", type=click.Path(), default="corpus/frame.csv",
              show_default=True, help="Where to write the AI-candidate frame.")
@click.option("--per-apk", is_flag=True,
              help="Keep every version of each app. Default is one row per "
                   "package (newest version), making the app the unit of analysis.")
def corpus_frame(index_path: str | None, output: str, per_apk: bool) -> None:
    """Filter the AndroZoo index down to AI-candidate apps.

    This builds the *population* the sample is later drawn from. Run it once;
    the resulting frame.csv is the auditable record of what was eligible.

    By default the frame holds one row per package, so RQ1's denominator is
    apps rather than APKs. Pass --per-apk to keep every indexed version.
    """
    out = Path(output)

    # Filter on the fly — only AI candidates touch RAM.
    candidates: list[tuple] = []
    total_rows = 0

    def _consume(source) -> None:
        """Drain a record stream into ``candidates``.

        This must run *inside* whatever context manager owns the underlying
        file. ``parse_metadata`` is a generator, so binding it to a name
        inside a ``with`` block and iterating after the block has exited
        raises ``ValueError: I/O operation on closed file`` — the generator
        is only consumed on iteration, by which point the handle is gone.
        """
        nonlocal total_rows
        for record in source:
            total_rows += 1
            if total_rows % 1_000_000 == 0:
                console.print(
                    f"  …processed {total_rows:,} rows, {len(candidates):,} candidates"
                )
            signals = ai_signals(record.pkg_name)
            if signals:
                candidates.append((record, signals))

    if index_path:
        # AndroZoo ships the index gzipped (~3.3GB compressed, ~25GB plain),
        # so reading the archive directly avoids decompressing it to disk.
        opener = gzip.open if str(index_path).endswith(".gz") else open
        console.print(f"Reading local index {index_path}; filtering as we go…")
        with opener(index_path, "rt", encoding="utf-8", errors="replace") as handle:
            _consume(parse_metadata(handle))
    else:
        console.print("Streaming AndroZoo index (~3.5GB gzipped); filtering as we go…")
        _consume(stream_androzoo_index())

    console.print(f"  …done: {total_rows:,} rows → {len(candidates):,} AI candidates")

    rows = candidates if per_apk else list(deduplicate_by_package(candidates))
    n = write_frame(rows, out)

    unit = "APKs (every version)" if per_apk else "apps (newest version each)"
    console.print(
        f"[green]Frame built:[/green] {n} {unit} from {len(candidates)} AI-candidate "
        f"APKs across {total_rows} indexed rows → {out}"
    )
    if not per_apk and len(candidates) != n:
        console.print(
            f"  Collapsed {len(candidates) - n} duplicate-package rows "
            f"({100 * (len(candidates) - n) / len(candidates):.1f}% of candidates)."
        )


@corpus.command(name="run")
@click.option(
    "--manifest", "manifest_path", type=click.Path(exists=True, dir_okay=False),
    default="corpus/manifest.csv", show_default=True,
    help="Manifest CSV produced by 'corpus sample'.",
)
@click.option("--ledger", "ledger_path", type=click.Path(), default="corpus/ledger.db",
              show_default=True, help="Durable run state (enables resume).")
@click.option("--limit", type=int, default=None,
              help="Only process the next N pending APKs.")
@click.option("--workers", type=int, default=None,
              help="Scan processes (default: CPU count, capped at 8).")
@click.option("--download-workers", type=int, default=3, show_default=True,
              help="Concurrent downloads; keep low to stay polite to AndroZoo.")
@click.option("--max-staged", type=int, default=16, show_default=True,
              help="Hard cap on APKs on disk at once. Peak disk = this x APK size.")
@click.option("--delay", type=float, default=0.5, show_default=True,
              help="Seconds between downloads, per downloader thread.")
@click.option("--keep-apks", is_flag=True,
              help="Keep every APK after scanning. Needs ~164GB for a full 10k run.")
@click.option("--keep-with-findings", is_flag=True,
              help="Keep only APKs that produced a finding or are LLM-integrated. "
                   "Enough to re-scan after a detector fix, or to inspect "
                   "findings by hand, without retaining the whole corpus.")
@click.option("--keep-negatives", type=int, default=0, show_default=True,
              help="Also keep this many APKs that produced NO findings, so "
                   "false negatives can be audited. Retaining only positives "
                   "makes a missed detection invisible by construction.")
@click.option("--keep-budget-gb", type=float, default=20.0, show_default=True,
              help="Cap on disk retained by --keep-with-findings. Once reached, "
                   "further APKs are discarded and the run continues.")
@click.option("--findings", "findings_path", type=click.Path(),
              default="results/findings.jsonl", show_default=True,
              help="Append-only JSONL summary. Point a trial run elsewhere so "
                   "it cannot contaminate the real corpus results.")
@click.option("--detail-dir", type=click.Path(), default="results/detail",
              show_default=True, help="Gzipped full findings for interesting APKs.")
@click.option("--staging-dir", type=click.Path(), default="corpus/staging",
              show_default=True, help="Transient APK area; emptied as the run proceeds.")
@click.option("--download-timeout", type=float, default=None,
              help="Hard wall-clock limit per download, in seconds. Lower it if "
                   "AndroZoo stalls connections often. [default: 180]")
def corpus_run(
    manifest_path: str, ledger_path: str, limit: int | None, workers: int | None,
    download_workers: int, max_staged: int, delay: float, keep_apks: bool,
    keep_with_findings: bool, keep_negatives: int, keep_budget_gb: float,
    findings_path: str, detail_dir: str, staging_dir: str,
    download_timeout: float | None,
) -> None:
    """Download, scan and discard each APK in the manifest.

    Resumable: state lives in the ledger, so re-running continues where the
    last run stopped rather than starting over. APKs are deleted after
    scanning unless --keep-apks is given, which is what keeps a 10,000-APK
    run inside a disk budget far smaller than the corpus.
    """
    try:
        client = AndroZooClient.from_env()
    except AndroZooError as exc:
        console.print(f"[red]{exc}[/red]")
        sys.exit(2)

    if download_timeout is not None:
        # Wrap the default fetcher so the deadline reaches urlopen. Without a
        # total deadline a stalled connection blocks its downloader thread
        # forever; with three threads that deadlocks the whole run.
        client.fetcher = make_fetcher(total_timeout=download_timeout)

    with Ledger(Path(ledger_path)) as ledger:
        added = ledger.enroll(read_manifest(Path(manifest_path)))
        todo = ledger.pending(limit=limit)
        if not todo:
            console.print("[green]Nothing pending — corpus is fully scanned.[/green]")
            _print_ledger_status(ledger)
            return

        console.print(
            f"Enrolled {added} new APKs; {len(todo)} pending this run "
            f"(peak disk ≈ {max_staged * 17}MB)."
        )
        run_id = ledger.start_run(note=f"limit={limit}")

        def _event(event: str, payload: dict[str, object]) -> None:
            if event == "scanned":
                mark = "[bold green]LLM[/bold green]" if payload.get("llm") else "   "
                console.print(
                    f"  {mark} {str(payload.get('pkg'))[:44]:44} "
                    f"{payload.get('findings')} findings"
                )
            elif event in {"download_failed", "scan_failed"}:
                console.print(
                    f"  [red]{event}[/red] {str(payload.get('sha256'))[:16]}… "
                    f"{str(payload.get('error'))[:60]}"
                )

        report = run_stream(
            todo, ledger=ledger, client=client,
            staging_dir=Path(staging_dir),
            findings_path=Path(findings_path),
            detail_dir=Path(detail_dir),
            workers=workers, download_workers=download_workers,
            max_staged=max_staged, delay=delay, keep_apks=keep_apks,
            keep_with_findings=keep_with_findings,
            keep_negatives=keep_negatives,
            keep_budget_bytes=int(keep_budget_gb * 1024**3),
            on_event=_event,
        )
        ledger.finish_run(run_id)

        console.print(
            f"\n[bold]Run complete[/bold] in {report.elapsed / 60:.1f} min: "
            f"{report.scanned} scanned, {report.llm_integrated} LLM-integrated, "
            f"{report.download_failed} download failures, "
            f"{report.scan_failed} scan failures, "
            f"{report.bytes_downloaded / 1e9:.1f}GB transferred."
        )
        _print_ledger_status(ledger)


@corpus.command(name="retry")
@click.option("--ledger", "ledger_path", type=click.Path(exists=True),
              default="corpus/ledger.db", show_default=True)
@click.option("--rescan", is_flag=True,
              help="Also reset already-scanned APKs, forcing a full re-run. "
                   "Under streaming this means re-downloading them.")
@click.option("--yes", is_flag=True, help="Skip the confirmation prompt.")
def corpus_retry(ledger_path: str, rescan: bool, yes: bool) -> None:
    """Return failed APKs to the queue so the next run picks them up.

    By default resets download and scan failures. With --rescan it also resets
    successfully scanned APKs, which is what you want after changing a scanner
    rule — but note that streaming discarded those binaries, so they must be
    downloaded again.
    """
    states = [DOWNLOAD_FAILED, SCAN_FAILED]
    if rescan:
        states.append(SCANNED)

    with Ledger(Path(ledger_path)) as ledger:
        counts = ledger.counts()
        affected = sum(counts.get(state, 0) for state in states)
        if not affected:
            console.print("[green]Nothing to reset.[/green]")
            return
        if rescan and not yes:
            console.print(
                f"[yellow]--rescan will reset {affected} APKs, including "
                f"{counts.get(SCANNED, 0)} already scanned.[/yellow]\n"
                "Streaming discarded those binaries, so they will be "
                "re-downloaded from AndroZoo."
            )
            click.confirm("Proceed?", abort=True)

        n = ledger.reset(states)
        console.print(f"[green]Reset {n} APKs to pending.[/green]")
        _print_ledger_status(ledger)


@corpus.command(name="status")
@click.option("--ledger", "ledger_path", type=click.Path(exists=True),
              default="corpus/ledger.db", show_default=True)
def corpus_status(ledger_path: str) -> None:
    """Show corpus progress: how much is downloaded, scanned, and outstanding."""
    with Ledger(Path(ledger_path)) as ledger:
        _print_ledger_status(ledger)


def _print_ledger_status(ledger: Ledger) -> None:
    stats = ledger.stats()

    table = Table(title="Corpus status")
    table.add_column("State")
    table.add_column("APKs", justify="right")
    for state, n in sorted(stats.by_state.items(), key=lambda kv: -kv[1]):
        table.add_row(state, str(n))
    table.add_row("[bold]total[/bold]", f"[bold]{stats.total}[/bold]")
    console.print(table)

    if stats.scanned:
        console.print(
            f"LLM-integrated: {stats.llm_integrated}/{stats.scanned} "
            f"({100 * stats.llm_rate:.1f}%)  ·  "
            f"findings: {stats.findings}  ·  "
            f"scan time: {stats.scan_seconds / 3600:.1f}h"
        )
        providers = ledger.provider_counts()
        if providers:
            console.print("Providers: " + ", ".join(f"{k}={v}" for k, v in providers.items()))


@main.group()
def validate() -> None:
    """Measure filter accuracy against hand-labelled ground truth."""


@validate.command(name="sample")
@click.option("--findings", "findings_path", type=click.Path(exists=True, dir_okay=False),
              default="results/findings.jsonl", show_default=True,
              help="Findings JSONL produced by 'corpus run'.")
@click.option("--per-stratum", type=int, default=100, show_default=True,
              help="APKs to draw from each of the flagged/unflagged strata.")
@click.option("--seed", type=int, default=20260811, show_default=True,
              help="RNG seed for a reproducible draw.")
@click.option("-o", "--output-dir", type=click.Path(), default="validation",
              show_default=True)
def validate_sample(
    findings_path: str, per_stratum: int, seed: int, output_dir: str
) -> None:
    """Draw the stratified validation sub-sample for hand-labelling.

    Writes two files: a blind worksheet with no predictions in it, and a
    separate answer key. The coder must only ever open the worksheet.
    """
    rows = read_findings(Path(findings_path))
    sample = draw_validation_sample(rows, per_stratum=per_stratum, seed=seed)

    out = Path(output_dir)
    label_path = out / "sample_labels.csv"
    key_path = out / "sample_key.csv"
    write_sample(sample, label_path=label_path, key_path=key_path)

    table = Table(title="Validation sample")
    table.add_column("Stratum")
    table.add_column("Pool", justify="right")
    table.add_column("Drawn", justify="right")
    table.add_row("flagged (predicted LLM)", str(sample.pool_flagged), str(len(sample.flagged)))
    table.add_row("unflagged", str(sample.pool_unflagged), str(len(sample.unflagged)))
    console.print(table)

    if sample.is_short:
        console.print(
            f"[yellow]Short of {per_stratum} per stratum.[/yellow] Precision and "
            "recall from a sample this size carry wide confidence intervals; "
            "scan a larger corpus before drawing the final validation set."
        )
    console.print(f"Worksheet (blind) → {label_path}")
    console.print(f"Answer key       → {key_path}  [dim]do not open while labelling[/dim]")
    console.print(
        "\nNext: download these APKs with --keep-apks, then run\n"
        f"  python scripts/label_validation_set.py --apks <dir> --worksheet {label_path}"
    )


@validate.command(name="credentials")
@click.option(
    "--staging", "staging_path", type=click.Path(exists=True, dir_okay=False),
    required=True,
    help="CSV from the 0600 credential staging area: provider,key,sha256,package. "
         "Never committed (docs/DATA_HANDLING.md §3).",
)
@click.option("--live", is_flag=True,
              help="Make real introspection calls. Without this, dry run only.")
@click.option("--yes", is_flag=True, help="Skip the confirmation prompt.")
@click.option("--timeout", default=30.0, show_default=True,
              help="HTTP timeout in seconds for each introspection call.")
def validate_credentials_cmd(staging_path: str, live: bool, yes: bool, timeout: float) -> None:
    """Check which discovered credentials are still live.

    Calls only non-revocable introspection endpoints — no content generation,
    no billable inference, no impersonation (proposal §8.2, ETHICS.md §2.1).
    Providers without such an endpoint are abstained from, not guessed at.

    This separates "contains a key-shaped string" from "exposes a working
    credential". Only the second supports a prevalence claim, and reporting
    already-revoked keys wastes vendor security-team time.
    """
    records: list[CredentialRecord] = []
    with open(staging_path, encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                provider = Provider(row["provider"])
            except ValueError:
                console.print(f"[yellow]Unknown provider {row['provider']!r}, skipping[/yellow]")
                continue
            records.append(
                CredentialRecord(
                    provider=provider, key=row["key"],
                    sha256=row.get("sha256", ""), package=row.get("package", ""),
                )
            )

    if not records:
        console.print("[red]No credentials read from staging file.[/red]")
        sys.exit(1)

    abstained = unvalidatable_providers(records)
    if abstained:
        console.print(
            "[yellow]Abstaining from:[/yellow] "
            + ", ".join(sorted(p.value for p in abstained))
            + "  (no non-revocable introspection endpoint; ETHICS.md §5.2)"
        )

    distinct = len({r.key for r in records})
    if live and not yes:
        console.print(
            f"[yellow]This will make {distinct} real request(s) to provider "
            "introspection endpoints using credentials belonging to other "
            "developers.[/yellow]\nPermitted under proposal §8.2 because these "
            "endpoints do not generate content, consume quota, or impersonate. "
            "Each distinct key is called at most once."
        )
        click.confirm("Proceed?", abort=True)

    summary = validate_credentials(records, live=live, timeout_seconds=timeout)
    console.print(render_report(summary))

    if not live:
        console.print("\n[dim]Dry run — no requests made. Re-run with --live.[/dim]")
        return

    disclose = summary.to_disclose()
    console.print(
        f"\n[bold]{len(disclose)} credential(s) warrant disclosure[/bold] "
        f"({summary.live} live, {summary.unknown} unverifiable); "
        f"{summary.revoked} already revoked and will not be reported."
    )


@validate.command(name="score")
@click.option("--labels", "labels_path", type=click.Path(exists=True, dir_okay=False),
              default="validation/manual_labels.jsonl", show_default=True)
@click.option("--key", "key_path", type=click.Path(exists=True, dir_okay=False),
              default="validation/sample_key.csv", show_default=True)
def validate_score(labels_path: str, key_path: str) -> None:
    """Score hand labels against the filter's predictions.

    Reports precision, recall and F1 with Wilson intervals — the numbers that
    make the RQ1 prevalence estimate defensible.
    """
    store = LabelStore(Path(labels_path))
    truth: dict[str, bool | None] = {
        sha.upper(): record.is_llm_integrated
        for sha, record in store.all_labels().items()
    }
    predicted_all = read_key(Path(key_path))
    predicted = {sha: pred for sha, pred in predicted_all.items() if sha in truth}

    if not predicted:
        console.print("[red]No overlap between labels and key.[/red]")
        sys.exit(1)

    metrics = compute_metrics(predicted, truth)
    uncertain = sum(1 for v in truth.values() if v is None)

    table = Table(title="Filter accuracy vs hand-labelled ground truth")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_column("95% CI (Wilson)", justify="right")
    p_lo, p_hi = metrics.precision_ci
    r_lo, r_hi = metrics.recall_ci
    table.add_row("Precision", f"{metrics.precision:.3f}", f"[{p_lo:.3f}, {p_hi:.3f}]")
    table.add_row("Recall", f"{metrics.recall:.3f}", f"[{r_lo:.3f}, {r_hi:.3f}]")
    table.add_row("F1", f"{metrics.f1:.3f}", "—")
    console.print(table)
    console.print(
        f"TP={metrics.tp}  FP={metrics.fp}  FN={metrics.fn}  TN={metrics.tn}"
        f"  ·  {len(predicted)} scored, {uncertain} uncertain (excluded)"
    )
    if metrics.recall < 0.85:
        console.print(
            f"[yellow]Recall {metrics.recall:.1%} is below the ≥85% success "
            "criterion in proposal §9.[/yellow] Review the false negatives before "
            "reporting prevalence."
        )


def _walk_apks(apk_dir: str) -> list[Path]:
    """Every APK under ``apk_dir``, skipping macOS AppleDouble stubs.

    Files named ``._<name>.apk`` are 163-byte resource-fork sidecars created
    when a directory is copied via macOS. They match ``*.apk`` but are not
    ZIP archives, so every one of them fails to parse. An earlier batch run
    globbed them and reported 100% scan failure across the corpus.
    """
    return [
        Path(root) / fn
        for root, _, files in os.walk(apk_dir)
        for fn in files
        if fn.lower().endswith(".apk") and not fn.startswith("._")
    ]


def _print_summary(result) -> None:  # type: ignore[no-untyped-def]
    table = Table(title=f"Summary: {result.apk.package_name}")
    table.add_column("Severity")
    table.add_column("Count", justify="right")
    for sev, count in result.summary.items():
        if count:
            table.add_row(sev, str(count))
    console.print(table)


if __name__ == "__main__":
    main()
