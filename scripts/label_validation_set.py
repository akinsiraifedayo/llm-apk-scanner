"""Blind interactive labeller for the validation sub-sample (METHODOLOGY §3).

Presents each APK's raw evidence — hosts, SDK paths, AI-vocabulary strings in
context — and asks for a ground-truth judgement. Writes to an append-only
JSONL store, so a long labelling session can be interrupted and resumed.

Two properties make the resulting labels usable as ground truth:

1. **Blind.** The scanner's prediction is never shown. Anchoring the coder on
   the system's own answer would produce a ground truth that agrees with it by
   construction, and precision/recall computed from that measure nothing.
2. **Evidence-bearing.** The coder sees what is actually in the APK, including
   signals the scanner does not act on — which is the only way a false
   negative can ever be spotted.

Usage:
    python scripts/label_validation_set.py --apks validation/apks/ \\
        --worksheet validation/sample_labels.csv

Labels are recorded by SHA-256, so re-running skips what is already done.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import sys
from pathlib import Path

from llm_apk_scanner.validation import LabelRecord, LabelStore
from llm_apk_scanner.validation.evidence import extract_evidence, render_evidence

_MENU = (
    "\n  [y] LLM-integrated   [n] not LLM-integrated   "
    "[u] uncertain   [s] skip   [q] save & quit\n"
)


def prompt_for_label() -> tuple[bool | None, str, bool]:
    """Ask for one judgement. Returns ``(label, notes, should_quit)``."""
    while True:
        answer = input("  Ground truth? [y/n/u/s/q]: ").strip().lower()
        if answer in {"y", "yes"}:
            return True, input("  Notes (what convinced you?): ").strip(), False
        if answer in {"n", "no"}:
            return False, input("  Notes (optional): ").strip(), False
        if answer in {"u", "uncertain"}:
            return None, input("  Why uncertain?: ").strip(), False
        if answer in {"s", "skip"}:
            return None, "__SKIP__", False
        if answer in {"q", "quit"}:
            return None, "__SKIP__", True
        print("  Please answer y, n, u, s or q.")


def _worksheet_order(worksheet: Path | None, apks: list[Path]) -> list[Path]:
    """Order APKs by the worksheet, so both strata are interleaved.

    Labelling all flagged APKs first and all unflagged afterwards would let
    the coder infer the stratum from position and start pattern-matching
    rather than judging each app on its evidence.
    """
    if worksheet is None or not worksheet.exists():
        return sorted(apks)
    with worksheet.open(encoding="utf-8") as handle:
        wanted = [row["sha256"].upper() for row in csv.DictReader(handle)]
    by_sha = {p.stem.upper(): p for p in apks}
    ordered = [by_sha[sha] for sha in wanted if sha in by_sha]
    missing = [p for p in apks if p.stem.upper() not in set(wanted)]
    return ordered + sorted(missing)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apks", type=Path, required=True,
                        help="Directory of validation APKs (downloaded with --keep-apks).")
    parser.add_argument("--worksheet", type=Path, default=None,
                        help="Blind worksheet CSV; fixes the labelling order.")
    parser.add_argument("-o", "--out", type=Path,
                        default=Path("validation/manual_labels.jsonl"))
    args = parser.parse_args()

    apks = [p for p in args.apks.rglob("*.apk") if not p.name.startswith("._")]
    if not apks:
        print(f"No APKs found under {args.apks}")
        return 1

    store = LabelStore(args.out)
    already = set(store.all_labels())
    queue = [p for p in _worksheet_order(args.worksheet, apks) if p.stem.upper() not in already]

    print(f"{len(apks)} APKs in set; {len(already)} already labelled; "
          f"{len(queue)} remaining.")
    print("The scanner's prediction is deliberately hidden — judge on evidence only.")
    print(_MENU)

    labelled = 0
    for position, path in enumerate(queue, 1):
        print(f"\n[{position}/{len(queue)}]")
        evidence = extract_evidence(path)
        print(render_evidence(evidence))

        truth, notes, should_quit = prompt_for_label()
        if should_quit:
            break
        if notes == "__SKIP__":
            continue

        store.append(
            LabelRecord(
                apk_sha256=path.stem.upper(),
                apk_path=str(path),
                is_llm_integrated=truth,
                notes=notes,
                labelled_at=dt.datetime.now(dt.UTC).isoformat(),
                labeller="candidate",
            )
        )
        labelled += 1

    print(f"\n{labelled} new labels written to {args.out}")
    print(f"{len(already) + labelled}/{len(apks)} of the validation set is labelled.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
