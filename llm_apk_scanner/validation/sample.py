"""Draw the validation sub-sample used to measure filter accuracy.

``docs/METHODOLOGY.md`` §3 specifies a stratified sub-sample of 200 APKs —
100 the filter flagged as LLM-integrated, 100 it did not — hand-labelled to
give ground truth. Comparing labels against predictions yields the precision
and recall (with Wilson intervals) that make the RQ1 prevalence figure
defensible rather than merely asserted.

Stratifying by the *prediction* is deliberate. Drawing 200 APKs at random
would, at the observed ~6% flag rate, yield about a dozen positives — far too
few to estimate precision with any useful interval. Sampling both strata to
equal size measures each direction of error properly; the strata are then
reweighted to corpus proportions when computing corpus-level rates.

The draw is seeded, so the same corpus and seed always produce the same
validation set — the sample is a publishable artefact, not a one-off.
"""

from __future__ import annotations

import csv
import json
import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ValidationSample:
    """The drawn validation set, split by what the filter predicted."""

    flagged: list[dict[str, object]] = field(default_factory=list)
    unflagged: list[dict[str, object]] = field(default_factory=list)
    requested_per_stratum: int = 0
    pool_flagged: int = 0
    pool_unflagged: int = 0
    seed: int = 0

    @property
    def size(self) -> int:
        return len(self.flagged) + len(self.unflagged)

    @property
    def is_short(self) -> bool:
        """Whether either stratum could not be filled from the pool."""
        return (
            len(self.flagged) < self.requested_per_stratum
            or len(self.unflagged) < self.requested_per_stratum
        )

    def all_rows(self) -> list[dict[str, object]]:
        """Both strata in one deterministically shuffled order.

        Shuffled rather than sorted, so the coder cannot infer a row's stratum
        from where it sits in the worksheet. Sorting by SHA-256 would *usually*
        interleave the strata — real hashes are effectively random — but that
        is an incidental property to rest blinding on, not a guarantee. The
        shuffle is seeded, so the worksheet is still byte-stable across runs.
        """
        rows = sorted(self.flagged + self.unflagged, key=lambda r: str(r["sha256"]))
        random.Random(self.seed).shuffle(rows)
        return rows


def read_findings(path: Path) -> list[dict[str, object]]:
    """Load a findings JSONL file, skipping blank lines."""
    rows: list[dict[str, object]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def draw_validation_sample(
    rows: Iterable[dict[str, object]],
    *,
    per_stratum: int = 100,
    seed: int = 20260811,
) -> ValidationSample:
    """Draw ``per_stratum`` APKs from each of the flagged/unflagged strata.

    If a stratum holds fewer than ``per_stratum`` rows the whole stratum is
    taken and ``is_short`` reports it — silently returning a smaller sample
    would understate the confidence intervals computed from it.
    """
    flagged_pool: list[dict[str, object]] = []
    unflagged_pool: list[dict[str, object]] = []
    for row in rows:
        (flagged_pool if row.get("llm_integrated") else unflagged_pool).append(row)

    # Sort before sampling so the seeded draw does not depend on input order.
    flagged_pool.sort(key=lambda r: str(r["sha256"]))
    unflagged_pool.sort(key=lambda r: str(r["sha256"]))

    rng = random.Random(seed)
    return ValidationSample(
        flagged=_take(rng, flagged_pool, per_stratum),
        unflagged=_take(rng, unflagged_pool, per_stratum),
        requested_per_stratum=per_stratum,
        pool_flagged=len(flagged_pool),
        pool_unflagged=len(unflagged_pool),
        seed=seed,
    )


def _take(
    rng: random.Random, pool: Sequence[dict[str, object]], n: int
) -> list[dict[str, object]]:
    if len(pool) <= n:
        return list(pool)
    return rng.sample(list(pool), n)


#: Written for the labeller. Deliberately **excludes** the prediction: showing
#: it would anchor the coder onto the scanner's answer, and a ground truth
#: that agrees with the system by construction measures nothing.
_LABEL_FIELDS = ["sha256", "pkg_name", "version", "apk_size"]

#: Written for the analyst. Retains the prediction so labels can be scored.
_KEY_FIELDS = ["sha256", "pkg_name", "llm_integrated", "providers", "n_findings"]


def write_sample(sample: ValidationSample, *, label_path: Path, key_path: Path) -> None:
    """Write the blind labelling worksheet and the separate answer key.

    Two files, because the coder must not see the prediction while labelling
    but the analyst needs it afterwards to build the confusion matrix.
    """
    rows = sample.all_rows()

    label_path = Path(label_path)
    label_path.parent.mkdir(parents=True, exist_ok=True)
    with label_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_LABEL_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in _LABEL_FIELDS})

    key_path = Path(key_path)
    key_path.parent.mkdir(parents=True, exist_ok=True)
    with key_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_KEY_FIELDS)
        writer.writeheader()
        for row in rows:
            record = {k: row.get(k, "") for k in _KEY_FIELDS}
            providers = row.get("providers")
            if isinstance(providers, list):
                record["providers"] = "|".join(str(p) for p in providers)
            writer.writerow(record)


def read_key(path: Path) -> dict[str, bool]:
    """Read the answer key as ``sha256 -> predicted LLM-integrated``."""
    predictions: dict[str, bool] = {}
    with Path(path).open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            predictions[row["sha256"]] = str(row["llm_integrated"]).lower() in {
                "true", "1", "yes",
            }
    return predictions
