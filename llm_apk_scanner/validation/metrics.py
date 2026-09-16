"""
Precision, recall, F1, and Wilson confidence intervals for the
filter and the per-detector validation outcomes.
"""

from __future__ import annotations

from dataclasses import dataclass

from llm_apk_scanner.stats import wilson_interval


@dataclass
class Metrics:
    """Standard binary-classifier metrics with CIs."""

    tp: int
    fp: int
    fn: int
    tn: int

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def precision_ci(self) -> tuple[float, float]:
        return wilson_interval(self.tp, self.tp + self.fp)

    @property
    def recall_ci(self) -> tuple[float, float]:
        return wilson_interval(self.tp, self.tp + self.fn)

    def as_table_row(self, label: str) -> dict[str, str | float | int]:
        p_lo, p_hi = self.precision_ci
        r_lo, r_hi = self.recall_ci
        return {
            "label": label,
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "tn": self.tn,
            "precision": round(self.precision, 4),
            "precision_ci_lower": round(p_lo, 4),
            "precision_ci_upper": round(p_hi, 4),
            "recall": round(self.recall, 4),
            "recall_ci_lower": round(r_lo, 4),
            "recall_ci_upper": round(r_hi, 4),
            "f1": round(self.f1, 4),
        }


def compute_metrics(
    predicted: dict[str, bool],
    actual: dict[str, bool | None],
) -> Metrics:
    """
    Compute confusion-matrix metrics.

    Args:
        predicted: sha256 -> True if the system flagged the APK.
        actual:    sha256 -> True/False/None ground truth (None
                   means uncertain; uncertain APKs are excluded
                   from metrics).

    Returns:
        Metrics over the intersecting sha256 set.
    """
    tp = fp = fn = tn = 0
    for sha, pred in predicted.items():
        truth = actual.get(sha)
        if truth is None:
            continue  # uncertain — exclude
        if pred and truth:
            tp += 1
        elif pred and not truth:
            fp += 1
        elif (not pred) and truth:
            fn += 1
        else:
            tn += 1
    return Metrics(tp=tp, fp=fp, fn=fn, tn=tn)
