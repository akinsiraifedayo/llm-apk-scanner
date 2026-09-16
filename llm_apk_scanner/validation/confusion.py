"""Render a 2x2 confusion matrix as Markdown for inclusion in dissertation §6.3."""

from __future__ import annotations

from llm_apk_scanner.validation.metrics import Metrics


def render_confusion_matrix(metrics: Metrics, label: str = "filter") -> str:
    p_lo, p_hi = metrics.precision_ci
    r_lo, r_hi = metrics.recall_ci
    lines = [
        f"## Confusion matrix — {label}",
        "",
        "| | predicted positive | predicted negative |",
        "|---|---|---|",
        f"| **actual positive** | TP = {metrics.tp} | FN = {metrics.fn} |",
        f"| **actual negative** | FP = {metrics.fp} | TN = {metrics.tn} |",
        "",
        f"- Precision: {metrics.precision:.4f} "
        f"(95% CI {p_lo:.4f}–{p_hi:.4f})",
        f"- Recall:    {metrics.recall:.4f} "
        f"(95% CI {r_lo:.4f}–{r_hi:.4f})",
        f"- F1:        {metrics.f1:.4f}",
    ]
    return "\n".join(lines)
