"""
Filter and detector validation harness.

Implements the methodology in proposal §3.3 (filter validation
on a 200-APK manually-labelled sample) and §4.4 (detector recall
on seeded ground-truth).

Modules:
    - labeller.py:    record manual labels for the validation set.
    - metrics.py:     compute precision, recall, F1, Wilson CI.
    - confusion.py:   render a confusion matrix in markdown form.
"""

from llm_apk_scanner.validation.confusion import render_confusion_matrix
from llm_apk_scanner.validation.labeller import (
    LabelRecord,
    LabelStore,
)
from llm_apk_scanner.validation.metrics import (
    Metrics,
    compute_metrics,
)

__all__ = [
    "LabelRecord",
    "LabelStore",
    "Metrics",
    "compute_metrics",
    "render_confusion_matrix",
]
