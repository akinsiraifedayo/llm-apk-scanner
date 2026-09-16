"""
Manual-label store.

The validation harness asks the candidate (single-coder per the
proposal §8.4 limitation) to label each APK in a sub-sample with
the ground-truth answer:

    - is_llm_integrated: True | False | uncertain

The store is JSON Lines so it can be incrementally appended
during long labelling sessions and so labelling decisions are
auditable.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class LabelRecord:
    """A single manual label for an APK in the validation sample."""

    apk_sha256: str
    apk_path: str
    is_llm_integrated: bool | None  # None == "uncertain"
    notes: str = ""
    labelled_at: str = ""  # ISO 8601
    labeller: str = "candidate"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class LabelStore:
    """Append-only JSONL store of LabelRecord objects."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: LabelRecord) -> None:
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record.to_dict(), ensure_ascii=False))
            fh.write("\n")

    def __iter__(self) -> Iterator[LabelRecord]:
        if not self.path.exists():
            return iter(())

        def _gen() -> Iterator[LabelRecord]:
            with self.path.open(encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    d = json.loads(line)
                    yield LabelRecord(**d)

        return _gen()

    def latest_for(self, apk_sha256: str) -> LabelRecord | None:
        """Last record for a given APK, or None if not labelled."""
        last: LabelRecord | None = None
        for rec in self:
            if rec.apk_sha256 == apk_sha256:
                last = rec
        return last

    def all_labels(self) -> dict[str, LabelRecord]:
        """Latest label per APK, keyed by sha256."""
        out: dict[str, LabelRecord] = {}
        for rec in self:
            out[rec.apk_sha256] = rec
        return out
