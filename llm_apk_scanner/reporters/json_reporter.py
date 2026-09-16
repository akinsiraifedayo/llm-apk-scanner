"""JSON serialiser for ScanResult."""

from __future__ import annotations

import json
from typing import Any

from llm_apk_scanner.models import ScanResult


class JsonReporter:
    """Stable JSON serialiser. The schema is part of the public contract."""

    def render(self, result: ScanResult, indent: int = 2) -> str:
        return json.dumps(
            result.to_dict(),
            indent=indent,
            sort_keys=False,
            ensure_ascii=False,
            default=self._default,
        )

    @staticmethod
    def _default(obj: Any) -> Any:
        if hasattr(obj, "value"):
            return obj.value
        if hasattr(obj, "to_dict"):
            return obj.to_dict()
        raise TypeError(f"Cannot serialise {type(obj).__name__}")
