"""Common base class for static scanners."""

from __future__ import annotations

from abc import ABC, abstractmethod

from llm_apk_scanner.models import Finding


class BaseScanner(ABC):
    """
    Common interface for the four detector modules.

    Each scanner consumes (source_file, content) tuples and emits a
    list of Findings. Scanners are stateless — no shared mutable
    state, so they can be run in parallel across APKs in a corpus run.
    """

    name: str = "base"

    @abstractmethod
    def scan(self, sources: list[tuple[str, str]]) -> list[Finding]:
        """Scan a list of (source_file, content) blobs."""
        raise NotImplementedError
