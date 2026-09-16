"""
Native-library string extractor.

The LLM-app identification pipeline is biased toward Java/Kotlin
binaries. Many apps push their inference client into a native
library — `lib*.so` files inside `lib/<abi>/` — to obfuscate
endpoint or token references. This module extracts printable
strings from such libraries so the lexical filter sees them.

Implementation is a pure-Python port of the classic `strings(1)`
utility, sufficient for the scanner's lexical signal needs. It
does not attempt full ELF parsing (which would require a
dependency such as pyelftools); the scanner is concerned only
with string-level signal.

Threshold defaults follow GNU strings: minimum 4 printable
characters in a run. The encoder picks up both ASCII and UTF-16-LE
runs (the latter common in C++ wide-string literals).
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from typing import BinaryIO


PRINTABLE_BYTES = bytes(range(0x20, 0x7F)) + b"\t"


def extract_strings(
    fh: BinaryIO,
    min_length: int = 4,
    chunk_size: int = 1 << 20,
) -> Iterator[str]:
    """Yield ASCII and UTF-16-LE printable runs from a binary stream."""
    yield from _extract_ascii(fh, min_length=min_length, chunk_size=chunk_size)
    fh.seek(0)
    yield from _extract_utf16_le(fh, min_length=min_length, chunk_size=chunk_size)


def _extract_ascii(
    fh: BinaryIO, *, min_length: int, chunk_size: int
) -> Iterator[str]:
    run: bytearray = bytearray()
    while True:
        chunk = fh.read(chunk_size)
        if not chunk:
            break
        for b in chunk:
            if b in PRINTABLE_BYTES:
                run.append(b)
            else:
                if len(run) >= min_length:
                    yield run.decode("ascii", errors="replace")
                run.clear()
    if len(run) >= min_length:
        yield run.decode("ascii", errors="replace")


def _extract_utf16_le(
    fh: BinaryIO, *, min_length: int, chunk_size: int
) -> Iterator[str]:
    """
    Stream-friendly UTF-16-LE printable run extraction.

    A UTF-16-LE printable code unit is one where the high byte is 0
    and the low byte is in the printable ASCII range. This is a
    well-known heuristic — it misses non-Latin scripts but catches
    the C++ wide-string idiom that motivates this extractor.
    """
    buf = bytearray()
    while True:
        chunk = fh.read(chunk_size)
        if not chunk:
            break
        buf.extend(chunk)
        # Process pairs.
        end = len(buf) & ~1
        run: bytearray = bytearray()
        for i in range(0, end, 2):
            low = buf[i]
            high = buf[i + 1]
            if high == 0 and low in PRINTABLE_BYTES:
                run.append(low)
            else:
                if len(run) >= min_length:
                    yield run.decode("ascii", errors="replace")
                run.clear()
        if len(run) >= min_length:
            yield run.decode("ascii", errors="replace")
        # Carry over the unpaired tail byte if any.
        del buf[:end]


def extract_from_file(
    path: str | os.PathLike[str],
    min_length: int = 4,
) -> list[str]:
    """Convenience wrapper that opens the file and collects all strings."""
    with open(path, "rb") as fh:
        return list(extract_strings(fh, min_length=min_length))
