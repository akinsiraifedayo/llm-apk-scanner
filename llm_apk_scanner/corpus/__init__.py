"""Corpus acquisition: AndroZoo metadata parsing, stratified sampling, download.

This subpackage implements the sample-frame construction described in
``docs/METHODOLOGY.md`` §2.3 (two-stage stratified sample) and the data
handling described in ``docs/DATA_HANDLING.md``. The AndroZoo API key is
read from the ``ANDROZOO_API_KEY`` environment variable and is never
stored in code or committed to the repository.
"""

from __future__ import annotations

from llm_apk_scanner.corpus.model import ApkRecord
from llm_apk_scanner.corpus.sample import SampleResult, stratified_sample

__all__ = ["ApkRecord", "SampleResult", "stratified_sample"]
