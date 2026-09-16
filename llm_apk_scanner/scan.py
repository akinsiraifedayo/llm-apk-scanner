"""
End-to-end scan orchestration.

Composes the loader, the two-stage filter, and the four detectors
into a single `scan_apk(path) -> ScanResult` function. The CLI is a
thin wrapper around this.
"""

from __future__ import annotations

import time
from dataclasses import replace

from llm_apk_scanner import __version__
from llm_apk_scanner.apk_loader import ApkCorpus, load_apk
from llm_apk_scanner.filters import LexicalFilter, StructuralFilter
from llm_apk_scanner.models import (
    Finding,
    FindingKind,
    Provider,
    ScanResult,
    Severity,
    SourceLocation,
)
from llm_apk_scanner.scanners import (
    RagArtefactScanner,
    SecretScanner,
    SystemPromptScanner,
    ToolSchemaScanner,
)

#: Rule name of the ambiguous Google key. See `_attribute_google_keys`.
_GOOGLE_KEY_RULE = "Google API key (service unknown)"


def _attribute_google_keys(
    findings: list[Finding], providers: list[Provider]
) -> list[Finding]:
    """Attribute generic Google API keys using app-level context.

    An ``AIza…`` string authorises *some* Google API — Maps, Firebase,
    YouTube, Gemini — and the key itself does not say which. The detector
    therefore reports it as GENERIC/MEDIUM, and this promotes it to
    GOOGLE_GEMINI/HIGH only when the same APK also references a Gemini or
    Vertex inference endpoint. Without that context the classification is a
    guess, and guessing "Gemini" made every Firebase key look like
    inference-credential exposure.

    Firebase and Maps keys are meant to ship in the client and are
    restricted by package name and signing certificate, so an uncorroborated
    match is usually not an exposure at all — which is why the demoted
    severity is MEDIUM rather than HIGH.
    """
    corroborated = Provider.GOOGLE_GEMINI in providers
    if not corroborated:
        return findings

    promoted: list[Finding] = []
    for finding in findings:
        if _GOOGLE_KEY_RULE in finding.title:
            promoted.append(
                replace(
                    finding,
                    provider=Provider.GOOGLE_GEMINI,
                    severity=Severity.HIGH,
                    description=(
                        finding.description
                        + " Attributed to Gemini: this APK also references a "
                        "Google generative-AI inference endpoint."
                    ),
                )
            )
        else:
            promoted.append(finding)
    return promoted


def _deduplicate(findings: list[Finding]) -> list[Finding]:
    """Collapse findings that describe the same artefact.

    A credential is one exposure whether the literal appears once or five
    times, and whether it sits in ``classes3.dex``, ``resources.arsc`` or
    both. Counting occurrences instead of artefacts inflates every per-app
    figure, and it does so unevenly: the inflation tracks how many times a
    build tool happened to duplicate a string, not how exposed the app is.

    This became material once assets were read properly (Defect 14). The same
    Google key routinely appears in both ``google-services-desktop.json`` and
    the compiled ``resources.arsc``, and embedding-model identifiers appear
    once per call site.

    The highest-confidence instance is kept, so the retained finding also
    carries the most informative location. Occurrence count and the full set
    of locations are preserved in metadata, because "this key appears in four
    places" is worth reporting even though it is one exposure.
    """
    best: dict[tuple, Finding] = {}
    extra: dict[tuple, list[str]] = {}
    for f in findings:
        key = (f.kind, f.provider, f.evidence)
        where = f.location.file_path if f.location else ""
        extra.setdefault(key, []).append(where)
        incumbent = best.get(key)
        if incumbent is None or f.confidence > incumbent.confidence:
            best[key] = f

    out: list[Finding] = []
    for key, f in best.items():
        locations = sorted({w for w in extra[key] if w})
        if len(extra[key]) > 1:
            meta = dict(f.metadata or {})
            meta["occurrences"] = len(extra[key])
            meta["locations"] = locations
            f = replace(f, metadata=meta)
        out.append(f)
    return out


def scan_corpus(corpus: ApkCorpus) -> ScanResult:
    """Run the full pipeline over an in-memory corpus."""
    started = time.perf_counter()

    lexical = LexicalFilter()
    structural = StructuralFilter()

    lexical_matches = lexical.scan_corpus(corpus.sources)
    candidate_providers = LexicalFilter.detected_providers(lexical_matches)
    structural_evidence = structural.scan(corpus.sources, candidate_providers)
    is_llm = StructuralFilter.is_confirmed(lexical_matches, structural_evidence)

    findings: list[Finding] = []

    # Surface lexical hits as low-severity informational findings; they
    # are themselves part of the dissertation's prevalence statistics.
    for m in lexical_matches:
        findings.append(
            Finding(
                kind=(
                    FindingKind.LLM_ENDPOINT
                    if m.is_endpoint
                    else FindingKind.LLM_SDK_REFERENCE
                ),
                severity=Severity.INFO,
                title=(
                    f"{m.provider.value} endpoint reference"
                    if m.is_endpoint
                    else f"{m.provider.value} SDK package reference"
                ),
                description=(
                    "Lexical match for an LLM-provider signature. "
                    "Used as evidence for app population identification."
                ),
                location=SourceLocation(file_path=m.source_file),
                evidence=m.matched_text,
                confidence=0.9 if m.is_endpoint else 0.7,
                provider=m.provider,
            )
        )

    # The four detectors run regardless — this preserves a fair
    # baseline rate of findings against putatively non-LLM apps.
    findings.extend(
        _attribute_google_keys(
            SecretScanner().scan(corpus.sources), candidate_providers
        )
    )
    findings.extend(SystemPromptScanner().scan(corpus.sources))
    findings.extend(ToolSchemaScanner().scan(corpus.sources))
    findings.extend(RagArtefactScanner().scan(corpus.sources))

    findings = _deduplicate(findings)

    duration = time.perf_counter() - started

    return ScanResult(
        apk=corpus.info,
        is_llm_integrated=is_llm,
        detected_providers=candidate_providers,
        findings=findings,
        scan_duration_seconds=round(duration, 4),
        scanner_version=__version__,
    )


def scan_apk(path: str) -> ScanResult:
    """Load an APK from disk and run the full scan."""
    corpus = load_apk(path)
    return scan_corpus(corpus)
