"""Batch credential validation against non-revocable introspection endpoints.

Answers the question a prevalence figure cannot answer on its own: of the
credential-shaped strings found in the corpus, how many are **live**?

That distinction decides what RQ1 may claim. "N apps contain a string matching
the OpenAI key format" is weak — developers routinely rotate a key and leave
the old literal in the APK, so a format match alone proves only that a key was
*once* there. "N apps expose a credential that still authenticates today" is
the strong claim, and it needs validation to make.

It also protects the disclosure channel. Reporting a hundred keys of which
ninety are already revoked burns vendor security-team time and the goodwill
that responsible disclosure depends on.

Ethics envelope (proposal §8.2, `docs/ETHICS.md` §2.1, §5.2): validation calls
**only** introspection endpoints that do not generate content, do not consume
billable inference, are idempotent, and leave no side effect beyond a log
line. Providers without such an endpoint are abstained from, not guessed at.

Two safety properties enforced here rather than left to the caller:

* **Dry-run by default.** A live call requires an explicit opt-in.
* **At most one call per distinct key.** Results are cached by key, so a key
  appearing in twenty APKs is checked once. Repeatedly probing someone else's
  credential is neither necessary nor defensible.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone

from llm_apk_scanner.models import Provider
from llm_apk_scanner.validators.key_validator import KeyValidationResult, KeyValidator


@dataclass(frozen=True)
class CredentialRecord:
    """One credential found in the corpus, awaiting validation."""

    provider: Provider
    key: str = field(repr=False)
    sha256: str = ""
    package: str = ""

    @property
    def fingerprint(self) -> str:
        """Stable, non-reversible identifier for logs and reports.

        Never log the key. The fingerprint lets two findings be recognised as
        the same credential without the value appearing in any artefact.
        """
        return hashlib.sha256(self.key.encode()).hexdigest()[:16]


@dataclass
class ValidationOutcome:
    """The result of validating one credential."""

    record: CredentialRecord
    valid: bool | None
    http_status: int | None
    note: str
    checked_at: str

    @property
    def state(self) -> str:
        if self.valid is True:
            return "live"
        if self.valid is False:
            return "revoked"
        return "unknown"


@dataclass
class ValidationSummary:
    """Corpus-level totals — the numbers RQ1 reports."""

    outcomes: list[ValidationOutcome] = field(default_factory=list)

    @property
    def live(self) -> int:
        return sum(1 for o in self.outcomes if o.valid is True)

    @property
    def revoked(self) -> int:
        return sum(1 for o in self.outcomes if o.valid is False)

    @property
    def unknown(self) -> int:
        return sum(1 for o in self.outcomes if o.valid is None)

    @property
    def total(self) -> int:
        return len(self.outcomes)

    @property
    def live_rate(self) -> float:
        """Share of *checked* credentials still live.

        Abstained and errored checks are excluded from the denominator —
        counting an abstention as "revoked" would understate exposure, and
        counting it as "live" would overstate it.
        """
        checked = self.live + self.revoked
        return self.live / checked if checked else 0.0

    def to_disclose(self) -> list[ValidationOutcome]:
        """Credentials warranting a vendor report.

        Live keys always. Unknown-state keys too, because abstention is our
        limitation and not evidence of safety — an Anthropic key we cannot
        check may well be live, and the vendor can determine that in a way we
        deliberately cannot. Confirmed-revoked keys are not reported.
        """
        return [o for o in self.outcomes if o.valid is not False]


def validate_credentials(
    records: Sequence[CredentialRecord],
    *,
    live: bool = False,
    validator: KeyValidator | None = None,
    timeout_seconds: float = 30.0,
) -> ValidationSummary:
    """Validate each distinct credential at most once.

    Args:
        records: credentials recovered from the corpus.
        live: perform real introspection calls. Default False (dry run) —
            a live call must be a deliberate act, not a default.
        validator: injected for testing.

    Returns:
        A ``ValidationSummary`` with one outcome per input record; records
        sharing a key share the cached result.
    """
    validator = validator or KeyValidator()
    summary = ValidationSummary()
    cache: dict[str, KeyValidationResult] = {}

    for record in records:
        cached = cache.get(record.key)
        if cached is None:
            cached = validator.validate(
                record.provider, record.key, dry_run=not live,
                timeout_seconds=timeout_seconds,
            )
            cache[record.key] = cached
        summary.outcomes.append(
            ValidationOutcome(
                record=record,
                valid=cached.valid,
                http_status=cached.http_status,
                note=cached.note,
                checked_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            )
        )
    return summary


def render_report(summary: ValidationSummary) -> str:
    """Format outcomes for the disclosure log. Contains no key material."""
    lines = [
        "credential validation report",
        f"  checked : {summary.total}",
        f"  live    : {summary.live}",
        f"  revoked : {summary.revoked}",
        f"  unknown : {summary.unknown} (abstained or unreachable)",
    ]
    if summary.live or summary.revoked:
        lines.append(f"  live rate: {100 * summary.live_rate:.0f}% of those checked")
    lines.append("")
    for outcome in summary.outcomes:
        lines.append(
            f"  [{outcome.state:7}] {outcome.record.provider.value:14} "
            f"{outcome.record.fingerprint} {outcome.record.package[:34]:34} "
            f"{outcome.note[:48]}"
        )
    return "\n".join(lines)


def unvalidatable_providers(records: Iterable[CredentialRecord]) -> set[Provider]:
    """Providers present in the input that have no introspection endpoint.

    Surfaced explicitly so the write-up can state which providers were
    abstained from rather than leaving the gap implicit in the totals.
    """
    from llm_apk_scanner.validators.key_validator import PROVIDER_INTROSPECTION

    return {
        r.provider
        for r in records
        if not (PROVIDER_INTROSPECTION.get(r.provider) or {}).get("url")
    }
