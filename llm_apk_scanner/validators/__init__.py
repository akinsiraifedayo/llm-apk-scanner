"""
Credential validators.

These exist as a deliberately separate module so that the scanner
core never accidentally exercises credentials. Validation is opt-in
and is performed only via *non-revocable, idempotent introspection*
endpoints (e.g. `GET /v1/models` for OpenAI).

Validators are NEVER called automatically as part of `scan_apk`.
They are invoked only by an explicit CLI flag and only against a
manually-curated subset.
"""

from llm_apk_scanner.validators.key_validator import (
    KeyValidationResult,
    KeyValidator,
)
from llm_apk_scanner.validators.runner import (
    CredentialRecord,
    ValidationOutcome,
    ValidationSummary,
    render_report,
    unvalidatable_providers,
    validate_credentials,
)

__all__ = [
    "CredentialRecord",
    "KeyValidationResult",
    "KeyValidator",
    "ValidationOutcome",
    "ValidationSummary",
    "render_report",
    "unvalidatable_providers",
    "validate_credentials",
]
