"""
Non-revocable key validation.

Ethics policy (proposal §8.2): validity is confirmed only through
endpoints that:
  1. Do not generate content.
  2. Do not consume billable inference quota.
  3. Are idempotent and rate-limited by the provider.
  4. Have no side effects observable by the developer beyond log
     entries describing an introspection call.

For most providers these are the model-listing endpoints.

This module DOES NOT make HTTP calls when imported. It is opt-in
via `KeyValidator.validate(...)`. The default `dry_run=True` mode
stops at building the request URL and headers, so the rest of the
pipeline can be tested without network access at all.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from llm_apk_scanner.models import Provider

if TYPE_CHECKING:
    pass  # urllib imported lazily inside validate()


@dataclass
class KeyValidationResult:
    provider: Provider
    valid: bool | None  # None means we didn't actually call out
    http_status: int | None
    note: str


# Map provider -> introspection endpoint metadata.
PROVIDER_INTROSPECTION = {
    Provider.OPENAI: {
        "url": "https://api.openai.com/v1/models",
        "auth_header": "Authorization",
        "auth_format": "Bearer {key}",
    },
    Provider.ANTHROPIC: {
        # GET /v1/models — confirmed live 2026-08-12 per platform.claude.com/docs.
        # Safe introspection: no content generation, no billing, idempotent.
        "url": "https://api.anthropic.com/v1/models?limit=1",
        "auth_header": "X-Api-Key",
        "auth_format": "{key}",
        "extra_headers": {"anthropic-version": "2023-06-01"},
    },
    Provider.GOOGLE_GEMINI: {
        "url": "https://generativelanguage.googleapis.com/v1beta/models",
        "auth_header": None,  # API-key on query string
        "auth_format": "?key={key}",
    },
    Provider.HUGGINGFACE: {
        "url": "https://huggingface.co/api/whoami-v2",
        "auth_header": "Authorization",
        "auth_format": "Bearer {key}",
    },
}


class KeyValidator:
    """Confirms key validity through non-revocable introspection."""

    def validate(
        self,
        provider: Provider,
        key: str,
        dry_run: bool = True,
        timeout_seconds: float = 10.0,
    ) -> KeyValidationResult:
        meta = PROVIDER_INTROSPECTION.get(provider)
        if not meta or not meta.get("url"):
            return KeyValidationResult(
                provider=provider,
                valid=None,
                http_status=None,
                note="No non-revocable introspection endpoint available "
                "for this provider; abstaining.",
            )

        if dry_run:
            return KeyValidationResult(
                provider=provider,
                valid=None,
                http_status=None,
                note=f"DRY RUN: would GET {meta['url']}",
            )

        # Real call. Imported lazily so the validator can be imported
        # in environments without network libraries.
        import urllib.error
        import urllib.request

        url = meta["url"]
        headers: dict[str, str] = {"User-Agent": "llm-apk-scanner-research/0.1"}
        # Add any extra headers (e.g. anthropic-version).
        headers.update(meta.get("extra_headers", {}))
        if meta["auth_header"]:
            headers[meta["auth_header"]] = meta["auth_format"].format(key=key)
        else:
            # API-key on query string variant.
            url = url + meta["auth_format"].format(key=key)

        req = urllib.request.Request(url, headers=headers, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
                status = resp.status
                return KeyValidationResult(
                    provider=provider,
                    valid=200 <= status < 300,
                    http_status=status,
                    note="Confirmed via introspection endpoint.",
                )
        except urllib.error.HTTPError as e:
            return KeyValidationResult(
                provider=provider,
                valid=False if e.code in (401, 403) else None,
                http_status=e.code,
                note=f"HTTP {e.code} from introspection.",
            )
        except urllib.error.URLError as e:
            return KeyValidationResult(
                provider=provider,
                valid=None,
                http_status=None,
                note=f"Network error: {e.reason}",
            )
