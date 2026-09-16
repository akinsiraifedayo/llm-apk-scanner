"""Tests for batch credential validation.

The safety properties matter as much as the counting: a live call must be a
deliberate act, and someone else's credential must not be probed repeatedly.
"""

from __future__ import annotations

import pytest

from llm_apk_scanner.models import Provider
from llm_apk_scanner.validators.key_validator import KeyValidationResult
from llm_apk_scanner.validators.runner import (
    CredentialRecord,
    render_report,
    unvalidatable_providers,
    validate_credentials,
)


class _RecordingValidator:
    """Records every call and returns a scripted verdict."""

    def __init__(self, verdicts: dict[str, bool | None] | None = None) -> None:
        self.calls: list[tuple[Provider, str, bool]] = []
        self.verdicts = verdicts or {}

    def validate(self, provider, key, dry_run=True, timeout_seconds=10.0):
        self.calls.append((provider, key, dry_run))
        verdict = self.verdicts.get(key)
        return KeyValidationResult(
            provider=provider,
            valid=verdict,
            http_status=200 if verdict else None,
            note="scripted",
        )


def _rec(key: str, provider: Provider = Provider.OPENAI, pkg: str = "com.x"):
    return CredentialRecord(provider=provider, key=key, package=pkg)


class TestSafetyProperties:
    def test_dry_run_is_the_default(self):
        v = _RecordingValidator()
        validate_credentials([_rec("k1")], validator=v)
        assert v.calls[0][2] is True, "a live call must be opt-in"

    def test_live_flag_opts_in(self):
        v = _RecordingValidator()
        validate_credentials([_rec("k1")], live=True, validator=v)
        assert v.calls[0][2] is False

    def test_each_distinct_key_is_checked_once(self):
        """A key in twenty APKs is still someone's credential, checked once."""
        v = _RecordingValidator()
        records = [_rec("same", pkg=f"com.app{i}") for i in range(20)]
        summary = validate_credentials(records, live=True, validator=v)
        assert len(v.calls) == 1
        assert summary.total == 20

    def test_distinct_keys_are_each_checked(self):
        v = _RecordingValidator()
        validate_credentials([_rec("a"), _rec("b"), _rec("a")], live=True, validator=v)
        assert len({k for _, k, _ in v.calls}) == 2
        assert len(v.calls) == 2


class TestCounting:
    def _summary(self):
        v = _RecordingValidator({"live1": True, "live2": True, "dead": False})
        records = [_rec("live1"), _rec("live2"), _rec("dead"), _rec("abstain")]
        return validate_credentials(records, live=True, validator=v)

    def test_states_are_counted(self):
        s = self._summary()
        assert (s.live, s.revoked, s.unknown, s.total) == (2, 1, 1, 4)

    def test_live_rate_excludes_unknowns_from_denominator(self):
        """Abstention is our limitation, not evidence either way."""
        s = self._summary()
        assert s.live_rate == pytest.approx(2 / 3)

    def test_live_rate_is_zero_when_nothing_was_checked(self):
        v = _RecordingValidator()
        s = validate_credentials([_rec("x")], validator=v)
        assert s.live_rate == 0.0


class TestDisclosureSelection:
    def test_revoked_keys_are_not_reported(self):
        """Reporting dead keys burns the disclosure channel."""
        v = _RecordingValidator({"dead": False, "live": True})
        s = validate_credentials([_rec("dead"), _rec("live")], live=True, validator=v)
        assert [o.record.key for o in s.to_disclose()] == ["live"]

    def test_unknown_state_keys_are_still_reported(self):
        """We could not check it; the vendor can. Silence would be a guess."""
        v = _RecordingValidator({"dead": False})
        s = validate_credentials(
            [_rec("dead"), _rec("unchecked", Provider.ANTHROPIC)],
            live=True,
            validator=v,
        )
        assert [o.record.provider for o in s.to_disclose()] == [Provider.ANTHROPIC]


class TestRedaction:
    def test_fingerprint_is_stable_and_not_the_key(self):
        a, b = _rec("secret-value"), _rec("secret-value", pkg="com.y")
        assert a.fingerprint == b.fingerprint
        assert "secret" not in a.fingerprint
        assert len(a.fingerprint) == 16

    def test_report_contains_no_key_material(self):
        v = _RecordingValidator({"sk-supersecretvalue123": True})
        s = validate_credentials(
            [_rec("sk-supersecretvalue123")], live=True, validator=v
        )
        assert "sk-supersecretvalue123" not in render_report(s)

    def test_repr_does_not_leak_the_key(self):
        assert "secret-value" not in repr(_rec("secret-value"))


def test_unvalidatable_providers_are_surfaced():
    """Providers without introspection endpoints are surfaced."""
    providers = unvalidatable_providers(
        [_rec("a", Provider.OPENAI), _rec("b", Provider.COHERE)]
    )
    assert Provider.COHERE in providers
    assert Provider.OPENAI not in providers
