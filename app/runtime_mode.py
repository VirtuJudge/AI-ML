"""Shared provider and environment mode resolution policy for VirtuJudge AI-ML."""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from typing import Literal

ProviderMode = Literal["fake", "mixed", "live", "groq"]
CanonicalProviderMode = Literal["fake", "mixed", "live"]

LIVE_PROVIDER_MODES: frozenset[str] = frozenset({"live", "groq"})
FAKE_PROVIDER_MODES: frozenset[str] = frozenset({"fake"})
MIXED_PROVIDER_MODES: frozenset[str] = frozenset({"mixed"})
_PROVIDER_MODE_MAP: dict[str, ProviderMode] = {
    "fake": "fake",
    "mixed": "mixed",
    "live": "live",
    "groq": "groq",
}
SUPPORTED_PROVIDER_MODES: frozenset[str] = frozenset(_PROVIDER_MODE_MAP)
PRODUCTION_ENV_NAMES: frozenset[str] = frozenset({"production", "prod"})
TEST_ENV_NAMES: frozenset[str] = frozenset({"test", "testing"})


@dataclass(frozen=True)
class RuntimeModePolicy:
    """Normalized provider and environment policy shared across worker startup and stages."""

    provider_mode: ProviderMode
    provider_mode_explicit: bool
    environment: str
    is_live: bool
    is_mixed: bool
    is_fake: bool
    is_production: bool
    is_test_env: bool
    allow_synthetic_report_fixtures: bool

    @property
    def canonical_mode(self) -> CanonicalProviderMode:
        """Return canonical provider mode ('live' for both 'live' and 'groq')."""
        if self.is_live:
            return "live"
        if self.is_mixed:
            return "mixed"
        return "fake"

    def with_synthetic_fixtures(self, allow: bool) -> RuntimeModePolicy:
        """Return a policy copy with an explicit synthetic-fixture setting."""
        if allow and (self.is_live or self.is_mixed or self.is_production):
            raise ValueError(
                f"Synthetic fixtures cannot be enabled in {self.provider_mode!r} mode "
                f"(environment={self.environment!r}, production={self.is_production})."
            )
        return replace(self, allow_synthetic_report_fixtures=allow)


def resolve_runtime_mode(
    provider_mode: str | None = None,
    *,
    app_env: str | None = None,
    allow_synthetic_fixtures: bool | None = None,
) -> RuntimeModePolicy:
    """Resolve and normalize AI_PROVIDER_MODE and environment configuration once."""
    if provider_mode is None:
        raw_mode = os.getenv("AI_PROVIDER_MODE")
        if raw_mode is None:
            normalized_mode = "fake"
            provider_mode_explicit = False
        else:
            normalized_mode = raw_mode.strip().lower()
            provider_mode_explicit = True
    else:
        normalized_mode = provider_mode.strip().lower()
        provider_mode_explicit = True

    typed_mode = _PROVIDER_MODE_MAP.get(normalized_mode)
    if typed_mode is None:
        raise ValueError("AI_PROVIDER_MODE must be one of: fake, mixed, live, or groq.")

    is_live = typed_mode in LIVE_PROVIDER_MODES
    is_mixed = typed_mode in MIXED_PROVIDER_MODES
    is_fake = typed_mode in FAKE_PROVIDER_MODES

    if app_env is not None:
        raw_env = app_env
    else:
        raw_env = os.getenv("APP_ENV") or os.getenv("ENVIRONMENT") or "development"
    environment = raw_env.strip().lower() or "development"

    is_production = environment in PRODUCTION_ENV_NAMES or bool(os.getenv("SPACE_ID", "").strip())
    is_test_env = environment in TEST_ENV_NAMES or bool(os.getenv("PYTEST_CURRENT_TEST"))

    if allow_synthetic_fixtures is not None:
        if allow_synthetic_fixtures and (is_live or is_mixed or is_production):
            raise ValueError(
                f"Synthetic fixtures cannot be enabled in {typed_mode!r} mode "
                f"(environment={environment!r}, production={is_production})."
            )
        allow_synthetic = allow_synthetic_fixtures
    else:
        allow_synthetic = (
            not is_live
            and not is_mixed
            and not is_production
            and (is_test_env or (provider_mode_explicit and is_fake))
        )

    return RuntimeModePolicy(
        provider_mode=typed_mode,
        provider_mode_explicit=provider_mode_explicit,
        environment=environment,
        is_live=is_live,
        is_mixed=is_mixed,
        is_fake=is_fake,
        is_production=is_production,
        is_test_env=is_test_env,
        allow_synthetic_report_fixtures=allow_synthetic,
    )


__all__ = [
    "FAKE_PROVIDER_MODES",
    "LIVE_PROVIDER_MODES",
    "MIXED_PROVIDER_MODES",
    "PRODUCTION_ENV_NAMES",
    "SUPPORTED_PROVIDER_MODES",
    "TEST_ENV_NAMES",
    "CanonicalProviderMode",
    "ProviderMode",
    "RuntimeModePolicy",
    "resolve_runtime_mode",
]
