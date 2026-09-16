"""Shared pytest fixtures."""

from __future__ import annotations

import pytest

from tests.fixtures.seeded_strings import (
    NEGATIVE_PROMPTS,
    NEGATIVE_SECRETS,
    POSITIVE_PROMPTS,
    POSITIVE_RAG_PATHS,
    POSITIVE_SECRETS,
    POSITIVE_TOOL_SCHEMAS,
)


@pytest.fixture
def positive_secrets() -> list[tuple[str, str]]:
    return POSITIVE_SECRETS


@pytest.fixture
def negative_secrets() -> list[tuple[str, str]]:
    return NEGATIVE_SECRETS


@pytest.fixture
def positive_prompts() -> list[str]:
    return POSITIVE_PROMPTS


@pytest.fixture
def negative_prompts() -> list[str]:
    return NEGATIVE_PROMPTS


@pytest.fixture
def positive_tool_schemas() -> list[str]:
    return POSITIVE_TOOL_SCHEMAS


@pytest.fixture
def positive_rag_paths() -> list[str]:
    return POSITIVE_RAG_PATHS
