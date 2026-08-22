"""Pytest configuration and test fixtures."""

import pytest

from tribune.config import reset_settings_cache


@pytest.fixture(autouse=True)
def clean_settings_cache():
    """Ensure settings cache is clean before and after every test."""
    reset_settings_cache()
    yield
    reset_settings_cache()
