"""Shared fixtures: no test reads the configuration files of the machine it runs on."""

from __future__ import annotations

import pytest

import generation.config as config_module


@pytest.fixture(autouse=True)
def _isolated_from_this_machine(monkeypatch: pytest.MonkeyPatch) -> None:
    """Never open the real ``.env.local`` nor the user env file of this machine."""
    monkeypatch.setattr(config_module, "LOCAL_ENV_FILE", None)
    monkeypatch.setattr(config_module, "ENV_FILE", None)
