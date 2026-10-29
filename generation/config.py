"""Configuration for the generation layer, read from the environment.

Every setting is a ``GROUNDED_RAG_*`` variable.  Values are read from
``~/.config/grounded-rag/env`` first (``CLAVE=valor`` lines, read with the
standard library only) and then from the real environment, which wins.

The API key and the IP salt are secrets: this module never logs them, its
``repr`` masks them and :meth:`Config.public` is the only shape meant for
reports.  See ``tests/test_config.py``.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Mapping

ENV_FILE = Path("~/.config/grounded-rag/env").expanduser()
PREFIX = "GROUNDED_RAG_"
MASK = "***"

DEFAULT_BASE_URL = ""
DEFAULT_MODEL = ""
DEFAULT_DAILY_TOKEN_CAP = 200_000
DEFAULT_RATE_PER_IP = "10/hour"
DEFAULT_PORT = 8790
USAGE_DB = Path(__file__).resolve().parent.parent / "data" / "usage.sqlite"


class ConfigError(ValueError):
    """A setting is malformed."""


class LLMNotConfigured(ConfigError):
    """The generation model is not configured (no network was used)."""


def read_env_file(path: Path | str = ENV_FILE) -> dict[str, str]:
    """Parse a ``CLAVE=valor`` file; blank lines and ``#`` comments are skipped.

    No dependency is used for this.  Surrounding single/double quotes are
    removed, ``export`` prefixes are tolerated and lines without ``=`` are
    ignored instead of failing the whole file.
    """
    values: dict[str, str] = {}
    path = Path(path)
    if not path.exists():
        return values
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        key, separator, value = line.partition("=")
        if not separator:
            continue
        key = key.strip()
        if not key:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[key] = value
    return values


def parse_bool(value: str, *, name: str) -> bool:
    lowered = str(value).strip().lower()
    if lowered in {"on", "1", "true", "yes", "y"}:
        return True
    if lowered in {"off", "0", "false", "no", "n"}:
        return False
    raise ConfigError(f"{name} must be 'on'/'off' (got {value!r})")


RATE_RE = re.compile(r"^\s*(\d+)\s*/\s*([a-zA-Z]+)\s*$")
RATE_UNITS = {
    "s": 1.0,
    "sec": 1.0,
    "second": 1.0,
    "seconds": 1.0,
    "m": 60.0,
    "min": 60.0,
    "minute": 60.0,
    "minutes": 60.0,
    "h": 3600.0,
    "hour": 3600.0,
    "hours": 3600.0,
    "d": 86400.0,
    "day": 86400.0,
    "days": 86400.0,
}


def parse_rate(text: str, *, name: str = "GROUNDED_RAG_RATE_PER_IP") -> tuple[int, float]:
    """Parse ``"10/hour"`` into ``(10, 3600.0)``."""
    match = RATE_RE.match(str(text))
    if match is None:
        raise ConfigError(f"{name} must look like '10/hour' (got {text!r})")
    limit = int(match.group(1))
    unit = match.group(2).lower()
    if unit not in RATE_UNITS:
        raise ConfigError(f"{name} has an unknown time unit {unit!r} (use second/minute/hour/day)")
    if limit <= 0:
        raise ConfigError(f"{name} must allow at least one request")
    return limit, RATE_UNITS[unit]


#: ``GROUNDED_RAG_*`` variable -> :class:`Config` field.
ENV_FIELDS = {
    f"{PREFIX}LLM_BASE_URL": "llm_base_url",
    f"{PREFIX}LLM_API_KEY": "llm_api_key",
    f"{PREFIX}LLM_MODEL": "llm_model",
    f"{PREFIX}GENERATION": "generation",
    f"{PREFIX}DAILY_TOKEN_CAP": "daily_token_cap",
    f"{PREFIX}RATE_PER_IP": "rate_per_ip",
    f"{PREFIX}IP_SALT": "ip_salt",
    f"{PREFIX}PORT": "port",
    # Test-only switch: ``GROUNDED_RAG_LLM_FAKE=mock`` uses the deterministic
    # offline model (see generation.mock) instead of a network client.
    f"{PREFIX}LLM_FAKE": "llm_fake",
}

SECRET_FIELDS = ("llm_api_key", "ip_salt")


@dataclass(frozen=True)
class Config:
    """Resolved settings.  ``repr``/``str`` mask the API key and the IP salt."""

    llm_base_url: str = DEFAULT_BASE_URL
    llm_api_key: str = ""
    llm_model: str = DEFAULT_MODEL
    generation: bool = True
    daily_token_cap: int = DEFAULT_DAILY_TOKEN_CAP
    rate_per_ip: str = DEFAULT_RATE_PER_IP
    ip_salt: str = ""
    port: int = DEFAULT_PORT
    llm_fake: str = ""
    usage_db: Path = USAGE_DB

    # ------------------------------------------------------------- secrets
    @property
    def llm_configured(self) -> bool:
        """True when a base URL, a model and an API key are all present."""
        return bool(self.llm_base_url and self.llm_model and self.llm_api_key)

    def require_llm(self) -> None:
        """Raise :class:`LLMNotConfigured` (no network) when the model is missing."""
        missing = [
            name
            for name, value in (
                ("GROUNDED_RAG_LLM_BASE_URL", self.llm_base_url),
                ("GROUNDED_RAG_LLM_MODEL", self.llm_model),
                ("GROUNDED_RAG_LLM_API_KEY", self.llm_api_key),
            )
            if not value
        ]
        if missing:
            raise LLMNotConfigured(
                "the generation model is not configured: "
                + ", ".join(missing)
                + " is empty (set it in the environment or in ~/.config/grounded-rag/env)"
            )

    def rate_limit(self) -> tuple[int, float]:
        return parse_rate(self.rate_per_ip)

    @property
    def masked(self) -> dict[str, Any]:
        """Public view for logs/reports: secrets become ``***`` (or '' if unset)."""
        return {
            name: (MASK if getattr(self, name) else "") for name in SECRET_FIELDS
        }

    def public(self) -> dict[str, Any]:
        """Serializable settings without secrets (safe for reports and logs)."""
        data: dict[str, Any] = {}
        for field in fields(self):
            value = getattr(self, field.name)
            if field.name in SECRET_FIELDS:
                data[field.name] = bool(value)
            elif isinstance(value, Path):
                data[field.name] = str(value)
            else:
                data[field.name] = value
        return data

    def __repr__(self) -> str:
        parts = []
        for field in fields(self):
            value = getattr(self, field.name)
            if field.name in SECRET_FIELDS:
                value = MASK if value else ""
            parts.append(f"{field.name}={value!r}")
        return f"Config({', '.join(parts)})"

    __str__ = __repr__


def load_config(
    environ: Mapping[str, str] | None = None,
    env_file: Path | str | None = ENV_FILE,
) -> Config:
    """Build :class:`Config` from the env file plus the real environment.

    The environment wins over the file; unknown ``GROUNDED_RAG_*`` variables are
    ignored.  A malformed value raises :class:`ConfigError`.
    """
    environ = os.environ if environ is None else environ
    values = read_env_file(env_file) if env_file is not None else {}
    for env_name, field_name in ENV_FIELDS.items():
        if env_name in environ:
            values[env_name] = environ[env_name]

    kwargs: dict[str, Any] = {}
    if f"{PREFIX}GENERATION" in values:
        kwargs["generation"] = parse_bool(values[f"{PREFIX}GENERATION"], name=f"{PREFIX}GENERATION")
    if f"{PREFIX}DAILY_TOKEN_CAP" in values:
        raw = values[f"{PREFIX}DAILY_TOKEN_CAP"].strip()
        try:
            kwargs["daily_token_cap"] = int(raw)
        except ValueError as error:
            raise ConfigError(f"{PREFIX}DAILY_TOKEN_CAP must be an integer (got {raw!r})") from error
        if kwargs["daily_token_cap"] < 0:
            raise ConfigError(f"{PREFIX}DAILY_TOKEN_CAP cannot be negative")
    if f"{PREFIX}PORT" in values:
        raw = values[f"{PREFIX}PORT"].strip()
        try:
            kwargs["port"] = int(raw)
        except ValueError as error:
            raise ConfigError(f"{PREFIX}PORT must be an integer (got {raw!r})") from error
    if f"{PREFIX}RATE_PER_IP" in values:
        parse_rate(values[f"{PREFIX}RATE_PER_IP"])  # fail fast on a malformed rate
        kwargs["rate_per_ip"] = values[f"{PREFIX}RATE_PER_IP"].strip()

    for env_name, field_name in ENV_FIELDS.items():
        if field_name in kwargs or field_name in {"generation", "daily_token_cap", "port", "rate_per_ip"}:
            continue
        if env_name in values:
            kwargs[field_name] = values[env_name].strip()

    if "llm_fake" in kwargs and kwargs["llm_fake"].lower() not in {"", "mock"}:
        raise ConfigError("GROUNDED_RAG_LLM_FAKE only accepts 'mock'")
    return Config(**kwargs)
