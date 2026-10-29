"""Configuration for the generation layer, read from the environment.

Every setting is a ``GROUNDED_RAG_*`` variable.  Values are read from
``~/.config/grounded-rag/env`` first (``CLAVE=valor`` lines, read with the
standard library only) and then from the real environment, which wins.

The API key and the IP salt are secrets: this module never logs them, its
``repr`` masks them and :meth:`Config.public` is the only shape meant for
reports.  The key can also live outside the environment, in a JSON file whose
location and dotted path are settings (``GROUNDED_RAG_LLM_KEY_FILE`` plus
``GROUNDED_RAG_LLM_KEY_PATH``); that file is read at call time by
:func:`resolve_llm_api_key`, never at import time and never into
:class:`Config`.  See ``tests/test_config.py``.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Mapping

ENV_FILE = Path("~/.config/grounded-rag/env").expanduser()
PREFIX = "GROUNDED_RAG_"
MASK = "***"

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "deepseek/deepseek-v4.1-flash"
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
    f"{PREFIX}LLM_KEY_FILE": "llm_key_file",
    f"{PREFIX}LLM_KEY_PATH": "llm_key_path",
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
    llm_key_file: str = ""
    llm_key_path: str = ""
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
    def key_file_ready(self) -> bool:
        """True when both file variables are set and the file exists (it is not read)."""
        if not (self.llm_key_file and self.llm_key_path):
            return False
        return Path(self.llm_key_file).expanduser().is_file()

    def describe_key_problem(self) -> str:
        """Why there is no API key, naming variables and paths — never a value."""
        if not (self.llm_key_file or self.llm_key_path):
            return "GROUNDED_RAG_LLM_API_KEY is empty"
        if not (self.llm_key_file and self.llm_key_path):
            missing = (
                "GROUNDED_RAG_LLM_KEY_PATH" if self.llm_key_file else "GROUNDED_RAG_LLM_KEY_FILE"
            )
            return (
                f"{missing} is missing, so the key file is unusable "
                f"GROUNDED_RAG_LLM_KEY_FILE={self.llm_key_file or '(unset)'} "
                f"GROUNDED_RAG_LLM_KEY_PATH={self.llm_key_path or '(unset)'}"
            )
        path = Path(self.llm_key_file).expanduser()
        if not path.is_file():
            return (
                f"GROUNDED_RAG_LLM_KEY_FILE points at {path}, which does not exist "
                f"(and GROUNDED_RAG_LLM_API_KEY is empty)"
            )
        return ""

    @property
    def llm_configured(self) -> bool:
        """True when a base URL, a model and a key (env or key file) are all present."""
        return bool(
            self.llm_base_url
            and self.llm_model
            and (self.llm_api_key or self.key_file_ready)
        )

    def require_llm(self) -> None:
        """Raise :class:`LLMNotConfigured` (no network) when the model is missing."""
        problems = [
            f"{name} is empty"
            for name, value in (
                ("GROUNDED_RAG_LLM_BASE_URL", self.llm_base_url),
                ("GROUNDED_RAG_LLM_MODEL", self.llm_model),
            )
            if not value
        ]
        if not self.llm_api_key and not self.key_file_ready:
            problems.append(self.describe_key_problem())
        if problems:
            raise LLMNotConfigured(
                "the generation model is not configured: "
                + "; ".join(problems)
                + " (set it in the environment or in ~/.config/grounded-rag/env)"
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


def read_key_from_file(file_path: str | Path, dotted_path: str) -> str:
    """Return the key at ``dotted_path`` inside the JSON file ``file_path``.

    Called at runtime by :func:`resolve_llm_api_key` (never at import time).
    Every failure raises :class:`LLMNotConfigured` naming the variable and the
    path; the value and the file contents never reach the message.
    """
    path = Path(file_path).expanduser()
    name = f"{PREFIX}LLM_KEY_FILE"
    path_name = f"{PREFIX}LLM_KEY_PATH"
    if not path.is_file():
        raise LLMNotConfigured(f"{name} points at {path}, which does not exist")
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        raise LLMNotConfigured(f"{name}: {path} cannot be read") from None
    try:
        data = json.loads(text)
    except ValueError:
        # The JSON decoder's message can quote the file: keep it out.
        raise LLMNotConfigured(f"{name}: {path} is not valid JSON") from None

    node: Any = data
    for part in dotted_path.split("."):
        if not part or not isinstance(node, dict) or part not in node:
            raise LLMNotConfigured(
                f"{path_name}: no such path {dotted_path!r} in {path} ({name})"
            )
        node = node[part]

    if not isinstance(node, str) or not node.strip():
        raise LLMNotConfigured(
            f"{path_name}: the value at {dotted_path!r} in {path} is empty or not a string ({name})"
        )
    return node.strip()


def resolve_llm_api_key(config: Config) -> str:
    """The API key for ``config``: the environment one wins over the key file.

    The file route is read here, at call time, and returned straight to the
    client: it is not stored on :class:`Config` nor written anywhere.
    """
    if config.llm_api_key:
        return config.llm_api_key
    if not (config.llm_key_file and config.llm_key_path):
        raise LLMNotConfigured(
            "the generation model is not configured: " + config.describe_key_problem()
        )
    return read_key_from_file(config.llm_key_file, config.llm_key_path)
