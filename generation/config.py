"""Configuration for the generation layer, read from the environment.

Every setting is a ``GROUNDED_RAG_*`` variable.  Values are read, in this
order, from ``~/.config/grounded-rag/env``, then from ``.env.local`` at the
root of this repository and finally from the real environment, which wins.
Both files hold ``CLAVE=valor`` lines and are read with the standard library
only; the repository root comes from this module's location, not from the
current directory.  An empty value means "not set", so a copy of
``.env.example`` with every value blank behaves exactly like no file at all.

The endpoint, the model and the key are never hardcoded: they only come from
``GROUNDED_RAG_LLM_BASE_URL``, ``GROUNDED_RAG_LLM_MODEL`` and
``GROUNDED_RAG_LLM_API_KEY`` (or the key file pair below).

The API key and the IP salt are secrets: this module never logs them, its
``repr`` masks them and :meth:`Config.public` is the only shape meant for
reports (it also drops the endpoint, every path and the model's provider
route).  The key can also live outside the environment, in a JSON file whose
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
REPO_ROOT = Path(__file__).resolve().parent.parent
#: Per-checkout overrides: git-ignored, same format as :data:`ENV_FILE`.
LOCAL_ENV_FILE = REPO_ROOT / ".env.local"
PREFIX = "GROUNDED_RAG_"
MASK = "***"

DEFAULT_BASE_URL = ""
DEFAULT_MODEL = ""
DEFAULT_PRICE_INPUT_PER_M = 0.0
DEFAULT_PRICE_OUTPUT_PER_M = 0.0
DEFAULT_BUDGET_USD_PER_MONTH = 0.50
DEFAULT_RATE_PER_IP = "10/hour"
DEFAULT_PORT = 8790
USAGE_DB = REPO_ROOT / "data" / "usage.sqlite"


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


def parse_amount(value: str, *, name: str) -> float:
    """A non-negative amount: USD per million tokens, or USD for the budget."""
    try:
        amount = float(str(value).strip())
    except ValueError as error:
        raise ConfigError(f"{name} must be a non-negative number (got {value!r})") from error
    if amount < 0:
        raise ConfigError(f"{name} cannot be negative (got {value!r})")
    return amount


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
    f"{PREFIX}LLM_REFERER": "llm_referer",
    f"{PREFIX}LLM_TITLE": "llm_title",
    f"{PREFIX}LLM_PRICE_INPUT_PER_M": "llm_price_input_per_m",
    f"{PREFIX}LLM_PRICE_OUTPUT_PER_M": "llm_price_output_per_m",
    f"{PREFIX}BUDGET_USD_PER_MONTH": "budget_usd_per_month",
    f"{PREFIX}GENERATION": "generation",
    f"{PREFIX}RATE_PER_IP": "rate_per_ip",
    f"{PREFIX}IP_SALT": "ip_salt",
    f"{PREFIX}PORT": "port",
    # Test-only switch: ``GROUNDED_RAG_LLM_FAKE=mock`` uses the deterministic
    # offline model (see generation.mock) instead of a network client.
    f"{PREFIX}LLM_FAKE": "llm_fake",
}

#: Fields the loop below converts explicitly instead of taking the raw string.
TYPED_FIELDS = frozenset(
    {
        "generation",
        "port",
        "rate_per_ip",
        "llm_price_input_per_m",
        "llm_price_output_per_m",
        "budget_usd_per_month",
    }
)

SECRET_FIELDS = ("llm_api_key", "ip_salt")


@dataclass(frozen=True)
class Config:
    """Resolved settings.  ``repr``/``str`` mask the API key and the IP salt."""

    llm_base_url: str = DEFAULT_BASE_URL
    llm_api_key: str = ""
    llm_key_file: str = ""
    llm_key_path: str = ""
    llm_model: str = DEFAULT_MODEL
    llm_referer: str = ""
    llm_title: str = ""
    llm_price_input_per_m: float = DEFAULT_PRICE_INPUT_PER_M
    llm_price_output_per_m: float = DEFAULT_PRICE_OUTPUT_PER_M
    budget_usd_per_month: float = DEFAULT_BUDGET_USD_PER_MONTH
    generation: bool = True
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
                + " (set it in the environment, in .env.local or in ~/.config/grounded-rag/env)"
            )

    def rate_limit(self) -> tuple[int, float]:
        return parse_rate(self.rate_per_ip)

    def cost_usd(self, prompt_tokens: int, completion_tokens: int) -> float:
        """USD cost of one completion at the configured prices per million tokens."""
        return (
            int(prompt_tokens) * self.llm_price_input_per_m
            + int(completion_tokens) * self.llm_price_output_per_m
        ) / 1_000_000

    @property
    def masked(self) -> dict[str, Any]:
        """Public view for logs/reports: secrets become ``***`` (or '' if unset)."""
        return {
            name: (MASK if getattr(self, name) else "") for name in SECRET_FIELDS
        }

    def public(self) -> dict[str, Any]:
        """The only settings shape meant for reports and logs.

        No secret, no path and no host: just the model name (the part after the
        last ``/``, so a provider route is not published), the two prices per
        million tokens, the monthly budget and the ``generation`` switch.  The
        endpoint, the key file, the key path and the usage database stay out.
        """
        return {
            "llm_model": self.llm_model.rsplit("/", 1)[-1],
            "llm_price_input_per_m": self.llm_price_input_per_m,
            "llm_price_output_per_m": self.llm_price_output_per_m,
            "budget_usd_per_month": self.budget_usd_per_month,
            "generation": self.generation,
        }

    def __repr__(self) -> str:
        parts = []
        for field in fields(self):
            value = getattr(self, field.name)
            if field.name in SECRET_FIELDS:
                value = MASK if value else ""
            parts.append(f"{field.name}={value!r}")
        return f"Config({', '.join(parts)})"

    __str__ = __repr__


def _present(values: Mapping[str, str], name: str) -> str | None:
    """The value of ``name``, or ``None`` when it is absent or blank."""
    text = values.get(name)
    if text is None:
        return None
    text = text.strip()
    return text or None


class _Unset:
    """Sentinel: the argument was omitted, so the module default path applies."""

    def __repr__(self) -> str:  # pragma: no cover - debugging aid only
        return "<unset>"


_UNSET = _Unset()


def load_config(
    environ: Mapping[str, str] | None = None,
    env_file: Path | str | None | _Unset = _UNSET,
    local_env_file: Path | str | None | _Unset = _UNSET,
) -> Config:
    """Build :class:`Config` from the two env files plus the real environment.

    Precedence, highest first: the environment, ``.env.local`` at the root of
    the repository (:data:`LOCAL_ENV_FILE`, git-ignored) and the user file
    (:data:`ENV_FILE`, ``~/.config/grounded-rag/env``); the defaults of the
    code come last.  A missing file is ignored, ``None`` disables its layer and
    a blank value never shadows a lower layer, so a blank ``.env.example`` copy
    changes nothing.  Unknown ``GROUNDED_RAG_*`` variables are ignored and a
    malformed value raises :class:`ConfigError`.

    ``env_file`` and ``local_env_file`` are injectable for the tests; omitting
    them keeps the defaults above, and passing ``None`` skips that file.
    """
    environ = os.environ if environ is None else environ
    if isinstance(env_file, _Unset):
        env_file = ENV_FILE
    if isinstance(local_env_file, _Unset):
        local_env_file = LOCAL_ENV_FILE

    values: dict[str, str] = {}
    for path in (env_file, local_env_file):
        if path is None:
            continue
        for env_name, value in read_env_file(path).items():
            if value.strip():
                values[env_name] = value
    for env_name in ENV_FIELDS:
        if env_name in environ:
            values[env_name] = environ[env_name]

    kwargs: dict[str, Any] = {}
    generation = _present(values, f"{PREFIX}GENERATION")
    if generation is not None:
        kwargs["generation"] = parse_bool(generation, name=f"{PREFIX}GENERATION")
    port = _present(values, f"{PREFIX}PORT")
    if port is not None:
        try:
            kwargs["port"] = int(port)
        except ValueError as error:
            raise ConfigError(f"{PREFIX}PORT must be an integer (got {port!r})") from error
    rate = _present(values, f"{PREFIX}RATE_PER_IP")
    if rate is not None:
        parse_rate(rate)  # fail fast on a malformed rate
        kwargs["rate_per_ip"] = rate
    for env_name, field_name in (
        (f"{PREFIX}LLM_PRICE_INPUT_PER_M", "llm_price_input_per_m"),
        (f"{PREFIX}LLM_PRICE_OUTPUT_PER_M", "llm_price_output_per_m"),
        (f"{PREFIX}BUDGET_USD_PER_MONTH", "budget_usd_per_month"),
    ):
        amount = _present(values, env_name)
        if amount is not None:
            kwargs[field_name] = parse_amount(amount, name=env_name)

    for env_name, field_name in ENV_FIELDS.items():
        if field_name in TYPED_FIELDS or field_name in kwargs:
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
