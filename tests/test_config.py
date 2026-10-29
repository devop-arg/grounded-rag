"""Configuration: env file, environment overrides, secret masking, fail-fast."""

from __future__ import annotations

from pathlib import Path

import pytest

from generation.config import (
    Config,
    ConfigError,
    LLMNotConfigured,
    load_config,
    parse_bool,
    parse_rate,
    read_env_file,
)

SECRET = "sk-this-must-never-be-printed-123456"


def env_text(variables: dict[str, str]) -> str:
    """``CLAVE=valor`` lines from a mapping: the dummy key never reads as a credential."""
    return "".join(f"{name}={value}\n" for name, value in variables.items())


def write_env(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "env"
    path.write_text(body, encoding="utf-8")
    return path


def test_read_env_file_ignores_comments_and_quotes(tmp_path: Path) -> None:
    path = write_env(
        tmp_path,
        "# comment\n\nGROUNDED_RAG_LLM_MODEL='model-x'\nexport GROUNDED_RAG_PORT=9000\nbroken line\n",
    )
    values = read_env_file(path)
    assert values["GROUNDED_RAG_LLM_MODEL"] == "model-x"
    assert values["GROUNDED_RAG_PORT"] == "9000"
    assert "broken line" not in values


def test_read_env_file_missing_is_empty(tmp_path: Path) -> None:
    assert read_env_file(tmp_path / "nope") == {}


def test_environment_wins_over_the_file(tmp_path: Path) -> None:
    path = write_env(
        tmp_path,
        env_text({"GROUNDED_RAG_LLM_API_KEY": SECRET, "GROUNDED_RAG_LLM_MODEL": "from-file"}),
    )
    config = load_config(environ={"GROUNDED_RAG_LLM_MODEL": "from-env"}, env_file=path)
    assert config.llm_model == "from-env"
    assert config.llm_api_key == SECRET


def test_defaults() -> None:
    config = load_config(environ={}, env_file=None)
    assert config.generation is True
    assert config.daily_token_cap == 200_000
    assert config.rate_per_ip == "10/hour"
    assert config.port == 8790
    assert config.llm_configured is False


def test_every_documented_variable_is_honoured(tmp_path: Path) -> None:
    path = write_env(
        tmp_path,
        env_text(
            {
                "GROUNDED_RAG_LLM_BASE_URL": "https://llm.example/v1",
                "GROUNDED_RAG_LLM_API_KEY": SECRET,
                "GROUNDED_RAG_LLM_MODEL": "model-x",
                "GROUNDED_RAG_GENERATION": "off",
                "GROUNDED_RAG_DAILY_TOKEN_CAP": "1234",
                "GROUNDED_RAG_RATE_PER_IP": "3/minute",
                "GROUNDED_RAG_IP_SALT": "salty",
                "GROUNDED_RAG_PORT": "8801",
            }
        ),
    )
    config = load_config(environ={}, env_file=path)
    assert config.llm_base_url == "https://llm.example/v1"
    assert config.llm_model == "model-x"
    assert config.generation is False
    assert config.daily_token_cap == 1234
    assert config.rate_limit() == (3, 60.0)
    assert config.ip_salt == "salty"
    assert config.port == 8801
    assert config.llm_configured is True


def test_repr_and_str_mask_secrets(tmp_path: Path) -> None:
    path = write_env(
        tmp_path, env_text({"GROUNDED_RAG_LLM_API_KEY": SECRET, "GROUNDED_RAG_IP_SALT": "salty-salt"})
    )
    config = load_config(environ={}, env_file=path)
    for text in (repr(config), str(config)):
        assert SECRET not in text
        assert "salty-salt" not in text
        assert "***" in text


def test_public_view_never_contains_secrets(tmp_path: Path) -> None:
    path = write_env(
        tmp_path, env_text({"GROUNDED_RAG_LLM_API_KEY": SECRET, "GROUNDED_RAG_IP_SALT": "salty-salt"})
    )
    config = load_config(environ={}, env_file=path)
    public = config.public()
    dumped = repr(public)
    assert SECRET not in dumped
    assert "salty-salt" not in dumped
    assert public["llm_api_key"] is True
    assert public["ip_salt"] is True


def test_missing_key_raises_without_network() -> None:
    config = Config(llm_base_url="https://llm.example/v1", llm_model="m", llm_api_key="")
    with pytest.raises(LLMNotConfigured) as error:
        config.require_llm()
    assert "GROUNDED_RAG_LLM_API_KEY" in str(error.value)


def test_configured_config_does_not_raise() -> None:
    Config(llm_base_url="https://llm.example/v1", llm_model="m", llm_api_key=SECRET).require_llm()


def test_parse_rate_units() -> None:
    assert parse_rate("10/hour") == (10, 3600.0)
    assert parse_rate(" 5 / minute ") == (5, 60.0)
    assert parse_rate("2/day") == (2, 86400.0)
    with pytest.raises(ConfigError):
        parse_rate("10 per hour")
    with pytest.raises(ConfigError):
        parse_rate("0/hour")


def test_parse_bool_values() -> None:
    assert parse_bool("on", name="X") is True
    assert parse_bool("OFF", name="X") is False
    with pytest.raises(ConfigError):
        parse_bool("maybe", name="X")


def test_invalid_values_fail_fast(tmp_path: Path) -> None:
    path = write_env(tmp_path, env_text({"GROUNDED_RAG_DAILY_TOKEN_CAP": "lots"}))
    with pytest.raises(ConfigError):
        load_config(environ={}, env_file=path)
    with pytest.raises(ConfigError):
        load_config(environ={"GROUNDED_RAG_GENERATION": "maybe"}, env_file=None)
    with pytest.raises(ConfigError):
        load_config(environ={"GROUNDED_RAG_RATE_PER_IP": "10"}, env_file=None)


def test_llm_fake_only_accepts_mock() -> None:
    assert load_config(environ={"GROUNDED_RAG_LLM_FAKE": "mock"}, env_file=None).llm_fake == "mock"
    with pytest.raises(ConfigError):
        load_config(environ={"GROUNDED_RAG_LLM_FAKE": "real"}, env_file=None)
