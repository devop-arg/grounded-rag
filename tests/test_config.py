"""Configuration: env file, environment overrides, secret masking, fail-fast."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from generation.config import (
    DEFAULT_BASE_URL,
    DEFAULT_BUDGET_USD_PER_MONTH,
    DEFAULT_MODEL,
    DEFAULT_PRICE_INPUT_PER_M,
    DEFAULT_PRICE_OUTPUT_PER_M,
    Config,
    ConfigError,
    LLMNotConfigured,
    load_config,
    parse_amount,
    parse_bool,
    parse_rate,
    read_env_file,
    read_key_from_file,
    resolve_llm_api_key,
)

SECRET = "sk-this-must-never-be-printed-123456"
FILE_SECRET = "sk-only-in-the-json-file-abcdef987654"
BASE_URL = "https://llm.example/v1"
MODEL = "deepseek-v4.1-flash"


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
    assert config.rate_per_ip == "10/hour"
    assert config.port == 8790
    assert config.budget_usd_per_month == DEFAULT_BUDGET_USD_PER_MONTH == 0.5
    assert config.llm_price_input_per_m == DEFAULT_PRICE_INPUT_PER_M == 0.0
    assert config.llm_price_output_per_m == DEFAULT_PRICE_OUTPUT_PER_M == 0.0
    assert (config.llm_referer, config.llm_title) == ("", "")
    assert config.llm_configured is False


def test_the_shipped_env_example_changes_nothing(tmp_path: Path) -> None:
    """Every value in ``.env.example`` is empty: defaults apply, nothing breaks."""
    source = Path(__file__).resolve().parent.parent / ".env.example"
    config = load_config(environ={}, env_file=source)
    assert config == load_config(environ={}, env_file=None)


def test_every_documented_variable_is_honoured(tmp_path: Path) -> None:
    path = write_env(
        tmp_path,
        env_text(
            {
                "GROUNDED_RAG_LLM_BASE_URL": BASE_URL,
                "GROUNDED_RAG_LLM_API_KEY": SECRET,
                "GROUNDED_RAG_LLM_MODEL": "model-x",
                "GROUNDED_RAG_LLM_REFERER": "https://example.com/docs",
                "GROUNDED_RAG_LLM_TITLE": "example",
                "GROUNDED_RAG_LLM_PRICE_INPUT_PER_M": "0.15",
                "GROUNDED_RAG_LLM_PRICE_OUTPUT_PER_M": "0.6",
                "GROUNDED_RAG_BUDGET_USD_PER_MONTH": "2.5",
                "GROUNDED_RAG_GENERATION": "off",
                "GROUNDED_RAG_RATE_PER_IP": "3/minute",
                "GROUNDED_RAG_IP_SALT": "salty",
                "GROUNDED_RAG_PORT": "8801",
            }
        ),
    )
    config = load_config(environ={}, env_file=path)
    assert config.llm_base_url == BASE_URL
    assert config.llm_model == "model-x"
    assert config.llm_referer == "https://example.com/docs"
    assert config.llm_title == "example"
    assert config.llm_price_input_per_m == 0.15
    assert config.llm_price_output_per_m == 0.6
    assert config.budget_usd_per_month == 2.5
    assert config.generation is False
    assert config.rate_limit() == (3, 60.0)
    assert config.ip_salt == "salty"
    assert config.port == 8801
    assert config.llm_configured is True


def test_prices_and_budget_come_from_the_environment() -> None:
    config = load_config(
        environ={
            "GROUNDED_RAG_LLM_PRICE_INPUT_PER_M": "1.5",
            "GROUNDED_RAG_LLM_PRICE_OUTPUT_PER_M": "4",
            "GROUNDED_RAG_BUDGET_USD_PER_MONTH": "10",
        },
        env_file=None,
    )
    assert (config.llm_price_input_per_m, config.llm_price_output_per_m) == (1.5, 4.0)
    assert config.budget_usd_per_month == 10.0
    assert config.cost_usd(2_000_000, 1_000_000) == 7.0


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
    assert public["budget_usd_per_month"] == 0.5
    assert public["llm_price_input_per_m"] == 0.0


def test_missing_key_raises_without_network() -> None:
    config = Config(llm_base_url=BASE_URL, llm_model="m", llm_api_key="")
    with pytest.raises(LLMNotConfigured) as error:
        config.require_llm()
    assert "GROUNDED_RAG_LLM_API_KEY" in str(error.value)


def test_configured_config_does_not_raise() -> None:
    Config(llm_base_url=BASE_URL, llm_model="m", llm_api_key=SECRET).require_llm()


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


def test_parse_amount_values() -> None:
    assert parse_amount("0.5", name="X") == 0.5
    assert parse_amount(" 3 ", name="X") == 3.0
    for malformed in ("lots", "-1", "1,5"):
        with pytest.raises(ConfigError):
            parse_amount(malformed, name="X")


def test_invalid_values_fail_fast(tmp_path: Path) -> None:
    path = write_env(tmp_path, env_text({"GROUNDED_RAG_BUDGET_USD_PER_MONTH": "lots"}))
    with pytest.raises(ConfigError):
        load_config(environ={}, env_file=path)
    with pytest.raises(ConfigError):
        load_config(environ={"GROUNDED_RAG_GENERATION": "maybe"}, env_file=None)
    with pytest.raises(ConfigError):
        load_config(environ={"GROUNDED_RAG_RATE_PER_IP": "10"}, env_file=None)
    with pytest.raises(ConfigError):
        load_config(environ={"GROUNDED_RAG_LLM_PRICE_INPUT_PER_M": "-0.1"}, env_file=None)
    with pytest.raises(ConfigError):
        load_config(environ={"GROUNDED_RAG_PORT": "eight"}, env_file=None)


def test_llm_fake_only_accepts_mock() -> None:
    assert load_config(environ={"GROUNDED_RAG_LLM_FAKE": "mock"}, env_file=None).llm_fake == "mock"
    with pytest.raises(ConfigError):
        load_config(environ={"GROUNDED_RAG_LLM_FAKE": "real"}, env_file=None)


# ------------------------------------------------------------------ provider


def test_there_is_no_default_endpoint_or_model() -> None:
    config = load_config(environ={}, env_file=None)
    assert DEFAULT_BASE_URL == ""
    assert DEFAULT_MODEL == ""
    assert config.llm_base_url == ""
    assert config.llm_model == ""
    assert (config.llm_key_file, config.llm_key_path) == ("", "")
    with pytest.raises(LLMNotConfigured) as error:
        config.require_llm()
    message = str(error.value)
    assert "GROUNDED_RAG_LLM_BASE_URL is empty" in message
    assert "GROUNDED_RAG_LLM_MODEL is empty" in message

    overridden = load_config(
        environ={
            "GROUNDED_RAG_LLM_BASE_URL": BASE_URL,
            "GROUNDED_RAG_LLM_MODEL": MODEL,
            "GROUNDED_RAG_LLM_API_KEY": SECRET,
        },
        env_file=None,
    )
    assert overridden.llm_base_url == BASE_URL
    assert overridden.llm_model == MODEL
    assert overridden.llm_configured is True


# ------------------------------------------------------------ key by file


def write_key_json(tmp_path: Path, payload, name: str = "auth.json") -> Path:
    """A fake credential file: the real ones are never opened by the tests."""
    path = tmp_path / name
    path.write_text(payload if isinstance(payload, str) else json.dumps(payload), encoding="utf-8")
    return path


def key_file_config(path: Path, dotted: str = "llm.key", **overrides) -> Config:
    base = dict(
        llm_base_url=BASE_URL,
        llm_model=MODEL,
        llm_key_file=str(path),
        llm_key_path=dotted,
    )
    base.update(overrides)
    return Config(**base)


def test_key_file_variables_load_and_configure_without_reading(tmp_path: Path) -> None:
    path = write_key_json(tmp_path, {"llm": {"key": FILE_SECRET}})
    config = load_config(
        environ={
            "GROUNDED_RAG_LLM_BASE_URL": BASE_URL,
            "GROUNDED_RAG_LLM_MODEL": MODEL,
            "GROUNDED_RAG_LLM_KEY_FILE": str(path),
            "GROUNDED_RAG_LLM_KEY_PATH": "llm.key",
        },
        env_file=None,
    )
    assert config.llm_key_file == str(path)
    assert config.llm_key_path == "llm.key"
    assert config.llm_api_key == ""  # the key never lands in Config
    assert config.llm_configured is True
    config.require_llm()  # both variables set and the file exists


def test_dotted_path_reads_the_key_at_call_time(tmp_path: Path) -> None:
    path = write_key_json(tmp_path, {"llm": {"key": FILE_SECRET}, "other": "x"})
    config = key_file_config(path)
    assert resolve_llm_api_key(config) == FILE_SECRET
    assert FILE_SECRET not in repr(config)  # read, returned, never stored
    assert config.llm_api_key == ""


def test_env_key_wins_over_the_file(tmp_path: Path) -> None:
    path = write_key_json(tmp_path, {"llm": {"key": FILE_SECRET}})
    config = key_file_config(path, llm_api_key=SECRET)
    assert resolve_llm_api_key(config) == SECRET
    assert SECRET not in repr(config)


def test_both_file_variables_are_required(tmp_path: Path) -> None:
    path = write_key_json(tmp_path, {"llm": {"key": FILE_SECRET}})
    only_file = Config(llm_base_url=BASE_URL, llm_model="m", llm_key_file=str(path))
    only_path = Config(llm_base_url=BASE_URL, llm_model="m", llm_key_path="llm.key")
    for config, missing in ((only_file, "GROUNDED_RAG_LLM_KEY_PATH"), (only_path, "GROUNDED_RAG_LLM_KEY_FILE")):
        assert config.llm_configured is False
        with pytest.raises(LLMNotConfigured) as error:
            config.require_llm()
        assert missing in str(error.value)
        with pytest.raises(LLMNotConfigured) as error:
            resolve_llm_api_key(config)
        assert missing in str(error.value)


def test_missing_key_file_raises_without_leaking(tmp_path: Path) -> None:
    path = tmp_path / "nope.json"
    config = key_file_config(path)
    assert config.llm_configured is False
    for call in (config.require_llm, lambda: resolve_llm_api_key(config)):
        with pytest.raises(LLMNotConfigured) as error:
            call()
        text = str(error.value)
        assert "GROUNDED_RAG_LLM_KEY_FILE" in text
        assert str(path) in text
        assert FILE_SECRET not in text


def test_invalid_json_raises_without_leaking(tmp_path: Path) -> None:
    path = write_key_json(tmp_path, '{"llm": {"key": "%s",}}' % FILE_SECRET)
    config = key_file_config(path)
    with pytest.raises(LLMNotConfigured) as error:
        resolve_llm_api_key(config)
    text = str(error.value)
    assert "GROUNDED_RAG_LLM_KEY_FILE" in text
    assert str(path) in text
    assert "not valid JSON" in text
    assert FILE_SECRET not in text


def test_missing_dotted_path_raises_without_leaking(tmp_path: Path) -> None:
    path = write_key_json(tmp_path, {"llm": {"key": FILE_SECRET}})
    config = key_file_config(path, dotted="llm.token")
    with pytest.raises(LLMNotConfigured) as error:
        resolve_llm_api_key(config)
    text = str(error.value)
    assert "GROUNDED_RAG_LLM_KEY_PATH" in text
    assert "llm.token" in text
    assert str(path) in text
    assert FILE_SECRET not in text


@pytest.mark.parametrize("value", ["", "   ", 12345, None, ["a"], {"nested": FILE_SECRET}])
def test_empty_or_non_string_value_raises_without_leaking(tmp_path: Path, value) -> None:
    path = write_key_json(tmp_path, {"llm": {"key": value}})
    config = key_file_config(path)
    with pytest.raises(LLMNotConfigured) as error:
        read_key_from_file(path, "llm.key")
    text = str(error.value)
    assert "GROUNDED_RAG_LLM_KEY_PATH" in text
    assert str(path) in text
    assert "not a string" in text
    assert FILE_SECRET not in text
    assert config.llm_configured is True  # the file exists: only reading can fail


def test_repr_and_public_show_the_paths_never_the_key(tmp_path: Path) -> None:
    path = write_key_json(tmp_path, {"llm": {"key": FILE_SECRET}})
    config = key_file_config(path)
    for text in (repr(config), str(config), repr(config.public())):
        assert FILE_SECRET not in text
        assert str(path) in text
        assert "llm.key" in text
    public = config.public()
    assert public["llm_key_file"] == str(path)
    assert public["llm_key_path"] == "llm.key"
    assert public["llm_api_key"] is False
