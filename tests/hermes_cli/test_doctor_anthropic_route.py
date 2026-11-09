"""Doctor must not validate a custom endpoint credential against Anthropic."""
from unittest.mock import Mock, patch
from hermes_cli.doctor_connectivity import _probe_anthropic


def test_custom_anthropic_endpoint_does_not_receive_official_api_probe():
    with patch("hermes_cli.auth.get_anthropic_key", return_value="proxy-test"), \
         patch("hermes_cli.config.get_env_value_prefer_dotenv", return_value="http://127.0.0.1:3457"), \
         patch("httpx.get") as get:
        result = _probe_anthropic()
    get.assert_not_called()
    assert "custom" in str(result.lines).lower()
    assert "skipped" in str(result.lines).lower()


def test_official_anthropic_endpoint_still_reports_invalid_credential():
    with patch("hermes_cli.auth.get_anthropic_key", return_value="invalid-test"), \
         patch("hermes_cli.config.get_env_value_prefer_dotenv", return_value="https://api.anthropic.com"), \
         patch("httpx.get", return_value=Mock(status_code=401)) as get:
        result = _probe_anthropic()
    assert get.call_args.args[0] == "https://api.anthropic.com/v1/models"
    assert "invalid API key" in str(result.lines)
