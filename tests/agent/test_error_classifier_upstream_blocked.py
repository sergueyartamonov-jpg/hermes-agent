"""A 403 written by a WAF/CDN in front of the provider is not an API-key rejection (#53099, #70566).

A relay that blocks the SDK User-Agent answers ``403 Your request was blocked.``; Cloudflare's
browser challenge answers 403 HTML. Both used to classify as ``auth`` and print key guidance.
"""
import pytest

from agent.error_classifier import FailoverReason, classify_api_error


class _APIError(Exception):
    def __init__(self, message, status_code):
        super().__init__(message)
        self.status_code = status_code


@pytest.mark.parametrize("body", [
    "Error code: 403 - Your request was blocked.",
    "<!doctype html><html><body>Enable JavaScript and cookies to continue</body></html>",
    "<!doctype html><html><script src='/cdn-cgi/challenge-platform/h/g/orchestrate/chl_page'></script></html>",
])
def test_403_waf_block_is_upstream_blocked_not_auth(body):
    result = classify_api_error(_APIError(body, 403), provider="openai-api")
    assert result.reason == FailoverReason.upstream_blocked
    assert result.retryable is False and result.should_fallback is True
    assert result.should_rotate_credential is False and result.is_auth is False


@pytest.mark.parametrize("body, status, reason", [
    ("<html><title>Forbidden</title><body>Access denied</body></html>", 403, FailoverReason.auth),
    ("<html>Enable JavaScript and cookies to continue</html>", 401, FailoverReason.auth),
])
def test_generic_403_and_all_401_keep_auth(body, status, reason):
    assert classify_api_error(_APIError(body, status), provider="openai-api").reason == reason


_OPENAI_EDGE_PAGE = """<html><body><div class="message"><p>Unable to load site</p></div>
<span class="explanation">Please try again later. If you are using a VPN, try turning it off.</span>
<span class="explanation">[IP: 192.0.2.1 | Ray ID: test-ray]</span></body></html>"""


def test_codex_unable_to_load_page_does_not_refresh_credentials():
    result = classify_api_error(_APIError(_OPENAI_EDGE_PAGE, 403), provider="openai-codex")
    assert result.reason == FailoverReason.upstream_blocked
    assert result.should_fallback is True and result.retryable is False
    assert result.is_auth is False and result.should_rotate_credential is False


@pytest.mark.parametrize("body,status", [
    (_OPENAI_EDGE_PAGE, 401),
    ("Unable to load site: token expired", 403),
    ("<html><p>Unable to load site: permission denied</p></html>", 403),
    ("Invalid authentication credentials", 403),
])
def test_codex_auth_errors_keep_auth_classification(body, status):
    result = classify_api_error(_APIError(body, status), provider="openai-codex")
    assert result.reason == FailoverReason.auth and result.is_auth is True
