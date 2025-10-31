import pytest

from ..utils.validation import (
    validate_email,
    validate_url,
    validate_instagram_username,
    sanitize_html,
    validate_caption_content,
    CaptionRequest,
    ContentType,
    ToneType,
)


def test_validate_email_lowercases_domain_and_user():
    out = validate_email(email="A@EXAMPLE.COM")
    assert out["is_valid"] is True
    assert out["email"] == "a@example.com"


def test_validate_url_with_port_and_path():
    out = validate_url(url="http://example.com:8080/p?q=1#f")
    assert out["is_valid"] is True


def test_validate_url_missing_scheme():
    out = validate_url(url="example.com/path")
    assert out["is_valid"] is False


def test_validate_username_length_boundaries():
    # too short (empty)
    assert validate_instagram_username(username=" ")["is_valid"] is False
    # 30 chars ok
    assert validate_instagram_username(username="a" * 30)["is_valid"] is True
    # 31 chars not ok
    assert validate_instagram_username(username="a" * 31)["is_valid"] is False


def test_sanitize_html_comments_and_whitelist():
    res = sanitize_html(html_content="<!--c--><b>x</b><u>y</u>")
    # default allowed does not include <u>, so it should be removed
    assert "<b>" in res["sanitized"] and "u>" not in res["sanitized"]
    assert "<!--" not in res["sanitized"]


def test_validate_caption_content_reel_length_limit():
    too_long = "x" * 1001
    bad = validate_caption_content(caption=too_long, content_type=ContentType.REEL)
    assert bad["is_valid"] is False


def test_normalize_timezone_string_unknown_kept():
    from ..utils.u_ti_ls import normalize_timezone_string
    tz = normalize_timezone_string(timezone_str=" Asia/Kolkata ")
    assert tz["timezone"] == "Asia/Kolkata"


def test_caption_request_hashtags_order_preserved_unique():
    tags = ["#one", "#two", "#one", "#three", "#two"]
    req = CaptionRequest(prompt="ok", content_type=ContentType.POST, tone=ToneType.CASUAL, hashtags=tags)
    assert req.hashtags == ["#one", "#two", "#three"]



