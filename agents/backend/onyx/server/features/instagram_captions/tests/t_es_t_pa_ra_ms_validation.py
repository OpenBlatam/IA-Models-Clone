import pytest

from ..utils.validation import (
    validate_email,
    validate_url,
    validate_instagram_username,
)
from ..utils.u_ti_ls import normalize_timezone_string


@pytest.mark.parametrize(
    "email,ok,normalized",
    [
        ("USER@Example.COM", True, "user@example.com"),
        ("a@b.co", True, "a@b.co"),
        ("bad-email", False, None),
        ("", False, None),
    ],
)
def test_emails_param(email, ok, normalized):
    res = validate_email(email=email)
    assert res["is_valid"] is ok
    if ok:
        assert res["email"] == normalized


@pytest.mark.parametrize(
    "url,ok",
    [
        ("https://a.co", True),
        ("http://a.com:8080/p?q=1#f", True),
        ("ftp://x.com", False),
        ("http://", False),
        ("", False),
    ],
)
def test_urls_param(url, ok):
    res = validate_url(url=url)
    assert res["is_valid"] is ok


@pytest.mark.parametrize(
    "username,ok",
    [
        ("User1", True),
        ("user.name", False),
        ("admin", False),
        ("a" * 31, False),
        ("a", True),
        ("User_1", False),
    ],
)
def test_usernames_param(username, ok):
    res = validate_instagram_username(username=username)
    assert res["is_valid"] is ok


@pytest.mark.parametrize(
    "tz_in,tz_out",
    [
        ("est", "US/Eastern"),
        ("pst", "US/Pacific"),
        ("cst", "US/Central"),
        ("mst", "US/Mountain"),
        ("utc", "UTC"),
        ("gmt", "UTC"),
    ],
)
def test_timezone_mapping_param(tz_in, tz_out):
    assert normalize_timezone_string(timezone_str=tz_in)["timezone"] == tz_out



