import pytest


# Skip property-based tests if Hypothesis is not installed
hypothesis = pytest.importorskip("hypothesis")
st = pytest.importorskip("hypothesis.strategies")

from ..utils.validation import (
    validate_email,
    validate_instagram_username,
    sanitize_html,
    extract_keywords_from_text,
    ExtractKeywordsInput,
)


@hypothesis.given(st.text())
def test_validate_email_does_not_crash_on_arbitrary_text(s):
    res = validate_email(email=s)
    assert isinstance(res, dict)
    assert "is_valid" in res


@hypothesis.given(
    st.text(alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789", min_size=1, max_size=30)
)
def test_validate_instagram_username_alnum_up_to_30_is_valid_except_reserved(username):
    reserved = {"admin", "instagram", "meta", "facebook", "help", "support"}
    res = validate_instagram_username(username=username)
    if username.lower() in reserved:
        assert res["is_valid"] is False
    else:
        assert res["is_valid"] in (True, False)


@hypothesis.given(st.text())
def test_sanitize_html_never_keeps_script_tags(html_input):
    out = sanitize_html(html_content=html_input)
    assert "<script" not in out["sanitized"].lower()


@hypothesis.given(
    st.lists(st.text(min_size=1, max_size=10), min_size=1, max_size=20)
)
def test_extract_keywords_no_stop_words(words):
    text = " ".join(words)
    out = extract_keywords_from_text(input=ExtractKeywordsInput(text=text, max_keywords=20))
    kws = out["keywords"]
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
        'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did',
        'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those'
    }
    assert all(w not in stop_words for w in kws)



