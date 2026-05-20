import pytest


hypothesis = pytest.importorskip("hypothesis")
st = pytest.importorskip("hypothesis.strategies")

from ..utils.validation import validate_url
from ..utils.u_ti_ls import sanitize_hashtags, HashtagsInput


@hypothesis.given(st.text())
def test_validate_url_never_crashes(s):
    out = validate_url(url=s)
    assert isinstance(out, dict)
    assert "is_valid" in out


@hypothesis.given(st.lists(st.text(), min_size=0, max_size=50))
def test_sanitize_hashtags_properties_random_inputs(items):
    out = sanitize_hashtags(input=HashtagsInput(hashtags=items))["hashtags"]
    # All hashtags must start with '#', be lowercase, length within [1, 30] for content,
    # and content must be alnum allowing underscores
    for tag in out:
        assert tag.startswith("#")
        assert tag == tag.lower()
        content = tag[1:]
        assert 1 <= len(content) <= 30
        assert content.replace("_", "").isalnum()
    # No duplicates
    assert len(out) == len(set(out))



