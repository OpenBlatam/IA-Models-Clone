from ..utils.u_ti_ls import sanitize_hashtags, HashtagsInput


def test_hashtags_without_hash_prefix_are_prefixed_and_lowercased():
    out = sanitize_hashtags(input=HashtagsInput(hashtags=["AI", "Data_Science"]))
    assert out["hashtags"] == ["#ai", "#data_science"]


def test_hashtags_remove_invalid_chars_and_enforce_length():
    out = sanitize_hashtags(input=HashtagsInput(hashtags=["#In*valid!", "#a" * 40, "#ok_tag"]))
    assert "#ok_tag" in out["hashtags"]
    # Long repeated '#a' becomes long but will be trimmed by length rule, resulting list may exclude it
    assert all(1 <= len(tag[1:]) <= 30 for tag in out["hashtags"])


def test_hashtags_uniqueness_preserves_first_occurrence():
    out = sanitize_hashtags(input=HashtagsInput(hashtags=["#One", "#TWO", "#one", "#two"]))
    assert out["hashtags"] == ["#one", "#two"]



