import pytest

from agents.backend.onyx.server.features.instagram_captions.utils.validation import (
    CaptionRequest,
    ContentType,
)


def test_total_length_exceeds_when_including_hashtags_raises():
    prompt = "a" * 2195
    hashtags = ["#too", "#long"]
    with pytest.raises(ValueError):
        CaptionRequest(prompt=prompt, content_type=ContentType.POST, hashtags=hashtags, include_hashtags=True)


def test_total_length_within_limit_ok():
    prompt = "Hello world"
    hashtags = ["#ai", "#ml"]
    req = CaptionRequest(prompt=prompt, content_type=ContentType.POST, hashtags=hashtags, include_hashtags=True)
    assert req.prompt == "Hello world"
    assert set(req.hashtags).issuperset({"#ai", "#ml"})


def test_hashtags_sanitization_dedup_and_limit():
    tags = ["#AI", "ai", "#a!@#$", "#this_tag_is_way_too_long_!!!!!!!!!!!!!!!!!!!!!!!!!!!!", "#ml"]
    req = CaptionRequest(prompt="x", content_type=ContentType.POST, hashtags=tags)
    # invalid chars removed, lowercased, deduped; long tag dropped
    assert req.hashtags[0] == "#ai"
    assert "#ml" in req.hashtags
    assert all(len(t) <= 31 for t in req.hashtags)  # includes '#'



