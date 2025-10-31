import pytest

from ..utils.validation import (
    validate_caption_content,
    CaptionRequest,
    ContentType,
    ToneType,
)


def test_caption_content_hashtag_and_mention_boundaries_valid():
    # Exactly 30 hashtags is valid
    caption = "ok " + " ".join([f"#h{i}" for i in range(30)])
    res = validate_caption_content(caption=caption, content_type=ContentType.POST)
    assert res["is_valid"] is True
    assert res["hashtag_count"] == 30

    # Exactly 20 mentions is valid
    caption = "ok " + " ".join([f"@u{i}" for i in range(20)])
    res = validate_caption_content(caption=caption, content_type=ContentType.POST)
    assert res["is_valid"] is True
    assert res["mention_count"] == 20


@pytest.mark.parametrize(
    "ctype,limit",
    [
        (ContentType.POST, 2200),
        (ContentType.STORY, 500),
        (ContentType.REEL, 1000),
        (ContentType.CAROUSEL, 2200),
        (ContentType.IGTV, 2200),
    ],
)
def test_caption_content_length_at_limit_is_valid(ctype, limit):
    caption = "x" * limit
    res = validate_caption_content(caption=caption, content_type=ctype)
    assert res["is_valid"] is True
    assert res["length"] == limit


def test_caption_request_boundary_total_length_with_hashtags_valid():
    # Build case where prompt + hashtags exactly equals max_length
    max_len = 50
    prompt_len = 46  # 46 prompt + 1 space + 3 hashtag chars (#a1) = 50
    req = CaptionRequest(
        prompt="x" * prompt_len,
        content_type=ContentType.POST,
        tone=ToneType.CASUAL,
        hashtags=["#a1"],
        max_length=max_len,
        include_hashtags=True,
    )
    assert len(req.prompt) + (len("#a1") + 1) == max_len



