import pytest

from agents.backend.onyx.server.features.instagram_captions.utils.validation import (
    CaptionRequest,
    ContentType,
)


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
def test_max_length_not_exceeding_per_content_type(ctype, limit):
    ok = CaptionRequest(prompt="x", content_type=ctype, max_length=limit)
    assert ok.max_length == limit
    with pytest.raises(ValueError):
        CaptionRequest(prompt="x", content_type=ctype, max_length=limit + 1)



