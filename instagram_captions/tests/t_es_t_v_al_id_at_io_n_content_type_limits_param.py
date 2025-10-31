import pytest

from ..utils.validation import CaptionRequest, ContentType, ToneType


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
def test_caption_request_max_length_per_content_type_within_limit(ctype, limit):
    req = CaptionRequest(prompt="ok", content_type=ctype, tone=ToneType.CASUAL, max_length=limit)
    assert req.max_length == limit


@pytest.mark.parametrize(
    "ctype,limit",
    [
        (ContentType.POST, 2201),
        (ContentType.STORY, 501),
        (ContentType.REEL, 1001),
        (ContentType.CAROUSEL, 2201),
        (ContentType.IGTV, 2201),
    ],
)
def test_caption_request_max_length_per_content_type_exceeds_limit_raises(ctype, limit):
    with pytest.raises(ValueError):
        CaptionRequest(prompt="hi", content_type=ctype, tone=ToneType.CASUAL, max_length=limit)



