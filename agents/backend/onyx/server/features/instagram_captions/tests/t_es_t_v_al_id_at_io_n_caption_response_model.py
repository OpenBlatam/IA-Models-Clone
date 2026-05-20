import pytest

from agents.backend.onyx.server.features.instagram_captions.utils.validation import (
    CaptionResponse,
    ContentType,
    ToneType,
)


def test_caption_response_validates_fields():
    resp = CaptionResponse(
        caption="hello",
        content_type=ContentType.POST,
        length=5,
        hashtags=["#hi"],
        tone=ToneType.PROFESSIONAL,
        generation_time=0.01,
        quality_score=95.0,
    )
    assert resp.length == 5
    with pytest.raises(ValueError):
        CaptionResponse(
            caption="x" * 2300,
            content_type=ContentType.POST,
            length=2300,
            hashtags=[],
            tone=ToneType.CASUAL,
            generation_time=0.01,
        )



