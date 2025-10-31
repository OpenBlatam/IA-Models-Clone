import pytest

from ..utils.validation import CaptionResponse, ContentType, ToneType


def test_caption_response_quality_score_bounds():
    ok = CaptionResponse(
        caption="ok",
        content_type=ContentType.POST,
        length=2,
        hashtags=[],
        tone=ToneType.CASUAL,
        generation_time=0.01,
        quality_score=0.0,
    )
    assert ok.quality_score == 0.0

    ok2 = CaptionResponse(
        caption="ok",
        content_type=ContentType.POST,
        length=2,
        hashtags=[],
        tone=ToneType.CASUAL,
        generation_time=0.01,
        quality_score=100.0,
    )
    assert ok2.quality_score == 100.0

    with pytest.raises(ValueError):
        CaptionResponse(
            caption="ok",
            content_type=ContentType.POST,
            length=2,
            hashtags=[],
            tone=ToneType.CASUAL,
            generation_time=0.01,
            quality_score=-1,
        )

    with pytest.raises(ValueError):
        CaptionResponse(
            caption="ok",
            content_type=ContentType.POST,
            length=2,
            hashtags=[],
            tone=ToneType.CASUAL,
            generation_time=0.01,
            quality_score=101,
        )



