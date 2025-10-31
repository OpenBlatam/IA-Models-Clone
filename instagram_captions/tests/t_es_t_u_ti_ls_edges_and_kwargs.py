import pytest

from ..utils.u_ti_ls import (
    generate_cache_key,
    deserialize_from_cache,
    format_duration_human_readable,
)
from ..utils.validation import (
    CaptionRequest,
    CaptionResponse,
    ContentType,
    ToneType,
    validate_caption_request,
)


def test_generate_cache_key_unserializable_kwargs_returns_error():
    out = generate_cache_key(args=(), kwargs={"bad": set([1, 2])})
    assert out["cache_key"] is None
    assert "error" in out


def test_deserialize_from_cache_model_mismatch_returns_error():
    # JSON missing required fields for CaptionRequest should raise internally and return error dict
    bad_json = "{}"
    out = deserialize_from_cache(data=bad_json, model_class=CaptionRequest)
    assert out["deserialized"] is None
    assert "error" in out


def test_format_duration_boundaries():
    assert format_duration_human_readable(input=dict(seconds=1))["duration"].endswith("s")
    assert format_duration_human_readable(input=dict(seconds=60))["duration"].endswith("m")
    assert format_duration_human_readable(input=dict(seconds=3600))["duration"].endswith("h")


@validate_caption_request
def _echo_kw_only(*, request: CaptionRequest):
    return request


def test_validate_caption_request_with_kwargs_request():
    data = {
        "prompt": "hi",
        "content_type": ContentType.POST,
        "tone": ToneType.CASUAL,
    }
    result = _echo_kw_only(request=data)
    assert isinstance(result, CaptionRequest)
    assert result.prompt == "hi"


def test_caption_response_quality_score_optional():
    resp = CaptionResponse(
        caption="ok",
        content_type=ContentType.POST,
        length=2,
        hashtags=[],
        tone=ToneType.CASUAL,
        generation_time=0.01,
        quality_score=None,
    )
    assert resp.quality_score is None



