from ..utils.validation import (
    validate_caption_request,
    CaptionRequest,
    ContentType,
    ToneType,
)


@validate_caption_request
def _echo_request(request: CaptionRequest):
    return request


def test_validate_caption_request_success():
    data = {
        "prompt": "  hello   world  ",
        "content_type": ContentType.POST,
        "tone": ToneType.PROFESSIONAL,
        "hashtags": ["#X"],
        "max_length": 200,
        "include_hashtags": True,
    }
    result = _echo_request(data)
    assert isinstance(result, CaptionRequest)
    assert result.prompt == "hello world"


def test_validate_caption_request_failure_returns_error_dict():
    bad = {"prompt": " ", "content_type": ContentType.POST, "tone": ToneType.CASUAL}
    result = _echo_request(bad)
    assert isinstance(result, dict)
    assert result.get("is_valid") is False
    assert "error" in result


def test_validate_caption_request_passthrough_instance():
    # Passing a model instance should be returned as-is
    inst = CaptionRequest(prompt="ok", content_type=ContentType.POST, tone=ToneType.CASUAL)
    out = _echo_request(inst)
    assert out is inst


