import pytest

from ..utils.validation import (
    validate_caption_request,
    CaptionRequest,
    ContentType,
    ToneType,
    sanitize_html,
)
from ..utils.u_ti_ls import validate_caption_length, CaptionLengthInput


@validate_caption_request
def _echo_any(request):
    return request


def test_validate_caption_request_with_invalid_type_returns_error_dict():
    res = _echo_any(["not", "a", "dict"])  # invalid type
    assert isinstance(res, dict)
    assert res.get("is_valid") is False


def test_sanitize_html_with_empty_allowed_list_removes_all_tags():
    res = sanitize_html(html_content="<b>x</b><i>y</i>", allowed_tags=[])
    assert res["sanitized"] == "xy"


def test_validate_caption_length_exceeds_for_story_is_false():
    out = validate_caption_length(input=CaptionLengthInput(caption="x" * 600, content_type="story"))
    assert out["is_valid"] is False and out["max_length"] == 500



