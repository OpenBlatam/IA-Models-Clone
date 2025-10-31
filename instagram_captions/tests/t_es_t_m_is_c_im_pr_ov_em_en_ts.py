import pytest

from ..utils.u_ti_ls import format_duration_human_readable, calculate_improvement_percentage
from ..utils.validation import sanitize_html
from ..utils.validation import CaptionRequest, ContentType, ToneType
from ..utils.validation import validate_caption_request


def test_format_duration_threshold_seconds():
    # 59.9s should still be seconds, not minutes
    out = format_duration_human_readable(input=dict(seconds=59.9))
    assert out["duration"].endswith("s")


def test_calculate_improvement_negative_original_score():
    pos = calculate_improvement_percentage(input=dict(original_score=-10, new_score=5))
    assert pos["improvement_percentage"] == 100.0
    zero = calculate_improvement_percentage(input=dict(original_score=-10, new_score=0))
    assert zero["improvement_percentage"] == 0.0


def test_sanitize_html_removed_tags_are_unique():
    res = sanitize_html(html_content="<x>x</x><x>y</x>")
    # removed_tags should have no duplicates
    assert len(res["removed_tags"]) == len(set(res["removed_tags"]))


@validate_caption_request
def _no_request_func():
    return "ok"


def test_validate_caption_request_without_request_data_passthrough():
    # When no request-like arg provided, function executes normally
    assert _no_request_func() == "ok"



