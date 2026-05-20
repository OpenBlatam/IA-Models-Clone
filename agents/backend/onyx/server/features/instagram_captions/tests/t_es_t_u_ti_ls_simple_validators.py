from ..utils.u_ti_ls import (
    validate_non_empty_string,
    validate_list_not_empty,
    validate_numeric_range,
    validate_caption_length,
    CaptionLengthInput,
)
from ..utils.validation import extract_keywords_from_text, ExtractKeywordsInput, sanitize_html


def test_simple_validators_error_messages():
    e1 = validate_non_empty_string(value=" ", field_name="name")
    assert e1["is_valid"] is False and e1["error"] == "name cannot be empty"

    e2 = validate_list_not_empty(value=[], field_name="items")
    assert e2["is_valid"] is False and e2["error"] == "items cannot be empty"

    e3 = validate_numeric_range(value=0, min_val=1, max_val=2, field_name="n")
    assert e3["is_valid"] is False and "between 1 and 2" in e3["error"]


def test_extract_keywords_uniqueness():
    text = "growth growth scaling scaling scaling product product-led"
    out = extract_keywords_from_text(input=ExtractKeywordsInput(text=text, max_keywords=10))
    kws = out["keywords"]
    assert len(kws) == len(set(kws))


def test_validate_caption_length_unknown_type_defaults_2200():
    out = validate_caption_length(input=CaptionLengthInput(caption="x" * 2200, content_type="unknown"))
    assert out["is_valid"] is True and out["max_length"] == 2200


def test_sanitize_html_custom_allowed_tags_no_duplicates():
    res = sanitize_html(html_content="<b>x</b><i>y</i><i>z</i>", allowed_tags=["b", "i"])
    assert res["sanitized"].count("<i>") == 2
    assert len(res["removed_tags"]) == len(set(res["removed_tags"]))



