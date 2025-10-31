import pytest

from ..utils.validation import (
    validate_email,
    validate_url,
    validate_instagram_username,
    sanitize_html,
    validate_caption_content,
    BatchCaptionRequest,
    CaptionRequest,
    ContentType,
    ToneType,
)


class TestValidationEdges:
    def test_email_and_url_non_string_inputs(self):
        assert validate_email(email=123)["is_valid"] is False
        assert validate_email(email=None)["is_valid"] is False

        assert validate_url(url=123)["is_valid"] is False
        assert validate_url(url=None)["is_valid"] is False

    def test_username_non_string_input(self):
        assert validate_instagram_username(username=123)["is_valid"] is False

    def test_caption_content_invalid_types(self):
        bad_caption = validate_caption_content(caption=123, content_type=ContentType.POST)
        assert bad_caption["is_valid"] is False
        assert bad_caption["error"] == "Caption must be a string"

        bad_type = validate_caption_content(caption="ok", content_type="post")
        assert bad_type["is_valid"] is False
        assert bad_type["error"] == "Invalid content type"

    def test_batch_request_empty_raises(self):
        with pytest.raises(ValueError):
            BatchCaptionRequest(requests=[], batch_size=5)

    def test_sanitize_html_non_string(self):
        res = sanitize_html(html_content=None)
        assert res["sanitized"] == ""
        assert "invalid_type" in res["removed_tags"]



