import pytest

from ..utils.validation import (
    validate_email,
    validate_url,
    validate_instagram_username,
    sanitize_html,
    validate_caption_content,
    CaptionRequest,
    BatchCaptionRequest,
    CaptionResponse,
    ContentType,
    ToneType,
)


class TestValidationUtilities:
    def test_validate_email(self):
        assert validate_email(email="USER@example.com") == {
            "is_valid": True,
            "email": "user@example.com",
        }

        assert validate_email(email="bad-email") == {
            "is_valid": False,
            "error": "Invalid email format",
        }

        assert validate_email(email=" ") == {
            "is_valid": False,
            "error": "Email cannot be empty",
        }

    def test_validate_url(self):
        assert validate_url(url="https://example.com/path?q=1#x") == {
            "is_valid": True,
            "url": "https://example.com/path?q=1#x",
        }

        assert validate_url(url="ftp://example.com") == {
            "is_valid": False,
            "error": "Invalid URL format",
        }

        assert validate_url(url=" ") == {
            "is_valid": False,
            "error": "URL cannot be empty",
        }

    def test_validate_instagram_username(self):
        assert validate_instagram_username(username="JohnDoe123") == {
            "is_valid": True,
            "username": "JohnDoe123",
        }

        assert validate_instagram_username(username="john.doe") == {
            "is_valid": False,
            "error": "Username contains invalid characters",
        }

        assert validate_instagram_username(username="instagram") == {
            "is_valid": False,
            "error": "Username is reserved",
        }

    def test_sanitize_html(self):
        res = sanitize_html(html_content="<b>ok</b><script>bad()</script>")
        assert res["sanitized"].startswith("<b>ok</b>")
        assert "script" in res["removed_tags"]

        res_whitelist = sanitize_html(
            html_content="<i>ok</i><u>keep</u>",
            allowed_tags=["i", "u"],
        )
        assert res_whitelist["sanitized"] == "<i>ok</i><u>keep</u>"

    def test_validate_caption_content(self):
        # Valid case
        ok = validate_caption_content(
            caption="Great day at the beach! #sun #fun @friend",
            content_type=ContentType.POST,
        )
        assert ok["is_valid"] is True
        assert ok["hashtag_count"] == 2
        assert ok["mention_count"] == 1

        # Too many hashtags
        many_tags = "Look " + " ".join([f"#t{i}" for i in range(31)])
        bad = validate_caption_content(caption=many_tags, content_type=ContentType.POST)
        assert bad["is_valid"] is False
        assert bad["error"].startswith("Too many hashtags")


class TestCaptionRequestModel:
    def test_prompt_sanitization_and_hashtags(self):
        req = CaptionRequest(
            prompt="  hello   <b>world</b>  ",
            content_type=ContentType.POST,
            tone=ToneType.PROFESSIONAL,
            hashtags=["#AI", "#AI", "#Clean-Tag", "#bad!tag"],
            max_length=200,
            include_hashtags=True,
        )

        # Prompt should be stripped, whitespace collapsed, and HTML escaped
        assert req.prompt == "hello &lt;b&gt;world&lt;/b&gt;"

        # Hashtags: duplicates removed, sanitized, lowercased, max 30
        assert req.hashtags[0] == "#ai"
        assert "#clean-tag" in req.hashtags
        assert all(tag.startswith("#") for tag in req.hashtags)
        assert len(req.hashtags) <= 30

    def test_max_length_respected_per_content_type(self):
        # STORY limit is 500; setting higher should fail
        with pytest.raises(ValueError):
            CaptionRequest(
                prompt="ok",
                content_type=ContentType.STORY,
                tone=ToneType.CASUAL,
                max_length=10000,
            )

        # Valid when within limit
        req = CaptionRequest(
            prompt="ok",
            content_type=ContentType.STORY,
            tone=ToneType.CASUAL,
            max_length=400,
        )
        assert req.max_length == 400

    def test_total_length_validation_with_hashtags(self):
        # Construct case where prompt + hashtags would exceed limit 50
        with pytest.raises(ValueError):
            CaptionRequest(
                prompt="x" * 45,
                content_type=ContentType.POST,
                tone=ToneType.CASUAL,
                hashtags=["#tag" * 2],
                max_length=50,
                include_hashtags=True,
            )

    def test_include_hashtags_false_skips_total_length_check(self):
        # Long hashtags but excluded from length check
        req = CaptionRequest(
            prompt="x" * 45,
            content_type=ContentType.POST,
            tone=ToneType.CASUAL,
            hashtags=["#" + "t" * 40],
            max_length=46,
            include_hashtags=False,
        )
        assert req.max_length == 46


class TestBatchCaptionRequestModel:
    def test_batch_limits(self):
        # Valid small batch
        r1 = CaptionRequest(prompt="a", content_type=ContentType.POST, tone=ToneType.CASUAL)
        r2 = CaptionRequest(prompt="b", content_type=ContentType.POST, tone=ToneType.CASUAL)
        batch_ok = BatchCaptionRequest(requests=[r1, r2], batch_size=2)
        assert len(batch_ok.requests) == 2

        # Exceed batch limit
        with pytest.raises(ValueError):
            BatchCaptionRequest(requests=[r1] * 101, batch_size=10)


