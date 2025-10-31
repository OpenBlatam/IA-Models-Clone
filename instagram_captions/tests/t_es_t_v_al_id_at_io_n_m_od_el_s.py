import pytest

from ..utils.validation import (
    UserInputValidation,
    CaptionResponse,
    ContentType,
    ToneType,
    CaptionRequest,
)


class TestValidationModels:
    def test_user_input_validation_success_and_errors(self):
        ok = UserInputValidation(input_text="  hello  ", input_type="text", max_length=10)
        assert ok.input_text == "hello"
        assert ok.input_type == "text"

        with pytest.raises(ValueError):
            UserInputValidation(input_text="", input_type="text", max_length=10)

        with pytest.raises(ValueError):
            UserInputValidation(input_text="hi", input_type="invalid_type", max_length=10)

        with pytest.raises(ValueError):
            UserInputValidation(input_text="x" * 1001, input_type="text", max_length=10)

    def test_caption_response_length_validator(self):
        # Valid response
        ok = CaptionResponse(
            caption="nice caption",
            content_type=ContentType.POST,
            length=12,
            hashtags=["#a"],
            tone=ToneType.CASUAL,
            generation_time=0.1,
            quality_score=90.0,
        )
        assert ok.length == len(ok.caption)

        # Too long
        with pytest.raises(ValueError):
            CaptionResponse(
                caption="x" * 2500,
                content_type=ContentType.POST,
                length=2500,
                hashtags=[],
                tone=ToneType.CASUAL,
                generation_time=0.2,
            )

    def test_caption_request_content_type_limits(self):
        # REEL limit is 1000; setting higher max_length should fail
        with pytest.raises(ValueError):
            CaptionRequest(
                prompt="ok",
                content_type=ContentType.REEL,
                tone=ToneType.CASUAL,
                max_length=2000,
            )

        # Lower is ok
        req = CaptionRequest(
            prompt="hello",
            content_type=ContentType.REEL,
            tone=ToneType.CASUAL,
            max_length=800,
        )
        assert req.max_length == 800



