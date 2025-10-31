import pytest

from ..utils.validation import BatchCaptionRequest, CaptionRequest, ContentType, ToneType, UserInputValidation


def test_batch_caption_request_batch_size_bounds():
    r = CaptionRequest(prompt="a", content_type=ContentType.POST, tone=ToneType.CASUAL)
    ok = BatchCaptionRequest(requests=[r] * 5, batch_size=1)
    assert ok.batch_size == 1

    with pytest.raises(ValueError):
        BatchCaptionRequest(requests=[r] * 5, batch_size=0)

    with pytest.raises(ValueError):
        BatchCaptionRequest(requests=[r] * 5, batch_size=51)


def test_user_input_validation_accepts_known_types():
    ui = UserInputValidation(input_text=" hi ", input_type="email", max_length=10)
    assert ui.input_text == "hi"
    assert ui.input_type == "email"



