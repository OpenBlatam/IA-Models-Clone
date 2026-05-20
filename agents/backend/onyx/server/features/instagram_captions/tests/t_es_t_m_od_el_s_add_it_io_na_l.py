import datetime as dt
import pytest

from ..utils.validation import (
    CaptionRequest,
    CaptionResponse,
    BatchCaptionRequest,
    ContentType,
    ToneType,
)
from ..utils.u_ti_ls import get_current_utc_timestamp


class TestModelsAdditional:
    def test_caption_response_trims_and_limits(self):
        resp = CaptionResponse(
            caption="  hello  ",
            content_type=ContentType.POST,
            length=7,
            hashtags=[],
            tone=ToneType.CASUAL,
            generation_time=0.01,
        )
        # validator trims caption
        assert resp.caption == "hello"
        assert resp.length == len(resp.caption)

        # Exceeding 2200 should fail
        with pytest.raises(ValueError):
            CaptionResponse(
                caption="x" * 2300,
                content_type=ContentType.POST,
                length=2300,
                hashtags=[],
                tone=ToneType.CASUAL,
                generation_time=0.01,
            )

    def test_caption_request_empty_hashtags_removed(self):
        req = CaptionRequest(
            prompt="ok",
            content_type=ContentType.POST,
            tone=ToneType.CASUAL,
            hashtags=[" ", "#", "#bad!", "#good_1"],
        )
        assert req.hashtags == ["#good_1"]

    def test_batch_caption_request_size_constraints(self):
        r = CaptionRequest(prompt="a", content_type=ContentType.POST, tone=ToneType.CASUAL)
        ok = BatchCaptionRequest(requests=[r] * 10, batch_size=10)
        assert len(ok.requests) == 10

        with pytest.raises(ValueError):
            BatchCaptionRequest(requests=[r] * 101, batch_size=10)

    def test_current_utc_timestamp_is_isoformat(self):
        ts = get_current_utc_timestamp()["timestamp"]
        # fromisoformat supports timezone offset; ensure it parses
        try:
            dt.datetime.fromisoformat(ts.replace("Z", "+00:00"))
            ok = True
        except Exception:
            ok = False
        assert ok is True



