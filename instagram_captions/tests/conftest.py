import pytest

from ..utils.validation import CaptionRequest, ContentType, ToneType
from ..utils.u_ti_ls import serialize_for_cache, deserialize_from_cache


@pytest.fixture
def caption_req_factory():
    def _make(
        prompt: str = "hello",
        content_type: ContentType = ContentType.POST,
        tone: ToneType = ToneType.CASUAL,
        **kwargs,
    ) -> CaptionRequest:
        return CaptionRequest(prompt=prompt, content_type=content_type, tone=tone, **kwargs)

    return _make


@pytest.fixture
def roundtrip_model():
    def _roundtrip(model, model_class):
        ser = serialize_for_cache(data=model)["serialized"]
        return deserialize_from_cache(data=ser, model_class=model_class)["deserialized"]

    return _roundtrip



