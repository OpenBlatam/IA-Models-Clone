import asyncio
import pytest

from ..utils.u_ti_ls import (
    normalize_timezone_string,
    serialize_for_cache,
    deserialize_from_cache,
    calculate_improvement_percentage,
    validate_numeric_range,
    format_duration_human_readable,
    generate_cache_key,
    batch_process_with_concurrency,
)
from ..utils.validation import (
    CaptionRequest,
    ContentType,
    ToneType,
    validate_caption_content,
)
from ..utils.validation import CaptionRequest as CRModel


def test_timezone_normalization_edges():
    assert normalize_timezone_string(timezone_str="gmt")["timezone"] == "UTC"
    assert normalize_timezone_string(timezone_str=" Europe/Madrid ")["timezone"] == "Europe/Madrid"
    assert normalize_timezone_string(timezone_str="")["timezone"] == "UTC"
    assert normalize_timezone_string(timezone_str="PST")["timezone"] == "US/Pacific"


def test_serialize_deserialize_pydantic_model_roundtrip(caption_req_factory, roundtrip_model):
    model = caption_req_factory(prompt="hello")
    de = roundtrip_model(model, CRModel)
    assert de.prompt == "hello"
    assert isinstance(de, CRModel)

    # Unserializable object should fall back gracefully
    class X:  # not JSON serializable
        pass
    fallback = serialize_for_cache(data=X())
    assert isinstance(fallback["serialized"], str)


def test_improvement_percentage_edges_and_numeric_range_boundaries():
    zero_zero = calculate_improvement_percentage(input=dict(original_score=0, new_score=0))
    assert zero_zero["improvement_percentage"] == 0.0
    zero_pos = calculate_improvement_percentage(input=dict(original_score=0, new_score=5))
    assert zero_pos["improvement_percentage"] == 100.0

    # Boundaries inclusive
    assert validate_numeric_range(value=1, min_val=1, max_val=10, field_name="n")["is_valid"] is True
    assert validate_numeric_range(value=10, min_val=1, max_val=10, field_name="n")["is_valid"] is True

    # Cache key stability with complex types in args (uses default=str)
    import datetime as _dt
    key = generate_cache_key(args=("a", _dt.datetime(2020, 1, 1)), kwargs={})["cache_key"]
    assert isinstance(key, str) and len(key) == 32


def test_caption_content_too_many_mentions():
    mentions = " ".join(["@u" + str(i) for i in range(21)])
    bad = validate_caption_content(caption=mentions, content_type=ContentType.POST)
    assert bad["is_valid"] is False
    assert "Too many mentions" in bad["error"]


def test_hashtags_trimmed_to_30_unique():
    tags = [f"#tag{i}" for i in range(35)]
    req = CaptionRequest(prompt="ok", content_type=ContentType.POST, tone=ToneType.CASUAL, hashtags=tags)
    assert len(req.hashtags) == 30
    assert len(req.hashtags) == len(set(req.hashtags))


def test_format_duration_hours():
    dur = format_duration_human_readable(input=dict(seconds=7200))
    assert dur["duration"].endswith("h")


def test_truncate_with_suffix_longer_than_max():
    # When suffix is longer than max_length, ensure no crash and non-empty result
    from ..utils.u_ti_ls import TruncateTextInput, truncate_text
    out = truncate_text(input=TruncateTextInput(text="abcdef", max_length=2, suffix="...."))
    assert isinstance(out["text"], str)


def test_batch_process_with_concurrency_collects_exceptions():
    async def processor(x):
        if x == 2:
            raise RuntimeError("boom")
        return x

    res = batch_process_with_concurrency(
        input=dict(items=[1, 2, 3], max_concurrency=2),
        processor=processor,
    )
    # Should include an exception among results
    results = res["results"]
    assert len(results) == 3
    assert any(isinstance(r, Exception) for r in results)


