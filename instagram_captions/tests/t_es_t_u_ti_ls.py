import asyncio
import time
from types import SimpleNamespace

import pytest

from ..utils.u_ti_ls import (
    create_error_response,
    validate_non_empty_string,
    validate_list_not_empty,
    validate_numeric_range,
    generate_cache_key,
    serialize_for_cache,
    deserialize_from_cache,
    normalize_timezone_string,
    extract_keywords_from_text,
    calculate_improvement_percentage,
    validate_caption_length,
    sanitize_hashtags,
    calculate_readability_score,
    get_current_utc_timestamp,
    batch_process_with_concurrency,
    format_duration_human_readable,
    truncate_text,
    PerformanceMonitor,
    # Input models
    ExtractKeywordsInput,
    ImprovementInput,
    CaptionLengthInput,
    HashtagsInput,
    ReadabilityInput,
    BatchProcessInput,
    FormatDurationInput,
    TruncateTextInput,
)


def test_create_error_response_and_validations():
    resp = create_error_response(error_code="E_CODE", message="Oops")
    assert resp["error_code"] == "E_CODE"
    assert resp["message"] == "Oops"

    missing = create_error_response(error_code="", message="")
    assert missing["error_code"] == "INVALID_INPUT"

    assert validate_non_empty_string(value=" hi ", field_name="name")["is_valid"] is True
    assert validate_non_empty_string(value=" ", field_name="name")["is_valid"] is False

    assert validate_list_not_empty(value=[1], field_name="items")["is_valid"] is True
    assert validate_list_not_empty(value=[], field_name="items")["is_valid"] is False

    assert validate_numeric_range(value=5, min_val=1, max_val=10, field_name="n")["is_valid"] is True
    assert validate_numeric_range(value=0, min_val=1, max_val=10, field_name="n")["is_valid"] is False


def test_cache_helpers_and_timezone():
    key1 = generate_cache_key(args=(1, 2), kwargs={"b": 2, "a": 1})["cache_key"]
    key2 = generate_cache_key(args=(1, 2), kwargs={"a": 1, "b": 2})["cache_key"]
    assert key1 == key2
    assert len(key1) == 32

    data = {"x": 1}
    ser = serialize_for_cache(data=data)["serialized"]
    de = deserialize_from_cache(data=ser)["deserialized"]
    assert de == data

    bad = deserialize_from_cache(data="{bad json}")
    assert bad["deserialized"] is None
    assert "error" in bad

    assert normalize_timezone_string(timezone_str="est")["timezone"] == "US/Eastern"
    assert normalize_timezone_string(timezone_str=None)["timezone"] == "UTC"


def test_keyword_readability_hashtags_and_length():
    kw = extract_keywords_from_text(
        input=ExtractKeywordsInput(text="This is an amazing productivity guide for creators", max_keywords=5)
    )["keywords"]
    assert "amazing" in kw and "productivity" in kw
    assert len(kw) <= 5

    imp = calculate_improvement_percentage(input=ImprovementInput(original_score=50, new_score=75))
    assert imp["improvement_percentage"] == 50.0

    valid = validate_caption_length(input=CaptionLengthInput(caption="x" * 100, content_type="story"))
    assert valid["is_valid"] is True and valid["max_length"] == 500

    tags = sanitize_hashtags(input=HashtagsInput(hashtags=["AI", "#A!I", "#valid_tag", "#valid_tag"]))
    assert "#ai" in tags["hashtags"] and "#valid_tag" in tags["hashtags"]
    assert len(tags["hashtags"]) == len(set(tags["hashtags"]))

    score = calculate_readability_score(input=ReadabilityInput(text="Short sentence. Another one!"))
    assert 0.0 <= score["readability_score"] <= 100.0


def test_time_and_truncation_helpers():
    ts = get_current_utc_timestamp()["timestamp"]
    assert ts.endswith("Z") or "T" in ts

    dur_ms = format_duration_human_readable(input=FormatDurationInput(seconds=0.5))["duration"]
    dur_s = format_duration_human_readable(input=FormatDurationInput(seconds=10))["duration"]
    dur_m = format_duration_human_readable(input=FormatDurationInput(seconds=120))["duration"]
    assert dur_ms.endswith("ms") and dur_s.endswith("s") and dur_m.endswith("m")

    txt = truncate_text(input=TruncateTextInput(text="abcdef", max_length=4, suffix=".."))["text"]
    assert txt == "ab.."
    assert truncate_text(input=TruncateTextInput(text="abc", max_length=5))["text"] == "abc"


@pytest.mark.asyncio
async def test_timeout_and_batch_processing_and_measurement():
    async def succeed():
        await asyncio.sleep(0.01)
        return 7

    async def slow():
        await asyncio.sleep(0.05)
        return 1

    ok = await asyncio.wait_for(succeed(), timeout=0.2)
    assert ok == 7

    timed = await asyncio.wait_for(
        asyncio.create_task(
            # wrap via timeout_operation helper
            __import__("agents.backend.onyx.server.features.instagram_captions.utils.u_ti_ls", fromlist=['x']).timeout_operation(
                operation=succeed,
                timeout_seconds=0.2,
            )
        ),
        timeout=1.0,
    )
    assert timed["timed_out"] is False and timed["result"] == 7

    timedout = await __import__(
        "agents.backend.onyx.server.features.instagram_captions.utils.u_ti_ls",
        fromlist=['x'],
    ).timeout_operation(operation=slow, timeout_seconds=0.01)
    assert timedout["timed_out"] is True

    # batch_process_with_concurrency uses asyncio.run internally; use lightweight async processor
    async def processor(x):
        await asyncio.sleep(0)
        return x * 2

    batch = batch_process_with_concurrency(
        input=BatchProcessInput(items=[1, 2, 3, 4], max_concurrency=2),
        processor=processor,
    )
    assert batch["results"] == [2, 4, 6, 8]

    # PerformanceMonitor basic flow
    pm = PerformanceMonitor()
    pm.start_timer("op")
    await asyncio.sleep(0.001)
    dur = pm.end_timer("op")
    assert dur >= 0
    mets = pm.get_metrics()
    assert "op" in mets
    pm.reset()
    assert pm.get_metrics() == {}



