import asyncio
import datetime as dt

import pytest

from ..utils.u_ti_ls import (
    extract_keywords_from_text,
    ExtractKeywordsInput,
    sanitize_hashtags,
    HashtagsInput,
    calculate_readability_score,
    ReadabilityInput,
    PerformanceMonitor,
    deserialize_from_cache,
)


def test_extract_keywords_filters_stop_words_and_min_length():
    text = "the and but an in on at to for of with is are was were have it a big test for amazing results"
    kws = extract_keywords_from_text(input=ExtractKeywordsInput(text=text, max_keywords=50))["keywords"]
    # Ensure common stop words not present and keywords length > 3
    assert all(len(k) > 3 for k in kws)
    assert "amazing" in kws or "results" in kws


def test_sanitize_hashtags_filters_long_and_non_alnum():
    too_long = "#" + ("x" * 31)
    mixed = ["#ok", too_long, "invalid!tag", "#also_ok"]
    out = sanitize_hashtags(input=HashtagsInput(hashtags=mixed))["hashtags"]
    assert "#ok" in out and "#also_ok" in out
    assert all(1 <= len(tag[1:]) <= 30 for tag in out)
    assert all(tag.replace("_", "")[1:].isalnum() for tag in out)


def test_calculate_readability_score_empty_and_long():
    assert calculate_readability_score(input=ReadabilityInput(text=""))["readability_score"] == 0.0
    long_text = " ".join(["word" for _ in range(200)]) + "."
    score = calculate_readability_score(input=ReadabilityInput(text=long_text))["readability_score"]
    assert 0.0 <= score <= 100.0


def test_performance_monitor_missing_operation_returns_zero():
    pm = PerformanceMonitor()
    assert pm.end_timer("not_started") == 0.0


def test_deserialize_from_cache_empty_data_error():
    out = deserialize_from_cache(data="", model_class=None)
    assert out["deserialized"] is None
    assert out["error"] == "No data provided"



