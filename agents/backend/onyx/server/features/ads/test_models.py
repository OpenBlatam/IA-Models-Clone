from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
import pytest
from uuid6 import UUID, uuid7
from agents.backend.onyx.server.features.ads.models import Ad
from agents.backend.onyx.server.features.ads.schemas import AdCreate, AdRead
import orjson

from typing import Any, List, Dict, Optional
import logging
import asyncio
def test_ad_valid():
    
    """test_ad_valid function."""
ad = Ad(title="AdTitle", content="Some content", metadata={"type": "banner"})
    assert isinstance(ad.id, UUID)
    assert ad.title == "AdTitle"
    assert ad.content == "Some content"
    assert ad.metadata == {"type": "banner"}

def test_ad_empty_title_raises():
    
    """test_ad_empty_title_raises function."""
with pytest.raises(ValueError):
        Ad(title=" ", content="Content", metadata={})

def test_ad_empty_content_raises():
    
    """test_ad_empty_content_raises function."""
with pytest.raises(ValueError):
        Ad(title="Title", content=" ", metadata={})

def test_ad_metadata_not_dict():
    
    """test_ad_metadata_not_dict function."""
with pytest.raises(ValueError):
        Ad(title="Title", content="Content", metadata=[1,2,3])

def test_ad_serialization():
    
    """test_ad_serialization function."""
ad = Ad(title="AdTitle", content="Content", metadata={})
    data = ad.model_dump_json()
    assert '"title":"AdTitle"' in data

def test_ad_create_valid():
    
    """test_ad_create_valid function."""
schema = AdCreate(title="AdTitle", content="Content", metadata={})
    assert schema.title == "AdTitle"
    assert schema.content == "Content"

def test_ad_create_invalid_title():
    
    """test_ad_create_invalid_title function."""
with pytest.raises(ValueError):
        AdCreate(title=" ", content="Content", metadata={})

def test_ad_create_invalid_content():
    
    """test_ad_create_invalid_content function."""
with pytest.raises(ValueError):
        AdCreate(title="Title", content=" ", metadata={})

def test_ad_create_metadata_not_dict():
    
    """test_ad_create_metadata_not_dict function."""
with pytest.raises(ValueError):
        AdCreate(title="Title", content="Content", metadata=[1,2,3])

def test_ad_read_valid():
    
    """test_ad_read_valid function."""
schema = AdRead(id=uuid7(), title="AdTitle", content="Content", metadata={})
    assert schema.title == "AdTitle"
    assert schema.content == "Content"
    assert isinstance(schema.id, UUID)

def test_ad_example():
    
    """test_ad_example function."""
ad = Ad.example()
    assert isinstance(ad, Ad)
    assert ad.title
    assert ad.content
    assert isinstance(orjson.loads(ad.to_json()), dict)

def test_ad_random():
    
    """test_ad_random function."""
ad = Ad.random()
    assert isinstance(ad, Ad)
    assert ad.title
    assert ad.content

def test_ad_to_json_and_from_json():
    
    """test_ad_to_json_and_from_json function."""
ad = Ad.random()
    data = ad.to_json()
    ad2 = Ad.from_json(data)
    assert ad2.title == ad.title
    assert ad2.content == ad.content

def test_ad_to_training_example_and_from_training_example():
    
    """test_ad_to_training_example_and_from_training_example function."""
ad = Ad.random()
    ex = ad.to_training_example()
    ad2 = Ad.from_training_example(ex)
    assert ad2.title == ad.title
    assert ad2.content == ad.content 