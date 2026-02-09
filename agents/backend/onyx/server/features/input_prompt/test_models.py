from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
import pytest
from uuid6 import UUID, uuid7
from agents.backend.onyx.server.features.input_prompt.models import InputPrompt
from agents.backend.onyx.server.features.input_prompt.schemas import InputPromptCreate, InputPromptRead

from typing import Any, List, Dict, Optional
import logging
import asyncio
def test_input_prompt_valid():
    
    """test_input_prompt_valid function."""
prompt = InputPrompt(prompt="Say hi", metadata={"lang": "en"})
    assert isinstance(prompt.id, UUID)
    assert prompt.prompt == "Say hi"
    assert prompt.metadata == {"lang": "en"}

def test_input_prompt_empty_prompt_raises():
    
    """test_input_prompt_empty_prompt_raises function."""
with pytest.raises(ValueError):
        InputPrompt(prompt=" ", metadata={})

def test_input_prompt_metadata_not_dict():
    
    """test_input_prompt_metadata_not_dict function."""
with pytest.raises(ValueError):
        InputPrompt(prompt="Test", metadata=[1,2,3])

def test_input_prompt_serialization():
    
    """test_input_prompt_serialization function."""
prompt = InputPrompt(prompt="Say hi", metadata={})
    data = prompt.model_dump_json()
    assert '"prompt":"Say hi"' in data

def test_input_prompt_create_valid():
    
    """test_input_prompt_create_valid function."""
schema = InputPromptCreate(prompt="Say hi", metadata={})
    assert schema.prompt == "Say hi"

def test_input_prompt_create_invalid_prompt():
    
    """test_input_prompt_create_invalid_prompt function."""
with pytest.raises(ValueError):
        InputPromptCreate(prompt=" ", metadata={})

def test_input_prompt_read_valid():
    
    """test_input_prompt_read_valid function."""
schema = InputPromptRead(id=uuid7(), prompt="Say hi", metadata={})
    assert schema.prompt == "Say hi"
    assert isinstance(schema.id, UUID) 