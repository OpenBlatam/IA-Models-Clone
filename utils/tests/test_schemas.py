from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
import pytest
from agents.backend.onyx.server.features.utils.schemas import GenerationRequest, BatchGenerationRequest, GenerationResponse, BatchGenerationResponse, TokenResponse, RefreshTokenRequest
import orjson

from typing import Any, List, Dict, Optional
import logging
import asyncio
def test_generation_request_example():
    
    """test_generation_request_example function."""
obj = GenerationRequest.example()
    assert isinstance(obj, GenerationRequest)
    assert obj.prompt
    assert isinstance(orjson.loads(obj.json()), dict)

def test_generation_request_random():
    
    """test_generation_request_random function."""
obj = GenerationRequest.random()
    assert isinstance(obj, GenerationRequest)
    assert obj.prompt


def test_batch_generation_request_example():
    
    """test_batch_generation_request_example function."""
obj = BatchGenerationRequest.example()
    assert isinstance(obj, BatchGenerationRequest)
    assert obj.prompts
    assert isinstance(orjson.loads(obj.json()), dict)

def test_batch_generation_request_random():
    
    """test_batch_generation_request_random function."""
obj = BatchGenerationRequest.random()
    assert isinstance(obj, BatchGenerationRequest)
    assert obj.prompts


def test_generation_response_example():
    
    """test_generation_response_example function."""
obj = GenerationResponse.example()
    assert isinstance(obj, GenerationResponse)
    assert obj.result
    assert isinstance(orjson.loads(obj.json()), dict)

def test_generation_response_random():
    
    """test_generation_response_random function."""
obj = GenerationResponse.random()
    assert isinstance(obj, GenerationResponse)
    assert obj.result


def test_batch_generation_response_example():
    
    """test_batch_generation_response_example function."""
obj = BatchGenerationResponse.example()
    assert isinstance(obj, BatchGenerationResponse)
    assert obj.results
    assert isinstance(orjson.loads(obj.json()), dict)

def test_batch_generation_response_random():
    
    """test_batch_generation_response_random function."""
obj = BatchGenerationResponse.random()
    assert isinstance(obj, BatchGenerationResponse)
    assert obj.results


def test_token_response_example():
    
    """test_token_response_example function."""
obj = TokenResponse.example()
    assert isinstance(obj, TokenResponse)
    assert obj.access_token
    assert isinstance(orjson.loads(obj.json()), dict)

def test_token_response_random():
    
    """test_token_response_random function."""
obj = TokenResponse.random()
    assert isinstance(obj, TokenResponse)
    assert obj.access_token


def test_refresh_token_request_example():
    
    """test_refresh_token_request_example function."""
obj = RefreshTokenRequest.example()
    assert isinstance(obj, RefreshTokenRequest)
    assert obj.refresh_token
    assert isinstance(orjson.loads(obj.json()), dict)

def test_refresh_token_request_random():
    
    """test_refresh_token_request_random function."""
obj = RefreshTokenRequest.random()
    assert isinstance(obj, RefreshTokenRequest)
    assert obj.refresh_token 