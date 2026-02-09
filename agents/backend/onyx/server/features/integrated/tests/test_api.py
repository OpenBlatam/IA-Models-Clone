from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
import pytest
from datetime import datetime
from typing import Dict, Any
from ..api import (
from typing import Any, List, Dict, Optional
import logging
import asyncio
    DocumentRequest,
    AdsRequest,
    ChatRequest,
    FileRequest,
    NLPRequest,
    AgentRequest,
    IntegratedRequest,
    IntegratedResponse
)

# Test DocumentRequest
def test_document_request_validation():
    """Test DocumentRequest validation."""
    # Test valid request with URL
    valid_url_request = DocumentRequest(
        document_url="https://example.com/doc.pdf",
        document_type="pdf",
        language="en"
    )
    assert valid_url_request.document_url == "https://example.com/doc.pdf"
    assert valid_url_request.document_type == "pdf"
    assert valid_url_request.language == "en"

    # Test valid request with content
    valid_content_request = DocumentRequest(
        document_content="Sample content",
        document_type="text",
        language="es"
    )
    assert valid_content_request.document_content == "Sample content"
    assert valid_content_request.document_type == "text"
    assert valid_content_request.language == "es"

    # Test invalid document type
    with pytest.raises(ValueError, match="Document type must be one of"):
        DocumentRequest(
            document_content="Sample content",
            document_type="invalid",
            language="en"
        )

    # Test invalid language code
    with pytest.raises(ValueError, match="Language must be a 2-letter code"):
        DocumentRequest(
            document_content="Sample content",
            document_type="text",
            language="english"
        )

    # Test missing source
    with pytest.raises(ValueError, match="Either document_url or document_content must be provided"):
        DocumentRequest(
            document_type="text",
            language="en"
        )

# Test AdsRequest
def test_ads_request_validation():
    """Test AdsRequest validation."""
    # Test valid request
    valid_request = AdsRequest(
        ads_type="social",
        target_audience="young_adults",
        platform="facebook",
        brand_voice={"tone": "casual"}
    )
    assert valid_request.ads_type == "social"
    assert valid_request.platform == "facebook"
    assert valid_request.brand_voice == {"tone": "casual"}

    # Test invalid ads type
    with pytest.raises(ValueError, match="Ads type must be one of"):
        AdsRequest(
            ads_type="invalid",
            target_audience="young_adults",
            platform="facebook"
        )

    # Test invalid platform
    with pytest.raises(ValueError, match="Platform must be one of"):
        AdsRequest(
            ads_type="social",
            target_audience="young_adults",
            platform="invalid"
        )

# Test ChatRequest
def test_chat_request_validation():
    """Test ChatRequest validation."""
    # Test valid request
    valid_request = ChatRequest(
        chat_message="Hello, how can I help?",
        user_id="user123",
        session_id="session456"
    )
    assert valid_request.chat_message == "Hello, how can I help?"
    assert valid_request.user_id == "user123"
    assert valid_request.session_id == "session456"

    # Test empty message
    with pytest.raises(ValueError, match="Chat message cannot be empty"):
        ChatRequest(
            chat_message="   ",
            user_id="user123"
        )

# Test FileRequest
def test_file_request_validation():
    """Test FileRequest validation."""
    # Test valid request with URL
    valid_url_request = FileRequest(
        file_url="https://example.com/image.jpg",
        file_type="image",
        file_name="image.jpg"
    )
    assert valid_url_request.file_url == "https://example.com/image.jpg"
    assert valid_url_request.file_type == "image"
    assert valid_url_request.file_name == "image.jpg"

    # Test valid request with content
    valid_content_request = FileRequest(
        file_content=b"sample content",
        file_type="document",
        file_name="doc.txt"
    )
    assert valid_content_request.file_content == b"sample content"
    assert valid_content_request.file_type == "document"
    assert valid_content_request.file_name == "doc.txt"

    # Test invalid file type
    with pytest.raises(ValueError, match="File type must be one of"):
        FileRequest(
            file_url="https://example.com/file",
            file_type="invalid"
        )

    # Test missing source
    with pytest.raises(ValueError, match="Either file_url or file_content must be provided"):
        FileRequest(
            file_type="image"
        )

# Test NLPRequest
def test_nlp_request_validation():
    """Test NLPRequest validation."""
    # Test valid request
    valid_request = NLPRequest(
        text="Sample text for analysis",
        nlp_task="sentiment",
        language="en"
    )
    assert valid_request.text == "Sample text for analysis"
    assert valid_request.nlp_task == "sentiment"
    assert valid_request.language == "en"

    # Test invalid NLP task
    with pytest.raises(ValueError, match="NLP task must be one of"):
        NLPRequest(
            text="Sample text",
            nlp_task="invalid",
            language="en"
        )

    # Test invalid language code
    with pytest.raises(ValueError, match="Language must be a 2-letter code"):
        NLPRequest(
            text="Sample text",
            nlp_task="sentiment",
            language="english"
        )

# Test AgentRequest
def test_agent_request_validation():
    """Test AgentRequest validation."""
    # Test valid request
    valid_request = AgentRequest(
        agent_task="process_data",
        parameters={"param1": "value1"}
    )
    assert valid_request.agent_task == "process_data"
    assert valid_request.parameters == {"param1": "value1"}

    # Test empty task
    with pytest.raises(ValueError, match="Agent task cannot be empty"):
        AgentRequest(
            agent_task="   ",
            parameters={}
        )

# Test IntegratedRequest
def test_integrated_request_validation():
    """Test IntegratedRequest validation."""
    # Test valid request with document
    valid_doc_request = IntegratedRequest(
        document_request=DocumentRequest(
            document_content="Sample content",
            document_type="text",
            language="en"
        )
    )
    assert valid_doc_request.document_request is not None
    assert valid_doc_request.document_request.document_content == "Sample content"

    # Test valid request with multiple types
    valid_multi_request = IntegratedRequest(
        document_request=DocumentRequest(
            document_content="Sample content",
            document_type="text",
            language="en"
        ),
        chat_request=ChatRequest(
            chat_message="Hello",
            user_id="user123"
        )
    )
    assert valid_multi_request.document_request is not None
    assert valid_multi_request.chat_request is not None

    # Test empty request
    with pytest.raises(ValueError, match="At least one request type must be provided"):
        IntegratedRequest()

# Test IntegratedResponse
def test_integrated_response_validation():
    """Test IntegratedResponse validation."""
    # Test valid response
    valid_response = IntegratedResponse[str](
        request_id="req123",
        status="success",
        result="Processed successfully",
        metadata={"key": "value"},
        performance_metrics={"time": 1.5}
    )
    assert valid_response.request_id == "req123"
    assert valid_response.status == "success"
    assert valid_response.result == "Processed successfully"
    assert valid_response.metadata == {"key": "value"}
    assert valid_response.performance_metrics == {"time": 1.5}

    # Test response with different result type
    valid_response_dict = IntegratedResponse[Dict[str, Any]](
        request_id="req124",
        status="success",
        result={"key": "value"},
        metadata={}
    )
    assert valid_response_dict.result == {"key": "value"} 