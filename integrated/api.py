from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from typing import Dict, List, Optional, Union, Any, TypeVar, Generic
from datetime import datetime
import logging
from pydantic import Field, validator, root_validator, ValidationError as PydanticValidationError
from ...utils.base_model import OnyxBaseModel
from ..utils.error_system import (
from typing import Any, List, Dict, Optional
import asyncio
"""
Integrated API Models - Onyx Integration
Enhanced models for integrated API with advanced features and proper error handling.
"""
    error_factory,
    ErrorContext,
    ValidationError,
    handle_errors,
    ErrorCategory
)

logger = logging.getLogger(__name__)

T = TypeVar('T')

class DocumentRequest(OnyxBaseModel):
    """Enhanced document processing request with improved error handling."""
    
    document_url: Optional[str] = None
    document_content: Optional[str] = None
    document_type: str = Field(default="text")
    language: str = Field(default="en")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # Configure indexing
    index_fields = ["document_type", "language"]
    search_fields = ["document_content"]
    
    @root_validator
    def validate_document_source(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Validate that either URL or content is provided with user-friendly error messages."""
        document_url = values.get("document_url")
        document_content = values.get("document_content")
        
        # Guard clause: Check if both are missing
        if not document_url and not document_content:
            context = ErrorContext(
                operation="validate_document_source",
                additional_data={"document_url": document_url, "document_content": document_content}
            )
            raise error_factory.create_validation_error(
                "Neither URL nor content provided",
                field="document_source",
                validation_errors=["Either document_url or document_content must be provided"],
                context=context
            )
        
        # Guard clause: Check if both are provided (ambiguous)
        if document_url and document_content:
            logger.warning("Document request validation: Both URL and content provided, URL will take precedence")
            values["document_content"] = None  # Clear content to avoid confusion
        
        return values
    
    @validator("document_type")
    def validate_document_type(cls, v: str) -> str:
        """Validate document type with user-friendly error messages."""
        allowed_types = ["text", "pdf", "doc", "docx", "html"]
        
        # Guard clause: Check if type is valid
        if v not in allowed_types:
            context = ErrorContext(
                operation="validate_document_type",
                additional_data={"document_type": v, "allowed_types": allowed_types}
            )
            raise error_factory.create_validation_error(
                f"Document type '{v}' is not supported",
                field="document_type",
                value=v,
                validation_errors=[f"Document type must be one of: {', '.join(allowed_types)}"],
                context=context
            )
        
        return v
    
    @validator("language")
    def validate_language(cls, v: str) -> str:
        """Validate language code with user-friendly error messages."""
        # Guard clause: Check language code length
        if len(v) != 2:
            context = ErrorContext(
                operation="validate_language",
                additional_data={"language": v}
            )
            raise error_factory.create_validation_error(
                f"Language code '{v}' is invalid",
                field="language",
                value=v,
                validation_errors=["Language must be a 2-letter language code (e.g., 'en', 'es', 'fr')"],
                context=context
            )
        
        return v.lower()

class AdsRequest(OnyxBaseModel):
    """Enhanced ads generation request with improved error handling."""
    
    ads_type: str
    target_audience: str
    platform: str
    brand_voice: Optional[Dict[str, Any]] = None
    content_guidelines: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # Configure indexing
    index_fields = ["ads_type", "platform", "target_audience"]
    search_fields = ["content_guidelines"]
    
    @validator("ads_type")
    def validate_ads_type(cls, v: str) -> str:
        """Validate ads type with user-friendly error messages."""
        allowed_types = ["social", "display", "search", "video"]
        
        # Guard clause: Check if type is valid
        if v not in allowed_types:
            context = ErrorContext(
                operation="validate_ads_type",
                additional_data={"ads_type": v, "allowed_types": allowed_types}
            )
            raise error_factory.create_validation_error(
                f"Ads type '{v}' is not supported",
                field="ads_type",
                value=v,
                validation_errors=[f"Ads type must be one of: {', '.join(allowed_types)}"],
                context=context
            )
        
        return v
    
    @validator("platform")
    def validate_platform(cls, v: str) -> str:
        """Validate platform with user-friendly error messages."""
        allowed_platforms = ["facebook", "instagram", "twitter", "linkedin", "google"]
        
        # Guard clause: Check if platform is valid
        if v not in allowed_platforms:
            context = ErrorContext(
                operation="validate_platform",
                additional_data={"platform": v, "allowed_platforms": allowed_platforms}
            )
            raise error_factory.create_validation_error(
                f"Platform '{v}' is not supported",
                field="platform",
                value=v,
                validation_errors=[f"Platform must be one of: {', '.join(allowed_platforms)}"],
                context=context
            )
        
        return v

class ChatRequest(OnyxBaseModel):
    """Enhanced chat request with improved error handling."""
    
    chat_message: str
    context: Optional[Dict[str, Any]] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # Configure indexing
    index_fields = ["user_id", "session_id"]
    search_fields = ["chat_message"]
    
    @validator("chat_message")
    def validate_message(cls, v: str) -> str:
        """Validate chat message with user-friendly error messages."""
        # Guard clause: Check if message is empty or whitespace only
        if not v or not v.strip():
            context = ErrorContext(
                operation="validate_message",
                additional_data={"chat_message": v}
            )
            raise error_factory.create_validation_error(
                "Chat message cannot be empty",
                field="chat_message",
                value=v,
                validation_errors=["Please provide a message to continue"],
                context=context
            )
        
        return v.strip()

class FileRequest(OnyxBaseModel):
    """Enhanced file processing request with improved error handling."""
    
    file_url: Optional[str] = None
    file_content: Optional[bytes] = None
    file_type: str
    file_name: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # Configure indexing
    index_fields = ["file_type", "file_name"]
    search_fields = ["file_name"]
    
    @root_validator
    def validate_file_source(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Validate that either URL or content is provided with user-friendly error messages."""
        file_url = values.get("file_url")
        file_content = values.get("file_content")
        
        # Guard clause: Check if both are missing
        if not file_url and not file_content:
            context = ErrorContext(
                operation="validate_file_source",
                additional_data={"file_url": file_url, "file_content": file_content}
            )
            raise error_factory.create_validation_error(
                "Neither URL nor content provided",
                field="file_source",
                validation_errors=["Either file_url or file_content must be provided"],
                context=context
            )
        
        # Guard clause: Check if both are provided (ambiguous)
        if file_url and file_content:
            logger.warning("File request validation: Both URL and content provided, URL will take precedence")
            values["file_content"] = None  # Clear content to avoid confusion
        
        return values
    
    @validator("file_type")
    def validate_file_type(cls, v: str) -> str:
        """Validate file type with user-friendly error messages."""
        allowed_types = ["image", "video", "audio", "document"]
        
        # Guard clause: Check if type is valid
        if v not in allowed_types:
            context = ErrorContext(
                operation="validate_file_type",
                additional_data={"file_type": v, "allowed_types": allowed_types}
            )
            raise error_factory.create_validation_error(
                f"File type '{v}' is not supported",
                field="file_type",
                value=v,
                validation_errors=[f"File type must be one of: {', '.join(allowed_types)}"],
                context=context
            )
        
        return v

class NLPRequest(OnyxBaseModel):
    """Enhanced NLP request with improved error handling."""
    
    text: str
    nlp_task: str
    language: str = Field(default="en")
    options: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # Configure indexing
    index_fields = ["nlp_task", "language"]
    search_fields = ["text"]
    
    @validator("nlp_task")
    def validate_nlp_task(cls, v: str) -> str:
        """Validate NLP task with user-friendly error messages."""
        allowed_tasks = ["sentiment", "classification", "extraction", "summarization"]
        
        # Guard clause: Check if task is valid
        if v not in allowed_tasks:
            context = ErrorContext(
                operation="validate_nlp_task",
                additional_data={"nlp_task": v, "allowed_tasks": allowed_tasks}
            )
            raise error_factory.create_validation_error(
                f"NLP task '{v}' is not supported",
                field="nlp_task",
                value=v,
                validation_errors=[f"NLP task must be one of: {', '.join(allowed_tasks)}"],
                context=context
            )
        
        return v
    
    @validator("language")
    def validate_language(cls, v: str) -> str:
        """Validate language code with user-friendly error messages."""
        # Guard clause: Check language code length
        if len(v) != 2:
            context = ErrorContext(
                operation="validate_language",
                additional_data={"language": v}
            )
            raise error_factory.create_validation_error(
                f"Language code '{v}' is invalid",
                field="language",
                value=v,
                validation_errors=["Language must be a 2-letter language code (e.g., 'en', 'es', 'fr')"],
                context=context
            )
        
        return v.lower()

class AgentRequest(OnyxBaseModel):
    """Enhanced agent request with improved error handling."""
    
    agent_task: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    context: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # Configure indexing
    index_fields = ["agent_task"]
    search_fields = ["parameters"]
    
    @validator("agent_task")
    def validate_agent_task(cls, v: str) -> str:
        """Validate agent task with user-friendly error messages."""
        # Guard clause: Check if task is empty or whitespace only
        if not v or not v.strip():
            context = ErrorContext(
                operation="validate_agent_task",
                additional_data={"agent_task": v}
            )
            raise error_factory.create_validation_error(
                "Agent task cannot be empty",
                field="agent_task",
                value=v,
                validation_errors=["Please provide a task description"],
                context=context
            )
        
        return v.strip()

class IntegratedRequest(OnyxBaseModel):
    """Enhanced integrated request with improved error handling."""
    
    document_request: Optional[DocumentRequest] = None
    ads_request: Optional[AdsRequest] = None
    chat_request: Optional[ChatRequest] = None
    file_request: Optional[FileRequest] = None
    nlp_request: Optional[NLPRequest] = None
    agent_request: Optional[AgentRequest] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # Configure indexing
    index_fields = ["id"]
    search_fields = ["metadata"]
    
    @root_validator
    async def validate_request_types(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Validate that at least one request type is provided with user-friendly error messages."""
        request_types = [
            "document_request",
            "ads_request",
            "chat_request",
            "file_request",
            "nlp_request",
            "agent_request"
        ]
        
        # Guard clause: Check if at least one request type is provided
        if not any(values.get(rt) for rt in request_types):
            context = ErrorContext(
                operation="validate_request_types",
                additional_data={"request_types": request_types}
            )
            raise error_factory.create_validation_error(
                "No request types provided",
                field="request_types",
                validation_errors=[
                    "At least one request type must be provided. "
                    "Please include one of: document_request, ads_request, chat_request, "
                    "file_request, nlp_request, or agent_request."
                ],
                context=context
            )
        
        return values

class IntegratedResponse(OnyxBaseModel, Generic[T]):
    """Enhanced integrated response with improved error handling."""
    
    request_id: str
    status: str
    result: T
    metadata: Dict[str, Any] = Field(default_factory=dict)
    performance_metrics: Dict[str, float] = Field(default_factory=dict)
    
    # Configure indexing
    index_fields = ["request_id", "status"]
    search_fields = ["metadata"]
    
    @validator("status")
    def validate_status(cls, v: str) -> str:
        """Validate status with user-friendly error messages."""
        allowed_statuses = ["success", "error", "processing"]
        
        # Guard clause: Check if status is valid
        if v not in allowed_statuses:
            context = ErrorContext(
                operation="validate_status",
                additional_data={"status": v, "allowed_statuses": allowed_statuses}
            )
            raise error_factory.create_validation_error(
                f"Status '{v}' is not valid",
                field="status",
                value=v,
                validation_errors=[f"Status must be one of: {', '.join(allowed_statuses)}"],
                context=context
            )
        
        return v

# Example usage:
"""
# Create document request
doc_request = DocumentRequest(
    document_url="https://example.com/doc.pdf",
    document_type="pdf",
    language="en"
)

# Create ads request
ads_request = AdsRequest(
    ads_type="social",
    target_audience="Tech Professionals",
    platform="linkedin",
    brand_voice={
        "tone": "professional",
        "style": "formal"
    }
)

# Create chat request
chat_request = ChatRequest(
    chat_message="Hello, how can I help you?",
    user_id="user123",
    session_id="session456"
)

# Create integrated request
integrated_request = IntegratedRequest(
    document_request=doc_request,
    ads_request=ads_request,
    chat_request=chat_request,
    metadata={"priority": "high"}
)

# Index request
redis_indexer = RedisIndexer()
integrated_request.index(redis_indexer)

# Create response
response = IntegratedResponse(
    request_id="req123",
    status="success",
    result={
        "document": {"processed": True},
        "ads": {"generated": True},
        "chat": {"response": "I can help you with that!"}
    }
)

# Index response
response.index(redis_indexer)
""" 