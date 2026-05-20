"""
Pydantic schemas for API requests and responses.
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime


# ==================== Chat Schemas ====================

class MessageSchema(BaseModel):
    """Schema for a chat message."""
    role: str = Field(..., description="Message role: system, user, or assistant")
    content: str = Field(..., description="Message content")
    
    class Config:
        json_schema_extra = {
            "example": {
                "role": "user",
                "content": "Hello, how are you?"
            }
        }


class ChatRequest(BaseModel):
    """Request to send a chat message."""
    message: str = Field(..., min_length=1, max_length=50000, description="User message")
    conversation_id: Optional[str] = Field(None, description="Conversation ID (creates new if not provided)")
    system_prompt: Optional[str] = Field(None, max_length=10000, description="System prompt override")
    model: Optional[str] = Field(None, description="Model to use")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: Optional[int] = Field(None, ge=1, le=8000, description="Maximum tokens to generate")
    stream: bool = Field(False, description="Enable streaming response")
    
    class Config:
        json_schema_extra = {
            "example": {
                "message": "Explain what is Python programming",
                "model": "deepseek/deepseek-chat",
                "temperature": 0.7
            }
        }


class ChatMessageResponse(BaseModel):
    """Response message from chat."""
    role: str
    content: str
    message_id: str
    timestamp: str


class ChatResponse(BaseModel):
    """Response from a chat request."""
    success: bool
    message: Optional[ChatMessageResponse] = None
    conversation_id: str
    model: str
    usage: Optional[Dict[str, Any]] = None
    latency_ms: Optional[float] = None
    error: Optional[str] = None


# ==================== Generate Schemas ====================

class GenerateRequest(BaseModel):
    """Request to generate text."""
    prompt: str = Field(..., min_length=1, max_length=50000, description="User prompt")
    model: Optional[str] = Field(None, description="Model to use")
    system_prompt: Optional[str] = Field(None, max_length=10000, description="System prompt")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: Optional[int] = Field(None, ge=1, le=8000, description="Maximum tokens")
    
    class Config:
        json_schema_extra = {
            "example": {
                "prompt": "Write a haiku about programming",
                "model": "deepseek/deepseek-chat",
                "temperature": 0.8
            }
        }


class GenerateResponse(BaseModel):
    """Response from text generation."""
    success: bool
    content: str = ""
    model: str = ""
    usage: Optional[Dict[str, Any]] = None
    latency_ms: Optional[float] = None
    cached: bool = False
    error: Optional[str] = None


class StreamRequest(BaseModel):
    """Request for streaming generation."""
    prompt: str = Field(..., min_length=1, max_length=50000)
    model: Optional[str] = None
    system_prompt: Optional[str] = None
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(None, ge=1, le=8000)


# ==================== Parallel Generation Schemas ====================

class ParallelGenerateRequest(BaseModel):
    """Request to generate from multiple models."""
    prompt: str = Field(..., min_length=1, max_length=50000)
    models: Optional[List[str]] = Field(None, description="List of models (max 10)")
    system_prompt: Optional[str] = None
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(None, ge=1, le=8000)


class ModelResponse(BaseModel):
    """Response from a single model."""
    model: str
    content: str
    usage: Optional[Dict[str, Any]] = None
    latency_ms: Optional[float] = None
    error: Optional[str] = None


class ParallelGenerateResponse(BaseModel):
    """Response from parallel generation."""
    success: bool
    responses: Dict[str, ModelResponse] = {}
    total_models: int = 0
    successful_models: int = 0


# ==================== Conversation Schemas ====================

class CreateConversationRequest(BaseModel):
    """Request to create a new conversation."""
    system_prompt: Optional[str] = None
    model: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class ConversationResponse(BaseModel):
    """Response with conversation details."""
    conversation_id: str
    message_count: int
    model: Optional[str] = None
    created_at: str
    updated_at: str


class ConversationListResponse(BaseModel):
    """Response with list of conversations."""
    conversations: List[ConversationResponse]
    total: int


class ConversationHistoryResponse(BaseModel):
    """Response with conversation history."""
    conversation_id: str
    messages: List[Dict[str, Any]]
    total: int


# ==================== Models Schemas ====================

class ModelInfo(BaseModel):
    """Information about an available model."""
    id: str
    name: Optional[str] = None
    context_length: Optional[int] = None
    pricing: Optional[Dict[str, Any]] = None


class ModelsResponse(BaseModel):
    """Response with available models."""
    models: List[Dict[str, Any]]
    total: int


# ==================== Stats Schemas ====================

class StatsResponse(BaseModel):
    """Response with service statistics."""
    uptime_seconds: float
    requests: Dict[str, Any]
    cache: Dict[str, Any]
    latency: Dict[str, Any]
    tokens: Dict[str, Any]


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    uptime_seconds: float
    models_available: bool


# ==================== Code Analysis Schemas ====================

class CodeAnalysisRequest(BaseModel):
    """Request to analyze code."""
    code: str = Field(..., min_length=1, max_length=100000)
    language: Optional[str] = None
    analysis_type: str = Field("general", description="general, bugs, performance, security")
    model: Optional[str] = None


class CodeAnalysisResponse(BaseModel):
    """Response from code analysis."""
    success: bool
    analysis: str = ""
    model: str = ""
    latency_ms: Optional[float] = None
    error: Optional[str] = None


# ==================== PDF Export Schemas ====================

class PDFDocumentSchema(BaseModel):
    """Schema for a document to export."""
    id: str = Field(..., description="Document ID")
    content: str = Field(..., description="Document content (text/markdown)")
    documentNumber: Optional[int] = Field(None, description="Document number in sequence")
    title: Optional[str] = Field(None, description="Optional document title")


class PDFExportRequest(BaseModel):
    """Request to export documents to PDF."""
    documents: List[PDFDocumentSchema] = Field(..., min_length=1, max_length=100, description="Documents to export")
    title: Optional[str] = Field("Bulk Export", description="PDF title")
    include_metadata: bool = Field(True, description="Include metadata in PDF")
    page_size: str = Field("A4", description="Page size: A4 or letter")
    
    class Config:
        json_schema_extra = {
            "example": {
                "documents": [
                    {"id": "1", "content": "# Document 1\n\nThis is the first document.", "documentNumber": 1},
                    {"id": "2", "content": "# Document 2\n\nThis is the second document.", "documentNumber": 2}
                ],
                "title": "My Bulk Export",
                "include_metadata": True,
                "page_size": "A4"
            }
        }


class PDFExportResponse(BaseModel):
    """Response from PDF export."""
    success: bool
    filename: str = ""
    size_bytes: int = 0
    message: str = ""
    error: Optional[str] = None


