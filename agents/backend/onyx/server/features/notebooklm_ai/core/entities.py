from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from enum import Enum
import uuid
from pydantic import BaseModel, Field
import json
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Core Entities - NotebookLM AI Domain Models
Advanced document intelligence with AI-powered analysis and citation.
"""



class DocumentType(str, Enum):
    """Document types supported by NotebookLM."""
    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"
    MD = "markdown"
    HTML = "html"
    JSON = "json"
    CSV = "csv"
    EXCEL = "excel"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    CODE = "code"


class SourceType(str, Enum):
    """Source types for citations."""
    DOCUMENT = "document"
    WEBSITE = "website"
    DATABASE = "database"
    API = "api"
    FILE = "file"
    NOTEBOOK = "notebook"


class QueryType(str, Enum):
    """Query types for different interactions."""
    GENERAL = "general"
    ANALYTICAL = "analytical"
    COMPARATIVE = "comparative"
    CREATIVE = "creative"
    TECHNICAL = "technical"
    RESEARCH = "research"


@dataclass
class DocumentId:
    """Document identifier."""
    value: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class NotebookId:
    """Notebook identifier."""
    value: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class SourceId:
    """Source identifier."""
    value: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class QueryId:
    """Query identifier."""
    value: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class ResponseId:
    """Response identifier."""
    value: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class UserId:
    """User identifier."""
    value: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class Document:
    """
    Document entity with AI-powered analysis capabilities.
    """
    id: DocumentId
    title: str
    content: str
    document_type: DocumentType
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    # AI Analysis Fields
    summary: Optional[str] = None
    key_points: List[str] = field(default_factory=list)
    entities: List[str] = field(default_factory=list)
    topics: List[str] = field(default_factory=list)
    sentiment: Optional[float] = None
    readability_score: Optional[float] = None
    word_count: int = 0
    character_count: int = 0
    
    # Processing Status
    is_processed: bool = False
    processing_errors: List[str] = field(default_factory=list)
    
    def __post_init__(self) -> Any:
        """Initialize computed fields."""
        self.word_count = len(self.content.split())
        self.character_count = len(self.content)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id.value,
            "title": self.title,
            "content": self.content,
            "document_type": self.document_type.value,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "summary": self.summary,
            "key_points": self.key_points,
            "entities": self.entities,
            "topics": self.topics,
            "sentiment": self.sentiment,
            "readability_score": self.readability_score,
            "word_count": self.word_count,
            "character_count": self.character_count,
            "is_processed": self.is_processed,
            "processing_errors": self.processing_errors
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Document':
        """Create from dictionary."""
        return cls(
            id=DocumentId(data["id"]),
            title=data["title"],
            content=data["content"],
            document_type=DocumentType(data["document_type"]),
            metadata=data.get("metadata", {}),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            summary=data.get("summary"),
            key_points=data.get("key_points", []),
            entities=data.get("entities", []),
            topics=data.get("topics", []),
            sentiment=data.get("sentiment"),
            readability_score=data.get("readability_score"),
            is_processed=data.get("is_processed", False),
            processing_errors=data.get("processing_errors", [])
        )


@dataclass
class Source:
    """
    Source entity for citations and references.
    """
    id: SourceId
    title: str
    url: Optional[str] = None
    source_type: SourceType = SourceType.DOCUMENT
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    # Citation Information
    authors: List[str] = field(default_factory=list)
    publication_date: Optional[datetime] = None
    publisher: Optional[str] = None
    doi: Optional[str] = None
    isbn: Optional[str] = None
    
    # Content Analysis
    summary: Optional[str] = None
    key_insights: List[str] = field(default_factory=list)
    relevance_score: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id.value,
            "title": self.title,
            "url": self.url,
            "source_type": self.source_type.value,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "authors": self.authors,
            "publication_date": self.publication_date.isoformat() if self.publication_date else None,
            "publisher": self.publisher,
            "doi": self.doi,
            "isbn": self.isbn,
            "summary": self.summary,
            "key_insights": self.key_insights,
            "relevance_score": self.relevance_score
        }


@dataclass
class Citation:
    """
    Citation entity for referencing sources.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_id: SourceId = None
    document_id: Optional[DocumentId] = None
    text: str = ""
    page_number: Optional[int] = None
    line_number: Optional[int] = None
    confidence_score: float = 1.0
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "source_id": self.source_id.value if self.source_id else None,
            "document_id": self.document_id.value if self.document_id else None,
            "text": self.text,
            "page_number": self.page_number,
            "line_number": self.line_number,
            "confidence_score": self.confidence_score,
            "created_at": self.created_at.isoformat()
        }


@dataclass
class Query:
    """
    Query entity for user interactions.
    """
    id: QueryId
    text: str
    query_type: QueryType = QueryType.GENERAL
    user_id: UserId = None
    notebook_id: Optional[NotebookId] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    # Query Analysis
    intent: Optional[str] = None
    entities: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    complexity_score: Optional[float] = None
    
    # Context
    context_documents: List[DocumentId] = field(default_factory=list)
    previous_queries: List[QueryId] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id.value,
            "text": self.text,
            "query_type": self.query_type.value,
            "user_id": self.user_id.value if self.user_id else None,
            "notebook_id": self.notebook_id.value if self.notebook_id else None,
            "created_at": self.created_at.isoformat(),
            "intent": self.intent,
            "entities": self.entities,
            "keywords": self.keywords,
            "complexity_score": self.complexity_score,
            "context_documents": [doc.value for doc in self.context_documents],
            "previous_queries": [query.value for query in self.previous_queries]
        }


@dataclass
class Response:
    """
    AI-generated response with citations.
    """
    id: ResponseId
    query_id: QueryId
    content: str
    citations: List[Citation] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    # Response Analysis
    confidence_score: float = 1.0
    relevance_score: float = 1.0
    completeness_score: float = 1.0
    
    # Metadata
    model_used: Optional[str] = None
    processing_time: Optional[float] = None
    token_count: Optional[int] = None
    
    # Additional Information
    suggestions: List[str] = field(default_factory=list)
    follow_up_questions: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id.value,
            "query_id": self.query_id.value,
            "content": self.content,
            "citations": [citation.to_dict() for citation in self.citations],
            "created_at": self.created_at.isoformat(),
            "confidence_score": self.confidence_score,
            "relevance_score": self.relevance_score,
            "completeness_score": self.completeness_score,
            "model_used": self.model_used,
            "processing_time": self.processing_time,
            "token_count": self.token_count,
            "suggestions": self.suggestions,
            "follow_up_questions": self.follow_up_questions
        }


@dataclass
class Conversation:
    """
    Conversation entity for multi-turn interactions.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: UserId = None
    notebook_id: Optional[NotebookId] = None
    title: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    # Conversation Flow
    queries: List[Query] = field(default_factory=list)
    responses: List[Response] = field(default_factory=list)
    
    # Analysis
    topic: Optional[str] = None
    sentiment: Optional[float] = None
    complexity_level: Optional[str] = None
    
    def add_interaction(self, query: Query, response: Response):
        """Add a query-response interaction."""
        self.queries.append(query)
        self.responses.append(response)
        self.updated_at = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "user_id": self.user_id.value if self.user_id else None,
            "notebook_id": self.notebook_id.value if self.notebook_id else None,
            "title": self.title,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "queries": [query.to_dict() for query in self.queries],
            "responses": [response.to_dict() for response in self.responses],
            "topic": self.topic,
            "sentiment": self.sentiment,
            "complexity_level": self.complexity_level
        }


@dataclass
class Notebook:
    """
    Notebook entity - the main workspace for document intelligence.
    """
    id: NotebookId
    title: str
    description: str = ""
    user_id: UserId = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    # Content
    documents: List[Document] = field(default_factory=list)
    sources: List[Source] = field(default_factory=list)
    conversations: List[Conversation] = field(default_factory=list)
    
    # Settings
    is_public: bool = False
    allow_collaboration: bool = False
    ai_settings: Dict[str, Any] = field(default_factory=dict)
    
    # Analytics
    total_documents: int = 0
    total_conversations: int = 0
    last_activity: Optional[datetime] = None
    
    def add_document(self, document: Document):
        """Add a document to the notebook."""
        self.documents.append(document)
        self.total_documents = len(self.documents)
        self.updated_at = datetime.utcnow()
        self.last_activity = datetime.utcnow()
    
    def add_source(self, source: Source):
        """Add a source to the notebook."""
        self.sources.append(source)
        self.updated_at = datetime.utcnow()
    
    def add_conversation(self, conversation: Conversation):
        """Add a conversation to the notebook."""
        self.conversations.append(conversation)
        self.total_conversations = len(self.conversations)
        self.updated_at = datetime.utcnow()
        self.last_activity = datetime.utcnow()
    
    def get_document_by_id(self, document_id: DocumentId) -> Optional[Document]:
        """Get document by ID."""
        for doc in self.documents:
            if doc.id.value == document_id.value:
                return doc
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id.value,
            "title": self.title,
            "description": self.description,
            "user_id": self.user_id.value if self.user_id else None,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "documents": [doc.to_dict() for doc in self.documents],
            "sources": [source.to_dict() for source in self.sources],
            "conversations": [conv.to_dict() for conv in self.conversations],
            "is_public": self.is_public,
            "allow_collaboration": self.allow_collaboration,
            "ai_settings": self.ai_settings,
            "total_documents": self.total_documents,
            "total_conversations": self.total_conversations,
            "last_activity": self.last_activity.isoformat() if self.last_activity else None
        }


@dataclass
class User:
    """
    User entity for authentication and personalization.
    """
    id: UserId
    username: str
    email: str
    full_name: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    # Preferences
    preferences: Dict[str, Any] = field(default_factory=dict)
    ai_settings: Dict[str, Any] = field(default_factory=dict)
    
    # Usage Statistics
    total_notebooks: int = 0
    total_queries: int = 0
    total_documents: int = 0
    
    # Subscription
    subscription_tier: str = "free"
    subscription_expires: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id.value,
            "username": self.username,
            "email": self.email,
            "full_name": self.full_name,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "preferences": self.preferences,
            "ai_settings": self.ai_settings,
            "total_notebooks": self.total_notebooks,
            "total_queries": self.total_queries,
            "total_documents": self.total_documents,
            "subscription_tier": self.subscription_tier,
            "subscription_expires": self.subscription_expires.isoformat() if self.subscription_expires else None
        } 