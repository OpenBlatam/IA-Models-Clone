from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from .core.entities import (
from .core.value_objects import (
from .core.repositories import (
from .application.use_cases import (
from .infrastructure.ai_engines import (
from .presentation.api import (
from .shared.config import (
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
NotebookLM AI - Advanced Document Intelligence System
Inspired by Google's NotebookLM with latest AI libraries and optimizations.
"""

    Document,
    Notebook,
    Source,
    Citation,
    Query,
    Response,
    Conversation,
    User
)

    DocumentId,
    NotebookId,
    SourceId,
    QueryId,
    ResponseId,
    UserId,
    DocumentType,
    SourceType,
    QueryType
)

    DocumentRepository,
    NotebookRepository,
    SourceRepository,
    ConversationRepository,
    UserRepository
)

    CreateNotebookUseCase,
    AddDocumentUseCase,
    QueryNotebookUseCase,
    GenerateResponseUseCase,
    ManageSourcesUseCase,
    AnalyzeDocumentsUseCase
)

    AdvancedLLMEngine,
    DocumentProcessor,
    CitationGenerator,
    ResponseOptimizer,
    MultiModalProcessor
)

    NotebookLMRouter,
    create_notebooklm_app
)

    NotebookLMConfig,
    AIEngineConfig,
    DatabaseConfig
)

__version__ = "1.0.0"
__author__ = "NotebookLM AI Team"

__all__ = [
    # Core Entities
    "Document",
    "Notebook", 
    "Source",
    "Citation",
    "Query",
    "Response",
    "Conversation",
    "User",
    
    # Value Objects
    "DocumentId",
    "NotebookId",
    "SourceId",
    "QueryId",
    "ResponseId",
    "UserId",
    "DocumentType",
    "SourceType",
    "QueryType",
    
    # Repositories
    "DocumentRepository",
    "NotebookRepository",
    "SourceRepository",
    "ConversationRepository",
    "UserRepository",
    
    # Use Cases
    "CreateNotebookUseCase",
    "AddDocumentUseCase",
    "QueryNotebookUseCase",
    "GenerateResponseUseCase",
    "ManageSourcesUseCase",
    "AnalyzeDocumentsUseCase",
    
    # AI Engines
    "AdvancedLLMEngine",
    "DocumentProcessor",
    "CitationGenerator",
    "ResponseOptimizer",
    "MultiModalProcessor",
    
    # API
    "NotebookLMRouter",
    "create_notebooklm_app",
    
    # Config
    "NotebookLMConfig",
    "AIEngineConfig",
    "DatabaseConfig"
] 