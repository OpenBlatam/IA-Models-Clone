"""
NotebookLM AI - Advanced Document Intelligence System
=====================================================

Inspired by Google's NotebookLM with latest AI libraries and optimizations.
"""

from typing import Any, List, Dict, Optional
import logging
import asyncio

from .core.entities import (
    Document,
    Notebook,
    Source,
    Citation,
    Query,
    Response,
    Conversation,
    User,
)
from .core.value_objects import (
    DocumentId,
    NotebookId,
    SourceId,
    QueryId,
    ResponseId,
    UserId,
    DocumentType,
    SourceType,
    QueryType,
)
from .core.repositories import (
    DocumentRepository,
    NotebookRepository,
    SourceRepository,
    ConversationRepository,
    UserRepository,
)
from .application.use_cases import (
    CreateNotebookUseCase,
    AddDocumentUseCase,
    QueryNotebookUseCase,
    GenerateResponseUseCase,
    ManageSourcesUseCase,
    AnalyzeDocumentsUseCase,
)
from .infrastructure.ai_engines import (
    AdvancedLLMEngine,
    DocumentProcessor,
    CitationGenerator,
    ResponseOptimizer,
    MultiModalProcessor,
)
from .presentation.api import (
    NotebookLMRouter,
    create_notebooklm_app,
)
from .shared.config import (
    NotebookLMConfig,
    AIEngineConfig,
    DatabaseConfig,
)

__version__ = "1.0.0"
__author__ = "Blatam Academy"
__license__ = "Proprietary"
__description__ = "Advanced document intelligence system inspired by Google's NotebookLM with latest AI libraries and optimizations"

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