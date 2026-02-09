"""
Document Management
==================

Document management modules.
"""

from aws.modules.document.document_manager import DocumentManager, Document, DocumentStatus
from aws.modules.document.version_control import VersionControl, DocumentVersion
from aws.modules.document.search_engine import SearchEngine, SearchResult

__all__ = [
    "DocumentManager",
    "Document",
    "DocumentStatus",
    "VersionControl",
    "DocumentVersion",
    "SearchEngine",
    "SearchResult",
]

