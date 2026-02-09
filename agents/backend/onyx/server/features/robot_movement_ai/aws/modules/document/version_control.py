"""
Version Control
===============

Document version control.
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class DocumentVersion:
    """Document version."""
    version: str
    document_id: str
    content: str
    author: str
    created_at: datetime
    message: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class VersionControl:
    """Document version control."""
    
    def __init__(self):
        self._versions: Dict[str, List[DocumentVersion]] = {}  # document_id -> versions
    
    def create_version(
        self,
        document_id: str,
        content: str,
        author: str,
        message: Optional[str] = None
    ) -> DocumentVersion:
        """Create new version."""
        if document_id not in self._versions:
            self._versions[document_id] = []
        
        version_number = len(self._versions[document_id]) + 1
        version = DocumentVersion(
            version=f"v{version_number}",
            document_id=document_id,
            content=content,
            author=author,
            created_at=datetime.now(),
            message=message
        )
        
        self._versions[document_id].append(version)
        logger.info(f"Created version {version.version} for document {document_id}")
        return version
    
    def get_versions(self, document_id: str) -> List[DocumentVersion]:
        """Get all versions for document."""
        return self._versions.get(document_id, []).copy()
    
    def get_version(self, document_id: str, version: str) -> Optional[DocumentVersion]:
        """Get specific version."""
        versions = self.get_versions(document_id)
        for v in versions:
            if v.version == version:
                return v
        return None
    
    def get_latest_version(self, document_id: str) -> Optional[DocumentVersion]:
        """Get latest version."""
        versions = self.get_versions(document_id)
        return versions[-1] if versions else None
    
    def compare_versions(
        self,
        document_id: str,
        version1: str,
        version2: str
    ) -> Dict[str, Any]:
        """Compare two versions."""
        v1 = self.get_version(document_id, version1)
        v2 = self.get_version(document_id, version2)
        
        if not v1 or not v2:
            return {"error": "Version not found"}
        
        # Simple diff (in production, use proper diff algorithm)
        return {
            "version1": version1,
            "version2": version2,
            "content_diff": len(v2.content) - len(v1.content),
            "created_at_diff": (v2.created_at - v1.created_at).total_seconds()
        }
    
    def get_version_stats(self) -> Dict[str, Any]:
        """Get version statistics."""
        return {
            "total_documents": len(self._versions),
            "total_versions": sum(len(versions) for versions in self._versions.values()),
            "avg_versions_per_document": (
                sum(len(versions) for versions in self._versions.values()) / len(self._versions)
                if self._versions else 0
            )
        }















