"""
Content Lifecycle Engine

This module provides content lifecycle management capabilities including
content creation, versioning, search, and lifecycle operations.
"""

import hashlib
import time
from typing import Dict, List, Any, Optional
import logging

from ..core.base import BaseEngine
from ..core.config import SystemConfig
from ..core.exceptions import StorageError, ValidationError

logger = logging.getLogger(__name__)


class ContentLifecycleEngine(BaseEngine[Dict[str, Any]]):
    """Advanced content lifecycle management engine"""
    
    def __init__(self, config: Optional[SystemConfig] = None):
        super().__init__(config)
        self._capabilities = [
            "content_creation",
            "content_versioning",
            "content_search",
            "content_update",
            "content_deletion",
            "lifecycle_management"
        ]
        self._content_store = {}  # In-memory store for demo purposes
        self._version_store = {}  # Version history store
    
    async def _process(self, data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Process content lifecycle operations"""
        try:
            operation = kwargs.get("operation", "create")
            
            if operation == "create":
                content = data.get("content", "")
                content_type = data.get("content_type", "text")
                metadata = data.get("metadata", {})
                return await self.create_content(content, content_type, metadata, **kwargs)
            elif operation == "update":
                content_id = data.get("content_id", "")
                content = data.get("content", "")
                metadata = data.get("metadata", {})
                return await self.update_content(content_id, content, metadata, **kwargs)
            elif operation == "search":
                query = data.get("query", "")
                filters = data.get("filters", {})
                limit = data.get("limit", 10)
                offset = data.get("offset", 0)
                return await self.search_content(query, filters, limit, offset, **kwargs)
            else:
                raise ValidationError(f"Unknown operation: {operation}")
                
        except Exception as e:
            logger.error(f"Content lifecycle processing failed: {e}")
            raise StorageError(f"Content lifecycle processing failed: {str(e)}", operation="processing")
    
    async def create_content(self, content: str, content_type: str = "text", 
                           metadata: Dict[str, Any] = None, **kwargs) -> Dict[str, Any]:
        """Create new content entry"""
        try:
            if not content:
                raise ValidationError("Content cannot be empty")
            
            # Generate content ID and hash
            content_hash = hashlib.md5(content.encode()).hexdigest()
            content_id = f"content_{content_hash}_{int(time.time())}"
            
            # Create content entry
            content_entry = {
                "content_id": content_id,
                "content": content,
                "content_type": content_type,
                "content_hash": content_hash,
                "metadata": metadata or {},
                "created_at": time.time(),
                "updated_at": time.time(),
                "version": 1,
                "status": "active"
            }
            
            # Store content
            self._content_store[content_id] = content_entry
            
            # Create initial version
            await self.create_version(content_id, "Initial version", metadata)
            
            return {
                "content_id": content_id,
                "content_hash": content_hash,
                "version": 1,
                "status": "created",
                "metadata": content_entry
            }
            
        except Exception as e:
            logger.error(f"Content creation failed: {e}")
            raise StorageError(f"Content creation failed: {str(e)}", operation="create")
    
    async def update_content(self, content_id: str, content: str, 
                           metadata: Dict[str, Any] = None, **kwargs) -> Dict[str, Any]:
        """Update existing content"""
        try:
            if content_id not in self._content_store:
                raise ValidationError(f"Content {content_id} not found")
            
            # Get current content
            current_content = self._content_store[content_id]
            
            # Generate new hash
            new_hash = hashlib.md5(content.encode()).hexdigest()
            
            # Update content entry
            current_content.update({
                "content": content,
                "content_hash": new_hash,
                "metadata": metadata or current_content.get("metadata", {}),
                "updated_at": time.time(),
                "version": current_content.get("version", 1) + 1
            })
            
            # Store updated content
            self._content_store[content_id] = current_content
            
            # Create new version
            version_notes = kwargs.get("version_notes", "Content updated")
            await self.create_version(content_id, version_notes, metadata)
            
            return {
                "content_id": content_id,
                "content_hash": new_hash,
                "version": current_content["version"],
                "status": "updated",
                "metadata": current_content
            }
            
        except Exception as e:
            logger.error(f"Content update failed: {e}")
            raise StorageError(f"Content update failed: {str(e)}", operation="update")
    
    async def delete_content(self, content_id: str) -> Dict[str, Any]:
        """Delete content (soft delete)"""
        try:
            if content_id not in self._content_store:
                raise ValidationError(f"Content {content_id} not found")
            
            # Soft delete - mark as deleted
            content = self._content_store[content_id]
            content.update({
                "status": "deleted",
                "deleted_at": time.time()
            })
            
            self._content_store[content_id] = content
            
            return {
                "content_id": content_id,
                "status": "deleted",
                "deleted_at": content["deleted_at"]
            }
            
        except Exception as e:
            logger.error(f"Content deletion failed: {e}")
            raise StorageError(f"Content deletion failed: {str(e)}", operation="delete")
    
    async def search_content(self, query: str, filters: Dict[str, Any] = None, 
                           limit: int = 10, offset: int = 0, **kwargs) -> List[Dict[str, Any]]:
        """Search content"""
        try:
            if not query:
                raise ValidationError("Search query cannot be empty")
            
            results = []
            query_lower = query.lower()
            
            # Search through content store
            for content_id, content_entry in self._content_store.items():
                if content_entry.get("status") != "active":
                    continue
                
                # Simple text search
                content_text = content_entry.get("content", "").lower()
                if query_lower in content_text:
                    # Apply filters
                    if self._matches_filters(content_entry, filters or {}):
                        results.append({
                            "content_id": content_id,
                            "content_preview": content_entry.get("content", "")[:200] + "...",
                            "content_type": content_entry.get("content_type", "text"),
                            "created_at": content_entry.get("created_at"),
                            "updated_at": content_entry.get("updated_at"),
                            "version": content_entry.get("version", 1),
                            "metadata": content_entry.get("metadata", {})
                        })
            
            # Apply pagination
            start_idx = offset
            end_idx = offset + limit
            paginated_results = results[start_idx:end_idx]
            
            return paginated_results
            
        except Exception as e:
            logger.error(f"Content search failed: {e}")
            raise StorageError(f"Content search failed: {str(e)}", operation="search")
    
    async def create_version(self, content_id: str, version_notes: str = "", 
                           metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Create a new version of content"""
        try:
            if content_id not in self._content_store:
                raise ValidationError(f"Content {content_id} not found")
            
            content = self._content_store[content_id]
            version_number = content.get("version", 1)
            
            # Create version entry
            version_id = f"version_{content_id}_{version_number}_{int(time.time())}"
            version_entry = {
                "version_id": version_id,
                "content_id": content_id,
                "version_number": version_number,
                "content": content.get("content", ""),
                "content_hash": content.get("content_hash", ""),
                "version_notes": version_notes,
                "metadata": metadata or {},
                "created_at": time.time(),
                "created_by": "system"
            }
            
            # Store version
            if content_id not in self._version_store:
                self._version_store[content_id] = []
            self._version_store[content_id].append(version_entry)
            
            return {
                "version_id": version_id,
                "content_id": content_id,
                "version_number": version_number,
                "status": "created"
            }
            
        except Exception as e:
            logger.error(f"Version creation failed: {e}")
            raise StorageError(f"Version creation failed: {str(e)}", operation="version")
    
    def _matches_filters(self, content_entry: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Check if content matches the given filters"""
        for filter_key, filter_value in filters.items():
            if filter_key == "content_type":
                if content_entry.get("content_type") != filter_value:
                    return False
            elif filter_key == "created_after":
                if content_entry.get("created_at", 0) < filter_value:
                    return False
            elif filter_key == "created_before":
                if content_entry.get("created_at", 0) > filter_value:
                    return False
            elif filter_key == "status":
                if content_entry.get("status") != filter_value:
                    return False
            # Add more filter types as needed
        
        return True
    
    def get_capabilities(self) -> List[str]:
        """Get list of engine capabilities"""
        return self._capabilities
    
    async def get_content(self, content_id: str) -> Optional[Dict[str, Any]]:
        """Get content by ID"""
        return self._content_store.get(content_id)
    
    async def get_content_versions(self, content_id: str) -> List[Dict[str, Any]]:
        """Get all versions of content"""
        return self._version_store.get(content_id, [])
    
    async def get_content_count(self) -> int:
        """Get total number of content items"""
        return len([c for c in self._content_store.values() if c.get("status") == "active"])





















