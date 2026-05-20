"""
Storage Service
===============

Robust document storage service with persistence, error handling, and recovery.
"""

import asyncio
import logging
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import aiofiles
from aiofiles.os import makedirs
import hashlib
from ..utils.robust_helpers import robust_retry, validate_input, safe_json_dumps, safe_json_loads, generate_id

logger = logging.getLogger(__name__)

class StorageService:
    """
    Robust storage service for documents with:
    - Async file operations
    - Error recovery
    - Metadata management
    - Backup support
    """
    
    def __init__(self, storage_path: str = "./storage", backup_path: str = "./storage/backups"):
        self.storage_path = Path(storage_path)
        self.backup_path = Path(backup_path)
        self.metadata_file = self.storage_path / "metadata.json"
        self.metadata = {}
        self.is_initialized = False
        
    async def initialize(self):
        """Initialize storage service."""
        try:
            # Create directories
            await makedirs(self.storage_path, exist_ok=True)
            await makedirs(self.backup_path, exist_ok=True)
            
            # Load existing metadata
            await self._load_metadata()
            
            self.is_initialized = True
            logger.info(f"Storage service initialized at {self.storage_path}")
            
        except Exception as e:
            logger.error(f"Failed to initialize storage service: {e}")
            raise
    
    async def _load_metadata(self):
        """Load metadata from file."""
        try:
            if self.metadata_file.exists():
                async with aiofiles.open(self.metadata_file, 'r') as f:
                    content = await f.read()
                    self.metadata = json.loads(content)
                    logger.info(f"Loaded {len(self.metadata)} document metadata entries")
            else:
                self.metadata = {}
        except Exception as e:
            logger.warning(f"Failed to load metadata: {e}, starting with empty metadata")
            self.metadata = {}
    
    async def _save_metadata(self):
        """Save metadata to file."""
        try:
            async with aiofiles.open(self.metadata_file, 'w') as f:
                await f.write(json.dumps(self.metadata, indent=2))
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")
    
    @robust_retry(max_attempts=3, base_delay=1.0, max_delay=10.0)
    async def save_document(
        self, 
        document: Dict[str, Any], 
        task_id: str,
        format: str = "json"
    ) -> str:
        """
        Save document with robust error handling.
        
        Args:
            document: Document data
            task_id: Task ID
            format: Format (json, markdown, txt)
            
        Returns:
            File path where document was saved
        """
        if not self.is_initialized:
            await self.initialize()
        
        # Validate input
        is_valid, error_msg = validate_input(
            document,
            required_fields=["id", "content"],
            field_validators={
                "id": lambda x: isinstance(x, str) and len(x) > 0,
                "content": lambda x: x is not None
            }
        )
        if not is_valid:
            raise ValueError(f"Invalid document data: {error_msg}")
        
        try:
            # Generate document ID
            doc_id = document.get("id", f"doc_{hashlib.md5(task_id.encode()).hexdigest()[:8]}")
            
            # Create task directory
            task_dir = self.storage_path / task_id
            await makedirs(task_dir, exist_ok=True)
            
            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{doc_id}_{timestamp}.{format}"
            file_path = task_dir / filename
            
            # Save document based on format
            if format == "json":
                async with aiofiles.open(file_path, 'w', encoding='utf-8') as f:
                    await f.write(safe_json_dumps(document))
            elif format == "markdown":
                content = document.get("content", "")
                async with aiofiles.open(file_path, 'w', encoding='utf-8') as f:
                    await f.write(content)
            else:
                content = document.get("content", "")
                async with aiofiles.open(file_path, 'w', encoding='utf-8') as f:
                    await f.write(content)
            
            # Update metadata
            self.metadata[doc_id] = {
                "task_id": task_id,
                "file_path": str(file_path),
                "format": format,
                "created_at": document.get("timestamp", datetime.now().isoformat()),
                "quality_score": document.get("quality_score", 0.0),
                "model_used": document.get("model_used", "unknown"),
                "size_bytes": file_path.stat().st_size if file_path.exists() else 0
            }
            
            # Save metadata
            await self._save_metadata()
            
            logger.info(f"Document saved: {file_path}")
            return str(file_path)
            
        except Exception as e:
            logger.error(f"Failed to save document: {e}")
            # Try backup location
            try:
                backup_file = self.backup_path / f"{task_id}_{datetime.now().timestamp()}.{format}"
                async with aiofiles.open(backup_file, 'w') as f:
                    await f.write(json.dumps(document))
                logger.info(f"Document saved to backup: {backup_file}")
                return str(backup_file)
            except Exception as backup_error:
                logger.error(f"Backup save also failed: {backup_error}")
                raise
    
    async def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve document by ID."""
        try:
            if doc_id not in self.metadata:
                return None
            
            file_path = Path(self.metadata[doc_id]["file_path"])
            if not file_path.exists():
                return None
            
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                content = await f.read()
                
            if file_path.suffix == ".json":
                return safe_json_loads(content, default={})
            else:
                return {
                    "id": doc_id,
                    "content": content,
                    "metadata": self.metadata[doc_id]
                }
                
        except Exception as e:
            logger.error(f"Failed to get document {doc_id}: {e}")
            return None
    
    async def get_task_documents(self, task_id: str, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """Get all documents for a task."""
        try:
            documents = []
            task_docs = [
                (doc_id, meta) 
                for doc_id, meta in self.metadata.items() 
                if meta.get("task_id") == task_id
            ]
            
            # Sort by created_at
            task_docs.sort(key=lambda x: x[1].get("created_at", ""), reverse=True)
            
            # Apply pagination
            paginated_docs = task_docs[offset:offset + limit]
            
            for doc_id, meta in paginated_docs:
                doc = await self.get_document(doc_id)
                if doc:
                    documents.append(doc)
            
            return documents
            
        except Exception as e:
            logger.error(f"Failed to get task documents: {e}")
            return []
    
    async def delete_document(self, doc_id: str) -> bool:
        """Delete document."""
        try:
            if doc_id not in self.metadata:
                return False
            
            file_path = Path(self.metadata[doc_id]["file_path"])
            if file_path.exists():
                file_path.unlink()
            
            del self.metadata[doc_id]
            await self._save_metadata()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete document {doc_id}: {e}")
            return False
    
    async def cleanup_old_documents(self, days: int = 30):
        """Cleanup documents older than specified days."""
        try:
            cutoff_date = datetime.now().timestamp() - (days * 24 * 60 * 60)
            deleted_count = 0
            
            for doc_id, meta in list(self.metadata.items()):
                created_at = meta.get("created_at", "")
                try:
                    doc_timestamp = datetime.fromisoformat(created_at).timestamp()
                    if doc_timestamp < cutoff_date:
                        await self.delete_document(doc_id)
                        deleted_count += 1
                except:
                    continue
            
            logger.info(f"Cleaned up {deleted_count} old documents")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Failed to cleanup old documents: {e}")
            return 0
    
    async def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        try:
            total_docs = len(self.metadata)
            total_size = 0
            
            for meta in self.metadata.values():
                file_path = Path(meta.get("file_path", ""))
                if file_path.exists():
                    total_size += file_path.stat().st_size
            
            return {
                "total_documents": total_docs,
                "total_size_bytes": total_size,
                "total_size_mb": round(total_size / (1024 * 1024), 2),
                "storage_path": str(self.storage_path),
                "backup_path": str(self.backup_path)
            }
            
        except Exception as e:
            logger.error(f"Failed to get storage stats: {e}")
            return {}
    
    async def cleanup(self):
        """Cleanup resources."""
        try:
            await self._save_metadata()
            logger.info("Storage service cleaned up")
        except Exception as e:
            logger.error(f"Failed to cleanup storage service: {e}")

