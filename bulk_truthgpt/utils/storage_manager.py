"""
Storage Manager
==============

Advanced storage system for documents and files.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
import json
import os
from datetime import datetime
from pathlib import Path
import aiofiles

logger = logging.getLogger(__name__)

class StorageManager:
    """
    Advanced storage manager for documents and files.
    
    Features:
    - File storage
    - Document indexing
    - Search functionality
    - Metadata management
    - Backup and recovery
    """
    
    def __init__(self, storage_path: str = "./storage"):
        self.storage_path = Path(storage_path)
        self.documents_path = self.storage_path / "documents"
        self.metadata_path = self.storage_path / "metadata"
        self.index_path = self.storage_path / "index"
        
    async def initialize(self):
        """Initialize storage manager."""
        logger.info("Initializing Storage Manager...")
        
        try:
            # Create storage directories
            self.storage_path.mkdir(parents=True, exist_ok=True)
            self.documents_path.mkdir(parents=True, exist_ok=True)
            self.metadata_path.mkdir(parents=True, exist_ok=True)
            self.index_path.mkdir(parents=True, exist_ok=True)
            
            logger.info("Storage Manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Storage Manager: {str(e)}")
            raise
    
    async def store_document(self, document_id: str, document_data: Dict[str, Any]) -> bool:
        """Store document in storage."""
        try:
            # Store document content
            content_file = self.documents_path / f"{document_id}.txt"
            async with aiofiles.open(content_file, 'w', encoding='utf-8') as f:
                await f.write(document_data.get('content', ''))
            
            # Store metadata
            metadata_file = self.metadata_path / f"{document_id}.json"
            async with aiofiles.open(metadata_file, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(document_data, default=str, ensure_ascii=False))
            
            # Update index
            await self._update_index(document_id, document_data)
            
            logger.info(f"Stored document {document_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store document {document_id}: {str(e)}")
            return False
    
    async def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get document from storage."""
        try:
            metadata_file = self.metadata_path / f"{document_id}.json"
            
            if not metadata_file.exists():
                return None
            
            async with aiofiles.open(metadata_file, 'r', encoding='utf-8') as f:
                content = await f.read()
                return json.loads(content)
                
        except Exception as e:
            logger.error(f"Failed to get document {document_id}: {str(e)}")
            return None
    
    async def get_task_documents(
        self, 
        task_id: str, 
        limit: int = 100, 
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """Get documents for a task."""
        try:
            # Load index
            index_file = self.index_path / "task_index.json"
            if not index_file.exists():
                return []
            
            async with aiofiles.open(index_file, 'r', encoding='utf-8') as f:
                index_data = json.loads(await f.read())
            
            # Get task documents
            task_documents = index_data.get(task_id, [])
            
            # Apply pagination
            start = offset
            end = offset + limit
            paginated_docs = task_documents[start:end]
            
            # Load document data
            documents = []
            for doc_id in paginated_docs:
                doc_data = await self.get_document(doc_id)
                if doc_data:
                    documents.append(doc_data)
            
            return documents
            
        except Exception as e:
            logger.error(f"Failed to get task documents: {str(e)}")
            return []
    
    async def search_documents(
        self, 
        query: str, 
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Search documents."""
        try:
            # Simple text search implementation
            # In production, this would use a proper search engine like Elasticsearch
            
            documents = []
            search_count = 0
            
            # Search through all metadata files
            for metadata_file in self.metadata_path.glob("*.json"):
                try:
                    async with aiofiles.open(metadata_file, 'r', encoding='utf-8') as f:
                        doc_data = json.loads(await f.read())
                    
                    # Check if document matches query
                    if self._matches_query(doc_data, query, filters):
                        documents.append(doc_data)
                        search_count += 1
                        
                        if search_count >= limit:
                            break
                            
                except Exception as e:
                    logger.warning(f"Failed to read metadata file {metadata_file}: {str(e)}")
                    continue
            
            return documents
            
        except Exception as e:
            logger.error(f"Failed to search documents: {str(e)}")
            return []
    
    def _matches_query(self, doc_data: Dict[str, Any], query: str, filters: Optional[Dict[str, Any]] = None) -> bool:
        """Check if document matches search query."""
        try:
            # Text search
            content = doc_data.get('content', '').lower()
            if query.lower() not in content:
                return False
            
            # Apply filters
            if filters:
                for key, value in filters.items():
                    if key in doc_data.get('metadata', {}):
                        if doc_data['metadata'][key] != value:
                            return False
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to check query match: {str(e)}")
            return False
    
    async def delete_document(self, document_id: str) -> bool:
        """Delete document from storage."""
        try:
            # Delete content file
            content_file = self.documents_path / f"{document_id}.txt"
            if content_file.exists():
                content_file.unlink()
            
            # Delete metadata file
            metadata_file = self.metadata_path / f"{document_id}.json"
            if metadata_file.exists():
                metadata_file.unlink()
            
            # Update index
            await self._remove_from_index(document_id)
            
            logger.info(f"Deleted document {document_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete document {document_id}: {str(e)}")
            return False
    
    async def _update_index(self, document_id: str, document_data: Dict[str, Any]):
        """Update search index."""
        try:
            index_file = self.index_path / "task_index.json"
            
            # Load existing index
            index_data = {}
            if index_file.exists():
                async with aiofiles.open(index_file, 'r', encoding='utf-8') as f:
                    index_data = json.loads(await f.read())
            
            # Update index
            task_id = document_data.get('task_id')
            if task_id:
                if task_id not in index_data:
                    index_data[task_id] = []
                
                if document_id not in index_data[task_id]:
                    index_data[task_id].append(document_id)
            
            # Save index
            async with aiofiles.open(index_file, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(index_data, ensure_ascii=False))
                
        except Exception as e:
            logger.error(f"Failed to update index: {str(e)}")
    
    async def _remove_from_index(self, document_id: str):
        """Remove document from index."""
        try:
            index_file = self.index_path / "task_index.json"
            
            if not index_file.exists():
                return
            
            # Load index
            async with aiofiles.open(index_file, 'r', encoding='utf-8') as f:
                index_data = json.loads(await f.read())
            
            # Remove document from all tasks
            for task_id, doc_ids in index_data.items():
                if document_id in doc_ids:
                    doc_ids.remove(document_id)
            
            # Save updated index
            async with aiofiles.open(index_file, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(index_data, ensure_ascii=False))
                
        except Exception as e:
            logger.error(f"Failed to remove from index: {str(e)}")
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get storage statistics."""
        try:
            # Count documents
            doc_count = len(list(self.documents_path.glob("*.txt")))
            
            # Count metadata files
            meta_count = len(list(self.metadata_path.glob("*.json")))
            
            # Calculate total size
            total_size = 0
            for file_path in self.storage_path.rglob("*"):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
            
            return {
                'total_documents': doc_count,
                'metadata_files': meta_count,
                'total_size_bytes': total_size,
                'storage_path': str(self.storage_path)
            }
            
        except Exception as e:
            logger.error(f"Failed to get storage statistics: {str(e)}")
            return {}
    
    async def backup(self, backup_path: str) -> bool:
        """Create backup of storage."""
        try:
            import shutil
            
            backup_dir = Path(backup_path)
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy storage directory
            shutil.copytree(self.storage_path, backup_dir / "storage")
            
            logger.info(f"Created backup at {backup_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create backup: {str(e)}")
            return False
    
    async def restore(self, backup_path: str) -> bool:
        """Restore from backup."""
        try:
            import shutil
            
            backup_dir = Path(backup_path)
            storage_backup = backup_dir / "storage"
            
            if not storage_backup.exists():
                logger.error(f"Backup not found at {backup_path}")
                return False
            
            # Remove existing storage
            if self.storage_path.exists():
                shutil.rmtree(self.storage_path)
            
            # Restore from backup
            shutil.copytree(storage_backup, self.storage_path)
            
            logger.info(f"Restored from backup {backup_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to restore from backup: {str(e)}")
            return False
    
    async def cleanup(self):
        """Cleanup storage manager."""
        try:
            logger.info("Storage Manager cleaned up successfully")
            
        except Exception as e:
            logger.error(f"Failed to cleanup Storage Manager: {str(e)}")











