"""
Document Service
================

Advanced document service with validation, processing, and quality checks.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path

from ..utils.robust_helpers import robust_retry, validate_input
from ..utils.intelligent_cache import intelligent_cache
from ..utils.compression_manager import compression_manager, CompressionAlgorithm
from ..utils.event_bus import event_bus
from .storage_service import StorageService

logger = logging.getLogger(__name__)

class DocumentService:
    """Advanced document service."""
    
    def __init__(self, storage_service: StorageService):
        self.storage_service = storage_service
        self.quality_threshold = 0.7
        self.processing_queue = asyncio.Queue()
    
    async def process_document(
        self,
        document: Dict[str, Any],
        task_id: str,
        compress: bool = False
    ) -> Dict[str, Any]:
        """Process document with validation and quality checks."""
        # Validate document
        is_valid, error = validate_input(
            document,
            required_fields=["id", "content"],
            field_validators={
                "id": lambda x: isinstance(x, str) and len(x) > 0,
                "content": lambda x: isinstance(x, str) and len(x) > 10
            }
        )
        
        if not is_valid:
            raise ValueError(f"Invalid document: {error}")
        
        # Check quality
        quality_score = self._calculate_quality(document)
        document["quality_score"] = quality_score
        
        if quality_score < self.quality_threshold:
            logger.warning(f"Document {document['id']} has low quality score: {quality_score}")
        
        # Compress if requested
        if compress:
            content_bytes = document["content"].encode()
            compressed, stats = compression_manager.compress(
                content_bytes,
                algorithm=CompressionAlgorithm.GZIP
            )
            document["compressed"] = {
                "data": compressed.hex(),
                "stats": stats
            }
        
        # Save to storage
        file_path = await self.storage_service.save_document(
            document=document,
            task_id=task_id,
            format="json"
        )
        
        document["file_path"] = file_path
        
        # Publish event
        await event_bus.publish("document.generated", {
            "document_id": document["id"],
            "task_id": task_id,
            "quality_score": quality_score,
            "file_path": file_path
        })
        
        return document
    
    def _calculate_quality(self, document: Dict[str, Any]) -> float:
        """Calculate document quality score."""
        content = document.get("content", "")
        if not content:
            return 0.0
        
        score = 0.0
        
        # Length check
        if len(content) > 100:
            score += 0.2
        elif len(content) > 50:
            score += 0.1
        
        # Structure check
        if '\n' in content:
            score += 0.2
        if '.' in content:
            score += 0.2
        
        # Metadata check
        if document.get("metadata"):
            score += 0.2
        
        # Quality score from generation
        if "quality_score" in document:
            score += document["quality_score"] * 0.2
        
        return min(1.0, score)
    
    async def process_batch(
        self,
        documents: List[Dict[str, Any]],
        task_id: str
    ) -> List[Dict[str, Any]]:
        """Process batch of documents."""
        results = []
        
        for document in documents:
            try:
                processed = await self.process_document(document, task_id)
                results.append(processed)
            except Exception as e:
                logger.error(f"Failed to process document {document.get('id')}: {e}")
                results.append({
                    "id": document.get("id"),
                    "error": str(e),
                    "status": "failed"
                })
        
        return results
    
    async def get_document_quality_report(self, document_id: str) -> Dict[str, Any]:
        """Get quality report for document."""
        document = await self.storage_service.get_document(document_id)
        
        if not document:
            return {"error": "Document not found"}
        
        return {
            "document_id": document_id,
            "quality_score": document.get("quality_score", 0.0),
            "length": len(document.get("content", "")),
            "has_metadata": bool(document.get("metadata")),
            "timestamp": document.get("timestamp")
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get document service statistics."""
        return {
            "queue_size": self.processing_queue.qsize(),
            "quality_threshold": self.quality_threshold
        }
































