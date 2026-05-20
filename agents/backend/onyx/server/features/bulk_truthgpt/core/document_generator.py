"""
Document Generator
=================

Advanced document generation system with TruthGPT integration.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import json
import uuid
from pathlib import Path

from ..models.schemas import GenerationConfig, DocumentStatus
from ..utils.storage_manager import StorageManager
from ..utils.template_engine import TemplateEngine
from ..utils.format_converter import FormatConverter

logger = logging.getLogger(__name__)

class DocumentGenerator:
    """
    Document Generator for TruthGPT-based bulk generation.
    
    Features:
    - Multiple document formats
    - Template-based generation
    - Content optimization
    - Batch processing
    - Quality validation
    """
    
    def __init__(self, truthgpt_engine):
        self.truthgpt_engine = truthgpt_engine
        self.storage_manager = StorageManager()
        self.template_engine = TemplateEngine()
        self.format_converter = FormatConverter()
        self.generation_queue = asyncio.Queue()
        self.active_generations = {}
        
    async def initialize(self):
        """Initialize the document generator."""
        logger.info("Initializing Document Generator...")
        
        try:
            await self.storage_manager.initialize()
            await self.template_engine.initialize()
            await self.format_converter.initialize()
            
            logger.info("Document Generator initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Document Generator: {str(e)}")
            raise
    
    async def generate_document(
        self, 
        query: str, 
        config: GenerationConfig,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate a single document.
        
        Args:
            query: Generation query
            config: Generation configuration
            context: Additional context
            
        Returns:
            Generated document with metadata
        """
        try:
            logger.info(f"Generating document for query: {query}")
            
            # Create generation context
            from ..core.truthgpt_engine import GenerationContext
            generation_context = GenerationContext(
                query=query,
                config=config,
                knowledge_base=context.get("knowledge_base") if context else None,
                previous_documents=context.get("previous_documents") if context else None,
                optimization_hints=context.get("optimization_hints") if context else None
            )
            
            # Generate using TruthGPT engine
            document_data = await self.truthgpt_engine.generate_document(generation_context)
            
            # Apply template if specified
            if config.template:
                document_data = await self._apply_template(document_data, config.template)
            
            # Convert to requested format
            if config.output_format:
                document_data = await self._convert_format(document_data, config.output_format)
            
            # Store document
            document_id = await self._store_document(document_data, config)
            
            # Add document ID to the data
            document_data["document_id"] = document_id
            
            logger.info(f"Document generated successfully: {document_id}")
            return document_data
            
        except Exception as e:
            logger.error(f"Failed to generate document: {str(e)}")
            raise
    
    async def generate_batch(
        self,
        queries: List[str],
        config: GenerationConfig,
        context: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate multiple documents in batch.
        
        Args:
            queries: List of generation queries
            config: Generation configuration
            context: Additional context
            
        Returns:
            List of generated documents
        """
        try:
            logger.info(f"Generating batch of {len(queries)} documents")
            
            # Create tasks for parallel generation
            tasks = []
            for query in queries:
                task = self.generate_document(query, config, context)
                tasks.append(task)
            
            # Execute tasks in parallel
            documents = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter out exceptions and log errors
            successful_documents = []
            for i, result in enumerate(documents):
                if isinstance(result, Exception):
                    logger.error(f"Failed to generate document for query {i}: {str(result)}")
                else:
                    successful_documents.append(result)
            
            logger.info(f"Batch generation completed: {len(successful_documents)}/{len(queries)} successful")
            return successful_documents
            
        except Exception as e:
            logger.error(f"Failed to generate batch: {str(e)}")
            raise
    
    async def _apply_template(self, document_data: Dict[str, Any], template_name: str) -> Dict[str, Any]:
        """Apply template to document."""
        try:
            template = await self.template_engine.get_template(template_name)
            if template:
                document_data["content"] = await self.template_engine.render_template(
                    template,
                    document_data["content"],
                    document_data.get("metadata", {})
                )
                document_data["template_applied"] = template_name
            return document_data
            
        except Exception as e:
            logger.error(f"Failed to apply template: {str(e)}")
            return document_data
    
    async def _convert_format(self, document_data: Dict[str, Any], output_format: str) -> Dict[str, Any]:
        """Convert document to specified format."""
        try:
            converted_content = await self.format_converter.convert(
                document_data["content"],
                output_format,
                document_data.get("metadata", {})
            )
            
            document_data["content"] = converted_content
            document_data["format"] = output_format
            
            return document_data
            
        except Exception as e:
            logger.error(f"Failed to convert format: {str(e)}")
            return document_data
    
    async def _store_document(self, document_data: Dict[str, Any], config: GenerationConfig) -> str:
        """Store document in storage system."""
        try:
            document_id = str(uuid.uuid4())
            
            # Prepare document for storage
            storage_data = {
                "id": document_id,
                "content": document_data["content"],
                "metadata": document_data.get("metadata", {}),
                "analysis": document_data.get("analysis", {}),
                "optimization": document_data.get("optimization", {}),
                "created_at": datetime.utcnow().isoformat(),
                "config": config.dict()
            }
            
            # Store in storage manager
            await self.storage_manager.store_document(document_id, storage_data)
            
            return document_id
            
        except Exception as e:
            logger.error(f"Failed to store document: {str(e)}")
            raise
    
    async def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get document by ID."""
        try:
            return await self.storage_manager.get_document(document_id)
        except Exception as e:
            logger.error(f"Failed to get document {document_id}: {str(e)}")
            return None
    
    async def get_task_documents(
        self, 
        task_id: str, 
        limit: int = 100, 
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """Get documents generated by a specific task."""
        try:
            return await self.storage_manager.get_task_documents(task_id, limit, offset)
        except Exception as e:
            logger.error(f"Failed to get task documents: {str(e)}")
            return []
    
    async def delete_document(self, document_id: str) -> bool:
        """Delete document by ID."""
        try:
            return await self.storage_manager.delete_document(document_id)
        except Exception as e:
            logger.error(f"Failed to delete document {document_id}: {str(e)}")
            return False
    
    async def search_documents(
        self, 
        query: str, 
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Search documents by content or metadata."""
        try:
            return await self.storage_manager.search_documents(query, filters, limit)
        except Exception as e:
            logger.error(f"Failed to search documents: {str(e)}")
            return []
    
    async def get_document_statistics(self) -> Dict[str, Any]:
        """Get document generation statistics."""
        try:
            return await self.storage_manager.get_statistics()
        except Exception as e:
            logger.error(f"Failed to get document statistics: {str(e)}")
            return {}
    
    async def cleanup(self):
        """Cleanup resources."""
        try:
            await self.storage_manager.cleanup()
            await self.template_engine.cleanup()
            await self.format_converter.cleanup()
            logger.info("Document Generator cleaned up successfully")
        except Exception as e:
            logger.error(f"Failed to cleanup Document Generator: {str(e)}")











