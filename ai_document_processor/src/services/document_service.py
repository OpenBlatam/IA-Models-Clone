"""
Document Service - Document Management
====================================

Service for document management and processing operations.
"""

import asyncio
import logging
from typing import Optional, List, Dict, Any, Union
from datetime import datetime
from pathlib import Path
import uuid

from ..core.exceptions import ProcessingError, ValidationError, FileError
from ..core.config import get_config
from ..models.document import Document, DocumentType, DocumentStatus, DocumentCollection
from ..models.processing import ProcessingResult, ProcessingStatus, ProcessingConfig
from .file_service import FileService
from .validation_service import ValidationService
from .ai_service import AIService
from .transform_service import TransformService

logger = logging.getLogger(__name__)


class DocumentService:
    """Service for document management and processing."""
    
    def __init__(
        self,
        file_service: Optional[FileService] = None,
        validation_service: Optional[ValidationService] = None,
        ai_service: Optional[AIService] = None,
        transform_service: Optional[TransformService] = None
    ):
        """
        Initialize document service.
        
        Args:
            file_service: File service instance
            validation_service: Validation service instance
            ai_service: AI service instance
            transform_service: Transform service instance
        """
        self.file_service = file_service or FileService()
        self.validation_service = validation_service or ValidationService()
        self.ai_service = ai_service or AIService()
        self.transform_service = transform_service or TransformService()
        
        # Get configuration
        self.config = get_config('processing')
        
        # Document storage (in production, this would be a database)
        self._documents: Dict[str, Document] = {}
        self._processing_results: Dict[str, ProcessingResult] = {}
    
    async def create_document(
        self,
        file_path: Union[str, Path],
        filename: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Document:
        """
        Create a new document from file.
        
        Args:
            file_path: Path to the document file
            filename: Optional filename override
            metadata: Optional document metadata
            
        Returns:
            Created document instance
            
        Raises:
            FileError: If file cannot be read
            ValidationError: If file validation fails
        """
        try:
            file_path = Path(file_path)
            
            # Validate file exists
            if not file_path.exists():
                raise FileError(f"File not found: {file_path}")
            
            # Get file info
            file_stat = file_path.stat()
            file_size = file_stat.st_size
            
            # Check file size
            max_size_bytes = self.config.max_file_size_mb * 1024 * 1024
            if file_size > max_size_bytes:
                raise ValidationError(f"File too large: {file_size} bytes (max: {max_size_bytes})")
            
            # Determine document type
            document_type = self._determine_document_type(file_path)
            
            # Create document
            document = Document(
                id=str(uuid.uuid4()),
                filename=filename or file_path.name,
                file_path=str(file_path),
                document_type=document_type,
                metadata={
                    'file_size': file_size,
                    'created_at': datetime.fromtimestamp(file_stat.st_ctime),
                    'modified_at': datetime.fromtimestamp(file_stat.st_mtime),
                    **(metadata or {})
                }
            )
            
            # Store document
            self._documents[document.id] = document
            
            logger.info(f"Created document: {document.id} ({document.filename})")
            return document
            
        except Exception as e:
            logger.error(f"Failed to create document from {file_path}: {e}")
            raise
    
    async def get_document(self, document_id: str) -> Optional[Document]:
        """
        Get document by ID.
        
        Args:
            document_id: Document ID
            
        Returns:
            Document instance or None if not found
        """
        return self._documents.get(document_id)
    
    async def list_documents(
        self,
        document_type: Optional[DocumentType] = None,
        status: Optional[DocumentStatus] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Document]:
        """
        List documents with optional filtering.
        
        Args:
            document_type: Filter by document type
            status: Filter by document status
            limit: Maximum number of documents to return
            offset: Number of documents to skip
            
        Returns:
            List of documents
        """
        documents = list(self._documents.values())
        
        # Apply filters
        if document_type:
            documents = [d for d in documents if d.document_type == document_type]
        
        if status:
            documents = [d for d in documents if d.status == status]
        
        # Apply pagination
        return documents[offset:offset + limit]
    
    async def delete_document(self, document_id: str) -> bool:
        """
        Delete document.
        
        Args:
            document_id: Document ID
            
        Returns:
            True if deleted, False if not found
        """
        if document_id in self._documents:
            del self._documents[document_id]
            logger.info(f"Deleted document: {document_id}")
            return True
        return False
    
    async def process_document(
        self,
        document_id: str,
        processing_config: Optional[ProcessingConfig] = None
    ) -> ProcessingResult:
        """
        Process a document.
        
        Args:
            document_id: Document ID
            processing_config: Optional processing configuration
            
        Returns:
            Processing result
            
        Raises:
            ProcessingError: If processing fails
        """
        # Get document
        document = await self.get_document(document_id)
        if not document:
            raise ProcessingError(f"Document not found: {document_id}")
        
        # Create processing result
        result = ProcessingResult(
            id=str(uuid.uuid4()),
            document_id=document_id,
            status=ProcessingStatus.IN_PROGRESS
        )
        
        try:
            # Update document status
            document.update_status(DocumentStatus.PROCESSING)
            
            # Store processing result
            self._processing_results[result.id] = result
            
            # Process document
            await self._process_document_stages(document, result, processing_config)
            
            # Mark as completed
            result.complete()
            document.update_status(DocumentStatus.COMPLETED)
            
            logger.info(f"Document processed successfully: {document_id}")
            return result
            
        except Exception as e:
            # Mark as failed
            error_msg = str(e)
            result.fail(error_msg)
            document.update_status(DocumentStatus.FAILED, error_msg)
            
            logger.error(f"Document processing failed: {document_id} - {error_msg}")
            raise ProcessingError(f"Processing failed: {error_msg}")
    
    async def _process_document_stages(
        self,
        document: Document,
        result: ProcessingResult,
        processing_config: Optional[ProcessingConfig]
    ):
        """Process document through all stages."""
        config = processing_config or ProcessingConfig()
        
        # Stage 1: Validation
        validation_stage = result.get_stage(ProcessingStage.VALIDATION)
        if not validation_stage:
            validation_stage = ProcessingStageInfo(stage=ProcessingStage.VALIDATION, status=ProcessingStatus.PENDING)
            result.add_stage(validation_stage)
        
        validation_stage.start()
        try:
            await self.validation_service.validate_document(document, config)
            validation_stage.complete()
        except Exception as e:
            validation_stage.fail(str(e))
            raise
        
        # Stage 2: Text Extraction
        extraction_stage = ProcessingStageInfo(stage=ProcessingStage.EXTRACTION, status=ProcessingStatus.PENDING)
        result.add_stage(extraction_stage)
        
        extraction_stage.start()
        try:
            extracted_text = await self.file_service.extract_text(document)
            result.extracted_text = extracted_text
            extraction_stage.complete()
        except Exception as e:
            extraction_stage.fail(str(e))
            raise
        
        # Stage 3: AI Classification (if enabled)
        if config.enable_ai_classification:
            classification_stage = ProcessingStageInfo(stage=ProcessingStage.CLASSIFICATION, status=ProcessingStatus.PENDING)
            result.add_stage(classification_stage)
            
            classification_stage.start()
            try:
                classified_type = await self.ai_service.classify_document(document, extracted_text)
                result.classified_type = classified_type
                classification_stage.complete()
            except Exception as e:
                classification_stage.fail(str(e))
                result.add_warning(f"Classification failed: {e}")
        
        # Stage 4: Transformation (if enabled)
        if config.enable_ai_transformation:
            transformation_stage = ProcessingStageInfo(stage=ProcessingStage.TRANSFORMATION, status=ProcessingStatus.PENDING)
            result.add_stage(transformation_stage)
            
            transformation_stage.start()
            try:
                transformed_content = await self.transform_service.transform_document(
                    document, extracted_text, result.classified_type
                )
                result.transformed_content = transformed_content
                transformation_stage.complete()
            except Exception as e:
                transformation_stage.fail(str(e))
                result.add_warning(f"Transformation failed: {e}")
    
    async def process_batch(
        self,
        document_ids: List[str],
        processing_config: Optional[ProcessingConfig] = None
    ) -> BatchProcessingResult:
        """
        Process multiple documents in batch.
        
        Args:
            document_ids: List of document IDs
            processing_config: Optional processing configuration
            
        Returns:
            Batch processing result
        """
        batch_id = str(uuid.uuid4())
        batch_result = BatchProcessingResult(
            id=batch_id,
            total_documents=len(document_ids),
            status=ProcessingStatus.IN_PROGRESS
        )
        
        try:
            # Process documents in parallel
            tasks = []
            for doc_id in document_ids:
                task = self.process_document(doc_id, processing_config)
                tasks.append(task)
            
            # Wait for all tasks to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    # Create failed result
                    failed_result = ProcessingResult(
                        id=str(uuid.uuid4()),
                        document_id=document_ids[i],
                        status=ProcessingStatus.FAILED,
                        error_message=str(result)
                    )
                    batch_result.add_result(failed_result)
                else:
                    batch_result.add_result(result)
            
            # Mark batch as completed
            batch_result.complete()
            
            logger.info(f"Batch processing completed: {batch_id} ({batch_result.get_success_rate():.1f}% success)")
            return batch_result
            
        except Exception as e:
            batch_result.fail(str(e))
            logger.error(f"Batch processing failed: {batch_id} - {e}")
            raise ProcessingError(f"Batch processing failed: {e}")
    
    async def get_processing_result(self, result_id: str) -> Optional[ProcessingResult]:
        """
        Get processing result by ID.
        
        Args:
            result_id: Processing result ID
            
        Returns:
            Processing result or None if not found
        """
        return self._processing_results.get(result_id)
    
    async def get_document_processing_results(self, document_id: str) -> List[ProcessingResult]:
        """
        Get all processing results for a document.
        
        Args:
            document_id: Document ID
            
        Returns:
            List of processing results
        """
        return [r for r in self._processing_results.values() if r.document_id == document_id]
    
    def _determine_document_type(self, file_path: Path) -> DocumentType:
        """Determine document type from file extension."""
        extension = file_path.suffix.lower()
        
        type_mapping = {
            '.md': DocumentType.MARKDOWN,
            '.pdf': DocumentType.PDF,
            '.docx': DocumentType.WORD,
            '.doc': DocumentType.WORD,
            '.txt': DocumentType.TEXT,
            '.html': DocumentType.HTML,
            '.htm': DocumentType.HTML,
            '.xml': DocumentType.XML,
        }
        
        return type_mapping.get(extension, DocumentType.UNKNOWN)
    
    async def get_document_collection(
        self,
        document_type: Optional[DocumentType] = None,
        status: Optional[DocumentStatus] = None
    ) -> DocumentCollection:
        """
        Get document collection with optional filtering.
        
        Args:
            document_type: Filter by document type
            status: Filter by document status
            
        Returns:
            Document collection
        """
        documents = await self.list_documents(document_type, status)
        
        collection = DocumentCollection(
            documents=documents,
            total_count=len(documents),
            metadata={
                'filtered_by_type': document_type,
                'filtered_by_status': status,
                'created_at': datetime.utcnow()
            }
        )
        
        return collection

















