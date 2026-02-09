"""
Services for AI Document Processor
"""

import asyncio
import logging
import os
import shutil
import tempfile
import uuid
from typing import List, Dict, Any, Optional, BinaryIO
from datetime import datetime
import json

from fastapi import UploadFile, HTTPException
from sqlalchemy import create_engine, Column, String, DateTime, Text, Float, Integer, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import redis
import pandas as pd

from config import settings
from models import (
    DocumentUpload, BatchDocumentUpload, DocumentSearchQuery,
    DocumentComparisonRequest, DocumentProcessingResult,
    BatchProcessingResult, DocumentComparisonResult,
    ProcessingStatus, AnalysisType, DocumentMetadata
)
from document_processor import document_processor

logger = logging.getLogger(__name__)

# Database setup
Base = declarative_base()

class DocumentRecord(Base):
    """Database model for document records"""
    __tablename__ = "documents"
    
    id = Column(String, primary_key=True)
    filename = Column(String, nullable=False)
    content_type = Column(String, nullable=False)
    file_size = Column(Integer, nullable=False)
    language = Column(String, nullable=False)
    status = Column(String, nullable=False)
    metadata = Column(Text)
    results = Column(Text)
    created_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime, nullable=False)
    processing_time = Column(Float)
    error_message = Column(Text)

class BatchRecord(Base):
    """Database model for batch records"""
    __tablename__ = "batches"
    
    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    status = Column(String, nullable=False)
    total_documents = Column(Integer, nullable=False)
    processed_documents = Column(Integer, default=0)
    failed_documents = Column(Integer, default=0)
    results = Column(Text)
    created_at = Column(DateTime, nullable=False)
    completed_at = Column(DateTime)
    processing_time = Column(Float)

# Initialize database
if settings.database_url:
    engine = create_engine(settings.database_url)
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
else:
    engine = None
    SessionLocal = None

# Initialize Redis
redis_client = None
if settings.redis_url:
    try:
        redis_client = redis.from_url(settings.redis_url)
        redis_client.ping()
    except Exception as e:
        logger.warning(f"Redis connection failed: {e}")
        redis_client = None


class DocumentService:
    """Service for document processing operations"""
    
    def __init__(self):
        self.upload_path = settings.upload_path
        self.processed_path = settings.processed_path
        self.temp_path = settings.temp_path
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Ensure required directories exist"""
        for path in [self.upload_path, self.processed_path, self.temp_path]:
            os.makedirs(path, exist_ok=True)
    
    async def upload_document(self, file: UploadFile, analysis_types: List[AnalysisType],
                            metadata: Dict[str, Any] = None) -> str:
        """Upload and process a single document"""
        try:
            # Validate file
            if not self._validate_file(file):
                raise HTTPException(status_code=400, detail="Invalid file format or size")
            
            # Generate document ID
            document_id = str(uuid.uuid4())
            
            # Save file
            file_path = os.path.join(self.upload_path, f"{document_id}_{file.filename}")
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            # Process document
            result = await document_processor.process_document(
                file_path, analysis_types, metadata
            )
            result.document_id = document_id
            
            # Save to database
            await self._save_document_record(result)
            
            # Cache result
            await self._cache_result(document_id, result)
            
            # Move to processed directory
            processed_path = os.path.join(self.processed_path, f"{document_id}_{file.filename}")
            shutil.move(file_path, processed_path)
            
            return document_id
            
        except Exception as e:
            logger.error(f"Error uploading document: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def upload_batch_documents(self, files: List[UploadFile], 
                                   batch_upload: BatchDocumentUpload) -> str:
        """Upload and process multiple documents in batch"""
        try:
            batch_id = str(uuid.uuid4())
            
            # Create batch record
            batch_record = BatchRecord(
                id=batch_id,
                name=batch_upload.batch_name,
                status=ProcessingStatus.PROCESSING,
                total_documents=len(files),
                processed_documents=0,
                failed_documents=0,
                created_at=datetime.now()
            )
            
            if SessionLocal:
                db = SessionLocal()
                try:
                    db.add(batch_record)
                    db.commit()
                finally:
                    db.close()
            
            # Process documents asynchronously
            asyncio.create_task(self._process_batch_async(
                batch_id, files, batch_upload.analysis_types
            ))
            
            return batch_id
            
        except Exception as e:
            logger.error(f"Error uploading batch documents: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def _process_batch_async(self, batch_id: str, files: List[UploadFile],
                                 analysis_types: List[AnalysisType]):
        """Process batch documents asynchronously"""
        results = []
        start_time = datetime.now()
        
        try:
            for file in files:
                try:
                    # Save file temporarily
                    temp_path = os.path.join(self.temp_path, f"{batch_id}_{file.filename}")
                    with open(temp_path, "wb") as buffer:
                        shutil.copyfileobj(file.file, buffer)
                    
                    # Process document
                    result = await document_processor.process_document(
                        temp_path, analysis_types
                    )
                    results.append(result)
                    
                    # Update batch progress
                    await self._update_batch_progress(batch_id, success=True)
                    
                    # Clean up temp file
                    os.remove(temp_path)
                    
                except Exception as e:
                    logger.error(f"Error processing document {file.filename}: {e}")
                    await self._update_batch_progress(batch_id, success=False)
            
            # Create batch result
            batch_result = BatchProcessingResult(
                batch_id=batch_id,
                batch_name="",  # Will be updated from database
                status=ProcessingStatus.COMPLETED,
                total_documents=len(files),
                processed_documents=len([r for r in results if r.status == ProcessingStatus.COMPLETED]),
                failed_documents=len([r for r in results if r.status == ProcessingStatus.FAILED]),
                results=results,
                processing_time=(datetime.now() - start_time).total_seconds(),
                created_at=start_time,
                completed_at=datetime.now()
            )
            
            # Save batch result
            await self._save_batch_result(batch_result)
            
        except Exception as e:
            logger.error(f"Error processing batch {batch_id}: {e}")
            await self._update_batch_status(batch_id, ProcessingStatus.FAILED)
    
    async def get_document_result(self, document_id: str) -> Optional[DocumentProcessingResult]:
        """Get document processing result"""
        try:
            # Try cache first
            cached_result = await self._get_cached_result(document_id)
            if cached_result:
                return cached_result
            
            # Try database
            if SessionLocal:
                db = SessionLocal()
                try:
                    record = db.query(DocumentRecord).filter(DocumentRecord.id == document_id).first()
                    if record:
                        return DocumentProcessingResult.parse_raw(record.results)
                finally:
                    db.close()
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting document result: {e}")
            return None
    
    async def get_batch_result(self, batch_id: str) -> Optional[BatchProcessingResult]:
        """Get batch processing result"""
        try:
            if SessionLocal:
                db = SessionLocal()
                try:
                    record = db.query(BatchRecord).filter(BatchRecord.id == batch_id).first()
                    if record:
                        return BatchProcessingResult.parse_raw(record.results)
                finally:
                    db.close()
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting batch result: {e}")
            return None
    
    async def search_documents(self, query: DocumentSearchQuery) -> Dict[str, Any]:
        """Search documents using semantic search"""
        try:
            # This is a simplified implementation
            # In production, you'd use a proper vector database like ChromaDB or Elasticsearch
            
            if not SessionLocal:
                return {"results": [], "total": 0, "processing_time": 0.0}
            
            db = SessionLocal()
            try:
                # Simple text search for now
                records = db.query(DocumentRecord).filter(
                    DocumentRecord.filename.contains(query.query) |
                    DocumentRecord.metadata.contains(query.query)
                ).limit(query.limit).offset(query.offset).all()
                
                results = []
                for record in records:
                    try:
                        result = DocumentProcessingResult.parse_raw(record.results)
                        results.append({
                            "document_id": record.id,
                            "filename": record.filename,
                            "metadata": result.metadata,
                            "relevance_score": 0.8  # Placeholder
                        })
                    except:
                        continue
                
                return {
                    "results": results,
                    "total": len(results),
                    "processing_time": 0.1
                }
                
            finally:
                db.close()
                
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            return {"results": [], "total": 0, "processing_time": 0.0}
    
    async def compare_documents(self, request: DocumentComparisonRequest) -> DocumentComparisonResult:
        """Compare multiple documents"""
        try:
            start_time = datetime.now()
            comparison_id = str(uuid.uuid4())
            
            # Get document results
            documents = []
            for doc_id in request.document_ids:
                result = await self.get_document_result(doc_id)
                if result and result.status == ProcessingStatus.COMPLETED:
                    documents.append(result)
            
            if len(documents) < 2:
                raise HTTPException(status_code=400, detail="Need at least 2 valid documents for comparison")
            
            # Calculate similarity scores
            similarity_scores = []
            similar_sections = []
            
            for i in range(len(documents)):
                for j in range(i + 1, len(documents)):
                    # Simple similarity calculation based on keywords
                    doc1_keywords = set()
                    doc2_keywords = set()
                    
                    if documents[i].keyword_result:
                        doc1_keywords = set([kw["word"] for kw in documents[i].keyword_result.keywords])
                    
                    if documents[j].keyword_result:
                        doc2_keywords = set([kw["word"] for kw in documents[j].keyword_result.keywords])
                    
                    if doc1_keywords and doc2_keywords:
                        intersection = doc1_keywords.intersection(doc2_keywords)
                        union = doc1_keywords.union(doc2_keywords)
                        similarity = len(intersection) / len(union) if union else 0.0
                    else:
                        similarity = 0.0
                    
                    similarity_scores.append(similarity)
                    
                    if similarity > request.threshold:
                        similar_sections.append({
                            "document1_id": documents[i].document_id,
                            "document2_id": documents[j].document_id,
                            "similarity": similarity,
                            "common_keywords": list(intersection) if 'intersection' in locals() else []
                        })
            
            # Determine if plagiarism detected
            is_plagiarized = any(score > request.threshold for score in similarity_scores)
            
            return DocumentComparisonResult(
                comparison_id=comparison_id,
                document_ids=request.document_ids,
                comparison_type=request.comparison_type,
                similarity_scores=similarity_scores,
                is_plagiarized=is_plagiarized,
                similar_sections=similar_sections,
                processing_time=(datetime.now() - start_time).total_seconds(),
                created_at=start_time
            )
            
        except Exception as e:
            logger.error(f"Error comparing documents: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        try:
            stats = {
                "total_documents": 0,
                "total_batches": 0,
                "active_processing": 0,
                "system_uptime": 0.0,
                "average_processing_time": 0.0,
                "success_rate": 0.0,
                "timestamp": datetime.now()
            }
            
            if SessionLocal:
                db = SessionLocal()
                try:
                    # Count documents
                    total_docs = db.query(DocumentRecord).count()
                    completed_docs = db.query(DocumentRecord).filter(
                        DocumentRecord.status == ProcessingStatus.COMPLETED
                    ).count()
                    
                    # Count batches
                    total_batches = db.query(BatchRecord).count()
                    
                    # Count active processing
                    active_processing = db.query(DocumentRecord).filter(
                        DocumentRecord.status == ProcessingStatus.PROCESSING
                    ).count()
                    
                    # Calculate average processing time
                    avg_time = db.query(DocumentRecord).filter(
                        DocumentRecord.processing_time.isnot(None)
                    ).with_entities(DocumentRecord.processing_time).all()
                    
                    avg_processing_time = sum([t[0] for t in avg_time]) / len(avg_time) if avg_time else 0.0
                    
                    # Calculate success rate
                    success_rate = completed_docs / total_docs if total_docs > 0 else 0.0
                    
                    stats.update({
                        "total_documents": total_docs,
                        "total_batches": total_batches,
                        "active_processing": active_processing,
                        "average_processing_time": avg_processing_time,
                        "success_rate": success_rate
                    })
                    
                finally:
                    db.close()
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting system stats: {e}")
            return {
                "total_documents": 0,
                "total_batches": 0,
                "active_processing": 0,
                "system_uptime": 0.0,
                "average_processing_time": 0.0,
                "success_rate": 0.0,
                "timestamp": datetime.now()
            }
    
    def _validate_file(self, file: UploadFile) -> bool:
        """Validate uploaded file"""
        try:
            # Check file size
            if file.size > settings.max_file_size:
                return False
            
            # Check file extension
            file_extension = os.path.splitext(file.filename)[1].lower().lstrip('.')
            if file_extension not in settings.supported_formats:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating file: {e}")
            return False
    
    async def _save_document_record(self, result: DocumentProcessingResult):
        """Save document record to database"""
        if not SessionLocal:
            return
        
        try:
            db = SessionLocal()
            try:
                record = DocumentRecord(
                    id=result.document_id,
                    filename=result.metadata.filename,
                    content_type=result.metadata.content_type,
                    file_size=result.metadata.file_size,
                    language=result.metadata.language,
                    status=result.status,
                    metadata=result.metadata.json(),
                    results=result.json(),
                    created_at=result.created_at,
                    updated_at=datetime.now(),
                    processing_time=result.processing_time,
                    error_message=result.error_message
                )
                
                db.add(record)
                db.commit()
                
            finally:
                db.close()
                
        except Exception as e:
            logger.error(f"Error saving document record: {e}")
    
    async def _save_batch_result(self, result: BatchProcessingResult):
        """Save batch result to database"""
        if not SessionLocal:
            return
        
        try:
            db = SessionLocal()
            try:
                record = db.query(BatchRecord).filter(BatchRecord.id == result.batch_id).first()
                if record:
                    record.status = result.status
                    record.processed_documents = result.processed_documents
                    record.failed_documents = result.failed_documents
                    record.results = result.json()
                    record.completed_at = result.completed_at
                    record.processing_time = result.processing_time
                    
                    db.commit()
                
            finally:
                db.close()
                
        except Exception as e:
            logger.error(f"Error saving batch result: {e}")
    
    async def _update_batch_progress(self, batch_id: str, success: bool):
        """Update batch processing progress"""
        if not SessionLocal:
            return
        
        try:
            db = SessionLocal()
            try:
                record = db.query(BatchRecord).filter(BatchRecord.id == batch_id).first()
                if record:
                    if success:
                        record.processed_documents += 1
                    else:
                        record.failed_documents += 1
                    
                    db.commit()
                
            finally:
                db.close()
                
        except Exception as e:
            logger.error(f"Error updating batch progress: {e}")
    
    async def _update_batch_status(self, batch_id: str, status: ProcessingStatus):
        """Update batch status"""
        if not SessionLocal:
            return
        
        try:
            db = SessionLocal()
            try:
                record = db.query(BatchRecord).filter(BatchRecord.id == batch_id).first()
                if record:
                    record.status = status
                    db.commit()
                
            finally:
                db.close()
                
        except Exception as e:
            logger.error(f"Error updating batch status: {e}")
    
    async def _cache_result(self, document_id: str, result: DocumentProcessingResult):
        """Cache document result"""
        if not redis_client:
            return
        
        try:
            cache_key = f"document_result:{document_id}"
            redis_client.setex(
                cache_key,
                settings.cache_ttl,
                result.json()
            )
        except Exception as e:
            logger.error(f"Error caching result: {e}")
    
    async def _get_cached_result(self, document_id: str) -> Optional[DocumentProcessingResult]:
        """Get cached document result"""
        if not redis_client:
            return None
        
        try:
            cache_key = f"document_result:{document_id}"
            cached_data = redis_client.get(cache_key)
            if cached_data:
                return DocumentProcessingResult.parse_raw(cached_data)
        except Exception as e:
            logger.error(f"Error getting cached result: {e}")
        
        return None


# Global service instance
document_service = DocumentService()














