"""
Data Models for AI Document Processor
"""

from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field, validator
from datetime import datetime
from enum import Enum


class DocumentType(str, Enum):
    """Supported document types"""
    PDF = "pdf"
    DOCX = "docx"
    DOC = "doc"
    TXT = "txt"
    RTF = "rtf"
    ODT = "odt"
    PPTX = "pptx"
    XLSX = "xlsx"
    CSV = "csv"
    IMAGE = "image"


class ProcessingStatus(str, Enum):
    """Document processing status"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AnalysisType(str, Enum):
    """Types of document analysis"""
    OCR = "ocr"
    CLASSIFICATION = "classification"
    ENTITY_EXTRACTION = "entity_extraction"
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    TOPIC_MODELING = "topic_modeling"
    SUMMARIZATION = "summarization"
    KEYWORD_EXTRACTION = "keyword_extraction"
    SEMANTIC_SEARCH = "semantic_search"
    PLAGIARISM_DETECTION = "plagiarism_detection"
    CONTENT_ANALYSIS = "content_analysis"


# Input Models
class DocumentUpload(BaseModel):
    """Model for document upload"""
    filename: str = Field(..., description="Document filename")
    content_type: str = Field(..., description="MIME type of the document")
    file_size: int = Field(..., description="File size in bytes")
    analysis_types: List[AnalysisType] = Field(default=[AnalysisType.CONTENT_ANALYSIS], description="Types of analysis to perform")
    language: Optional[str] = Field(default="en", description="Document language")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    @validator('file_size')
    def validate_file_size(cls, v):
        if v > 100 * 1024 * 1024:  # 100MB
            raise ValueError('File size cannot exceed 100MB')
        return v


class BatchDocumentUpload(BaseModel):
    """Model for batch document upload"""
    documents: List[DocumentUpload] = Field(..., min_items=1, max_items=100, description="List of documents to process")
    batch_name: str = Field(..., description="Name for the batch")
    analysis_types: List[AnalysisType] = Field(default=[AnalysisType.CONTENT_ANALYSIS], description="Default analysis types")
    priority: int = Field(default=1, ge=1, le=10, description="Processing priority")


class DocumentSearchQuery(BaseModel):
    """Model for document search query"""
    query: str = Field(..., min_length=1, description="Search query")
    search_type: str = Field(default="semantic", description="Type of search: semantic, keyword, hybrid")
    filters: Dict[str, Any] = Field(default_factory=dict, description="Search filters")
    limit: int = Field(default=10, ge=1, le=100, description="Maximum number of results")
    offset: int = Field(default=0, ge=0, description="Result offset")


class DocumentComparisonRequest(BaseModel):
    """Model for document comparison request"""
    document_ids: List[str] = Field(..., min_items=2, max_items=10, description="IDs of documents to compare")
    comparison_type: str = Field(default="similarity", description="Type of comparison: similarity, plagiarism, content")
    threshold: float = Field(default=0.8, ge=0.0, le=1.0, description="Similarity threshold")


# Output Models
class DocumentMetadata(BaseModel):
    """Model for document metadata"""
    document_id: str = Field(..., description="Unique document ID")
    filename: str = Field(..., description="Original filename")
    content_type: str = Field(..., description="MIME type")
    file_size: int = Field(..., description="File size in bytes")
    language: str = Field(..., description="Detected language")
    page_count: Optional[int] = Field(default=None, description="Number of pages")
    word_count: Optional[int] = Field(default=None, description="Number of words")
    character_count: Optional[int] = Field(default=None, description="Number of characters")
    created_at: datetime = Field(..., description="Creation timestamp")
    modified_at: Optional[datetime] = Field(default=None, description="Last modification timestamp")
    author: Optional[str] = Field(default=None, description="Document author")
    title: Optional[str] = Field(default=None, description="Document title")
    subject: Optional[str] = Field(default=None, description="Document subject")
    keywords: Optional[List[str]] = Field(default=None, description="Document keywords")


class OCRResult(BaseModel):
    """Model for OCR results"""
    text: str = Field(..., description="Extracted text")
    confidence: float = Field(..., ge=0.0, le=1.0, description="OCR confidence score")
    language: str = Field(..., description="Detected language")
    page_results: List[Dict[str, Any]] = Field(..., description="Per-page OCR results")
    processing_time: float = Field(..., description="Processing time in seconds")


class ClassificationResult(BaseModel):
    """Model for document classification results"""
    category: str = Field(..., description="Predicted category")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Classification confidence")
    all_categories: List[Dict[str, float]] = Field(..., description="All category predictions")
    processing_time: float = Field(..., description="Processing time in seconds")


class EntityExtractionResult(BaseModel):
    """Model for entity extraction results"""
    entities: List[Dict[str, Any]] = Field(..., description="Extracted entities")
    entity_count: int = Field(..., description="Number of entities found")
    entity_types: List[str] = Field(..., description="Types of entities found")
    processing_time: float = Field(..., description="Processing time in seconds")


class SentimentAnalysisResult(BaseModel):
    """Model for sentiment analysis results"""
    sentiment: str = Field(..., description="Overall sentiment")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Sentiment confidence")
    polarity: float = Field(..., ge=-1.0, le=1.0, description="Sentiment polarity")
    subjectivity: float = Field(..., ge=0.0, le=1.0, description="Text subjectivity")
    processing_time: float = Field(..., description="Processing time in seconds")


class TopicModelingResult(BaseModel):
    """Model for topic modeling results"""
    topics: List[Dict[str, Any]] = Field(..., description="Identified topics")
    topic_count: int = Field(..., description="Number of topics")
    dominant_topic: str = Field(..., description="Most prominent topic")
    processing_time: float = Field(..., description="Processing time in seconds")


class SummarizationResult(BaseModel):
    """Model for document summarization results"""
    summary: str = Field(..., description="Generated summary")
    original_length: int = Field(..., description="Original text length")
    summary_length: int = Field(..., description="Summary length")
    compression_ratio: float = Field(..., ge=0.0, le=1.0, description="Compression ratio")
    key_points: List[str] = Field(..., description="Key points extracted")
    processing_time: float = Field(..., description="Processing time in seconds")


class KeywordExtractionResult(BaseModel):
    """Model for keyword extraction results"""
    keywords: List[Dict[str, Any]] = Field(..., description="Extracted keywords with scores")
    keyword_count: int = Field(..., description="Number of keywords")
    processing_time: float = Field(..., description="Processing time in seconds")


class SemanticSearchResult(BaseModel):
    """Model for semantic search results"""
    query: str = Field(..., description="Search query")
    results: List[Dict[str, Any]] = Field(..., description="Search results")
    total_results: int = Field(..., description="Total number of results")
    processing_time: float = Field(..., description="Processing time in seconds")


class PlagiarismDetectionResult(BaseModel):
    """Model for plagiarism detection results"""
    is_plagiarized: bool = Field(..., description="Whether document is plagiarized")
    similarity_score: float = Field(..., ge=0.0, le=1.0, description="Similarity score")
    similar_documents: List[Dict[str, Any]] = Field(..., description="Similar documents found")
    processing_time: float = Field(..., description="Processing time in seconds")


class ContentAnalysisResult(BaseModel):
    """Model for content analysis results"""
    readability_score: float = Field(..., description="Readability score")
    complexity_score: float = Field(..., description="Complexity score")
    quality_rating: str = Field(..., description="Content quality rating")
    suggestions: List[str] = Field(..., description="Improvement suggestions")
    processing_time: float = Field(..., description="Processing time in seconds")


class DocumentProcessingResult(BaseModel):
    """Model for complete document processing results"""
    document_id: str = Field(..., description="Document ID")
    status: ProcessingStatus = Field(..., description="Processing status")
    metadata: DocumentMetadata = Field(..., description="Document metadata")
    ocr_result: Optional[OCRResult] = Field(default=None, description="OCR results")
    classification_result: Optional[ClassificationResult] = Field(default=None, description="Classification results")
    entity_result: Optional[EntityExtractionResult] = Field(default=None, description="Entity extraction results")
    sentiment_result: Optional[SentimentAnalysisResult] = Field(default=None, description="Sentiment analysis results")
    topic_result: Optional[TopicModelingResult] = Field(default=None, description="Topic modeling results")
    summarization_result: Optional[SummarizationResult] = Field(default=None, description="Summarization results")
    keyword_result: Optional[KeywordExtractionResult] = Field(default=None, description="Keyword extraction results")
    content_analysis_result: Optional[ContentAnalysisResult] = Field(default=None, description="Content analysis results")
    processing_time: float = Field(..., description="Total processing time")
    created_at: datetime = Field(..., description="Processing timestamp")
    error_message: Optional[str] = Field(default=None, description="Error message if failed")


class BatchProcessingResult(BaseModel):
    """Model for batch processing results"""
    batch_id: str = Field(..., description="Batch ID")
    batch_name: str = Field(..., description="Batch name")
    status: ProcessingStatus = Field(..., description="Batch status")
    total_documents: int = Field(..., description="Total documents in batch")
    processed_documents: int = Field(..., description="Successfully processed documents")
    failed_documents: int = Field(..., description="Failed documents")
    results: List[DocumentProcessingResult] = Field(..., description="Individual document results")
    processing_time: float = Field(..., description="Total batch processing time")
    created_at: datetime = Field(..., description="Batch creation timestamp")
    completed_at: Optional[datetime] = Field(default=None, description="Batch completion timestamp")


class DocumentComparisonResult(BaseModel):
    """Model for document comparison results"""
    comparison_id: str = Field(..., description="Comparison ID")
    document_ids: List[str] = Field(..., description="Compared document IDs")
    comparison_type: str = Field(..., description="Type of comparison")
    similarity_scores: List[float] = Field(..., description="Similarity scores between documents")
    is_plagiarized: bool = Field(..., description="Whether plagiarism detected")
    similar_sections: List[Dict[str, Any]] = Field(..., description="Similar sections found")
    processing_time: float = Field(..., description="Processing time")
    created_at: datetime = Field(..., description="Comparison timestamp")


# System Models
class HealthResponse(BaseModel):
    """Model for health check response"""
    status: str = Field(..., description="System status")
    version: str = Field(..., description="Application version")
    timestamp: datetime = Field(..., description="Check timestamp")
    components: Dict[str, str] = Field(..., description="Component statuses")


class ErrorResponse(BaseModel):
    """Model for error responses"""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(default=None, description="Error detail")
    timestamp: datetime = Field(..., description="Error timestamp")
    request_id: Optional[str] = Field(default=None, description="Request ID")


class StatsResponse(BaseModel):
    """Model for system statistics"""
    total_documents: int = Field(..., description="Total documents processed")
    total_batches: int = Field(..., description="Total batches processed")
    active_processing: int = Field(..., description="Currently processing documents")
    system_uptime: float = Field(..., description="System uptime in seconds")
    average_processing_time: float = Field(..., description="Average processing time")
    success_rate: float = Field(..., description="Processing success rate")
    timestamp: datetime = Field(..., description="Statistics timestamp")















