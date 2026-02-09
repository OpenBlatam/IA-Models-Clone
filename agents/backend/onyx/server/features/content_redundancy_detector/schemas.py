"""
Type definitions and schemas for the content redundancy detector
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, field_validator
from datetime import datetime


# Input Models
class ContentInput(BaseModel):
    """Input model for content analysis"""
    content: str = Field(..., min_length=1, max_length=50000, description="Content to analyze")
    threshold: float = Field(default=0.8, ge=0.0, le=1.0, description="Similarity threshold")
    
    @field_validator('content')
    @classmethod
    def validate_content(cls, v: str) -> str:
        if not v.strip():
            raise ValueError('Content cannot be empty')
        return v.strip()


class SimilarityInput(BaseModel):
    """Input model for similarity comparison"""
    text1: str = Field(..., min_length=1, max_length=50000, description="First text")
    text2: str = Field(..., min_length=1, max_length=50000, description="Second text")
    threshold: float = Field(default=0.8, ge=0.0, le=1.0, description="Similarity threshold")
    
    @field_validator('text1', 'text2')
    @classmethod
    def validate_texts(cls, v: str) -> str:
        if not v.strip():
            raise ValueError('Texts cannot be empty')
        return v.strip()


class BatchAnalysisInput(BaseModel):
    """Input model for batch analysis"""
    texts: List[str] = Field(..., min_items=1, max_items=100, description="List of texts to analyze")
    
    @field_validator('texts')
    @classmethod
    def validate_texts(cls, v: List[str]) -> List[str]:
        if not v:
            raise ValueError('Texts list cannot be empty')
        return [text.strip() for text in v if text.strip()]


class TopicExtractionInput(BaseModel):
    """Input model for topic extraction"""
    texts: List[str] = Field(..., min_items=2, max_items=1000, description="List of texts for topic extraction")
    num_topics: int = Field(default=5, ge=2, le=20, description="Number of topics to extract")
    
    @field_validator('texts')
    @classmethod
    def validate_texts(cls, v: List[str]) -> List[str]:
        if not v:
            raise ValueError('Texts list cannot be empty')
        return [text.strip() for text in v if text.strip()]


class PlagiarismDetectionInput(BaseModel):
    """Input model for plagiarism detection"""
    content: str = Field(..., min_length=1, max_length=50000, description="Content to check for plagiarism")
    reference_texts: List[str] = Field(..., min_items=1, max_items=100, description="Reference texts to compare against")
    threshold: float = Field(default=0.8, ge=0.0, le=1.0, description="Similarity threshold for plagiarism")
    
    @field_validator('content')
    @classmethod
    def validate_content(cls, v: str) -> str:
        if not v.strip():
            raise ValueError('Content cannot be empty')
        return v.strip()
    
    @field_validator('reference_texts')
    @classmethod
    def validate_reference_texts(cls, v: List[str]) -> List[str]:
        if not v:
            raise ValueError('Reference texts cannot be empty')
        return [text.strip() for text in v if text.strip()]


class SummaryInput(BaseModel):
    """Input model for text summarization"""
    content: str = Field(..., min_length=50, max_length=50000, description="Content to summarize")
    max_length: int = Field(default=150, ge=30, le=500, description="Maximum length of summary")
    
    @field_validator('content')
    @classmethod
    def validate_content(cls, v: str) -> str:
        if not v.strip():
            raise ValueError('Content cannot be empty')
        return v.strip()


# Output Models
class AnalysisResult(BaseModel):
    """Result model for content analysis"""
    content_hash: str = Field(..., description="MD5 hash of content")
    word_count: int = Field(..., description="Total word count")
    character_count: int = Field(..., description="Total character count")
    unique_words: int = Field(..., description="Number of unique words")
    redundancy_score: float = Field(..., ge=0.0, le=1.0, description="Redundancy score")
    analysis_time: float = Field(..., description="Analysis time in seconds")
    timestamp: str = Field(..., description="Analysis timestamp")


class SimilarityResult(BaseModel):
    """Result model for similarity comparison"""
    similarity_score: float = Field(..., ge=0.0, le=1.0, description="Similarity score")
    is_similar: bool = Field(..., description="Whether texts are similar based on threshold")
    common_words: List[str] = Field(..., description="List of common words")
    analysis_time: float = Field(..., description="Analysis time in seconds")
    timestamp: str = Field(..., description="Analysis timestamp")


class QualityResult(BaseModel):
    """Result model for quality assessment"""
    readability_score: float = Field(..., ge=0.0, le=100.0, description="Readability score")
    complexity_score: float = Field(..., ge=0.0, le=100.0, description="Complexity score")
    quality_rating: str = Field(..., description="Quality rating")
    suggestions: List[str] = Field(..., description="Improvement suggestions")
    analysis_time: float = Field(..., description="Analysis time in seconds")
    timestamp: str = Field(..., description="Analysis timestamp")


class SentimentResult(BaseModel):
    """Result model for sentiment analysis"""
    dominant_sentiment: str = Field(..., description="Dominant sentiment (positive/negative/neutral)")
    sentiment_scores: Dict[str, float] = Field(..., description="Sentiment scores for all categories")
    polarity: float = Field(..., ge=-1.0, le=1.0, description="Text polarity")
    subjectivity: float = Field(..., ge=0.0, le=1.0, description="Text subjectivity")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    timestamp: str = Field(..., description="Analysis timestamp")


class LanguageResult(BaseModel):
    """Result model for language detection"""
    language: str = Field(..., description="Detected language code")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence")
    timestamp: str = Field(..., description="Analysis timestamp")


class TopicResult(BaseModel):
    """Result model for topic extraction"""
    topics: List[Dict[str, Any]] = Field(..., description="Extracted topics with words and weights")
    num_topics: int = Field(..., description="Number of topics extracted")
    timestamp: str = Field(..., description="Analysis timestamp")


class SemanticSimilarityResult(BaseModel):
    """Result model for semantic similarity"""
    similarity_score: float = Field(..., ge=0.0, le=1.0, description="Semantic similarity score")
    similarity_percentage: float = Field(..., ge=0.0, le=100.0, description="Similarity as percentage")
    method: str = Field(..., description="Method used for calculation")
    timestamp: str = Field(..., description="Analysis timestamp")


class PlagiarismResult(BaseModel):
    """Result model for plagiarism detection"""
    is_plagiarized: bool = Field(..., description="Whether content is plagiarized")
    max_similarity: float = Field(..., ge=0.0, le=1.0, description="Maximum similarity score")
    similarities: List[Dict[str, Any]] = Field(..., description="Similarity scores for all references")
    threshold: float = Field(..., ge=0.0, le=1.0, description="Threshold used for detection")
    timestamp: str = Field(..., description="Analysis timestamp")


class EntityResult(BaseModel):
    """Result model for entity extraction"""
    entities: List[Dict[str, Any]] = Field(..., description="Extracted named entities")
    entity_count: int = Field(..., description="Number of entities found")
    timestamp: str = Field(..., description="Analysis timestamp")


class SummaryResult(BaseModel):
    """Result model for text summarization"""
    summary: str = Field(..., description="Generated summary")
    original_length: int = Field(..., description="Length of original text")
    summary_length: int = Field(..., description="Length of summary")
    compression_ratio: float = Field(..., ge=0.0, le=1.0, description="Compression ratio")
    timestamp: str = Field(..., description="Analysis timestamp")


class ReadabilityResult(BaseModel):
    """Result model for advanced readability analysis"""
    flesch_score: float = Field(..., description="Flesch reading ease score")
    grade_level: float = Field(..., description="Estimated grade level")
    avg_sentence_length: float = Field(..., description="Average sentence length")
    avg_word_length: float = Field(..., description="Average word length")
    sentence_count: int = Field(..., description="Number of sentences")
    word_count: int = Field(..., description="Number of words")
    character_count: int = Field(..., description="Number of characters")
    timestamp: str = Field(..., description="Analysis timestamp")


class ComprehensiveAnalysisResult(BaseModel):
    """Result model for comprehensive analysis"""
    text_hash: str = Field(..., description="Hash of analyzed text")
    text_length: int = Field(..., description="Length of text")
    sentiment: Dict[str, Any] = Field(..., description="Sentiment analysis results")
    language: Dict[str, Any] = Field(..., description="Language detection results")
    entities: Dict[str, Any] = Field(..., description="Entity extraction results")
    summary: Dict[str, Any] = Field(..., description="Text summarization results")
    readability: Dict[str, Any] = Field(..., description="Readability analysis results")
    timestamp: str = Field(..., description="Analysis timestamp")


class BatchAnalysisResult(BaseModel):
    """Result model for batch analysis"""
    results: List[Dict[str, Any]] = Field(..., description="List of analysis results")
    total_processed: int = Field(..., description="Total number of texts processed")
    successful_analyses: int = Field(..., description="Number of successful analyses")
    failed_analyses: int = Field(..., description="Number of failed analyses")
    timestamp: str = Field(..., description="Analysis timestamp")


# System Models
class ErrorResponse(BaseModel):
    """Error response model"""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Error detail")
    timestamp: str = Field(..., description="Error timestamp")


class HealthResponse(BaseModel):
    """Health check response model"""
    status: str = Field(..., description="System status")
    timestamp: str = Field(..., description="Check timestamp")
    version: str = Field(..., description="Application version")


class StatsResponse(BaseModel):
    """Statistics response model"""
    total_endpoints: int = Field(..., description="Total number of endpoints")
    features: List[str] = Field(..., description="List of features")
    version: str = Field(..., description="Application version")
    status: str = Field(..., description="System status")


# Internal Types
AnalysisData = Dict[str, Any]
SimilarityData = Dict[str, Any]
QualityData = Dict[str, Any]



