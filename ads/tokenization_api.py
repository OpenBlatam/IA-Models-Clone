from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field
import asyncio
import json
from datetime import datetime
from onyx.utils.logger import setup_logger
from onyx.server.features.ads.tokenization_service import TokenizationService
from onyx.server.features.ads.optimized_config import settings
from typing import Any, List, Dict, Optional
import logging
"""
API endpoints for advanced tokenization and sequence handling.
"""


logger = setup_logger()
router = APIRouter(prefix="/tokenization", tags=["tokenization"])

# Initialize tokenization service
tokenization_service = TokenizationService()

class TextPreprocessingRequest(BaseModel):
    """Request model for text preprocessing."""
    text: str = Field(..., description="Text to preprocess")
    remove_stopwords: bool = Field(False, description="Whether to remove stopwords")
    normalize: bool = Field(True, description="Whether to normalize text")

class TextPreprocessingResponse(BaseModel):
    """Response model for text preprocessing."""
    original_text: str
    processed_text: str
    word_count: int
    character_count: int
    processing_time_ms: float

class TokenizationRequest(BaseModel):
    """Request model for tokenization."""
    text: str = Field(..., description="Text to tokenize")
    max_length: int = Field(512, description="Maximum sequence length")
    model_name: str = Field("microsoft/DialoGPT-medium", description="Model name for tokenizer")
    include_analysis: bool = Field(True, description="Include text analysis")

class TokenizationResponse(BaseModel):
    """Response model for tokenization."""
    text: str
    token_count: int
    token_ids: List[int]
    attention_mask: List[int]
    analysis: Optional[Dict[str, Any]] = None
    processing_time_ms: float

class AdsPromptTokenizationRequest(BaseModel):
    """Request model for ads prompt tokenization."""
    prompt: str = Field(..., description="Ad prompt")
    target_audience: Optional[str] = Field(None, description="Target audience")
    keywords: Optional[List[str]] = Field(None, description="Keywords")
    brand: Optional[str] = Field(None, description="Brand name")
    max_length: int = Field(512, description="Maximum sequence length")
    model_name: str = Field("microsoft/DialoGPT-medium", description="Model name")

class SequenceOptimizationRequest(BaseModel):
    """Request model for sequence optimization."""
    texts: List[str] = Field(..., description="List of texts to optimize")
    target_token_count: int = Field(512, description="Target token count per sequence")
    model_name: str = Field("microsoft/DialoGPT-medium", description="Model name")

class SequenceOptimizationResponse(BaseModel):
    """Response model for sequence optimization."""
    original_count: int
    optimized_count: int
    optimized_texts: List[str]
    statistics: Dict[str, Any]
    processing_time_ms: float

class TextAnalysisRequest(BaseModel):
    """Request model for text analysis."""
    text: str = Field(..., description="Text to analyze")
    model_name: str = Field("microsoft/DialoGPT-medium", description="Model name")

class TextAnalysisResponse(BaseModel):
    """Response model for text analysis."""
    text: str
    analysis: Dict[str, Any]
    processing_time_ms: float

class BatchTokenizationRequest(BaseModel):
    """Request model for batch tokenization."""
    texts: List[str] = Field(..., description="List of texts to tokenize")
    max_length: int = Field(512, description="Maximum sequence length")
    model_name: str = Field("microsoft/DialoGPT-medium", description="Model name")
    include_analysis: bool = Field(True, description="Include text analysis")

class BatchTokenizationResponse(BaseModel):
    """Response model for batch tokenization."""
    total_texts: int
    successful_tokenizations: int
    failed_tokenizations: int
    results: List[Dict[str, Any]]
    statistics: Dict[str, Any]
    processing_time_ms: float

@router.post("/preprocess", response_model=TextPreprocessingResponse)
async def preprocess_text(request: TextPreprocessingRequest):
    """Preprocess text for ads generation."""
    try:
        start_time = datetime.now()
        
        # Preprocess text
        if request.normalize:
            processed_text = tokenization_service.preprocessor.normalize_text(request.text)
        else:
            processed_text = request.text
        
        if request.remove_stopwords:
            processed_text = tokenization_service.preprocessor.clean_ads_text(
                processed_text, remove_stopwords=True
            )
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return TextPreprocessingResponse(
            original_text=request.text,
            processed_text=processed_text,
            word_count=len(processed_text.split()),
            character_count=len(processed_text),
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.exception("Error preprocessing text")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/tokenize", response_model=TokenizationResponse)
async def tokenize_text(request: TokenizationRequest):
    """Tokenize text using advanced tokenization."""
    try:
        start_time = datetime.now()
        
        # Get tokenization service for specific model
        model_tokenization_service = TokenizationService(request.model_name)
        
        # Tokenize text
        tokens = model_tokenization_service.tokenizer.tokenize_text(
            request.text,
            max_length=request.max_length,
            return_tensors="pt"
        )
        
        # Convert to lists for response
        token_ids = tokens['input_ids'][0].tolist()
        attention_mask = tokens['attention_mask'][0].tolist()
        
        # Analyze text if requested
        analysis = None
        if request.include_analysis:
            analysis = await model_tokenization_service.analyze_text_complexity(request.text)
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return TokenizationResponse(
            text=request.text,
            token_count=len(token_ids),
            token_ids=token_ids,
            attention_mask=attention_mask,
            analysis=analysis,
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.exception("Error tokenizing text")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/tokenize-ads-prompt", response_model=TokenizationResponse)
async def tokenize_ads_prompt(request: AdsPromptTokenizationRequest):
    """Tokenize ads prompt with structured format."""
    try:
        start_time = datetime.now()
        
        # Get tokenization service for specific model
        model_tokenization_service = TokenizationService(request.model_name)
        
        # Tokenize ads prompt
        tokens = model_tokenization_service.tokenizer.tokenize_ads_prompt(
            prompt=request.prompt,
            target_audience=request.target_audience,
            keywords=request.keywords,
            brand=request.brand,
            max_length=request.max_length
        )
        
        # Convert to lists for response
        token_ids = tokens['input_ids'][0].tolist()
        attention_mask = tokens['attention_mask'][0].tolist()
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return TokenizationResponse(
            text=request.prompt,
            token_count=len(token_ids),
            token_ids=token_ids,
            attention_mask=attention_mask,
            analysis=None,
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.exception("Error tokenizing ads prompt")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/optimize-sequences", response_model=SequenceOptimizationResponse)
async def optimize_sequences(request: SequenceOptimizationRequest):
    """Optimize sequence lengths for batch processing."""
    try:
        start_time = datetime.now()
        
        # Get tokenization service for specific model
        model_tokenization_service = TokenizationService(request.model_name)
        
        # Optimize sequences
        optimized_texts = await model_tokenization_service.optimize_sequence_length(
            request.texts, request.target_token_count
        )
        
        # Calculate statistics
        original_lengths = [len(text.split()) for text in request.texts]
        optimized_lengths = [len(text.split()) for text in optimized_texts]
        
        statistics = {
            'original_avg_length': sum(original_lengths) / len(original_lengths) if original_lengths else 0,
            'optimized_avg_length': sum(optimized_lengths) / len(optimized_lengths) if optimized_lengths else 0,
            'original_max_length': max(original_lengths) if original_lengths else 0,
            'optimized_max_length': max(optimized_lengths) if optimized_lengths else 0,
            'segmentation_ratio': len(optimized_texts) / len(request.texts) if request.texts else 0
        }
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return SequenceOptimizationResponse(
            original_count=len(request.texts),
            optimized_count=len(optimized_texts),
            optimized_texts=optimized_texts,
            statistics=statistics,
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.exception("Error optimizing sequences")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/analyze-text", response_model=TextAnalysisResponse)
async def analyze_text(request: TextAnalysisRequest):
    """Analyze text complexity and characteristics."""
    try:
        start_time = datetime.now()
        
        # Get tokenization service for specific model
        model_tokenization_service = TokenizationService(request.model_name)
        
        # Analyze text
        analysis = await model_tokenization_service.analyze_text_complexity(request.text)
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return TextAnalysisResponse(
            text=request.text,
            analysis=analysis,
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.exception("Error analyzing text")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/batch-tokenize", response_model=BatchTokenizationResponse)
async def batch_tokenize_texts(request: BatchTokenizationRequest):
    """Tokenize multiple texts in batch."""
    try:
        start_time = datetime.now()
        
        # Get tokenization service for specific model
        model_tokenization_service = TokenizationService(request.model_name)
        
        results = []
        successful_count = 0
        failed_count = 0
        
        for i, text in enumerate(request.texts):
            try:
                # Tokenize text
                tokens = model_tokenization_service.tokenizer.tokenize_text(
                    text,
                    max_length=request.max_length,
                    return_tensors="pt"
                )
                
                # Analyze if requested
                analysis = None
                if request.include_analysis:
                    analysis = await model_tokenization_service.analyze_text_complexity(text)
                
                results.append({
                    'index': i,
                    'text': text,
                    'token_count': len(tokens['input_ids'][0]),
                    'token_ids': tokens['input_ids'][0].tolist(),
                    'attention_mask': tokens['attention_mask'][0].tolist(),
                    'analysis': analysis,
                    'success': True
                })
                successful_count += 1
                
            except Exception as e:
                results.append({
                    'index': i,
                    'text': text,
                    'error': str(e),
                    'success': False
                })
                failed_count += 1
        
        # Calculate statistics
        token_counts = [r['token_count'] for r in results if r.get('success')]
        statistics = {
            'total_texts': len(request.texts),
            'successful_tokenizations': successful_count,
            'failed_tokenizations': failed_count,
            'avg_token_count': sum(token_counts) / len(token_counts) if token_counts else 0,
            'min_token_count': min(token_counts) if token_counts else 0,
            'max_token_count': max(token_counts) if token_counts else 0
        }
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return BatchTokenizationResponse(
            total_texts=len(request.texts),
            successful_tokenizations=successful_count,
            failed_tokenizations=failed_count,
            results=results,
            statistics=statistics,
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.exception("Error in batch tokenization")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/vocabulary-info")
async def get_vocabulary_info(model_name: str = "microsoft/DialoGPT-medium"):
    """Get vocabulary information for a specific model."""
    try:
        # Get tokenization service for specific model
        model_tokenization_service = TokenizationService(model_name)
        
        vocab_size = model_tokenization_service.tokenizer.get_vocab_size()
        special_tokens = model_tokenization_service.tokenizer.get_special_tokens()
        
        return {
            'model_name': model_name,
            'vocabulary_size': vocab_size,
            'special_tokens': special_tokens,
            'additional_special_tokens': [
                '[URL]', '[EMAIL]', '[PHONE]', '[AD_START]', '[AD_END]',
                '[TARGET_AUDIENCE]', '[KEYWORDS]', '[BRAND]', '[CTA]'
            ]
        }
        
    except Exception as e:
        logger.exception("Error getting vocabulary info")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def tokenization_health_check():
    """Health check for tokenization service."""
    try:
        # Test basic tokenization
        test_text = "Hello world"
        tokens = tokenization_service.tokenizer.tokenize_text(test_text)
        
        return {
            'status': 'healthy',
            'service': 'tokenization',
            'test_tokenization': len(tokens['input_ids'][0]) > 0,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.exception("Tokenization service health check failed")
        raise HTTPException(status_code=503, detail=str(e))

@router.on_event("shutdown")
async def shutdown_tokenization_service():
    """Cleanup tokenization service on shutdown."""
    try:
        await tokenization_service.close()
        logger.info("Tokenization service shutdown complete")
    except Exception as e:
        logger.exception("Error during tokenization service shutdown") 