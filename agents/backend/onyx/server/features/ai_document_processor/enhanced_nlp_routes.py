"""
Enhanced NLP Routes for AI Document Processor
API routes for enhanced Natural Language Processing features
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import logging
from enhanced_nlp_system import enhanced_nlp_system

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/enhanced-nlp", tags=["Enhanced NLP"])

# Pydantic models
class TextInput(BaseModel):
    text: str = Field(..., description="Text to process")
    method: Optional[str] = Field("spacy", description="Processing method")
    include_phrases: Optional[bool] = Field(True, description="Include phrases")
    include_entities: Optional[bool] = Field(True, description="Include entities")

class SentimentInput(BaseModel):
    text: str = Field(..., description="Text to analyze")
    method: Optional[str] = Field("nltk", description="Analysis method")
    include_emotions: Optional[bool] = Field(True, description="Include emotions")

class PreprocessingInput(BaseModel):
    text: str = Field(..., description="Text to preprocess")
    steps: Optional[List[str]] = Field(None, description="Preprocessing steps")

class KeywordInput(BaseModel):
    text: str = Field(..., description="Text to extract keywords from")
    method: Optional[str] = Field("tfidf", description="Extraction method")
    top_k: Optional[int] = Field(10, description="Number of top keywords")
    include_phrases: Optional[bool] = Field(True, description="Include phrases")

class SimilarityInput(BaseModel):
    text1: str = Field(..., description="First text")
    text2: str = Field(..., description="Second text")
    method: Optional[str] = Field("cosine", description="Similarity method")
    include_semantic: Optional[bool] = Field(True, description="Include semantic analysis")

class TopicModelingInput(BaseModel):
    texts: List[str] = Field(..., description="Texts to analyze")
    method: Optional[str] = Field("lda", description="Topic modeling method")
    num_topics: Optional[int] = Field(5, description="Number of topics")
    include_coherence: Optional[bool] = Field(True, description="Include coherence analysis")

class ClassificationInput(BaseModel):
    text: str = Field(..., description="Text to classify")
    categories: List[str] = Field(..., description="Available categories")
    method: Optional[str] = Field("naive_bayes", description="Classification method")
    include_confidence: Optional[bool] = Field(True, description="Include confidence scores")

class SummarizationInput(BaseModel):
    text: str = Field(..., description="Text to summarize")
    method: Optional[str] = Field("extractive", description="Summarization method")
    max_sentences: Optional[int] = Field(3, description="Maximum sentences in summary")
    include_ranking: Optional[bool] = Field(True, description="Include sentence rankings")

class NetworkInput(BaseModel):
    texts: List[str] = Field(..., description="Texts to build network from")
    min_frequency: Optional[int] = Field(2, description="Minimum word frequency")

class ReadabilityInput(BaseModel):
    text: str = Field(..., description="Text to analyze readability")

# Enhanced tokenization endpoints
@router.post("/tokenize")
async def enhanced_tokenization(input_data: TextInput):
    """Enhanced tokenization with phrases and entities"""
    try:
        result = await enhanced_nlp_system.enhanced_tokenization(
            text=input_data.text,
            method=input_data.method,
            include_phrases=input_data.include_phrases,
            include_entities=input_data.include_entities
        )
        return result
    except Exception as e:
        logger.error(f"Error in enhanced tokenization: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/sentiment")
async def enhanced_sentiment_analysis(input_data: SentimentInput):
    """Enhanced sentiment analysis with emotions"""
    try:
        result = await enhanced_nlp_system.enhanced_sentiment_analysis(
            text=input_data.text,
            method=input_data.method,
            include_emotions=input_data.include_emotions
        )
        return result
    except Exception as e:
        logger.error(f"Error in enhanced sentiment analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/preprocess")
async def enhanced_text_preprocessing(input_data: PreprocessingInput):
    """Enhanced text preprocessing"""
    try:
        result = await enhanced_nlp_system.enhanced_text_preprocessing(
            text=input_data.text,
            steps=input_data.steps
        )
        return result
    except Exception as e:
        logger.error(f"Error in enhanced text preprocessing: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/keywords")
async def enhanced_keyword_extraction(input_data: KeywordInput):
    """Enhanced keyword extraction with phrases"""
    try:
        result = await enhanced_nlp_system.enhanced_keyword_extraction(
            text=input_data.text,
            method=input_data.method,
            top_k=input_data.top_k,
            include_phrases=input_data.include_phrases
        )
        return result
    except Exception as e:
        logger.error(f"Error in enhanced keyword extraction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/similarity")
async def enhanced_similarity_calculation(input_data: SimilarityInput):
    """Enhanced similarity calculation with semantic analysis"""
    try:
        result = await enhanced_nlp_system.enhanced_similarity_calculation(
            text1=input_data.text1,
            text2=input_data.text2,
            method=input_data.method,
            include_semantic=input_data.include_semantic
        )
        return result
    except Exception as e:
        logger.error(f"Error in enhanced similarity calculation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/topics")
async def enhanced_topic_modeling(input_data: TopicModelingInput):
    """Enhanced topic modeling with coherence analysis"""
    try:
        result = await enhanced_nlp_system.enhanced_topic_modeling(
            texts=input_data.texts,
            method=input_data.method,
            num_topics=input_data.num_topics,
            include_coherence=input_data.include_coherence
        )
        return result
    except Exception as e:
        logger.error(f"Error in enhanced topic modeling: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/classify")
async def enhanced_text_classification(input_data: ClassificationInput):
    """Enhanced text classification with confidence scores"""
    try:
        result = await enhanced_nlp_system.enhanced_text_classification(
            text=input_data.text,
            categories=input_data.categories,
            method=input_data.method,
            include_confidence=input_data.include_confidence
        )
        return result
    except Exception as e:
        logger.error(f"Error in enhanced text classification: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/summarize")
async def enhanced_text_summarization(input_data: SummarizationInput):
    """Enhanced text summarization with sentence ranking"""
    try:
        result = await enhanced_nlp_system.enhanced_text_summarization(
            text=input_data.text,
            method=input_data.method,
            max_sentences=input_data.max_sentences,
            include_ranking=input_data.include_ranking
        )
        return result
    except Exception as e:
        logger.error(f"Error in enhanced text summarization: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/network")
async def build_word_network(input_data: NetworkInput):
    """Build word co-occurrence network"""
    try:
        result = await enhanced_nlp_system.build_word_network(
            texts=input_data.texts,
            min_frequency=input_data.min_frequency
        )
        return result
    except Exception as e:
        logger.error(f"Error building word network: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/readability")
async def calculate_readability_metrics(input_data: ReadabilityInput):
    """Calculate enhanced readability metrics"""
    try:
        result = await enhanced_nlp_system.calculate_readability_metrics(
            text=input_data.text
        )
        return result
    except Exception as e:
        logger.error(f"Error calculating readability metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Model management endpoints
@router.post("/models/spacy/{model_name}")
async def load_spacy_model(model_name: str):
    """Load enhanced spaCy model"""
    try:
        result = await enhanced_nlp_system.load_enhanced_spacy_model(model_name)
        return result
    except Exception as e:
        logger.error(f"Error loading spaCy model: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/models/nltk")
async def load_nltk_components():
    """Load enhanced NLTK components"""
    try:
        result = await enhanced_nlp_system.load_enhanced_nltk_components()
        return result
    except Exception as e:
        logger.error(f"Error loading NLTK components: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Statistics and monitoring endpoints
@router.get("/stats")
async def get_enhanced_nlp_stats():
    """Get enhanced NLP processing statistics"""
    try:
        result = enhanced_nlp_system.get_enhanced_nlp_stats()
        return result
    except Exception as e:
        logger.error(f"Error getting enhanced NLP stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def enhanced_nlp_health():
    """Enhanced NLP system health check"""
    try:
        stats = enhanced_nlp_system.get_enhanced_nlp_stats()
        return {
            "status": "healthy",
            "uptime_seconds": stats["uptime_seconds"],
            "success_rate": stats["success_rate"],
            "total_requests": stats["stats"]["total_nlp_requests"],
            "successful_requests": stats["stats"]["successful_nlp_requests"],
            "failed_requests": stats["stats"]["failed_nlp_requests"]
        }
    except Exception as e:
        logger.error(f"Error in enhanced NLP health check: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Batch processing endpoints
@router.post("/batch/tokenize")
async def batch_enhanced_tokenization(texts: List[str], method: str = "spacy"):
    """Batch enhanced tokenization"""
    try:
        results = []
        for text in texts:
            result = await enhanced_nlp_system.enhanced_tokenization(
                text=text,
                method=method,
                include_phrases=True,
                include_entities=True
            )
            results.append(result)
        return {
            "status": "success",
            "results": results,
            "total_texts": len(texts)
        }
    except Exception as e:
        logger.error(f"Error in batch enhanced tokenization: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/batch/sentiment")
async def batch_enhanced_sentiment_analysis(texts: List[str], method: str = "nltk"):
    """Batch enhanced sentiment analysis"""
    try:
        results = []
        for text in texts:
            result = await enhanced_nlp_system.enhanced_sentiment_analysis(
                text=text,
                method=method,
                include_emotions=True
            )
            results.append(result)
        return {
            "status": "success",
            "results": results,
            "total_texts": len(texts)
        }
    except Exception as e:
        logger.error(f"Error in batch enhanced sentiment analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/batch/preprocess")
async def batch_enhanced_text_preprocessing(texts: List[str], steps: List[str] = None):
    """Batch enhanced text preprocessing"""
    try:
        if steps is None:
            steps = ["lowercase", "remove_punctuation", "remove_stopwords", "lemmatize"]
        
        results = []
        for text in texts:
            result = await enhanced_nlp_system.enhanced_text_preprocessing(
                text=text,
                steps=steps
            )
            results.append(result)
        return {
            "status": "success",
            "results": results,
            "total_texts": len(texts)
        }
    except Exception as e:
        logger.error(f"Error in batch enhanced text preprocessing: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/batch/keywords")
async def batch_enhanced_keyword_extraction(texts: List[str], method: str = "tfidf", top_k: int = 10):
    """Batch enhanced keyword extraction"""
    try:
        results = []
        for text in texts:
            result = await enhanced_nlp_system.enhanced_keyword_extraction(
                text=text,
                method=method,
                top_k=top_k,
                include_phrases=True
            )
            results.append(result)
        return {
            "status": "success",
            "results": results,
            "total_texts": len(texts)
        }
    except Exception as e:
        logger.error(f"Error in batch enhanced keyword extraction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/batch/summarize")
async def batch_enhanced_text_summarization(texts: List[str], method: str = "extractive", max_sentences: int = 3):
    """Batch enhanced text summarization"""
    try:
        results = []
        for text in texts:
            result = await enhanced_nlp_system.enhanced_text_summarization(
                text=text,
                method=method,
                max_sentences=max_sentences,
                include_ranking=True
            )
            results.append(result)
        return {
            "status": "success",
            "results": results,
            "total_texts": len(texts)
        }
    except Exception as e:
        logger.error(f"Error in batch enhanced text summarization: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Advanced analysis endpoints
@router.post("/analyze/comprehensive")
async def comprehensive_text_analysis(text: str):
    """Comprehensive text analysis with all enhanced NLP features"""
    try:
        results = {}
        
        # Tokenization
        tokenization_result = await enhanced_nlp_system.enhanced_tokenization(
            text=text,
            method="spacy",
            include_phrases=True,
            include_entities=True
        )
        results["tokenization"] = tokenization_result
        
        # Sentiment analysis
        sentiment_result = await enhanced_nlp_system.enhanced_sentiment_analysis(
            text=text,
            method="nltk",
            include_emotions=True
        )
        results["sentiment"] = sentiment_result
        
        # Keyword extraction
        keyword_result = await enhanced_nlp_system.enhanced_keyword_extraction(
            text=text,
            method="tfidf",
            top_k=10,
            include_phrases=True
        )
        results["keywords"] = keyword_result
        
        # Text summarization
        summarization_result = await enhanced_nlp_system.enhanced_text_summarization(
            text=text,
            method="extractive",
            max_sentences=3,
            include_ranking=True
        )
        results["summarization"] = summarization_result
        
        # Readability metrics
        readability_result = await enhanced_nlp_system.calculate_readability_metrics(text)
        results["readability"] = readability_result
        
        return {
            "status": "success",
            "comprehensive_analysis": results,
            "text_length": len(text)
        }
        
    except Exception as e:
        logger.error(f"Error in comprehensive text analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/analyze/comparison")
async def compare_texts(text1: str, text2: str):
    """Compare two texts using enhanced NLP features"""
    try:
        results = {}
        
        # Analyze first text
        analysis1 = await comprehensive_text_analysis(text1)
        results["text1_analysis"] = analysis1["comprehensive_analysis"]
        
        # Analyze second text
        analysis2 = await comprehensive_text_analysis(text2)
        results["text2_analysis"] = analysis2["comprehensive_analysis"]
        
        # Calculate similarity
        similarity_result = await enhanced_nlp_system.enhanced_similarity_calculation(
            text1=text1,
            text2=text2,
            method="cosine",
            include_semantic=True
        )
        results["similarity"] = similarity_result
        
        return {
            "status": "success",
            "comparison": results,
            "text1_length": len(text1),
            "text2_length": len(text2)
        }
        
    except Exception as e:
        logger.error(f"Error in text comparison: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Utility endpoints
@router.get("/methods")
async def get_available_methods():
    """Get available processing methods"""
    return {
        "tokenization_methods": ["spacy", "nltk", "tweet"],
        "sentiment_methods": ["nltk", "spacy"],
        "preprocessing_steps": [
            "lowercase", "remove_punctuation", "remove_numbers",
            "remove_stopwords", "remove_stopwords_advanced", "lemmatize",
            "stem", "lancaster_stem", "snowball_stem",
            "remove_extra_whitespace", "remove_urls", "remove_emails"
        ],
        "keyword_methods": ["tfidf", "frequency", "yake"],
        "similarity_methods": ["cosine", "jaccard", "euclidean", "manhattan"],
        "topic_modeling_methods": ["lda", "nmf", "lsa"],
        "classification_methods": ["naive_bayes", "ensemble"],
        "summarization_methods": ["extractive", "abstractive", "hybrid"]
    }

@router.get("/models/status")
async def get_models_status():
    """Get status of loaded models"""
    try:
        return {
            "spacy_models": {
                model: "loaded" if enhanced_nlp_system.nlp_models.get(model) is not None else "not_loaded"
                for model in enhanced_nlp_system.nlp_models
            },
            "nltk_components": {
                component: "loaded" if enhanced_nlp_system.nlp_pipelines.get(component) is not None else "not_loaded"
                for component in enhanced_nlp_system.nlp_pipelines
            }
        }
    except Exception as e:
        logger.error(f"Error getting models status: {e}")
        raise HTTPException(status_code=500, detail=str(e))












