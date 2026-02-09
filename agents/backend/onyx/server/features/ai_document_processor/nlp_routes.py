"""
NLP Routes
Real, working Natural Language Processing endpoints for AI document processing
"""

import logging
from fastapi import APIRouter, HTTPException, Form
from fastapi.responses import JSONResponse
from typing import Optional, List
import asyncio
from nlp_system import nlp_system

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/v1/nlp", tags=["Natural Language Processing"])

@router.post("/load-spacy-model")
async def load_spacy_model(
    model_name: str = Form("en_core_web_sm")
):
    """Load spaCy model"""
    try:
        result = await nlp_system.load_spacy_model(model_name)
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Error loading spaCy model: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/load-nltk-components")
async def load_nltk_components():
    """Load NLTK components"""
    try:
        result = await nlp_system.load_nltk_components()
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Error loading NLTK components: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/tokenize-text")
async def tokenize_text(
    text: str = Form(...),
    method: str = Form("spacy")
):
    """Tokenize text using different methods"""
    try:
        result = await nlp_system.tokenize_text(text, method)
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Error tokenizing text: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/sentence-segmentation")
async def sentence_segmentation(
    text: str = Form(...),
    method: str = Form("spacy")
):
    """Segment text into sentences"""
    try:
        result = await nlp_system.sentence_segmentation(text, method)
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Error segmenting sentences: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/pos-tagging")
async def pos_tagging(
    text: str = Form(...),
    method: str = Form("spacy")
):
    """Perform part-of-speech tagging"""
    try:
        result = await nlp_system.pos_tagging(text, method)
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Error performing POS tagging: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/named-entity-recognition")
async def named_entity_recognition(
    text: str = Form(...),
    method: str = Form("spacy")
):
    """Perform named entity recognition"""
    try:
        result = await nlp_system.named_entity_recognition(text, method)
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Error performing NER: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/sentiment-analysis")
async def sentiment_analysis(
    text: str = Form(...),
    method: str = Form("nltk")
):
    """Perform sentiment analysis"""
    try:
        result = await nlp_system.sentiment_analysis(text, method)
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Error performing sentiment analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/text-preprocessing")
async def text_preprocessing(
    text: str = Form(...),
    steps: List[str] = Form(["lowercase", "remove_punctuation", "remove_stopwords", "lemmatize"])
):
    """Comprehensive text preprocessing"""
    try:
        result = await nlp_system.text_preprocessing(text, steps)
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Error preprocessing text: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/extract-keywords")
async def extract_keywords(
    text: str = Form(...),
    method: str = Form("tfidf"),
    top_k: int = Form(10)
):
    """Extract keywords from text"""
    try:
        result = await nlp_system.extract_keywords(text, method, top_k)
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Error extracting keywords: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/calculate-similarity")
async def calculate_similarity(
    text1: str = Form(...),
    text2: str = Form(...),
    method: str = Form("cosine")
):
    """Calculate similarity between two texts"""
    try:
        result = await nlp_system.calculate_similarity(text1, text2, method)
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Error calculating similarity: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/topic-modeling")
async def topic_modeling(
    texts: List[str] = Form(...),
    method: str = Form("lda"),
    num_topics: int = Form(5)
):
    """Perform topic modeling on a collection of texts"""
    try:
        result = await nlp_system.topic_modeling(texts, method, num_topics)
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Error performing topic modeling: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/text-classification")
async def text_classification(
    text: str = Form(...),
    categories: List[str] = Form(...),
    method: str = Form("naive_bayes")
):
    """Classify text into categories"""
    try:
        result = await nlp_system.text_classification(text, categories, method)
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Error classifying text: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/text-summarization")
async def text_summarization(
    text: str = Form(...),
    method: str = Form("extractive"),
    max_sentences: int = Form(3)
):
    """Summarize text using different methods"""
    try:
        result = await nlp_system.text_summarization(text, method, max_sentences)
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Error summarizing text: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/nlp-models")
async def get_nlp_models():
    """Get all NLP models"""
    try:
        result = nlp_system.get_nlp_models()
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Error getting NLP models: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/nlp-corpus")
async def get_nlp_corpus():
    """Get NLP corpus information"""
    try:
        result = nlp_system.get_nlp_corpus()
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Error getting NLP corpus: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/nlp-vocabulary")
async def get_nlp_vocabulary():
    """Get NLP vocabulary information"""
    try:
        result = nlp_system.get_nlp_vocabulary()
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Error getting NLP vocabulary: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/nlp-embeddings")
async def get_nlp_embeddings():
    """Get NLP embeddings information"""
    try:
        result = nlp_system.get_nlp_embeddings()
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Error getting NLP embeddings: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/nlp-similarity")
async def get_nlp_similarity():
    """Get NLP similarity information"""
    try:
        result = nlp_system.get_nlp_similarity()
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Error getting NLP similarity: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/nlp-clusters")
async def get_nlp_clusters():
    """Get NLP clusters information"""
    try:
        result = nlp_system.get_nlp_clusters()
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Error getting NLP clusters: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/nlp-topics")
async def get_nlp_topics():
    """Get NLP topics information"""
    try:
        result = nlp_system.get_nlp_topics()
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Error getting NLP topics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/nlp-stats")
async def get_nlp_stats():
    """Get NLP processing statistics"""
    try:
        result = nlp_system.get_nlp_stats()
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Error getting NLP stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/nlp-dashboard")
async def get_nlp_dashboard():
    """Get comprehensive NLP dashboard"""
    try:
        # Get all NLP data
        models = nlp_system.get_nlp_models()
        corpus = nlp_system.get_nlp_corpus()
        vocabulary = nlp_system.get_nlp_vocabulary()
        embeddings = nlp_system.get_nlp_embeddings()
        similarity = nlp_system.get_nlp_similarity()
        clusters = nlp_system.get_nlp_clusters()
        topics = nlp_system.get_nlp_topics()
        stats = nlp_system.get_nlp_stats()
        
        # Calculate additional metrics
        total_requests = stats["stats"]["total_nlp_requests"]
        successful_requests = stats["stats"]["successful_nlp_requests"]
        failed_requests = stats["stats"]["failed_nlp_requests"]
        total_tokens = stats["stats"]["total_tokens_processed"]
        total_sentences = stats["stats"]["total_sentences_processed"]
        total_documents = stats["stats"]["total_documents_processed"]
        
        # Calculate success rate
        success_rate = (successful_requests / total_requests * 100) if total_requests > 0 else 0
        
        dashboard_data = {
            "timestamp": stats["uptime_seconds"],
            "overview": {
                "total_requests": total_requests,
                "successful_requests": successful_requests,
                "failed_requests": failed_requests,
                "success_rate": round(success_rate, 2),
                "total_tokens": total_tokens,
                "total_sentences": total_sentences,
                "total_documents": total_documents,
                "uptime_hours": stats["uptime_hours"]
            },
            "nlp_metrics": {
                "total_nlp_requests": stats["stats"]["total_nlp_requests"],
                "successful_nlp_requests": stats["stats"]["successful_nlp_requests"],
                "failed_nlp_requests": stats["stats"]["failed_nlp_requests"],
                "total_tokens_processed": stats["stats"]["total_tokens_processed"],
                "total_sentences_processed": stats["stats"]["total_sentences_processed"],
                "total_documents_processed": stats["stats"]["total_documents_processed"],
                "vocabulary_size": stats["stats"]["vocabulary_size"],
                "corpus_size": stats["stats"]["corpus_size"]
            },
            "models": models,
            "corpus": corpus,
            "vocabulary": vocabulary,
            "embeddings": embeddings,
            "similarity": similarity,
            "clusters": clusters,
            "topics": topics
        }
        
        return JSONResponse(content=dashboard_data)
    except Exception as e:
        logger.error(f"Error getting NLP dashboard: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/nlp-performance")
async def get_nlp_performance():
    """Get NLP processing performance analysis"""
    try:
        stats = nlp_system.get_nlp_stats()
        models = nlp_system.get_nlp_models()
        vocabulary = nlp_system.get_nlp_vocabulary()
        
        # Calculate performance metrics
        total_requests = stats["stats"]["total_nlp_requests"]
        successful_requests = stats["stats"]["successful_nlp_requests"]
        failed_requests = stats["stats"]["failed_nlp_requests"]
        total_tokens = stats["stats"]["total_tokens_processed"]
        total_sentences = stats["stats"]["total_sentences_processed"]
        vocabulary_size = stats["stats"]["vocabulary_size"]
        
        # Calculate metrics
        success_rate = (successful_requests / total_requests * 100) if total_requests > 0 else 0
        failure_rate = (failed_requests / total_requests * 100) if total_requests > 0 else 0
        average_tokens_per_request = stats["average_tokens_per_request"]
        average_sentences_per_request = stats["average_sentences_per_request"]
        
        performance_data = {
            "timestamp": stats["uptime_seconds"],
            "performance_metrics": {
                "success_rate": round(success_rate, 2),
                "failure_rate": round(failure_rate, 2),
                "total_requests": total_requests,
                "successful_requests": successful_requests,
                "failed_requests": failed_requests,
                "total_tokens": total_tokens,
                "total_sentences": total_sentences,
                "vocabulary_size": vocabulary_size
            },
            "processing_efficiency": {
                "average_tokens_per_request": round(average_tokens_per_request, 2),
                "average_sentences_per_request": round(average_sentences_per_request, 2),
                "tokens_per_second": round(total_tokens / stats["uptime_seconds"], 2) if stats["uptime_seconds"] > 0 else 0,
                "sentences_per_second": round(total_sentences / stats["uptime_seconds"], 2) if stats["uptime_seconds"] > 0 else 0
            },
            "model_performance": {
                "spacy_models_loaded": models["model_count"],
                "nltk_components_loaded": len([c for c in models["nltk_components"].values() if c]),
                "total_models": len(models["spacy_models"]) + len(models["nltk_components"])
            },
            "vocabulary_performance": {
                "vocabulary_size": vocabulary["vocabulary_size"],
                "word_frequencies": vocabulary["word_frequencies"],
                "pos_tags": vocabulary["pos_tags"],
                "named_entities": vocabulary["named_entities"]
            }
        }
        
        return JSONResponse(content=performance_data)
    except Exception as e:
        logger.error(f"Error getting NLP performance: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health-nlp")
async def health_check_nlp():
    """NLP system health check"""
    try:
        stats = nlp_system.get_nlp_stats()
        models = nlp_system.get_nlp_models()
        vocabulary = nlp_system.get_nlp_vocabulary()
        
        return JSONResponse(content={
            "status": "healthy",
            "service": "NLP System",
            "version": "1.0.0",
            "features": {
                "spacy_models": True,
                "nltk_components": True,
                "tokenization": True,
                "sentence_segmentation": True,
                "pos_tagging": True,
                "named_entity_recognition": True,
                "sentiment_analysis": True,
                "text_preprocessing": True,
                "keyword_extraction": True,
                "similarity_calculation": True,
                "topic_modeling": True,
                "text_classification": True,
                "text_summarization": True
            },
            "nlp_stats": stats["stats"],
            "system_status": {
                "total_nlp_requests": stats["stats"]["total_nlp_requests"],
                "successful_nlp_requests": stats["stats"]["successful_nlp_requests"],
                "failed_nlp_requests": stats["stats"]["failed_nlp_requests"],
                "total_tokens_processed": stats["stats"]["total_tokens_processed"],
                "total_sentences_processed": stats["stats"]["total_sentences_processed"],
                "total_documents_processed": stats["stats"]["total_documents_processed"],
                "vocabulary_size": stats["stats"]["vocabulary_size"],
                "corpus_size": stats["stats"]["corpus_size"],
                "uptime_hours": stats["uptime_hours"]
            },
            "available_models": {
                "spacy_models": list(models["spacy_models"].keys()),
                "nltk_components": list(models["nltk_components"].keys())
            },
            "processing_capabilities": {
                "tokenization_methods": ["spacy", "nltk", "regex"],
                "sentence_segmentation_methods": ["spacy", "nltk", "regex"],
                "pos_tagging_methods": ["spacy", "nltk"],
                "ner_methods": ["spacy", "nltk"],
                "sentiment_analysis_methods": ["nltk", "spacy"],
                "preprocessing_steps": ["lowercase", "remove_punctuation", "remove_stopwords", "lemmatize", "stem"],
                "keyword_extraction_methods": ["tfidf", "frequency"],
                "similarity_methods": ["cosine", "jaccard", "euclidean"],
                "topic_modeling_methods": ["lda", "kmeans"],
                "classification_methods": ["naive_bayes"],
                "summarization_methods": ["extractive", "abstractive"]
            }
        })
    except Exception as e:
        logger.error(f"Error in NLP health check: {e}")
        raise HTTPException(status_code=500, detail=str(e))












