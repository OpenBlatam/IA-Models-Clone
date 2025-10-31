"""
Enhanced NLP Application for AI Document Processor
Main FastAPI application with enhanced NLP features
"""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import logging
import time
import asyncio
from typing import Dict, Any
import uvicorn

# Import all systems
from enhanced_nlp_system import enhanced_nlp_system
from enhanced_nlp_routes import router as enhanced_nlp_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("Starting Enhanced NLP Application...")
    
    # Initialize enhanced NLP system
    try:
        await enhanced_nlp_system.load_enhanced_nltk_components()
        await enhanced_nlp_system.load_enhanced_spacy_model("en_core_web_sm")
        logger.info("Enhanced NLP system initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing enhanced NLP system: {e}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Enhanced NLP Application...")

# Create FastAPI application
app = FastAPI(
    title="Enhanced NLP AI Document Processor",
    description="Advanced Natural Language Processing system for document analysis",
    version="2.0.0",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

# Add timing middleware
@app.middleware("http")
async def timing_middleware(request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Global exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)}
    )

# Include routers
app.include_router(enhanced_nlp_router)

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with system information"""
    return {
        "message": "Enhanced NLP AI Document Processor",
        "version": "2.0.0",
        "status": "running",
        "features": [
            "Enhanced Tokenization",
            "Advanced Sentiment Analysis",
            "Text Preprocessing",
            "Keyword Extraction",
            "Similarity Calculation",
            "Topic Modeling",
            "Text Classification",
            "Text Summarization",
            "Word Networks",
            "Readability Metrics",
            "Batch Processing",
            "Comprehensive Analysis"
        ],
        "endpoints": {
            "enhanced_nlp": "/enhanced-nlp",
            "health": "/health",
            "stats": "/enhanced-nlp/stats",
            "docs": "/docs"
        }
    }

# Health check endpoint
@app.get("/health")
async def health_check():
    """System health check"""
    try:
        # Check enhanced NLP system
        nlp_stats = enhanced_nlp_system.get_enhanced_nlp_stats()
        
        return {
            "status": "healthy",
            "timestamp": time.time(),
            "uptime_seconds": nlp_stats["uptime_seconds"],
            "enhanced_nlp": {
                "status": "healthy",
                "success_rate": nlp_stats["success_rate"],
                "total_requests": nlp_stats["stats"]["total_nlp_requests"],
                "successful_requests": nlp_stats["stats"]["successful_nlp_requests"],
                "failed_requests": nlp_stats["stats"]["failed_nlp_requests"]
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "timestamp": time.time(),
            "error": str(e)
        }

# System information endpoint
@app.get("/info")
async def system_info():
    """Get system information"""
    try:
        nlp_stats = enhanced_nlp_system.get_enhanced_nlp_stats()
        
        return {
            "system": "Enhanced NLP AI Document Processor",
            "version": "2.0.0",
            "status": "running",
            "enhanced_nlp_stats": nlp_stats,
            "available_features": [
                "Enhanced Tokenization (spacy, nltk, tweet)",
                "Advanced Sentiment Analysis (nltk, spacy) with emotions",
                "Text Preprocessing (12+ steps)",
                "Keyword Extraction (tfidf, frequency, yake)",
                "Similarity Calculation (cosine, jaccard, euclidean, manhattan)",
                "Topic Modeling (lda, nmf, lsa) with coherence",
                "Text Classification (naive_bayes, ensemble)",
                "Text Summarization (extractive, abstractive, hybrid)",
                "Word Networks (co-occurrence analysis)",
                "Readability Metrics (flesch, smog, coleman-liau)",
                "Batch Processing (all features)",
                "Comprehensive Analysis (all features combined)",
                "Text Comparison (side-by-side analysis)"
            ],
            "processing_methods": {
                "tokenization": ["spacy", "nltk", "tweet"],
                "sentiment": ["nltk", "spacy"],
                "preprocessing": [
                    "lowercase", "remove_punctuation", "remove_numbers",
                    "remove_stopwords", "remove_stopwords_advanced", "lemmatize",
                    "stem", "lancaster_stem", "snowball_stem",
                    "remove_extra_whitespace", "remove_urls", "remove_emails"
                ],
                "keywords": ["tfidf", "frequency", "yake"],
                "similarity": ["cosine", "jaccard", "euclidean", "manhattan"],
                "topics": ["lda", "nmf", "lsa"],
                "classification": ["naive_bayes", "ensemble"],
                "summarization": ["extractive", "abstractive", "hybrid"]
            }
        }
    except Exception as e:
        logger.error(f"Error getting system info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Performance metrics endpoint
@app.get("/metrics")
async def get_metrics():
    """Get system performance metrics"""
    try:
        nlp_stats = enhanced_nlp_system.get_enhanced_nlp_stats()
        
        return {
            "performance_metrics": {
                "uptime_seconds": nlp_stats["uptime_seconds"],
                "uptime_hours": nlp_stats["uptime_hours"],
                "success_rate": nlp_stats["success_rate"],
                "average_tokens_per_request": nlp_stats["average_tokens_per_request"],
                "average_sentences_per_request": nlp_stats["average_sentences_per_request"],
                "embeddings_created": nlp_stats["embeddings_created"],
                "similarities_calculated": nlp_stats["similarities_calculated"],
                "clusters_created": nlp_stats["clusters_created"],
                "topics_discovered": nlp_stats["topics_discovered"],
                "classifications_made": nlp_stats["classifications_made"],
                "networks_built": nlp_stats["networks_built"]
            },
            "request_statistics": {
                "total_requests": nlp_stats["stats"]["total_nlp_requests"],
                "successful_requests": nlp_stats["stats"]["successful_nlp_requests"],
                "failed_requests": nlp_stats["stats"]["failed_nlp_requests"],
                "total_tokens_processed": nlp_stats["stats"]["total_tokens_processed"],
                "total_sentences_processed": nlp_stats["stats"]["total_sentences_processed"],
                "total_documents_processed": nlp_stats["stats"]["total_documents_processed"]
            }
        }
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Comparison endpoint
@app.post("/compare")
async def compare_texts(text1: str, text2: str):
    """Compare two texts using enhanced NLP features"""
    try:
        # This will use the comparison endpoint from enhanced_nlp_routes
        from enhanced_nlp_routes import compare_texts
        result = await compare_texts(text1, text2)
        return result
    except Exception as e:
        logger.error(f"Error in text comparison: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Comprehensive analysis endpoint
@app.post("/analyze")
async def comprehensive_analysis(text: str):
    """Comprehensive text analysis with all enhanced NLP features"""
    try:
        # This will use the comprehensive analysis endpoint from enhanced_nlp_routes
        from enhanced_nlp_routes import comprehensive_text_analysis
        result = await comprehensive_text_analysis(text)
        return result
    except Exception as e:
        logger.error(f"Error in comprehensive analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Batch processing endpoint
@app.post("/batch/analyze")
async def batch_comprehensive_analysis(texts: list):
    """Batch comprehensive analysis for multiple texts"""
    try:
        results = []
        for text in texts:
            result = await comprehensive_analysis(text)
            results.append(result)
        
        return {
            "status": "success",
            "results": results,
            "total_texts": len(texts)
        }
    except Exception as e:
        logger.error(f"Error in batch comprehensive analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Main execution
if __name__ == "__main__":
    uvicorn.run(
        "enhanced_nlp_app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )












