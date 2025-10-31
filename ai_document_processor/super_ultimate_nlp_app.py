"""
Super Ultimate NLP Application for AI Document Processor
Main FastAPI application with all super advanced NLP features
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
from advanced_nlp_features import advanced_nlp_features
from advanced_nlp_routes import router as advanced_nlp_router
from super_advanced_nlp import super_advanced_nlp_system
from super_advanced_nlp_routes import router as super_advanced_nlp_router

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
    logger.info("Starting Super Ultimate NLP Application...")
    
    # Initialize enhanced NLP system
    try:
        await enhanced_nlp_system.load_enhanced_nltk_components()
        await enhanced_nlp_system.load_enhanced_spacy_model("en_core_web_sm")
        logger.info("Enhanced NLP system initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing enhanced NLP system: {e}")
    
    # Initialize advanced NLP features
    try:
        await advanced_nlp_features.load_dependency_parser("spacy")
        logger.info("Advanced NLP features initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing advanced NLP features: {e}")
    
    # Initialize super advanced NLP system
    try:
        await super_advanced_nlp_system.load_transformer_model("bert-base-uncased")
        await super_advanced_nlp_system.load_embedding_model("sentence-transformers/all-MiniLM-L6-v2")
        logger.info("Super Advanced NLP system initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing super advanced NLP system: {e}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Super Ultimate NLP Application...")

# Create FastAPI application
app = FastAPI(
    title="Super Ultimate NLP AI Document Processor",
    description="Super Ultimate Natural Language Processing system with all advanced features",
    version="4.0.0",
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
app.include_router(advanced_nlp_router)
app.include_router(super_advanced_nlp_router)

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with system information"""
    return {
        "message": "Super Ultimate NLP AI Document Processor",
        "version": "4.0.0",
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
            "Dependency Parsing",
            "Coreference Resolution",
            "Entity Linking",
            "Discourse Analysis",
            "Word Embeddings",
            "Semantic Networks",
            "Knowledge Graphs",
            "Super Advanced Classification",
            "Super Advanced Sentiment",
            "Super Advanced Generation",
            "Super Advanced QA",
            "Super Advanced NER",
            "Super Advanced Summarization",
            "Transformer Models",
            "Embedding Models",
            "Creative Writing",
            "Analytical Analysis",
            "Batch Processing",
            "Comprehensive Analysis"
        ],
        "endpoints": {
            "enhanced_nlp": "/enhanced-nlp",
            "advanced_nlp": "/advanced-nlp",
            "super_advanced_nlp": "/super-advanced-nlp",
            "health": "/health",
            "stats": "/stats",
            "docs": "/docs"
        }
    }

# Health check endpoint
@app.get("/health")
async def health_check():
    """System health check"""
    try:
        # Check enhanced NLP system
        enhanced_stats = enhanced_nlp_system.get_enhanced_nlp_stats()
        
        # Check advanced NLP features
        advanced_stats = advanced_nlp_features.get_advanced_nlp_stats()
        
        # Check super advanced NLP system
        super_advanced_stats = super_advanced_nlp_system.get_super_advanced_nlp_stats()
        
        return {
            "status": "healthy",
            "timestamp": time.time(),
            "enhanced_nlp": {
                "status": "healthy",
                "uptime_seconds": enhanced_stats["uptime_seconds"],
                "success_rate": enhanced_stats["success_rate"],
                "total_requests": enhanced_stats["stats"]["total_nlp_requests"],
                "successful_requests": enhanced_stats["stats"]["successful_nlp_requests"],
                "failed_requests": enhanced_stats["stats"]["failed_nlp_requests"]
            },
            "advanced_nlp": {
                "status": "healthy",
                "uptime_seconds": advanced_stats["uptime_seconds"],
                "success_rate": advanced_stats["success_rate"],
                "total_requests": advanced_stats["stats"]["total_advanced_requests"],
                "successful_requests": advanced_stats["stats"]["successful_advanced_requests"],
                "failed_requests": advanced_stats["stats"]["failed_advanced_requests"]
            },
            "super_advanced_nlp": {
                "status": "healthy",
                "uptime_seconds": super_advanced_stats["uptime_seconds"],
                "success_rate": super_advanced_stats["success_rate"],
                "total_requests": super_advanced_stats["stats"]["total_super_advanced_requests"],
                "successful_requests": super_advanced_stats["stats"]["successful_super_advanced_requests"],
                "failed_requests": super_advanced_stats["stats"]["failed_super_advanced_requests"]
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
        enhanced_stats = enhanced_nlp_system.get_enhanced_nlp_stats()
        advanced_stats = advanced_nlp_features.get_advanced_nlp_stats()
        super_advanced_stats = super_advanced_nlp_system.get_super_advanced_nlp_stats()
        
        return {
            "system": "Super Ultimate NLP AI Document Processor",
            "version": "4.0.0",
            "status": "running",
            "enhanced_nlp_stats": enhanced_stats,
            "advanced_nlp_stats": advanced_stats,
            "super_advanced_nlp_stats": super_advanced_stats,
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
                "Dependency Parsing (spacy, nltk, stanford)",
                "Coreference Resolution (spacy, rule_based)",
                "Entity Linking (spacy, rule_based)",
                "Discourse Analysis (rhetorical, coherence)",
                "Word Embeddings (word2vec, tfidf, count)",
                "Semantic Networks (co_occurrence, semantic_similarity)",
                "Knowledge Graphs (entity_relation, dependency_based)",
                "Super Advanced Classification (transformer, ensemble)",
                "Super Advanced Sentiment (transformer with emotions and aspects)",
                "Super Advanced Generation (transformer, creative)",
                "Super Advanced QA (transformer, retrieval)",
                "Super Advanced NER (transformer, rule_based)",
                "Super Advanced Summarization (transformer, extractive)",
                "Transformer Models (bert, roberta, distilbert, gpt2, etc.)",
                "Embedding Models (sentence-transformers, word2vec, etc.)",
                "Creative Writing (style-based generation)",
                "Analytical Analysis (comprehensive text analysis)",
                "Batch Processing (all features)",
                "Comprehensive Analysis (all features combined)",
                "Text Comparison (side-by-side analysis)"
            ],
            "processing_methods": {
                "tokenization": ["spacy", "nltk", "tweet"],
                "sentiment": ["nltk", "spacy", "transformer"],
                "preprocessing": [
                    "lowercase", "remove_punctuation", "remove_numbers",
                    "remove_stopwords", "remove_stopwords_advanced", "lemmatize",
                    "stem", "lancaster_stem", "snowball_stem",
                    "remove_extra_whitespace", "remove_urls", "remove_emails"
                ],
                "keywords": ["tfidf", "frequency", "yake"],
                "similarity": ["cosine", "jaccard", "euclidean", "manhattan"],
                "topics": ["lda", "nmf", "lsa"],
                "classification": ["naive_bayes", "ensemble", "transformer"],
                "summarization": ["extractive", "abstractive", "hybrid", "transformer"],
                "dependencies": ["spacy", "nltk", "stanford"],
                "coreferences": ["spacy", "rule_based"],
                "entities": ["spacy", "rule_based", "transformer"],
                "discourse": ["rhetorical", "coherence"],
                "embeddings": ["word2vec", "tfidf", "count", "transformer"],
                "networks": ["co_occurrence", "semantic_similarity"],
                "graphs": ["entity_relation", "dependency_based"],
                "generation": ["transformer", "creative"],
                "qa": ["transformer", "retrieval"],
                "ner": ["transformer", "rule_based"]
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
        enhanced_stats = enhanced_nlp_system.get_enhanced_nlp_stats()
        advanced_stats = advanced_nlp_features.get_advanced_nlp_stats()
        super_advanced_stats = super_advanced_nlp_system.get_super_advanced_nlp_stats()
        
        return {
            "performance_metrics": {
                "enhanced_nlp": {
                    "uptime_seconds": enhanced_stats["uptime_seconds"],
                    "uptime_hours": enhanced_stats["uptime_hours"],
                    "success_rate": enhanced_stats["success_rate"],
                    "average_tokens_per_request": enhanced_stats["average_tokens_per_request"],
                    "average_sentences_per_request": enhanced_stats["average_sentences_per_request"],
                    "embeddings_created": enhanced_stats["embeddings_created"],
                    "similarities_calculated": enhanced_stats["similarities_calculated"],
                    "clusters_created": enhanced_stats["clusters_created"],
                    "topics_discovered": enhanced_stats["topics_discovered"],
                    "classifications_made": enhanced_stats["classifications_made"],
                    "networks_built": enhanced_stats["networks_built"]
                },
                "advanced_nlp": {
                    "uptime_seconds": advanced_stats["uptime_seconds"],
                    "uptime_hours": advanced_stats["uptime_hours"],
                    "success_rate": advanced_stats["success_rate"],
                    "dependencies_parsed": advanced_stats["dependencies_parsed"],
                    "coreferences_resolved": advanced_stats["coreferences_resolved"],
                    "entities_linked": advanced_stats["entities_linked"],
                    "discourse_analyzed": advanced_stats["discourse_analyzed"],
                    "embeddings_created": advanced_stats["embeddings_created"],
                    "semantic_networks_built": advanced_stats["semantic_networks_built"],
                    "knowledge_graphs_created": advanced_stats["knowledge_graphs_created"]
                },
                "super_advanced_nlp": {
                    "uptime_seconds": super_advanced_stats["uptime_seconds"],
                    "uptime_hours": super_advanced_stats["uptime_hours"],
                    "success_rate": super_advanced_stats["success_rate"],
                    "transformer_requests": super_advanced_stats["transformer_requests"],
                    "embedding_requests": super_advanced_stats["embedding_requests"],
                    "classification_requests": super_advanced_stats["classification_requests"],
                    "generation_requests": super_advanced_stats["generation_requests"],
                    "qa_requests": super_advanced_stats["qa_requests"],
                    "ner_requests": super_advanced_stats["ner_requests"],
                    "sentiment_requests": super_advanced_stats["sentiment_requests"]
                }
            },
            "request_statistics": {
                "enhanced_nlp": {
                    "total_requests": enhanced_stats["stats"]["total_nlp_requests"],
                    "successful_requests": enhanced_stats["stats"]["successful_nlp_requests"],
                    "failed_requests": enhanced_stats["stats"]["failed_nlp_requests"],
                    "total_tokens_processed": enhanced_stats["stats"]["total_tokens_processed"],
                    "total_sentences_processed": enhanced_stats["stats"]["total_sentences_processed"],
                    "total_documents_processed": enhanced_stats["stats"]["total_documents_processed"]
                },
                "advanced_nlp": {
                    "total_requests": advanced_stats["stats"]["total_advanced_requests"],
                    "successful_requests": advanced_stats["stats"]["successful_advanced_requests"],
                    "failed_requests": advanced_stats["stats"]["failed_advanced_requests"],
                    "dependencies_parsed": advanced_stats["stats"]["total_dependencies_parsed"],
                    "coreferences_resolved": advanced_stats["stats"]["total_coreferences_resolved"],
                    "entities_linked": advanced_stats["stats"]["total_entities_linked"],
                    "discourse_analyzed": advanced_stats["stats"]["total_discourse_analyzed"],
                    "embeddings_created": advanced_stats["stats"]["total_embeddings_created"],
                    "semantic_networks_built": advanced_stats["stats"]["total_semantic_networks_built"],
                    "knowledge_graphs_created": advanced_stats["stats"]["total_knowledge_graphs_created"]
                },
                "super_advanced_nlp": {
                    "total_requests": super_advanced_stats["stats"]["total_super_advanced_requests"],
                    "successful_requests": super_advanced_stats["stats"]["successful_super_advanced_requests"],
                    "failed_requests": super_advanced_stats["stats"]["failed_super_advanced_requests"],
                    "transformer_requests": super_advanced_stats["stats"]["total_transformer_requests"],
                    "embedding_requests": super_advanced_stats["stats"]["total_embedding_requests"],
                    "classification_requests": super_advanced_stats["stats"]["total_classification_requests"],
                    "generation_requests": super_advanced_stats["stats"]["total_generation_requests"],
                    "qa_requests": super_advanced_stats["stats"]["total_qa_requests"],
                    "ner_requests": super_advanced_stats["stats"]["total_ner_requests"],
                    "sentiment_requests": super_advanced_stats["stats"]["total_sentiment_requests"]
                }
            }
        }
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Super ultimate analysis endpoint
@app.post("/analyze/super-ultimate")
async def super_ultimate_analysis(text: str):
    """Super ultimate comprehensive analysis with all NLP features"""
    try:
        results = {}
        
        # Enhanced NLP analysis
        enhanced_results = {}
        
        # Tokenization
        tokenization_result = await enhanced_nlp_system.enhanced_tokenization(
            text=text,
            method="spacy",
            include_phrases=True,
            include_entities=True
        )
        enhanced_results["tokenization"] = tokenization_result
        
        # Sentiment analysis
        sentiment_result = await enhanced_nlp_system.enhanced_sentiment_analysis(
            text=text,
            method="nltk",
            include_emotions=True
        )
        enhanced_results["sentiment"] = sentiment_result
        
        # Keyword extraction
        keyword_result = await enhanced_nlp_system.enhanced_keyword_extraction(
            text=text,
            method="tfidf",
            top_k=10,
            include_phrases=True
        )
        enhanced_results["keywords"] = keyword_result
        
        # Text summarization
        summarization_result = await enhanced_nlp_system.enhanced_text_summarization(
            text=text,
            method="extractive",
            max_sentences=3,
            include_ranking=True
        )
        enhanced_results["summarization"] = summarization_result
        
        # Readability metrics
        readability_result = await enhanced_nlp_system.calculate_readability_metrics(text)
        enhanced_results["readability"] = readability_result
        
        # Advanced NLP analysis
        advanced_results = {}
        
        # Dependency parsing
        dependencies_result = await advanced_nlp_features.parse_dependencies(text, "spacy")
        advanced_results["dependencies"] = dependencies_result
        
        # Coreference resolution
        coreferences_result = await advanced_nlp_features.resolve_coreferences(text, "spacy")
        advanced_results["coreferences"] = coreferences_result
        
        # Entity linking
        entities_result = await advanced_nlp_features.link_entities(text, "spacy")
        advanced_results["entities"] = entities_result
        
        # Discourse analysis
        discourse_result = await advanced_nlp_features.analyze_discourse(text, "rhetorical")
        advanced_results["discourse"] = discourse_result
        
        # Word embeddings
        embeddings_result = await advanced_nlp_features.create_word_embeddings(text, "word2vec")
        advanced_results["embeddings"] = embeddings_result
        
        # Semantic network
        network_result = await advanced_nlp_features.build_semantic_network(text, "co_occurrence")
        advanced_results["semantic_network"] = network_result
        
        # Knowledge graph
        knowledge_graph_result = await advanced_nlp_features.create_knowledge_graph(text, "entity_relation")
        advanced_results["knowledge_graph"] = knowledge_graph_result
        
        # Super Advanced NLP analysis
        super_advanced_results = {}
        
        # Super advanced classification
        classification_result = await super_advanced_nlp_system.super_advanced_text_classification(
            text=text,
            categories=["technology", "business", "science", "health", "education"],
            method="transformer",
            include_confidence=True
        )
        super_advanced_results["classification"] = classification_result
        
        # Super advanced sentiment
        super_sentiment_result = await super_advanced_nlp_system.super_advanced_sentiment_analysis(
            text=text,
            method="transformer",
            include_emotions=True,
            include_aspects=True
        )
        super_advanced_results["sentiment"] = super_sentiment_result
        
        # Super advanced NER
        ner_result = await super_advanced_nlp_system.super_advanced_entity_recognition(
            text=text,
            method="transformer"
        )
        super_advanced_results["ner"] = ner_result
        
        # Super advanced summarization
        super_summarization_result = await super_advanced_nlp_system.super_advanced_text_summarization(
            text=text,
            method="transformer",
            max_length=100,
            include_highlights=True
        )
        super_advanced_results["summarization"] = super_summarization_result
        
        results = {
            "enhanced_nlp": enhanced_results,
            "advanced_nlp": advanced_results,
            "super_advanced_nlp": super_advanced_results
        }
        
        return {
            "status": "success",
            "super_ultimate_analysis": results,
            "text_length": len(text),
            "total_features": len(enhanced_results) + len(advanced_results) + len(super_advanced_results)
        }
        
    except Exception as e:
        logger.error(f"Error in super ultimate analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Batch super ultimate analysis endpoint
@app.post("/batch/analyze/super-ultimate")
async def batch_super_ultimate_analysis(texts: list):
    """Batch super ultimate analysis for multiple texts"""
    try:
        results = []
        for text in texts:
            result = await super_ultimate_analysis(text)
            results.append(result)
        
        return {
            "status": "success",
            "results": results,
            "total_texts": len(texts)
        }
    except Exception as e:
        logger.error(f"Error in batch super ultimate analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Comparison endpoint
@app.post("/compare/super-ultimate")
async def super_ultimate_text_comparison(text1: str, text2: str):
    """Super ultimate text comparison with all NLP features"""
    try:
        # Analyze first text
        analysis1 = await super_ultimate_analysis(text1)
        
        # Analyze second text
        analysis2 = await super_ultimate_analysis(text2)
        
        # Calculate similarity using enhanced NLP
        similarity_result = await enhanced_nlp_system.enhanced_similarity_calculation(
            text1=text1,
            text2=text2,
            method="cosine",
            include_semantic=True
        )
        
        return {
            "status": "success",
            "text1_analysis": analysis1["super_ultimate_analysis"],
            "text2_analysis": analysis2["super_ultimate_analysis"],
            "similarity": similarity_result,
            "text1_length": len(text1),
            "text2_length": len(text2)
        }
        
    except Exception as e:
        logger.error(f"Error in super ultimate text comparison: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Main execution
if __name__ == "__main__":
    uvicorn.run(
        "super_ultimate_nlp_app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )












