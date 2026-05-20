"""
Servidor principal para Music Analyzer AI
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
import time
import logging

from api import router
from api.ml_music_api import router as ml_router
from config.settings import settings
from config.service_registry import register_all_services
from config.di_setup import setup_dependencies
from middleware.rate_limiter import RateLimitMiddleware, RateLimiter
from services.analytics_service import analytics_service

# Setup dependency injection (new improved system)
try:
    setup_dependencies()
    logger.info("Enhanced DI system initialized")
except Exception as e:
    logger.warning(f"Enhanced DI setup failed, falling back to legacy: {e}")
    # Fallback to legacy registration
    register_all_services()

# Gradio integration
try:
    from gradio.music_analyzer_ui import create_gradio_app
    import gradio as gr
    GRADIO_AVAILABLE = True
except ImportError:
    GRADIO_AVAILABLE = False
    gr = None

# Configurar logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Music Analyzer AI API",
    description="Sistema de análisis musical con integración a Spotify y coaching musical",
    version="2.21.0"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, especificar dominios permitidos
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configurar Rate Limiting
rate_limiter = RateLimiter(max_requests=100, window_seconds=60)
app.add_middleware(RateLimitMiddleware, rate_limiter=rate_limiter)

# Importar y configurar middlewares avanzados (opcional)
try:
    from utils.advanced.advanced_middleware import setup_advanced_middleware
    setup_advanced_middleware(
        app,
        service_name="music_analyzer_ai",
        enable_opentelemetry=True,
        opentelemetry_endpoint=None  # Configurar si está disponible
    )
    logger.info("Advanced middleware configured")
except ImportError:
    logger.warning("Advanced middleware not available. Install dependencies from requirements.txt")
except Exception as e:
    logger.warning(f"Could not setup advanced middleware: {e}")

# Middleware para tracking de analytics
@app.middleware("http")
async def analytics_middleware(request: Request, call_next):
    start_time = time.time()
    user_id = request.headers.get("X-User-ID")
    
    response = await call_next(request)
    
    response_time = time.time() - start_time
    endpoint = request.url.path
    method = request.method
    
    analytics_service.track_request(
        endpoint=endpoint,
        method=method,
        user_id=user_id,
        response_time=response_time,
        status_code=response.status_code
    )
    
    return response

# Incluir routers
app.include_router(router)
app.include_router(ml_router)

# Include new v1 API router (using use cases)
try:
    from api.v1.routes import v1_router
    from api.v1.middleware.error_handler import ErrorHandlerMiddleware
    
    # Add error handler middleware (applies to all routes)
    app.add_middleware(ErrorHandlerMiddleware)
    
    app.include_router(v1_router)
    logger.info("API v1 router (use cases) registered with error handling")
except ImportError as e:
    logger.warning(f"Could not register v1 router: {e}")

# Health check endpoints (optimized for cloud deployments)
try:
    from deployment.health.cloud_health import router as health_router
    app.include_router(health_router)
except ImportError:
    # Fallback health check if deployment module not available
    @app.get("/health")
    async def health():
        return {"status": "healthy", "service": "music-analyzer-ai"}

# Gradio interface
if GRADIO_AVAILABLE and gr:
    try:
        gradio_app = create_gradio_app()
        # Mount Gradio app
        app = gr.mount_gradio_app(app, gradio_app, path="/gradio")
        logger.info("Gradio interface mounted at /gradio")
    except Exception as e:
        logger.warning(f"Could not mount Gradio: {str(e)}")


@app.get("/")
async def root():
    """Endpoint raíz"""
    return {
        "service": "Music Analyzer AI",
        "version": "2.21.0",
        "status": "running",
        "endpoints": {
            "search": "/music/search",
            "analyze": "/music/analyze",
            "coaching": "/music/coaching",
            "compare": "/music/compare",
            "recommendations": "/music/track/{track_id}/recommendations",
            "track_info": "/music/track/{track_id}/info",
            "audio_features": "/music/track/{track_id}/audio-features",
            "audio_analysis": "/music/track/{track_id}/audio-analysis",
            "cache_stats": "/music/cache/stats",
            "cache_clear": "/music/cache/clear",
            "export": "/music/export/{track_id}",
            "history": "/music/history",
            "history_stats": "/music/history/stats",
            "analytics": "/music/analytics",
            "favorites": "/music/favorites",
            "tags": "/music/tags",
            "webhooks": "/music/webhooks",
            "health": "/music/health",
            "ml": {
                "analyze_comprehensive": "/music/ml/analyze-comprehensive",
                "compare_tracks": "/music/ml/compare-tracks",
                "predict_genre": "/music/ml/predict/genre",
                "predict_multi_task": "/music/ml/predict/multi-task",
                "pipeline_info": "/music/ml/pipeline/info",
                "performance_stats": "/music/ml/performance/stats",
                "benchmark": "/music/ml/performance/benchmark"
            },
            "deep_learning": {
                "initialize": "/music/ml/deep-learning/initialize",
                "predict": "/music/ml/deep-learning/predict/{track_id}",
                "batch_predict": "/music/ml/deep-learning/batch-predict",
                "model_info": "/music/ml/deep-learning/model-info",
                "train": "/music/ml/deep-learning/train",
                "train_with_validation": "/music/ml/deep-learning/train-with-validation",
                "evaluate_advanced": "/music/ml/deep-learning/evaluate-advanced",
                "extract_embeddings": "/music/ml/deep-learning/extract-embeddings",
                "lyrics_analyze": "/music/ml/deep-learning/lyrics/analyze",
                "save_model": "/music/ml/deep-learning/save",
                "load_model": "/music/ml/deep-learning/load",
                "save_training_history": "/music/ml/deep-learning/save-training-history",
                "experiment_initialize": "/music/ml/deep-learning/experiment/initialize",
                "find_similar": "/music/ml/deep-learning/find-similar",
                "recommend_embeddings": "/music/ml/deep-learning/recommend-embeddings",
                "optimize_hyperparameters": "/music/ml/deep-learning/optimize-hyperparameters",
                "cluster_tracks": "/music/ml/deep-learning/cluster-tracks",
                "analyze_feature_importance": "/music/ml/deep-learning/analyze-feature-importance",
                "compare_models": "/music/ml/deep-learning/compare-models",
                "export_results": "/music/ml/deep-learning/export-results",
                "analyze_embedding_trends": "/music/ml/deep-learning/analyze-embedding-trends",
                "analyze_bias_fairness": "/music/ml/deep-learning/analyze-bias-fairness",
                "training_report": "/music/ml/deep-learning/training-report",
                "fine_tune": "/music/ml/deep-learning/fine-tune",
                "explain": "/music/ml/deep-learning/explain/{track_id}",
                "ab_test": "/music/ml/deep-learning/ab-test",
                "analyze_robustness": "/music/ml/deep-learning/analyze-robustness",
                "version": "/music/ml/deep-learning/version",
                "versions": "/music/ml/deep-learning/versions",
                "production_metrics": "/music/ml/deep-learning/production-metrics",
                "detect_drift": "/music/ml/deep-learning/detect-drift",
                "check_degradation": "/music/ml/deep-learning/check-degradation",
                "auto_retrain": "/music/ml/deep-learning/auto-retrain",
                "analyze_confidence": "/music/ml/deep-learning/analyze-confidence",
                "detect_outliers": "/music/ml/deep-learning/detect-outliers",
                "create_ensemble": "/music/ml/deep-learning/create-ensemble",
                "predict_ensemble": "/music/ml/deep-learning/predict-ensemble/{track_id}",
                "batch_advanced": "/music/ml/deep-learning/batch-advanced",
                "clear_cache": "/music/ml/deep-learning/clear-cache",
                "calibrate": "/music/ml/deep-learning/calibrate",
                "analyze_uncertainty": "/music/ml/deep-learning/analyze-uncertainty",
                "active_learning": "/music/ml/deep-learning/active-learning",
                "transfer_learning": "/music/ml/deep-learning/transfer-learning",
                "detect_adversarial": "/music/ml/deep-learning/detect-adversarial/{track_id}",
                "meta_learning": "/music/ml/deep-learning/meta-learning",
                "few_shot": "/music/ml/deep-learning/few-shot",
                "analyze_causality": "/music/ml/deep-learning/analyze-causality",
                "explain_advanced": "/music/ml/deep-learning/explain-advanced/{track_id}",
                "analyze_concepts": "/music/ml/deep-learning/analyze-concepts"
            },
            "trends": {
                "popularity": "/music/trends/popularity",
                "artists": "/music/trends/artists",
                "predict_success": "/music/predict/success",
                "rhythmic_patterns": "/music/rhythmic/patterns"
            },
            "collaborations": {
                "analyze": "/music/collaborations/analyze",
                "network": "/music/collaborations/network",
                "compare_versions": "/music/versions/compare"
            },
            "alerts": {
                "check": "/music/alerts/check",
                "list": "/music/alerts",
                "mark_read": "/music/alerts/{alert_id}/read",
                "delete": "/music/alerts/{alert_id}"
            },
            "temporal": {
                "structure": "/music/temporal/structure",
                "energy": "/music/temporal/energy",
                "tempo": "/music/temporal/tempo"
            },
            "quality": {
                "analyze": "/music/quality/analyze"
            },
            "contextual_recommendations": {
                "contextual": "/music/recommendations/contextual",
                "time_of_day": "/music/recommendations/time-of-day",
                "activity": "/music/recommendations/activity",
                "mood": "/music/recommendations/mood"
            },
            "playlist_analysis": {
                "analyze": "/music/playlists/analyze",
                "suggest_improvements": "/music/playlists/suggest-improvements",
                "optimize_order": "/music/playlists/optimize-order"
            },
            "artist_analysis": {
                "compare": "/music/artists/compare",
                "evolution": "/music/artists/evolution"
            },
            "discovery": {
                "similar_artists": "/music/discovery/similar-artists",
                "underground": "/music/discovery/underground",
                "mood_transition": "/music/discovery/mood-transition",
                "fresh": "/music/discovery/fresh"
            },
            "covers_remixes": {
                "analyze_cover": "/music/covers/analyze",
                "analyze_remix": "/music/remixes/analyze",
                "find": "/music/covers/find"
            },
            "gradio": "/gradio" if GRADIO_AVAILABLE else None,
            "docs": "/docs"
        }
    }


if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting Music Analyzer AI server on {settings.HOST}:{settings.PORT}")
    uvicorn.run(app, host=settings.HOST, port=settings.PORT)
