"""
API Routes Module
Aggregates all route modules into a single api_router
"""

from fastapi import APIRouter

api_router = APIRouter()

# Core analysis routers
try:
    from .analysis import router as analysis_router
    api_router.include_router(analysis_router, prefix="/analyze", tags=["Analysis"])
except ImportError:
    pass

try:
    from .similarity import router as similarity_router
    api_router.include_router(similarity_router, prefix="/similarity", tags=["Similarity"])
except ImportError:
    pass

try:
    from .quality import router as quality_router
    api_router.include_router(quality_router, prefix="/quality", tags=["Quality"])
except ImportError:
    pass

# System routers
try:
    from .health import router as health_router
    api_router.include_router(health_router, prefix="/health", tags=["Health"])
except ImportError:
    pass

try:
    from .metrics import router as metrics_router
    api_router.include_router(metrics_router, prefix="/metrics", tags=["Metrics"])
except ImportError:
    pass

try:
    from .stats import router as stats_router
    api_router.include_router(stats_router, prefix="/stats", tags=["System Stats"])
except ImportError:
    pass

try:
    from .cache import router as cache_router
    api_router.include_router(cache_router, prefix="/cache", tags=["Cache"])
except ImportError:
    pass

# AI/ML routers
try:
    from .ai_ml import router as ai_ml_router
    api_router.include_router(ai_ml_router, prefix="/ai", tags=["AI/ML Core"])
except ImportError:
    pass

try:
    from .ai_sentiment import router as ai_sentiment_router
    api_router.include_router(ai_sentiment_router, prefix="/ai/sentiment", tags=["AI - Sentiment"])
except ImportError:
    pass

try:
    from .ai_topics import router as ai_topics_router
    api_router.include_router(ai_topics_router, prefix="/ai/topics", tags=["AI - Topics"])
except ImportError:
    pass

try:
    from .ai_semantic import router as ai_semantic_router
    api_router.include_router(ai_semantic_router, prefix="/ai/semantic-similarity", tags=["AI - Semantic Similarity"])
except ImportError:
    pass

try:
    from .ai_plagiarism import router as ai_plagiarism_router
    api_router.include_router(ai_plagiarism_router, prefix="/ai/plagiarism", tags=["AI - Plagiarism"])
except ImportError:
    pass

try:
    from .ai_predict import router as ai_predict_router
    api_router.include_router(ai_predict_router, prefix="/ai/predict", tags=["AI - Prediction"])
except ImportError:
    pass

# Batch and export
try:
    from .batch import router as batch_router
    api_router.include_router(batch_router, prefix="/batch", tags=["Batch Processing"])
except ImportError:
    pass

try:
    from .export import router as export_router
    api_router.include_router(export_router, prefix="/export", tags=["Export"])
except ImportError:
    pass

# Webhooks
try:
    from .webhooks import router as webhooks_router
    api_router.include_router(webhooks_router, prefix="/webhooks", tags=["Webhooks"])
except ImportError:
    pass

try:
    from .webhooks_improved import router as webhooks_improved_router
    api_router.include_router(webhooks_improved_router, prefix="/webhooks", tags=["Webhooks Improved"])
except ImportError:
    pass

# Analytics
try:
    from .analytics import router as analytics_router
    api_router.include_router(analytics_router, prefix="/analytics", tags=["Analytics"])
except ImportError:
    pass

# Monitoring
try:
    from .monitoring import router as monitoring_router
    api_router.include_router(monitoring_router, prefix="/monitoring", tags=["Monitoring"])
except ImportError:
    pass

# Security
try:
    from .security import router as security_router
    api_router.include_router(security_router, prefix="/security", tags=["Security"])
except ImportError:
    pass

# Cloud
try:
    from .cloud import router as cloud_router
    api_router.include_router(cloud_router, prefix="/cloud", tags=["Cloud Integration"])
except ImportError:
    pass

# Automation
try:
    from .automation import router as automation_router
    api_router.include_router(automation_router, prefix="/automation", tags=["Automation"])
except ImportError:
    pass

# Training
try:
    from .training import router as training_router
    api_router.include_router(training_router, prefix="/training", tags=["AI - Training"])
except ImportError:
    pass

# Multimodal
try:
    from .multimodal import router as multimodal_router
    api_router.include_router(multimodal_router, prefix="/multimodal", tags=["Multimodal"])
except ImportError:
    pass

# Realtime
try:
    from .realtime import router as realtime_router
    api_router.include_router(realtime_router, prefix="/realtime", tags=["Real-time"])
except ImportError:
    pass

# Root
try:
    from .root import router as root_router
    api_router.include_router(root_router, tags=["Root"])
except ImportError:
    pass

# Policy (optional)
try:
    from .policy import router as policy_router
    api_router.include_router(policy_router, prefix="/policy", tags=["Policy"])
except ImportError:
    pass

__all__ = ["api_router"]
