"""
API v1 router
"""

from fastapi import APIRouter

from .endpoints import blog_posts, users, comments, health, files, ai, websocket, recommendations, admin, blockchain, quantum, ml_pipeline, advanced_analytics, workflows, security, monitoring, integrations, advanced_caching, notifications, advanced_search, content_management, media, social, ecommerce, gamification, learning, ai_assistant, automation, iot, ar_vr, metaverse

api_router = APIRouter()

# Include endpoint routers
api_router.include_router(
    blog_posts.router,
    prefix="/blog-posts",
    tags=["Blog Posts"]
)

api_router.include_router(
    users.router,
    prefix="/users",
    tags=["Users"]
)

api_router.include_router(
    comments.router,
    prefix="/comments",
    tags=["Comments"]
)

api_router.include_router(
    files.router,
    prefix="/files",
    tags=["File Management"]
)

api_router.include_router(
    ai.router,
    prefix="/ai",
    tags=["AI Features"]
)

api_router.include_router(
    websocket.router,
    prefix="/ws",
    tags=["WebSocket"]
)

api_router.include_router(
    recommendations.router,
    prefix="/recommendations",
    tags=["Recommendations"]
)

api_router.include_router(
    admin.router,
    prefix="/admin",
    tags=["Admin"]
)

api_router.include_router(
    blockchain.router,
    prefix="/blockchain",
    tags=["Blockchain"]
)

api_router.include_router(
    quantum.router,
    prefix="/quantum",
    tags=["Quantum Computing"]
)

api_router.include_router(
    ml_pipeline.router,
    prefix="/ml",
    tags=["Machine Learning"]
)

api_router.include_router(
    advanced_analytics.router,
    prefix="/analytics",
    tags=["Advanced Analytics"]
)

api_router.include_router(
    workflows.router,
    prefix="/workflows",
    tags=["Workflows"]
)

api_router.include_router(
    security.router,
    prefix="/security",
    tags=["Security"]
)

api_router.include_router(
    monitoring.router,
    prefix="/monitoring",
    tags=["Monitoring"]
)

api_router.include_router(
    integrations.router,
    prefix="/integrations",
    tags=["Integrations"]
)

api_router.include_router(
    advanced_caching.router,
    prefix="/cache",
    tags=["Advanced Caching"]
)

api_router.include_router(
    notifications.router,
    prefix="/notifications",
    tags=["Notifications"]
)

api_router.include_router(
    advanced_search.router,
    prefix="/search",
    tags=["Advanced Search"]
)

api_router.include_router(
    content_management.router,
    prefix="/content",
    tags=["Content Management"]
)

api_router.include_router(
    media.router,
    prefix="/media",
    tags=["Media Management"]
)

api_router.include_router(
    social.router,
    prefix="/social",
    tags=["Social Features"]
)

api_router.include_router(
    ecommerce.router,
    prefix="/ecommerce",
    tags=["E-commerce"]
)

api_router.include_router(
    gamification.router,
    prefix="/gamification",
    tags=["Gamification"]
)

api_router.include_router(
    learning.router,
    prefix="/learning",
    tags=["Learning & Education"]
)

api_router.include_router(
    ai_assistant.router,
    prefix="/ai-assistant",
    tags=["AI Assistant"]
)

api_router.include_router(
    automation.router,
    prefix="/automation",
    tags=["Advanced Automation"]
)

api_router.include_router(
    iot.router,
    prefix="/iot",
    tags=["IoT Management"]
)

api_router.include_router(
    ar_vr.router,
    prefix="/ar-vr",
    tags=["AR/VR Management"]
)

api_router.include_router(
    metaverse.router,
    prefix="/metaverse",
    tags=["Metaverse Management"]
)

api_router.include_router(
    health.router,
    prefix="/health",
    tags=["Health"]
)
