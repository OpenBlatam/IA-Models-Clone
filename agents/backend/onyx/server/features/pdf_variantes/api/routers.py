"""
PDF Variantes API - Router Configuration
Centralized router registration
"""

from fastapi import FastAPI

from .routes import (
    pdf_router,
    variant_router,
    topic_router,
    brainstorm_router,
    collaboration_router,
    export_router,
    analytics_router,
    search_router,
    batch_router,
    health_router,
)


def register_routers(app: FastAPI) -> None:
    """Register all routers with the application"""
    
    # Core PDF operations
    app.include_router(
        pdf_router,
        prefix="/api/v1/pdf",
        tags=["PDF Processing"]
    )
    
    # Variant generation
    app.include_router(
        variant_router,
        prefix="/api/v1/variants",
        tags=["Variant Generation"]
    )
    
    # Topic extraction
    app.include_router(
        topic_router,
        prefix="/api/v1/topics",
        tags=["Topic Extraction"]
    )
    
    # Brainstorming
    app.include_router(
        brainstorm_router,
        prefix="/api/v1/brainstorm",
        tags=["Brainstorming"]
    )
    
    # Collaboration
    app.include_router(
        collaboration_router,
        prefix="/api/v1/collaboration",
        tags=["Collaboration"]
    )
    
    # Export operations
    app.include_router(
        export_router,
        prefix="/api/v1/export",
        tags=["Export"]
    )
    
    # Analytics
    app.include_router(
        analytics_router,
        prefix="/api/v1/analytics",
        tags=["Analytics"]
    )
    
    # Search
    app.include_router(
        search_router,
        prefix="/api/v1/search",
        tags=["Search"]
    )
    
    # Batch processing
    app.include_router(
        batch_router,
        prefix="/api/v1/batch",
        tags=["Batch Processing"]
    )
    
    # Health and monitoring
    app.include_router(
        health_router,
        prefix="/api/v1/health",
        tags=["Health"]
    )
    
    # Enhanced routers with advanced features
    try:
        from .enhanced_routes import (
            enhanced_pdf_router,
            enhanced_variant_router,
            stats_router
        )
        
        app.include_router(enhanced_pdf_router)
        app.include_router(enhanced_variant_router)
        app.include_router(stats_router)
    except ImportError:
        # Enhanced routes are optional
        pass
    
    # Fast routers with performance optimizations
    try:
        from .fast_routes import fast_router
        app.include_router(fast_router)
    except ImportError:
        # Fast routes are optional
        pass
    
    # Robust routers with resilience patterns
    try:
        from .robust_routes import robust_router
        app.include_router(robust_router)
    except ImportError:
        # Robust routes are optional
        pass
    
    # Modular architecture routers
    try:
        from .modules.integration import setup_modules
        # Auto-discover and register all modules
        module_manager = setup_modules(app)
        # Store in app state for access
        app.state.module_manager = module_manager
    except ImportError:
        # Module system is optional
        pass
    
    # Ultra-fast optimized routes
    try:
        from .optimization.fast_routes_optimized import ultra_fast_router
        app.include_router(ultra_fast_router)
    except ImportError:
        # Ultra-fast routes are optional
        pass

