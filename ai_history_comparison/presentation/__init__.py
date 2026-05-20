"""
Presentation Module - Capa de Presentación
Módulo de presentación con APIs, CLI y interfaces web
"""

from .api import (
    create_app,
    create_router,
    setup_middleware,
    setup_routes
)

from .api.v1 import (
    content_router,
    analysis_router,
    comparison_router,
    report_router,
    system_router
)

from .api.v2 import (
    advanced_router,
    ml_router
)

from .api.websocket import (
    realtime_router,
    streaming_router
)

from .cli import (
    CLIApp,
    ContentCommands,
    AnalysisCommands,
    SystemCommands
)

__all__ = [
    # API Factory
    "create_app",
    "create_router", 
    "setup_middleware",
    "setup_routes",
    
    # API v1
    "content_router",
    "analysis_router",
    "comparison_router",
    "report_router",
    "system_router",
    
    # API v2
    "advanced_router",
    "ml_router",
    
    # WebSocket
    "realtime_router",
    "streaming_router",
    
    # CLI
    "CLIApp",
    "ContentCommands",
    "AnalysisCommands",
    "SystemCommands"
]







