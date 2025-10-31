"""
Gamma App - AI-Powered Content Generation System
Advanced presentation, document, and web page creation with AI
"""

__version__ = "1.0.0"
__author__ = "Gamma App Team"
__email__ = "team@gammaapp.com"
__description__ = "AI-Powered Content Generation System"

# Core imports
from .core.content_generator import ContentGenerator
from .core.design_engine import DesignEngine
from .core.collaboration_engine import CollaborationEngine

# Engine imports
from .engines.presentation_engine import PresentationEngine
from .engines.document_engine import DocumentEngine
from .engines.web_page_engine import WebPageEngine
from .engines.ai_models_engine import AIModelsEngine
from .engines.export_engine import AdvancedExportEngine

# Service imports
from .services.cache_service import AdvancedCacheService, cache_service
from .services.security_service import AdvancedSecurityService, security_service
from .services.performance_service import PerformanceService, performance_service
from .services.analytics_service import AnalyticsService
from .services.collaboration_service import CollaborationService
from .services.health_service import HealthService, health_service

# Utility imports
from .utils.config import get_settings, get_ai_config, get_security_config
from .utils.auth import create_access_token, verify_token, hash_password
from .utils.logging_config import setup_logging, get_logger

# Database imports
from .models.database import Base, init_database, get_db

# CLI imports
from .cli.main import app as cli_app

__all__ = [
    # Core
    "ContentGenerator",
    "DesignEngine", 
    "CollaborationEngine",
    
    # Engines
    "PresentationEngine",
    "DocumentEngine",
    "WebPageEngine",
    "AIModelsEngine",
    "AdvancedExportEngine",
    
    # Services
    "AdvancedCacheService",
    "cache_service",
    "AdvancedSecurityService", 
    "security_service",
    "PerformanceService",
    "performance_service",
    "AnalyticsService",
    "CollaborationService",
    "HealthService",
    "health_service",
    
    # Utils
    "get_settings",
    "get_ai_config",
    "get_security_config",
    "create_access_token",
    "verify_token",
    "hash_password",
    "setup_logging",
    "get_logger",
    
    # Database
    "Base",
    "init_database",
    "get_db",
    
    # CLI
    "cli_app",
    
    # Metadata
    "__version__",
    "__author__",
    "__email__",
    "__description__"
]

# Main application class
class GammaApp:
    """
    Main Gamma App application class
    """
    
    def __init__(self, config: dict = None):
        """Initialize Gamma App"""
        self.config = config or {}
        self.content_generator = None
        self.design_engine = None
        self.collaboration_engine = None
        self.presentation_engine = None
        self.document_engine = None
        self.web_page_engine = None
        self.ai_models_engine = None
        self.export_engine = None
        self.cache_service = None
        self.security_service = None
        self.performance_service = None
        self.analytics_service = None
        self.collaboration_service = None
        self.health_service = None
        
        self._initialized = False
    
    async def initialize(self):
        """Initialize all components"""
        if self._initialized:
            return
        
        try:
            # Initialize core components
            self.content_generator = ContentGenerator(self.config)
            self.design_engine = DesignEngine(self.config)
            self.collaboration_engine = CollaborationEngine(self.config)
            
            # Initialize engines
            self.presentation_engine = PresentationEngine(self.config)
            self.document_engine = DocumentEngine(self.config)
            self.web_page_engine = WebPageEngine(self.config)
            self.ai_models_engine = AIModelsEngine(self.config)
            self.export_engine = AdvancedExportEngine(self.config)
            
            # Initialize services
            self.cache_service = AdvancedCacheService(self.config)
            self.security_service = AdvancedSecurityService(self.config)
            self.performance_service = PerformanceService(self.config)
            self.analytics_service = AnalyticsService(self.config)
            self.collaboration_service = CollaborationService(self.config)
            self.health_service = HealthService(self.config)
            
            self._initialized = True
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Gamma App: {e}")
    
    async def generate_presentation(self, topic: str, slides: int = 10, 
                                  style: str = "modern", **kwargs):
        """Generate a presentation"""
        if not self._initialized:
            await self.initialize()
        
        return await self.presentation_engine.generate_presentation(
            topic=topic,
            slides=slides,
            style=style,
            **kwargs
        )
    
    async def generate_document(self, title: str, content_type: str = "report",
                              sections: list = None, **kwargs):
        """Generate a document"""
        if not self._initialized:
            await self.initialize()
        
        return await self.document_engine.generate_document(
            title=title,
            content_type=content_type,
            sections=sections or [],
            **kwargs
        )
    
    async def generate_web_page(self, title: str, page_type: str = "landing_page",
                              sections: list = None, **kwargs):
        """Generate a web page"""
        if not self._initialized:
            await self.initialize()
        
        return await self.web_page_engine.create_web_page(
            content={
                "title": title,
                "sections": sections or []
            },
            page_type=page_type,
            **kwargs
        )
    
    async def export_content(self, content, format: str = "pdf", **kwargs):
        """Export content to specified format"""
        if not self._initialized:
            await self.initialize()
        
        return await self.export_engine.export_content(
            content=content,
            export_config=kwargs,
            format=format
        )
    
    async def get_health_status(self):
        """Get system health status"""
        if not self._initialized:
            await self.initialize()
        
        return await self.health_service.get_system_health()
    
    async def close(self):
        """Close all components"""
        if not self._initialized:
            return
        
        try:
            if self.cache_service:
                await self.cache_service.close()
            if self.security_service:
                await self.security_service.close()
            if self.performance_service:
                self.performance_service.stop_monitoring()
            if self.ai_models_engine:
                await self.ai_models_engine.cleanup()
        except Exception as e:
            print(f"Error closing components: {e}")
        
        self._initialized = False

# Convenience function for quick initialization
async def create_app(config: dict = None) -> GammaApp:
    """Create and initialize Gamma App instance"""
    app = GammaApp(config)
    await app.initialize()
    return app

# Version information
VERSION_INFO = {
    "version": __version__,
    "author": __author__,
    "email": __email__,
    "description": __description__,
    "features": [
        "AI-Powered Content Generation",
        "Advanced Export Options", 
        "Real-time Collaboration",
        "Enterprise Security",
        "Performance Monitoring",
        "Multi-level Caching",
        "Health Monitoring",
        "CLI Interface"
    ]
}

def get_version_info():
    """Get version information"""
    return VERSION_INFO.copy()

























