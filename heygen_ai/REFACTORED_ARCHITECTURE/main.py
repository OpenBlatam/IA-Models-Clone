"""
HeyGen AI - Refactored Main Application

This module serves as the main entry point for the refactored HeyGen AI application.
It orchestrates the different layers and components of the application.
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware

from .domain.entities.ai_model import AIModel, ModelType, ModelStatus
from .domain.services.ai_model_service import AIModelService
from .domain.repositories.base_repository import BaseRepository
from .infrastructure.repositories.ai_model_repository_impl import AIModelRepositoryImpl
from .application.use_cases.ai_model_use_cases import (
    CreateModelUseCase,
    GetModelUseCase,
    ListModelsUseCase,
    UpdateModelUseCase,
    DeleteModelUseCase,
    TrainModelUseCase,
    CompleteTrainingUseCase,
    DeployModelUseCase,
    ArchiveModelUseCase,
    SearchModelsUseCase,
    GetModelVersionsUseCase,
    GetLatestModelVersionUseCase
)
from .presentation.controllers.ai_model_controller import AIModelController
from .infrastructure.database import DatabaseManager
from .infrastructure.config import Settings


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Application:
    """Main application class that orchestrates all components."""
    
    def __init__(self):
        """Initialize the application."""
        self.settings = Settings()
        self.database_manager = None
        self.model_repository = None
        self.model_service = None
        self.use_cases = {}
        self.controllers = {}
        self.app = None
    
    async def initialize(self):
        """Initialize all application components."""
        try:
            logger.info("🚀 Initializing HeyGen AI Refactored Application...")
            
            # Initialize database
            self.database_manager = DatabaseManager(self.settings.database_url)
            await self.database_manager.connect()
            
            # Initialize repositories
            self.model_repository = AIModelRepositoryImpl(self.database_manager.connection)
            
            # Initialize services
            self.model_service = AIModelService(self.model_repository)
            
            # Initialize use cases
            self.use_cases = {
                'create_model': CreateModelUseCase(self.model_service),
                'get_model': GetModelUseCase(self.model_service),
                'list_models': ListModelsUseCase(self.model_service),
                'update_model': UpdateModelUseCase(self.model_service),
                'delete_model': DeleteModelUseCase(self.model_service),
                'train_model': TrainModelUseCase(self.model_service),
                'complete_training': CompleteTrainingUseCase(self.model_service),
                'deploy_model': DeployModelUseCase(self.model_service),
                'archive_model': ArchiveModelUseCase(self.model_service),
                'search_models': SearchModelsUseCase(self.model_service),
                'get_model_versions': GetModelVersionsUseCase(self.model_service),
                'get_latest_model_version': GetLatestModelVersionUseCase(self.model_service)
            }
            
            # Initialize controllers
            self.controllers = {
                'ai_model': AIModelController(**self.use_cases)
            }
            
            # Initialize FastAPI app
            self.app = self._create_fastapi_app()
            
            logger.info("✅ Application initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize application: {e}")
            raise
    
    def _create_fastapi_app(self) -> FastAPI:
        """Create and configure the FastAPI application."""
        
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            """Application lifespan manager."""
            # Startup
            await self.initialize()
            yield
            # Shutdown
            await self.cleanup()
        
        app = FastAPI(
            title="HeyGen AI - Refactored API",
            description="Refactored HeyGen AI system with clean architecture",
            version="2.0.0",
            lifespan=lifespan
        )
        
        # Add middleware
        app.add_middleware(
            CORSMiddleware,
            allow_origins=self.settings.allowed_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=self.settings.allowed_hosts
        )
        
        # Include routers
        app.include_router(self.controllers['ai_model'].router)
        
        # Add health check endpoint
        @app.get("/health")
        async def health_check():
            """Health check endpoint."""
            return {
                "status": "healthy",
                "version": "2.0.0",
                "database": "connected" if self.database_manager else "disconnected"
            }
        
        return app
    
    async def cleanup(self):
        """Cleanup application resources."""
        try:
            logger.info("🧹 Cleaning up application resources...")
            
            if self.database_manager:
                await self.database_manager.disconnect()
            
            logger.info("✅ Application cleanup completed")
            
        except Exception as e:
            logger.error(f"❌ Error during cleanup: {e}")
    
    async def run(self, host: str = "0.0.0.0", port: int = 8000):
        """Run the application."""
        if not self.app:
            await self.initialize()
        
        import uvicorn
        uvicorn.run(self.app, host=host, port=port)


# Global application instance
app_instance = Application()


async def main():
    """Main entry point."""
    try:
        await app_instance.run()
    except KeyboardInterrupt:
        logger.info("🛑 Application stopped by user")
    except Exception as e:
        logger.error(f"❌ Application error: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())