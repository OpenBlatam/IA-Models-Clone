"""
MANS Unified System - Main Application Entry Point

This is the main entry point for the refactored MANS (Más Avanzadas y Nuevas) system,
integrating all advanced AI and space technology capabilities under a unified architecture.

Features:
- Advanced AI with neural networks and deep learning
- Generative AI and large language models
- Computer vision and image processing
- Natural language processing
- Reinforcement learning and transfer learning
- Federated learning and explainable AI
- AI ethics and safety
- Space technology with satellite communication
- Space-based data processing and orbital mechanics
- Space weather monitoring and debris tracking
- Interplanetary networking and space-based AI
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from datetime import datetime

from .unified_mans_config import UnifiedMANSConfig
from .unified_mans_manager import initialize_mans_system, shutdown_mans_system, get_unified_mans_manager
from .unified_mans_api import unified_mans_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global configuration
config = UnifiedMANSConfig()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("Starting MANS Unified System...")
    try:
        await initialize_mans_system()
        logger.info("MANS system initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize MANS system: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down MANS Unified System...")
    try:
        await shutdown_mans_system()
        logger.info("MANS system shut down successfully")
    except Exception as e:
        logger.error(f"Error during MANS system shutdown: {e}")

# Create FastAPI application
app = FastAPI(
    title="MANS - Unified Advanced Technology System",
    description="""
    ## MANS - Más Avanzadas y Nuevas (More Advanced and New)
    
    A unified system integrating cutting-edge AI and space technology capabilities:
    
    ### Advanced AI Technologies
    - **Neural Networks**: Advanced deep learning architectures
    - **Generative AI**: Large language models and content generation
    - **Computer Vision**: Image processing and analysis
    - **Natural Language Processing**: Text understanding and generation
    - **Reinforcement Learning**: Adaptive learning algorithms
    - **Transfer Learning**: Domain adaptation and knowledge transfer
    - **Federated Learning**: Distributed training and privacy-preserving ML
    - **Explainable AI**: Interpretable and transparent AI systems
    - **AI Ethics**: Fairness, transparency, and responsible AI
    - **AI Safety**: Robustness, alignment, and safety measures
    
    ### Space Technology Systems
    - **Satellite Communication**: Orbital communication systems
    - **Space Weather**: Monitoring and prediction of space conditions
    - **Space Debris**: Tracking and collision avoidance
    - **Interplanetary Networking**: Deep space communication protocols
    
    ### Key Features
    - **Unified Architecture**: Single system integrating all technologies
    - **High Performance**: Optimized for speed and efficiency
    - **Scalable**: Designed to handle enterprise-level workloads
    - **Secure**: Built-in security and privacy protections
    - **Observable**: Comprehensive monitoring and metrics
    - **Extensible**: Easy to add new technologies and capabilities
    """,
    version="1.0.0",
    contact={
        "name": "MANS Development Team",
        "email": "mans-dev@example.com",
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT",
    },
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if config.debug_mode else ["https://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"] if config.debug_mode else ["localhost", "127.0.0.1"]
)

# Custom middleware for request logging
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests"""
    start_time = datetime.utcnow()
    
    # Log request
    logger.info(f"Request: {request.method} {request.url}")
    
    # Process request
    response = await call_next(request)
    
    # Log response
    process_time = (datetime.utcnow() - start_time).total_seconds()
    logger.info(f"Response: {response.status_code} - {process_time:.3f}s")
    
    return response

# Custom middleware for error handling
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": str(exc) if config.debug_mode else "An unexpected error occurred",
            "timestamp": datetime.utcnow().isoformat()
        }
    )

# Include routers
app.include_router(unified_mans_router)

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "MANS - Unified Advanced Technology System",
        "description": "Advanced AI and Space Technology Integration",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat(),
        "docs_url": "/docs",
        "redoc_url": "/redoc",
        "api_prefix": config.api_prefix,
        "environment": config.environment.value,
        "debug_mode": config.debug_mode,
        "features": {
            "advanced_ai": config.advanced_ai.enabled,
            "space_technology": config.space_technology.enabled,
            "neural_networks": config.neural_network.enabled,
            "generative_ai": config.generative_ai.enabled,
            "computer_vision": config.computer_vision.enabled,
            "nlp": config.nlp.enabled,
            "reinforcement_learning": config.reinforcement_learning.enabled,
            "transfer_learning": config.transfer_learning.enabled,
            "federated_learning": config.federated_learning.enabled,
            "explainable_ai": config.explainable_ai.enabled,
            "ai_ethics": config.ai_ethics.enabled,
            "ai_safety": config.ai_safety.enabled,
            "satellite_communication": config.satellite_communication.enabled,
            "space_weather": config.space_weather.enabled,
            "space_debris": config.space_debris.enabled,
            "interplanetary_networking": config.interplanetary_networking.enabled
        }
    }

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        mans_manager = get_unified_mans_manager()
        status = mans_manager.get_system_status()
        
        return {
            "status": "healthy" if status["initialized"] else "unhealthy",
            "timestamp": datetime.utcnow().isoformat(),
            "mans_system": {
                "initialized": status["initialized"],
                "active_services": status["services"],
                "metrics": status["metrics"]
            },
            "config": {
                "environment": config.environment.value,
                "debug_mode": config.debug_mode,
                "enabled_features": config.get_all_enabled_features()
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(e) if config.debug_mode else "System health check failed"
        }

# Metrics endpoint
@app.get("/metrics")
async def get_metrics():
    """Get system metrics"""
    try:
        mans_manager = get_unified_mans_manager()
        status = mans_manager.get_system_status()
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "system_metrics": status["metrics"],
            "service_status": status["services"],
            "configuration": {
                "total_features": 16,
                "enabled_features": len(config.get_all_enabled_features()),
                "environment": config.environment.value
            }
        }
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        return {
            "error": "Failed to retrieve metrics",
            "timestamp": datetime.utcnow().isoformat()
        }

# Configuration endpoint
@app.get("/config")
async def get_configuration():
    """Get system configuration"""
    try:
        return config.get_system_summary()
    except Exception as e:
        logger.error(f"Error getting configuration: {e}")
        return {
            "error": "Failed to retrieve configuration",
            "timestamp": datetime.utcnow().isoformat()
        }

# Features endpoint
@app.get("/features")
async def get_features():
    """Get available features"""
    try:
        enabled_features = config.get_all_enabled_features()
        
        return {
            "enabled_features": enabled_features,
            "total_features": 16,
            "enabled_count": len(enabled_features),
            "feature_categories": {
                "advanced_ai": [
                    "neural_networks", "generative_ai", "computer_vision",
                    "nlp", "reinforcement_learning", "transfer_learning",
                    "federated_learning", "explainable_ai", "ai_ethics", "ai_safety"
                ],
                "space_technology": [
                    "satellite_communication", "space_weather", "space_debris",
                    "interplanetary_networking"
                ]
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting features: {e}")
        return {
            "error": "Failed to retrieve features",
            "timestamp": datetime.utcnow().isoformat()
        }

# Custom OpenAPI schema
def custom_openapi():
    """Custom OpenAPI schema"""
    if app.openapi_schema:
        return app.openapi_schema
    
    from fastapi.openapi.utils import get_openapi
    
    openapi_schema = get_openapi(
        title="MANS - Unified Advanced Technology System",
        version="1.0.0",
        description=app.description,
        routes=app.routes,
    )
    
    # Add custom tags
    openapi_schema["tags"] = [
        {
            "name": "MANS - Unified Advanced Technology",
            "description": "Advanced AI and Space Technology Integration"
        },
        {
            "name": "Advanced AI",
            "description": "Neural Networks, Generative AI, Computer Vision, NLP, and more"
        },
        {
            "name": "Space Technology",
            "description": "Satellite Communication, Space Weather, Debris Tracking, and more"
        }
    ]
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

# Main execution
if __name__ == "__main__":
    logger.info("Starting MANS Unified System server...")
    
    # Run the application
    uvicorn.run(
        "refactored_mans_system.main:app",
        host="0.0.0.0",
        port=8000,
        reload=config.debug_mode,
        log_level=config.log_level.value.lower(),
        access_log=True
    )





















