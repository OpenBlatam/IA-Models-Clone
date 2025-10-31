"""
Unified API System

This module provides a unified FastAPI application that integrates all advanced
features including quantum computing, blockchain, IoT, AR/VR, edge computing,
and performance optimizations into a single, cohesive API.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

from .unified_config import UnifiedConfig, get_config
from .unified_manager import UnifiedSystemManager, get_unified_manager

logger = logging.getLogger(__name__)

# Pydantic models
class RequestModel(BaseModel):
    """Unified request model"""
    service_type: str = Field(..., description="Type of service to use")
    operation: str = Field(..., description="Operation to perform")
    data: Dict[str, Any] = Field(default_factory=dict, description="Request data")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Request metadata")

class ResponseModel(BaseModel):
    """Unified response model"""
    success: bool = Field(..., description="Whether the request was successful")
    result: Optional[Any] = Field(None, description="Response result")
    error: Optional[str] = Field(None, description="Error message if any")
    service_type: str = Field(..., description="Service type used")
    timestamp: str = Field(..., description="Response timestamp")
    processing_time_ms: Optional[float] = Field(None, description="Processing time in milliseconds")

class SystemStatusModel(BaseModel):
    """System status model"""
    initialized: bool
    uptime_seconds: float
    total_requests: int
    active_connections: int
    error_count: int
    last_health_check: Optional[str]
    systems: Dict[str, bool]
    configuration: Dict[str, Any]
    metadata: Dict[str, Any]

class FeatureStatusModel(BaseModel):
    """Feature status model"""
    quantum_computing: bool
    blockchain_integration: bool
    iot_integration: bool
    ar_vr_support: bool
    edge_computing: bool
    performance_optimization: bool
    advanced_security: bool
    real_time_monitoring: bool
    ai_ml_enhancement: bool

# Global variables
config: UnifiedConfig = None
unified_manager: UnifiedSystemManager = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global config, unified_manager
    
    # Startup
    logger.info("Starting Unified AI History Comparison System...")
    
    try:
        # Initialize configuration
        config = get_config()
        logger.info("Configuration loaded successfully")
        
        # Initialize unified manager
        unified_manager = get_unified_manager()
        success = await unified_manager.initialize()
        
        if not success:
            raise Exception("Failed to initialize unified system")
        
        logger.info("Unified system initialized successfully")
        
        yield
        
    except Exception as e:
        logger.error(f"Failed to start unified system: {e}")
        raise
    
    finally:
        # Shutdown
        logger.info("Shutting down Unified AI History Comparison System...")
        
        if unified_manager:
            await unified_manager.shutdown()
            logger.info("Unified system shut down successfully")

# Create FastAPI application
app = FastAPI(
    title="Unified AI History Comparison System",
    description="""
    A comprehensive AI History Comparison System that integrates advanced features including:
    
    - **Quantum Computing**: Advanced quantum algorithms and machine learning
    - **Blockchain Integration**: Multi-chain support with DeFi and NFT capabilities
    - **IoT Integration**: Multi-protocol device management and real-time processing
    - **AR/VR Support**: Immersive 3D visualization and interaction
    - **Edge Computing**: Distributed processing and edge AI/ML
    - **Performance Optimization**: Advanced optimization techniques
    - **Advanced Security**: Multi-layer security and compliance
    - **Real-time Monitoring**: Comprehensive observability and analytics
    - **AI/ML Enhancement**: Advanced AI models and machine learning
    
    This unified system provides a single API interface for all advanced capabilities.
    """,
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.cors_origins if config else ["*"],
    allow_credentials=True,
    allow_methods=config.cors_methods if config else ["*"],
    allow_headers=config.cors_headers if config else ["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # Configure appropriately for production
)

# Dependency functions
async def get_unified_manager_dependency() -> UnifiedSystemManager:
    """Get unified manager dependency"""
    if unified_manager is None:
        raise HTTPException(status_code=503, detail="Unified system not initialized")
    return unified_manager

async def get_config_dependency() -> UnifiedConfig:
    """Get configuration dependency"""
    if config is None:
        raise HTTPException(status_code=503, detail="Configuration not loaded")
    return config

# API Routes

@app.get("/", response_model=Dict[str, Any])
async def root():
    """Root endpoint with system information"""
    return {
        "name": "Unified AI History Comparison System",
        "version": "1.0.0",
        "description": "Advanced AI system with quantum computing, blockchain, IoT, AR/VR, and edge computing",
        "features": [
            "Quantum Computing",
            "Blockchain Integration", 
            "IoT Integration",
            "AR/VR Support",
            "Edge Computing",
            "Performance Optimization",
            "Advanced Security",
            "Real-time Monitoring",
            "AI/ML Enhancement"
        ],
        "endpoints": {
            "health": "/health",
            "status": "/status",
            "features": "/features",
            "process": "/process",
            "docs": "/docs",
            "redoc": "/redoc"
        },
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/health", response_model=Dict[str, Any])
async def health_check():
    """Health check endpoint"""
    try:
        if unified_manager is None:
            return {
                "status": "unhealthy",
                "message": "Unified system not initialized",
                "timestamp": datetime.utcnow().isoformat()
            }
        
        system_status = unified_manager.get_system_status()
        
        return {
            "status": "healthy" if system_status["initialized"] else "unhealthy",
            "uptime_seconds": system_status["uptime_seconds"],
            "systems": system_status["systems"],
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

@app.get("/status", response_model=SystemStatusModel)
async def get_system_status(manager: UnifiedSystemManager = Depends(get_unified_manager_dependency)):
    """Get comprehensive system status"""
    try:
        status = manager.get_system_status()
        return SystemStatusModel(**status)
    except Exception as e:
        logger.error(f"Failed to get system status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/features", response_model=FeatureStatusModel)
async def get_feature_status(manager: UnifiedSystemManager = Depends(get_unified_manager_dependency)):
    """Get feature status"""
    try:
        features = manager.get_feature_status()
        return FeatureStatusModel(**features)
    except Exception as e:
        logger.error(f"Failed to get feature status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process", response_model=ResponseModel)
async def process_request(
    request: RequestModel,
    background_tasks: BackgroundTasks,
    manager: UnifiedSystemManager = Depends(get_unified_manager_dependency)
):
    """Process unified request"""
    try:
        start_time = datetime.utcnow()
        
        # Convert request to dict
        request_data = {
            "service_type": request.service_type,
            "operation": request.operation,
            "data": request.data,
            "metadata": request.metadata
        }
        
        # Process request
        result = await manager.process_request(request_data)
        
        # Calculate processing time
        processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        # Add processing time to result
        result["processing_time_ms"] = processing_time
        
        return ResponseModel(**result)
        
    except Exception as e:
        logger.error(f"Failed to process request: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Service-specific endpoints

@app.post("/quantum/algorithm")
async def run_quantum_algorithm(
    algorithm: str,
    data: Dict[str, Any],
    manager: UnifiedSystemManager = Depends(get_unified_manager_dependency)
):
    """Run quantum algorithm"""
    try:
        request_data = {
            "service_type": "quantum",
            "operation": "run_algorithm",
            "data": {"algorithm": algorithm, **data}
        }
        
        result = await manager.process_request(request_data)
        return result
        
    except Exception as e:
        logger.error(f"Failed to run quantum algorithm: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/blockchain/account")
async def create_blockchain_account(
    blockchain_type: str,
    manager: UnifiedSystemManager = Depends(get_unified_manager_dependency)
):
    """Create blockchain account"""
    try:
        request_data = {
            "service_type": "blockchain",
            "operation": "create_account",
            "data": {"blockchain_type": blockchain_type}
        }
        
        result = await manager.process_request(request_data)
        return result
        
    except Exception as e:
        logger.error(f"Failed to create blockchain account: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/iot/device")
async def register_iot_device(
    name: str,
    device_type: str,
    protocol: str,
    manager: UnifiedSystemManager = Depends(get_unified_manager_dependency)
):
    """Register IoT device"""
    try:
        request_data = {
            "service_type": "iot",
            "operation": "register_device",
            "data": {
                "name": name,
                "device_type": device_type,
                "protocol": protocol
            }
        }
        
        result = await manager.process_request(request_data)
        return result
        
    except Exception as e:
        logger.error(f"Failed to register IoT device: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ar-vr/scene")
async def create_ar_vr_scene(
    name: str,
    scene_type: str,
    manager: UnifiedSystemManager = Depends(get_unified_manager_dependency)
):
    """Create AR/VR scene"""
    try:
        request_data = {
            "service_type": "ar_vr",
            "operation": "create_scene",
            "data": {
                "name": name,
                "scene_type": scene_type
            }
        }
        
        result = await manager.process_request(request_data)
        return result
        
    except Exception as e:
        logger.error(f"Failed to create AR/VR scene: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/edge/node")
async def register_edge_node(
    name: str,
    node_type: str,
    location: Dict[str, float],
    manager: UnifiedSystemManager = Depends(get_unified_manager_dependency)
):
    """Register edge node"""
    try:
        request_data = {
            "service_type": "edge",
            "operation": "register_node",
            "data": {
                "name": name,
                "node_type": node_type,
                "location": location
            }
        }
        
        result = await manager.process_request(request_data)
        return result
        
    except Exception as e:
        logger.error(f"Failed to register edge node: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/performance/optimize")
async def optimize_performance(
    manager: UnifiedSystemManager = Depends(get_unified_manager_dependency)
):
    """Optimize system performance"""
    try:
        request_data = {
            "service_type": "performance",
            "operation": "optimize"
        }
        
        result = await manager.process_request(request_data)
        return result
        
    except Exception as e:
        logger.error(f"Failed to optimize performance: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Feature management endpoints

@app.post("/features/{feature_name}/enable")
async def enable_feature(
    feature_name: str,
    manager: UnifiedSystemManager = Depends(get_unified_manager_dependency)
):
    """Enable a feature"""
    try:
        success = await manager.enable_feature(feature_name)
        
        if success:
            return {"success": True, "message": f"Feature {feature_name} enabled"}
        else:
            raise HTTPException(status_code=400, detail=f"Failed to enable feature {feature_name}")
            
    except Exception as e:
        logger.error(f"Failed to enable feature {feature_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/features/{feature_name}/disable")
async def disable_feature(
    feature_name: str,
    manager: UnifiedSystemManager = Depends(get_unified_manager_dependency)
):
    """Disable a feature"""
    try:
        success = await manager.disable_feature(feature_name)
        
        if success:
            return {"success": True, "message": f"Feature {feature_name} disabled"}
        else:
            raise HTTPException(status_code=400, detail=f"Failed to disable feature {feature_name}")
            
    except Exception as e:
        logger.error(f"Failed to disable feature {feature_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Configuration endpoints

@app.get("/config")
async def get_configuration(config: UnifiedConfig = Depends(get_config_dependency)):
    """Get system configuration"""
    try:
        return config.get_summary()
    except Exception as e:
        logger.error(f"Failed to get configuration: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal server error",
            "detail": str(exc) if config and config.debug else "An error occurred",
            "timestamp": datetime.utcnow().isoformat()
        }
    )

# Main function
def main():
    """Main function to run the unified API"""
    global config
    
    # Load configuration
    config = get_config()
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, config.api.log_level.value),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Run the application
    uvicorn.run(
        "refactored_unified_system.unified_api:app",
        host=config.api.host,
        port=config.api.port,
        workers=config.api.workers,
        reload=config.api.reload,
        log_level=config.api.log_level.value.lower()
    )

if __name__ == "__main__":
    main()





















