#!/usr/bin/env python3
"""
🚀 Quick Start: FastAPI Dependency Injection for Video-OpusClip

This script demonstrates how to use FastAPI's dependency injection system
for managing state and shared resources in the Video-OpusClip system.

Features demonstrated:
- Database connection management
- Cache management
- Model loading and caching
- Performance optimization
- Error handling
- Health monitoring
- Testing support
"""

import asyncio
import time
import os
from typing import Dict, Any, Optional
from contextlib import asynccontextmanager

import torch
import torch.nn as nn
from fastapi import FastAPI, Depends, HTTPException, status, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import structlog

# Import Video-OpusClip components
from fastapi_dependency_injection import (
    DependencyContainer, AppConfig, get_app_config,
    get_dependency_container, set_dependency_container,
    create_app, add_health_endpoints
)

# Configure logging
logger = structlog.get_logger(__name__)

# =============================================================================
# Configuration Setup
# =============================================================================

def setup_environment():
    """Setup environment variables for testing."""
    os.environ.update({
        "DATABASE_URL": "sqlite+aiosqlite:///./test_video_opusclip.db",
        "REDIS_URL": "redis://localhost:6379",
        "MODEL_CACHE_DIR": "./models",
        "DEVICE": "cpu",  # Use CPU for testing
        "MIXED_PRECISION": "false",
        "MAX_WORKERS": "2",
        "REQUEST_TIMEOOUT": "30.0",
        "ENABLE_METRICS": "true",
        "ENABLE_HEALTH_CHECKS": "true",
        "SECRET_KEY": "test-secret-key",
        "DEBUG": "true"
    })

# =============================================================================
# Request/Response Models
# =============================================================================

class VideoProcessingRequest(BaseModel):
    """Request model for video processing."""
    video_url: str = Field(..., description="URL of the video to process")
    processing_type: str = Field(default="caption", description="Type of processing")
    max_length: int = Field(default=100, ge=1, le=500, description="Maximum output length")
    
    class Config:
        schema_extra = {
            "example": {
                "video_url": "https://example.com/video.mp4",
                "processing_type": "caption",
                "max_length": 100
            }
        }

class VideoProcessingResponse(BaseModel):
    """Response model for video processing."""
    success: bool = Field(..., description="Whether processing was successful")
    result: Optional[Dict[str, Any]] = Field(None, description="Processing result")
    processing_time: float = Field(..., description="Processing time in seconds")
    model_used: str = Field(..., description="Model used for processing")

# =============================================================================
# Mock Models and Services
# =============================================================================

class MockVideoProcessor(nn.Module):
    """Mock video processor for demonstration."""
    
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, 10)
    
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class MockCaptionGenerator(nn.Module):
    """Mock caption generator for demonstration."""
    
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(100, 256)
        self.decoder = nn.Linear(256, 1000)  # Vocabulary size
    
    def forward(self, x):
        encoded = torch.relu(self.encoder(x))
        decoded = self.decoder(encoded)
        return decoded

# =============================================================================
# Dependency Injection Examples
# =============================================================================

async def demonstrate_dependency_injection():
    """Demonstrate dependency injection patterns."""
    
    print("🔧 Setting up dependency injection system...")
    
    # Setup environment
    setup_environment()
    
    # Get configuration
    config = get_app_config()
    print(f"✅ Configuration loaded: {config.database_url}")
    
    # Create dependency container
    container = DependencyContainer(config)
    set_dependency_container(container)
    
    # Initialize container
    await container.initialize()
    print("✅ Dependency container initialized")
    
    # Demonstrate different dependency patterns
    await demonstrate_singleton_dependencies(container)
    await demonstrate_cached_dependencies(container)
    await demonstrate_scoped_dependencies(container)
    await demonstrate_error_handling(container)
    await demonstrate_health_monitoring(container)
    
    # Cleanup
    await container.cleanup()
    print("✅ Dependency container cleaned up")

async def demonstrate_singleton_dependencies(container: DependencyContainer):
    """Demonstrate singleton dependency pattern."""
    
    print("\n🎯 Demonstrating Singleton Dependencies...")
    
    # Database manager is singleton
    db_manager1 = container.db_manager
    db_manager2 = container.db_manager
    print(f"✅ Database managers are same instance: {db_manager1 is db_manager2}")
    
    # Cache manager is singleton
    cache_manager1 = container.cache_manager
    cache_manager2 = container.cache_manager
    print(f"✅ Cache managers are same instance: {cache_manager1 is cache_manager2}")
    
    # Model manager is singleton
    model_manager1 = container.model_manager
    model_manager2 = container.model_manager
    print(f"✅ Model managers are same instance: {model_manager1 is model_manager2}")

async def demonstrate_cached_dependencies(container: DependencyContainer):
    """Demonstrate cached dependency pattern."""
    
    print("\n💾 Demonstrating Cached Dependencies...")
    
    # Load model (should be cached)
    start_time = time.time()
    model1 = await container.model_manager.load_model("video_processor", {})
    load_time1 = time.time() - start_time
    print(f"✅ First model load: {load_time1:.3f}s")
    
    # Load same model again (should use cache)
    start_time = time.time()
    model2 = await container.model_manager.load_model("video_processor", {})
    load_time2 = time.time() - start_time
    print(f"✅ Second model load (cached): {load_time2:.3f}s")
    
    # Verify it's the same model
    print(f"✅ Models are same instance: {model1 is model2}")
    print(f"✅ Cache performance improvement: {load_time1/load_time2:.1f}x faster")

async def demonstrate_scoped_dependencies(container: DependencyContainer):
    """Demonstrate scoped dependency pattern."""
    
    print("\n🔄 Demonstrating Scoped Dependencies...")
    
    # Request-scoped database session
    async with container.db_manager.get_session() as session1:
        print(f"✅ Database session 1: {session1}")
        
        async with container.db_manager.get_session() as session2:
            print(f"✅ Database session 2: {session2}")
            print(f"✅ Sessions are different: {session1 is not session2}")
    
    # Singleton performance optimizer
    optimizer1 = container.performance_optimizer
    optimizer2 = container.performance_optimizer
    print(f"✅ Performance optimizers are same: {optimizer1 is optimizer2}")

async def demonstrate_error_handling(container: DependencyContainer):
    """Demonstrate error handling in dependencies."""
    
    print("\n🛡️ Demonstrating Error Handling...")
    
    try:
        # Try to get non-existent model
        model = await container.model_manager.get_model("non_existent_model")
        print("❌ Should have raised exception")
    except Exception as e:
        print(f"✅ Properly handled missing model: {type(e).__name__}")
    
    try:
        # Try to get database session (should work)
        async with container.db_manager.get_session() as session:
            print("✅ Database session created successfully")
    except Exception as e:
        print(f"❌ Database session failed: {e}")

async def demonstrate_health_monitoring(container: DependencyContainer):
    """Demonstrate health monitoring."""
    
    print("\n🏥 Demonstrating Health Monitoring...")
    
    # Check database health
    db_healthy = await container.db_manager.health_check()
    print(f"✅ Database health: {'🟢 Healthy' if db_healthy else '🔴 Unhealthy'}")
    
    # Check cache health
    cache_healthy = await container.cache_manager.health_check()
    print(f"✅ Cache health: {'🟢 Healthy' if cache_healthy else '🔴 Unhealthy'}")
    
    # Check model health
    model_healthy = await container.model_manager.health_check()
    print(f"✅ Model health: {'🟢 Healthy' if model_healthy else '🔴 Unhealthy'}")
    
    # Check service health
    service_healthy = await container.service_manager.health_check()
    print(f"✅ Service health: {'🟢 Healthy' if service_healthy else '🔴 Unhealthy'}")

# =============================================================================
# FastAPI Application Examples
# =============================================================================

def create_demo_app() -> FastAPI:
    """Create demo FastAPI application with dependency injection."""
    
    # Create app with lifespan management
    app = create_app()
    add_health_endpoints(app)
    
    # Add demo routes
    add_demo_routes(app)
    
    return app

def add_demo_routes(app: FastAPI):
    """Add demo routes to FastAPI app."""
    
    @app.post("/demo/video/process", response_model=VideoProcessingResponse)
    async def demo_process_video(
        request: VideoProcessingRequest,
        model: nn.Module = Depends(get_dependency_container().get_model_dependency("video_processor")),
        optimizer = Depends(get_dependency_container().get_performance_optimizer_dependency()),
        error_handler = Depends(get_dependency_container().get_error_handler_dependency())
    ):
        """Demo video processing with dependency injection."""
        
        start_time = time.time()
        
        try:
            # Simulate video processing
            await asyncio.sleep(0.1)  # Simulate processing time
            
            # Use injected dependencies
            with optimizer.optimize_context():
                # Mock processing
                input_tensor = torch.randn(1, 3, 224, 224)
                result = model(input_tensor)
                
                processing_result = {
                    "video_url": request.video_url,
                    "processing_type": request.processing_type,
                    "output": result.tolist(),
                    "confidence": 0.95
                }
            
            processing_time = time.time() - start_time
            
            return VideoProcessingResponse(
                success=True,
                result=processing_result,
                processing_time=processing_time,
                model_used="video_processor"
            )
            
        except Exception as e:
            await error_handler.handle_error(e)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Video processing failed: {str(e)}"
            )
    
    @app.get("/demo/dependencies/status")
    async def demo_dependencies_status():
        """Demo endpoint to check dependency status."""
        
        container = get_dependency_container()
        
        status = {
            "database": await container.db_manager.health_check(),
            "cache": await container.cache_manager.health_check(),
            "models": await container.model_manager.health_check(),
            "services": await container.service_manager.health_check(),
            "performance_optimizer": True,  # Always available
            "training_logger": True,        # Always available
            "error_handler": True,          # Always available
            "mixed_precision_manager": True,
            "gradient_accumulator": True,
            "multi_gpu_trainer": True,
            "code_profiler": True
        }
        
        overall_health = all(status.values())
        
        return {
            "overall_health": overall_health,
            "dependencies": status,
            "timestamp": time.time()
        }
    
    @app.get("/demo/models/loaded")
    async def demo_loaded_models():
        """Demo endpoint to show loaded models."""
        
        container = get_dependency_container()
        models = list(container.model_manager.models.keys())
        
        return {
            "loaded_models": models,
            "total_models": len(models),
            "device": str(container.model_manager._device)
        }

# =============================================================================
# Testing Examples
# =============================================================================

async def demonstrate_testing():
    """Demonstrate testing with dependency injection."""
    
    print("\n🧪 Demonstrating Testing Support...")
    
    # Create test app
    app = create_demo_app()
    
    # Test health endpoint
    from fastapi.testclient import TestClient
    client = TestClient(app)
    
    # Test health check
    response = client.get("/health")
    print(f"✅ Health check status: {response.status_code}")
    print(f"✅ Health check response: {response.json()}")
    
    # Test detailed health check
    response = client.get("/health/detailed")
    print(f"✅ Detailed health check status: {response.status_code}")
    
    # Test video processing
    video_request = {
        "video_url": "https://example.com/test.mp4",
        "processing_type": "caption",
        "max_length": 100
    }
    
    response = client.post("/demo/video/process", json=video_request)
    print(f"✅ Video processing status: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Processing time: {result['processing_time']:.3f}s")
        print(f"✅ Model used: {result['model_used']}")
    
    # Test dependencies status
    response = client.get("/demo/dependencies/status")
    print(f"✅ Dependencies status: {response.status_code}")
    
    # Test loaded models
    response = client.get("/demo/models/loaded")
    print(f"✅ Loaded models: {response.json()}")

# =============================================================================
# Performance Examples
# =============================================================================

async def demonstrate_performance():
    """Demonstrate performance optimization with dependencies."""
    
    print("\n⚡ Demonstrating Performance Optimization...")
    
    container = get_dependency_container()
    
    # Test model loading performance
    print("Testing model loading performance...")
    
    start_time = time.time()
    model1 = await container.model_manager.load_model("video_processor", {})
    first_load_time = time.time() - start_time
    
    start_time = time.time()
    model2 = await container.model_manager.load_model("video_processor", {})
    second_load_time = time.time() - start_time
    
    print(f"✅ First load: {first_load_time:.3f}s")
    print(f"✅ Second load (cached): {second_load_time:.3f}s")
    print(f"✅ Performance improvement: {first_load_time/second_load_time:.1f}x")
    
    # Test database connection pooling
    print("\nTesting database connection pooling...")
    
    start_time = time.time()
    async with container.db_manager.get_session() as session1:
        async with container.db_manager.get_session() as session2:
            async with container.db_manager.get_session() as session3:
                pass
    pool_time = time.time() - start_time
    
    print(f"✅ Multiple sessions created in: {pool_time:.3f}s")
    
    # Test performance optimizer
    print("\nTesting performance optimizer...")
    
    with container.performance_optimizer.optimize_context():
        # Simulate optimized operation
        await asyncio.sleep(0.05)
        print("✅ Performance optimization applied")

# =============================================================================
# Main Execution
# =============================================================================

async def main():
    """Main function demonstrating FastAPI dependency injection."""
    
    print("🚀 FastAPI Dependency Injection Quick Start")
    print("=" * 50)
    
    try:
        # Demonstrate core dependency injection
        await demonstrate_dependency_injection()
        
        # Demonstrate testing
        await demonstrate_testing()
        
        # Demonstrate performance
        await demonstrate_performance()
        
        print("\n✅ All demonstrations completed successfully!")
        print("\n📚 Next Steps:")
        print("1. Review the fastapi_dependency_injection.py file")
        print("2. Check the FASTAPI_DEPENDENCY_INJECTION_GUIDE.md")
        print("3. Implement your own dependencies")
        print("4. Add health monitoring to your routes")
        print("5. Write tests for your dependencies")
        
    except Exception as e:
        print(f"❌ Error during demonstration: {e}")
        logger.error("Demonstration failed", error=str(e), exc_info=True)

if __name__ == "__main__":
    # Run the demonstration
    asyncio.run(main()) 