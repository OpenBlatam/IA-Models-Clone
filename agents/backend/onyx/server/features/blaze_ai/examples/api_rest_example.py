"""
Blaze AI REST API Example

This example demonstrates how to use the REST API module to access
all Blaze AI capabilities through HTTP endpoints.
"""

import asyncio
import logging
import time
import base64
import json
from typing import Dict, Any

# Import the modular system
from ..modules import (
    ModuleRegistry,
    create_module_registry,
    create_cache_module,
    create_monitoring_module,
    create_optimization_module,
    create_storage_module,
    create_execution_module,
    create_engines_module,
    create_ml_module,
    create_data_analysis_module,
    create_ai_intelligence_module,
    create_api_rest_module
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# API CLIENT FOR TESTING
# ============================================================================

class APIClient:
    """Simple API client for testing endpoints."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.api_key = "test_api_key_12345"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    async def test_health_check(self):
        """Test health check endpoint."""
        logger.info("🏥 Testing Health Check...")
        try:
            import httpx
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.base_url}/health")
                logger.info(f"✅ Health check: {response.status_code} - {response.json()}")
                return response.json()
        except Exception as e:
            logger.error(f"❌ Health check failed: {e}")
            return None
    
    async def test_api_info(self):
        """Test API info endpoint."""
        logger.info("📋 Testing API Info...")
        try:
            import httpx
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.base_url}/")
                logger.info(f"✅ API info: {response.status_code} - {response.json()}")
                return response.json()
        except Exception as e:
            logger.error(f"❌ API info failed: {e}")
            return None
    
    async def test_nlp_sentiment(self, text: str):
        """Test NLP sentiment analysis."""
        logger.info(f"🧠 Testing NLP Sentiment: '{text[:30]}...'")
        try:
            import httpx
            payload = {
                "text": text,
                "task": "sentiment",
                "language": "en"
            }
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/api/v1/nlp/sentiment",
                    json=payload,
                    headers=self.headers
                )
                logger.info(f"✅ NLP sentiment: {response.status_code} - {response.json()}")
                return response.json()
        except Exception as e:
            logger.error(f"❌ NLP sentiment failed: {e}")
            return None
    
    async def test_vision_detection(self, image_description: str):
        """Test computer vision object detection."""
        logger.info(f"👁️ Testing Vision Detection: {image_description}")
        try:
            import httpx
            # Simulate image data (base64 encoded)
            simulated_image = base64.b64encode(f"simulated_image_data_{image_description}".encode()).decode()
            
            payload = {
                "image_data": simulated_image,
                "task": "object_detection",
                "format": "jpeg"
            }
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/api/v1/vision/detect",
                    json=payload,
                    headers=self.headers
                )
                logger.info(f"✅ Vision detection: {response.status_code} - {response.json()}")
                return response.json()
        except Exception as e:
            logger.error(f"❌ Vision detection failed: {e}")
            return None
    
    async def test_reasoning(self, query: str, reasoning_type: str = "logical"):
        """Test automated reasoning."""
        logger.info(f"🧮 Testing Reasoning ({reasoning_type}): '{query[:40]}...'")
        try:
            import httpx
            payload = {
                "query": query,
                "reasoning_type": reasoning_type,
                "context": {"domain": "philosophy"}
            }
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/api/v1/reasoning/process",
                    json=payload,
                    headers=self.headers
                )
                logger.info(f"✅ Reasoning: {response.status_code} - {response.json()}")
                return response.json()
        except Exception as e:
            logger.error(f"❌ Reasoning failed: {e}")
            return None
    
    async def test_multimodal(self, text: str, image_description: str):
        """Test multimodal processing."""
        logger.info(f"🔄 Testing Multimodal: '{text[:30]}...' + {image_description}")
        try:
            import httpx
            # Simulate image data
            simulated_image = base64.b64encode(f"simulated_image_data_{image_description}".encode()).decode()
            
            payload = {
                "text": text,
                "image_data": simulated_image,
                "task": "analysis"
            }
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/api/v1/multimodal/process",
                    json=payload,
                    headers=self.headers
                )
                logger.info(f"✅ Multimodal: {response.status_code} - {response.json()}")
                return response.json()
        except Exception as e:
            logger.error(f"❌ Multimodal failed: {e}")
            return None
    
    async def test_cache_operations(self):
        """Test cache operations."""
        logger.info("💾 Testing Cache Operations...")
        try:
            import httpx
            
            # Set cache value
            set_payload = {
                "key": "test_key",
                "value": {"message": "Hello from API", "timestamp": time.time()},
                "ttl": 3600,
                "tags": ["test", "api"]
            }
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/api/v1/cache",
                    json=set_payload,
                    headers=self.headers
                )
                logger.info(f"✅ Cache set: {response.status_code} - {response.json()}")
            
            # Get cache value
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.base_url}/api/v1/cache/test_key",
                    headers=self.headers
                )
                logger.info(f"✅ Cache get: {response.status_code} - {response.json()}")
                return response.json()
        except Exception as e:
            logger.error(f"❌ Cache operations failed: {e}")
            return None
    
    async def test_system_status(self):
        """Test system status endpoint."""
        logger.info("📊 Testing System Status...")
        try:
            import httpx
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.base_url}/api/v1/system/status",
                    headers=self.headers
                )
                logger.info(f"✅ System status: {response.status_code} - {response.json()}")
                return response.json()
        except Exception as e:
            logger.error(f"❌ System status failed: {e}")
            return None
    
    async def test_system_metrics(self):
        """Test system metrics endpoint."""
        logger.info("📈 Testing System Metrics...")
        try:
            import httpx
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.base_url}/api/v1/system/metrics",
                    headers=self.headers
                )
                logger.info(f"✅ System metrics: {response.status_code} - {response.json()}")
                return response.json()
        except Exception as e:
            logger.error(f"❌ System metrics failed: {e}")
            return None

# ============================================================================
# MAIN EXAMPLE
# ============================================================================

async def main():
    """Main example function."""
    logger.info("🚀 Starting Blaze AI REST API Example")
    
    # Create module registry
    registry = create_module_registry()
    
    try:
        # Initialize registry
        await registry.initialize()
        
        # ========================================================================
        # CREATE AND REGISTER MODULES
        # ========================================================================
        
        logger.info("📦 Creating and registering modules...")
        
        # Create core modules
        cache = create_cache_module("api_cache", max_size=1000, priority=1)
        monitoring = create_monitoring_module("api_monitoring", collection_interval=5.0, priority=2)
        optimization = create_optimization_module("api_optimization", optimization_type="GENETIC", priority=3)
        storage = create_storage_module("api_storage", storage_path="./api_data", priority=4)
        execution = create_execution_module("api_execution", max_workers=8, priority=5)
        engines = create_engines_module("api_engines", priority=6)
        ml_module = create_ml_module("api_ml", max_training_jobs=3, priority=7)
        data_analysis = create_data_analysis_module("api_data_analysis", max_concurrent_jobs=2, priority=8)
        ai_intelligence = create_ai_intelligence_module(
            "api_ai_intelligence",
            enable_nlp=True,
            enable_vision=True,
            enable_reasoning=True,
            enable_multimodal=True,
            priority=9
        )
        
        # Create REST API module
        api_rest = create_api_rest_module(
            "blaze_ai_api",
            host="0.0.0.0",
            port=8000,
            api_keys=["test_api_key_12345"],
            enable_cors=True,
            rate_limit_enabled=True,
            enable_documentation=True
        )
        
        # Register all modules
        modules = [
            cache, monitoring, optimization, storage, execution,
            engines, ml_module, data_analysis, ai_intelligence, api_rest
        ]
        
        for module in modules:
            await registry.register_module(module)
        
        logger.info("✅ All modules registered successfully")
        
        # ========================================================================
        # CONFIGURE API MODULE WITH OTHER MODULES
        # ========================================================================
        
        logger.info("🔗 Configuring API module with other modules...")
        
        # Set module references in API module
        api_rest.registry = registry
        api_rest.ai_intelligence = ai_intelligence
        api_rest.ml_module = ml_module
        api_rest.data_analysis = data_analysis
        api_rest.cache = cache
        api_rest.monitoring = monitoring
        api_rest.optimization = optimization
        api_rest.storage = storage
        api_rest.execution = execution
        api_rest.engines = engines
        
        logger.info("✅ API module configured successfully")
        
        # ========================================================================
        # WAIT FOR MODULES TO BE READY
        # ========================================================================
        
        logger.info("⏳ Waiting for modules to be ready...")
        await asyncio.sleep(3)
        
        # ========================================================================
        # TEST API ENDPOINTS
        # ========================================================================
        
        logger.info("🧪 Testing API endpoints...")
        
        # Create API client
        client = APIClient("http://localhost:8000")
        
        # Test basic endpoints
        await client.test_health_check()
        await client.test_api_info()
        
        # Test AI capabilities
        await client.test_nlp_sentiment("I absolutely love this amazing product! It's incredible!")
        await client.test_nlp_sentiment("This is the worst experience I've ever had. Terrible service!")
        
        await client.test_vision_detection("sunset_over_mountains")
        await client.test_vision_detection("busy_city_street")
        
        await client.test_reasoning(
            "If all humans are mortal and Socrates is human, what can we conclude?",
            "logical"
        )
        await client.test_reasoning(
            "What is the symbolic representation of a logical AND operation?",
            "symbolic"
        )
        
        await client.test_multimodal(
            "A beautiful landscape with mountains and trees",
            "mountain_landscape"
        )
        await client.test_multimodal(
            "A modern office building in the city center",
            "office_building"
        )
        
        # Test utility endpoints
        await client.test_cache_operations()
        await client.test_system_status()
        await client.test_system_metrics()
        
        # ========================================================================
        # PERFORMANCE TESTING
        # ========================================================================
        
        logger.info("⚡ Running API performance tests...")
        
        # Test concurrent requests
        start_time = time.time()
        tasks = []
        for i in range(10):
            task = client.test_nlp_sentiment(f"Test text {i} for performance testing")
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        end_time = time.time()
        
        successful_requests = sum(1 for r in results if r is not None)
        logger.info(f"📊 Performance test: {successful_requests}/10 requests successful in {end_time - start_time:.3f}s")
        
        # ========================================================================
        # API DOCUMENTATION INFO
        # ========================================================================
        
        logger.info("📚 API Documentation available at:")
        logger.info(f"   Swagger UI: http://localhost:8000/docs")
        logger.info(f"   ReDoc: http://localhost:8000/redoc")
        logger.info(f"   OpenAPI JSON: http://localhost:8000/openapi.json")
        
        # ========================================================================
        # SYSTEM STATUS
        # ========================================================================
        
        logger.info("📈 Final system status...")
        
        # Get final status of all modules
        for module_name in registry.list_modules():
            module = registry.get_module(module_name)
            if module:
                status = module.get_status()
                logger.info(f"Module {module_name}: {status['status']}")
        
        logger.info("🎉 API REST example completed successfully!")
        logger.info("🌐 API is now running and accessible at http://localhost:8000")
        logger.info("🔑 Use API key: test_api_key_12345 for authentication")
        
        # Keep the system running for manual testing
        logger.info("⏸️ System will continue running for manual testing...")
        logger.info("🛑 Press Ctrl+C to stop the system")
        
        try:
            # Keep running for manual testing
            while True:
                await asyncio.sleep(10)
                # Log periodic status
                logger.info("💓 System heartbeat - API is running...")
        except KeyboardInterrupt:
            logger.info("🛑 Shutdown requested by user")
        
    except Exception as e:
        logger.error(f"❌ Example failed: {e}")
        raise
    
    finally:
        # Shutdown registry
        logger.info("🔄 Shutting down system...")
        await registry.shutdown()
        logger.info("✅ System shutdown completed")

# ============================================================================
# RUN EXAMPLE
# ============================================================================

if __name__ == "__main__":
    asyncio.run(main())
