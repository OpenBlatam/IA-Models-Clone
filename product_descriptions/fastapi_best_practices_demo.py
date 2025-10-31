from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
TIMEOUT_SECONDS = 60

import asyncio
import json
import time
from datetime import datetime
from typing import Dict, Any, List, Optional
import logging
from fastapi import FastAPI, Depends, HTTPException, status, Query, Path, Body
from fastapi.testclient import TestClient
from fastapi.responses import JSONResponse
from models.fastapi_models import (
from operations.fastapi_operations import router as operations_router
from middleware.fastapi_middleware import (
from typing import Any, List, Dict, Optional
"""
FastAPI Best Practices Demo

This demo showcases FastAPI best practices for Data Models, Path Operations,
and Middleware following the official documentation guidelines.
"""


# Import FastAPI components

# Import our models and middleware
    User, UserCreate, ProductDescription, ProductDescriptionRequest,
    ProductDescriptionResponse, PaginationParams, ErrorResponse
)
    MiddlewareStack, RequestLoggingMiddleware, PerformanceMonitoringMiddleware,
    ErrorHandlingMiddleware, SecurityHeadersMiddleware, RateLimitingMiddleware
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FastAPIBestPracticesDemo:
    """Demo class for showcasing FastAPI best practices."""
    
    def __init__(self) -> Any:
        self.demo_results = []
        self.start_time = None
        self.app = None
        self.client = None
    
    async def run_demo(self) -> Any:
        """Run the complete FastAPI best practices demo."""
        logger.info("🚀 Starting FastAPI Best Practices Demo")
        self.start_time = time.time()
        
        try:
            # Demo sections
            await self.demo_data_models()
            await self.demo_path_operations()
            await self.demo_middleware()
            await self.demo_integration()
            await self.demo_testing()
            
            await self.print_demo_summary()
            
        except Exception as e:
            logger.error(f"Demo failed: {e}")
            raise
    
    async def demo_data_models(self) -> Any:
        """Demo FastAPI data models best practices."""
        logger.info("\n📋 Demo 1: Data Models Best Practices")
        
        # Create sample data models
        user_create = UserCreate(
            email="john.doe@example.com",
            username="john_doe",
            password="SecurePass123!",
            role="user"
        )
        
        product_request = ProductDescriptionRequest(
            product_name="iPhone 15 Pro",
            category="electronics",
            features=["5G connectivity", "A17 Pro chip", "48MP camera"],
            target_audience="Tech enthusiasts and professionals",
            tone="professional",
            length="medium",
            language="en"
        )
        
        # Demonstrate model validation
        validation_examples = {
            "valid_user": {
                "email": "valid@example.com",
                "username": "valid_user",
                "password": "ValidPass123!",
                "role": "user"
            },
            "invalid_user": {
                "email": "invalid-email",
                "username": "ab",  # Too short
                "password": "weak",  # Too weak
                "role": "invalid_role"
            },
            "valid_product": {
                "product_name": "Valid Product",
                "category": "electronics",
                "features": ["Feature 1", "Feature 2"],
                "tone": "professional",
                "length": "medium",
                "language": "en"
            }
        }
        
        self.demo_results.append({
            "section": "Data Models",
            "description": "FastAPI data models with validation and best practices",
            "data": {
                "user_create_model": user_create.model_dump(),
                "product_request_model": product_request.model_dump(),
                "validation_examples": validation_examples
            }
        })
        
        logger.info("✅ Data Models Best Practices:")
        logger.info("   - Pydantic models with comprehensive validation")
        logger.info("   - Field constraints and custom validators")
        logger.info("   - Enum types for constrained choices")
        logger.info("   - Computed fields and mixins")
        logger.info("   - Proper error handling and messages")
    
    async def demo_path_operations(self) -> Any:
        """Demo FastAPI path operations best practices."""
        logger.info("\n🔧 Demo 2: Path Operations Best Practices")
        
        # Create FastAPI app for testing
        app = FastAPI(
            title="FastAPI Best Practices Demo",
            description="Demonstrating FastAPI best practices",
            version="1.0.0"
        )
        
        # Add operations router
        app.include_router(operations_router, prefix="/api/v1")
        
        # Create test client
        client = TestClient(app)
        
        # Test path operations
        path_operation_examples = {
            "health_check": {
                "method": "GET",
                "path": "/api/v1/health",
                "expected_status": 200,
                "description": "Health check endpoint"
            },
            "generate_description": {
                "method": "POST",
                "path": "/api/v1/product-descriptions/generate",
                "expected_status": 201,
                "description": "Generate product description"
            },
            "get_description": {
                "method": "GET",
                "path": "/api/v1/product-descriptions/{description_id}",
                "expected_status": 200,
                "description": "Get specific description"
            },
            "list_descriptions": {
                "method": "GET",
                "path": "/api/v1/product-descriptions",
                "expected_status": 200,
                "description": "List descriptions with pagination"
            }
        }
        
        # Test health endpoint
        try:
            response = client.get("/api/v1/health")
            health_status = response.status_code == 200
        except Exception as e:
            health_status = f"Error: {str(e)}"
        
        self.demo_results.append({
            "section": "Path Operations",
            "description": "FastAPI path operations with proper decorators and validation",
            "data": {
                "path_operation_examples": path_operation_examples,
                "health_endpoint_test": health_status,
                "best_practices": [
                    "Proper HTTP status codes",
                    "Comprehensive response models",
                    "Input validation with Path, Query, Body",
                    "Error handling with HTTPException",
                    "Background tasks for async operations",
                    "Dependency injection for shared resources"
                ]
            }
        })
        
        logger.info("✅ Path Operations Best Practices:")
        logger.info("   - Proper HTTP methods and status codes")
        logger.info("   - Comprehensive response models")
        logger.info("   - Input validation with Path, Query, Body")
        logger.info("   - Error handling with HTTPException")
        logger.info("   - Background tasks for async operations")
        logger.info("   - Dependency injection for shared resources")
    
    async def demo_middleware(self) -> Any:
        """Demo FastAPI middleware best practices."""
        logger.info("\n🛡️ Demo 3: Middleware Best Practices")
        
        # Create middleware stack
        middleware_stack = MiddlewareStack()
        middleware_stack.configure_default_stack()
        
        # Test middleware configuration
        middleware_config = middleware_stack.get_middleware_config()
        
        # Create app with middleware
        app = FastAPI(title="Middleware Demo")
        
        # Add middleware in proper order
        for middleware_class, kwargs in middleware_config:
            app.add_middleware(middleware_class, **kwargs)
        
        # Test middleware functionality
        client = TestClient(app)
        
        # Add a test endpoint
        @app.get("/test")
        async def test_endpoint():
            
    """test_endpoint function."""
return {"message": "Test endpoint with middleware"}
        
        # Test middleware
        try:
            response = client.get("/test")
            middleware_test = {
                "status_code": response.status_code,
                "headers": dict(response.headers),
                "has_security_headers": "X-Content-Type-Options" in response.headers,
                "has_request_id": "X-Request-ID" in response.headers
            }
        except Exception as e:
            middleware_test = f"Error: {str(e)}"
        
        self.demo_results.append({
            "section": "Middleware",
            "description": "FastAPI middleware stack with best practices",
            "data": {
                "middleware_config": [
                    {
                        "name": middleware_class.__name__,
                        "order": i + 1,
                        "purpose": self.get_middleware_purpose(middleware_class.__name__)
                    }
                    for i, (middleware_class, kwargs) in enumerate(middleware_config)
                ],
                "middleware_test": middleware_test,
                "best_practices": [
                    "Proper middleware order",
                    "Security middleware first",
                    "Logging and monitoring",
                    "Error handling",
                    "Performance optimization",
                    "Rate limiting"
                ]
            }
        })
        
        logger.info("✅ Middleware Best Practices:")
        logger.info("   - Proper middleware order (security first)")
        logger.info("   - Comprehensive logging and monitoring")
        logger.info("   - Error handling and recovery")
        logger.info("   - Security headers and rate limiting")
        logger.info("   - Performance monitoring and optimization")
        logger.info("   - Request context management")
    
    async def demo_integration(self) -> Any:
        """Demo integration of all FastAPI best practices."""
        logger.info("\n🔗 Demo 4: Integration Best Practices")
        
        # Create comprehensive FastAPI app
        app = FastAPI(
            title="FastAPI Best Practices Integration",
            description="Complete integration of FastAPI best practices",
            version="1.0.0",
            docs_url="/docs",
            redoc_url="/redoc"
        )
        
        # Configure middleware
        middleware_stack = MiddlewareStack()
        middleware_stack.configure_default_stack()
        
        for middleware_class, kwargs in middleware_stack.get_middleware_config():
            app.add_middleware(middleware_class, **kwargs)
        
        # Add routers
        app.include_router(operations_router, prefix="/api/v1")
        
        # Add custom endpoints
        @app.get("/", tags=["Root"])
        async def root():
            
    """root function."""
return {
                "message": "FastAPI Best Practices Demo",
                "version": "1.0.0",
                "docs": "/docs",
                "health": "/api/v1/health"
            }
        
        @app.get("/api/v1/status", tags=["Status"])
        async def status_endpoint():
            
    """status_endpoint function."""
return {
                "status": "healthy",
                "timestamp": datetime.utcnow().isoformat(),
                "middleware_count": len(middleware_stack.get_middleware_config())
            }
        
        # Test integration
        client = TestClient(app)
        
        integration_tests = {}
        
        # Test root endpoint
        try:
            response = client.get("/")
            integration_tests["root_endpoint"] = {
                "status_code": response.status_code,
                "has_security_headers": "X-Content-Type-Options" in response.headers
            }
        except Exception as e:
            integration_tests["root_endpoint"] = f"Error: {str(e)}"
        
        # Test status endpoint
        try:
            response = client.get("/api/v1/status")
            integration_tests["status_endpoint"] = {
                "status_code": response.status_code,
                "response_data": response.json()
            }
        except Exception as e:
            integration_tests["status_endpoint"] = f"Error: {str(e)}"
        
        # Test health endpoint
        try:
            response = client.get("/api/v1/health")
            integration_tests["health_endpoint"] = {
                "status_code": response.status_code,
                "response_data": response.json()
            }
        except Exception as e:
            integration_tests["health_endpoint"] = f"Error: {str(e)}"
        
        self.demo_results.append({
            "section": "Integration",
            "description": "Complete integration of FastAPI best practices",
            "data": {
                "integration_tests": integration_tests,
                "app_configuration": {
                    "title": app.title,
                    "version": app.version,
                    "docs_url": app.docs_url,
                    "redoc_url": app.redoc_url
                },
                "features": [
                    "Comprehensive data models",
                    "Proper path operations",
                    "Middleware stack",
                    "Error handling",
                    "Documentation",
                    "Testing support"
                ]
            }
        })
        
        logger.info("✅ Integration Best Practices:")
        logger.info("   - Complete FastAPI application setup")
        logger.info("   - Middleware integration")
        logger.info("   - Router organization")
        logger.info("   - Error handling")
        logger.info("   - Documentation generation")
        logger.info("   - Testing support")
    
    async def demo_testing(self) -> Any:
        """Demo testing best practices for FastAPI."""
        logger.info("\n🧪 Demo 5: Testing Best Practices")
        
        # Create test app
        app = FastAPI(title="Testing Demo")
        app.include_router(operations_router, prefix="/api/v1")
        
        client = TestClient(app)
        
        # Test cases
        test_cases = {
            "health_check": {
                "method": "GET",
                "path": "/api/v1/health",
                "expected_status": 200,
                "description": "Health check should return 200"
            },
            "invalid_endpoint": {
                "method": "GET",
                "path": "/api/v1/nonexistent",
                "expected_status": 404,
                "description": "Nonexistent endpoint should return 404"
            },
            "method_not_allowed": {
                "method": "POST",
                "path": "/api/v1/health",
                "expected_status": 405,
                "description": "POST to GET endpoint should return 405"
            }
        }
        
        test_results = {}
        
        for test_name, test_case in test_cases.items():
            try:
                if test_case["method"] == "GET":
                    response = client.get(test_case["path"])
                elif test_case["method"] == "POST":
                    response = client.post(test_case["path"])
                
                test_results[test_name] = {
                    "expected_status": test_case["expected_status"],
                    "actual_status": response.status_code,
                    "passed": response.status_code == test_case["expected_status"],
                    "response_data": response.json() if response.content else None
                }
            except Exception as e:
                test_results[test_name] = {
                    "error": str(e),
                    "passed": False
                }
        
        self.demo_results.append({
            "section": "Testing",
            "description": "FastAPI testing best practices",
            "data": {
                "test_cases": test_cases,
                "test_results": test_results,
                "testing_best_practices": [
                    "Use TestClient for integration testing",
                    "Test all HTTP methods and status codes",
                    "Test error scenarios",
                    "Test middleware functionality",
                    "Test data validation",
                    "Use fixtures for test data"
                ]
            }
        })
        
        logger.info("✅ Testing Best Practices:")
        logger.info("   - Use TestClient for integration testing")
        logger.info("   - Test all HTTP methods and status codes")
        logger.info("   - Test error scenarios and edge cases")
        logger.info("   - Test middleware functionality")
        logger.info("   - Test data validation")
        logger.info("   - Use fixtures for test data")
    
    def get_middleware_purpose(self, middleware_name: str) -> str:
        """Get middleware purpose description."""
        purposes = {
            "TrustedHostMiddleware": "Security - Validate trusted hosts",
            "CORSMiddleware": "Cross-origin - Handle CORS requests",
            "GZipMiddleware": "Performance - Compress responses",
            "RequestContextMiddleware": "Context - Manage request state",
            "RequestLoggingMiddleware": "Logging - Log requests and responses",
            "PerformanceMonitoringMiddleware": "Monitoring - Track performance metrics",
            "RateLimitingMiddleware": "Security - Limit request rates",
            "SecurityHeadersMiddleware": "Security - Add security headers",
            "ErrorHandlingMiddleware": "Error handling - Catch and format errors"
        }
        return purposes.get(middleware_name, "Unknown purpose")
    
    async def print_demo_summary(self) -> Any:
        """Print a summary of the demo results."""
        logger.info("\n" + "="*60)
        logger.info("📋 FASTAPI BEST PRACTICES DEMO SUMMARY")
        logger.info("="*60)
        
        total_sections = len(self.demo_results)
        
        logger.info(f"📊 Statistics:")
        logger.info(f"   - Demo sections: {total_sections}")
        logger.info(f"   - Demo duration: {time.time() - self.start_time:.2f}s")
        
        logger.info(f"\n🏗️ FastAPI Best Practices Covered:")
        logger.info(f"   ✅ Data Models with Pydantic")
        logger.info(f"   ✅ Path Operations with proper decorators")
        logger.info(f"   ✅ Middleware stack configuration")
        logger.info(f"   ✅ Error handling and validation")
        logger.info(f"   ✅ Security and performance")
        logger.info(f"   ✅ Testing and documentation")
        
        logger.info(f"\n🎯 Key Benefits:")
        logger.info(f"   📈 Automatic API documentation")
        logger.info(f"   🔧 Type safety and validation")
        logger.info(f"   🛡️ Security and performance")
        logger.info(f"   🧪 Easy testing and debugging")
        logger.info(f"   📚 Self-documenting code")
        logger.info(f"   🚀 Production-ready structure")
        
        logger.info(f"\n📁 File Structure:")
        logger.info(f"   models/fastapi_models.py (Data models)")
        logger.info(f"   operations/fastapi_operations.py (Path operations)")
        logger.info(f"   middleware/fastapi_middleware.py (Middleware)")
        logger.info(f"   fastapi_best_practices_demo.py (Demo)")
        
        logger.info(f"\n🚀 Demo completed successfully!")
        logger.info("="*60)

async def main():
    """Main demo function."""
    demo = FastAPIBestPracticesDemo()
    await demo.run_demo()

match __name__:
    case "__main__":
    asyncio.run(main()) 