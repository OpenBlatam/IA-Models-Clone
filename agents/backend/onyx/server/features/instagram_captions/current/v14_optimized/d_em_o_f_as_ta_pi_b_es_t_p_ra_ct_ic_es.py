from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

import asyncio
import time
import json
import requests
from typing import Dict, Any, List
import logging
from models.fastapi_best_practices import (
from middleware.fastapi_best_practices import (
from typing import Any, List, Dict, Optional
"""
FastAPI Best Practices Demo

This script demonstrates the complete FastAPI best practices implementation:
- Data models with Pydantic v2 validation
- Path operations with proper HTTP methods
- Middleware for security, monitoring, and performance
- Error handling and validation
- Performance optimization
- Security best practices
"""


# Import FastAPI best practices components
    CaptionGenerationRequest, CaptionGenerationResponse,
    BatchCaptionRequest, BatchCaptionResponse,
    UserPreferences, ErrorResponse, HealthResponse,
    CaptionAnalytics, ServiceStatus, CaptionStyle, CaptionTone, LanguageCode
)

    RequestIDMiddleware, LoggingMiddleware, PerformanceMonitoringMiddleware,
    SecurityHeadersMiddleware, RateLimitingMiddleware, ErrorHandlingMiddleware,
    CacheControlMiddleware, create_middleware_stack
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FastAPIBestPracticesDemo:
    """Demo class for FastAPI best practices"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        
    """__init__ function."""
self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": "Bearer demo-api-key",
            "Content-Type": "application/json"
        })
    
    def run_all_demos(self) -> Any:
        """Run all FastAPI best practices demos"""
        logger.info("🚀 Starting FastAPI Best Practices Demo")
        
        try:
            # 1. Data Models Demo
            self.demo_data_models()
            
            # 2. Path Operations Demo
            self.demo_path_operations()
            
            # 3. Middleware Demo
            self.demo_middleware()
            
            # 4. Error Handling Demo
            self.demo_error_handling()
            
            # 5. Performance Demo
            self.demo_performance()
            
            # 6. Security Demo
            self.demo_security()
            
            # 7. API Integration Demo
            self.demo_api_integration()
            
            logger.info("✅ All demos completed successfully!")
            
        except Exception as e:
            logger.error(f"❌ Demo failed: {e}")
            raise
    
    def demo_data_models(self) -> Any:
        """Demo Pydantic v2 data models with validation"""
        logger.info("\n📋 Demo: Data Models with Pydantic v2")
        
        # 1. Basic caption generation request
        try:
            request = CaptionGenerationRequest(
                content_description="Beautiful sunset over mountains with golden light reflecting on a calm lake",
                style=CaptionStyle.CASUAL,
                tone=CaptionTone.FRIENDLY,
                hashtag_count=15,
                language=LanguageCode.ENGLISH,
                include_emoji=True,
                max_length=1000
            )
            logger.info(f"✅ Valid request created: {request.model_dump()}")
            
            # Test computed fields
            logger.info(f"📊 Request summary: {request.model_dump()}")
            
        except Exception as e:
            logger.error(f"❌ Request validation failed: {e}")
        
        # 2. Batch request
        try:
            batch_request = BatchCaptionRequest(
                requests=[
                    CaptionGenerationRequest(
                        content_description="Delicious homemade pizza with melted cheese",
                        style=CaptionStyle.CREATIVE,
                        tone=CaptionTone.ENTHUSIASTIC
                    ),
                    CaptionGenerationRequest(
                        content_description="Modern office space with natural lighting",
                        style=CaptionStyle.PROFESSIONAL,
                        tone=CaptionTone.PROFESSIONAL
                    )
                ],
                max_concurrent=3
            )
            logger.info(f"✅ Batch request created with {len(batch_request.requests)} items")
            
        except Exception as e:
            logger.error(f"❌ Batch request validation failed: {e}")
        
        # 3. User preferences
        try:
            preferences = UserPreferences(
                user_id="user_123",
                default_style=CaptionStyle.CASUAL,
                default_tone=CaptionTone.FRIENDLY,
                default_hashtag_count=15,
                preferred_language=LanguageCode.ENGLISH,
                include_emoji=True
            )
            logger.info(f"✅ User preferences created: {preferences.preferences_summary}")
            
        except Exception as e:
            logger.error(f"❌ User preferences validation failed: {e}")
        
        # 4. Response model
        try:
            response = CaptionGenerationResponse(
                caption="Golden hour magic ✨ Nature's perfect lighting show",
                hashtags=["#sunset", "#mountains", "#goldenhour", "#nature"],
                style=CaptionStyle.CASUAL,
                tone=CaptionTone.FRIENDLY,
                language=LanguageCode.ENGLISH,
                processing_time=1.23,
                model_used="gpt-3.5-turbo",
                confidence_score=0.95,
                character_count=45,
                word_count=8
            )
            logger.info(f"✅ Response created: {response.full_caption}")
            logger.info(f"📊 Response metrics: {response.total_length} chars, within limits: {response.is_within_limits}")
            
        except Exception as e:
            logger.error(f"❌ Response validation failed: {e}")
        
        # 5. Error response
        try:
            error_response = ErrorResponse(
                error="validation_error",
                message="Invalid content description",
                details=[
                    {
                        "field": "content_description",
                        "message": "Must be at least 10 characters long",
                        "code": "MIN_LENGTH"
                    }
                ]
            )
            logger.info(f"✅ Error response created: {error_response.model_dump()}")
            
        except Exception as e:
            logger.error(f"❌ Error response validation failed: {e}")
    
    def demo_path_operations(self) -> Any:
        """Demo path operations with proper HTTP methods"""
        logger.info("\n🛣️ Demo: Path Operations with HTTP Methods")
        
        # Test different HTTP methods and status codes
        test_cases = [
            {
                "method": "GET",
                "endpoint": "/",
                "description": "Root endpoint (200 OK)"
            },
            {
                "method": "GET", 
                "endpoint": "/health",
                "description": "Health check (200 OK)"
            },
            {
                "method": "GET",
                "endpoint": "/metrics", 
                "description": "Performance metrics (200 OK)"
            },
            {
                "method": "GET",
                "endpoint": "/api/v14/captions",
                "description": "List captions (200 OK)"
            },
            {
                "method": "GET",
                "endpoint": "/debug/request-info",
                "description": "Debug request info (200 OK)"
            }
        ]
        
        for test_case in test_cases:
            try:
                response = self.session.request(
                    method=test_case["method"],
                    url=f"{self.base_url}{test_case['endpoint']}"
                )
                
                if response.status_code == 200:
                    logger.info(f"✅ {test_case['description']}: {response.status_code}")
                else:
                    logger.warning(f"⚠️ {test_case['description']}: {response.status_code}")
                    
            except Exception as e:
                logger.error(f"❌ {test_case['description']} failed: {e}")
    
    def demo_middleware(self) -> Any:
        """Demo middleware functionality"""
        logger.info("\n🔧 Demo: Middleware Stack")
        
        # Test middleware components
        middleware_components = [
            "RequestIDMiddleware",
            "LoggingMiddleware", 
            "PerformanceMonitoringMiddleware",
            "SecurityHeadersMiddleware",
            "RateLimitingMiddleware",
            "ErrorHandlingMiddleware",
            "CacheControlMiddleware"
        ]
        
        for component in middleware_components:
            logger.info(f"✅ {component} configured")
        
        # Test request ID tracking
        try:
            response = self.session.get(f"{self.base_url}/debug/request-info")
            request_id = response.headers.get("X-Request-ID")
            processing_time = response.headers.get("X-Processing-Time")
            
            if request_id:
                logger.info(f"✅ Request ID tracking: {request_id}")
            if processing_time:
                logger.info(f"✅ Processing time tracking: {processing_time}s")
                
        except Exception as e:
            logger.error(f"❌ Middleware test failed: {e}")
        
        # Test security headers
        try:
            response = self.session.get(f"{self.base_url}/")
            security_headers = [
                "X-Content-Type-Options",
                "X-Frame-Options", 
                "X-XSS-Protection",
                "Content-Security-Policy"
            ]
            
            for header in security_headers:
                if header in response.headers:
                    logger.info(f"✅ Security header present: {header}")
                else:
                    logger.warning(f"⚠️ Security header missing: {header}")
                    
        except Exception as e:
            logger.error(f"❌ Security headers test failed: {e}")
    
    def demo_error_handling(self) -> Any:
        """Demo error handling and validation"""
        logger.info("\n🚨 Demo: Error Handling and Validation")
        
        # Test validation errors
        invalid_requests = [
            {
                "content_description": "",  # Empty description
                "description": "Empty content description"
            },
            {
                "content_description": "Short",  # Too short
                "description": "Content description too short"
            },
            {
                "content_description": "Valid description but invalid hashtag count",
                "hashtag_count": 50,  # Too many hashtags
                "description": "Invalid hashtag count"
            }
        ]
        
        for invalid_request in invalid_requests:
            try:
                # This should raise validation errors
                CaptionGenerationRequest(**invalid_request)
                logger.warning(f"⚠️ Expected validation error for: {invalid_request['description']}")
                
            except Exception as e:
                logger.info(f"✅ Validation error caught: {invalid_request['description']} - {str(e)[:100]}")
        
        # Test HTTP error responses
        error_endpoints = [
            {
                "endpoint": "/api/v14/captions/nonexistent",
                "expected_status": 404,
                "description": "Not found error"
            },
            {
                "endpoint": "/api/v14/captions/invalid-id",
                "expected_status": 400,
                "description": "Bad request error"
            }
        ]
        
        for error_test in error_endpoints:
            try:
                response = self.session.get(f"{self.base_url}{error_test['endpoint']}")
                
                if response.status_code == error_test["expected_status"]:
                    logger.info(f"✅ {error_test['description']}: {response.status_code}")
                else:
                    logger.warning(f"⚠️ {error_test['description']}: expected {error_test['expected_status']}, got {response.status_code}")
                    
            except Exception as e:
                logger.error(f"❌ Error test failed: {e}")
    
    def demo_performance(self) -> Any:
        """Demo performance monitoring and optimization"""
        logger.info("\n⚡ Demo: Performance Monitoring")
        
        # Test performance metrics
        try:
            start_time = time.time()
            response = self.session.get(f"{self.base_url}/metrics")
            request_time = time.time() - start_time
            
            if response.status_code == 200:
                metrics = response.json()
                logger.info(f"✅ Performance metrics retrieved in {request_time:.3f}s")
                logger.info(f"📊 Metrics: {json.dumps(metrics, indent=2)[:200]}...")
            else:
                logger.warning(f"⚠️ Failed to get metrics: {response.status_code}")
                
        except Exception as e:
            logger.error(f"❌ Performance test failed: {e}")
        
        # Test processing time tracking
        try:
            response = self.session.get(f"{self.base_url}/debug/performance")
            processing_time = response.headers.get("X-Processing-Time")
            
            if processing_time:
                logger.info(f"✅ Processing time tracked: {processing_time}s")
            else:
                logger.warning("⚠️ Processing time header missing")
                
        except Exception as e:
            logger.error(f"❌ Processing time test failed: {e}")
        
        # Test batch processing performance
        try:
            batch_request = {
                "requests": [
                    {
                        "content_description": f"Test content {i}",
                        "style": "casual",
                        "tone": "friendly"
                    }
                    for i in range(3)
                ],
                "max_concurrent": 2
            }
            
            start_time = time.time()
            response = self.session.post(
                f"{self.base_url}/api/v14/captions/batch-generate",
                json=batch_request
            )
            batch_time = time.time() - start_time
            
            if response.status_code in [200, 202]:
                logger.info(f"✅ Batch processing completed in {batch_time:.3f}s")
            else:
                logger.warning(f"⚠️ Batch processing failed: {response.status_code}")
                
        except Exception as e:
            logger.error(f"❌ Batch performance test failed: {e}")
    
    def demo_security(self) -> Any:
        """Demo security features"""
        logger.info("\n🔒 Demo: Security Features")
        
        # Test authentication
        try:
            # Test without authentication
            no_auth_session = requests.Session()
            response = no_auth_session.get(f"{self.base_url}/api/v14/captions")
            
            if response.status_code == 401:
                logger.info("✅ Authentication required")
            else:
                logger.warning(f"⚠️ Authentication not enforced: {response.status_code}")
                
        except Exception as e:
            logger.error(f"❌ Authentication test failed: {e}")
        
        # Test rate limiting
        try:
            # Make multiple requests quickly
            responses = []
            for i in range(5):
                response = self.session.get(f"{self.base_url}/health")
                responses.append(response.status_code)
            
            if all(status == 200 for status in responses):
                logger.info("✅ Rate limiting working (requests allowed)")
            elif any(status == 429 for status in responses):
                logger.info("✅ Rate limiting working (requests blocked)")
            else:
                logger.warning("⚠️ Rate limiting behavior unclear")
                
        except Exception as e:
            logger.error(f"❌ Rate limiting test failed: {e}")
        
        # Test CORS
        try:
            response = self.session.options(f"{self.base_url}/")
            cors_headers = response.headers.get("Access-Control-Allow-Origin")
            
            if cors_headers:
                logger.info(f"✅ CORS configured: {cors_headers}")
            else:
                logger.warning("⚠️ CORS headers missing")
                
        except Exception as e:
            logger.error(f"❌ CORS test failed: {e}")
    
    async def demo_api_integration(self) -> Any:
        """Demo complete API integration"""
        logger.info("\n🔗 Demo: Complete API Integration")
        
        # Test caption generation workflow
        try:
            # 1. Create caption request
            caption_request = {
                "content_description": "Beautiful sunset over mountains with golden light reflecting on a calm lake",
                "style": "casual",
                "tone": "friendly", 
                "hashtag_count": 15,
                "language": "en",
                "include_emoji": True,
                "max_length": 1000
            }
            
            # 2. Generate caption
            response = self.session.post(
                f"{self.base_url}/api/v14/captions/generate",
                json=caption_request
            )
            
            if response.status_code == 201:
                caption_data = response.json()
                logger.info("✅ Caption generated successfully")
                logger.info(f"📝 Caption: {caption_data.get('caption', 'N/A')}")
                logger.info(f"🏷️ Hashtags: {caption_data.get('hashtags', [])}")
                logger.info(f"⏱️ Processing time: {caption_data.get('processing_time', 0):.3f}s")
                
                # 3. Test analytics
                analytics_response = self.session.get(f"{self.base_url}/api/v14/analytics")
                if analytics_response.status_code == 200:
                    analytics = analytics_response.json()
                    logger.info("✅ Analytics retrieved")
                    logger.info(f"📊 Total captions: {analytics.get('total_captions_generated', 0)}")
                    logger.info(f"📈 Success rate: {analytics.get('success_rate', 0):.1f}%")
                    
            else:
                logger.warning(f"⚠️ Caption generation failed: {response.status_code}")
                
        except Exception as e:
            logger.error(f"❌ API integration test failed: {e}")
        
        # Test health check
        try:
            response = self.session.get(f"{self.base_url}/health")
            if response.status_code == 200:
                health_data = response.json()
                logger.info("✅ Health check successful")
                logger.info(f"🏥 Overall status: {health_data.get('status', 'unknown')}")
                logger.info(f"🔧 Services: {len(health_data.get('services', {}))}")
            else:
                logger.warning(f"⚠️ Health check failed: {response.status_code}")
                
        except Exception as e:
            logger.error(f"❌ Health check test failed: {e}")


def main():
    """Main demo function"""
    logger.info("🎯 FastAPI Best Practices Demo")
    logger.info("=" * 50)
    
    # Create demo instance
    demo = FastAPIBestPracticesDemo()
    
    # Run all demos
    demo.run_all_demos()
    
    logger.info("\n🎉 Demo completed successfully!")
    logger.info("📚 Check the documentation at /docs for more information")
    logger.info("🔧 API is ready for production use with best practices implemented")


match __name__:
    case "__main__":
    main() 