from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
import asyncio
import time
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from .error_middleware import (
from .http_exceptions import (
from typing import Any, List, Dict, Optional
"""
🚀 ERROR MIDDLEWARE EXAMPLES - REAL-WORLD AI VIDEO SCENARIOS
===========================================================

Practical examples of error middleware usage in AI Video applications:
- Middleware setup and configuration
- Error handling in video processing
- Performance monitoring
- Circuit breaker patterns
- Alerting and monitoring
- Real-world error scenarios
"""


    MiddlewareStack,
    ErrorTracker,
    StructuredLoggingMiddleware,
    ErrorHandlingMiddleware,
    PerformanceMonitoringMiddleware,
    ErrorType,
    ErrorAction,
    ErrorInfo,
    ErrorSeverity
)

    AIVideoHTTPException,
    SystemError,
    VideoGenerationError,
    ModelLoadError,
    DatabaseError,
    CacheError,
    RateLimitError,
    TimeoutError,
    MemoryError
)

logger = logging.getLogger(__name__)

# ============================================================================
# EXAMPLE 1: COMPLETE MIDDLEWARE SETUP
# ============================================================================

def create_ai_video_app_with_middleware() -> FastAPI:
    """Create FastAPI app with comprehensive error middleware."""
    
    app = FastAPI(
        title="AI Video API with Error Middleware",
        description="Comprehensive error handling and monitoring",
        version="1.0.0"
    )
    
    # Create middleware stack
    middleware_stack = MiddlewareStack()
    
    # Add custom middleware configurations
    middleware_stack.add_middleware(
        StructuredLoggingMiddleware,
        log_level="INFO"
    )
    
    middleware_stack.add_middleware(
        ErrorHandlingMiddleware,
        error_tracker=middleware_stack.error_tracker
    )
    
    middleware_stack.add_middleware(
        PerformanceMonitoringMiddleware
    )
    
    # Apply middleware stack
    app = middleware_stack.create_middleware_stack(app)
    
    # Store middleware stack for access in routes
    app.state.middleware_stack = middleware_stack
    
    # Add routes
    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {
            "status": "healthy",
            "timestamp": time.time(),
            "service": "ai_video_api"
        }
    
    @app.get("/videos/{video_id}")
    async def get_video(video_id: str, request: Request):
        """Get video endpoint with error handling."""
        # Simulate different error scenarios
        if video_id == "error_video":
            raise ValueError("Invalid video ID format")
        
        if video_id == "timeout_video":
            await asyncio.sleep(10)  # Simulate timeout
        
        if video_id == "memory_video":
            # Simulate memory error
            raise MemoryError("Insufficient memory")
        
        if video_id == "database_video":
            raise DatabaseError("Database connection failed")
        
        if video_id == "model_video":
            raise ModelLoadError("stable-diffusion", "Model loading failed")
        
        return {
            "video_id": video_id,
            "status": "available",
            "url": f"/videos/{video_id}.mp4"
        }
    
    @app.post("/videos/generate")
    async def generate_video(request: Request):
        """Generate video endpoint with error handling."""
        # Simulate video generation with potential errors
        await asyncio.sleep(2)  # Simulate processing time
        
        # Simulate different error scenarios based on request
        body = await request.json()
        
        if body.get("prompt") == "error_prompt":
            raise VideoGenerationError(
                detail="Generation failed due to inappropriate content",
                video_id=body.get("video_id"),
                model_name=body.get("model_name")
            )
        
        if body.get("model_name") == "invalid_model":
            raise ModelLoadError(
                model_name="invalid_model",
                detail="Model not found"
            )
        
        if body.get("timeout") == True:
            await asyncio.sleep(30)  # Simulate timeout
        
        return {
            "video_id": body.get("video_id"),
            "status": "generated",
            "processing_time": 2.0
        }
    
    @app.get("/errors/stats")
    async def get_error_stats(request: Request):
        """Get error statistics endpoint."""
        middleware_stack = request.app.state.middleware_stack
        return middleware_stack.get_error_stats()
    
    @app.get("/performance/stats")
    async def get_performance_stats(request: Request):
        """Get performance statistics endpoint."""
        # This would access the performance middleware metrics
        return {
            "message": "Performance stats endpoint",
            "timestamp": time.time()
        }
    
    @app.get("/circuit-breakers")
    async def get_circuit_breakers(request: Request):
        """Get circuit breaker status."""
        middleware_stack = request.app.state.middleware_stack
        return {
            "circuit_breakers": middleware_stack.error_tracker.circuit_breakers,
            "timestamp": time.time()
        }
    
    return app

# ============================================================================
# EXAMPLE 2: ERROR SCENARIOS AND TESTING
# ============================================================================

async def test_error_scenarios():
    """Test various error scenarios with middleware."""
    
    app = create_ai_video_app_with_middleware()
    client = TestClient(app)
    
    print("=== Testing Error Scenarios ===\n")
    
    # Test 1: Normal request
    print("1. Testing normal request...")
    response = client.get("/health")
    print(f"   Status: {response.status_code}")
    print(f"   Response: {response.json()}\n")
    
    # Test 2: Validation error
    print("2. Testing validation error...")
    response = client.get("/videos/error_video")
    print(f"   Status: {response.status_code}")
    print(f"   Response: {response.json()}\n")
    
    # Test 3: Memory error
    print("3. Testing memory error...")
    response = client.get("/videos/memory_video")
    print(f"   Status: {response.status_code}")
    print(f"   Response: {response.json()}\n")
    
    # Test 4: Database error
    print("4. Testing database error...")
    response = client.get("/videos/database_video")
    print(f"   Status: {response.status_code}")
    print(f"   Response: {response.json()}\n")
    
    # Test 5: Model error
    print("5. Testing model error...")
    response = client.get("/videos/model_video")
    print(f"   Status: {response.status_code}")
    print(f"   Response: {response.json()}\n")
    
    # Test 6: Video generation error
    print("6. Testing video generation error...")
    response = client.post("/videos/generate", json={
        "video_id": "test_video",
        "prompt": "error_prompt",
        "model_name": "stable-diffusion"
    })
    print(f"   Status: {response.status_code}")
    print(f"   Response: {response.json()}\n")
    
    # Test 7: Get error statistics
    print("7. Getting error statistics...")
    response = client.get("/errors/stats")
    print(f"   Status: {response.status_code}")
    print(f"   Response: {response.json()}\n")
    
    # Test 8: Get circuit breakers
    print("8. Getting circuit breakers...")
    response = client.get("/circuit-breakers")
    print(f"   Status: {response.status_code}")
    print(f"   Response: {response.json()}\n")

# ============================================================================
# EXAMPLE 3: CIRCUIT BREAKER PATTERNS
# ============================================================================

class CircuitBreakerExample:
    """Example of circuit breaker patterns in video processing."""
    
    def __init__(self) -> Any:
        self.error_tracker = ErrorTracker()
        self.request_count = 0
    
    async def process_video_with_circuit_breaker(self, video_id: str) -> Dict[str, Any]:
        """Process video with circuit breaker protection."""
        
        # Check if circuit breaker is active for video processing
        if self.error_tracker.is_circuit_broken(ErrorType.MODEL):
            return {
                "error": "Service temporarily unavailable due to model errors",
                "status": "circuit_broken",
                "video_id": video_id
            }
        
        try:
            # Simulate video processing
            self.request_count += 1
            
            # Simulate errors for demonstration
            if self.request_count % 3 == 0:  # Every 3rd request fails
                raise ModelLoadError("stable-diffusion", "Model loading failed")
            
            # Simulate successful processing
            await asyncio.sleep(1)
            
            return {
                "video_id": video_id,
                "status": "processed",
                "processing_time": 1.0
            }
            
        except Exception as exc:
            # Record error for circuit breaker
            error_info = ErrorInfo(
                error_type=ErrorType.MODEL,
                severity=ErrorSeverity.HIGH,
                action=ErrorAction.ALERT,
                alert_threshold=3,
                circuit_break_threshold=5
            )
            
            context = {
                "video_id": video_id,
                "request_count": self.request_count,
                "error_type": exc.__class__.__name__
            }
            
            self.error_tracker.record_error(ErrorType.MODEL, error_info, context)
            
            # Check if circuit breaker should be triggered
            if self.error_tracker.is_circuit_broken(ErrorType.MODEL):
                logger.critical(f"Circuit breaker triggered for video {video_id}")
            
            raise

# ============================================================================
# EXAMPLE 4: PERFORMANCE MONITORING
# ============================================================================

class PerformanceMonitoringExample:
    """Example of performance monitoring in video processing."""
    
    def __init__(self) -> Any:
        self.performance_middleware = PerformanceMonitoringMiddleware(None)
        self.slow_operations = []
    
    async def monitor_video_processing(self, video_id: str, processing_time: float):
        """Monitor video processing performance."""
        
        # Create mock metrics
        metrics = type('Metrics', (), {
            'request_id': f"req_{video_id}",
            'method': 'POST',
            'url': f'/videos/{video_id}/process',
            'duration': processing_time,
            'memory_before': 512.0,
            'memory_after': 768.0,
            'memory_delta': 256.0,
            'cpu_usage': 75.0,
            'status_code': 200,
            'timestamp': time.time()
        })()
        
        # Record metrics
        self.performance_middleware._record_metrics(metrics)
        
        # Check for performance issues
        self.performance_middleware._check_performance_issues(metrics)
        
        # Get performance stats
        stats = self.performance_middleware.get_performance_stats()
        
        return {
            "video_id": video_id,
            "processing_time": processing_time,
            "performance_stats": stats,
            "memory_delta": metrics.memory_delta,
            "cpu_usage": metrics.cpu_usage
        }
    
    async def simulate_performance_issues(self) -> Any:
        """Simulate various performance issues."""
        
        print("=== Simulating Performance Issues ===\n")
        
        # Normal processing
        result = await self.monitor_video_processing("video_1", 1.0)
        print(f"Normal processing: {result['processing_time']}s")
        
        # Slow processing
        result = await self.monitor_video_processing("video_2", 8.0)
        print(f"Slow processing: {result['processing_time']}s")
        
        # High memory usage
        result = await self.monitor_video_processing("video_3", 2.0)
        print(f"High memory processing: {result['memory_delta']}MB increase")
        
        # Get overall stats
        stats = self.performance_middleware.get_performance_stats()
        print(f"\nOverall performance stats: {stats}")

# ============================================================================
# EXAMPLE 5: ERROR RECOVERY STRATEGIES
# ============================================================================

class ErrorRecoveryExample:
    """Example of error recovery strategies."""
    
    def __init__(self) -> Any:
        self.error_tracker = ErrorTracker()
        self.retry_count = 0
        self.max_retries = 3
    
    async def process_with_retry(self, operation: str, video_id: str) -> Dict[str, Any]:
        """Process operation with retry logic."""
        
        for attempt in range(self.max_retries + 1):
            try:
                # Simulate operation
                await asyncio.sleep(1)
                
                # Simulate different error scenarios
                if operation == "load_model" and attempt < 2:
                    raise ModelLoadError("stable-diffusion", "Temporary loading error")
                
                if operation == "generate_video" and attempt < 1:
                    raise VideoGenerationError("Temporary generation error", video_id)
                
                # Success
                return {
                    "operation": operation,
                    "video_id": video_id,
                    "status": "success",
                    "attempt": attempt + 1
                }
                
            except Exception as exc:
                # Record error
                error_info = ErrorInfo(
                    error_type=ErrorType.MODEL if "model" in operation else ErrorType.SYSTEM,
                    severity=ErrorSeverity.MEDIUM,
                    action=ErrorAction.RETRY,
                    retry_count=attempt,
                    max_retries=self.max_retries
                )
                
                context = {
                    "operation": operation,
                    "video_id": video_id,
                    "attempt": attempt + 1,
                    "error": str(exc)
                }
                
                self.error_tracker.record_error(
                    error_info.error_type, 
                    error_info, 
                    context
                )
                
                if attempt == self.max_retries:
                    # Final failure
                    raise
                
                # Wait before retry
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
    
    async def process_with_fallback(self, primary_operation: str, fallback_operation: str, video_id: str):
        """Process with fallback operation."""
        
        try:
            # Try primary operation
            return await self.process_with_retry(primary_operation, video_id)
            
        except Exception as exc:
            logger.warning(f"Primary operation failed, trying fallback: {exc}")
            
            try:
                # Try fallback operation
                return await self.process_with_retry(fallback_operation, video_id)
                
            except Exception as fallback_exc:
                logger.error(f"Both primary and fallback operations failed: {fallback_exc}")
                raise

# ============================================================================
# EXAMPLE 6: ALERTING AND MONITORING
# ============================================================================

class AlertingExample:
    """Example of alerting and monitoring."""
    
    def __init__(self) -> Any:
        self.error_tracker = ErrorTracker()
        self.alert_history = []
    
    def send_alert(self, alert_type: str, message: str, severity: str):
        """Send alert (simulated)."""
        alert = {
            "type": alert_type,
            "message": message,
            "severity": severity,
            "timestamp": time.time()
        }
        
        self.alert_history.append(alert)
        logger.warning(f"ALERT: {alert_type} - {message}")
        
        return alert
    
    async def monitor_error_rates(self) -> Any:
        """Monitor error rates and send alerts."""
        
        # Simulate error recording
        for i in range(10):
            error_info = ErrorInfo(
                error_type=ErrorType.MODEL,
                severity=ErrorSeverity.HIGH,
                action=ErrorAction.ALERT,
                alert_threshold=5
            )
            
            context = {
                "error_count": i + 1,
                "timestamp": time.time()
            }
            
            self.error_tracker.record_error(ErrorType.MODEL, error_info, context)
            
            # Check for alerts
            stats = self.error_tracker.get_error_stats(window_minutes=1)
            
            if stats["error_rate"] > 5:  # More than 5 errors per minute
                self.send_alert(
                    "HIGH_ERROR_RATE",
                    f"Error rate is {stats['error_rate']:.2f} errors/minute",
                    "high"
                )
            
            await asyncio.sleep(0.1)  # Small delay
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get alert summary."""
        return {
            "total_alerts": len(self.alert_history),
            "alerts_by_type": self._group_alerts_by_type(),
            "recent_alerts": self.alert_history[-5:] if self.alert_history else []
        }
    
    def _group_alerts_by_type(self) -> Dict[str, int]:
        """Group alerts by type."""
        grouped = {}
        for alert in self.alert_history:
            alert_type = alert["type"]
            grouped[alert_type] = grouped.get(alert_type, 0) + 1
        return grouped

# ============================================================================
# EXAMPLE 7: INTEGRATED ERROR HANDLING SYSTEM
# ============================================================================

class IntegratedErrorHandlingSystem:
    """Integrated error handling system for AI Video applications."""
    
    def __init__(self) -> Any:
        self.error_tracker = ErrorTracker()
        self.performance_monitor = PerformanceMonitoringExample()
        self.error_recovery = ErrorRecoveryExample()
        self.alerting = AlertingExample()
    
    async async def process_video_request(self, video_id: str, prompt: str, model_name: str) -> Dict[str, Any]:
        """Process video request with comprehensive error handling."""
        
        start_time = time.time()
        
        try:
            # Step 1: Load model with retry
            model_result = await self.error_recovery.process_with_retry(
                "load_model", 
                video_id
            )
            
            # Step 2: Generate video with fallback
            video_result = await self.error_recovery.process_with_fallback(
                "generate_video",
                "generate_video_fallback",
                video_id
            )
            
            # Step 3: Monitor performance
            processing_time = time.time() - start_time
            performance_result = await self.performance_monitor.monitor_video_processing(
                video_id, 
                processing_time
            )
            
            return {
                "video_id": video_id,
                "status": "completed",
                "model_loading": model_result,
                "video_generation": video_result,
                "performance": performance_result,
                "total_time": processing_time
            }
            
        except Exception as exc:
            # Record error
            error_info = ErrorInfo(
                error_type=ErrorType.SYSTEM,
                severity=ErrorSeverity.HIGH,
                action=ErrorAction.ALERT,
                alert_threshold=3
            )
            
            context = {
                "video_id": video_id,
                "prompt": prompt,
                "model_name": model_name,
                "error": str(exc)
            }
            
            self.error_tracker.record_error(ErrorType.SYSTEM, error_info, context)
            
            # Send alert
            self.alerting.send_alert(
                "VIDEO_PROCESSING_FAILED",
                f"Video processing failed for {video_id}: {exc}",
                "high"
            )
            
            raise
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        return {
            "error_stats": self.error_tracker.get_error_stats(),
            "alert_summary": self.alerting.get_alert_summary(),
            "circuit_breakers": self.error_tracker.circuit_breakers,
            "timestamp": time.time()
        }

# ============================================================================
# USAGE EXAMPLES
# ============================================================================

async def run_error_middleware_examples():
    """Run all error middleware examples."""
    
    print("🚀 ERROR MIDDLEWARE EXAMPLES\n")
    print("=" * 50)
    
    # Example 1: Test error scenarios
    print("\n1. Testing Error Scenarios")
    print("-" * 30)
    await test_error_scenarios()
    
    # Example 2: Circuit breaker patterns
    print("\n2. Circuit Breaker Patterns")
    print("-" * 30)
    circuit_breaker = CircuitBreakerExample()
    
    for i in range(8):
        try:
            result = await circuit_breaker.process_video_with_circuit_breaker(f"video_{i}")
            print(f"   Video {i}: {result['status']}")
        except Exception as e:
            print(f"   Video {i}: Error - {e}")
    
    # Example 3: Performance monitoring
    print("\n3. Performance Monitoring")
    print("-" * 30)
    performance_monitor = PerformanceMonitoringExample()
    await performance_monitor.simulate_performance_issues()
    
    # Example 4: Error recovery
    print("\n4. Error Recovery Strategies")
    print("-" * 30)
    error_recovery = ErrorRecoveryExample()
    
    try:
        result = await error_recovery.process_with_retry("load_model", "video_retry")
        print(f"   Retry result: {result}")
    except Exception as e:
        print(f"   Retry failed: {e}")
    
    # Example 5: Alerting
    print("\n5. Alerting and Monitoring")
    print("-" * 30)
    alerting = AlertingExample()
    await alerting.monitor_error_rates()
    alert_summary = alerting.get_alert_summary()
    print(f"   Alert summary: {alert_summary}")
    
    # Example 6: Integrated system
    print("\n6. Integrated Error Handling System")
    print("-" * 30)
    integrated_system = IntegratedErrorHandlingSystem()
    
    try:
        result = await integrated_system.process_video_request(
            "integrated_video",
            "A beautiful sunset",
            "stable-diffusion"
        )
        print(f"   Integrated result: {result['status']}")
    except Exception as e:
        print(f"   Integrated system error: {e}")
    
    system_status = integrated_system.get_system_status()
    print(f"   System status: {system_status}")

if __name__ == "__main__":
    # Run examples
    asyncio.run(run_error_middleware_examples()) 