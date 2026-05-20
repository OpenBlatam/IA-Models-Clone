from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

# Constants
BUFFER_SIZE = 1024

import asyncio
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException, status, Request
from fastapi.responses import JSONResponse
from .pydantic_schemas import (
from .pydantic_validation import (
            from .pydantic_schemas import VideoMetadata
            import time
from typing import Any, List, Dict, Optional
"""
Pydantic Validation Examples
===========================

Comprehensive examples demonstrating how to use Pydantic BaseModel for
consistent input/output validation and response schemas in the AI Video system.

Examples include:
- Basic request/response validation
- Advanced validation with custom validators
- Batch processing validation
- Error handling and transformation
- Performance monitoring
- Middleware integration
"""



    VideoGenerationInput, BatchGenerationInput, VideoEditInput,
    VideoGenerationResponse, BatchGenerationResponse, VideoEditResponse,
    SystemHealth, UserQuota, APIError, VideoStatus, QualityLevel,
    ProcessingPriority, ModelType, VideoFormat,
    create_video_id, create_batch_id, create_error_response, create_success_response
)

    ValidationConfig, PydanticValidationMiddleware,
    validate_request, validate_response, validate_input_output,
    ValidationUtils, ValidationPerformanceMonitor,
    create_validation_middleware, create_performance_monitor
)

# Setup logging
logger = logging.getLogger(__name__)

# =============================================================================
# BASIC VALIDATION EXAMPLES
# =============================================================================

class BasicValidationExamples:
    """Basic validation examples for AI Video system."""
    
    @staticmethod
    def validate_video_generation_request():
        """Example: Validate video generation request."""
        try:
            # Valid request
            valid_request = VideoGenerationInput(
                prompt="A beautiful sunset over mountains",
                num_frames=16,
                height=512,
                width=512,
                quality=QualityLevel.HIGH,
                model_type=ModelType.STABLE_DIFFUSION
            )
            print(f"✅ Valid request: {valid_request.model_dump()}")
            
            # Invalid request - will raise ValidationError
            try:
                invalid_request = VideoGenerationInput(
                    prompt="",  # Empty prompt
                    height=100,  # Invalid height (not divisible by 8)
                    width=100    # Invalid width
                )
            except Exception as e:
                print(f"❌ Invalid request caught: {e}")
                
        except Exception as e:
            print(f"Error in validation example: {e}")
    
    @staticmethod
    def validate_batch_generation_request():
        """Example: Validate batch generation request."""
        try:
            # Create multiple video requests
            video_requests = [
                VideoGenerationInput(
                    prompt="A cat playing with a ball",
                    quality=QualityLevel.MEDIUM
                ),
                VideoGenerationInput(
                    prompt="A dog running in a park",
                    quality=QualityLevel.HIGH
                )
            ]
            
            # Valid batch request
            batch_request = BatchGenerationInput(
                requests=video_requests,
                batch_name="Pet Videos",
                priority=ProcessingPriority.HIGH
            )
            
            print(f"✅ Valid batch request: {batch_request.model_dump()}")
            print(f"📊 Total estimated size: {batch_request.total_estimated_size_mb:.2f} MB")
            
        except Exception as e:
            print(f"Error in batch validation example: {e}")
    
    @staticmethod
    def validate_video_edit_request():
        """Example: Validate video edit request."""
        try:
            # Valid edit request
            edit_request = VideoEditInput(
                video_id="550e8400-e29b-41d4-a716-446655440000",
                operation="resize",
                parameters={
                    "width": 1024,
                    "height": 768
                }
            )
            
            print(f"✅ Valid edit request: {edit_request.model_dump()}")
            
            # Speed adjustment request
            speed_request = VideoEditInput(
                video_id="550e8400-e29b-41d4-a716-446655440000",
                operation="speed",
                parameters={
                    "speed_factor": 2.0
                }
            )
            
            print(f"✅ Speed edit request: {speed_request.model_dump()}")
            
        except Exception as e:
            print(f"Error in edit validation example: {e}")

# =============================================================================
# ADVANCED VALIDATION EXAMPLES
# =============================================================================

class AdvancedValidationExamples:
    """Advanced validation examples with custom logic."""
    
    @staticmethod
    def validate_with_custom_business_rules():
        """Example: Validate with custom business rules."""
        try:
            # Custom validation function
            def validate_prompt_content(prompt: str) -> str:
                """Custom prompt validation."""
                forbidden_words = ['inappropriate', 'explicit', 'nsfw']
                
                for word in forbidden_words:
                    if word.lower() in prompt.lower():
                        raise ValueError(f"Prompt contains forbidden word: {word}")
                
                # Check minimum length
                if len(prompt.strip()) < 10:
                    raise ValueError("Prompt must be at least 10 characters long")
                
                return prompt.strip()
            
            # Test custom validation
            try:
                # Valid prompt
                valid_prompt = validate_prompt_content("A beautiful landscape with mountains and trees")
                print(f"✅ Valid prompt: {valid_prompt}")
                
                # Invalid prompt
                try:
                    invalid_prompt = validate_prompt_content("Short")
                except ValueError as e:
                    print(f"❌ Invalid prompt caught: {e}")
                    
            except Exception as e:
                print(f"Error in custom validation: {e}")
                
        except Exception as e:
            print(f"Error in business rules validation: {e}")
    
    @staticmethod
    def validate_user_quota():
        """Example: Validate user quota and limits."""
        try:
            # Create user quota
            user_quota = UserQuota(
                user_id="user123",
                daily_limit=10,
                daily_used=5,
                daily_reset=datetime.now() + timedelta(hours=6),
                monthly_limit=100,
                monthly_used=25,
                monthly_reset=datetime.now() + timedelta(days=15),
                storage_limit_mb=1024,
                storage_used_mb=256.5,
                max_priority=ProcessingPriority.HIGH
            )
            
            print(f"✅ User quota: {user_quota.model_dump()}")
            print(f"📊 Daily remaining: {user_quota.daily_remaining}")
            print(f"📊 Monthly remaining: {user_quota.monthly_remaining}")
            print(f"📊 Storage remaining: {user_quota.storage_remaining_mb:.2f} MB")
            print(f"📊 Can generate video: {user_quota.can_generate_video}")
            
            # Check quota before generation
            if not user_quota.can_generate_video:
                raise ValueError("User quota exceeded")
                
        except Exception as e:
            print(f"Error in quota validation: {e}")
    
    @staticmethod
    def validate_system_health():
        """Example: Validate system health status."""
        try:
            # Create system health status
            system_health = SystemHealth(
                status="healthy",
                version="1.0.0",
                uptime=3600.0,  # 1 hour
                cpu_usage=45.2,
                memory_usage=67.8,
                gpu_usage=89.1,
                disk_usage=34.5,
                active_requests=12,
                queue_size=5,
                average_response_time=0.85,
                database_status="healthy",
                cache_status="healthy",
                storage_status="healthy"
            )
            
            print(f"✅ System health: {system_health.model_dump()}")
            print(f"📊 Is healthy: {system_health.is_healthy}")
            print(f"📊 Resource usage critical: {system_health.resource_usage_critical}")
            
        except Exception as e:
            print(f"Error in system health validation: {e}")

# =============================================================================
# RESPONSE VALIDATION EXAMPLES
# =============================================================================

class ResponseValidationExamples:
    """Examples of response validation and transformation."""
    
    @staticmethod
    def create_video_generation_response():
        """Example: Create and validate video generation response."""
        try:
            # Create video metadata
            
            metadata = VideoMetadata(
                video_id=create_video_id(),
                prompt="A beautiful sunset over mountains",
                width=512,
                height=512,
                fps=8,
                num_frames=16,
                duration=2.0,
                file_size=2048576,  # 2MB
                format=VideoFormat.MP4,
                guidance_scale=7.5,
                num_inference_steps=50,
                seed=42,
                model_type=ModelType.STABLE_DIFFUSION,
                quality=QualityLevel.HIGH,
                created_at=datetime.now(),
                completed_at=datetime.now() + timedelta(minutes=2)
            )
            
            # Create response
            response = VideoGenerationResponse(
                video_id=metadata.video_id,
                status=VideoStatus.COMPLETED,
                message="Video generation completed successfully",
                video_url="https://example.com/videos/video123.mp4",
                thumbnail_url="https://example.com/thumbnails/video123.jpg",
                progress=100.0,
                metadata=metadata
            )
            
            print(f"✅ Video generation response: {response.model_dump()}")
            print(f"📊 Can download: {response.can_download}")
            print(f"📊 Processing time: {response.metadata.processing_time:.2f} seconds")
            
        except Exception as e:
            print(f"Error in response creation: {e}")
    
    @staticmethod
    def create_batch_generation_response():
        """Example: Create and validate batch generation response."""
        try:
            # Create batch response
            video_ids = [create_video_id() for _ in range(3)]
            
            response = BatchGenerationResponse(
                batch_id=create_batch_id(),
                batch_name="Test Batch",
                video_ids=video_ids,
                total_videos=3,
                completed_videos=2,
                failed_videos=0,
                processing_videos=1,
                overall_progress=66.7,
                status=VideoStatus.PROCESSING,
                message="Batch processing in progress"
            )
            
            print(f"✅ Batch generation response: {response.model_dump()}")
            print(f"📊 Success rate: {response.success_rate:.2%}")
            print(f"📊 Is completed: {response.is_completed}")
            
        except Exception as e:
            print(f"Error in batch response creation: {e}")
    
    @staticmethod
    def create_error_response():
        """Example: Create standardized error response."""
        try:
            # Create error response
            error_response = create_error_response(
                error_code="QUOTA_EXCEEDED",
                error_type="quota_limit",
                message="Daily video generation limit exceeded",
                details={
                    "daily_limit": 10,
                    "daily_used": 10,
                    "reset_time": "2024-01-01T00:00:00Z"
                },
                request_id="req123",
                endpoint="/api/v1/videos/generate"
            )
            
            print(f"✅ Error response: {error_response.model_dump()}")
            print(f"📊 Is retryable: {error_response.is_retryable}")
            
        except Exception as e:
            print(f"Error in error response creation: {e}")

# =============================================================================
# MIDDLEWARE INTEGRATION EXAMPLES
# =============================================================================

class MiddlewareIntegrationExamples:
    """Examples of middleware integration with validation."""
    
    @staticmethod
    def setup_validation_middleware():
        """Example: Setup validation middleware."""
        try:
            # Create validation configuration
            config = ValidationConfig(
                enable_request_validation=True,
                enable_response_validation=True,
                enable_performance_monitoring=True,
                enable_validation_caching=True,
                max_validation_time=1.0,
                detailed_error_messages=True,
                log_validation_errors=True
            )
            
            # Create middleware
            middleware = create_validation_middleware(config)
            
            # Create FastAPI app with middleware
            app = FastAPI(title="AI Video API")
            app.add_middleware(PydanticValidationMiddleware, config=config)
            
            print("✅ Validation middleware setup complete")
            return app
            
        except Exception as e:
            print(f"Error in middleware setup: {e}")
            return None
    
    @staticmethod
    def create_validated_endpoints():
        """Example: Create endpoints with validation decorators."""
        try:
            app = FastAPI(title="AI Video API with Validation")
            
            @app.post("/api/v1/videos/generate")
            @validate_input_output(VideoGenerationInput, VideoGenerationResponse)
            async def generate_video(request: Request, validated_input: VideoGenerationInput):
                """Generate video with input/output validation."""
                try:
                    # Process video generation
                    video_id = create_video_id()
                    
                    # Create response
                    response = create_success_response(
                        video_id=video_id,
                        status=VideoStatus.PROCESSING,
                        message="Video generation started"
                    )
                    
                    return response.model_dump()
                    
                except Exception as e:
                    logger.error(f"Video generation error: {e}")
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail="Video generation failed"
                    )
            
            @app.post("/api/v1/videos/batch")
            @validate_request(BatchGenerationInput)
            async def generate_batch(request: Request, validated_data: BatchGenerationInput):
                """Generate batch with request validation."""
                try:
                    batch_id = create_batch_id()
                    video_ids = [create_video_id() for _ in validated_data.requests]
                    
                    response = BatchGenerationResponse(
                        batch_id=batch_id,
                        video_ids=video_ids,
                        total_videos=len(validated_data.requests),
                        completed_videos=0,
                        failed_videos=0,
                        processing_videos=len(validated_data.requests),
                        overall_progress=0.0,
                        status=VideoStatus.PROCESSING,
                        message="Batch generation started"
                    )
                    
                    return response.model_dump()
                    
                except Exception as e:
                    logger.error(f"Batch generation error: {e}")
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail="Batch generation failed"
                    )
            
            print("✅ Validated endpoints created")
            return app
            
        except Exception as e:
            print(f"Error in endpoint creation: {e}")
            return None

# =============================================================================
# PERFORMANCE MONITORING EXAMPLES
# =============================================================================

class PerformanceMonitoringExamples:
    """Examples of performance monitoring with validation."""
    
    @staticmethod
    async def monitor_validation_performance():
        """Example: Monitor validation performance."""
        try:
            # Create performance monitor
            monitor = create_performance_monitor()
            
            # Simulate multiple validations
            for i in range(10):
                async with monitor.monitor_validation("VideoGenerationInput"):
                    # Simulate validation
                    await asyncio.sleep(0.1)
                    
                    # Simulate some cache hits/misses
                    if i % 3 == 0:
                        monitor.record_cache_hit()
                    else:
                        monitor.record_cache_miss()
            
            # Get performance stats
            stats = monitor.get_stats()
            print(f"✅ Performance stats: {stats}")
            
        except Exception as e:
            print(f"Error in performance monitoring: {e}")
    
    @staticmethod
    def validate_with_performance_tracking():
        """Example: Validate with performance tracking."""
        try:
            
            # Track validation time
            start_time = time.time()
            
            # Perform validation
            request = VideoGenerationInput(
                prompt="A beautiful landscape",
                quality=QualityLevel.HIGH
            )
            
            validation_time = time.time() - start_time
            print(f"✅ Validation completed in {validation_time:.4f} seconds")
            print(f"📊 Validated data: {request.model_dump()}")
            
        except Exception as e:
            print(f"Error in performance tracking: {e}")

# =============================================================================
# ERROR HANDLING EXAMPLES
# =============================================================================

class ErrorHandlingExamples:
    """Examples of error handling with Pydantic validation."""
    
    @staticmethod
    def handle_validation_errors():
        """Example: Handle validation errors gracefully."""
        try:
            # Test cases that will cause validation errors
            test_cases = [
                {
                    "prompt": "",  # Empty prompt
                    "height": 100,  # Invalid height
                    "width": 100    # Invalid width
                },
                {
                    "prompt": "Valid prompt",
                    "height": 512,
                    "width": 512,
                    "quality": "invalid_quality"  # Invalid quality
                },
                {
                    "prompt": "Valid prompt",
                    "height": 512,
                    "width": 2048,  # Invalid aspect ratio
                    "quality": "high"
                }
            ]
            
            for i, test_case in enumerate(test_cases, 1):
                try:
                    validated = VideoGenerationInput(**test_case)
                    print(f"✅ Test case {i}: Valid")
                    
                except Exception as e:
                    # Transform validation error
                    if hasattr(e, 'errors'):
                        transformed_error = ValidationUtils.transform_validation_error(e)
                        print(f"❌ Test case {i}: {transformed_error}")
                    else:
                        print(f"❌ Test case {i}: {e}")
                        
        except Exception as e:
            print(f"Error in error handling: {e}")
    
    @staticmethod
    def create_error_responses():
        """Example: Create different types of error responses."""
        try:
            # Rate limit error
            rate_limit_error = create_error_response(
                error_code="RATE_LIMIT_EXCEEDED",
                error_type="rate_limit",
                message="Too many requests",
                details={"retry_after": 60},
                retry_after=60
            )
            print(f"✅ Rate limit error: {rate_limit_error.model_dump()}")
            
            # Quota error
            quota_error = create_error_response(
                error_code="QUOTA_EXCEEDED",
                error_type="quota_limit",
                message="Daily limit exceeded",
                details={"daily_limit": 10, "daily_used": 10}
            )
            print(f"✅ Quota error: {quota_error.model_dump()}")
            
            # Validation error
            validation_error = create_error_response(
                error_code="VALIDATION_ERROR",
                error_type="validation_failed",
                message="Invalid input data",
                details={"errors": [{"field": "prompt", "message": "Required field"}]}
            )
            print(f"✅ Validation error: {validation_error.model_dump()}")
            
        except Exception as e:
            print(f"Error in error response creation: {e}")

# =============================================================================
# INTEGRATION EXAMPLES
# =============================================================================

class IntegrationExamples:
    """Examples of integrating Pydantic validation with the AI Video system."""
    
    @staticmethod
    def create_complete_workflow():
        """Example: Complete workflow with validation."""
        try:
            # 1. Validate user quota
            user_quota = UserQuota(
                user_id="user123",
                daily_limit=10,
                daily_used=5,
                daily_reset=datetime.now() + timedelta(hours=6),
                monthly_limit=100,
                monthly_used=25,
                monthly_reset=datetime.now() + timedelta(days=15),
                storage_limit_mb=1024,
                storage_used_mb=256.5,
                max_priority=ProcessingPriority.HIGH
            )
            
            if not user_quota.can_generate_video:
                raise ValueError("User quota exceeded")
            
            # 2. Validate video generation request
            video_request = VideoGenerationInput(
                prompt="A beautiful sunset over mountains",
                quality=QualityLevel.HIGH,
                model_type=ModelType.STABLE_DIFFUSION
            )
            
            # 3. Create video generation response
            video_id = create_video_id()
            response = VideoGenerationResponse(
                video_id=video_id,
                status=VideoStatus.PROCESSING,
                message="Video generation started",
                progress=0.0
            )
            
            print("✅ Complete workflow executed successfully")
            print(f"📊 User quota: {user_quota.daily_remaining} daily requests remaining")
            print(f"📊 Video request: {video_request.prompt}")
            print(f"📊 Video response: {response.video_id}")
            
        except Exception as e:
            print(f"Error in complete workflow: {e}")
    
    @staticmethod
    def create_batch_workflow():
        """Example: Batch workflow with validation."""
        try:
            # 1. Validate batch request
            batch_request = BatchGenerationInput(
                requests=[
                    VideoGenerationInput(
                        prompt="A cat playing with a ball",
                        quality=QualityLevel.MEDIUM
                    ),
                    VideoGenerationInput(
                        prompt="A dog running in a park",
                        quality=QualityLevel.HIGH
                    )
                ],
                batch_name="Pet Videos",
                priority=ProcessingPriority.HIGH
            )
            
            # 2. Create batch response
            batch_id = create_batch_id()
            video_ids = [create_video_id() for _ in batch_request.requests]
            
            batch_response = BatchGenerationResponse(
                batch_id=batch_id,
                video_ids=video_ids,
                total_videos=len(batch_request.requests),
                completed_videos=0,
                failed_videos=0,
                processing_videos=len(batch_request.requests),
                overall_progress=0.0,
                status=VideoStatus.PROCESSING,
                message="Batch generation started"
            )
            
            print("✅ Batch workflow executed successfully")
            print(f"📊 Batch ID: {batch_response.batch_id}")
            print(f"📊 Total videos: {batch_response.total_videos}")
            print(f"📊 Estimated size: {batch_request.total_estimated_size_mb:.2f} MB")
            
        except Exception as e:
            print(f"Error in batch workflow: {e}")

# =============================================================================
# MAIN EXAMPLE RUNNER
# =============================================================================

async def run_all_examples():
    """Run all validation examples."""
    print("🚀 Running Pydantic Validation Examples")
    print("=" * 50)
    
    # Basic validation examples
    print("\n📋 Basic Validation Examples:")
    BasicValidationExamples.validate_video_generation_request()
    BasicValidationExamples.validate_batch_generation_request()
    BasicValidationExamples.validate_video_edit_request()
    
    # Advanced validation examples
    print("\n🔧 Advanced Validation Examples:")
    AdvancedValidationExamples.validate_with_custom_business_rules()
    AdvancedValidationExamples.validate_user_quota()
    AdvancedValidationExamples.validate_system_health()
    
    # Response validation examples
    print("\n📤 Response Validation Examples:")
    ResponseValidationExamples.create_video_generation_response()
    ResponseValidationExamples.create_batch_generation_response()
    ResponseValidationExamples.create_error_response()
    
    # Middleware integration examples
    print("\n🔌 Middleware Integration Examples:")
    MiddlewareIntegrationExamples.setup_validation_middleware()
    MiddlewareIntegrationExamples.create_validated_endpoints()
    
    # Performance monitoring examples
    print("\n📊 Performance Monitoring Examples:")
    await PerformanceMonitoringExamples.monitor_validation_performance()
    PerformanceMonitoringExamples.validate_with_performance_tracking()
    
    # Error handling examples
    print("\n⚠️ Error Handling Examples:")
    ErrorHandlingExamples.handle_validation_errors()
    ErrorHandlingExamples.create_error_responses()
    
    # Integration examples
    print("\n🔗 Integration Examples:")
    IntegrationExamples.create_complete_workflow()
    IntegrationExamples.create_batch_workflow()
    
    print("\n✅ All examples completed successfully!")

match __name__:
    case "__main__":
    asyncio.run(run_all_examples()) 