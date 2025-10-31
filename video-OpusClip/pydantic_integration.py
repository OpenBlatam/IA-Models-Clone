"""
Pydantic Integration for Video-OpusClip System

Integration module that connects Pydantic models with existing API endpoints,
processors, validation systems, and error handling for seamless operation.
"""

from __future__ import annotations
from typing import List, Dict, Optional, Any, Union
from datetime import datetime
import asyncio
import json
from pathlib import Path

from fastapi import HTTPException, status
from pydantic import ValidationError as PydanticValidationError
import structlog

from .pydantic_models import (
    VideoClipRequest,
    VideoClipResponse,
    ViralVideoRequest,
    ViralVideoBatchResponse,
    BatchVideoRequest,
    BatchVideoResponse,
    VideoValidationResult,
    BatchValidationResult,
    VideoProcessingConfig,
    ViralProcessingConfig,
    ProcessingMetrics,
    ErrorInfo,
    VideoStatus,
    VideoQuality,
    VideoFormat,
    ProcessingPriority,
    ContentType,
    EngagementType,
    validate_video_request,
    validate_batch_request,
    create_video_clip_request,
    create_viral_video_request,
    create_batch_request,
    create_processing_config
)
from .error_handling import (
    ErrorHandler,
    ErrorCode,
    ValidationError,
    ProcessingError,
    ExternalServiceError,
    ResourceError,
    CriticalSystemError,
    SecurityError,
    ConfigurationError
)
from .validation import (
    validate_video_request_data,
    validate_batch_request_data,
    validate_and_sanitize_url,
    validate_system_health,
    validate_gpu_health,
    check_system_resources,
    check_gpu_availability
)

logger = structlog.get_logger()

# =============================================================================
# PYDANTIC VALIDATION INTEGRATION
# =============================================================================

class PydanticValidationIntegrator:
    """Integrates Pydantic validation with existing validation systems."""
    
    def __init__(self):
        self.error_handler = ErrorHandler()
    
    async def validate_and_convert_request(
        self, 
        request_data: Dict[str, Any], 
        request_type: str = "video_clip"
    ) -> Union[VideoClipRequest, ViralVideoRequest, BatchVideoRequest]:
        """
        Validate request data using Pydantic and convert to appropriate model.
        
        Args:
            request_data: Raw request data dictionary
            request_type: Type of request ("video_clip", "viral", "batch")
            
        Returns:
            Validated Pydantic model instance
            
        Raises:
            ValidationError: If validation fails
        """
        try:
            if request_type == "video_clip":
                return VideoClipRequest(**request_data)
            elif request_type == "viral":
                return ViralVideoRequest(**request_data)
            elif request_type == "batch":
                # Convert nested requests
                if "requests" in request_data:
                    requests_data = request_data["requests"]
                    validated_requests = []
                    for req_data in requests_data:
                        validated_requests.append(VideoClipRequest(**req_data))
                    request_data["requests"] = validated_requests
                return BatchVideoRequest(**request_data)
            else:
                raise ValidationError(f"Unknown request type: {request_type}")
                
        except PydanticValidationError as e:
            # Convert Pydantic validation errors to our format
            errors = []
            for error in e.errors():
                field_name = " -> ".join(str(loc) for loc in error["loc"])
                error_msg = f"{field_name}: {error['msg']}"
                errors.append(error_msg)
            
            raise ValidationError(
                f"Pydantic validation failed: {', '.join(errors)}",
                "pydantic_validation",
                request_data,
                ErrorCode.INVALID_REQUEST_DATA
            )
    
    async def validate_request_with_legacy_system(
        self, 
        pydantic_request: Union[VideoClipRequest, ViralVideoRequest, BatchVideoRequest]
    ) -> VideoValidationResult:
        """
        Validate Pydantic request using legacy validation system for compatibility.
        
        Args:
            pydantic_request: Validated Pydantic request model
            
        Returns:
            Validation result with detailed feedback
        """
        if isinstance(pydantic_request, VideoClipRequest):
            return await self._validate_video_clip_request(pydantic_request)
        elif isinstance(pydantic_request, ViralVideoRequest):
            return await self._validate_viral_request(pydantic_request)
        elif isinstance(pydantic_request, BatchVideoRequest):
            return await self._validate_batch_request(pydantic_request)
        else:
            raise ValidationError(f"Unsupported request type: {type(pydantic_request)}")
    
    async def _validate_video_clip_request(self, request: VideoClipRequest) -> VideoValidationResult:
        """Validate video clip request using legacy system."""
        try:
            # Use legacy validation functions
            validate_video_request_data(
                youtube_url=request.youtube_url,
                language=request.language,
                max_clip_length=request.max_clip_length,
                min_clip_length=request.min_clip_length,
                audience_profile=request.audience_profile
            )
            
            # Use Pydantic validation result
            return validate_video_request(request)
            
        except Exception as e:
            return VideoValidationResult(
                is_valid=False,
                errors=[f"Legacy validation failed: {str(e)}"],
                overall_score=0.0
            )
    
    async def _validate_viral_request(self, request: ViralVideoRequest) -> VideoValidationResult:
        """Validate viral video request."""
        # First validate as regular video request
        base_validation = await self._validate_video_clip_request(request)
        
        if not base_validation.is_valid:
            return base_validation
        
        # Additional viral-specific validation
        errors = base_validation.errors.copy()
        warnings = base_validation.warnings.copy()
        
        if request.n_variants > 20:
            warnings.append("Large number of variants may impact performance")
        
        if request.use_langchain and not request.viral_optimization:
            warnings.append("LangChain enabled but viral optimization disabled")
        
        return VideoValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            suggestions=base_validation.suggestions,
            quality_score=base_validation.quality_score,
            duration_score=base_validation.duration_score,
            format_score=base_validation.format_score,
            overall_score=base_validation.overall_score
        )
    
    async def _validate_batch_request(self, request: BatchVideoRequest) -> BatchValidationResult:
        """Validate batch request using legacy system."""
        try:
            # Use legacy batch validation
            validate_batch_request_data(
                requests=[{
                    "youtube_url": req.youtube_url,
                    "language": req.language,
                    "max_clip_length": req.max_clip_length,
                    "min_clip_length": req.min_clip_length,
                    "audience_profile": req.audience_profile
                } for req in request.requests],
                batch_size=len(request.requests)
            )
            
            # Use Pydantic validation result
            return validate_batch_request(request)
            
        except Exception as e:
            return BatchValidationResult(
                is_valid=False,
                errors=[f"Legacy batch validation failed: {str(e)}"],
                overall_score=0.0
            )

# =============================================================================
# API INTEGRATION
# =============================================================================

class PydanticAPIIntegrator:
    """Integrates Pydantic models with FastAPI endpoints."""
    
    def __init__(self):
        self.validator = PydanticValidationIntegrator()
    
    async def process_video_clip_request(
        self, 
        request_data: Dict[str, Any],
        processor: Any = None
    ) -> VideoClipResponse:
        """
        Process video clip request using Pydantic models.
        
        Args:
            request_data: Raw request data
            processor: Video processor instance
            
        Returns:
            VideoClipResponse with processing results
        """
        try:
            # Validate request with Pydantic
            pydantic_request = await self.validator.validate_and_convert_request(
                request_data, "video_clip"
            )
            
            # Additional validation with legacy system
            validation_result = await self.validator.validate_request_with_legacy_system(
                pydantic_request
            )
            
            if not validation_result.is_valid:
                raise ValidationError(
                    f"Request validation failed: {', '.join(validation_result.errors)}",
                    "request_validation",
                    request_data,
                    ErrorCode.INVALID_REQUEST_DATA
                )
            
            # Process the request (simulate processing)
            start_time = datetime.now()
            
            # Here you would integrate with your actual processor
            if processor:
                # result = await processor.process(pydantic_request)
                pass
            
            # Simulate processing result
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return VideoClipResponse(
                success=True,
                clip_id=f"clip_{pydantic_request.request_hash}",
                request_id=pydantic_request.request_hash,
                title="Processed Video",
                description="Video processed successfully",
                duration=pydantic_request.max_clip_length,
                language=pydantic_request.language,
                file_path="/path/to/processed/video.mp4",
                file_size=1024 * 1024 * 10,  # 10MB
                resolution="1920x1080",
                fps=30.0,
                bitrate=5000,
                processing_time=processing_time,
                quality=pydantic_request.quality,
                format=pydantic_request.format,
                status=VideoStatus.COMPLETED,
                warnings=validation_result.warnings,
                metadata={
                    "video_id": pydantic_request.video_id,
                    "validation_score": validation_result.overall_score
                }
            )
            
        except ValidationError as e:
            logger.warning("Request validation failed", error=str(e))
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": "Validation failed",
                    "message": str(e),
                    "field": e.field_name,
                    "code": e.error_code
                }
            )
        except Exception as e:
            logger.error("Video processing failed", error=str(e))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={
                    "error": "Processing failed",
                    "message": str(e)
                }
            )
    
    async def process_viral_video_request(
        self, 
        request_data: Dict[str, Any],
        processor: Any = None
    ) -> ViralVideoBatchResponse:
        """
        Process viral video request using Pydantic models.
        
        Args:
            request_data: Raw request data
            processor: Viral processor instance
            
        Returns:
            ViralVideoBatchResponse with generated variants
        """
        try:
            # Validate request with Pydantic
            pydantic_request = await self.validator.validate_and_convert_request(
                request_data, "viral"
            )
            
            # Additional validation
            validation_result = await self.validator.validate_request_with_legacy_system(
                pydantic_request
            )
            
            if not validation_result.is_valid:
                raise ValidationError(
                    f"Viral request validation failed: {', '.join(validation_result.errors)}",
                    "viral_validation",
                    request_data,
                    ErrorCode.INVALID_REQUEST_DATA
                )
            
            # Process viral variants (simulate)
            start_time = datetime.now()
            variants = []
            
            for i in range(pydantic_request.n_variants):
                variant = await self._create_viral_variant(pydantic_request, i + 1)
                variants.append(variant)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Calculate metrics
            viral_scores = [v.viral_score for v in variants]
            average_viral_score = sum(viral_scores) / len(viral_scores) if viral_scores else 0.0
            best_viral_score = max(viral_scores) if viral_scores else 0.0
            
            return ViralVideoBatchResponse(
                success=True,
                original_clip_id=pydantic_request.request_hash,
                batch_id=f"viral_batch_{pydantic_request.request_hash}",
                variants=variants,
                total_variants_generated=pydantic_request.n_variants,
                successful_variants=len(variants),
                processing_time=processing_time,
                average_viral_score=average_viral_score,
                best_viral_score=best_viral_score,
                warnings=validation_result.warnings,
                metadata={
                    "video_id": pydantic_request.video_id,
                    "validation_score": validation_result.overall_score,
                    "use_langchain": pydantic_request.use_langchain,
                    "viral_optimization": pydantic_request.viral_optimization
                }
            )
            
        except ValidationError as e:
            logger.warning("Viral request validation failed", error=str(e))
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": "Viral validation failed",
                    "message": str(e),
                    "field": e.field_name,
                    "code": e.error_code
                }
            )
        except Exception as e:
            logger.error("Viral video processing failed", error=str(e))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={
                    "error": "Viral processing failed",
                    "message": str(e)
                }
            )
    
    async def process_batch_request(
        self, 
        request_data: Dict[str, Any],
        processor: Any = None
    ) -> BatchVideoResponse:
        """
        Process batch video request using Pydantic models.
        
        Args:
            request_data: Raw request data
            processor: Batch processor instance
            
        Returns:
            BatchVideoResponse with batch processing results
        """
        try:
            # Validate request with Pydantic
            pydantic_request = await self.validator.validate_and_convert_request(
                request_data, "batch"
            )
            
            # Additional validation
            validation_result = await self.validator.validate_request_with_legacy_system(
                pydantic_request
            )
            
            if not validation_result.is_valid:
                raise ValidationError(
                    f"Batch request validation failed: {', '.join(validation_result.errors)}",
                    "batch_validation",
                    request_data,
                    ErrorCode.INVALID_REQUEST_DATA
                )
            
            # Process batch (simulate)
            start_time = datetime.now()
            results = []
            successful_requests = 0
            failed_requests = 0
            
            for request in pydantic_request.requests:
                try:
                    # Process individual request
                    result = await self.process_video_clip_request(
                        request.model_dump(), processor
                    )
                    results.append(result)
                    if result.success:
                        successful_requests += 1
                    else:
                        failed_requests += 1
                except Exception as e:
                    # Create error response for failed request
                    error_result = VideoClipResponse(
                        success=False,
                        request_id=request.request_hash,
                        status=VideoStatus.FAILED,
                        error=str(e),
                        processing_time=0.0
                    )
                    results.append(error_result)
                    failed_requests += 1
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return BatchVideoResponse(
                success=failed_requests == 0,
                batch_id=pydantic_request.batch_id or f"batch_{start_time.timestamp()}",
                results=results,
                total_requests=len(pydantic_request.requests),
                successful_requests=successful_requests,
                failed_requests=failed_requests,
                processing_time=processing_time,
                average_processing_time=processing_time / len(pydantic_request.requests) if pydantic_request.requests else 0.0,
                errors=validation_result.errors,
                warnings=validation_result.warnings
            )
            
        except ValidationError as e:
            logger.warning("Batch request validation failed", error=str(e))
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": "Batch validation failed",
                    "message": str(e),
                    "field": e.field_name,
                    "code": e.error_code
                }
            )
        except Exception as e:
            logger.error("Batch processing failed", error=str(e))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={
                    "error": "Batch processing failed",
                    "message": str(e)
                }
            )
    
    async def _create_viral_variant(
        self, 
        request: ViralVideoRequest, 
        variant_number: int
    ) -> Any:
        """Create a viral video variant (simulated)."""
        from .pydantic_models import ViralVideoVariant, ContentType, EngagementType
        
        # Simulate variant generation
        viral_score = 0.6 + (variant_number * 0.05)  # Increasing scores
        engagement_score = 0.5 + (variant_number * 0.03)
        retention_score = 0.7 + (variant_number * 0.02)
        
        return ViralVideoVariant(
            variant_id=f"variant_{request.request_hash}_{variant_number}",
            title=f"Viral Variant {variant_number}",
            description=f"Optimized viral variant {variant_number}",
            viral_score=min(viral_score, 1.0),
            engagement_prediction=min(engagement_score, 1.0),
            retention_score=min(retention_score, 1.0),
            duration=request.max_clip_length,
            content_type=ContentType.ENTERTAINMENT,
            engagement_type=EngagementType.VIRAL_POTENTIAL,
            target_audience=["18-24", "25-34"],
            viral_hooks=[f"Hook {variant_number}"],
            trending_elements=[f"Trend {variant_number}"],
            hashtags=[f"#viral{variant_number}", "#trending"],
            generation_time=1.5
        )

# =============================================================================
# CONFIGURATION INTEGRATION
# =============================================================================

class PydanticConfigIntegrator:
    """Integrates Pydantic configuration models with existing systems."""
    
    def __init__(self):
        self.default_config = create_processing_config()
        self.default_viral_config = ViralProcessingConfig()
    
    def create_config_from_dict(self, config_data: Dict[str, Any]) -> VideoProcessingConfig:
        """Create VideoProcessingConfig from dictionary."""
        try:
            return VideoProcessingConfig(**config_data)
        except PydanticValidationError as e:
            logger.warning("Invalid config data", errors=str(e))
            return self.default_config
    
    def create_viral_config_from_dict(self, config_data: Dict[str, Any]) -> ViralProcessingConfig:
        """Create ViralProcessingConfig from dictionary."""
        try:
            return ViralProcessingConfig(**config_data)
        except PydanticValidationError as e:
            logger.warning("Invalid viral config data", errors=str(e))
            return self.default_viral_config
    
    def merge_configs(
        self, 
        base_config: VideoProcessingConfig, 
        override_config: Dict[str, Any]
    ) -> VideoProcessingConfig:
        """Merge base config with override values."""
        try:
            # Convert base config to dict
            base_dict = base_config.model_dump()
            
            # Update with override values
            base_dict.update(override_config)
            
            # Create new config
            return VideoProcessingConfig(**base_dict)
        except PydanticValidationError as e:
            logger.warning("Config merge failed", errors=str(e))
            return base_config
    
    def validate_config(self, config: VideoProcessingConfig) -> bool:
        """Validate configuration settings."""
        try:
            # Pydantic validation is automatic, but we can add custom checks
            if config.max_workers > 32:
                logger.warning("Max workers exceeds recommended limit")
                return False
            
            if config.timeout > 3600:
                logger.warning("Timeout exceeds maximum allowed")
                return False
            
            return True
        except Exception as e:
            logger.error("Config validation failed", error=str(e))
            return False

# =============================================================================
# ERROR HANDLING INTEGRATION
# =============================================================================

class PydanticErrorIntegrator:
    """Integrates Pydantic error handling with existing error systems."""
    
    def __init__(self):
        self.error_handler = ErrorHandler()
    
    def convert_pydantic_error(self, pydantic_error: PydanticValidationError) -> ErrorInfo:
        """Convert Pydantic validation error to ErrorInfo."""
        errors = []
        for error in pydantic_error.errors():
            field_name = " -> ".join(str(loc) for loc in error["loc"])
            error_msg = f"{field_name}: {error['msg']}"
            errors.append(error_msg)
        
        return ErrorInfo(
            error_code="PYDANTIC_VALIDATION_ERROR",
            error_message="; ".join(errors),
            error_type="validation_error",
            field_name=errors[0].split(":")[0] if errors else None,
            field_value=None,
            additional_context={
                "pydantic_errors": pydantic_error.errors(),
                "model": pydantic_error.model.__name__ if hasattr(pydantic_error, 'model') else None
            }
        )
    
    def create_validation_error_from_pydantic(
        self, 
        pydantic_error: PydanticValidationError,
        request_data: Any
    ) -> ValidationError:
        """Create ValidationError from Pydantic error."""
        error_info = self.convert_pydantic_error(pydantic_error)
        
        return ValidationError(
            error_info.error_message,
            error_info.field_name or "pydantic_validation",
            request_data,
            ErrorCode.INVALID_REQUEST_DATA
        )
    
    def handle_pydantic_error(
        self, 
        pydantic_error: PydanticValidationError,
        context: str = "validation"
    ) -> Dict[str, Any]:
        """Handle Pydantic error and return user-friendly response."""
        error_info = self.convert_pydantic_error(pydantic_error)
        
        logger.warning(
            f"Pydantic {context} error",
            error_code=error_info.error_code,
            error_message=error_info.error_message,
            field_name=error_info.field_name
        )
        
        return {
            "success": False,
            "error": {
                "code": error_info.error_code,
                "message": error_info.error_message,
                "field": error_info.field_name,
                "type": "validation_error",
                "timestamp": error_info.timestamp.isoformat()
            },
            "suggestions": self._generate_error_suggestions(error_info)
        }
    
    def _generate_error_suggestions(self, error_info: ErrorInfo) -> List[str]:
        """Generate helpful suggestions based on error type."""
        suggestions = []
        
        if "youtube_url" in (error_info.field_name or "").lower():
            suggestions.extend([
                "Ensure the URL is a valid YouTube video URL",
                "Check that the video is publicly accessible",
                "Try using the standard YouTube watch URL format"
            ])
        
        if "language" in (error_info.field_name or "").lower():
            suggestions.append("Use a supported language code (e.g., 'en', 'es', 'fr')")
        
        if "duration" in (error_info.field_name or "").lower():
            suggestions.extend([
                "Duration must be between 3 and 600 seconds",
                "For short-form content, try 15-60 seconds"
            ])
        
        if "quality" in (error_info.field_name or "").lower():
            suggestions.append("Choose from: low, medium, high, ultra")
        
        return suggestions

# =============================================================================
# SERIALIZATION INTEGRATION
# =============================================================================

class PydanticSerializationIntegrator:
    """Integrates Pydantic serialization with existing systems."""
    
    def __init__(self):
        pass
    
    def serialize_response(self, response: Any) -> Dict[str, Any]:
        """Serialize Pydantic response to dictionary."""
        try:
            if hasattr(response, 'model_dump'):
                return response.model_dump()
            elif hasattr(response, 'dict'):
                return response.dict()
            else:
                return response
        except Exception as e:
            logger.error("Serialization failed", error=str(e))
            return {"error": "Serialization failed", "message": str(e)}
    
    def serialize_for_api(self, response: Any) -> Dict[str, Any]:
        """Serialize response specifically for API endpoints."""
        try:
            if hasattr(response, 'model_dump'):
                data = response.model_dump()
                
                # Add computed fields if available
                if hasattr(response, 'is_successful'):
                    data['is_successful'] = response.is_successful
                if hasattr(response, 'has_warnings'):
                    data['has_warnings'] = response.has_warnings
                if hasattr(response, 'file_size_mb'):
                    data['file_size_mb'] = response.file_size_mb
                
                return data
            else:
                return response
        except Exception as e:
            logger.error("API serialization failed", error=str(e))
            return {"error": "Serialization failed", "message": str(e)}
    
    def serialize_for_cache(self, obj: Any) -> str:
        """Serialize object for caching."""
        try:
            if hasattr(obj, 'model_dump_json'):
                return obj.model_dump_json()
            elif hasattr(obj, 'json'):
                return obj.json()
            else:
                return json.dumps(obj)
        except Exception as e:
            logger.error("Cache serialization failed", error=str(e))
            return json.dumps({"error": "Serialization failed"})
    
    def deserialize_from_cache(self, data: str, model_class: type) -> Any:
        """Deserialize object from cache."""
        try:
            if hasattr(model_class, 'model_validate_json'):
                return model_class.model_validate_json(data)
            elif hasattr(model_class, 'parse_raw'):
                return model_class.parse_raw(data)
            else:
                return json.loads(data)
        except Exception as e:
            logger.error("Cache deserialization failed", error=str(e))
            return None

# =============================================================================
# MAIN INTEGRATION CLASS
# =============================================================================

class VideoOpusClipPydanticIntegration:
    """Main integration class for Pydantic with Video-OpusClip system."""
    
    def __init__(self):
        self.validator = PydanticValidationIntegrator()
        self.api_integrator = PydanticAPIIntegrator()
        self.config_integrator = PydanticConfigIntegrator()
        self.error_integrator = PydanticErrorIntegrator()
        self.serializer = PydanticSerializationIntegrator()
    
    async def process_request(
        self, 
        request_data: Dict[str, Any], 
        request_type: str = "video_clip",
        processor: Any = None
    ) -> Dict[str, Any]:
        """
        Main entry point for processing requests with Pydantic integration.
        
        Args:
            request_data: Raw request data
            request_type: Type of request
            processor: Optional processor instance
            
        Returns:
            Serialized response dictionary
        """
        try:
            # Process based on request type
            if request_type == "video_clip":
                response = await self.api_integrator.process_video_clip_request(
                    request_data, processor
                )
            elif request_type == "viral":
                response = await self.api_integrator.process_viral_video_request(
                    request_data, processor
                )
            elif request_type == "batch":
                response = await self.api_integrator.process_batch_request(
                    request_data, processor
                )
            else:
                raise ValueError(f"Unknown request type: {request_type}")
            
            # Serialize response
            return self.serializer.serialize_for_api(response)
            
        except PydanticValidationError as e:
            return self.error_integrator.handle_pydantic_error(e, "request_processing")
        except Exception as e:
            logger.error("Request processing failed", error=str(e))
            return {
                "success": False,
                "error": {
                    "code": "PROCESSING_ERROR",
                    "message": str(e),
                    "type": "processing_error"
                }
            }
    
    def create_request_model(
        self, 
        youtube_url: str, 
        **kwargs
    ) -> VideoClipRequest:
        """Create a VideoClipRequest model with validation."""
        return create_video_clip_request(youtube_url, **kwargs)
    
    def create_viral_request_model(
        self, 
        youtube_url: str, 
        **kwargs
    ) -> ViralVideoRequest:
        """Create a ViralVideoRequest model with validation."""
        return create_viral_video_request(youtube_url, **kwargs)
    
    def create_batch_request_model(
        self, 
        requests: List[VideoClipRequest], 
        **kwargs
    ) -> BatchVideoRequest:
        """Create a BatchVideoRequest model with validation."""
        return create_batch_request(requests, **kwargs)
    
    def validate_request(self, request: Any) -> VideoValidationResult:
        """Validate a request model."""
        if isinstance(request, VideoClipRequest):
            return validate_video_request(request)
        elif isinstance(request, BatchVideoRequest):
            return validate_batch_request(request)
        else:
            raise ValueError(f"Unsupported request type: {type(request)}")

# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_pydantic_integration() -> VideoOpusClipPydanticIntegration:
    """Create and configure Pydantic integration instance."""
    return VideoOpusClipPydanticIntegration() 