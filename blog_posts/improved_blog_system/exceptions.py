"""
Advanced Blog Posts System Exceptions
====================================

Custom exception hierarchy for blog posts system with structured error handling.
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime
from enum import Enum


class BlogPostErrorType(str, Enum):
    """Blog post error types"""
    POST_NOT_FOUND = "post_not_found"
    POST_ALREADY_EXISTS = "post_already_exists"
    POST_INVALID_STATUS = "post_invalid_status"
    POST_PERMISSION_DENIED = "post_permission_denied"
    POST_VALIDATION_ERROR = "post_validation_error"
    POST_CONTENT_ERROR = "post_content_error"
    POST_SEO_ERROR = "post_seo_error"
    POST_ANALYTICS_ERROR = "post_analytics_error"
    POST_COLLABORATION_ERROR = "post_collaboration_error"
    POST_WORKFLOW_ERROR = "post_workflow_error"
    POST_TEMPLATE_ERROR = "post_template_error"
    POST_CATEGORY_ERROR = "post_category_error"
    POST_TAG_ERROR = "post_tag_error"
    POST_AUTHOR_ERROR = "post_author_error"
    POST_COMMENT_ERROR = "post_comment_error"
    POST_MEDIA_ERROR = "post_media_error"
    POST_PUBLISHING_ERROR = "post_publishing_error"
    POST_SCHEDULING_ERROR = "post_scheduling_error"
    POST_ARCHIVING_ERROR = "post_archiving_error"
    POST_DELETION_ERROR = "post_deletion_error"
    POST_SYSTEM_ERROR = "post_system_error"


class MLPipelineErrorType(str, Enum):
    """ML pipeline error types"""
    MODEL_NOT_FOUND = "model_not_found"
    MODEL_LOADING_ERROR = "model_loading_error"
    MODEL_INFERENCE_ERROR = "model_inference_error"
    MODEL_TRAINING_ERROR = "model_training_error"
    MODEL_VALIDATION_ERROR = "model_validation_error"
    PIPELINE_EXECUTION_ERROR = "pipeline_execution_error"
    PIPELINE_TIMEOUT = "pipeline_timeout"
    PIPELINE_QUEUE_FULL = "pipeline_queue_full"
    PIPELINE_RESOURCE_ERROR = "pipeline_resource_error"
    PIPELINE_CONFIGURATION_ERROR = "pipeline_configuration_error"


class ContentAnalysisErrorType(str, Enum):
    """Content analysis error types"""
    ANALYSIS_FAILED = "analysis_failed"
    CONTENT_TOO_SHORT = "content_too_short"
    CONTENT_TOO_LONG = "content_too_long"
    CONTENT_INVALID_FORMAT = "content_invalid_format"
    ANALYSIS_TIMEOUT = "analysis_timeout"
    ANALYSIS_RESOURCE_ERROR = "analysis_resource_error"
    ANALYSIS_MODEL_ERROR = "analysis_model_error"
    ANALYSIS_VALIDATION_ERROR = "analysis_validation_error"


class SEOOptimizationErrorType(str, Enum):
    """SEO optimization error types"""
    SEO_ANALYSIS_FAILED = "seo_analysis_failed"
    KEYWORD_ANALYSIS_FAILED = "keyword_analysis_failed"
    META_TAG_ERROR = "meta_tag_error"
    CONTENT_STRUCTURE_ERROR = "content_structure_error"
    SEO_SCORE_CALCULATION_ERROR = "seo_score_calculation_error"
    SEO_RECOMMENDATION_ERROR = "seo_recommendation_error"
    SEO_OPTIMIZATION_FAILED = "seo_optimization_failed"


class ContentGenerationErrorType(str, Enum):
    """Content generation error types"""
    GENERATION_FAILED = "generation_failed"
    AI_SERVICE_ERROR = "ai_service_error"
    PROMPT_VALIDATION_ERROR = "prompt_validation_error"
    CONTENT_VALIDATION_ERROR = "content_validation_error"
    GENERATION_TIMEOUT = "generation_timeout"
    GENERATION_QUOTA_EXCEEDED = "generation_quota_exceeded"
    GENERATION_QUALITY_ERROR = "generation_quality_error"


class BlogPostException(Exception):
    """Base exception for blog posts"""
    
    def __init__(
        self,
        error_type: BlogPostErrorType,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        post_id: Optional[str] = None,
        user_id: Optional[str] = None,
        request_id: Optional[str] = None
    ):
        self.error_type = error_type
        self.message = message
        self.details = details or {}
        self.post_id = post_id
        self.user_id = user_id
        self.request_id = request_id
        self.timestamp = datetime.utcnow()
        
        # Create detailed error message
        error_msg = f"[{error_type.value}] {message}"
        if post_id:
            error_msg += f" (Post: {post_id})"
        if user_id:
            error_msg += f" (User: {user_id})"
        if request_id:
            error_msg += f" (Request: {request_id})"
        
        super().__init__(error_msg)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary"""
        return {
            "error_type": self.error_type.value,
            "message": self.message,
            "details": self.details,
            "post_id": self.post_id,
            "user_id": self.user_id,
            "request_id": self.request_id,
            "timestamp": self.timestamp.isoformat()
        }


class MLPipelineException(Exception):
    """Base exception for ML pipeline"""
    
    def __init__(
        self,
        error_type: MLPipelineErrorType,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        pipeline_id: Optional[str] = None,
        model_name: Optional[str] = None,
        request_id: Optional[str] = None
    ):
        self.error_type = error_type
        self.message = message
        self.details = details or {}
        self.pipeline_id = pipeline_id
        self.model_name = model_name
        self.request_id = request_id
        self.timestamp = datetime.utcnow()
        
        # Create detailed error message
        error_msg = f"[{error_type.value}] {message}"
        if pipeline_id:
            error_msg += f" (Pipeline: {pipeline_id})"
        if model_name:
            error_msg += f" (Model: {model_name})"
        if request_id:
            error_msg += f" (Request: {request_id})"
        
        super().__init__(error_msg)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary"""
        return {
            "error_type": self.error_type.value,
            "message": self.message,
            "details": self.details,
            "pipeline_id": self.pipeline_id,
            "model_name": self.model_name,
            "request_id": self.request_id,
            "timestamp": self.timestamp.isoformat()
        }


class ContentAnalysisException(Exception):
    """Base exception for content analysis"""
    
    def __init__(
        self,
        error_type: ContentAnalysisErrorType,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        content_hash: Optional[str] = None,
        analysis_id: Optional[str] = None,
        request_id: Optional[str] = None
    ):
        self.error_type = error_type
        self.message = message
        self.details = details or {}
        self.content_hash = content_hash
        self.analysis_id = analysis_id
        self.request_id = request_id
        self.timestamp = datetime.utcnow()
        
        # Create detailed error message
        error_msg = f"[{error_type.value}] {message}"
        if content_hash:
            error_msg += f" (Content: {content_hash})"
        if analysis_id:
            error_msg += f" (Analysis: {analysis_id})"
        if request_id:
            error_msg += f" (Request: {request_id})"
        
        super().__init__(error_msg)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary"""
        return {
            "error_type": self.error_type.value,
            "message": self.message,
            "details": self.details,
            "content_hash": self.content_hash,
            "analysis_id": self.analysis_id,
            "request_id": self.request_id,
            "timestamp": self.timestamp.isoformat()
        }


class SEOOptimizationException(Exception):
    """Base exception for SEO optimization"""
    
    def __init__(
        self,
        error_type: SEOOptimizationErrorType,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        content_hash: Optional[str] = None,
        optimization_id: Optional[str] = None,
        request_id: Optional[str] = None
    ):
        self.error_type = error_type
        self.message = message
        self.details = details or {}
        self.content_hash = content_hash
        self.optimization_id = optimization_id
        self.request_id = request_id
        self.timestamp = datetime.utcnow()
        
        # Create detailed error message
        error_msg = f"[{error_type.value}] {message}"
        if content_hash:
            error_msg += f" (Content: {content_hash})"
        if optimization_id:
            error_msg += f" (Optimization: {optimization_id})"
        if request_id:
            error_msg += f" (Request: {request_id})"
        
        super().__init__(error_msg)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary"""
        return {
            "error_type": self.error_type.value,
            "message": self.message,
            "details": self.details,
            "content_hash": self.content_hash,
            "optimization_id": self.optimization_id,
            "request_id": self.request_id,
            "timestamp": self.timestamp.isoformat()
        }


class ContentGenerationException(Exception):
    """Base exception for content generation"""
    
    def __init__(
        self,
        error_type: ContentGenerationErrorType,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        topic: Optional[str] = None,
        generation_id: Optional[str] = None,
        request_id: Optional[str] = None
    ):
        self.error_type = error_type
        self.message = message
        self.details = details or {}
        self.topic = topic
        self.generation_id = generation_id
        self.request_id = request_id
        self.timestamp = datetime.utcnow()
        
        # Create detailed error message
        error_msg = f"[{error_type.value}] {message}"
        if topic:
            error_msg += f" (Topic: {topic})"
        if generation_id:
            error_msg += f" (Generation: {generation_id})"
        if request_id:
            error_msg += f" (Request: {request_id})"
        
        super().__init__(error_msg)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary"""
        return {
            "error_type": self.error_type.value,
            "message": self.message,
            "details": self.details,
            "topic": self.topic,
            "generation_id": self.generation_id,
            "request_id": self.request_id,
            "timestamp": self.timestamp.isoformat()
        }


# Specific blog post exceptions
class PostNotFoundError(BlogPostException):
    """Post not found error"""
    
    def __init__(self, post_id: str, message: str = "Blog post not found"):
        super().__init__(
            error_type=BlogPostErrorType.POST_NOT_FOUND,
            message=message,
            details={"post_id": post_id},
            post_id=post_id
        )


class PostAlreadyExistsError(BlogPostException):
    """Post already exists error"""
    
    def __init__(self, slug: str, message: str = "Blog post with this slug already exists"):
        super().__init__(
            error_type=BlogPostErrorType.POST_ALREADY_EXISTS,
            message=message,
            details={"slug": slug}
        )


class PostInvalidStatusError(BlogPostException):
    """Post invalid status error"""
    
    def __init__(self, post_id: str, current_status: str, required_status: str, message: str = "Invalid post status"):
        super().__init__(
            error_type=BlogPostErrorType.POST_INVALID_STATUS,
            message=message,
            details={
                "post_id": post_id,
                "current_status": current_status,
                "required_status": required_status
            },
            post_id=post_id
        )


class PostPermissionDeniedError(BlogPostException):
    """Post permission denied error"""
    
    def __init__(self, post_id: str, user_id: str, required_permission: str, message: str = "Permission denied"):
        super().__init__(
            error_type=BlogPostErrorType.POST_PERMISSION_DENIED,
            message=message,
            details={
                "post_id": post_id,
                "user_id": user_id,
                "required_permission": required_permission
            },
            post_id=post_id,
            user_id=user_id
        )


class PostValidationError(BlogPostException):
    """Post validation error"""
    
    def __init__(self, post_id: str, validation_field: str, validation_error: str, message: str = "Post validation error"):
        super().__init__(
            error_type=BlogPostErrorType.POST_VALIDATION_ERROR,
            message=message,
            details={
                "post_id": post_id,
                "validation_field": validation_field,
                "validation_error": validation_error
            },
            post_id=post_id
        )


class PostContentError(BlogPostException):
    """Post content error"""
    
    def __init__(self, post_id: str, content_error: str, message: str = "Post content error"):
        super().__init__(
            error_type=BlogPostErrorType.POST_CONTENT_ERROR,
            message=message,
            details={
                "post_id": post_id,
                "content_error": content_error
            },
            post_id=post_id
        )


class PostSEOError(BlogPostException):
    """Post SEO error"""
    
    def __init__(self, post_id: str, seo_error: str, message: str = "Post SEO error"):
        super().__init__(
            error_type=BlogPostErrorType.POST_SEO_ERROR,
            message=message,
            details={
                "post_id": post_id,
                "seo_error": seo_error
            },
            post_id=post_id
        )


class PostAnalyticsError(BlogPostException):
    """Post analytics error"""
    
    def __init__(self, post_id: str, analytics_error: str, message: str = "Post analytics error"):
        super().__init__(
            error_type=BlogPostErrorType.POST_ANALYTICS_ERROR,
            message=message,
            details={
                "post_id": post_id,
                "analytics_error": analytics_error
            },
            post_id=post_id
        )


class PostCollaborationError(BlogPostException):
    """Post collaboration error"""
    
    def __init__(self, post_id: str, collaboration_error: str, message: str = "Post collaboration error"):
        super().__init__(
            error_type=BlogPostErrorType.POST_COLLABORATION_ERROR,
            message=message,
            details={
                "post_id": post_id,
                "collaboration_error": collaboration_error
            },
            post_id=post_id
        )


class PostWorkflowError(BlogPostException):
    """Post workflow error"""
    
    def __init__(self, post_id: str, workflow_error: str, message: str = "Post workflow error"):
        super().__init__(
            error_type=BlogPostErrorType.POST_WORKFLOW_ERROR,
            message=message,
            details={
                "post_id": post_id,
                "workflow_error": workflow_error
            },
            post_id=post_id
        )


class PostTemplateError(BlogPostException):
    """Post template error"""
    
    def __init__(self, template_id: str, template_error: str, message: str = "Post template error"):
        super().__init__(
            error_type=BlogPostErrorType.POST_TEMPLATE_ERROR,
            message=message,
            details={
                "template_id": template_id,
                "template_error": template_error
            }
        )


class PostCategoryError(BlogPostException):
    """Post category error"""
    
    def __init__(self, category_id: str, category_error: str, message: str = "Post category error"):
        super().__init__(
            error_type=BlogPostErrorType.POST_CATEGORY_ERROR,
            message=message,
            details={
                "category_id": category_id,
                "category_error": category_error
            }
        )


class PostTagError(BlogPostException):
    """Post tag error"""
    
    def __init__(self, tag_id: str, tag_error: str, message: str = "Post tag error"):
        super().__init__(
            error_type=BlogPostErrorType.POST_TAG_ERROR,
            message=message,
            details={
                "tag_id": tag_id,
                "tag_error": tag_error
            }
        )


class PostAuthorError(BlogPostException):
    """Post author error"""
    
    def __init__(self, author_id: str, author_error: str, message: str = "Post author error"):
        super().__init__(
            error_type=BlogPostErrorType.POST_AUTHOR_ERROR,
            message=message,
            details={
                "author_id": author_id,
                "author_error": author_error
            }
        )


class PostCommentError(BlogPostException):
    """Post comment error"""
    
    def __init__(self, comment_id: str, comment_error: str, message: str = "Post comment error"):
        super().__init__(
            error_type=BlogPostErrorType.POST_COMMENT_ERROR,
            message=message,
            details={
                "comment_id": comment_id,
                "comment_error": comment_error
            }
        )


class PostMediaError(BlogPostException):
    """Post media error"""
    
    def __init__(self, post_id: str, media_error: str, message: str = "Post media error"):
        super().__init__(
            error_type=BlogPostErrorType.POST_MEDIA_ERROR,
            message=message,
            details={
                "post_id": post_id,
                "media_error": media_error
            },
            post_id=post_id
        )


class PostPublishingError(BlogPostException):
    """Post publishing error"""
    
    def __init__(self, post_id: str, publishing_error: str, message: str = "Post publishing error"):
        super().__init__(
            error_type=BlogPostErrorType.POST_PUBLISHING_ERROR,
            message=message,
            details={
                "post_id": post_id,
                "publishing_error": publishing_error
            },
            post_id=post_id
        )


class PostSchedulingError(BlogPostException):
    """Post scheduling error"""
    
    def __init__(self, post_id: str, scheduling_error: str, message: str = "Post scheduling error"):
        super().__init__(
            error_type=BlogPostErrorType.POST_SCHEDULING_ERROR,
            message=message,
            details={
                "post_id": post_id,
                "scheduling_error": scheduling_error
            },
            post_id=post_id
        )


class PostArchivingError(BlogPostException):
    """Post archiving error"""
    
    def __init__(self, post_id: str, archiving_error: str, message: str = "Post archiving error"):
        super().__init__(
            error_type=BlogPostErrorType.POST_ARCHIVING_ERROR,
            message=message,
            details={
                "post_id": post_id,
                "archiving_error": archiving_error
            },
            post_id=post_id
        )


class PostDeletionError(BlogPostException):
    """Post deletion error"""
    
    def __init__(self, post_id: str, deletion_error: str, message: str = "Post deletion error"):
        super().__init__(
            error_type=BlogPostErrorType.POST_DELETION_ERROR,
            message=message,
            details={
                "post_id": post_id,
                "deletion_error": deletion_error
            },
            post_id=post_id
        )


class PostSystemError(BlogPostException):
    """Post system error"""
    
    def __init__(self, message: str = "Blog post system error", details: Optional[Dict[str, Any]] = None):
        super().__init__(
            error_type=BlogPostErrorType.POST_SYSTEM_ERROR,
            message=message,
            details=details or {}
        )


# ML Pipeline exceptions
class ModelNotFoundError(MLPipelineException):
    """Model not found error"""
    
    def __init__(self, model_name: str, message: str = "ML model not found"):
        super().__init__(
            error_type=MLPipelineErrorType.MODEL_NOT_FOUND,
            message=message,
            details={"model_name": model_name},
            model_name=model_name
        )


class ModelLoadingError(MLPipelineException):
    """Model loading error"""
    
    def __init__(self, model_name: str, loading_error: str, message: str = "Model loading failed"):
        super().__init__(
            error_type=MLPipelineErrorType.MODEL_LOADING_ERROR,
            message=message,
            details={
                "model_name": model_name,
                "loading_error": loading_error
            },
            model_name=model_name
        )


class ModelInferenceError(MLPipelineException):
    """Model inference error"""
    
    def __init__(self, model_name: str, inference_error: str, message: str = "Model inference failed"):
        super().__init__(
            error_type=MLPipelineErrorType.MODEL_INFERENCE_ERROR,
            message=message,
            details={
                "model_name": model_name,
                "inference_error": inference_error
            },
            model_name=model_name
        )


class ModelTrainingError(MLPipelineException):
    """Model training error"""
    
    def __init__(self, model_name: str, training_error: str, message: str = "Model training failed"):
        super().__init__(
            error_type=MLPipelineErrorType.MODEL_TRAINING_ERROR,
            message=message,
            details={
                "model_name": model_name,
                "training_error": training_error
            },
            model_name=model_name
        )


class ModelValidationError(MLPipelineException):
    """Model validation error"""
    
    def __init__(self, model_name: str, validation_error: str, message: str = "Model validation failed"):
        super().__init__(
            error_type=MLPipelineErrorType.MODEL_VALIDATION_ERROR,
            message=message,
            details={
                "model_name": model_name,
                "validation_error": validation_error
            },
            model_name=model_name
        )


class PipelineExecutionError(MLPipelineException):
    """Pipeline execution error"""
    
    def __init__(self, pipeline_id: str, execution_error: str, message: str = "Pipeline execution failed"):
        super().__init__(
            error_type=MLPipelineErrorType.PIPELINE_EXECUTION_ERROR,
            message=message,
            details={
                "pipeline_id": pipeline_id,
                "execution_error": execution_error
            },
            pipeline_id=pipeline_id
        )


class PipelineTimeoutError(MLPipelineException):
    """Pipeline timeout error"""
    
    def __init__(self, pipeline_id: str, timeout_seconds: int, message: str = "Pipeline execution timed out"):
        super().__init__(
            error_type=MLPipelineErrorType.PIPELINE_TIMEOUT,
            message=message,
            details={
                "pipeline_id": pipeline_id,
                "timeout_seconds": timeout_seconds
            },
            pipeline_id=pipeline_id
        )


class PipelineQueueFullError(MLPipelineException):
    """Pipeline queue full error"""
    
    def __init__(self, queue_size: int, max_size: int, message: str = "Pipeline queue is full"):
        super().__init__(
            error_type=MLPipelineErrorType.PIPELINE_QUEUE_FULL,
            message=message,
            details={
                "queue_size": queue_size,
                "max_size": max_size
            }
        )


class PipelineResourceError(MLPipelineException):
    """Pipeline resource error"""
    
    def __init__(self, resource_type: str, resource_error: str, message: str = "Pipeline resource error"):
        super().__init__(
            error_type=MLPipelineErrorType.PIPELINE_RESOURCE_ERROR,
            message=message,
            details={
                "resource_type": resource_type,
                "resource_error": resource_error
            }
        )


class PipelineConfigurationError(MLPipelineException):
    """Pipeline configuration error"""
    
    def __init__(self, config_field: str, config_error: str, message: str = "Pipeline configuration error"):
        super().__init__(
            error_type=MLPipelineErrorType.PIPELINE_CONFIGURATION_ERROR,
            message=message,
            details={
                "config_field": config_field,
                "config_error": config_error
            }
        )


# Content Analysis exceptions
class AnalysisFailedError(ContentAnalysisException):
    """Analysis failed error"""
    
    def __init__(self, content_hash: str, analysis_error: str, message: str = "Content analysis failed"):
        super().__init__(
            error_type=ContentAnalysisErrorType.ANALYSIS_FAILED,
            message=message,
            details={
                "content_hash": content_hash,
                "analysis_error": analysis_error
            },
            content_hash=content_hash
        )


class ContentTooShortError(ContentAnalysisException):
    """Content too short error"""
    
    def __init__(self, content_length: int, min_length: int, message: str = "Content is too short for analysis"):
        super().__init__(
            error_type=ContentAnalysisErrorType.CONTENT_TOO_SHORT,
            message=message,
            details={
                "content_length": content_length,
                "min_length": min_length
            }
        )


class ContentTooLongError(ContentAnalysisException):
    """Content too long error"""
    
    def __init__(self, content_length: int, max_length: int, message: str = "Content is too long for analysis"):
        super().__init__(
            error_type=ContentAnalysisErrorType.CONTENT_TOO_LONG,
            message=message,
            details={
                "content_length": content_length,
                "max_length": max_length
            }
        )


class ContentInvalidFormatError(ContentAnalysisException):
    """Content invalid format error"""
    
    def __init__(self, content_format: str, supported_formats: list, message: str = "Content format is not supported"):
        super().__init__(
            error_type=ContentAnalysisErrorType.CONTENT_INVALID_FORMAT,
            message=message,
            details={
                "content_format": content_format,
                "supported_formats": supported_formats
            }
        )


class AnalysisTimeoutError(ContentAnalysisException):
    """Analysis timeout error"""
    
    def __init__(self, analysis_id: str, timeout_seconds: int, message: str = "Content analysis timed out"):
        super().__init__(
            error_type=ContentAnalysisErrorType.ANALYSIS_TIMEOUT,
            message=message,
            details={
                "analysis_id": analysis_id,
                "timeout_seconds": timeout_seconds
            },
            analysis_id=analysis_id
        )


class AnalysisResourceError(ContentAnalysisException):
    """Analysis resource error"""
    
    def __init__(self, resource_type: str, resource_error: str, message: str = "Analysis resource error"):
        super().__init__(
            error_type=ContentAnalysisErrorType.ANALYSIS_RESOURCE_ERROR,
            message=message,
            details={
                "resource_type": resource_type,
                "resource_error": resource_error
            }
        )


class AnalysisModelError(ContentAnalysisException):
    """Analysis model error"""
    
    def __init__(self, model_name: str, model_error: str, message: str = "Analysis model error"):
        super().__init__(
            error_type=ContentAnalysisErrorType.ANALYSIS_MODEL_ERROR,
            message=message,
            details={
                "model_name": model_name,
                "model_error": model_error
            }
        )


class AnalysisValidationError(ContentAnalysisException):
    """Analysis validation error"""
    
    def __init__(self, validation_field: str, validation_error: str, message: str = "Analysis validation error"):
        super().__init__(
            error_type=ContentAnalysisErrorType.ANALYSIS_VALIDATION_ERROR,
            message=message,
            details={
                "validation_field": validation_field,
                "validation_error": validation_error
            }
        )


# SEO Optimization exceptions
class SEOAnalysisFailedError(SEOOptimizationException):
    """SEO analysis failed error"""
    
    def __init__(self, content_hash: str, seo_error: str, message: str = "SEO analysis failed"):
        super().__init__(
            error_type=SEOOptimizationErrorType.SEO_ANALYSIS_FAILED,
            message=message,
            details={
                "content_hash": content_hash,
                "seo_error": seo_error
            },
            content_hash=content_hash
        )


class KeywordAnalysisFailedError(SEOOptimizationException):
    """Keyword analysis failed error"""
    
    def __init__(self, keywords: list, analysis_error: str, message: str = "Keyword analysis failed"):
        super().__init__(
            error_type=SEOOptimizationErrorType.KEYWORD_ANALYSIS_FAILED,
            message=message,
            details={
                "keywords": keywords,
                "analysis_error": analysis_error
            }
        )


class MetaTagError(SEOOptimizationException):
    """Meta tag error"""
    
    def __init__(self, meta_tag_type: str, meta_tag_error: str, message: str = "Meta tag error"):
        super().__init__(
            error_type=SEOOptimizationErrorType.META_TAG_ERROR,
            message=message,
            details={
                "meta_tag_type": meta_tag_type,
                "meta_tag_error": meta_tag_error
            }
        )


class ContentStructureError(SEOOptimizationException):
    """Content structure error"""
    
    def __init__(self, structure_element: str, structure_error: str, message: str = "Content structure error"):
        super().__init__(
            error_type=SEOOptimizationErrorType.CONTENT_STRUCTURE_ERROR,
            message=message,
            details={
                "structure_element": structure_element,
                "structure_error": structure_error
            }
        )


class SEOScoreCalculationError(SEOOptimizationException):
    """SEO score calculation error"""
    
    def __init__(self, calculation_error: str, message: str = "SEO score calculation failed"):
        super().__init__(
            error_type=SEOOptimizationErrorType.SEO_SCORE_CALCULATION_ERROR,
            message=message,
            details={
                "calculation_error": calculation_error
            }
        )


class SEORecommendationError(SEOOptimizationException):
    """SEO recommendation error"""
    
    def __init__(self, recommendation_error: str, message: str = "SEO recommendation generation failed"):
        super().__init__(
            error_type=SEOOptimizationErrorType.SEO_RECOMMENDATION_ERROR,
            message=message,
            details={
                "recommendation_error": recommendation_error
            }
        )


class SEOOptimizationFailedError(SEOOptimizationException):
    """SEO optimization failed error"""
    
    def __init__(self, optimization_error: str, message: str = "SEO optimization failed"):
        super().__init__(
            error_type=SEOOptimizationErrorType.SEO_OPTIMIZATION_FAILED,
            message=message,
            details={
                "optimization_error": optimization_error
            }
        )


# Content Generation exceptions
class GenerationFailedError(ContentGenerationException):
    """Generation failed error"""
    
    def __init__(self, topic: str, generation_error: str, message: str = "Content generation failed"):
        super().__init__(
            error_type=ContentGenerationErrorType.GENERATION_FAILED,
            message=message,
            details={
                "topic": topic,
                "generation_error": generation_error
            },
            topic=topic
        )


class AIServiceError(ContentGenerationException):
    """AI service error"""
    
    def __init__(self, service_name: str, service_error: str, message: str = "AI service error"):
        super().__init__(
            error_type=ContentGenerationErrorType.AI_SERVICE_ERROR,
            message=message,
            details={
                "service_name": service_name,
                "service_error": service_error
            }
        )


class PromptValidationError(ContentGenerationException):
    """Prompt validation error"""
    
    def __init__(self, prompt_field: str, validation_error: str, message: str = "Prompt validation error"):
        super().__init__(
            error_type=ContentGenerationErrorType.PROMPT_VALIDATION_ERROR,
            message=message,
            details={
                "prompt_field": prompt_field,
                "validation_error": validation_error
            }
        )


class ContentValidationError(ContentGenerationException):
    """Content validation error"""
    
    def __init__(self, content_field: str, validation_error: str, message: str = "Generated content validation error"):
        super().__init__(
            error_type=ContentGenerationErrorType.CONTENT_VALIDATION_ERROR,
            message=message,
            details={
                "content_field": content_field,
                "validation_error": validation_error
            }
        )


class GenerationTimeoutError(ContentGenerationException):
    """Generation timeout error"""
    
    def __init__(self, topic: str, timeout_seconds: int, message: str = "Content generation timed out"):
        super().__init__(
            error_type=ContentGenerationErrorType.GENERATION_TIMEOUT,
            message=message,
            details={
                "topic": topic,
                "timeout_seconds": timeout_seconds
            },
            topic=topic
        )


class GenerationQuotaExceededError(ContentGenerationException):
    """Generation quota exceeded error"""
    
    def __init__(self, quota_type: str, current_usage: int, quota_limit: int, message: str = "Generation quota exceeded"):
        super().__init__(
            error_type=ContentGenerationErrorType.GENERATION_QUOTA_EXCEEDED,
            message=message,
            details={
                "quota_type": quota_type,
                "current_usage": current_usage,
                "quota_limit": quota_limit
            }
        )


class GenerationQualityError(ContentGenerationException):
    """Generation quality error"""
    
    def __init__(self, quality_metric: str, quality_score: float, min_required: float, message: str = "Generated content quality below threshold"):
        super().__init__(
            error_type=ContentGenerationErrorType.GENERATION_QUALITY_ERROR,
            message=message,
            details={
                "quality_metric": quality_metric,
                "quality_score": quality_score,
                "min_required": min_required
            }
        )


# Utility functions for error handling
def create_blog_error(
    error_type: BlogPostErrorType,
    message: str,
    details: Optional[Dict[str, Any]] = None,
    post_id: Optional[str] = None,
    user_id: Optional[str] = None,
    request_id: Optional[str] = None
) -> BlogPostException:
    """Create a blog post exception"""
    return BlogPostException(
        error_type=error_type,
        message=message,
        details=details,
        post_id=post_id,
        user_id=user_id,
        request_id=request_id
    )


def create_ml_error(
    error_type: MLPipelineErrorType,
    message: str,
    details: Optional[Dict[str, Any]] = None,
    pipeline_id: Optional[str] = None,
    model_name: Optional[str] = None,
    request_id: Optional[str] = None
) -> MLPipelineException:
    """Create an ML pipeline exception"""
    return MLPipelineException(
        error_type=error_type,
        message=message,
        details=details,
        pipeline_id=pipeline_id,
        model_name=model_name,
        request_id=request_id
    )


def create_analysis_error(
    error_type: ContentAnalysisErrorType,
    message: str,
    details: Optional[Dict[str, Any]] = None,
    content_hash: Optional[str] = None,
    analysis_id: Optional[str] = None,
    request_id: Optional[str] = None
) -> ContentAnalysisException:
    """Create a content analysis exception"""
    return ContentAnalysisException(
        error_type=error_type,
        message=message,
        details=details,
        content_hash=content_hash,
        analysis_id=analysis_id,
        request_id=request_id
    )


def create_seo_error(
    error_type: SEOOptimizationErrorType,
    message: str,
    details: Optional[Dict[str, Any]] = None,
    content_hash: Optional[str] = None,
    optimization_id: Optional[str] = None,
    request_id: Optional[str] = None
) -> SEOOptimizationException:
    """Create an SEO optimization exception"""
    return SEOOptimizationException(
        error_type=error_type,
        message=message,
        details=details,
        content_hash=content_hash,
        optimization_id=optimization_id,
        request_id=request_id
    )


def create_generation_error(
    error_type: ContentGenerationErrorType,
    message: str,
    details: Optional[Dict[str, Any]] = None,
    topic: Optional[str] = None,
    generation_id: Optional[str] = None,
    request_id: Optional[str] = None
) -> ContentGenerationException:
    """Create a content generation exception"""
    return ContentGenerationException(
        error_type=error_type,
        message=message,
        details=details,
        topic=topic,
        generation_id=generation_id,
        request_id=request_id
    )


def log_blog_error(
    error: BlogPostException,
    logger: Optional[logging.Logger] = None
) -> None:
    """Log blog post error with structured logging"""
    if logger is None:
        logger = logging.getLogger(__name__)
    
    error_data = error.to_dict()
    
    # Log based on error type severity
    if error.error_type in [
        BlogPostErrorType.POST_SYSTEM_ERROR,
        BlogPostErrorType.POST_CONTENT_ERROR,
        BlogPostErrorType.POST_PUBLISHING_ERROR
    ]:
        logger.error(f"Blog Post Error: {error_data}")
    elif error.error_type in [
        BlogPostErrorType.POST_VALIDATION_ERROR,
        BlogPostErrorType.POST_PERMISSION_DENIED,
        BlogPostErrorType.POST_SEO_ERROR
    ]:
        logger.warning(f"Blog Post Warning: {error_data}")
    else:
        logger.info(f"Blog Post Info: {error_data}")


def log_ml_error(
    error: MLPipelineException,
    logger: Optional[logging.Logger] = None
) -> None:
    """Log ML pipeline error with structured logging"""
    if logger is None:
        logger = logging.getLogger(__name__)
    
    error_data = error.to_dict()
    
    # Log based on error type severity
    if error.error_type in [
        MLPipelineErrorType.MODEL_LOADING_ERROR,
        MLPipelineErrorType.MODEL_INFERENCE_ERROR,
        MLPipelineErrorType.PIPELINE_EXECUTION_ERROR
    ]:
        logger.error(f"ML Pipeline Error: {error_data}")
    elif error.error_type in [
        MLPipelineErrorType.MODEL_VALIDATION_ERROR,
        MLPipelineErrorType.PIPELINE_TIMEOUT,
        MLPipelineErrorType.PIPELINE_QUEUE_FULL
    ]:
        logger.warning(f"ML Pipeline Warning: {error_data}")
    else:
        logger.info(f"ML Pipeline Info: {error_data}")


def handle_blog_error(
    error: Exception,
    post_id: Optional[str] = None,
    user_id: Optional[str] = None,
    request_id: Optional[str] = None
) -> BlogPostException:
    """Handle and convert generic exceptions to blog post exceptions"""
    if isinstance(error, BlogPostException):
        return error
    
    # Convert common exceptions to blog post exceptions
    if isinstance(error, ValueError):
        return PostValidationError(
            post_id=post_id or "unknown",
            validation_field="input",
            validation_error=str(error),
            message=str(error)
        )
    elif isinstance(error, PermissionError):
        return PostPermissionDeniedError(
            post_id=post_id or "unknown",
            user_id=user_id or "unknown",
            required_permission="unknown",
            message=str(error)
        )
    elif isinstance(error, TimeoutError):
        return PostSystemError(
            message=f"Operation timed out: {str(error)}",
            details={
                "original_error": str(error),
                "error_type": type(error).__name__,
                "post_id": post_id,
                "user_id": user_id,
                "request_id": request_id
            }
        )
    else:
        return PostSystemError(
            message=f"Unexpected error: {str(error)}",
            details={
                "original_error": str(error),
                "error_type": type(error).__name__,
                "post_id": post_id,
                "user_id": user_id,
                "request_id": request_id
            }
        )


def handle_ml_error(
    error: Exception,
    pipeline_id: Optional[str] = None,
    model_name: Optional[str] = None,
    request_id: Optional[str] = None
) -> MLPipelineException:
    """Handle and convert generic exceptions to ML pipeline exceptions"""
    if isinstance(error, MLPipelineException):
        return error
    
    # Convert common exceptions to ML pipeline exceptions
    if isinstance(error, TimeoutError):
        return PipelineTimeoutError(
            pipeline_id=pipeline_id or "unknown",
            timeout_seconds=30,
            message=str(error)
        )
    elif isinstance(error, ConnectionError):
        return PipelineResourceError(
            resource_type="network",
            resource_error=str(error),
            message=str(error)
        )
    elif isinstance(error, ValueError):
        return ModelValidationError(
            model_name=model_name or "unknown",
            validation_error=str(error),
            message=str(error)
        )
    elif isinstance(error, PermissionError):
        return PipelineResourceError(
            resource_type="permission",
            resource_error=str(error),
            message=str(error)
        )
    else:
        return PipelineExecutionError(
            pipeline_id=pipeline_id or "unknown",
            execution_error=str(error),
            message=f"Unexpected error: {str(error)}"
        )


# Error response templates
ERROR_RESPONSES = {
    BlogPostErrorType.POST_NOT_FOUND: {
        "status_code": 404,
        "message": "Blog post not found",
        "suggestion": "Check post ID and ensure post exists"
    },
    BlogPostErrorType.POST_ALREADY_EXISTS: {
        "status_code": 409,
        "message": "Blog post already exists",
        "suggestion": "Use a different slug or update existing post"
    },
    BlogPostErrorType.POST_PERMISSION_DENIED: {
        "status_code": 403,
        "message": "Permission denied",
        "suggestion": "Check user permissions for this post"
    },
    BlogPostErrorType.POST_VALIDATION_ERROR: {
        "status_code": 400,
        "message": "Post validation error",
        "suggestion": "Check post data format and requirements"
    },
    BlogPostErrorType.POST_SYSTEM_ERROR: {
        "status_code": 500,
        "message": "Blog post system error",
        "suggestion": "Contact support if issue persists"
    },
    MLPipelineErrorType.MODEL_NOT_FOUND: {
        "status_code": 404,
        "message": "ML model not found",
        "suggestion": "Check model name and ensure model exists"
    },
    MLPipelineErrorType.MODEL_LOADING_ERROR: {
        "status_code": 500,
        "message": "Model loading failed",
        "suggestion": "Check model configuration and try again"
    },
    MLPipelineErrorType.PIPELINE_EXECUTION_ERROR: {
        "status_code": 500,
        "message": "Pipeline execution failed",
        "suggestion": "Check pipeline configuration and try again"
    },
    MLPipelineErrorType.PIPELINE_TIMEOUT: {
        "status_code": 408,
        "message": "Pipeline execution timed out",
        "suggestion": "Retry with longer timeout or simpler request"
    }
}


def get_error_response(error: Union[BlogPostException, MLPipelineException, ContentAnalysisException, SEOOptimizationException, ContentGenerationException]) -> Dict[str, Any]:
    """Get standardized error response"""
    error_info = ERROR_RESPONSES.get(error.error_type, {
        "status_code": 500,
        "message": "Unknown error",
        "suggestion": "Contact support"
    })
    
    return {
        "error": error.error_type.value,
        "message": error.message,
        "status_code": error_info["status_code"],
        "suggestion": error_info["suggestion"],
        "details": error.details,
        "timestamp": error.timestamp.isoformat(),
        "post_id": getattr(error, 'post_id', None),
        "pipeline_id": getattr(error, 'pipeline_id', None),
        "content_hash": getattr(error, 'content_hash', None),
        "optimization_id": getattr(error, 'optimization_id', None),
        "generation_id": getattr(error, 'generation_id', None),
        "request_id": getattr(error, 'request_id', None)
    }





























