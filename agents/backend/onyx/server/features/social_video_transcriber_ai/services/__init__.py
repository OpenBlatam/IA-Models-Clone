"""Services module for Social Video Transcriber AI"""

from .video_downloader import VideoDownloader, get_video_downloader
from .transcription_service import TranscriptionService, get_transcription_service
from .ai_analyzer import AIAnalyzer, get_ai_analyzer
from .variant_generator import VariantGenerator, get_variant_generator
from .openrouter_client import OpenRouterClient, get_openrouter_client
from .cache_service import CacheService, get_cache_service
from .advanced_analyzer import AdvancedAnalyzer, get_advanced_analyzer
from .batch_processor import BatchProcessor, get_batch_processor, BatchJob, BatchStatus
from .webhook_service import WebhookService, get_webhook_service, WebhookEvent
from .auth_service import AuthService, get_auth_service, UserTier, TIER_LIMITS
from .retry_handler import (
    RetryHandler,
    CircuitBreaker,
    ResilientExecutor,
    get_circuit_breaker,
    with_retry,
)
from .export_service import ExportService, get_export_service, ExportFormat, ExportOptions
from .translation_service import TranslationService, get_translation_service, SupportedLanguage
from .search_service import SearchService, get_search_service
from .queue_service import QueueService, get_queue_service, Priority
from .analytics_service import AnalyticsService, get_analytics_service
from .highlights_service import HighlightsService, get_highlights_service, HighlightType
from .rust_accelerator import (
    RustAccelerator,
    AcceleratorStats,
    get_rust_accelerator,
    is_rust_available,
    RUST_AVAILABLE,
)

__all__ = [
    # Video Downloader
    "VideoDownloader",
    "get_video_downloader",
    # Transcription
    "TranscriptionService",
    "get_transcription_service",
    # AI Analysis
    "AIAnalyzer",
    "get_ai_analyzer",
    "AdvancedAnalyzer",
    "get_advanced_analyzer",
    # Variants
    "VariantGenerator",
    "get_variant_generator",
    # OpenRouter
    "OpenRouterClient",
    "get_openrouter_client",
    # Cache
    "CacheService",
    "get_cache_service",
    # Batch Processing
    "BatchProcessor",
    "get_batch_processor",
    "BatchJob",
    "BatchStatus",
    # Webhooks
    "WebhookService",
    "get_webhook_service",
    "WebhookEvent",
    # Authentication
    "AuthService",
    "get_auth_service",
    "UserTier",
    "TIER_LIMITS",
    # Retry & Circuit Breaker
    "RetryHandler",
    "CircuitBreaker",
    "ResilientExecutor",
    "get_circuit_breaker",
    "with_retry",
    # Export
    "ExportService",
    "get_export_service",
    "ExportFormat",
    "ExportOptions",
    # Translation
    "TranslationService",
    "get_translation_service",
    "SupportedLanguage",
    # Search
    "SearchService",
    "get_search_service",
    # Queue
    "QueueService",
    "get_queue_service",
    "Priority",
    # Analytics
    "AnalyticsService",
    "get_analytics_service",
    # Highlights
    "HighlightsService",
    "get_highlights_service",
    "HighlightType",
    # Rust Accelerator ✨ HIGH PERFORMANCE
    "RustAccelerator",
    "AcceleratorStats",
    "get_rust_accelerator",
    "is_rust_available",
    "RUST_AVAILABLE",
]

