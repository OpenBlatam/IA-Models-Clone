"""
Core Professional Documents Module
===================================

Core functionality for document generation, including API, models, services, and templates.
"""

from .api import router
from .models import (
    DocumentGenerationRequest,
    DocumentGenerationResponse,
    DocumentExportRequest,
    DocumentExportResponse,
    DocumentTemplate,
    DocumentStyle,
    ProfessionalDocument,
    DocumentType,
    ExportFormat
)
from .services import (
    DocumentGenerationService,
    DocumentExportService,
    TemplateService
)
from .ai_service import AIDocumentGenerator
from .templates import template_manager
from .config import config, get_config
from .utils import handle_api_errors
from .storage import DocumentStorage, InMemoryDocumentStorage
from .exporters import (
    DocumentExporter,
    PDFExporter,
    MarkdownExporter,
    WordExporter,
    HTMLExporter
)
from .validators import (
    validate_document_exists,
    validate_export_request,
    validate_filename,
    validate_query_length
)
from .exceptions import (
    ProfessionalDocumentsError,
    DocumentNotFoundError,
    DocumentGenerationError,
    DocumentExportError,
    TemplateNotFoundError,
    ValidationError,
    AIServiceError,
    StorageError,
    InvalidFormatError
)
from .constants import (
    MIN_QUERY_LENGTH,
    WORDS_PER_PAGE,
    LENGTH_GUIDELINES,
    TONE_GUIDELINES,
    EXPORT_FORMAT_EXTENSIONS,
    DOCUMENT_TYPE_SUFFIXES,
    DEFAULT_EXPORT_DIR,
    DEFAULT_AI_MODEL,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE
)
from .helpers import (
    sanitize_title,
    calculate_word_count,
    process_sections_data,
    generate_document_title
)
from .export_helpers import (
    format_date,
    get_document_metadata,
    format_section_heading,
    has_metadata,
    build_markdown_metadata_lines,
    build_text_metadata_lines
)
from .dependencies import (
    get_document_generation_service,
    get_document_export_service,
    get_template_service
)
from .stats import calculate_document_stats
from .exporter_factory import ExporterFactory
from .performance import measure_time, PerformanceTimer
from .types import (
    SectionData,
    DocumentMetadata,
    SectionMetadata,
    DocumentTone,
    DocumentLength,
    LanguageCode,
    DocumentStatus,
    ExportStats,
    TemplateUsage
)
from .cache import TimedCache, template_cache, cached_template
from .builders import DocumentBuilder, create_document_builder
from .formatters import (
    format_file_size,
    format_duration,
    format_word_count,
    format_page_count,
    format_datetime,
    format_relative_time
)
from .retry import retry_with_backoff
from .sanitizers import (
    sanitize_html,
    sanitize_filename as sanitize_filename_util,
    sanitize_text,
    sanitize_url,
    sanitize_query
)
from .logging_utils import (
    log_function_call,
    log_document_operation,
    log_performance,
    log_error_with_context
)
from .query_builder import DocumentQueryBuilder, create_query_builder
from .middleware import (
    RequestLoggingMiddleware,
    RateLimitingMiddleware,
    add_request_id_middleware
)
from .batch_processor import (
    process_batch,
    batch_export_documents,
    batch_generate_documents
)
from .pagination import (
    PaginatedResult,
    paginate,
    calculate_offset,
    validate_pagination_params
)
from .rate_limit import TokenBucket, RateLimiter, default_rate_limiter

__all__ = [
    # API
    "router",
    
    # Models
    "DocumentGenerationRequest",
    "DocumentGenerationResponse",
    "DocumentExportRequest",
    "DocumentExportResponse",
    "DocumentTemplate",
    "DocumentStyle",
    "ProfessionalDocument",
    "DocumentType",
    "ExportFormat",
    
    # Services
    "DocumentGenerationService",
    "DocumentExportService",
    "TemplateService",
    "AIDocumentGenerator",
    "template_manager",
    
    # Configuration
    "config",
    "get_config",
    
    # Utilities
    "handle_api_errors",
    "DocumentStorage",
    "InMemoryDocumentStorage",
    
    # Exporters
    "DocumentExporter",
    "PDFExporter",
    "MarkdownExporter",
    "WordExporter",
    "HTMLExporter",
    "ExporterFactory",
    
    # Validators
    "validate_document_exists",
    "validate_export_request",
    "validate_filename",
    "validate_query_length",
    
    # Exceptions
    "ProfessionalDocumentsError",
    "DocumentNotFoundError",
    "DocumentGenerationError",
    "DocumentExportError",
    "TemplateNotFoundError",
    "ValidationError",
    "AIServiceError",
    "StorageError",
    "InvalidFormatError",
    
    # Constants
    "MIN_QUERY_LENGTH",
    "WORDS_PER_PAGE",
    "LENGTH_GUIDELINES",
    "TONE_GUIDELINES",
    "EXPORT_FORMAT_EXTENSIONS",
    "DOCUMENT_TYPE_SUFFIXES",
    "DEFAULT_EXPORT_DIR",
    "DEFAULT_AI_MODEL",
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_TEMPERATURE",
    
    # Helpers
    "sanitize_title",
    "calculate_word_count",
    "process_sections_data",
    "generate_document_title",
    "format_date",
    "get_document_metadata",
    "format_section_heading",
    "has_metadata",
    "build_markdown_metadata_lines",
    "build_text_metadata_lines",
    
    # Dependencies
    "get_document_generation_service",
    "get_document_export_service",
    "get_template_service",
    
    # Statistics
    "calculate_document_stats",
    
    # Performance
    "measure_time",
    "PerformanceTimer",
    
    # Types
    "SectionData",
    "DocumentMetadata",
    "SectionMetadata",
    "DocumentTone",
    "DocumentLength",
    "LanguageCode",
    "DocumentStatus",
    "ExportStats",
    "TemplateUsage",
    
    # Cache
    "TimedCache",
    "template_cache",
    "cached_template",
    
    # Builders
    "DocumentBuilder",
    "create_document_builder",
    
    # Formatters
    "format_file_size",
    "format_duration",
    "format_word_count",
    "format_page_count",
    "format_datetime",
    "format_relative_time",
    
    # Retry
    "retry_with_backoff",
    
    # Sanitizers
    "sanitize_html",
    "sanitize_filename_util",
    "sanitize_text",
    "sanitize_url",
    "sanitize_query",
    
    # Logging
    "log_function_call",
    "log_document_operation",
    "log_performance",
    "log_error_with_context",
    
    # Query Builder
    "DocumentQueryBuilder",
    "create_query_builder",
    
    # Middleware
    "RequestLoggingMiddleware",
    "RateLimitingMiddleware",
    "add_request_id_middleware",
    
    # Batch Processing
    "process_batch",
    "batch_export_documents",
    "batch_generate_documents",
    
    # Pagination
    "PaginatedResult",
    "paginate",
    "calculate_offset",
    "validate_pagination_params",
    
    # Rate Limiting
    "TokenBucket",
    "RateLimiter",
    "default_rate_limiter",
]
