"""
Utilities Module

Collection of utility functions and helpers.
"""

# Import existing utilities
from .image_utils import ImageUtils
from .detection_utils import DetectionUtils
from .performance_optimizer import PerformanceOptimizer
from .report_generator import ReportGenerator
from .visualization import QualityVisualizer

# Import new utilities
from .validation_utils import (
    validate_email,
    validate_url,
    validate_positive_number,
    validate_range,
    validate_not_empty,
    validate_required_fields,
)
from .performance_utils import (
    measure_time,
    retry_on_failure,
    throttle,
    PerformanceMonitor,
)
from .string_utils import (
    camel_to_snake,
    snake_to_camel,
    truncate,
    sanitize_filename,
    format_bytes,
    format_duration,
)
from .security_utils import (
    generate_token,
    hash_string,
    verify_hash,
    sanitize_input,
    encode_base64,
    decode_base64,
    mask_sensitive_data,
)
from .file_utils import (
    ensure_directory,
    get_file_size,
    get_file_extension,
    is_image_file,
    get_mime_type,
    list_files,
    safe_filename,
)
from .date_utils import (
    now_utc,
    now_local,
    to_utc,
    format_datetime,
    parse_datetime,
    time_ago,
    is_within_timeframe,
)
from .decorators import (
    singleton,
    deprecated,
    rate_limit,
    validate_args,
    cache_result,
)
from .async_utils import (
    run_in_executor,
    gather_with_limit,
    timeout_after,
    async_to_sync,
)
from .data_utils import (
    flatten_dict,
    unflatten_dict,
    deep_merge,
    filter_dict,
    exclude_dict,
    group_by,
    chunk_list,
    safe_json_loads,
    safe_json_dumps,
)
from .export_utils import (
    export_to_json,
    export_to_csv,
    export_to_dict,
)
from .format_utils import (
    format_number,
    format_percentage,
    format_currency,
    format_datetime_human,
    format_list,
)

# Test helpers (only import if testing)
try:
    from .test_helpers import (
        create_test_image,
        create_test_image_metadata,
        create_test_quality_score,
        create_test_defect,
        create_test_anomaly,
        create_test_inspection,
        assert_quality_score_valid,
        assert_inspection_valid,
    )
    TEST_HELPERS_AVAILABLE = True
except ImportError:
    TEST_HELPERS_AVAILABLE = False

__all__ = [
    # Existing
    "ImageUtils",
    "DetectionUtils",
    "PerformanceOptimizer",
    "ReportGenerator",
    "QualityVisualizer",
    # Validation
    "validate_email",
    "validate_url",
    "validate_positive_number",
    "validate_range",
    "validate_not_empty",
    "validate_required_fields",
    # Performance
    "measure_time",
    "retry_on_failure",
    "throttle",
    "PerformanceMonitor",
    # String
    "camel_to_snake",
    "snake_to_camel",
    "truncate",
    "sanitize_filename",
    "format_bytes",
    "format_duration",
    # Security
    "generate_token",
    "hash_string",
    "verify_hash",
    "sanitize_input",
    "encode_base64",
    "decode_base64",
    "mask_sensitive_data",
    # File
    "ensure_directory",
    "get_file_size",
    "get_file_extension",
    "is_image_file",
    "get_mime_type",
    "list_files",
    "safe_filename",
    # Date
    "now_utc",
    "now_local",
    "to_utc",
    "format_datetime",
    "parse_datetime",
    "time_ago",
    "is_within_timeframe",
    # Decorators
    "singleton",
    "deprecated",
    "rate_limit",
    "validate_args",
    "cache_result",
    # Async
    "run_in_executor",
    "gather_with_limit",
    "timeout_after",
    "async_to_sync",
    # Data
    "flatten_dict",
    "unflatten_dict",
    "deep_merge",
    "filter_dict",
    "exclude_dict",
    "group_by",
    "chunk_list",
    "safe_json_loads",
    "safe_json_dumps",
    # Export
    "export_to_json",
    "export_to_csv",
    "export_to_dict",
    # Format
    "format_number",
    "format_percentage",
    "format_currency",
    "format_datetime_human",
    "format_list",
]

# Add test helpers if available
if TEST_HELPERS_AVAILABLE:
    __all__.extend([
        "create_test_image",
        "create_test_image_metadata",
        "create_test_quality_score",
        "create_test_defect",
        "create_test_anomaly",
        "create_test_inspection",
        "assert_quality_score_valid",
        "assert_inspection_valid",
    ])
