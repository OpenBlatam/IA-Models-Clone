"""
Utility modules
Organized by functional categories for better modularity
"""

from utils.utility_factory import (
    UtilityFactory,
    get_utility_factory,
    get_utility_instance
)

from .errors import (
    APIError,
    ValidationError,
    NotFoundError,
    UnauthorizedError,
    ForbiddenError,
    ConflictError,
    create_http_exception,
    handle_service_error
)
from .validators import (
    validate_user_id,
    validate_email,
    validate_date_string,
    validate_date_not_future,
    validate_range,
    validate_required_fields,
    validate_enum_value,
    sanitize_string
)
from .response import (
    create_success_response,
    create_error_response,
    create_paginated_response
)
from .cache import (
    cache_result,
    clear_cache,
    get_cache_stats
)
from .async_helpers import (
    run_in_batches,
    run_parallel,
    async_retry,
    timeout_after
)
from .pydantic_helpers import (
    model_to_dict,
    models_to_dicts,
    validate_and_parse,
    partial_update_model
)
from .pagination import (
    calculate_pagination,
    paginate_items,
    validate_pagination_params
)
from .filters import (
    filter_by_field,
    filter_by_date_range,
    filter_by_custom_predicate,
    sort_items
)
from .security import (
    generate_secure_token,
    hash_string,
    validate_password_strength,
    sanitize_input,
    mask_sensitive_data
)
from .serialization import (
    serialize_for_json,
    to_json_string,
    prepare_response_data
)
from .query_params import (
    parse_date_query,
    parse_list_query,
    create_date_range_query,
    create_filter_query
)
from .date_helpers import (
    get_current_utc,
    parse_iso_date,
    format_iso_date,
    add_days,
    subtract_days,
    days_between,
    is_within_range,
    get_start_of_day,
    get_end_of_day,
    get_start_of_week,
    get_start_of_month
)
from .string_helpers import (
    truncate_string,
    normalize_string,
    capitalize_words,
    remove_special_chars,
    extract_words
)
from .math_helpers import (
    calculate_percentage,
    calculate_average,
    calculate_median,
    clamp_value,
    round_to_decimals
)
from .transformers import (
    transform_dict,
    flatten_dict,
    nest_dict,
    map_list,
    filter_list,
    group_by,
    pick_fields,
    omit_fields
)
from .response_builders import (
    build_success_response,
    build_error_response,
    build_paginated_response,
    build_list_response
)
from .type_converters import (
    to_int,
    to_float,
    to_bool,
    to_string,
    to_list,
    to_datetime
)
from .collection_helpers import (
    chunk_list,
    unique_list,
    merge_dicts,
    deep_merge_dicts,
    get_nested_value,
    set_nested_value
)
from .guards import (
    guard_not_none,
    guard_not_empty,
    guard_in_range,
    guard_in_list,
    guard_positive,
    guard_non_negative
)
from .api_docs import (
    create_endpoint_summary,
    create_response_examples,
    create_parameter_description
)
from .testing_helpers import (
    create_mock_assessment_data,
    create_mock_log_entry,
    create_mock_progress_data,
    assert_response_structure
)
from .performance_helpers import (
    measure_execution_time,
    batch_process,
    memoize_with_ttl
)
from .composers import (
    compose,
    pipe,
    curry,
    partial
)
from .functional_helpers import (
    map_function,
    filter_function,
    reduce_function,
    identity,
    constant,
    maybe
)
from .predicates import (
    is_none,
    is_not_none,
    is_empty,
    is_not_empty,
    is_positive,
    is_non_negative,
    is_negative,
    is_zero,
    is_in_range,
    is_greater_than,
    is_less_than,
    is_equal,
    is_not_equal,
    is_in,
    is_not_in,
    all_true,
    any_true
)
from .result_types import (
    Result,
    safe_call,
    safe_call_async
)
from .async_composers import (
    compose_async,
    pipe_async,
    parallel_map,
    retry_async
)
from .validation_combinators import (
    combine_validators,
    combine_validators_or,
    validate_and_transform,
    chain_validators
)
from .monads import (
    Maybe,
    Either,
    maybe_decorator,
    either_decorator
)
from .lenses import (
    Lens,
    lens,
    prop_lens,
    path_lens
)
from .functors import (
    Functor,
    ListFunctor,
    DictFunctor,
    list_functor,
    dict_functor
)
from .streams import (
    Stream,
    stream
)
from .trampolines import (
    Done,
    More,
    Trampoline,
    trampoline,
    make_trampoline
)
from .memoization import (
    memoize,
    memoize_with_key,
    memoize_with_hash,
    clear_memoization
)
from .observers import (
    Observer,
    create_observer
)
from .decorators import (
    retry,
    timeout,
    log_execution,
    validate_args,
    rate_limit
)
from .iterators import (
    take,
    drop,
    take_while,
    drop_while,
    chunk,
    pairwise,
    window,
    interleave
)
from .promises import (
    Promise,
    create_promise
)
from .futures import (
    all_futures,
    any_future,
    race_futures,
    timeout_future,
    create_future_from_value,
    create_future_from_error
)
from .schedulers import (
    Scheduler,
    create_scheduler
)
from .state_management import (
    State,
    create_state,
    state_reducer
)
from .event_emitters import (
    EventEmitter,
    create_event_emitter
)
from .middleware_utils import (
    create_middleware,
    request_logger,
    response_timer
)
from .batch_processors import (
    batch_map,
    batch_filter,
    batch_reduce
)
from .queue_utils import (
    AsyncQueue,
    create_queue,
    process_queue
)
from .semaphores import (
    RateLimiter,
    create_rate_limiter,
    with_semaphore,
    with_rate_limit
)
from .backpressure import (
    BackpressureController,
    create_backpressure_controller,
    with_backpressure
)
from .circuit_breakers import (
    CircuitBreaker,
    CircuitState,
    CircuitBreakerOpenError,
    create_circuit_breaker
)
from .retry_strategies import (
    RetryStrategy,
    retry_with_strategy,
    retry_with_jitter
)
from .pools import (
    ResourcePool,
    create_resource_pool,
    with_pool
)
from .workers import (
    WorkerPool,
    create_worker_pool
)
from .throttlers import (
    Throttler,
    Debouncer,
    create_throttler,
    create_debouncer,
    with_throttle
)
from .comparators import (
    compare_by,
    compare_multiple,
    natural_order,
    reverse_order
)
from .sorters import (
    sort_by,
    sort_by_multiple,
    stable_sort,
    partial_sort
)
from .aggregators import (
    group_by_key,
    aggregate,
    sum_by,
    count_by,
    average_by
)
from .formatters import (
    format_number,
    format_percentage,
    format_currency,
    format_duration,
    format_bytes
)
from .encoders import (
    encode_base64,
    decode_base64,
    encode_json_base64,
    decode_json_base64,
    url_encode,
    url_decode
)
from .parsers import (
    parse_json,
    parse_query_string,
    parse_csv_line,
    parse_key_value_pairs
)
from .validators import (
    is_email,
    is_url,
    is_phone,
    is_uuid,
    is_strong_password,
    is_credit_card,
    is_ip_address,
    is_alpha,
    is_alphanumeric,
    is_numeric,
    is_in_range,
    is_length,
    validate_with
)
from .sanitizers import (
    sanitize_html,
    sanitize_filename,
    sanitize_sql,
    sanitize_url,
    sanitize_email,
    sanitize_phone,
    sanitize_string,
    sanitize_number,
    sanitize_boolean,
    remove_whitespace,
    normalize_whitespace,
    remove_special_chars
)
from .generators import (
    generate_id,
    generate_uuid,
    generate_token,
    generate_password,
    generate_email,
    generate_phone,
    generate_string,
    generate_number,
    generate_float,
    generate_date,
    generate_list,
    generate_sequence,
    generate_choices
)
from .hashers import (
    hash_md5,
    hash_sha1,
    hash_sha256,
    hash_sha512,
    hash_blake2b,
    hash_file,
    hmac_hash,
    hash_password,
    verify_password,
    generate_salt,
    checksum,
    hash_multiple
)
from .file_utils import (
    read_file,
    write_file,
    read_json,
    write_json,
    file_exists,
    dir_exists,
    create_dir,
    delete_file,
    delete_dir,
    copy_file,
    move_file,
    get_file_size,
    get_file_extension,
    get_file_name,
    list_files,
    get_file_info
)
from .compression import (
    compress_gzip,
    decompress_gzip,
    compress_zlib,
    decompress_zlib,
    compress_json,
    decompress_json,
    compress_string,
    decompress_string,
    get_compression_ratio
)
from .logging_utils import (
    setup_logger,
    log_function_call,
    log_performance,
    log_error,
    log_info,
    log_warning,
    log_debug,
    create_log_context,
    format_log_message
)
from .config_utils import (
    Config,
    load_config_from_file,
    save_config_to_file,
    load_config_from_env,
    merge_configs,
    get_env_var,
    get_env_bool,
    get_env_int,
    get_env_float
)
from .time_utils import (
    get_timestamp,
    get_utc_now,
    get_local_now,
    datetime_to_timestamp,
    timestamp_to_datetime,
    add_time,
    subtract_time,
    time_difference,
    format_duration,
    is_business_day,
    get_business_days,
    sleep_seconds,
    sleep_milliseconds,
    measure_time,
    get_timezone_offset,
    convert_timezone
)
from .metrics import (
    Counter,
    Timer,
    MetricsCollector,
    track_time,
    calculate_rate,
    calculate_percentage,
    calculate_average,
    calculate_throughput
)
from .cache_advanced import (
    CacheEntry,
    AdvancedCache,
    cache_key,
    cached,
    invalidate_cache
)

__all__ = [
    # Errors
    "APIError",
    "ValidationError",
    "NotFoundError",
    "UnauthorizedError",
    "ForbiddenError",
    "ConflictError",
    "create_http_exception",
    "handle_service_error",
    # Validators
    "validate_user_id",
    "validate_email",
    "validate_date_string",
    "validate_date_not_future",
    "validate_range",
    "validate_required_fields",
    "validate_enum_value",
    "sanitize_string",
    # Response
    "create_success_response",
    "create_error_response",
    "create_paginated_response",
    # Cache
    "cache_result",
    "clear_cache",
    "get_cache_stats",
    # Async helpers
    "run_in_batches",
    "run_parallel",
    "async_retry",
    "timeout_after",
    # Pydantic helpers
    "model_to_dict",
    "models_to_dicts",
    "validate_and_parse",
    "partial_update_model",
    # Pagination
    "calculate_pagination",
    "paginate_items",
    "validate_pagination_params",
    # Filters
    "filter_by_field",
    "filter_by_date_range",
    "filter_by_custom_predicate",
    "sort_items",
    # Security
    "generate_secure_token",
    "hash_string",
    "validate_password_strength",
    "sanitize_input",
    "mask_sensitive_data",
    # Serialization
    "serialize_for_json",
    "to_json_string",
    "prepare_response_data",
    # Query params
    "parse_date_query",
    "parse_list_query",
    "create_date_range_query",
    "create_filter_query",
    # Date helpers
    "get_current_utc",
    "parse_iso_date",
    "format_iso_date",
    "add_days",
    "subtract_days",
    "days_between",
    "is_within_range",
    "get_start_of_day",
    "get_end_of_day",
    "get_start_of_week",
    "get_start_of_month",
    # String helpers
    "truncate_string",
    "normalize_string",
    "capitalize_words",
    "remove_special_chars",
    "extract_words",
    # Math helpers
    "calculate_percentage",
    "calculate_average",
    "calculate_median",
    "clamp_value",
    "round_to_decimals",
    # Transformers
    "transform_dict",
    "flatten_dict",
    "nest_dict",
    "map_list",
    "filter_list",
    "group_by",
    "pick_fields",
    "omit_fields",
    # Response builders
    "build_success_response",
    "build_error_response",
    "build_paginated_response",
    "build_list_response",
    # Type converters
    "to_int",
    "to_float",
    "to_bool",
    "to_string",
    "to_list",
    "to_datetime",
    # Collection helpers
    "chunk_list",
    "unique_list",
    "merge_dicts",
    "deep_merge_dicts",
    "get_nested_value",
    "set_nested_value",
    # Guards
    "guard_not_none",
    "guard_not_empty",
    "guard_in_range",
    "guard_in_list",
    "guard_positive",
    "guard_non_negative",
    # API docs
    "create_endpoint_summary",
    "create_response_examples",
    "create_parameter_description",
    # Testing helpers
    "create_mock_assessment_data",
    "create_mock_log_entry",
    "create_mock_progress_data",
    "assert_response_structure",
    # Performance helpers
    "measure_execution_time",
    "batch_process",
    "memoize_with_ttl",
    # Composers
    "compose",
    "pipe",
    "curry",
    "partial",
    # Functional helpers
    "map_function",
    "filter_function",
    "reduce_function",
    "identity",
    "constant",
    "maybe",
    # Predicates
    "is_none",
    "is_not_none",
    "is_empty",
    "is_not_empty",
    "is_positive",
    "is_non_negative",
    "is_negative",
    "is_zero",
    "is_in_range",
    "is_greater_than",
    "is_less_than",
    "is_equal",
    "is_not_equal",
    "is_in",
    "is_not_in",
    "all_true",
    "any_true",
    # Result types
    "Result",
    "safe_call",
    "safe_call_async",
    # Async composers
    "compose_async",
    "pipe_async",
    "parallel_map",
    "retry_async",
    # Validation combinators
    "combine_validators",
    "combine_validators_or",
    "validate_and_transform",
    "chain_validators",
    # Monads
    "Maybe",
    "Either",
    "maybe_decorator",
    "either_decorator",
    # Lenses
    "Lens",
    "lens",
    "prop_lens",
    "path_lens",
    # Functors
    "Functor",
    "ListFunctor",
    "DictFunctor",
    "list_functor",
    "dict_functor",
    # Streams
    "Stream",
    "stream",
    # Trampolines
    "Done",
    "More",
    "Trampoline",
    "trampoline",
    "make_trampoline",
    # Memoization
    "memoize",
    "memoize_with_key",
    "memoize_with_hash",
    "clear_memoization",
    # Observers
    "Observer",
    "create_observer",
    # Decorators
    "retry",
    "timeout",
    "log_execution",
    "validate_args",
    "rate_limit",
    # Iterators
    "take",
    "drop",
    "take_while",
    "drop_while",
    "chunk",
    "pairwise",
    "window",
    "interleave",
    # Promises
    "Promise",
    "create_promise",
    # Futures
    "all_futures",
    "any_future",
    "race_futures",
    "timeout_future",
    "create_future_from_value",
    "create_future_from_error",
    # Schedulers
    "Scheduler",
    "create_scheduler",
    # State management
    "State",
    "create_state",
    "state_reducer",
    # Event emitters
    "EventEmitter",
    "create_event_emitter",
    # Middleware utils
    "create_middleware",
    "request_logger",
    "response_timer",
    # Batch processors
    "batch_map",
    "batch_filter",
    "batch_reduce",
    # Queue utils
    "AsyncQueue",
    "create_queue",
    "process_queue",
    # Semaphores
    "RateLimiter",
    "create_rate_limiter",
    "with_semaphore",
    "with_rate_limit",
    # Backpressure
    "BackpressureController",
    "create_backpressure_controller",
    "with_backpressure",
    # Circuit breakers
    "CircuitBreaker",
    "CircuitState",
    "CircuitBreakerOpenError",
    "create_circuit_breaker",
    # Retry strategies
    "RetryStrategy",
    "retry_with_strategy",
    "retry_with_jitter",
    # Pools
    "ResourcePool",
    "create_resource_pool",
    "with_pool",
    # Workers
    "WorkerPool",
    "create_worker_pool",
    # Throttlers
    "Throttler",
    "Debouncer",
    "create_throttler",
    "create_debouncer",
    "with_throttle",
    # Comparators
    "compare_by",
    "compare_multiple",
    "natural_order",
    "reverse_order",
    # Sorters
    "sort_by",
    "sort_by_multiple",
    "stable_sort",
    "partial_sort",
    # Aggregators
    "group_by_key",
    "aggregate",
    "sum_by",
    "count_by",
    "average_by",
    # Formatters
    "format_number",
    "format_percentage",
    "format_currency",
    "format_duration",
    "format_bytes",
    # Encoders
    "encode_base64",
    "decode_base64",
    "encode_json_base64",
    "decode_json_base64",
    "url_encode",
    "url_decode",
    # Parsers
    "parse_json",
    "parse_query_string",
    "parse_csv_line",
    "parse_key_value_pairs",
    # Validators
    "is_email",
    "is_url",
    "is_phone",
    "is_uuid",
    "is_strong_password",
    "is_credit_card",
    "is_ip_address",
    "is_alpha",
    "is_alphanumeric",
    "is_numeric",
    "is_in_range",
    "is_length",
    "validate_with",
    # Sanitizers
    "sanitize_html",
    "sanitize_filename",
    "sanitize_sql",
    "sanitize_url",
    "sanitize_email",
    "sanitize_phone",
    "sanitize_string",
    "sanitize_number",
    "sanitize_boolean",
    "remove_whitespace",
    "normalize_whitespace",
    "remove_special_chars",
    # Generators
    "generate_id",
    "generate_uuid",
    "generate_token",
    "generate_password",
    "generate_email",
    "generate_phone",
    "generate_string",
    "generate_number",
    "generate_float",
    "generate_date",
    "generate_list",
    "generate_sequence",
    "generate_choices",
    # Hashers
    "hash_md5",
    "hash_sha1",
    "hash_sha256",
    "hash_sha512",
    "hash_blake2b",
    "hash_file",
    "hmac_hash",
    "hash_password",
    "verify_password",
    "generate_salt",
    "checksum",
    "hash_multiple",
    # File Utils
    "read_file",
    "write_file",
    "read_json",
    "write_json",
    "file_exists",
    "dir_exists",
    "create_dir",
    "delete_file",
    "delete_dir",
    "copy_file",
    "move_file",
    "get_file_size",
    "get_file_extension",
    "get_file_name",
    "list_files",
    "get_file_info",
    # Compression
    "compress_gzip",
    "decompress_gzip",
    "compress_zlib",
    "decompress_zlib",
    "compress_json",
    "decompress_json",
    "compress_string",
    "decompress_string",
    "get_compression_ratio",
    # Logging Utils
    "setup_logger",
    "log_function_call",
    "log_performance",
    "log_error",
    "log_info",
    "log_warning",
    "log_debug",
    "create_log_context",
    "format_log_message",
    # Config Utils
    "Config",
    "load_config_from_file",
    "save_config_to_file",
    "load_config_from_env",
    "merge_configs",
    "get_env_var",
    "get_env_bool",
    "get_env_int",
    "get_env_float",
    # Time Utils
    "get_timestamp",
    "get_utc_now",
    "get_local_now",
    "datetime_to_timestamp",
    "timestamp_to_datetime",
    "add_time",
    "subtract_time",
    "time_difference",
    "format_duration",
    "is_business_day",
    "get_business_days",
    "sleep_seconds",
    "sleep_milliseconds",
    "measure_time",
    "get_timezone_offset",
    "convert_timezone",
    # Metrics
    "Counter",
    "Timer",
    "MetricsCollector",
    "track_time",
    "calculate_rate",
    "calculate_percentage",
    "calculate_average",
    "calculate_throughput",
    # Advanced Cache
    "CacheEntry",
    "AdvancedCache",
    "cache_key",
    "cached",
    "invalidate_cache",
]
