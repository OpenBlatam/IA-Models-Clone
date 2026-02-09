"""
Native Extensions Module
========================

Módulo para extensiones nativas en C++ y otros lenguajes de alto rendimiento.
Proporciona bindings optimizados para operaciones críticas del sistema.
"""

# Intentar importar extensiones C++
try:
    from . import cpp_extensions
    from .cpp_extensions import (
        FastIK,
        FastTrajectoryOptimizer,
        FastMatrixOps,
        FastCollisionDetector,
        FastTransform3D,
        FastVectorOps,
        FastInterpolation,
        FastQuaternion,
        FastHomogeneousTransform,
        FastGeometry
    )
    CPP_AVAILABLE = True
except ImportError:
    CPP_AVAILABLE = False
    FastIK = None
    FastTrajectoryOptimizer = None
    FastMatrixOps = None
    FastCollisionDetector = None
    FastTransform3D = None
    FastVectorOps = None
    FastInterpolation = None
    FastQuaternion = None
    FastHomogeneousTransform = None
    FastGeometry = None

# Intentar importar extensiones Rust
try:
    from .rust_extensions import (
        fast_json_parse,
        fast_string_search,
        fast_hash,
        fast_array_sum,
        fast_array_max,
        fast_array_min,
        fast_array_mean,
        fast_array_std,
        fast_array_median,
        fast_array_percentile,
        fast_array_filter,
        fast_binary_search,
        fast_array_sort,
        fast_string_count,
        fast_string_replace,
        fast_string_split,
        fast_string_join,
        fast_string_trim,
        fast_string_upper,
        fast_string_lower,
        fast_string_find_all,
        fast_string_starts_with,
        fast_string_ends_with,
        fast_array_variance,
        fast_array_range,
        fast_array_cumsum,
        fast_array_cumprod,
        fast_json_validate,
        fast_json_stringify
    )
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False
    fast_json_parse = None
    fast_string_search = None
    fast_hash = None
    fast_array_sum = None
    fast_array_max = None
    fast_array_min = None
    fast_array_mean = None
    fast_array_std = None
    fast_array_median = None
    fast_array_percentile = None
    fast_array_filter = None
    fast_binary_search = None
    fast_array_sort = None
    fast_string_count = None
    fast_string_replace = None
    fast_string_split = None
    fast_string_join = None
    fast_string_trim = None
    fast_string_upper = None
    fast_string_lower = None
    fast_string_find_all = None
    fast_string_starts_with = None
    fast_string_ends_with = None
    fast_array_variance = None
    fast_array_range = None
    fast_array_cumsum = None
    fast_array_cumprod = None
    fast_json_validate = None
    fast_json_stringify = None

# Importar wrappers profesionales (siempre disponibles, con fallback)
from .wrapper import (
    NativeIKWrapper,
    NativeTrajectoryOptimizerWrapper,
    NativeMatrixOpsWrapper,
    NativeCollisionDetectorWrapper,
    NativeTransform3DWrapper,
    NativeVectorOpsWrapper,
    NativeInterpolationWrapper,
    NativeQuaternionWrapper,
    NativeHomogeneousTransformWrapper,
    NativeGeometryWrapper,
    NativeMathUtilsWrapper,
    json_parse,
    string_search,
    hash_data,
    array_mean,
    array_std,
    array_median,
    array_percentile,
    array_variance,
    array_range,
    array_cumsum,
    array_cumprod,
    array_filter,
    binary_search,
    string_count,
    string_replace,
    string_split,
    string_join,
    string_trim,
    string_upper,
    string_lower,
    string_find_all,
    string_starts_with,
    string_ends_with,
    json_validate,
    get_native_extensions_status,
    validate_array,
    handle_native_errors,
    performance_timer,
)

__all__ = [
    # Flags de disponibilidad
    'CPP_AVAILABLE',
    'RUST_AVAILABLE',
    # Clases C++ (pueden ser None si no están disponibles)
    'FastIK',
    'FastTrajectoryOptimizer',
    'FastMatrixOps',
    'FastCollisionDetector',
    'FastTransform3D',
    'FastVectorOps',
    'FastInterpolation',
    'FastQuaternion',
    'FastHomogeneousTransform',
    'FastGeometry',
    # Funciones Rust (pueden ser None si no están disponibles)
    'fast_json_parse',
    'fast_string_search',
    'fast_hash',
    'fast_array_sum',
    'fast_array_max',
    'fast_array_min',
    'fast_array_mean',
    'fast_array_std',
    'fast_array_median',
    'fast_array_percentile',
    'fast_array_filter',
    'fast_binary_search',
    'fast_array_sort',
    'fast_string_count',
    'fast_string_replace',
    'fast_string_split',
    'fast_string_join',
    'fast_string_trim',
    'fast_string_upper',
    'fast_string_lower',
    'fast_string_find_all',
    'fast_string_starts_with',
    'fast_string_ends_with',
    'fast_array_variance',
    'fast_array_range',
    'fast_array_cumsum',
    'fast_array_cumprod',
    'fast_json_validate',
    'fast_json_stringify',
    # Wrappers profesionales (siempre disponibles, recomendado)
    'NativeIKWrapper',
    'NativeTrajectoryOptimizerWrapper',
    'NativeMatrixOpsWrapper',
    'NativeCollisionDetectorWrapper',
    'NativeTransform3DWrapper',
    'NativeVectorOpsWrapper',
    'NativeInterpolationWrapper',
    'NativeQuaternionWrapper',
    'NativeHomogeneousTransformWrapper',
    'NativeGeometryWrapper',
    'NativeMathUtilsWrapper',
    'json_parse',
    'string_search',
    'hash_data',
    'array_mean',
    'array_std',
    'array_median',
    'array_percentile',
    'array_variance',
    'array_range',
    'array_cumsum',
    'array_cumprod',
    'array_filter',
    'binary_search',
    'string_count',
    'string_replace',
    'string_split',
    'string_join',
    'string_trim',
    'string_upper',
    'string_lower',
    'string_find_all',
    'string_starts_with',
    'string_ends_with',
    'json_validate',
    'get_native_extensions_status',
    # Utilidades profesionales
    'validate_array',
    'handle_native_errors',
    'performance_timer',
]

