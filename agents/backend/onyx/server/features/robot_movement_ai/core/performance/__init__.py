from .performance_utils import (
    euclidean_distance_fast,
    dot_product_fast,
    normalize_vector_fast,
    quaternion_multiply_fast,
    check_collision_fast,
    check_trajectory_collisions,
    trajectory_length_fast,
    batch_euclidean_distances,
    batch_dot_products,
    batch_normalize_vectors,
    batch_quaternion_multiply,
    batch_point_in_obstacle,
    batch_calculate_trajectory_distance,
    batch_calculate_trajectory_curvature,
    get_performance_info
)
from .performance import (
    measure_time,
    timeit,
    timeit_async,
    PerformanceProfiler,
    CacheManager
)

# Try to import optional modules
try:
    from .performance_monitor import PerformanceMonitor, PerformanceSnapshot, get_performance_monitor
except ImportError:
    PerformanceMonitor = None
    PerformanceSnapshot = None
    get_performance_monitor = None

try:
    from .performance_tuner import PerformanceTuner, TuningParameter, TuningResult, get_performance_tuner
except ImportError:
    PerformanceTuner = None
    TuningParameter = None
    TuningResult = None
    get_performance_tuner = None

try:
    from .fast_math import fast_dh_transform, fast_matrix_multiply
except ImportError:
    fast_dh_transform = None
    fast_matrix_multiply = None

__all__ = [
    'euclidean_distance_fast',
    'dot_product_fast',
    'normalize_vector_fast',
    'quaternion_multiply_fast',
    'check_collision_fast',
    'check_trajectory_collisions',
    'trajectory_length_fast',
    'batch_euclidean_distances',
    'batch_dot_products',
    'batch_normalize_vectors',
    'batch_quaternion_multiply',
    'batch_point_in_obstacle',
    'batch_calculate_trajectory_distance',
    'batch_calculate_trajectory_curvature',
    'get_performance_info',
    'measure_time',
    'timeit',
    'timeit_async',
    'PerformanceProfiler',
    'CacheManager',
    'PerformanceMonitor',
    'PerformanceSnapshot',
    'get_performance_monitor',
    'PerformanceTuner',
    'TuningParameter',
    'TuningResult',
    'get_performance_tuner',
    'fast_dh_transform',
    'fast_matrix_multiply',
]

