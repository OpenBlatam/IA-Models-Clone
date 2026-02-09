"""
Performance Utilities
=====================

Utilidades de alto rendimiento usando numba, JAX y optimizaciones.
"""

import numpy as np
from typing import Optional, Tuple, List

try:
    from numba import jit, njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range

try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

if NUMBA_AVAILABLE:
    @njit(cache=True, fastmath=True, inline='always')
    def euclidean_distance_fast(a: np.ndarray, b: np.ndarray) -> float:
        """Distancia euclidiana optimizada con numba."""
        diff = a - b
        return np.sqrt(diff[0]*diff[0] + diff[1]*diff[1] + diff[2]*diff[2])
    
    @njit(cache=True, fastmath=True)
    def dot_product_fast(a: np.ndarray, b: np.ndarray) -> float:
        """Producto punto optimizado."""
        return np.dot(a, b)
    
    @njit(cache=True, fastmath=True)
    def normalize_vector_fast(v: np.ndarray) -> np.ndarray:
        """Normalizar vector optimizado."""
        norm = np.sqrt(np.sum(v ** 2))
        if norm > 1e-10:
            return v / norm
        return v
    
    @njit(cache=True, fastmath=True)
    def quaternion_multiply_fast(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """Multiplicación de cuaterniones optimizada."""
        w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
        w2, x2, y2, z2 = q2[0], q2[1], q2[2], q2[3]
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])
    
    @njit(cache=True, fastmath=True, parallel=True)
    def batch_euclidean_distances(positions: np.ndarray) -> np.ndarray:
        """Calcular distancias euclidianas en batch."""
        n = positions.shape[0]
        distances = np.zeros((n, n))
        for i in prange(n):
            for j in prange(n):
                if i != j:
                    distances[i, j] = np.sqrt(np.sum((positions[i] - positions[j]) ** 2))
        return distances
    
    @njit(cache=True, fastmath=True)
    def trajectory_length_fast(trajectory: np.ndarray) -> float:
        """Calcular longitud de trayectoria optimizada."""
        if trajectory.shape[0] < 2:
            return 0.0
        total = 0.0
        for i in range(1, trajectory.shape[0]):
            total += np.sqrt(np.sum((trajectory[i] - trajectory[i-1]) ** 2))
        return total
    
    @njit(cache=True, fastmath=True)
    def check_collision_fast(point: np.ndarray, obstacle: np.ndarray) -> bool:
        """Verificar colisión punto-obstáculo optimizado."""
        min_bounds = obstacle[:3]
        max_bounds = obstacle[3:]
        return np.all(point >= min_bounds) and np.all(point <= max_bounds)
    
    @njit(cache=True, fastmath=True, parallel=True)
    def check_trajectory_collisions(trajectory: np.ndarray, obstacles: np.ndarray) -> np.ndarray:
        """Verificar colisiones de trayectoria con obstáculos."""
        n_points = trajectory.shape[0]
        n_obstacles = obstacles.shape[0]
        collisions = np.zeros(n_points, dtype=np.bool_)
        
        for i in prange(n_points):
            point = trajectory[i]
            for j in range(n_obstacles):
                if check_collision_fast(point, obstacles[j]):
                    collisions[i] = True
                    break
        
        return collisions

else:
    def euclidean_distance_fast(a: np.ndarray, b: np.ndarray) -> float:
        return np.linalg.norm(a - b)
    
    def dot_product_fast(a: np.ndarray, b: np.ndarray) -> float:
        return np.dot(a, b)
    
    def normalize_vector_fast(v: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(v)
        if norm > 1e-10:
            return v / norm
        return v
    
    def quaternion_multiply_fast(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
        w2, x2, y2, z2 = q2[0], q2[1], q2[2], q2[3]
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])
    
    def batch_euclidean_distances(positions: np.ndarray) -> np.ndarray:
        n = positions.shape[0]
        distances = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    distances[i, j] = np.linalg.norm(positions[i] - positions[j])
        return distances
    
    def trajectory_length_fast(trajectory: np.ndarray) -> float:
        if trajectory.shape[0] < 2:
            return 0.0
        diffs = np.diff(trajectory, axis=0)
        return np.sum(np.linalg.norm(diffs, axis=1))
    
    def check_collision_fast(point: np.ndarray, obstacle: np.ndarray) -> bool:
        min_bounds = obstacle[:3]
        max_bounds = obstacle[3:]
        return np.all(point >= min_bounds) and np.all(point <= max_bounds)
    
    def check_trajectory_collisions(trajectory: np.ndarray, obstacles: np.ndarray) -> np.ndarray:
        n_points = trajectory.shape[0]
        collisions = np.zeros(n_points, dtype=bool)
        for i in range(n_points):
            point = trajectory[i]
            for obstacle in obstacles:
                if check_collision_fast(point, obstacle):
                    collisions[i] = True
                    break
        return collisions

if JAX_AVAILABLE:
    @jax.jit
    def jax_euclidean_distance(a: jnp.ndarray, b: jnp.ndarray) -> float:
        """Distancia euclidiana con JAX (GPU/TPU)."""
        return jnp.linalg.norm(a - b)
    
    @jax.jit
    def jax_batch_distances(positions: jnp.ndarray) -> jnp.ndarray:
        """Distancias en batch con JAX."""
        return jnp.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=-1)
    
    @jax.jit
    def jax_trajectory_length(trajectory: jnp.ndarray) -> float:
        """Longitud de trayectoria con JAX."""
        if trajectory.shape[0] < 2:
            return 0.0
        diffs = jnp.diff(trajectory, axis=0)
        return jnp.sum(jnp.linalg.norm(diffs, axis=1))

# Batch operations stubs
def batch_dot_products(vectors1: np.ndarray, vectors2: np.ndarray) -> np.ndarray:
    """Calcular productos punto en batch."""
    return np.sum(vectors1 * vectors2, axis=-1)

def batch_normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    """Normalizar vectores en batch."""
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    norms = np.where(norms > 1e-10, norms, 1.0)
    return vectors / norms

def batch_quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Multiplicar cuaterniones en batch."""
    if q1.shape != q2.shape:
        raise ValueError("Quaternion arrays must have same shape")
    results = []
    for i in range(q1.shape[0]):
        results.append(quaternion_multiply_fast(q1[i], q2[i]))
    return np.array(results)

def batch_point_in_obstacle(points: np.ndarray, obstacles: np.ndarray) -> np.ndarray:
    """Verificar si puntos están en obstáculos."""
    n_points = points.shape[0]
    n_obstacles = obstacles.shape[0]
    results = np.zeros(n_points, dtype=bool)
    
    for i in range(n_points):
        point = points[i]
        for j in range(n_obstacles):
            if check_collision_fast(point, obstacles[j]):
                results[i] = True
                break
    
    return results

def batch_calculate_trajectory_distance(trajectories: List[np.ndarray]) -> np.ndarray:
    """Calcular distancias de trayectorias en batch."""
    distances = []
    for traj in trajectories:
        distances.append(trajectory_length_fast(traj))
    return np.array(distances)

def batch_calculate_trajectory_curvature(trajectories: List[np.ndarray]) -> np.ndarray:
    """Calcular curvatura de trayectorias en batch."""
    curvatures = []
    for traj in trajectories:
        if traj.shape[0] < 3:
            curvatures.append(0.0)
            continue
        # Simple curvature approximation
        diffs = np.diff(traj, axis=0)
        norms = np.linalg.norm(diffs, axis=1)
        if np.sum(norms) < 1e-10:
            curvatures.append(0.0)
        else:
            # Approximate curvature as variation in direction
            directions = diffs / (norms[:, np.newaxis] + 1e-10)
            direction_changes = np.linalg.norm(np.diff(directions, axis=0), axis=1)
            curvatures.append(np.mean(direction_changes) if len(direction_changes) > 0 else 0.0)
    return np.array(curvatures)

def get_performance_info() -> dict:
    """Obtener información de librerías de rendimiento disponibles."""
    try:
        from ..serialization.serialization import ORJSON_AVAILABLE, MSGPACK_AVAILABLE
    except ImportError:
        ORJSON_AVAILABLE = False
        MSGPACK_AVAILABLE = False
    
    return {
        "numba_available": NUMBA_AVAILABLE,
        "jax_available": JAX_AVAILABLE,
        "orjson_available": ORJSON_AVAILABLE,
        "msgpack_available": MSGPACK_AVAILABLE
    }

