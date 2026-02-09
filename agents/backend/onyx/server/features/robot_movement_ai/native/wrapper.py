"""
Native Extensions Wrapper - Professional Implementation
======================================================

Wrapper Python profesional que integra extensiones nativas C++ y Rust
con el código existente del sistema. Incluye manejo robusto de errores,
logging estructurado, validación de entrada y fallbacks elegantes.

Sigue las mejores prácticas de PyTorch, Transformers y desarrollo profesional.
"""

import numpy as np
import logging
from typing import List, Optional, Tuple, Union, Dict, Any
import warnings
from functools import wraps
from contextlib import contextmanager
import time

logger = logging.getLogger(__name__)

# ============================================================================
# Importación de Extensiones Nativas con Manejo Robusto de Errores
# ============================================================================

CPP_AVAILABLE = False
RUST_AVAILABLE = False

# Intentar importar extensiones C++
try:
    from . import cpp_extensions
    from .cpp_extensions import (
        FastIK,
        FastTrajectoryOptimizer,
        FastMatrixOps,
        FastCollisionDetector
    )
    CPP_AVAILABLE = True
    logger.info("✅ C++ extensions loaded successfully")
except ImportError as e:
    CPP_AVAILABLE = False
    logger.warning(f"⚠️  C++ extensions not available: {e}. Using Python fallback.")
    FastIK = None
    FastTrajectoryOptimizer = None
    FastMatrixOps = None
    FastCollisionDetector = None
except Exception as e:
    CPP_AVAILABLE = False
    logger.error(f"❌ Error loading C++ extensions: {e}", exc_info=True)
    FastIK = None
    FastTrajectoryOptimizer = None
    FastMatrixOps = None
    FastCollisionDetector = None

# Intentar importar extensiones Rust
try:
    from .rust_extensions import (
        fast_json_parse,
        fast_string_search,
        fast_hash,
        fast_array_sum,
        fast_array_max,
        fast_array_min
    )
    RUST_AVAILABLE = True
    logger.info("✅ Rust extensions loaded successfully")
except ImportError as e:
    RUST_AVAILABLE = False
    logger.warning(f"⚠️  Rust extensions not available: {e}. Using Python fallback.")
except Exception as e:
    RUST_AVAILABLE = False
    logger.error(f"❌ Error loading Rust extensions: {e}", exc_info=True)


# ============================================================================
# Utilidades de Validación y Manejo de Errores
# ============================================================================

def validate_array(
    arr: np.ndarray,
    shape: Optional[Tuple[int, ...]] = None,
    dtype: Optional[type] = None,
    name: str = "array"
) -> np.ndarray:
    """
    Validar y convertir array de NumPy.
    
    Args:
        arr: Array a validar
        shape: Shape esperado (None = cualquier shape)
        dtype: Tipo de datos esperado
        name: Nombre del array para mensajes de error
        
    Returns:
        Array validado
        
    Raises:
        ValueError: Si la validación falla
        TypeError: Si el tipo es incorrecto
    """
    if not isinstance(arr, np.ndarray):
        try:
            arr = np.asarray(arr, dtype=dtype)
        except Exception as e:
            raise TypeError(f"{name} must be convertible to numpy array: {e}")
    
    if dtype is not None and arr.dtype != dtype:
        try:
            arr = arr.astype(dtype)
        except Exception as e:
            raise ValueError(f"{name} cannot be converted to {dtype}: {e}")
    
    if shape is not None:
        if arr.shape != shape:
            raise ValueError(
                f"{name} must have shape {shape}, got {arr.shape}"
            )
    
    if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
        raise ValueError(f"{name} contains NaN or Inf values")
    
    return arr


def handle_native_errors(func):
    """
    Decorador para manejar errores de extensiones nativas con fallback.
    
    Automáticamente captura errores y usa implementación Python como fallback.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.warning(
                f"Native implementation failed in {func.__name__}: {e}. "
                "Using Python fallback."
            )
            # Intentar fallback si está disponible
            if hasattr(func, '_fallback'):
                return func._fallback(*args, **kwargs)
            raise
    return wrapper


@contextmanager
def performance_timer(operation_name: str):
    """
    Context manager para medir tiempo de operaciones.
    
    Args:
        operation_name: Nombre de la operación a medir
    """
    start_time = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start_time
        logger.debug(f"{operation_name} took {elapsed:.4f} seconds")


# ============================================================================
# Wrapper Profesional para Cinemática Inversa
# ============================================================================

class NativeIKWrapper:
    """
    Wrapper profesional para cinemática inversa nativa.
    
    Características:
    - Validación robusta de entrada
    - Manejo de errores con fallback
    - Logging estructurado
    - Medición de rendimiento
    - Soporte para múltiples configuraciones
    """
    
    def __init__(
        self,
        link_lengths: List[float],
        joint_limits: List[Tuple[float, float]],
        max_iterations: int = 100,
        tolerance: float = 1e-6,
        use_native: bool = True
    ):
        """
        Inicializar wrapper de cinemática inversa.
        
        Args:
            link_lengths: Longitudes de los eslabones
            joint_limits: Límites de articulaciones [(min, max), ...]
            max_iterations: Máximo número de iteraciones
            tolerance: Tolerancia para convergencia
            use_native: Si usar implementación nativa (si está disponible)
            
        Raises:
            ValueError: Si los parámetros son inválidos
        """
        # Validación de entrada
        if not link_lengths or len(link_lengths) == 0:
            raise ValueError("link_lengths cannot be empty")
        
        if any(l <= 0 for l in link_lengths):
            raise ValueError("All link lengths must be positive")
        
        if len(joint_limits) != len(link_lengths):
            raise ValueError(
                f"joint_limits length ({len(joint_limits)}) must match "
                f"link_lengths length ({len(link_lengths)})"
            )
        
        if max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        
        if tolerance <= 0:
            raise ValueError("tolerance must be positive")
        
        self.link_lengths = np.array(link_lengths, dtype=np.float64)
        self.joint_limits = joint_limits
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.use_native = use_native and CPP_AVAILABLE and FastIK is not None
        
        # Inicializar solucionador
        if self.use_native:
            try:
                self.ik_solver = FastIK(
                    link_lengths,
                    joint_limits,
                    max_iterations,
                    tolerance
                )
                logger.info(
                    f"✅ Using native C++ inverse kinematics "
                    f"(links={len(link_lengths)}, max_iter={max_iterations})"
                )
            except Exception as e:
                logger.warning(f"Failed to initialize native IK: {e}")
                self.use_native = False
                self.ik_solver = None
        else:
            self.ik_solver = None
            logger.info("Using Python fallback for inverse kinematics")
    
    @handle_native_errors
    def solve(
        self,
        target_position: Union[List[float], np.ndarray],
        initial_guess: Optional[List[float]] = None,
        return_metadata: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, Any]]]:
        """
        Resolver cinemática inversa.
        
        Args:
            target_position: Posición objetivo [x, y, z]
            initial_guess: Estimación inicial de ángulos (opcional)
            return_metadata: Si retornar metadatos adicionales
            
        Returns:
            Ángulos de articulaciones o (ángulos, metadatos)
            
        Raises:
            ValueError: Si la entrada es inválida
            RuntimeError: Si no se puede resolver
        """
        # Validar entrada
        target_pos = validate_array(
            target_position,
            shape=(3,),
            dtype=np.float64,
            name="target_position"
        )
        
        if initial_guess is None:
            initial_guess = [0.0] * len(self.link_lengths)
        else:
            if len(initial_guess) != len(self.link_lengths):
                raise ValueError(
                    f"initial_guess length ({len(initial_guess)}) must match "
                    f"link_lengths length ({len(self.link_lengths)})"
                )
        
        # Resolver
        with performance_timer("IK solve"):
            if self.use_native:
                try:
                    # Convertir a formato C++
                    from .cpp_extensions import Vector3d
                    target_vec = Vector3d(
                        float(target_pos[0]),
                        float(target_pos[1]),
                        float(target_pos[2])
                    )
                    
                    solution = self.ik_solver.solve(target_vec, initial_guess)
                    angles = np.array(solution, dtype=np.float64)
                    
                    # Validar solución
                    self._validate_solution(angles)
                    
                    if return_metadata:
                        metadata = {
                            "method": "native_cpp",
                            "iterations": self.max_iterations,
                            "tolerance": self.tolerance
                        }
                        return angles, metadata
                    
                    return angles
                    
                except Exception as e:
                    logger.warning(f"Native IK failed: {e}. Using fallback.")
                    return self._python_solve(target_pos, initial_guess, return_metadata)
            else:
                return self._python_solve(target_pos, initial_guess, return_metadata)
    
    def _python_solve(
        self,
        target_position: np.ndarray,
        initial_guess: List[float],
        return_metadata: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, Any]]]:
        """Implementación Python pura como fallback."""
        try:
            from ..core.inverse_kinematics import InverseKinematicsSolver
            from ..core.inverse_kinematics import EndEffectorPose
            
            solver = InverseKinematicsSolver(
                num_joints=len(self.link_lengths)
            )
            
            pose = EndEffectorPose(
                position=target_position,
                orientation=np.array([0, 0, 0, 1])  # Quaternion identidad
            )
            
            result = solver.solve(pose, initial_guess)
            angles = np.array(result.angles, dtype=np.float64)
            
            self._validate_solution(angles)
            
            if return_metadata:
                metadata = {
                    "method": "python_fallback",
                    "iterations": self.max_iterations,
                    "tolerance": self.tolerance
                }
                return angles, metadata
            
            return angles
            
        except Exception as e:
            logger.error(f"Python IK fallback failed: {e}", exc_info=True)
            raise RuntimeError(f"Failed to solve inverse kinematics: {e}")
    
    def _validate_solution(self, angles: np.ndarray):
        """Validar solución de IK."""
        # Verificar límites de articulaciones
        for i, (angle, (min_angle, max_angle)) in enumerate(
            zip(angles, self.joint_limits)
        ):
            if angle < min_angle or angle > max_angle:
                logger.warning(
                    f"Joint {i} angle {angle:.4f} outside limits "
                    f"[{min_angle:.4f}, {max_angle:.4f}]"
                )
        
        # Verificar NaN/Inf
        if np.any(np.isnan(angles)) or np.any(np.isinf(angles)):
            raise ValueError("Solution contains NaN or Inf values")


# ============================================================================
# Wrapper Profesional para Optimización de Trayectorias
# ============================================================================

class NativeTrajectoryOptimizerWrapper:
    """
    Wrapper profesional para optimización de trayectorias nativa.
    
    Características:
    - Validación de trayectorias
    - Optimización multi-objetivo
    - Manejo de obstáculos
    - Logging de métricas
    """
    
    def __init__(
        self,
        energy_weight: float = 0.3,
        time_weight: float = 0.3,
        smoothness_weight: float = 0.2,
        use_native: bool = True
    ):
        """
        Inicializar optimizador de trayectorias.
        
        Args:
            energy_weight: Peso para minimizar energía
            time_weight: Peso para minimizar tiempo
            smoothness_weight: Peso para suavidad
            use_native: Si usar implementación nativa
        """
        # Validar pesos
        total_weight = energy_weight + time_weight + smoothness_weight
        if abs(total_weight - 1.0) > 1e-6:
            logger.warning(
                f"Weights sum to {total_weight}, normalizing to 1.0"
            )
            energy_weight /= total_weight
            time_weight /= total_weight
            smoothness_weight /= total_weight
        
        self.energy_weight = energy_weight
        self.time_weight = time_weight
        self.smoothness_weight = smoothness_weight
        self.use_native = use_native and CPP_AVAILABLE and FastTrajectoryOptimizer is not None
        
        if self.use_native:
            try:
                self.optimizer = FastTrajectoryOptimizer(
                    energy_weight,
                    time_weight,
                    smoothness_weight
                )
                logger.info("✅ Using native C++ trajectory optimizer")
            except Exception as e:
                logger.warning(f"Failed to initialize native optimizer: {e}")
                self.use_native = False
                self.optimizer = None
        else:
            self.optimizer = None
            logger.info("Using Python fallback for trajectory optimization")
    
    @handle_native_errors
    def optimize(
        self,
        trajectory: np.ndarray,
        obstacles: Optional[np.ndarray] = None,
        return_metrics: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, float]]]:
        """
        Optimizar trayectoria.
        
        Args:
            trajectory: Trayectoria Nx3 (posiciones)
            obstacles: Obstáculos Mx4 (centro x, y, z, radio)
            return_metrics: Si retornar métricas de optimización
            
        Returns:
            Trayectoria optimizada o (trayectoria, métricas)
        """
        # Validar trayectoria
        trajectory = validate_array(
            trajectory,
            shape=(None, 3),
            dtype=np.float64,
            name="trajectory"
        )
        
        if len(trajectory) < 2:
            raise ValueError("Trajectory must have at least 2 points")
        
        # Validar obstáculos
        if obstacles is None:
            obstacles = np.empty((0, 4), dtype=np.float64)
        else:
            obstacles = validate_array(
                obstacles,
                shape=(None, 4),
                dtype=np.float64,
                name="obstacles"
            )
        
        # Optimizar
        with performance_timer("Trajectory optimization"):
            if self.use_native:
                try:
                    result = self.optimizer.optimize(trajectory, obstacles)
                    optimized = np.array(result, dtype=np.float64)
                    
                    if return_metrics:
                        metrics = self._compute_metrics(trajectory, optimized)
                        return optimized, metrics
                    
                    return optimized
                    
                except Exception as e:
                    logger.warning(f"Native optimization failed: {e}. Using fallback.")
                    return self._python_optimize(trajectory, obstacles, return_metrics)
            else:
                return self._python_optimize(trajectory, obstacles, return_metrics)
    
    def _python_optimize(
        self,
        trajectory: np.ndarray,
        obstacles: np.ndarray,
        return_metrics: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, float]]]:
        """Implementación Python pura como fallback."""
        try:
            from ..core.trajectory_optimizer import TrajectoryOptimizer
            from ..core.trajectory_optimizer import TrajectoryPoint
            
            optimizer = TrajectoryOptimizer()
            
            # Convertir a formato esperado
            traj_points = [
                TrajectoryPoint(
                    position=point,
                    orientation=np.array([0, 0, 0, 1])
                )
                for point in trajectory
            ]
            
            start = traj_points[0]
            goal = traj_points[-1]
            obs_list = [obs[:3] for obs in obstacles] if len(obstacles) > 0 else None
            
            optimized_points = optimizer.optimize(start, goal, obs_list)
            optimized = np.array([p.position for p in optimized_points], dtype=np.float64)
            
            if return_metrics:
                metrics = self._compute_metrics(trajectory, optimized)
                return optimized, metrics
            
            return optimized
            
        except Exception as e:
            logger.error(f"Python optimization fallback failed: {e}", exc_info=True)
            raise RuntimeError(f"Failed to optimize trajectory: {e}")
    
    def _compute_metrics(
        self,
        original: np.ndarray,
        optimized: np.ndarray
    ) -> Dict[str, float]:
        """Calcular métricas de optimización."""
        # Longitud de trayectoria
        def trajectory_length(traj):
            return np.sum(np.linalg.norm(np.diff(traj, axis=0), axis=1))
        
        original_length = trajectory_length(original)
        optimized_length = trajectory_length(optimized)
        
        # Suavidad (curvatura)
        def smoothness(traj):
            if len(traj) < 3:
                return 0.0
            diffs = np.diff(traj, axis=0)
            angles = np.arccos(
                np.clip(
                    np.sum(diffs[:-1] * diffs[1:], axis=1) /
                    (np.linalg.norm(diffs[:-1], axis=1) * np.linalg.norm(diffs[1:], axis=1)),
                    -1, 1
                )
            )
            return np.mean(angles)
        
        return {
            "original_length": float(original_length),
            "optimized_length": float(optimized_length),
            "length_reduction": float((original_length - optimized_length) / original_length * 100),
            "original_smoothness": float(smoothness(original)),
            "optimized_smoothness": float(smoothness(optimized)),
            "smoothness_improvement": float(smoothness(original) - smoothness(optimized))
        }


# ============================================================================
# Wrapper Profesional para Operaciones Matriciales
# ============================================================================

class NativeMatrixOpsWrapper:
    """
    Wrapper profesional para operaciones matriciales nativas.
    
    Proporciona operaciones matriciales optimizadas con validación
    y manejo de errores robusto.
    """
    
    @staticmethod
    @handle_native_errors
    def matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Multiplicación de matrices optimizada.
        
        Args:
            a: Matriz A [m, n]
            b: Matriz B [n, p]
            
        Returns:
            Matriz resultado [m, p]
        """
        a = validate_array(a, shape=(None, None), dtype=np.float64, name="a")
        b = validate_array(b, shape=(None, None), dtype=np.float64, name="b")
        
        if a.shape[1] != b.shape[0]:
            raise ValueError(
                f"Incompatible matrix dimensions: a.shape={a.shape}, b.shape={b.shape}"
            )
        
        if CPP_AVAILABLE and FastMatrixOps is not None:
            try:
                with performance_timer("Native matrix multiplication"):
                    return FastMatrixOps.matmul(a, b)
            except Exception as e:
                logger.warning(f"Native matmul failed: {e}. Using numpy.")
                return np.matmul(a, b)
        else:
            return np.matmul(a, b)
    
    @staticmethod
    @handle_native_errors
    def inv(a: np.ndarray) -> np.ndarray:
        """
        Inversa de matriz.
        
        Args:
            a: Matriz cuadrada [n, n]
            
        Returns:
            Matriz inversa [n, n]
        """
        a = validate_array(a, shape=(None, None), dtype=np.float64, name="a")
        
        if a.shape[0] != a.shape[1]:
            raise ValueError(f"Matrix must be square, got shape {a.shape}")
        
        if CPP_AVAILABLE and FastMatrixOps is not None:
            try:
                with performance_timer("Native matrix inverse"):
                    return FastMatrixOps.inv(a)
            except Exception as e:
                logger.warning(f"Native inv failed: {e}. Using numpy.")
                return np.linalg.inv(a)
        else:
            return np.linalg.inv(a)
    
    @staticmethod
    @handle_native_errors
    def det(a: np.ndarray) -> float:
        """
        Determinante de matriz.
        
        Args:
            a: Matriz cuadrada [n, n]
            
        Returns:
            Determinante
        """
        a = validate_array(a, shape=(None, None), dtype=np.float64, name="a")
        
        if a.shape[0] != a.shape[1]:
            raise ValueError(f"Matrix must be square, got shape {a.shape}")
        
        if CPP_AVAILABLE and FastMatrixOps is not None:
            try:
                with performance_timer("Native matrix determinant"):
                    return float(FastMatrixOps.det(a))
            except Exception as e:
                logger.warning(f"Native det failed: {e}. Using numpy.")
                return float(np.linalg.det(a))
        else:
            return float(np.linalg.det(a))
    
    @staticmethod
    @handle_native_errors
    def transpose(a: np.ndarray) -> np.ndarray:
        """
        Transpuesta de matriz.
        
        Args:
            a: Matriz [m, n]
            
        Returns:
            Matriz transpuesta [n, m]
        """
        a = validate_array(a, shape=(None, None), dtype=np.float64, name="a")
        
        if CPP_AVAILABLE and FastMatrixOps is not None:
            try:
                with performance_timer("Native matrix transpose"):
                    return FastMatrixOps.transpose(a)
            except Exception as e:
                logger.warning(f"Native transpose failed: {e}. Using numpy.")
                return np.transpose(a)
        else:
            return np.transpose(a)
    
    @staticmethod
    @handle_native_errors
    def norm(a: np.ndarray) -> float:
        """
        Norma de Frobenius de matriz.
        
        Args:
            a: Matriz [m, n]
            
        Returns:
            Norma de Frobenius
        """
        a = validate_array(a, shape=(None, None), dtype=np.float64, name="a")
        
        if CPP_AVAILABLE and FastMatrixOps is not None:
            try:
                with performance_timer("Native matrix norm"):
                    return float(FastMatrixOps.norm(a))
            except Exception as e:
                logger.warning(f"Native norm failed: {e}. Using numpy.")
                return float(np.linalg.norm(a))
        else:
            return float(np.linalg.norm(a))
    
    @staticmethod
    @handle_native_errors
    def trace(a: np.ndarray) -> float:
        """
        Traza de matriz.
        
        Args:
            a: Matriz cuadrada [n, n]
            
        Returns:
            Traza
        """
        a = validate_array(a, shape=(None, None), dtype=np.float64, name="a")
        
        if a.shape[0] != a.shape[1]:
            raise ValueError(f"Matrix must be square, got shape {a.shape}")
        
        if CPP_AVAILABLE and FastMatrixOps is not None:
            try:
                with performance_timer("Native matrix trace"):
                    return float(FastMatrixOps.trace(a))
            except Exception as e:
                logger.warning(f"Native trace failed: {e}. Using numpy.")
                return float(np.trace(a))
        else:
            return float(np.trace(a))


# ============================================================================
# Wrapper Profesional para Detección de Colisiones
# ============================================================================

class NativeCollisionDetectorWrapper:
    """
    Wrapper profesional para detección de colisiones nativa.
    
    Proporciona detección rápida de colisiones con validación
    y reportes detallados.
    """
    
    @staticmethod
    @handle_native_errors
    def check_trajectory_collision(
        trajectory: np.ndarray,
        obstacles: np.ndarray,
        return_details: bool = False
    ) -> Union[bool, Tuple[bool, Dict[str, Any]]]:
        """
        Verificar colisión de trayectoria con obstáculos.
        
        Args:
            trajectory: Trayectoria Nx3
            obstacles: Obstáculos Mx4 (centro x, y, z, radio)
            return_details: Si retornar detalles de colisión
            
        Returns:
            True si hay colisión o (bool, detalles)
        """
        trajectory = validate_array(
            trajectory,
            shape=(None, 3),
            dtype=np.float64,
            name="trajectory"
        )
        
        obstacles = validate_array(
            obstacles,
            shape=(None, 4),
            dtype=np.float64,
            name="obstacles"
        )
        
        if CPP_AVAILABLE and FastCollisionDetector is not None:
            try:
                with performance_timer("Native collision detection"):
                    has_collision = FastCollisionDetector.trajectory_collision(
                        trajectory, obstacles
                    )
                    
                    if return_details:
                        details = NativeCollisionDetectorWrapper._compute_collision_details(
                            trajectory, obstacles
                        )
                        return has_collision, details
                    
                    return has_collision
                    
            except Exception as e:
                logger.warning(f"Native collision check failed: {e}. Using fallback.")
                return NativeCollisionDetectorWrapper._python_check(
                    trajectory, obstacles, return_details
                )
        else:
            return NativeCollisionDetectorWrapper._python_check(
                trajectory, obstacles, return_details
            )
    
    @staticmethod
    def _python_check(
        trajectory: np.ndarray,
        obstacles: np.ndarray,
        return_details: bool = False
    ) -> Union[bool, Tuple[bool, Dict[str, Any]]]:
        """Implementación Python pura."""
        collisions = []
        
        for i, point in enumerate(trajectory):
            for j, obs in enumerate(obstacles):
                center = obs[:3]
                radius = obs[3]
                distance = np.linalg.norm(point - center)
                
                if distance < radius:
                    collisions.append({
                        "trajectory_point": i,
                        "obstacle": j,
                        "distance": float(distance),
                        "point": point.tolist(),
                        "obstacle_center": center.tolist(),
                        "obstacle_radius": float(radius)
                    })
        
        has_collision = len(collisions) > 0
        
        if return_details:
            details = {
                "has_collision": has_collision,
                "num_collisions": len(collisions),
                "collisions": collisions
            }
            return has_collision, details
        
        return has_collision
    
    @staticmethod
    def _compute_collision_details(
        trajectory: np.ndarray,
        obstacles: np.ndarray
    ) -> Dict[str, Any]:
        """Calcular detalles de colisiones."""
        min_distances = []
        
        for point in trajectory:
            min_dist = float('inf')
            for obs in obstacles:
                center = obs[:3]
                radius = obs[3]
                distance = np.linalg.norm(point - center) - radius
                min_dist = min(min_dist, distance)
            min_distances.append(min_dist)
        
        return {
            "min_distance": float(np.min(min_distances)),
            "mean_distance": float(np.mean(min_distances)),
            "closest_point": int(np.argmin(min_distances)),
            "safety_margin": float(np.min(min_distances))
        }


# ============================================================================
# Wrapper para Transformaciones 3D C++
# ============================================================================

class NativeTransform3DWrapper:
    """
    Wrapper para transformaciones 3D nativas.
    """
    
    @staticmethod
    def rotation_x(angle: float) -> np.ndarray:
        """Matriz de rotación alrededor del eje X."""
        if CPP_AVAILABLE:
            try:
                from .cpp_extensions import FastTransform3D
                return FastTransform3D.rotation_x(angle)
            except Exception as e:
                logger.warning(f"Native rotation_x failed: {e}. Using scipy.")
                from scipy.spatial.transform import Rotation
                r = Rotation.from_euler('x', angle)
                return r.as_matrix()
        else:
            from scipy.spatial.transform import Rotation
            r = Rotation.from_euler('x', angle)
            return r.as_matrix()
    
    @staticmethod
    def rotation_y(angle: float) -> np.ndarray:
        """Matriz de rotación alrededor del eje Y."""
        if CPP_AVAILABLE:
            try:
                from .cpp_extensions import FastTransform3D
                return FastTransform3D.rotation_y(angle)
            except Exception as e:
                logger.warning(f"Native rotation_y failed: {e}. Using scipy.")
                from scipy.spatial.transform import Rotation
                r = Rotation.from_euler('y', angle)
                return r.as_matrix()
        else:
            from scipy.spatial.transform import Rotation
            r = Rotation.from_euler('y', angle)
            return r.as_matrix()
    
    @staticmethod
    def rotation_z(angle: float) -> np.ndarray:
        """Matriz de rotación alrededor del eje Z."""
        if CPP_AVAILABLE:
            try:
                from .cpp_extensions import FastTransform3D
                return FastTransform3D.rotation_z(angle)
            except Exception as e:
                logger.warning(f"Native rotation_z failed: {e}. Using scipy.")
                from scipy.spatial.transform import Rotation
                r = Rotation.from_euler('z', angle)
                return r.as_matrix()
        else:
            from scipy.spatial.transform import Rotation
            r = Rotation.from_euler('z', angle)
            return r.as_matrix()
    
    @staticmethod
    def rotate_point(rotation_matrix: np.ndarray, point: np.ndarray) -> np.ndarray:
        """Aplicar rotación a un punto."""
        if CPP_AVAILABLE:
            try:
                from .cpp_extensions import FastTransform3D
                return FastTransform3D.rotate_point(rotation_matrix, point)
            except Exception as e:
                logger.warning(f"Native rotate_point failed: {e}. Using numpy.")
                return np.dot(rotation_matrix, point)
        else:
            return np.dot(rotation_matrix, point)


# ============================================================================
# Wrapper para Operaciones Vectoriales C++
# ============================================================================

class NativeVectorOpsWrapper:
    """
    Wrapper para operaciones vectoriales nativas.
    """
    
    @staticmethod
    def normalize(vec: np.ndarray) -> np.ndarray:
        """Normalizar vector."""
        if CPP_AVAILABLE:
            try:
                from .cpp_extensions import FastVectorOps
                return FastVectorOps.normalize(vec)
            except Exception as e:
                logger.warning(f"Native normalize failed: {e}. Using numpy.")
                norm = np.linalg.norm(vec)
                if norm < 1e-10:
                    raise ValueError("Cannot normalize zero vector")
                return vec / norm
        else:
            norm = np.linalg.norm(vec)
            if norm < 1e-10:
                raise ValueError("Cannot normalize zero vector")
            return vec / norm
    
    @staticmethod
    def distance(a: np.ndarray, b: np.ndarray) -> float:
        """Distancia euclidiana."""
        if CPP_AVAILABLE:
            try:
                from .cpp_extensions import FastVectorOps
                return FastVectorOps.distance(a, b)
            except Exception as e:
                logger.warning(f"Native distance failed: {e}. Using numpy.")
                return float(np.linalg.norm(a - b))
        else:
            return float(np.linalg.norm(a - b))
    
    @staticmethod
    def dot(a: np.ndarray, b: np.ndarray) -> float:
        """Producto punto."""
        if CPP_AVAILABLE:
            try:
                from .cpp_extensions import FastVectorOps
                return FastVectorOps.dot(a, b)
            except Exception as e:
                logger.warning(f"Native dot failed: {e}. Using numpy.")
                return float(np.dot(a, b))
        else:
            return float(np.dot(a, b))
    
    @staticmethod
    def cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Producto cruz 3D."""
        if CPP_AVAILABLE:
            try:
                from .cpp_extensions import FastVectorOps
                return FastVectorOps.cross(a, b)
            except Exception as e:
                logger.warning(f"Native cross failed: {e}. Using numpy.")
                return np.cross(a, b)
        else:
            return np.cross(a, b)


# ============================================================================
# Wrapper para Interpolación C++
# ============================================================================

class NativeInterpolationWrapper:
    """
    Wrapper para interpolación nativa.
    """
    
    @staticmethod
    def linear(points: np.ndarray, num_output: int) -> np.ndarray:
        """Interpolación lineal."""
        if CPP_AVAILABLE:
            try:
                from .cpp_extensions import FastInterpolation
                return FastInterpolation.linear(points, num_output)
            except Exception as e:
                logger.warning(f"Native interpolation failed: {e}. Using scipy.")
                from scipy.interpolate import interp1d
                t_old = np.linspace(0, 1, len(points))
                t_new = np.linspace(0, 1, num_output)
                f = interp1d(t_old, points, axis=0, kind='linear')
                return f(t_new)
        else:
            from scipy.interpolate import interp1d
            t_old = np.linspace(0, 1, len(points))
            t_new = np.linspace(0, 1, num_output)
            f = interp1d(t_old, points, axis=0, kind='linear')
            return f(t_new)


# ============================================================================
# Wrapper para Utilidades Matemáticas C++
# ============================================================================

class NativeMathUtilsWrapper:
    """
    Wrapper para utilidades matemáticas nativas.
    """
    
    # Mantener compatibilidad con código existente
    @staticmethod
    def linear_interpolate(points: np.ndarray, num_output_points: int) -> np.ndarray:
        """Interpolación lineal (compatibilidad)."""
        return NativeInterpolationWrapper.linear(points, num_output_points)
    
    @staticmethod
    def normalize_vector(vec: np.ndarray) -> np.ndarray:
        """Normalizar vector (compatibilidad)."""
        return NativeVectorOpsWrapper.normalize(vec)
    
    @staticmethod
    def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
        """Distancia euclidiana (compatibilidad)."""
        return NativeVectorOpsWrapper.distance(a, b)
    
    @staticmethod
    def cross_product(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Producto cruz (compatibilidad)."""
        return NativeVectorOpsWrapper.cross(a, b)
    
    @staticmethod
    def dot_product(a: np.ndarray, b: np.ndarray) -> float:
        """Producto punto (compatibilidad)."""
        return NativeVectorOpsWrapper.dot(a, b)


# ============================================================================
# Funciones de Utilidad Rust (con fallback profesional)
# ============================================================================

def json_parse(json_str: str, fallback_on_error: bool = True):
    """
    Parse JSON usando extensión Rust si está disponible.
    
    Args:
        json_str: String JSON a parsear
        fallback_on_error: Si usar fallback Python en caso de error
        
    Returns:
        Objeto Python parseado
    """
    if RUST_AVAILABLE:
        try:
            with performance_timer("Rust JSON parse"):
                return fast_json_parse(json_str)
        except Exception as e:
            if fallback_on_error:
                logger.warning(f"Rust JSON parse failed: {e}. Using Python json.")
                import json
                return json.loads(json_str)
            raise
    else:
        import json
        return json.loads(json_str)


def string_search(text: str, pattern: str) -> List[int]:
    """
    Búsqueda de string usando extensión Rust si está disponible.
    
    Args:
        text: Texto en el que buscar
        pattern: Patrón a buscar
        
    Returns:
        Lista de posiciones donde se encuentra el patrón
    """
    if RUST_AVAILABLE:
        try:
            with performance_timer("Rust string search"):
                return fast_string_search(text, pattern)
        except Exception as e:
            logger.warning(f"Rust string search failed: {e}. Using Python.")
            positions = []
            start = 0
            while True:
                pos = text.find(pattern, start)
                if pos == -1:
                    break
                positions.append(pos)
                start = pos + 1
            return positions
    else:
        positions = []
        start = 0
        while True:
            pos = text.find(pattern, start)
            if pos == -1:
                break
            positions.append(pos)
            start = pos + 1
        return positions


def hash_data(data: str) -> int:
    """
    Hash rápido usando extensión Rust si está disponible.
    
    Args:
        data: Datos a hashear
        
    Returns:
        Hash entero
    """
    if RUST_AVAILABLE:
        try:
            return fast_hash(data)
        except Exception as e:
            logger.warning(f"Rust hash failed: {e}. Using Python hash.")
            return hash(data)
    else:
        return hash(data)


def array_mean(numbers: List[float]) -> float:
    """Media de array usando Rust si está disponible."""
    if RUST_AVAILABLE:
        try:
            from .rust_extensions import fast_array_mean
            return fast_array_mean(numbers)
        except Exception as e:
            logger.warning(f"Rust mean failed: {e}. Using numpy.")
            return float(np.mean(numbers))
    else:
        return float(np.mean(numbers))


def array_std(numbers: List[float]) -> float:
    """Desviación estándar usando Rust si está disponible."""
    if RUST_AVAILABLE:
        try:
            from .rust_extensions import fast_array_std
            return fast_array_std(numbers)
        except Exception as e:
            logger.warning(f"Rust std failed: {e}. Using numpy.")
            return float(np.std(numbers))
    else:
        return float(np.std(numbers))


def binary_search(numbers: List[float], target: float) -> Optional[int]:
    """Búsqueda binaria usando Rust si está disponible."""
    if RUST_AVAILABLE:
        try:
            from .rust_extensions import fast_binary_search
            result = fast_binary_search(numbers, target)
            return result
        except Exception as e:
            logger.warning(f"Rust binary search failed: {e}. Using bisect.")
            import bisect
            pos = bisect.bisect_left(numbers, target)
            if pos < len(numbers) and numbers[pos] == target:
                return pos
            return None
    else:
        import bisect
        pos = bisect.bisect_left(numbers, target)
        if pos < len(numbers) and numbers[pos] == target:
            return pos
        return None


def string_count(text: str, pattern: str) -> int:
    """Contar ocurrencias de patrón en string."""
    if RUST_AVAILABLE:
        try:
            from .rust_extensions import fast_string_count
            return fast_string_count(text, pattern)
        except Exception as e:
            logger.warning(f"Rust string count failed: {e}. Using Python.")
            return text.count(pattern)
    else:
        return text.count(pattern)


def string_replace(text: str, pattern: str, replacement: str) -> str:
    """Reemplazar todas las ocurrencias en string."""
    if RUST_AVAILABLE:
        try:
            from .rust_extensions import fast_string_replace
            return fast_string_replace(text, pattern, replacement)
        except Exception as e:
            logger.warning(f"Rust string replace failed: {e}. Using Python.")
            return text.replace(pattern, replacement)
    else:
        return text.replace(pattern, replacement)


def json_validate(json_str: str) -> bool:
    """Validar formato JSON."""
    if RUST_AVAILABLE:
        try:
            from .rust_extensions import fast_json_validate
            return fast_json_validate(json_str)
        except Exception as e:
            logger.warning(f"Rust JSON validate failed: {e}. Using Python.")
            import json
            try:
                json.loads(json_str)
                return True
            except:
                return False
    else:
        import json
        try:
            json.loads(json_str)
            return True
        except:
            return False


def array_median(numbers: List[float]) -> float:
    """Mediana de array usando Rust."""
    if RUST_AVAILABLE:
        try:
            from .rust_extensions import fast_array_median
            return fast_array_median(numbers)
        except Exception as e:
            logger.warning(f"Rust median failed: {e}. Using numpy.")
            return float(np.median(numbers))
    else:
        return float(np.median(numbers))


def array_percentile(numbers: List[float], percentile: float) -> float:
    """Percentil de array usando Rust."""
    if RUST_AVAILABLE:
        try:
            from .rust_extensions import fast_array_percentile
            return fast_array_percentile(numbers, percentile)
        except Exception as e:
            logger.warning(f"Rust percentile failed: {e}. Using numpy.")
            return float(np.percentile(numbers, percentile))
    else:
        return float(np.percentile(numbers, percentile))


def array_filter(numbers: List[float], threshold: float, op: str = "gt") -> List[float]:
    """Filtrar array por condición."""
    if RUST_AVAILABLE:
        try:
            from .rust_extensions import fast_array_filter
            return fast_array_filter(numbers, threshold, op)
        except Exception as e:
            logger.warning(f"Rust filter failed: {e}. Using Python.")
            if op == "gt":
                return [x for x in numbers if x > threshold]
            elif op == "gte":
                return [x for x in numbers if x >= threshold]
            elif op == "lt":
                return [x for x in numbers if x < threshold]
            elif op == "lte":
                return [x for x in numbers if x <= threshold]
            elif op == "eq":
                return [x for x in numbers if abs(x - threshold) < 1e-10]
            else:
                raise ValueError(f"Unknown operation: {op}")
    else:
        if op == "gt":
            return [x for x in numbers if x > threshold]
        elif op == "gte":
            return [x for x in numbers if x >= threshold]
        elif op == "lt":
            return [x for x in numbers if x < threshold]
        elif op == "lte":
            return [x for x in numbers if x <= threshold]
        elif op == "eq":
            return [x for x in numbers if abs(x - threshold) < 1e-10]
        else:
            raise ValueError(f"Unknown operation: {op}")


def string_split(text: str, delimiter: str) -> List[str]:
    """Split string por delimitador."""
    if RUST_AVAILABLE:
        try:
            from .rust_extensions import fast_string_split
            return fast_string_split(text, delimiter)
        except Exception as e:
            logger.warning(f"Rust split failed: {e}. Using Python.")
            return text.split(delimiter)
    else:
        return text.split(delimiter)


def string_join(strings: List[str], separator: str = " ") -> str:
    """Join strings."""
    if RUST_AVAILABLE:
        try:
            from .rust_extensions import fast_string_join
            return fast_string_join(strings, separator)
        except Exception as e:
            logger.warning(f"Rust join failed: {e}. Using Python.")
            return separator.join(strings)
    else:
        return separator.join(strings)


def string_trim(text: str) -> str:
    """Trim whitespace."""
    if RUST_AVAILABLE:
        try:
            from .rust_extensions import fast_string_trim
            return fast_string_trim(text)
        except Exception as e:
            logger.warning(f"Rust trim failed: {e}. Using Python.")
            return text.strip()
    else:
        return text.strip()


def string_upper(text: str) -> str:
    """Convertir a mayúsculas."""
    if RUST_AVAILABLE:
        try:
            from .rust_extensions import fast_string_upper
            return fast_string_upper(text)
        except Exception as e:
            logger.warning(f"Rust upper failed: {e}. Using Python.")
            return text.upper()
    else:
        return text.upper()


def string_lower(text: str) -> str:
    """Convertir a minúsculas."""
    if RUST_AVAILABLE:
        try:
            from .rust_extensions import fast_string_lower
            return fast_string_lower(text)
        except Exception as e:
            logger.warning(f"Rust lower failed: {e}. Using Python.")
            return text.lower()
    else:
        return text.lower()


def string_find_all(text: str, pattern: str) -> List[int]:
    """Encontrar todas las ocurrencias de un patrón."""
    if RUST_AVAILABLE:
        try:
            from .rust_extensions import fast_string_find_all
            return fast_string_find_all(text, pattern)
        except Exception as e:
            logger.warning(f"Rust find_all failed: {e}. Using Python.")
            positions = []
            start = 0
            while True:
                pos = text.find(pattern, start)
                if pos == -1:
                    break
                positions.append(pos)
                start = pos + 1
            return positions
    else:
        positions = []
        start = 0
        while True:
            pos = text.find(pattern, start)
            if pos == -1:
                break
            positions.append(pos)
            start = pos + 1
        return positions


def string_starts_with(text: str, pattern: str) -> bool:
    """Verificar si string empieza con patrón."""
    if RUST_AVAILABLE:
        try:
            from .rust_extensions import fast_string_starts_with
            return fast_string_starts_with(text, pattern)
        except Exception as e:
            logger.warning(f"Rust starts_with failed: {e}. Using Python.")
            return text.startswith(pattern)
    else:
        return text.startswith(pattern)


def string_ends_with(text: str, pattern: str) -> bool:
    """Verificar si string termina con patrón."""
    if RUST_AVAILABLE:
        try:
            from .rust_extensions import fast_string_ends_with
            return fast_string_ends_with(text, pattern)
        except Exception as e:
            logger.warning(f"Rust ends_with failed: {e}. Using Python.")
            return text.endswith(pattern)
    else:
        return text.endswith(pattern)


def array_variance(numbers: List[float]) -> float:
    """Varianza de array."""
    if RUST_AVAILABLE:
        try:
            from .rust_extensions import fast_array_variance
            return fast_array_variance(numbers)
        except Exception as e:
            logger.warning(f"Rust variance failed: {e}. Using numpy.")
            return float(np.var(numbers))
    else:
        return float(np.var(numbers))


def array_range(numbers: List[float]) -> float:
    """Rango (max - min) de array."""
    if RUST_AVAILABLE:
        try:
            from .rust_extensions import fast_array_range
            return fast_array_range(numbers)
        except Exception as e:
            logger.warning(f"Rust range failed: {e}. Using Python.")
            return max(numbers) - min(numbers)
    else:
        return max(numbers) - min(numbers)


def array_cumsum(numbers: List[float]) -> List[float]:
    """Suma acumulativa."""
    if RUST_AVAILABLE:
        try:
            from .rust_extensions import fast_array_cumsum
            return fast_array_cumsum(numbers)
        except Exception as e:
            logger.warning(f"Rust cumsum failed: {e}. Using numpy.")
            return np.cumsum(numbers).tolist()
    else:
        return np.cumsum(numbers).tolist()


def array_cumprod(numbers: List[float]) -> List[float]:
    """Producto acumulativo."""
    if RUST_AVAILABLE:
        try:
            from .rust_extensions import fast_array_cumprod
            return fast_array_cumprod(numbers)
        except Exception as e:
            logger.warning(f"Rust cumprod failed: {e}. Using numpy.")
            return np.cumprod(numbers).tolist()
    else:
        return np.cumprod(numbers).tolist()


# ============================================================================
# Wrapper para Quaternions C++
# ============================================================================

class NativeQuaternionWrapper:
    """
    Wrapper para operaciones con quaternions nativas.
    """
    
    @staticmethod
    def from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
        """Crear quaternion desde eje y ángulo."""
        if CPP_AVAILABLE:
            try:
                from .cpp_extensions import FastQuaternion
                return FastQuaternion.from_axis_angle(axis, angle)
            except Exception as e:
                logger.warning(f"Native quaternion from_axis_angle failed: {e}. Using scipy.")
                from scipy.spatial.transform import Rotation
                r = Rotation.from_rotvec(axis * angle)
                return np.array([r.as_quat()[3], r.as_quat()[0], r.as_quat()[1], r.as_quat()[2]])
        else:
            from scipy.spatial.transform import Rotation
            r = Rotation.from_rotvec(axis * angle)
            return np.array([r.as_quat()[3], r.as_quat()[0], r.as_quat()[1], r.as_quat()[2]])
    
    @staticmethod
    def multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """Multiplicar quaternions."""
        if CPP_AVAILABLE:
            try:
                from .cpp_extensions import FastQuaternion
                return FastQuaternion.multiply(q1, q2)
            except Exception as e:
                logger.warning(f"Native quaternion multiply failed: {e}. Using scipy.")
                from scipy.spatial.transform import Rotation
                r1 = Rotation.from_quat([q1[1], q1[2], q1[3], q1[0]])
                r2 = Rotation.from_quat([q2[1], q2[2], q2[3], q2[0]])
                r = r1 * r2
                q = r.as_quat()
                return np.array([q[3], q[0], q[1], q[2]])
        else:
            from scipy.spatial.transform import Rotation
            r1 = Rotation.from_quat([q1[1], q1[2], q1[3], q1[0]])
            r2 = Rotation.from_quat([q2[1], q2[2], q2[3], q2[0]])
            r = r1 * r2
            q = r.as_quat()
            return np.array([q[3], q[0], q[1], q[2]])
    
    @staticmethod
    def to_rotation_matrix(q: np.ndarray) -> np.ndarray:
        """Convertir quaternion a matriz de rotación."""
        if CPP_AVAILABLE:
            try:
                from .cpp_extensions import FastQuaternion
                return FastQuaternion.to_rotation_matrix(q)
            except Exception as e:
                logger.warning(f"Native quaternion to_rotation_matrix failed: {e}. Using scipy.")
                from scipy.spatial.transform import Rotation
                r = Rotation.from_quat([q[1], q[2], q[3], q[0]])
                return r.as_matrix()
        else:
            from scipy.spatial.transform import Rotation
            r = Rotation.from_quat([q[1], q[2], q[3], q[0]])
            return r.as_matrix()
    
    @staticmethod
    def normalize(q: np.ndarray) -> np.ndarray:
        """Normalizar quaternion."""
        if CPP_AVAILABLE:
            try:
                from .cpp_extensions import FastQuaternion
                return FastQuaternion.normalize(q)
            except Exception as e:
                logger.warning(f"Native quaternion normalize failed: {e}. Using numpy.")
                norm = np.linalg.norm(q)
                if norm < 1e-10:
                    raise ValueError("Cannot normalize zero quaternion")
                return q / norm
        else:
            norm = np.linalg.norm(q)
            if norm < 1e-10:
                raise ValueError("Cannot normalize zero quaternion")
            return q / norm


# ============================================================================
# Wrapper para Transformaciones Homogéneas C++
# ============================================================================

class NativeHomogeneousTransformWrapper:
    """
    Wrapper para transformaciones homogéneas nativas.
    """
    
    @staticmethod
    def from_rotation_translation(
        rotation: np.ndarray,
        translation: np.ndarray
    ) -> np.ndarray:
        """Crear transformación homogénea desde rotación y traslación."""
        if CPP_AVAILABLE:
            try:
                from .cpp_extensions import FastHomogeneousTransform
                return FastHomogeneousTransform.from_rotation_translation(
                    rotation, translation
                )
            except Exception as e:
                logger.warning(f"Native homogeneous transform failed: {e}. Using numpy.")
                transform = np.eye(4)
                transform[:3, :3] = rotation
                transform[:3, 3] = translation
                return transform
        else:
            transform = np.eye(4)
            transform[:3, :3] = rotation
            transform[:3, 3] = translation
            return transform
    
    @staticmethod
    def transform_point(transform: np.ndarray, point: np.ndarray) -> np.ndarray:
        """Aplicar transformación homogénea a punto."""
        if CPP_AVAILABLE:
            try:
                from .cpp_extensions import FastHomogeneousTransform
                return FastHomogeneousTransform.transform_point(transform, point)
            except Exception as e:
                logger.warning(f"Native transform_point failed: {e}. Using numpy.")
                point_hom = np.append(point, 1.0)
                result_hom = transform @ point_hom
                return result_hom[:3]
        else:
            point_hom = np.append(point, 1.0)
            result_hom = transform @ point_hom
            return result_hom[:3]
    
    @staticmethod
    def inverse(transform: np.ndarray) -> np.ndarray:
        """Inversa de transformación homogénea."""
        if CPP_AVAILABLE:
            try:
                from .cpp_extensions import FastHomogeneousTransform
                return FastHomogeneousTransform.inverse(transform)
            except Exception as e:
                logger.warning(f"Native homogeneous inverse failed: {e}. Using numpy.")
                return np.linalg.inv(transform)
        else:
            return np.linalg.inv(transform)


# ============================================================================
# Wrapper para Geometría C++
# ============================================================================

class NativeGeometryWrapper:
    """
    Wrapper para operaciones geométricas nativas.
    """
    
    @staticmethod
    def point_to_line_distance(
        point: np.ndarray,
        line_start: np.ndarray,
        line_end: np.ndarray
    ) -> float:
        """Distancia de punto a segmento de línea."""
        if CPP_AVAILABLE:
            try:
                from .cpp_extensions import FastGeometry
                return FastGeometry.point_to_line_distance(point, line_start, line_end)
            except Exception as e:
                logger.warning(f"Native point_to_line_distance failed: {e}. Using numpy.")
                # Implementación Python
                ab = line_end - line_start
                ap = point - line_start
                ab_norm_sq = np.dot(ab, ab)
                if ab_norm_sq < 1e-10:
                    return float(np.linalg.norm(point - line_start))
                t = np.dot(ap, ab) / ab_norm_sq
                t = max(0.0, min(1.0, t))
                closest = line_start + t * ab
                return float(np.linalg.norm(point - closest))
        else:
            ab = line_end - line_start
            ap = point - line_start
            ab_norm_sq = np.dot(ab, ab)
            if ab_norm_sq < 1e-10:
                return float(np.linalg.norm(point - line_start))
            t = np.dot(ap, ab) / ab_norm_sq
            t = max(0.0, min(1.0, t))
            closest = line_start + t * ab
            return float(np.linalg.norm(point - closest))
    
    @staticmethod
    def triangle_area(
        a: np.ndarray,
        b: np.ndarray,
        c: np.ndarray
    ) -> float:
        """Área de triángulo."""
        if CPP_AVAILABLE:
            try:
                from .cpp_extensions import FastGeometry
                return FastGeometry.triangle_area(a, b, c)
            except Exception as e:
                logger.warning(f"Native triangle_area failed: {e}. Using numpy.")
                ab = b - a
                ac = c - a
                cross = np.cross(ab, ac)
                return float(0.5 * np.linalg.norm(cross))
        else:
            ab = b - a
            ac = c - a
            cross = np.cross(ab, ac)
            return float(0.5 * np.linalg.norm(cross))


# ============================================================================
# Información del Sistema
# ============================================================================

def get_native_extensions_status() -> Dict[str, Any]:
    """
    Obtener estado de extensiones nativas.
    
    Returns:
        Diccionario con información de extensiones disponibles
    """
    return {
        "cpp_available": CPP_AVAILABLE,
        "rust_available": RUST_AVAILABLE,
        "extensions": {
            "cpp": {
                "available": CPP_AVAILABLE,
                "modules": [
                    "FastIK",
                    "FastTrajectoryOptimizer",
                    "FastMatrixOps",
                    "FastCollisionDetector"
                ] if CPP_AVAILABLE else []
            },
            "rust": {
                "available": RUST_AVAILABLE,
                "functions": [
                    "fast_json_parse",
                    "fast_string_search",
                    "fast_hash"
                ] if RUST_AVAILABLE else []
            }
        },
        "recommendations": {
            "install_cpp": not CPP_AVAILABLE,
            "install_rust": not RUST_AVAILABLE
        }
    }


# ============================================================================
# Exportar Funciones Principales
# ============================================================================

__all__ = [
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
    'CPP_AVAILABLE',
    'RUST_AVAILABLE',
    'validate_array',
    'handle_native_errors',
    'performance_timer',
]
