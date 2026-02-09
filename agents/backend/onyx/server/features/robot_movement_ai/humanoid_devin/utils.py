"""
Utilities for Humanoid Devin Robot (Optimizado)
===============================================

Utilidades y helpers para el robot humanoide.
Incluye validaciones comunes, conversiones, y funciones auxiliares.
"""

import logging
from typing import Union, List, Tuple, Optional, Dict, Any
import numpy as np
from math import sqrt, atan2, asin

logger = logging.getLogger(__name__)


def normalize_quaternion(q: Union[np.ndarray, List[float], Tuple[float, ...]]) -> np.ndarray:
    """
    Normalizar quaternion a unidad (optimizado).
    
    Args:
        q: Quaternion [x, y, z, w] o [w, x, y, z]
        
    Returns:
        Quaternion normalizado como numpy array [x, y, z, w]
        
    Raises:
        ValueError: Si el quaternion es inválido
    """
    try:
        q_array = np.array(q, dtype=np.float64)
        
        if q_array.shape[0] != 4:
            raise ValueError(f"Quaternion must have 4 components, got {q_array.shape[0]}")
        
        if not np.all(np.isfinite(q_array)):
            raise ValueError("All quaternion components must be finite numbers")
        
        # Detectar orden: [w, x, y, z] o [x, y, z, w]
        norm = np.linalg.norm(q_array)
        if norm < 1e-10:
            raise ValueError("Quaternion norm is too small (near zero)")
        
        q_normalized = q_array / norm
        
        # Convertir a formato [x, y, z, w] si está en [w, x, y, z]
        if abs(q_normalized[0]) > 0.9:  # Probablemente w está primero
            q_normalized = np.array([q_normalized[1], q_normalized[2], q_normalized[3], q_normalized[0]])
        
        return q_normalized
    except (ValueError, TypeError) as e:
        raise ValueError(f"Invalid quaternion: {e}") from e


def quaternion_to_euler(q: Union[np.ndarray, List[float]]) -> Tuple[float, float, float]:
    """
    Convertir quaternion a ángulos de Euler (roll, pitch, yaw) (optimizado).
    
    Args:
        q: Quaternion [x, y, z, w]
        
    Returns:
        Tupla (roll, pitch, yaw) en radianes
        
    Raises:
        ValueError: Si el quaternion es inválido
    """
    q_norm = normalize_quaternion(q)
    x, y, z, w = q_norm
    
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = atan2(sinr_cosp, cosr_cosp)
    
    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = asin(1.0) if sinp > 0 else asin(-1.0)  # Use 90 degrees if out of range
    else:
        pitch = asin(sinp)
    
    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = atan2(siny_cosp, cosy_cosp)
    
    return (roll, pitch, yaw)


def euler_to_quaternion(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """
    Convertir ángulos de Euler a quaternion (optimizado).
    
    Args:
        roll: Rotación alrededor del eje x (radianes)
        pitch: Rotación alrededor del eje y (radianes)
        yaw: Rotación alrededor del eje z (radianes)
        
    Returns:
        Quaternion [x, y, z, w] normalizado
    """
    # Validar que los ángulos sean finitos
    if not all(np.isfinite([roll, pitch, yaw])):
        raise ValueError("All Euler angles must be finite numbers")
    
    # Calcular quaternion
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    
    q = np.array([
        sr * cp * cy - cr * sp * sy,  # x
        cr * sp * cy + sr * cp * sy,  # y
        cr * cp * sy - sr * sp * cy,  # z
        cr * cp * cy + sr * sp * sy   # w
    ], dtype=np.float64)
    
    return normalize_quaternion(q)


def clamp(value: float, min_val: float, max_val: float) -> float:
    """
    Limitar valor entre mínimo y máximo (optimizado).
    
    Args:
        value: Valor a limitar
        min_val: Valor mínimo
        max_val: Valor máximo
        
    Returns:
        Valor limitado
        
    Raises:
        ValueError: Si min_val > max_val
    """
    if min_val > max_val:
        raise ValueError(f"min_val ({min_val}) must be <= max_val ({max_val})")
    
    return max(min_val, min(value, max_val))


def normalize_angle(angle: float) -> float:
    """
    Normalizar ángulo a rango [-π, π] (optimizado).
    
    Args:
        angle: Ángulo en radianes
        
    Returns:
        Ángulo normalizado en [-π, π]
    """
    if not np.isfinite(angle):
        raise ValueError("Angle must be a finite number")
    
    return ((angle + np.pi) % (2 * np.pi)) - np.pi


def validate_joint_positions(
    positions: Union[List[float], np.ndarray],
    dof: int,
    joint_limits: Optional[List[Tuple[float, float]]] = None
) -> np.ndarray:
    """
    Validar y normalizar posiciones de articulaciones (optimizado).
    
    Args:
        positions: Lista o array de posiciones
        dof: Número de grados de libertad esperados
        joint_limits: Lista opcional de límites (min, max) por articulación
        
    Returns:
        Array validado de posiciones
        
    Raises:
        ValueError: Si las posiciones son inválidas
    """
    # Validar tipo y longitud
    if not isinstance(positions, (list, tuple, np.ndarray)):
        raise ValueError(f"positions must be a list, tuple, or numpy array, got {type(positions)}")
    
    positions_array = np.array(positions, dtype=np.float64)
    
    if len(positions_array) != dof:
        raise ValueError(f"Expected {dof} joint positions, got {len(positions_array)}")
    
    if not np.all(np.isfinite(positions_array)):
        raise ValueError("All joint positions must be finite numbers")
    
    # Validar límites si se proporcionan
    if joint_limits is not None:
        if len(joint_limits) != dof:
            raise ValueError(f"joint_limits must have {dof} entries, got {len(joint_limits)}")
        
        for i, (pos, (min_val, max_val)) in enumerate(zip(positions_array, joint_limits)):
            if not isinstance(min_val, (int, float)) or not isinstance(max_val, (int, float)):
                raise ValueError(f"Joint limit {i} must be a tuple of numbers")
            
            if min_val > max_val:
                raise ValueError(f"Joint limit {i}: min ({min_val}) > max ({max_val})")
            
            if pos < min_val or pos > max_val:
                logger.warning(
                    f"Joint {i} position {pos} is outside limits [{min_val}, {max_val}], "
                    f"clamping to {clamp(pos, min_val, max_val)}"
                )
                positions_array[i] = clamp(pos, min_val, max_val)
    
    return positions_array


def interpolate_joint_positions(
    start: Union[List[float], np.ndarray],
    end: Union[List[float], np.ndarray],
    num_steps: int = 10
) -> np.ndarray:
    """
    Interpolar entre posiciones de articulaciones (optimizado).
    
    Args:
        start: Posiciones iniciales
        end: Posiciones finales
        num_steps: Número de pasos de interpolación
        
    Returns:
        Array de trayectoria (num_steps, dof)
        
    Raises:
        ValueError: Si los parámetros son inválidos
    """
    if num_steps < 2:
        raise ValueError(f"num_steps must be >= 2, got {num_steps}")
    
    start_array = np.array(start, dtype=np.float64)
    end_array = np.array(end, dtype=np.float64)
    
    if start_array.shape != end_array.shape:
        raise ValueError(
            f"start and end must have the same shape, got {start_array.shape} and {end_array.shape}"
        )
    
    if not np.all(np.isfinite(start_array)) or not np.all(np.isfinite(end_array)):
        raise ValueError("All positions must be finite numbers")
    
    # Interpolación lineal
    alphas = np.linspace(0, 1, num_steps)
    trajectory = np.array([
        start_array + alpha * (end_array - start_array)
        for alpha in alphas
    ])
    
    return trajectory


def calculate_distance(
    pos1: Union[np.ndarray, List[float], Tuple[float, ...]],
    pos2: Union[np.ndarray, List[float], Tuple[float, ...]]
) -> float:
    """
    Calcular distancia euclidiana entre dos posiciones (optimizado).
    
    Args:
        pos1: Primera posición [x, y, z]
        pos2: Segunda posición [x, y, z]
        
    Returns:
        Distancia en metros
        
    Raises:
        ValueError: Si las posiciones son inválidas
    """
    pos1_array = np.array(pos1, dtype=np.float64)
    pos2_array = np.array(pos2, dtype=np.float64)
    
    if pos1_array.shape != pos2_array.shape:
        raise ValueError(
            f"pos1 and pos2 must have the same shape, got {pos1_array.shape} and {pos2_array.shape}"
        )
    
    if not np.all(np.isfinite(pos1_array)) or not np.all(np.isfinite(pos2_array)):
        raise ValueError("All position coordinates must be finite numbers")
    
    return float(np.linalg.norm(pos2_array - pos1_array))


def smooth_trajectory(
    trajectory: np.ndarray,
    window_size: int = 5
) -> np.ndarray:
    """
    Suavizar trayectoria usando media móvil (optimizado).
    
    Args:
        trajectory: Trayectoria (num_steps, dof)
        window_size: Tamaño de ventana para suavizado (debe ser impar)
        
    Returns:
        Trayectoria suavizada
        
    Raises:
        ValueError: Si los parámetros son inválidos
    """
    if trajectory is None or trajectory.size == 0:
        raise ValueError("trajectory cannot be None or empty")
    
    if len(trajectory.shape) != 2:
        raise ValueError(f"trajectory must be 2D array, got shape {trajectory.shape}")
    
    if window_size < 1:
        raise ValueError(f"window_size must be >= 1, got {window_size}")
    
    if window_size % 2 == 0:
        window_size += 1  # Hacer impar
        logger.debug(f"Adjusted window_size to {window_size} (must be odd)")
    
    if window_size > len(trajectory):
        logger.warning(f"window_size ({window_size}) > trajectory length ({len(trajectory)}), using trajectory length")
        window_size = len(trajectory)
        if window_size % 2 == 0:
            window_size = max(1, window_size - 1)
    
    # Aplicar media móvil
    smoothed = np.zeros_like(trajectory)
    half_window = window_size // 2
    
    for i in range(len(trajectory)):
        start_idx = max(0, i - half_window)
        end_idx = min(len(trajectory), i + half_window + 1)
        smoothed[i] = np.mean(trajectory[start_idx:end_idx], axis=0)
    
    return smoothed


def validate_pose(
    position: Union[np.ndarray, List[float], Tuple[float, ...]],
    orientation: Union[np.ndarray, List[float], Tuple[float, ...]]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Validar y normalizar pose (posición + orientación) (optimizado).
    
    Args:
        position: Posición [x, y, z]
        orientation: Orientación quaternion [x, y, z, w]
        
    Returns:
        Tupla (posición validada, orientación normalizada)
        
    Raises:
        ValueError: Si la pose es inválida
    """
    # Validar posición
    pos_array = np.array(position, dtype=np.float64)
    if pos_array.shape != (3,):
        raise ValueError(f"position must have shape (3,), got {pos_array.shape}")
    
    if not np.all(np.isfinite(pos_array)):
        raise ValueError("All position coordinates must be finite numbers")
    
    # Validar y normalizar orientación
    ori_array = normalize_quaternion(orientation)
    
    return pos_array, ori_array


def get_joint_velocity(
    current_positions: Union[List[float], np.ndarray],
    previous_positions: Union[List[float], np.ndarray],
    dt: float = 0.01
) -> np.ndarray:
    """
    Calcular velocidad de articulaciones (optimizado).
    
    Args:
        current_positions: Posiciones actuales
        previous_positions: Posiciones anteriores
        dt: Intervalo de tiempo en segundos
        
    Returns:
        Velocidades de articulaciones (rad/s)
        
    Raises:
        ValueError: Si los parámetros son inválidos
    """
    if dt <= 0:
        raise ValueError(f"dt must be positive, got {dt}")
    
    current = np.array(current_positions, dtype=np.float64)
    previous = np.array(previous_positions, dtype=np.float64)
    
    if current.shape != previous.shape:
        raise ValueError(
            f"current_positions and previous_positions must have the same shape, "
            f"got {current.shape} and {previous.shape}"
        )
    
    if not np.all(np.isfinite(current)) or not np.all(np.isfinite(previous)):
        raise ValueError("All positions must be finite numbers")
    
    velocities = (current - previous) / dt
    return velocities

