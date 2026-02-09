"""
Math Utilities - Utilidades matemáticas y geométricas
======================================================

Utilidades para cálculos matemáticos y geométricos útiles para robots.
"""

import math
from typing import List, Tuple, Optional, Union
import numpy as np


def euclidean_distance(p1: Union[List[float], Tuple[float, ...], np.ndarray],
                       p2: Union[List[float], Tuple[float, ...], np.ndarray]) -> float:
    """
    Calcular distancia euclidiana entre dos puntos.
    
    Args:
        p1: Primer punto (x, y) o (x, y, z)
        p2: Segundo punto (x, y) o (x, y, z)
    
    Returns:
        Distancia euclidiana
    """
    p1 = np.array(p1)
    p2 = np.array(p2)
    return float(np.linalg.norm(p1 - p2))


def manhattan_distance(p1: Union[List[float], Tuple[float, ...], np.ndarray],
                       p2: Union[List[float], Tuple[float, ...], np.ndarray]) -> float:
    """
    Calcular distancia Manhattan entre dos puntos.
    
    Args:
        p1: Primer punto
        p2: Segundo punto
    
    Returns:
        Distancia Manhattan
    """
    p1 = np.array(p1)
    p2 = np.array(p2)
    return float(np.sum(np.abs(p1 - p2)))


def angle_between_vectors(v1: Union[List[float], np.ndarray],
                          v2: Union[List[float], np.ndarray]) -> float:
    """
    Calcular ángulo entre dos vectores en radianes.
    
    Args:
        v1: Primer vector
        v2: Segundo vector
    
    Returns:
        Ángulo en radianes
    """
    v1 = np.array(v1)
    v2 = np.array(v2)
    
    dot_product = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    cos_angle = np.clip(dot_product / (norm1 * norm2), -1.0, 1.0)
    return float(np.arccos(cos_angle))


def normalize_vector(v: Union[List[float], np.ndarray]) -> np.ndarray:
    """
    Normalizar vector a longitud unitaria.
    
    Args:
        v: Vector
    
    Returns:
        Vector normalizado
    """
    v = np.array(v)
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def dot_product(v1: Union[List[float], np.ndarray],
                v2: Union[List[float], np.ndarray]) -> float:
    """
    Calcular producto punto entre dos vectores.
    
    Args:
        v1: Primer vector
        v2: Segundo vector
    
    Returns:
        Producto punto
    """
    v1 = np.array(v1)
    v2 = np.array(v2)
    return float(np.dot(v1, v2))


def cross_product(v1: Union[List[float], np.ndarray],
                  v2: Union[List[float], np.ndarray]) -> np.ndarray:
    """
    Calcular producto cruz entre dos vectores 3D.
    
    Args:
        v1: Primer vector 3D
        v2: Segundo vector 3D
    
    Returns:
        Producto cruz
    """
    v1 = np.array(v1)
    v2 = np.array(v2)
    return np.cross(v1, v2)


def radians_to_degrees(radians: float) -> float:
    """
    Convertir radianes a grados.
    
    Args:
        radians: Ángulo en radianes
    
    Returns:
        Ángulo en grados
    """
    return math.degrees(radians)


def degrees_to_radians(degrees: float) -> float:
    """
    Convertir grados a radianes.
    
    Args:
        degrees: Ángulo en grados
    
    Returns:
        Ángulo en radianes
    """
    return math.radians(degrees)


def normalize_angle(angle: float, use_degrees: bool = False) -> float:
    """
    Normalizar ángulo al rango [0, 2π) o [0, 360°).
    
    Args:
        angle: Ángulo
        use_degrees: Si True, usar grados
    
    Returns:
        Ángulo normalizado
    """
    if use_degrees:
        max_angle = 360.0
    else:
        max_angle = 2 * math.pi
    
    normalized = angle % max_angle
    if normalized < 0:
        normalized += max_angle
    return normalized


def quaternion_multiply(q1: Union[List[float], np.ndarray],
                        q2: Union[List[float], np.ndarray]) -> np.ndarray:
    """
    Multiplicar dos quaternions.
    
    Args:
        q1: Primer quaternion [x, y, z, w]
        q2: Segundo quaternion [x, y, z, w]
    
    Returns:
        Quaternion resultante
    """
    q1 = np.array(q1)
    q2 = np.array(q2)
    
    w1, x1, y1, z1 = q1[3], q1[0], q1[1], q1[2]
    w2, x2, y2, z2 = q2[3], q2[0], q2[1], q2[2]
    
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    
    return np.array([x, y, z, w])


def quaternion_conjugate(q: Union[List[float], np.ndarray]) -> np.ndarray:
    """
    Calcular conjugado de quaternion.
    
    Args:
        q: Quaternion [x, y, z, w]
    
    Returns:
        Quaternion conjugado
    """
    q = np.array(q)
    return np.array([-q[0], -q[1], -q[2], q[3]])


def quaternion_normalize(q: Union[List[float], np.ndarray]) -> np.ndarray:
    """
    Normalizar quaternion.
    
    Args:
        q: Quaternion [x, y, z, w]
    
    Returns:
        Quaternion normalizado
    """
    q = np.array(q)
    norm = np.linalg.norm(q)
    if norm == 0:
        return np.array([0.0, 0.0, 0.0, 1.0])
    return q / norm


def quaternion_to_euler(q: Union[List[float], np.ndarray]) -> Tuple[float, float, float]:
    """
    Convertir quaternion a ángulos de Euler (roll, pitch, yaw).
    
    Args:
        q: Quaternion [x, y, z, w]
    
    Returns:
        Tupla (roll, pitch, yaw) en radianes
    """
    q = np.array(q)
    x, y, z, w = q[0], q[1], q[2], q[3]
    
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(2 * (w * y - z * x))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    
    return (roll, pitch, yaw)


def euler_to_quaternion(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """
    Convertir ángulos de Euler a quaternion.
    
    Args:
        roll: Roll en radianes
        pitch: Pitch en radianes
        yaw: Yaw en radianes
    
    Returns:
        Quaternion [x, y, z, w]
    """
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    w = cr * cp * cy + sr * sp * sy
    
    return np.array([x, y, z, w])


def rotate_vector_by_quaternion(v: Union[List[float], np.ndarray],
                                 q: Union[List[float], np.ndarray]) -> np.ndarray:
    """
    Rotar vector por quaternion.
    
    Args:
        v: Vector 3D
        q: Quaternion [x, y, z, w]
    
    Returns:
        Vector rotado
    """
    v = np.array(v)
    q = quaternion_normalize(q)
    
    q_conj = quaternion_conjugate(q)
    v_quat = np.array([v[0], v[1], v[2], 0.0])
    
    result = quaternion_multiply(quaternion_multiply(q, v_quat), q_conj)
    return result[:3]


def point_in_circle(point: Union[List[float], Tuple[float, ...]],
                    center: Union[List[float], Tuple[float, ...]],
                    radius: float) -> bool:
    """
    Verificar si punto está dentro de círculo.
    
    Args:
        point: Punto (x, y)
        center: Centro del círculo (x, y)
        radius: Radio del círculo
    
    Returns:
        True si el punto está dentro
    """
    return euclidean_distance(point[:2], center[:2]) <= radius


def point_in_sphere(point: Union[List[float], Tuple[float, ...]],
                    center: Union[List[float], Tuple[float, ...]],
                    radius: float) -> bool:
    """
    Verificar si punto está dentro de esfera.
    
    Args:
        point: Punto (x, y, z)
        center: Centro de la esfera (x, y, z)
        radius: Radio de la esfera
    
    Returns:
        True si el punto está dentro
    """
    return euclidean_distance(point, center) <= radius


def point_in_box(point: Union[List[float], Tuple[float, ...]],
                 box_min: Union[List[float], Tuple[float, ...]],
                 box_max: Union[List[float], Tuple[float, ...]]) -> bool:
    """
    Verificar si punto está dentro de caja (bounding box).
    
    Args:
        point: Punto (x, y, z)
        box_min: Esquina mínima (x, y, z)
        box_max: Esquina máxima (x, y, z)
    
    Returns:
        True si el punto está dentro
    """
    point = np.array(point)
    box_min = np.array(box_min)
    box_max = np.array(box_max)
    
    return np.all(point >= box_min) and np.all(point <= box_max)


def lerp(start: float, end: float, t: float) -> float:
    """
    Interpolación lineal entre dos valores.
    
    Args:
        start: Valor inicial
        end: Valor final
        t: Factor de interpolación [0, 1]
    
    Returns:
        Valor interpolado
    """
    t = np.clip(t, 0.0, 1.0)
    return start + (end - start) * t


def slerp(q1: Union[List[float], np.ndarray],
          q2: Union[List[float], np.ndarray],
          t: float) -> np.ndarray:
    """
    Interpolación esférica (SLERP) entre dos quaternions.
    
    Args:
        q1: Primer quaternion
        q2: Segundo quaternion
        t: Factor de interpolación [0, 1]
    
    Returns:
        Quaternion interpolado
    """
    q1 = quaternion_normalize(q1)
    q2 = quaternion_normalize(q2)
    
    dot = np.dot(q1, q2)
    
    if dot < 0.0:
        q2 = -q2
        dot = -dot
    
    if abs(dot) > 0.9995:
        result = q1 + t * (q2 - q1)
        return quaternion_normalize(result)
    
    theta_0 = math.acos(abs(dot))
    sin_theta_0 = math.sin(theta_0)
    theta = theta_0 * t
    sin_theta = math.sin(theta)
    
    s0 = math.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    
    return quaternion_normalize(s0 * q1 + s1 * q2)


def clamp(value: float, min_val: float, max_val: float) -> float:
    """
    Limitar valor a rango [min_val, max_val].
    
    Args:
        value: Valor
        min_val: Valor mínimo
        max_val: Valor máximo
    
    Returns:
        Valor limitado
    """
    return max(min_val, min(value, max_val))


def smooth_step(edge0: float, edge1: float, x: float) -> float:
    """
    Función smooth step para transiciones suaves.
    
    Args:
        edge0: Borde inferior
        edge1: Borde superior
        x: Valor de entrada
    
    Returns:
        Valor entre 0 y 1
    """
    t = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)

