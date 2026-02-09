"""
Fast Math Operations
===================

Operaciones matemáticas optimizadas con numba para máximo rendimiento.
"""

import numpy as np

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

if NUMBA_AVAILABLE:
    @njit(cache=True, fastmath=True, inline='always')
    def fast_cos_sin(theta: float) -> tuple:
        """Calcular cos y sin simultáneamente."""
        return np.cos(theta), np.sin(theta)
    
    @njit(cache=True, fastmath=True, inline='always')
    def fast_norm2(v: np.ndarray) -> float:
        """Norma al cuadrado (más rápido que norma)."""
        return np.sum(v * v)
    
    @njit(cache=True, fastmath=True, inline='always')
    def fast_norm(v: np.ndarray) -> float:
        """Norma optimizada."""
        return np.sqrt(np.sum(v * v))
    
    @njit(cache=True, fastmath=True)
    def fast_dh_transform(a: float, d: float, alpha: float, theta: float) -> np.ndarray:
        """Calcular matriz de transformación DH optimizada."""
        ct, st = np.cos(theta), np.sin(theta)
        ca, sa = np.cos(alpha), np.sin(alpha)
        
        return np.array([
            [ct, -st * ca, st * sa, a * ct],
            [st, ct * ca, -ct * sa, a * st],
            [0.0, sa, ca, d],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float64)
    
    @njit(cache=True, fastmath=True)
    def fast_matrix_multiply(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Multiplicación de matrices 4x4 optimizada."""
        C = np.zeros((4, 4), dtype=np.float64)
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    C[i, j] += A[i, k] * B[k, j]
        return C

else:
    def fast_cos_sin(theta: float) -> tuple:
        return np.cos(theta), np.sin(theta)
    
    def fast_norm2(v: np.ndarray) -> float:
        return np.sum(v * v)
    
    def fast_norm(v: np.ndarray) -> float:
        return np.linalg.norm(v)
    
    def fast_dh_transform(a: float, d: float, alpha: float, theta: float) -> np.ndarray:
        ct, st = np.cos(theta), np.sin(theta)
        ca, sa = np.cos(alpha), np.sin(alpha)
        return np.array([
            [ct, -st * ca, st * sa, a * ct],
            [st, ct * ca, -ct * sa, a * st],
            [0.0, sa, ca, d],
            [0.0, 0.0, 0.0, 1.0]
        ])
    
    def fast_matrix_multiply(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        return A @ B

