"""
Ejemplo de Uso de Extensiones Nativas
======================================

Ejemplos de cómo usar las extensiones C++ y Rust para mejorar el rendimiento.
"""

import numpy as np
import time
from native.wrapper import (
    NativeIKWrapper,
    NativeTrajectoryOptimizerWrapper,
    NativeMatrixOpsWrapper,
    NativeCollisionDetectorWrapper,
    json_parse,
    string_search,
    hash_data,
    CPP_AVAILABLE,
    RUST_AVAILABLE
)

def example_inverse_kinematics():
    """Ejemplo de cinemática inversa nativa."""
    print("=" * 60)
    print("Ejemplo: Cinemática Inversa Nativa")
    print("=" * 60)
    
    # Configurar robot
    link_lengths = [0.3, 0.3, 0.3, 0.2, 0.1, 0.05]
    joint_limits = [(-3.14, 3.14)] * 6
    
    # Crear wrapper
    ik = NativeIKWrapper(link_lengths, joint_limits)
    
    # Resolver cinemática inversa
    target_pos = [0.5, 0.3, 0.2]
    initial_guess = [0.0] * 6
    
    start_time = time.time()
    solution = ik.solve(target_pos, initial_guess)
    elapsed = time.time() - start_time
    
    print(f"Target position: {target_pos}")
    print(f"Solution: {solution}")
    print(f"Time: {elapsed*1000:.2f}ms")
    print(f"Using native: {ik.use_native}")
    print()


def example_trajectory_optimization():
    """Ejemplo de optimización de trayectorias."""
    print("=" * 60)
    print("Ejemplo: Optimización de Trayectorias")
    print("=" * 60)
    
    # Crear trayectoria inicial
    trajectory = np.array([
        [0.0, 0.0, 0.0],
        [0.2, 0.1, 0.1],
        [0.4, 0.2, 0.2],
        [0.6, 0.3, 0.3],
        [0.8, 0.4, 0.4],
    ])
    
    # Crear obstáculos
    obstacles = np.array([
        [0.3, 0.15, 0.15, 0.05],  # Esfera en (0.3, 0.15, 0.15) con radio 0.05
        [0.5, 0.25, 0.25, 0.08],
    ])
    
    # Crear optimizador
    optimizer = NativeTrajectoryOptimizerWrapper(
        energy_weight=0.3,
        time_weight=0.3,
        smoothness_weight=0.2
    )
    
    # Optimizar
    start_time = time.time()
    optimized = optimizer.optimize(trajectory, obstacles)
    elapsed = time.time() - start_time
    
    print(f"Original trajectory shape: {trajectory.shape}")
    print(f"Optimized trajectory shape: {optimized.shape}")
    print(f"Time: {elapsed*1000:.2f}ms")
    print(f"Using native: {optimizer.use_native}")
    print(f"First point: {optimized[0]}")
    print(f"Last point: {optimized[-1]}")
    print()


def example_matrix_operations():
    """Ejemplo de operaciones matriciales."""
    print("=" * 60)
    print("Ejemplo: Operaciones Matriciales")
    print("=" * 60)
    
    # Crear matrices grandes
    size = 500
    a = np.random.rand(size, size)
    b = np.random.rand(size, size)
    
    # Multiplicación
    start_time = time.time()
    result = NativeMatrixOpsWrapper.matmul(a, b)
    elapsed = time.time() - start_time
    print(f"Matrix multiplication ({size}x{size}): {elapsed*1000:.2f}ms")
    
    # Inversa
    start_time = time.time()
    inverse = NativeMatrixOpsWrapper.inv(a)
    elapsed = time.time() - start_time
    print(f"Matrix inverse ({size}x{size}): {elapsed*1000:.2f}ms")
    
    # Determinante
    start_time = time.time()
    det = NativeMatrixOpsWrapper.det(a)
    elapsed = time.time() - start_time
    print(f"Matrix determinant ({size}x{size}): {elapsed*1000:.2f}ms")
    print(f"Determinant value: {det:.6e}")
    print()


def example_collision_detection():
    """Ejemplo de detección de colisiones."""
    print("=" * 60)
    print("Ejemplo: Detección de Colisiones")
    print("=" * 60)
    
    # Crear trayectoria
    trajectory = np.array([
        [0.0, 0.0, 0.0],
        [0.1, 0.1, 0.1],
        [0.2, 0.2, 0.2],
        [0.3, 0.3, 0.3],
        [0.4, 0.4, 0.4],
    ])
    
    # Crear obstáculos
    obstacles = np.array([
        [0.25, 0.25, 0.25, 0.1],  # Esfera que intersecta la trayectoria
        [0.6, 0.6, 0.6, 0.1],     # Esfera fuera de la trayectoria
    ])
    
    # Verificar colisiones
    start_time = time.time()
    has_collision = NativeCollisionDetectorWrapper.check_trajectory_collision(
        trajectory, obstacles
    )
    elapsed = time.time() - start_time
    
    print(f"Trajectory points: {len(trajectory)}")
    print(f"Obstacles: {len(obstacles)}")
    print(f"Collision detected: {has_collision}")
    print(f"Time: {elapsed*1000:.2f}ms")
    print()


def example_rust_extensions():
    """Ejemplo de extensiones Rust."""
    print("=" * 60)
    print("Ejemplo: Extensiones Rust")
    print("=" * 60)
    
    if not RUST_AVAILABLE:
        print("Rust extensions not available")
        return
    
    # Parsing JSON
    json_str = '{"name": "Robot", "position": [0.5, 0.3, 0.2], "active": true}'
    start_time = time.time()
    data = json_parse(json_str)
    elapsed = time.time() - start_time
    print(f"JSON parsing: {elapsed*1000:.2f}ms")
    print(f"Parsed data: {data}")
    
    # Búsqueda de strings
    text = "Hello world, hello Python, hello Rust, hello C++"
    pattern = "hello"
    start_time = time.time()
    positions = string_search(text, pattern)
    elapsed = time.time() - start_time
    print(f"\nString search: {elapsed*1000:.2f}ms")
    print(f"Pattern '{pattern}' found at positions: {positions}")
    
    # Hash
    data_str = "some important data to hash"
    start_time = time.time()
    hash_value = hash_data(data_str)
    elapsed = time.time() - start_time
    print(f"\nHash: {elapsed*1000:.2f}ms")
    print(f"Hash value: {hash_value}")
    print()


def benchmark_comparison():
    """Comparación de rendimiento entre Python y extensiones nativas."""
    print("=" * 60)
    print("Benchmark: Comparación de Rendimiento")
    print("=" * 60)
    
    # Cinemática inversa
    link_lengths = [0.3, 0.3, 0.3, 0.2, 0.1, 0.05]
    joint_limits = [(-3.14, 3.14)] * 6
    ik = NativeIKWrapper(link_lengths, joint_limits)
    
    target_pos = [0.5, 0.3, 0.2]
    initial_guess = [0.0] * 6
    
    # Ejecutar múltiples veces
    n_iterations = 100
    
    start_time = time.time()
    for _ in range(n_iterations):
        solution = ik.solve(target_pos, initial_guess)
    elapsed = time.time() - start_time
    
    avg_time = elapsed / n_iterations * 1000
    print(f"Inverse Kinematics ({n_iterations} iterations):")
    print(f"  Average time: {avg_time:.2f}ms")
    print(f"  Using native: {ik.use_native}")
    print()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Robot Movement AI - Native Extensions Examples")
    print("=" * 60)
    print(f"C++ Extensions Available: {CPP_AVAILABLE}")
    print(f"Rust Extensions Available: {RUST_AVAILABLE}")
    print("=" * 60 + "\n")
    
    try:
        example_inverse_kinematics()
        example_trajectory_optimization()
        example_matrix_operations()
        example_collision_detection()
        
        if RUST_AVAILABLE:
            example_rust_extensions()
        
        benchmark_comparison()
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

