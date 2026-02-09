"""
Ejemplos Avanzados de Uso de Extensiones Nativas
================================================

Ejemplos completos de todas las nuevas funcionalidades agregadas.
"""

import numpy as np
import time
from native.wrapper import (
    NativeTransform3DWrapper,
    NativeVectorOpsWrapper,
    NativeInterpolationWrapper,
    NativeMatrixOpsWrapper,
    NativeIKWrapper,
    NativeTrajectoryOptimizerWrapper,
    NativeCollisionDetectorWrapper,
    array_mean,
    array_std,
    array_median,
    array_percentile,
    array_filter,
    binary_search,
    string_split,
    string_join,
    string_trim,
    string_upper,
    string_lower,
    json_validate,
    get_native_extensions_status,
    CPP_AVAILABLE,
    RUST_AVAILABLE
)


def example_3d_transformations():
    """Ejemplo de transformaciones 3D."""
    print("=" * 60)
    print("Ejemplo: Transformaciones 3D")
    print("=" * 60)
    
    # Crear rotaciones
    angle_x = np.pi / 4
    angle_y = np.pi / 6
    angle_z = np.pi / 3
    
    rot_x = NativeTransform3DWrapper.rotation_x(angle_x)
    rot_y = NativeTransform3DWrapper.rotation_y(angle_y)
    rot_z = NativeTransform3DWrapper.rotation_z(angle_z)
    
    print(f"Rotation X matrix:\n{rot_x}")
    print(f"Rotation Y matrix:\n{rot_y}")
    print(f"Rotation Z matrix:\n{rot_z}")
    
    # Rotar punto
    point = np.array([1.0, 0.0, 0.0])
    rotated = NativeTransform3DWrapper.rotate_point(rot_z, point)
    print(f"Point {point} rotated by Z: {rotated}")
    print()


def example_vector_operations():
    """Ejemplo de operaciones vectoriales."""
    print("=" * 60)
    print("Ejemplo: Operaciones Vectoriales")
    print("=" * 60)
    
    # Normalizar
    vec = np.array([3.0, 4.0, 0.0])
    normalized = NativeVectorOpsWrapper.normalize(vec)
    print(f"Vector {vec} normalized: {normalized}")
    
    # Distancia
    a = np.array([0.0, 0.0, 0.0])
    b = np.array([1.0, 1.0, 1.0])
    dist = NativeVectorOpsWrapper.distance(a, b)
    print(f"Distance between {a} and {b}: {dist:.4f}")
    
    # Producto punto
    dot = NativeVectorOpsWrapper.dot(a, b)
    print(f"Dot product: {dot}")
    
    # Producto cruz
    v1 = np.array([1.0, 0.0, 0.0])
    v2 = np.array([0.0, 1.0, 0.0])
    cross = NativeVectorOpsWrapper.cross(v1, v2)
    print(f"Cross product of {v1} and {v2}: {cross}")
    print()


def example_interpolation():
    """Ejemplo de interpolación."""
    print("=" * 60)
    print("Ejemplo: Interpolación Lineal")
    print("=" * 60)
    
    # Waypoints
    waypoints = np.array([
        [0.0, 0.0, 0.0],
        [0.3, 0.2, 0.1],
        [0.6, 0.4, 0.2],
        [1.0, 0.6, 0.3],
    ])
    
    # Interpolar a 100 puntos
    smooth = NativeInterpolationWrapper.linear(waypoints, 100)
    
    print(f"Original waypoints: {len(waypoints)}")
    print(f"Interpolated points: {len(smooth)}")
    print(f"First point: {smooth[0]}")
    print(f"Last point: {smooth[-1]}")
    print()


def example_matrix_operations():
    """Ejemplo de operaciones matriciales extendidas."""
    print("=" * 60)
    print("Ejemplo: Operaciones Matriciales Extendidas")
    print("=" * 60)
    
    # Crear matriz
    matrix = np.random.rand(5, 5)
    
    # Transpuesta
    transposed = NativeMatrixOpsWrapper.transpose(matrix)
    print(f"Original shape: {matrix.shape}")
    print(f"Transposed shape: {transposed.shape}")
    
    # Norma
    norm = NativeMatrixOpsWrapper.norm(matrix)
    print(f"Matrix norm: {norm:.4f}")
    
    # Traza
    trace = NativeMatrixOpsWrapper.trace(matrix)
    print(f"Matrix trace: {trace:.4f}")
    print()


def example_rust_arrays():
    """Ejemplo de operaciones Rust con arrays."""
    print("=" * 60)
    print("Ejemplo: Operaciones Rust - Arrays")
    print("=" * 60)
    
    data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    
    # Estadísticas
    mean_val = array_mean(data)
    std_val = array_std(data)
    median_val = array_median(data)
    p75 = array_percentile(data, 75.0)
    
    print(f"Data: {data}")
    print(f"Mean: {mean_val:.2f}")
    print(f"Std: {std_val:.2f}")
    print(f"Median: {median_val:.2f}")
    print(f"75th percentile: {p75:.2f}")
    
    # Filtrar
    filtered = array_filter(data, 5.0, "gt")
    print(f"Values > 5.0: {filtered}")
    
    # Búsqueda binaria
    sorted_data = sorted(data)
    index = binary_search(sorted_data, 5.0)
    print(f"Binary search for 5.0: index {index}")
    print()


def example_rust_strings():
    """Ejemplo de operaciones Rust con strings."""
    print("=" * 60)
    print("Ejemplo: Operaciones Rust - Strings")
    print("=" * 60)
    
    text = "  Hello World from Robot Movement AI  "
    
    # Trim
    trimmed = string_trim(text)
    print(f"Original: '{text}'")
    print(f"Trimmed: '{trimmed}'")
    
    # Upper/Lower
    upper = string_upper(text)
    lower = string_lower(text)
    print(f"Upper: '{upper}'")
    print(f"Lower: '{lower}'")
    
    # Split
    csv = "a,b,c,d,e"
    parts = string_split(csv, ",")
    print(f"Split '{csv}': {parts}")
    
    # Join
    joined = string_join(parts, "-")
    print(f"Joined: '{joined}'")
    print()


def example_complete_workflow():
    """Ejemplo de flujo completo de trabajo."""
    print("=" * 60)
    print("Ejemplo: Flujo Completo de Trabajo")
    print("=" * 60)
    
    # 1. Crear trayectoria inicial
    waypoints = np.array([
        [0.0, 0.0, 0.0],
        [0.5, 0.3, 0.2],
        [1.0, 0.6, 0.4],
    ])
    
    # 2. Interpolar para suavizar
    smooth_traj = NativeInterpolationWrapper.linear(waypoints, 50)
    
    # 3. Optimizar trayectoria
    optimizer = NativeTrajectoryOptimizerWrapper()
    obstacles = np.array([[0.3, 0.2, 0.1, 0.05]])
    optimized = optimizer.optimize(smooth_traj, obstacles)
    
    # 4. Verificar colisiones
    has_collision = NativeCollisionDetectorWrapper.check_trajectory_collision(
        optimized, obstacles
    )
    
    # 5. Calcular estadísticas de la trayectoria
    distances = []
    for i in range(len(optimized) - 1):
        dist = NativeVectorOpsWrapper.distance(
            optimized[i], optimized[i + 1]
        )
        distances.append(dist)
    
    total_distance = sum(distances)
    mean_dist = array_mean(distances)
    std_dist = array_std(distances)
    
    print(f"Original waypoints: {len(waypoints)}")
    print(f"Interpolated points: {len(smooth_traj)}")
    print(f"Optimized points: {len(optimized)}")
    print(f"Has collision: {has_collision}")
    print(f"Total distance: {total_distance:.4f}")
    print(f"Mean segment distance: {mean_dist:.4f}")
    print(f"Std segment distance: {std_dist:.4f}")
    print()


def benchmark_all():
    """Benchmark de todas las operaciones."""
    print("=" * 60)
    print("Benchmark: Todas las Operaciones")
    print("=" * 60)
    
    n_iterations = 100
    
    # Matrices
    matrix = np.random.rand(100, 100)
    start = time.time()
    for _ in range(n_iterations):
        NativeMatrixOpsWrapper.transpose(matrix)
    elapsed = (time.time() - start) / n_iterations * 1000
    print(f"Matrix transpose: {elapsed:.3f}ms")
    
    # Vectores
    vec = np.random.rand(1000)
    start = time.time()
    for _ in range(n_iterations):
        NativeVectorOpsWrapper.normalize(vec)
    elapsed = (time.time() - start) / n_iterations * 1000
    print(f"Vector normalize: {elapsed:.3f}ms")
    
    # Interpolación
    points = np.random.rand(10, 3)
    start = time.time()
    for _ in range(n_iterations):
        NativeInterpolationWrapper.linear(points, 100)
    elapsed = (time.time() - start) / n_iterations * 1000
    print(f"Interpolation: {elapsed:.3f}ms")
    
    # Arrays Rust
    data = list(range(1000))
    start = time.time()
    for _ in range(n_iterations):
        array_median(data)
    elapsed = (time.time() - start) / n_iterations * 1000
    print(f"Array median: {elapsed:.3f}ms")
    print()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Robot Movement AI - Advanced Native Extensions Examples")
    print("=" * 60)
    print(f"C++ Extensions Available: {CPP_AVAILABLE}")
    print(f"Rust Extensions Available: {RUST_AVAILABLE}")
    print("=" * 60 + "\n")
    
    # Mostrar estado
    status = get_native_extensions_status()
    print("Native Extensions Status:")
    print(f"  C++: {status['cpp_available']}")
    print(f"  Rust: {status['rust_available']}")
    print()
    
    try:
        if CPP_AVAILABLE:
            example_3d_transformations()
            example_vector_operations()
            example_interpolation()
            example_matrix_operations()
            example_complete_workflow()
        
        if RUST_AVAILABLE:
            example_rust_arrays()
            example_rust_strings()
        
        benchmark_all()
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

