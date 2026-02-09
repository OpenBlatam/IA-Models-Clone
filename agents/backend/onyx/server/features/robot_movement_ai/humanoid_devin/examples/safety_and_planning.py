"""
Ejemplos de Seguridad y Planificación - Humanoid Devin Robot
============================================================

Ejemplos de uso del monitor de seguridad y planificador de trayectorias.
"""

import asyncio
import numpy as np

from ..drivers.humanoid_devin_driver import HumanoidDevinDriver
from ..helpers.safety_monitor import SafetyMonitor, SafetyError
from ..helpers.trajectory_planner import TrajectoryPlanner, TrajectoryPlannerError


async def example_safety_monitoring():
    """
    Ejemplo 1: Monitoreo de seguridad básico.
    """
    print("=== Ejemplo 1: Monitoreo de Seguridad ===")
    
    # Crear monitor de seguridad
    safety = SafetyMonitor(
        max_joint_velocity=2.0,
        max_cartesian_velocity=1.0,
        joint_limits=[(-np.pi, np.pi)] * 32,
        workspace_limits={
            "x": (-1.0, 1.0),
            "y": (-1.0, 1.0),
            "z": (0.0, 2.0)
        }
    )
    
    print("✓ Monitor de seguridad creado")
    
    # Validar posiciones de articulaciones
    positions = [0.0] * 32
    valid, error = safety.validate_joint_positions(positions)
    
    if valid:
        print("✓ Posiciones de articulaciones válidas")
    else:
        print(f"✗ Error: {error}")
    
    # Validar movimiento cartesiano
    start_pos = np.array([0.0, 0.0, 1.0])
    end_pos = np.array([0.5, 0.0, 1.0])
    valid, error = safety.validate_cartesian_movement(start_pos, end_pos, dt=0.1)
    
    if valid:
        print("✓ Movimiento cartesiano válido")
    else:
        print(f"✗ Error: {error}")
    
    # Obtener estado
    status = safety.get_status()
    print(f"✓ Estado del monitor: {status['is_safe']}")


async def example_emergency_stop():
    """
    Ejemplo 2: Parada de emergencia.
    """
    print("\n=== Ejemplo 2: Parada de Emergencia ===")
    
    safety = SafetyMonitor(emergency_stop_enabled=True)
    
    # Activar parada de emergencia
    safety.emergency_stop()
    print("✓ Parada de emergencia activada")
    
    # Intentar validar movimiento (debe fallar)
    positions = [0.0] * 32
    valid, error = safety.validate_joint_positions(positions)
    
    if not valid:
        print(f"✓ Movimiento bloqueado: {error}")
    
    # Limpiar parada de emergencia
    safety.clear_emergency_stop()
    print("✓ Parada de emergencia limpiada")
    
    # Ahora el movimiento debería ser válido
    valid, error = safety.validate_joint_positions(positions)
    if valid:
        print("✓ Movimiento ahora válido")


async def example_trajectory_planning():
    """
    Ejemplo 3: Planificación de trayectorias.
    """
    print("\n=== Ejemplo 3: Planificación de Trayectorias ===")
    
    # Crear planificador
    planner = TrajectoryPlanner(
        method="cubic",
        smooth=True,
        max_acceleration=1.0
    )
    
    print("✓ Planificador creado")
    
    # Planificar trayectoria de articulaciones
    start = [0.0] * 32
    end = [0.5] * 32
    
    trajectory = planner.plan_joint_trajectory(
        start=start,
        end=end,
        duration=2.0,
        num_points=50
    )
    
    print(f"✓ Trayectoria planificada: {trajectory.shape}")
    print(f"  Puntos: {len(trajectory)}")
    print(f"  DOF: {trajectory.shape[1]}")
    
    # Calcular velocidades
    velocities = planner.get_trajectory_velocity(trajectory, dt=0.04)
    print(f"✓ Velocidades calculadas: {velocities.shape}")
    print(f"  Velocidad máxima: {np.max(np.abs(velocities)):.3f} rad/s")
    
    # Calcular aceleraciones
    accelerations = planner.get_trajectory_acceleration(trajectory, dt=0.04)
    print(f"✓ Aceleraciones calculadas: {accelerations.shape}")
    print(f"  Aceleración máxima: {np.max(np.abs(accelerations)):.3f} rad/s²")


async def example_cartesian_trajectory():
    """
    Ejemplo 4: Trayectoria cartesiana con orientación.
    """
    print("\n=== Ejemplo 4: Trayectoria Cartesiana ===")
    
    planner = TrajectoryPlanner(method="quintic")
    
    # Posiciones
    start_pos = np.array([0.0, 0.0, 1.0])
    end_pos = np.array([0.5, 0.2, 1.2])
    
    # Orientaciones
    start_ori = np.array([0.0, 0.0, 0.0, 1.0])
    end_ori = np.array([0.0, 0.0, 0.707, 0.707])  # 90° en Z
    
    # Planificar
    trajectory = planner.plan_cartesian_trajectory(
        start_position=start_pos,
        end_position=end_pos,
        start_orientation=start_ori,
        end_orientation=end_ori,
        duration=2.0,
        num_points=50
    )
    
    print(f"✓ Trayectoria cartesiana planificada")
    print(f"  Posiciones: {trajectory['positions'].shape}")
    print(f"  Orientaciones: {trajectory['orientations'].shape}")
    
    # Verificar que las orientaciones están normalizadas
    norms = np.linalg.norm(trajectory['orientations'], axis=1)
    print(f"  Normas de quaterniones: min={np.min(norms):.6f}, max={np.max(norms):.6f}")


async def example_safe_trajectory_execution():
    """
    Ejemplo 5: Ejecución segura de trayectoria.
    """
    print("\n=== Ejemplo 5: Ejecución Segura de Trayectoria ===")
    
    # Crear monitor de seguridad y planificador
    safety = SafetyMonitor(
        max_joint_velocity=2.0,
        joint_limits=[(-np.pi, np.pi)] * 32
    )
    
    planner = TrajectoryPlanner(method="cubic")
    
    # Planificar trayectoria
    start = [0.0] * 32
    end = [0.3] * 32
    
    trajectory = planner.plan_joint_trajectory(
        start=start,
        end=end,
        duration=1.0,
        num_points=20
    )
    
    print(f"✓ Trayectoria planificada: {len(trajectory)} puntos")
    
    # Validar cada punto de la trayectoria
    valid_points = 0
    for i, positions in enumerate(trajectory):
        previous = trajectory[i-1] if i > 0 else start
        valid, error = safety.validate_joint_positions(
            positions,
            previous_positions=previous,
            dt=0.05
        )
        
        if valid:
            valid_points += 1
        else:
            print(f"  ⚠ Punto {i} inválido: {error}")
    
    print(f"✓ Puntos válidos: {valid_points}/{len(trajectory)}")
    
    if valid_points == len(trajectory):
        print("✓ Trayectoria completamente segura para ejecutar")
    else:
        print("⚠ Trayectoria contiene puntos inseguros")


async def example_different_planning_methods():
    """
    Ejemplo 6: Comparar diferentes métodos de planificación.
    """
    print("\n=== Ejemplo 6: Comparación de Métodos ===")
    
    start = [0.0] * 32
    end = [0.5] * 32
    
    methods = ["linear", "cubic", "quintic"]
    
    for method in methods:
        planner = TrajectoryPlanner(method=method)
        trajectory = planner.plan_joint_trajectory(
            start=start,
            end=end,
            duration=1.0,
            num_points=50
        )
        
        # Calcular suavidad (variación de velocidades)
        velocities = planner.get_trajectory_velocity(trajectory, dt=0.02)
        velocity_changes = np.diff(velocities, axis=0)
        smoothness = np.mean(np.abs(velocity_changes))
        
        print(f"  {method.capitalize()}:")
        print(f"    Suavidad: {smoothness:.6f}")
        print(f"    Velocidad máxima: {np.max(np.abs(velocities)):.3f} rad/s")


async def main():
    """
    Ejecutar todos los ejemplos de seguridad y planificación.
    """
    print("=" * 60)
    print("Ejemplos de Seguridad y Planificación")
    print("=" * 60)
    
    examples = [
        example_safety_monitoring,
        example_emergency_stop,
        example_trajectory_planning,
        example_cartesian_trajectory,
        example_safe_trajectory_execution,
        example_different_planning_methods
    ]
    
    for example in examples:
        try:
            await example()
        except Exception as e:
            print(f"✗ Error en ejemplo: {e}")
        print()
    
    print("=" * 60)
    print("Ejemplos completados")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())

