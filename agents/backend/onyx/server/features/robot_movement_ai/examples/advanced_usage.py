"""
Advanced Usage Examples
=======================

Ejemplos avanzados de uso del sistema.
"""

import numpy as np
import asyncio
from robot_movement_ai.config.robot_config import RobotConfig, RobotBrand
from robot_movement_ai.core.trajectory_optimizer import TrajectoryOptimizer, TrajectoryPoint
from robot_movement_ai.core.movement_engine import RobotMovementEngine
from robot_movement_ai.core.logging_config import setup_logging, get_logger
from robot_movement_ai.core.constants import OptimizationAlgorithm

setup_logging(level="INFO", colored=True)
logger = get_logger(__name__)


async def example_multi_waypoint_path():
    """Ejemplo de trayectoria con múltiples waypoints."""
    logger.info("=== Ejemplo: Trayectoria Multi-Waypoint ===")
    
    config = RobotConfig(robot_brand=RobotBrand.GENERIC, ros_enabled=False)
    engine = RobotMovementEngine(config)
    
    # Definir waypoints
    waypoints = [
        TrajectoryPoint(position=np.array([0.0, 0.0, 0.0]), orientation=np.array([0, 0, 0, 1])),
        TrajectoryPoint(position=np.array([0.5, 0.5, 0.5]), orientation=np.array([0, 0, 0, 1])),
        TrajectoryPoint(position=np.array([1.0, 1.0, 1.0]), orientation=np.array([0, 0, 0, 1])),
        TrajectoryPoint(position=np.array([1.5, 0.5, 1.0]), orientation=np.array([0, 0, 0, 1])),
    ]
    
    # Convertir a formato de API
    from robot_movement_ai.core.inverse_kinematics import EndEffectorPose
    poses = [
        EndEffectorPose(
            position=wp.position.tolist(),
            orientation=wp.orientation.tolist()
        )
        for wp in waypoints
    ]
    
    logger.info(f"Ejecutando trayectoria con {len(waypoints)} waypoints")
    # En un caso real, usaríamos: await engine.move_along_path(poses)
    logger.info("Trayectoria multi-waypoint completada")


def example_custom_optimization_params():
    """Ejemplo con parámetros de optimización personalizados."""
    logger.info("=== Ejemplo: Parámetros Personalizados ===")
    
    from robot_movement_ai.core.trajectory_optimizer import OptimizationParams
    
    # Crear parámetros personalizados
    custom_params = OptimizationParams(
        energy_weight=0.4,      # Priorizar energía
        time_weight=0.2,        # Menos prioridad a tiempo
        smoothness_weight=0.2,
        safety_weight=0.2,
        max_iterations=200,     # Más iteraciones
        convergence_threshold=1e-7
    )
    
    optimizer = TrajectoryOptimizer(optimization_params=custom_params)
    
    start = TrajectoryPoint(
        position=np.array([0.0, 0.0, 0.0]),
        orientation=np.array([0.0, 0.0, 0.0, 1.0])
    )
    
    goal = TrajectoryPoint(
        position=np.array([1.0, 1.0, 1.0]),
        orientation=np.array([0.0, 0.0, 0.0, 1.0])
    )
    
    trajectory = optimizer.optimize_trajectory(start, goal)
    logger.info(f"Trayectoria optimizada con parámetros personalizados: {len(trajectory)} puntos")


def example_astar_optimization():
    """Ejemplo usando algoritmo A*."""
    logger.info("=== Ejemplo: Optimización A* ===")
    
    optimizer = TrajectoryOptimizer()
    
    start = TrajectoryPoint(
        position=np.array([0.0, 0.0, 0.0]),
        orientation=np.array([0.0, 0.0, 0.0, 1.0])
    )
    
    goal = TrajectoryPoint(
        position=np.array([2.0, 2.0, 2.0]),
        orientation=np.array([0.0, 0.0, 0.0, 1.0])
    )
    
    obstacles = [
        np.array([0.5, 0.5, 0.5, 1.5, 1.5, 1.5])
    ]
    
    # Usar A*
    trajectory = optimizer.optimize_with_astar(
        start,
        goal,
        obstacles=obstacles,
        grid_resolution=0.1
    )
    
    logger.info(f"Trayectoria A* con {len(trajectory)} puntos")


def example_rrt_optimization():
    """Ejemplo usando algoritmo RRT."""
    logger.info("=== Ejemplo: Optimización RRT ===")
    
    optimizer = TrajectoryOptimizer()
    
    start = TrajectoryPoint(
        position=np.array([0.0, 0.0, 0.0]),
        orientation=np.array([0.0, 0.0, 0.0, 1.0])
    )
    
    goal = TrajectoryPoint(
        position=np.array([2.0, 2.0, 2.0]),
        orientation=np.array([0.0, 0.0, 0.0, 1.0])
    )
    
    obstacles = [
        np.array([0.5, 0.5, 0.5, 1.5, 1.5, 1.5])
    ]
    
    # Usar RRT
    trajectory = optimizer.optimize_with_rrt(
        start,
        goal,
        obstacles=obstacles,
        max_iterations=2000,
        step_size=0.15
    )
    
    logger.info(f"Trayectoria RRT con {len(trajectory)} puntos")


def example_multi_objective_optimization():
    """Ejemplo de optimización multi-objetivo."""
    logger.info("=== Ejemplo: Optimización Multi-Objetivo ===")
    
    optimizer = TrajectoryOptimizer()
    
    start = TrajectoryPoint(
        position=np.array([0.0, 0.0, 0.0]),
        orientation=np.array([0.0, 0.0, 0.0, 1.0])
    )
    
    goal = TrajectoryPoint(
        position=np.array([1.0, 1.0, 1.0]),
        orientation=np.array([0.0, 0.0, 0.0, 1.0])
    )
    
    # Optimización multi-objetivo
    trajectory = optimizer.optimize_multi_objective(
        start,
        goal,
        objectives=["time", "energy", "smoothness", "safety"]
    )
    
    logger.info(f"Trayectoria multi-objetivo con {len(trajectory)} puntos")


def example_performance_profiling():
    """Ejemplo de profiling de performance."""
    logger.info("=== Ejemplo: Performance Profiling ===")
    
    from robot_movement_ai.core.performance import PerformanceProfiler, measure_time
    
    optimizer = TrajectoryOptimizer()
    
    start = TrajectoryPoint(
        position=np.array([0.0, 0.0, 0.0]),
        orientation=np.array([0.0, 0.0, 0.0, 1.0])
    )
    
    goal = TrajectoryPoint(
        position=np.array([1.0, 1.0, 1.0]),
        orientation=np.array([0.0, 0.0, 0.0, 1.0])
    )
    
    # Profiling
    profiler = PerformanceProfiler()
    
    with profiler.section("optimization"):
        trajectory = optimizer.optimize_trajectory(start, goal)
    
    with profiler.section("analysis"):
        analysis = optimizer.analyze_trajectory(trajectory)
    
    # Reporte
    report = profiler.get_report()
    logger.info("Reporte de performance:")
    logger.info(f"  Tiempo total: {report['total_time']:.4f}s")
    for section, stats in report['sections'].items():
        logger.info(f"  {section}: {stats['total_time']:.4f}s ({stats['percentage']:.1f}%)")


def example_extension_system():
    """Ejemplo del sistema de extensiones."""
    logger.info("=== Ejemplo: Sistema de Extensiones ===")
    
    from robot_movement_ai.core.extensions import Extension, get_extension_manager
    
    class CustomOptimizerExtension(Extension):
        def initialize(self):
            logger.info(f"Extension {self.name} initialized")
            return True
        
        def shutdown(self):
            logger.info(f"Extension {self.name} shutdown")
    
    # Registrar extensión
    manager = get_extension_manager()
    extension = CustomOptimizerExtension("custom_optimizer")
    manager.register(extension)
    
    # Usar hook
    def on_trajectory_optimized(trajectory):
        logger.info(f"Trajectory optimized: {len(trajectory)} points")
    
    manager.register_hook("trajectory_optimized", on_trajectory_optimized)
    
    # Simular optimización
    optimizer = TrajectoryOptimizer()
    start = TrajectoryPoint(
        position=np.array([0.0, 0.0, 0.0]),
        orientation=np.array([0.0, 0.0, 0.0, 1.0])
    )
    goal = TrajectoryPoint(
        position=np.array([1.0, 1.0, 1.0]),
        orientation=np.array([0.0, 0.0, 0.0, 1.0])
    )
    
    trajectory = optimizer.optimize_trajectory(start, goal)
    manager.call_hook("trajectory_optimized", trajectory)
    
    # Limpiar
    manager.unregister("custom_optimizer")


if __name__ == "__main__":
    logger.info("Ejemplos Avanzados de Robot Movement AI")
    logger.info("=" * 60)
    
    try:
        example_custom_optimization_params()
        print()
        
        example_astar_optimization()
        print()
        
        example_rrt_optimization()
        print()
        
        example_multi_objective_optimization()
        print()
        
        example_performance_profiling()
        print()
        
        example_extension_system()
        print()
        
        # Ejemplo async
        asyncio.run(example_multi_waypoint_path())
        print()
        
        logger.info("=" * 60)
        logger.info("Todos los ejemplos avanzados completados!")
        
    except Exception as e:
        logger.error(f"Error en ejemplos: {e}", exc_info=True)






