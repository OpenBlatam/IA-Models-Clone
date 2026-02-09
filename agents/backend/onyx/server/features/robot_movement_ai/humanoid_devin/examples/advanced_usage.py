"""
Ejemplos Avanzados del Humanoid Devin Robot (Optimizado)
========================================================

Ejemplos avanzados mostrando capacidades profesionales del robot.
"""

import asyncio
import numpy as np
from typing import List, Dict, Any

from ..drivers.humanoid_devin_driver import HumanoidDevinDriver, RobotType
from ..utils import (
    quaternion_to_euler,
    euler_to_quaternion,
    calculate_distance,
    get_joint_velocity
)
from ..exceptions import HumanoidRobotError


async def example_complex_trajectory():
    """
    Ejemplo avanzado: Trayectoria compleja con múltiples waypoints.
    """
    print("=== Ejemplo Avanzado 1: Trayectoria Compleja ===")
    
    robot = HumanoidDevinDriver(
        robot_ip="192.168.1.100",
        dof=32,
        use_ml=True,
        use_diffusion=True
    )
    
    try:
        await robot.connect()
        
        # Definir waypoints
        waypoints = [
            np.array([0.2, -0.1, 0.9]),
            np.array([0.3, -0.2, 1.0]),
            np.array([0.4, -0.1, 0.95]),
            np.array([0.3, 0.0, 0.9])
        ]
        
        # Orientación base
        orientation = np.array([0.0, 0.0, 0.0, 1.0])
        
        print(f"✓ Ejecutando trayectoria con {len(waypoints)} waypoints")
        
        for i, waypoint in enumerate(waypoints):
            # Generar trayectoria suave entre waypoints
            if i > 0:
                trajectory = await robot.generate_smooth_trajectory(
                    start_position=waypoints[i-1],
                    end_position=waypoint,
                    num_steps=30
                )
                
                if trajectory is not None:
                    print(f"  Waypoint {i+1}: Trayectoria suave generada ({len(trajectory)} pasos)")
            
            # Mover a waypoint
            success = await robot.move_to_pose(
                position=waypoint,
                orientation=orientation,
                hand="right"
            )
            
            if success:
                print(f"  ✓ Waypoint {i+1} alcanzado")
            else:
                print(f"  ✗ Error en waypoint {i+1}")
            
            await asyncio.sleep(0.5)
        
        print("✓ Trayectoria compleja completada")
        await robot.disconnect()
        
    except Exception as e:
        print(f"✗ Error: {e}")


async def example_bimanual_manipulation():
    """
    Ejemplo avanzado: Manipulación bimanual coordinada.
    """
    print("\n=== Ejemplo Avanzado 2: Manipulación Bimanual ===")
    
    robot = HumanoidDevinDriver(
        robot_ip="192.168.1.100",
        dof=32
    )
    
    try:
        await robot.connect()
        
        # Posiciones para ambas manos
        left_hand_pos = np.array([0.3, 0.2, 1.0])
        right_hand_pos = np.array([0.3, -0.2, 1.0])
        orientation = np.array([0.0, 0.0, 0.0, 1.0])
        
        print("✓ Moviendo ambas manos simultáneamente")
        
        # Mover ambas manos en paralelo
        tasks = [
            robot.move_to_pose(left_hand_pos, orientation, hand="left"),
            robot.move_to_pose(right_hand_pos, orientation, hand="right")
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for i, result in enumerate(results):
            hand = "izquierda" if i == 0 else "derecha"
            if isinstance(result, Exception):
                print(f"  ✗ Error moviendo mano {hand}: {result}")
            else:
                print(f"  ✓ Mano {hand} movida exitosamente")
        
        # Agarrar con ambas manos
        await robot.grasp(hand="left")
        await robot.grasp(hand="right")
        print("✓ Ambas manos agarrando")
        
        await asyncio.sleep(1)
        
        # Soltar
        await robot.release(hand="left")
        await robot.release(hand="right")
        print("✓ Ambas manos soltadas")
        
        await robot.disconnect()
        
    except Exception as e:
        print(f"✗ Error: {e}")


async def example_adaptive_control():
    """
    Ejemplo avanzado: Control adaptativo usando feedback.
    """
    print("\n=== Ejemplo Avanzado 3: Control Adaptativo ===")
    
    robot = HumanoidDevinDriver(
        robot_ip="192.168.1.100",
        use_ml=True
    )
    
    try:
        await robot.connect()
        
        # Obtener estado inicial
        initial_positions = np.array(await robot.get_joint_positions())
        target_positions = initial_positions + 0.2
        
        print("✓ Iniciando control adaptativo")
        
        # Control en bucle con feedback
        current_positions = initial_positions.copy()
        max_iterations = 10
        tolerance = 0.01
        
        for iteration in range(max_iterations):
            # Calcular error
            error = np.linalg.norm(target_positions - current_positions)
            print(f"  Iteración {iteration+1}: Error = {error:.4f}")
            
            if error < tolerance:
                print("  ✓ Objetivo alcanzado")
                break
            
            # Predecir movimiento usando Transformer
            if robot.use_ml and robot.transformer_model:
                predicted = await robot.predict_joint_motion(
                    current_joints=current_positions,
                    target_joints=target_positions
                )
                
                if predicted is not None and len(predicted) > 0:
                    next_positions = predicted[0]
                else:
                    # Fallback: movimiento proporcional
                    next_positions = current_positions + 0.1 * (target_positions - current_positions)
            else:
                # Movimiento proporcional simple
                next_positions = current_positions + 0.1 * (target_positions - current_positions)
            
            # Aplicar movimiento
            await robot.set_joint_positions(next_positions.tolist())
            
            # Actualizar estado
            current_positions = np.array(await robot.get_joint_positions())
            
            await asyncio.sleep(0.1)
        
        print("✓ Control adaptativo completado")
        await robot.disconnect()
        
    except Exception as e:
        print(f"✗ Error: {e}")


async def example_obstacle_avoidance():
    """
    Ejemplo avanzado: Evasión de obstáculos usando PCL.
    """
    print("\n=== Ejemplo Avanzado 4: Evasión de Obstáculos ===")
    
    robot = HumanoidDevinDriver(
        robot_ip="192.168.1.100",
        use_pcl=True,
        use_nav2=True
    )
    
    try:
        await robot.connect()
        
        if robot.pcl_processor and robot.pcl_processor.available:
            # Simular nube de puntos (en producción vendría de sensor)
            # Crear puntos que representan un obstáculo
            obstacle_points = np.array([
                [0.5, 0.0, 0.5],
                [0.5, 0.1, 0.5],
                [0.5, -0.1, 0.5],
                [0.6, 0.0, 0.5],
                [0.4, 0.0, 0.5],
            ])
            
            # Detectar obstáculos
            obstacles = robot.pcl_processor.detect_obstacles(
                points=obstacle_points,
                min_points=3,
                max_distance=0.2
            )
            
            print(f"✓ Obstáculos detectados: {len(obstacles)}")
            
            for i, obstacle in enumerate(obstacles):
                center = obstacle['center']
                print(f"  Obstáculo {i+1}: Centro en {center}")
                
                # Calcular distancia al obstáculo
                robot_pos = np.array([0.0, 0.0, 0.0])  # Posición actual del robot
                distance = calculate_distance(robot_pos, center)
                print(f"    Distancia: {distance:.2f}m")
                
                # Si el obstáculo está en el camino, ajustar ruta
                if distance < 0.8:
                    print(f"    ⚠ Obstáculo cercano, ajustando ruta")
                    # En producción, aquí se planificaría una ruta alternativa
        else:
            print("⚠ PCL no disponible")
        
        await robot.disconnect()
        
    except Exception as e:
        print(f"✗ Error: {e}")


async def example_gesture_sequence():
    """
    Ejemplo avanzado: Secuencia de gestos coordinados.
    """
    print("\n=== Ejemplo Avanzado 5: Secuencia de Gestos ===")
    
    robot = HumanoidDevinDriver(
        robot_ip="192.168.1.100",
        dof=32
    )
    
    try:
        await robot.connect()
        
        print("✓ Ejecutando secuencia de gestos")
        
        # Secuencia de gestos
        gestures = [
            ("wave", "right"),
            ("wave", "left"),
            ("stand", None),
            ("wave", "right")
        ]
        
        for gesture, hand in gestures:
            if gesture == "wave":
                success = await robot.wave(hand=hand)
                print(f"  ✓ Saludando con mano {hand}")
            elif gesture == "stand":
                success = await robot.stand()
                print(f"  ✓ De pie")
            
            await asyncio.sleep(1)
        
        print("✓ Secuencia de gestos completada")
        await robot.disconnect()
        
    except Exception as e:
        print(f"✗ Error: {e}")


async def example_performance_monitoring():
    """
    Ejemplo avanzado: Monitoreo de rendimiento.
    """
    print("\n=== Ejemplo Avanzado 6: Monitoreo de Rendimiento ===")
    
    import time
    
    robot = HumanoidDevinDriver(
        robot_ip="192.168.1.100",
        use_ml=True
    )
    
    try:
        await robot.connect()
        
        # Monitorear tiempo de respuesta
        operations = [
            ("get_status", lambda: robot.get_status()),
            ("get_joint_positions", lambda: robot.get_joint_positions()),
            ("set_joint_positions", lambda: robot.set_joint_positions([0.0] * 32)),
        ]
        
        print("✓ Monitoreando rendimiento de operaciones")
        
        for op_name, op_func in operations:
            start_time = time.time()
            try:
                await op_func()
                duration = time.time() - start_time
                print(f"  {op_name}: {duration*1000:.2f}ms")
            except Exception as e:
                print(f"  {op_name}: Error - {e}")
        
        # Obtener estado completo
        status = await robot.get_status()
        print(f"\n✓ Estado del sistema:")
        print(f"  Integraciones activas: {sum(1 for v in status['integrations'].values() if v)}")
        print(f"  ML habilitado: {status['integrations']['deep_learning']['ml_enabled']}")
        
        await robot.disconnect()
        
    except Exception as e:
        print(f"✗ Error: {e}")


async def main():
    """
    Ejecutar todos los ejemplos avanzados.
    """
    print("=" * 60)
    print("Ejemplos Avanzados - Humanoid Devin Robot")
    print("=" * 60)
    
    examples = [
        example_complex_trajectory,
        example_bimanual_manipulation,
        example_adaptive_control,
        example_obstacle_avoidance,
        example_gesture_sequence,
        example_performance_monitoring
    ]
    
    for example in examples:
        try:
            await example()
        except Exception as e:
            print(f"✗ Error en ejemplo: {e}")
        print()
    
    print("=" * 60)
    print("Ejemplos avanzados completados")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())

