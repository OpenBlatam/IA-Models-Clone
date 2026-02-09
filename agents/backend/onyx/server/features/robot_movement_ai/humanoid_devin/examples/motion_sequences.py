"""
Ejemplos de Secuencias de Movimiento - Humanoid Devin Robot
============================================================

Ejemplos de uso del secuenciador de movimientos y biblioteca de gestos.
"""

import asyncio
import numpy as np

from ..drivers.humanoid_devin_driver import HumanoidDevinDriver
from ..helpers.motion_sequencer import MotionSequencer, MotionType
from ..helpers.gesture_library import GestureLibrary


async def example_simple_sequence():
    """
    Ejemplo 1: Secuencia simple de movimientos.
    """
    print("=== Ejemplo 1: Secuencia Simple ===")
    
    robot = HumanoidDevinDriver(robot_ip="192.168.1.100", dof=32)
    await robot.connect()
    
    sequencer = MotionSequencer(robot)
    
    # Agregar pasos
    sequencer.add_step(
        MotionType.STAND,
        {},
        duration=1.0,
        name="stand"
    )
    
    sequencer.add_step(
        MotionType.WAVE,
        {"hand": "right"},
        duration=2.0,
        name="wave_right"
    )
    
    sequencer.add_step(
        MotionType.WAVE,
        {"hand": "left"},
        duration=2.0,
        name="wave_left"
    )
    
    print(f"✓ Secuencia creada: {len(sequencer.sequence)} pasos")
    
    # Obtener información
    info = sequencer.get_sequence_info()
    print(f"  Duración total: {info['total_duration']:.1f}s")
    
    # Ejecutar secuencia
    results = await sequencer.execute_sequence()
    print(f"✓ Secuencia ejecutada: {results['steps_executed']}/{results['steps_total']} pasos")
    
    await robot.disconnect()


async def example_joint_sequence():
    """
    Ejemplo 2: Secuencia de posiciones de articulaciones.
    """
    print("\n=== Ejemplo 2: Secuencia de Articulaciones ===")
    
    robot = HumanoidDevinDriver(robot_ip="192.168.1.100", dof=32)
    await robot.connect()
    
    sequencer = MotionSequencer(robot)
    
    # Obtener posiciones iniciales
    initial = await robot.get_joint_positions()
    
    # Crear secuencia de movimientos de cabeza
    head_yaw_idx = 0
    head_pitch_idx = 1
    
    positions_list = [
        initial.copy(),
        initial.copy(),
        initial.copy(),
        initial.copy(),
        initial.copy()
    ]
    
    # Mover cabeza
    positions_list[1][head_yaw_idx] = 0.3
    positions_list[2][head_yaw_idx] = -0.3
    positions_list[3][head_pitch_idx] = 0.2
    positions_list[4][head_pitch_idx] = -0.2
    
    for i, positions in enumerate(positions_list):
        sequencer.add_step(
            MotionType.JOINT_POSITIONS,
            {"positions": positions},
            duration=0.5,
            name=f"head_movement_{i}"
        )
    
    print(f"✓ Secuencia de articulaciones creada: {len(sequencer.sequence)} pasos")
    
    # Ejecutar
    results = await sequencer.execute_sequence()
    print(f"✓ Secuencia ejecutada exitosamente")
    
    await robot.disconnect()


async def example_pose_sequence():
    """
    Ejemplo 3: Secuencia de poses.
    """
    print("\n=== Ejemplo 3: Secuencia de Poses ===")
    
    robot = HumanoidDevinDriver(robot_ip="192.168.1.100", dof=32)
    await robot.connect()
    
    sequencer = MotionSequencer(robot)
    
    # Definir poses
    poses = [
        {
            "position": [0.3, -0.2, 1.0],
            "orientation": [0.0, 0.0, 0.0, 1.0]
        },
        {
            "position": [0.4, 0.0, 1.1],
            "orientation": [0.0, 0.0, 0.0, 1.0]
        },
        {
            "position": [0.3, 0.2, 1.0],
            "orientation": [0.0, 0.0, 0.0, 1.0]
        }
    ]
    
    for i, pose in enumerate(poses):
        sequencer.add_step(
            MotionType.POSE,
            {
                "position": np.array(pose["position"]),
                "orientation": np.array(pose["orientation"]),
                "hand": "right"
            },
            duration=1.0,
            name=f"pose_{i}"
        )
    
    print(f"✓ Secuencia de poses creada: {len(sequencer.sequence)} pasos")
    
    # Ejecutar
    results = await sequencer.execute_sequence()
    print(f"✓ Secuencia ejecutada: {results['success']}")
    
    await robot.disconnect()


async def example_gesture_library():
    """
    Ejemplo 4: Usar biblioteca de gestos.
    """
    print("\n=== Ejemplo 4: Biblioteca de Gestos ===")
    
    robot = HumanoidDevinDriver(robot_ip="192.168.1.100", dof=32)
    await robot.connect()
    
    gesture_lib = GestureLibrary(robot)
    sequencer = MotionSequencer(robot)
    
    # Obtener gesto de saludo
    wave_steps = gesture_lib.get_wave_gesture(hand="right", repetitions=3)
    print(f"✓ Gesto de saludo obtenido: {len(wave_steps)} pasos")
    
    # Agregar pasos del gesto a la secuencia
    for step in wave_steps:
        motion_type = MotionType(step["type"])
        sequencer.add_step(
            motion_type,
            {k: v for k, v in step.items() if k != "type" and k != "duration"},
            duration=step.get("duration", 1.0),
            name=f"wave_step_{len(sequencer.sequence)}"
        )
    
    # Ejecutar
    results = await sequencer.execute_sequence()
    print(f"✓ Gesto ejecutado: {results['success']}")
    
    # Listar todos los gestos disponibles
    all_gestures = gesture_lib.get_all_gestures()
    print(f"✓ Gestos disponibles: {list(all_gestures.keys())}")
    
    await robot.disconnect()


async def example_complex_sequence():
    """
    Ejemplo 5: Secuencia compleja con múltiples tipos de movimientos.
    """
    print("\n=== Ejemplo 5: Secuencia Compleja ===")
    
    robot = HumanoidDevinDriver(robot_ip="192.168.1.100", dof=32)
    await robot.connect()
    
    sequencer = MotionSequencer(robot)
    
    # Secuencia: pararse -> saludar -> señalar -> agarrar -> soltar -> inclinarse
    sequencer.add_step(MotionType.STAND, {}, duration=1.0, name="stand")
    sequencer.add_step(MotionType.WAVE, {"hand": "right"}, duration=2.0, name="wave")
    
    # Señalar
    sequencer.add_step(
        MotionType.POSE,
        {
            "position": np.array([0.5, 0.0, 1.0]),
            "orientation": np.array([0.0, 0.0, 0.0, 1.0]),
            "hand": "right"
        },
        duration=1.0,
        name="point"
    )
    
    sequencer.add_step(MotionType.GRASP, {"hand": "right"}, duration=1.0, name="grasp")
    sequencer.add_step(MotionType.RELEASE, {"hand": "right"}, duration=1.0, name="release")
    
    # Inclinarse (usando gesto)
    gesture_lib = GestureLibrary(robot)
    bow_steps = gesture_lib.get_bowing_gesture(depth=0.3)
    
    for step in bow_steps:
        motion_type = MotionType(step["type"])
        sequencer.add_step(
            motion_type,
            {k: v for k, v in step.items() if k != "type" and k != "duration"},
            duration=step.get("duration", 1.0),
            name=f"bow_step_{len(sequencer.sequence)}"
        )
    
    print(f"✓ Secuencia compleja creada: {len(sequencer.sequence)} pasos")
    
    # Obtener información
    info = sequencer.get_sequence_info()
    print(f"  Duración total estimada: {info['total_duration']:.1f}s")
    
    # Ejecutar
    results = await sequencer.execute_sequence()
    print(f"✓ Secuencia ejecutada: {results['steps_executed']}/{results['steps_total']} pasos")
    
    if results["errors"]:
        print(f"  Errores: {len(results['errors'])}")
        for error in results["errors"]:
            print(f"    - {error}")
    
    await robot.disconnect()


async def example_conditional_sequence():
    """
    Ejemplo 6: Secuencia con condiciones.
    """
    print("\n=== Ejemplo 6: Secuencia con Condiciones ===")
    
    robot = HumanoidDevinDriver(robot_ip="192.168.1.100", dof=32)
    await robot.connect()
    
    sequencer = MotionSequencer(robot)
    
    # Variable de condición
    should_wave = True
    
    # Agregar paso condicional
    sequencer.add_step(
        MotionType.WAVE,
        {"hand": "right"},
        duration=2.0,
        name="conditional_wave",
        condition=lambda: should_wave
    )
    
    # Agregar paso siempre ejecutado
    sequencer.add_step(
        MotionType.STAND,
        {},
        duration=1.0,
        name="always_stand"
    )
    
    print(f"✓ Secuencia condicional creada: {len(sequencer.sequence)} pasos")
    
    # Ejecutar
    results = await sequencer.execute_sequence()
    print(f"✓ Secuencia ejecutada: {results['steps_executed']} pasos ejecutados")
    
    await robot.disconnect()


async def main():
    """
    Ejecutar todos los ejemplos de secuencias de movimiento.
    """
    print("=" * 60)
    print("Ejemplos de Secuencias de Movimiento")
    print("=" * 60)
    
    examples = [
        example_simple_sequence,
        example_joint_sequence,
        example_pose_sequence,
        example_gesture_library,
        example_complex_sequence,
        example_conditional_sequence
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

