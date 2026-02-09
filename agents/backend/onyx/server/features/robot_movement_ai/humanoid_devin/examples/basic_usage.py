"""
Ejemplos de uso básico del Humanoid Devin Robot (Optimizado)
============================================================

Ejemplos profesionales de cómo usar el robot humanoide con todas sus capacidades.
"""

import asyncio
import numpy as np
from typing import List

# Importar el driver y utilidades
from ..drivers.humanoid_devin_driver import HumanoidDevinDriver, RobotType
from ..utils import (
    normalize_quaternion,
    interpolate_joint_positions,
    smooth_trajectory,
    validate_pose
)
from ..exceptions import HumanoidRobotError, RobotConnectionError


async def example_basic_connection():
    """
    Ejemplo 1: Conexión básica al robot.
    """
    print("=== Ejemplo 1: Conexión Básica ===")
    
    # Crear driver
    robot = HumanoidDevinDriver(
        robot_ip="192.168.1.100",
        robot_port=30001,
        dof=32,
        robot_type=RobotType.GENERIC,
        use_ml=True,
        use_diffusion=True
    )
    
    try:
        # Conectar
        await robot.connect()
        print("✓ Robot conectado exitosamente")
        
        # Obtener estado
        status = await robot.get_status()
        print(f"✓ Estado del robot: {status['robot_type']}")
        print(f"✓ DOF: {status['dof']}")
        print(f"✓ ML habilitado: {status['integrations']['deep_learning']['ml_enabled']}")
        
        # Desconectar
        await robot.disconnect()
        print("✓ Robot desconectado")
        
    except RobotConnectionError as e:
        print(f"✗ Error de conexión: {e}")
    except HumanoidRobotError as e:
        print(f"✗ Error del robot: {e}")


async def example_joint_control():
    """
    Ejemplo 2: Control de articulaciones con validación.
    """
    print("\n=== Ejemplo 2: Control de Articulaciones ===")
    
    robot = HumanoidDevinDriver(
        robot_ip="192.168.1.100",
        dof=32
    )
    
    try:
        await robot.connect()
        
        # Obtener posiciones actuales
        current_positions = await robot.get_joint_positions()
        print(f"✓ Posiciones actuales: {len(current_positions)} articulaciones")
        
        # Crear posiciones objetivo (ejemplo: mover brazos)
        target_positions = current_positions.copy()
        target_positions[3] = 0.5   # left_shoulder_pitch
        target_positions[10] = -0.5  # right_shoulder_pitch
        
        # Establecer posiciones
        success = await robot.set_joint_positions(target_positions)
        print(f"✓ Posiciones establecidas: {success}")
        
        await robot.disconnect()
        
    except Exception as e:
        print(f"✗ Error: {e}")


async def example_smooth_movement():
    """
    Ejemplo 3: Movimiento suave usando interpolación y suavizado.
    """
    print("\n=== Ejemplo 3: Movimiento Suave ===")
    
    robot = HumanoidDevinDriver(
        robot_ip="192.168.1.100",
        dof=32,
        use_ml=True
    )
    
    try:
        await robot.connect()
        
        # Posiciones iniciales y finales
        start_positions = await robot.get_joint_positions()
        end_positions = start_positions.copy()
        end_positions[0] = 0.3   # head_yaw
        end_positions[1] = -0.2  # head_pitch
        
        # Interpolar trayectoria
        trajectory = interpolate_joint_positions(
            start=start_positions,
            end=end_positions,
            num_steps=20
        )
        print(f"✓ Trayectoria interpolada: {len(trajectory)} pasos")
        
        # Suavizar trayectoria
        smooth_traj = smooth_trajectory(trajectory, window_size=5)
        print(f"✓ Trayectoria suavizada")
        
        # Ejecutar trayectoria paso a paso
        for i, positions in enumerate(smooth_traj):
            await robot.set_joint_positions(positions)
            if i % 5 == 0:
                print(f"  Paso {i}/{len(smooth_traj)}")
        
        print("✓ Movimiento suave completado")
        await robot.disconnect()
        
    except Exception as e:
        print(f"✗ Error: {e}")


async def example_pose_control():
    """
    Ejemplo 4: Control de pose del efector final.
    """
    print("\n=== Ejemplo 4: Control de Pose ===")
    
    robot = HumanoidDevinDriver(
        robot_ip="192.168.1.100",
        use_moveit2=True
    )
    
    try:
        await robot.connect()
        
        # Posición objetivo
        target_position = np.array([0.3, -0.2, 1.0], dtype=np.float64)
        target_orientation = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        
        # Validar pose
        valid_pos, valid_ori = validate_pose(target_position, target_orientation)
        print(f"✓ Pose validada: posición {valid_pos}, orientación {valid_ori}")
        
        # Mover a pose
        success = await robot.move_to_pose(
            position=valid_pos,
            orientation=valid_ori,
            hand="right"
        )
        print(f"✓ Movimiento a pose: {success}")
        
        await robot.disconnect()
        
    except Exception as e:
        print(f"✗ Error: {e}")


async def example_walking():
    """
    Ejemplo 5: Caminar usando Nav2 o cmd_vel.
    """
    print("\n=== Ejemplo 5: Caminar ===")
    
    robot = HumanoidDevinDriver(
        robot_ip="192.168.1.100",
        use_nav2=True,
        use_diffusion=True
    )
    
    try:
        await robot.connect()
        
        # Caminar hacia adelante
        success = await robot.walk(
            direction="forward",
            distance=1.0,
            speed=0.5
        )
        print(f"✓ Caminar hacia adelante: {success}")
        
        # Esperar un momento
        await asyncio.sleep(2)
        
        # Girar a la derecha
        success = await robot.walk(
            direction="turn_right",
            distance=0.0,
            speed=0.3
        )
        print(f"✓ Girar a la derecha: {success}")
        
        await robot.disconnect()
        
    except Exception as e:
        print(f"✗ Error: {e}")


async def example_deep_learning():
    """
    Ejemplo 6: Usar modelos de Deep Learning para movimiento.
    """
    print("\n=== Ejemplo 6: Deep Learning ===")
    
    robot = HumanoidDevinDriver(
        robot_ip="192.168.1.100",
        use_ml=True,
        use_diffusion=True
    )
    
    try:
        await robot.connect()
        
        # Verificar estado de modelos DL
        status = await robot.get_status()
        dl_status = status['integrations']['deep_learning']
        print(f"✓ ML habilitado: {dl_status['ml_enabled']}")
        print(f"✓ Diffusion habilitado: {dl_status['diffusion_enabled']}")
        print(f"✓ Transformer disponible: {dl_status['transformer_model']}")
        
        # Generar trayectoria suave usando difusión
        start_pos = np.array([0.0, 0.0, 0.0])
        end_pos = np.array([0.5, 0.0, 0.0])
        
        trajectory = await robot.generate_smooth_trajectory(
            start_position=start_pos,
            end_position=end_pos,
            num_steps=50
        )
        
        if trajectory is not None:
            print(f"✓ Trayectoria generada con difusión: {len(trajectory)} pasos")
        else:
            print("⚠ Difusión no disponible, usando método estándar")
        
        # Predecir movimiento de articulaciones
        current_joints = np.array(await robot.get_joint_positions())
        target_joints = current_joints + 0.1
        
        predicted = await robot.predict_joint_motion(
            current_joints=current_joints,
            target_joints=target_joints
        )
        
        if predicted is not None:
            print(f"✓ Movimiento predicho usando Transformer")
        else:
            print("⚠ Transformer no disponible")
        
        await robot.disconnect()
        
    except Exception as e:
        print(f"✗ Error: {e}")


async def example_vision_integration():
    """
    Ejemplo 7: Integración con visión.
    """
    print("\n=== Ejemplo 7: Integración con Visión ===")
    
    robot = HumanoidDevinDriver(
        robot_ip="192.168.1.100",
        use_opencv=True
    )
    
    try:
        await robot.connect()
        
        if robot.vision and robot.vision.available:
            # Simular imagen (en producción vendría de cámara)
            dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
            
            # Detectar caras
            faces = robot.vision.detect_faces(dummy_image)
            print(f"✓ Caras detectadas: {len(faces)}")
            
            # Obtener información de imagen
            info = robot.vision.get_image_info(dummy_image)
            print(f"✓ Información de imagen: {info['width']}x{info['height']}")
            
            # Detectar bordes
            edges = robot.vision.detect_edges(dummy_image)
            edge_pixels = np.sum(edges > 0)
            print(f"✓ Bordes detectados: {edge_pixels} píxeles")
        else:
            print("⚠ Visión no disponible")
        
        await robot.disconnect()
        
    except Exception as e:
        print(f"✗ Error: {e}")


async def example_error_handling():
    """
    Ejemplo 8: Manejo robusto de errores.
    """
    print("\n=== Ejemplo 8: Manejo de Errores ===")
    
    from ..exceptions import (
        RobotConnectionError,
        RobotControlError,
        ValidationError
    )
    
    robot = HumanoidDevinDriver(
        robot_ip="192.168.1.100",
        dof=32
    )
    
    try:
        # Intentar conectar con IP inválida
        robot_invalid = HumanoidDevinDriver(
            robot_ip="",  # IP vacía - debería fallar
            dof=32
        )
    except ValidationError as e:
        print(f"✓ Validación capturada correctamente: {e}")
    
    try:
        await robot.connect()
        
        # Intentar establecer posiciones inválidas
        try:
            invalid_positions = [0.0] * 20  # Solo 20 en lugar de 32
            await robot.set_joint_positions(invalid_positions)
        except ValidationError as e:
            print(f"✓ Error de validación capturado: {e}")
        
        await robot.disconnect()
        
    except RobotConnectionError as e:
        print(f"✓ Error de conexión manejado: {e}")
    except Exception as e:
        print(f"✗ Error inesperado: {e}")


async def main():
    """
    Ejecutar todos los ejemplos.
    """
    print("=" * 60)
    print("Ejemplos de Uso - Humanoid Devin Robot")
    print("=" * 60)
    
    examples = [
        example_basic_connection,
        example_joint_control,
        example_smooth_movement,
        example_pose_control,
        example_walking,
        example_deep_learning,
        example_vision_integration,
        example_error_handling
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

