"""
Ejemplo de Uso de Configuración - Humanoid Devin Robot
=======================================================

Ejemplo de cómo usar el sistema de configuración.
"""

import asyncio
from pathlib import Path

from ..config.config_loader import ConfigLoader, load_config
from ..drivers.humanoid_devin_driver import HumanoidDevinDriver
from ..helpers.performance_monitor import PerformanceMonitor


async def example_load_config_from_file():
    """
    Ejemplo 1: Cargar configuración desde archivo.
    """
    print("=== Ejemplo 1: Cargar Configuración desde Archivo ===")
    
    # Ruta al archivo de configuración
    config_path = Path(__file__).parent.parent / "config" / "default_config.yaml"
    
    if config_path.exists():
        # Cargar configuración
        config = load_config(str(config_path))
        print(f"✓ Configuración cargada desde: {config_path}")
        
        # Obtener valores
        robot_ip = config.get("robot.ip")
        robot_port = config.get("robot.port")
        dof = config.get("robot.dof")
        
        print(f"  Robot IP: {robot_ip}")
        print(f"  Robot Port: {robot_port}")
        print(f"  DOF: {dof}")
        
        # Obtener configuración de integraciones
        integrations = config.get_integrations_config()
        print(f"  ML habilitado: {integrations.get('use_ml', False)}")
        print(f"  ROS 2 habilitado: {integrations.get('use_ros2', False)}")
    else:
        print(f"⚠ Archivo de configuración no encontrado: {config_path}")


async def example_use_default_config():
    """
    Ejemplo 2: Usar configuración por defecto.
    """
    print("\n=== Ejemplo 2: Usar Configuración por Defecto ===")
    
    # Cargar configuración por defecto
    config = load_config()
    print("✓ Configuración por defecto cargada")
    
    # Crear driver usando configuración
    driver_config = config.create_driver_config()
    print(f"✓ Configuración del driver creada")
    print(f"  Parámetros: {list(driver_config.keys())}")


async def example_create_driver_from_config():
    """
    Ejemplo 3: Crear driver desde configuración.
    """
    print("\n=== Ejemplo 3: Crear Driver desde Configuración ===")
    
    # Cargar configuración
    config = load_config()
    
    # Crear configuración del driver
    driver_config = config.create_driver_config()
    
    # Crear driver
    robot = HumanoidDevinDriver(**driver_config)
    print(f"✓ Driver creado desde configuración")
    print(f"  IP: {robot.robot_ip}")
    print(f"  Port: {robot.robot_port}")
    print(f"  DOF: {robot.dof}")
    print(f"  ML: {robot.use_ml}")
    print(f"  Diffusion: {robot.use_diffusion}")


async def example_performance_monitoring():
    """
    Ejemplo 4: Monitoreo de rendimiento.
    """
    print("\n=== Ejemplo 4: Monitoreo de Rendimiento ===")
    
    # Crear monitor
    monitor = PerformanceMonitor(enabled=True)
    
    # Simular operaciones
    operations = [
        "connect",
        "get_status",
        "get_joint_positions",
        "set_joint_positions",
        "move_to_pose"
    ]
    
    print("✓ Simulando operaciones...")
    
    for op_name in operations:
        # Iniciar operación
        monitor.start_operation(op_name)
        
        # Simular tiempo de ejecución
        await asyncio.sleep(0.1)
        
        # Finalizar operación (simular éxito)
        duration = monitor.end_operation(op_name, success=True)
        print(f"  {op_name}: {duration*1000:.2f}ms")
    
    # Obtener métricas
    metrics = monitor.get_metrics()
    print(f"\n✓ Métricas obtenidas para {len(metrics)} operaciones")
    
    # Imprimir resumen
    monitor.print_summary()


async def example_config_with_performance():
    """
    Ejemplo 5: Usar configuración con monitoreo de rendimiento.
    """
    print("\n=== Ejemplo 5: Configuración con Monitoreo ===")
    
    # Cargar configuración
    config = load_config()
    driver_config = config.create_driver_config()
    
    # Crear monitor
    monitor = PerformanceMonitor(enabled=True)
    
    # Crear driver
    robot = HumanoidDevinDriver(**driver_config)
    
    try:
        # Conectar con monitoreo
        await monitor.measure_async_operation(
            "connect",
            robot.connect
        )
        print("✓ Robot conectado (monitoreado)")
        
        # Obtener estado con monitoreo
        status = await monitor.measure_async_operation(
            "get_status",
            robot.get_status
        )
        print("✓ Estado obtenido (monitoreado)")
        
        # Obtener métricas
        connect_metrics = monitor.get_metrics("connect")
        status_metrics = monitor.get_metrics("get_status")
        
        print(f"\nMétricas de conexión:")
        print(f"  Tiempo promedio: {connect_metrics.get('average_time', 0)*1000:.2f}ms")
        print(f"  Tasa de éxito: {connect_metrics.get('success_rate', 0):.2f}%")
        
        print(f"\nMétricas de estado:")
        print(f"  Tiempo promedio: {status_metrics.get('average_time', 0)*1000:.2f}ms")
        
        await robot.disconnect()
        
    except Exception as e:
        print(f"✗ Error: {e}")


async def main():
    """
    Ejecutar todos los ejemplos de configuración.
    """
    print("=" * 60)
    print("Ejemplos de Configuración - Humanoid Devin Robot")
    print("=" * 60)
    
    examples = [
        example_load_config_from_file,
        example_use_default_config,
        example_create_driver_from_config,
        example_performance_monitoring,
        example_config_with_performance
    ]
    
    for example in examples:
        try:
            await example()
        except Exception as e:
            print(f"✗ Error en ejemplo: {e}")
        print()
    
    print("=" * 60)
    print("Ejemplos de configuración completados")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())

