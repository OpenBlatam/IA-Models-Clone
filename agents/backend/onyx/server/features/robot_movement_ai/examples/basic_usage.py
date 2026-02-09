"""
Ejemplos básicos de uso de Robot Movement AI v2.0
"""

import asyncio
from core.architecture.di_setup import setup_di, resolve_service
from core.architecture.application_layer import MoveRobotCommand, MoveRobotUseCase
from core.architecture.logging_config import setup_logging, get_logger
from sdk.python.robot_client import RobotClient


async def example_basic_movement():
    """Ejemplo básico de movimiento de robot"""
    print("=== Ejemplo: Movimiento Básico ===\n")
    
    # Setup logging
    logger = setup_logging(log_level="INFO", enable_colors=True)
    
    # Setup DI
    setup_di()
    
    # Resolver use case
    move_use_case = resolve_service(MoveRobotUseCase)
    
    # Crear comando
    command = MoveRobotCommand(
        robot_id="robot-1",
        target_x=0.5,
        target_y=0.3,
        target_z=0.2
    )
    
    # Ejecutar
    try:
        result = await move_use_case.execute(command)
        logger.info(f"Movimiento exitoso: {result}")
    except Exception as e:
        logger.error(f"Error en movimiento: {e}")


async def example_api_client():
    """Ejemplo de uso del SDK de API"""
    print("\n=== Ejemplo: Cliente API ===\n")
    
    async with RobotClient(base_url="http://localhost:8010") as client:
        # Health check
        health = await client.health_check()
        print(f"API Status: {health['status']}")
        
        # Listar robots
        robots = await client.list_robots()
        print(f"Robots disponibles: {len(robots)}")
        
        # Mover robot si existe
        if robots:
            robot = robots[0]
            result = await client.move_robot(
                robot.id,
                target_x=0.5,
                target_y=0.3,
                target_z=0.2
            )
            print(f"Resultado del movimiento: {result.status}")


def example_logging():
    """Ejemplo de logging avanzado"""
    print("\n=== Ejemplo: Logging ===\n")
    
    logger = setup_logging(log_level="DEBUG", enable_colors=True)
    
    logger.debug("Mensaje de debug")
    logger.info("Mensaje de información")
    logger.warning("Mensaje de advertencia")
    logger.error("Mensaje de error")
    
    # Con contexto
    from core.architecture.logging_config import LoggingContext
    
    with LoggingContext(logger, robot_id="robot-1", operation="move"):
        logger.info("Movimiento iniciado")
        logger.info("Movimiento completado")


def example_performance():
    """Ejemplo de uso de performance monitoring"""
    print("\n=== Ejemplo: Performance Monitoring ===\n")
    
    from core.architecture.performance import timed, get_performance_monitor
    
    @timed
    def slow_function():
        import time
        time.sleep(0.1)
        return "done"
    
    # Ejecutar función varias veces
    for _ in range(10):
        slow_function()
    
    # Ver estadísticas
    monitor = get_performance_monitor()
    stats = monitor.get_stats("slow_function")
    print(f"Estadísticas de slow_function:")
    print(f"  Promedio: {stats['avg']:.4f}s")
    print(f"  P95: {stats['p95']:.4f}s")


def example_caching():
    """Ejemplo de uso de cache"""
    print("\n=== Ejemplo: Caching ===\n")
    
    from core.architecture.performance import cached, get_performance_cache
    from datetime import timedelta
    
    @cached(ttl=timedelta(minutes=5))
    def expensive_operation(key: str):
        print(f"Ejecutando operación costosa para {key}")
        return f"result-{key}"
    
    # Primera llamada - ejecuta función
    result1 = expensive_operation("test")
    print(f"Resultado 1: {result1}")
    
    # Segunda llamada - usa cache
    result2 = expensive_operation("test")
    print(f"Resultado 2: {result2}")
    
    # Ver estadísticas de cache
    cache = get_performance_cache()
    stats = cache.get_stats()
    print(f"\nEstadísticas de cache:")
    print(f"  Hit Rate: {stats['hit_rate']}")
    print(f"  Tamaño: {stats['size']}/{stats['max_size']}")


async def main():
    """Ejecutar todos los ejemplos"""
    print("=" * 60)
    print("Ejemplos de Robot Movement AI v2.0")
    print("=" * 60)
    
    # Ejemplos síncronos
    example_logging()
    example_performance()
    example_caching()
    
    # Ejemplos async
    await example_basic_movement()
    await example_api_client()


if __name__ == "__main__":
    asyncio.run(main())
