# Guía de Ejemplos - Robot Movement AI v2.0
## Ejemplos Prácticos de Uso

---

## 📚 Ejemplos Disponibles

### 1. Uso Básico

**Archivo**: `examples/basic_usage.py`

Ejemplos de:
- Movimiento básico de robot
- Uso del SDK de API
- Logging avanzado
- Performance monitoring
- Caching

**Ejecutar**:
```bash
python examples/basic_usage.py
```

---

### 2. SDK Python

```python
from sdk.python.robot_client import RobotClient

async with RobotClient() as client:
    # Health check
    health = await client.health_check()
    
    # Listar robots
    robots = await client.list_robots()
    
    # Mover robot
    result = await client.move_robot("robot-1", 0.5, 0.3, 0.2)
```

---

### 3. Use Cases

```python
from core.architecture.di_setup import setup_di, resolve_service
from core.architecture.application_layer import MoveRobotCommand, MoveRobotUseCase

setup_di()
move_use_case = resolve_service(MoveRobotUseCase)

command = MoveRobotCommand(
    robot_id="robot-1",
    target_x=0.5,
    target_y=0.3,
    target_z=0.2
)

result = await move_use_case.execute(command)
```

---

### 4. Logging

```python
from core.architecture.logging_config import setup_logging, LoggingContext

logger = setup_logging(log_level="INFO", enable_colors=True)

# Logging básico
logger.info("Mensaje de información")

# Con contexto
with LoggingContext(logger, robot_id="robot-1", operation="move"):
    logger.info("Movimiento iniciado")
```

---

### 5. Performance

```python
from core.architecture.performance import timed, cached, get_performance_monitor
from datetime import timedelta

# Medir tiempo
@timed
async def my_function():
    # código aquí
    pass

# Cachear resultados
@cached(ttl=timedelta(minutes=5))
async def expensive_operation():
    # código aquí
    pass

# Ver estadísticas
monitor = get_performance_monitor()
stats = monitor.get_stats("my_function")
```

---

### 6. Configuración

```python
from core.architecture.config import load_config, get_config

# Cargar configuración
config = load_config()

# Obtener configuración actual
config = get_config()

# Acceder a valores
print(config.api_port)
print(config.database_url)
```

---

### 7. Migraciones

```bash
# Aplicar migraciones
python scripts/migrate.py migrate

# Ver estado
python scripts/migrate.py status

# Revertir migración
python scripts/migrate.py rollback --version 001
```

---

## 🚀 Ejemplos por Caso de Uso

### Caso 1: Control Básico de Robot

```python
from sdk.python.robot_client import RobotClient

async def control_robot():
    async with RobotClient() as client:
        # Obtener estado
        robot = await client.get_robot("robot-1")
        print(f"Estado: {robot.status}")
        
        # Mover
        result = await client.move_robot("robot-1", 0.5, 0.3, 0.2)
        print(f"Movimiento: {result.status}")
```

### Caso 2: Monitoreo Continuo

```python
import asyncio
from sdk.python.robot_client import RobotClient

async def monitor_robot():
    async with RobotClient() as client:
        while True:
            health = await client.health_check()
            metrics = await client.get_metrics()
            print(f"Health: {health['status']}")
            await asyncio.sleep(5)
```

### Caso 3: Batch Operations

```python
from sdk.python.robot_client import RobotClient

async def batch_movements():
    async with RobotClient() as client:
        movements = [
            (0.5, 0.3, 0.2),
            (0.6, 0.4, 0.3),
            (0.7, 0.5, 0.4),
        ]
        
        for x, y, z in movements:
            result = await client.move_robot("robot-1", x, y, z)
            print(f"Movido a ({x}, {y}, {z}): {result.status}")
```

---

## 📖 Más Información

- [START_HERE.md](./START_HERE.md) - Guía de inicio
- [MASTER_ARCHITECTURE_GUIDE.md](./MASTER_ARCHITECTURE_GUIDE.md) - Arquitectura completa
- [API Documentation](./api/openapi_config.py) - Documentación de API

---

**Versión**: 1.0.0  
**Última actualización**: 2025-01-27




