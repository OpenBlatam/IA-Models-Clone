# Ejemplos de Integración - Nueva Arquitectura
## Robot Movement AI v2.0

---

## 🚀 Ejemplos Prácticos de Uso

### Ejemplo 1: Setup Completo con FastAPI

```python
from fastapi import FastAPI, Depends
from core.architecture.di_setup import setup_dependency_injection, create_dependency
from core.architecture.application_layer import MoveRobotUseCase, MoveRobotCommand

app = FastAPI()

@app.on_event("startup")
async def startup():
    await setup_dependency_injection({
        'repository_type': 'in_memory'
    })

@app.post("/api/v2/robots/{robot_id}/move")
async def move_robot(
    robot_id: str,
    target_x: float,
    target_y: float,
    target_z: float,
    use_case: MoveRobotUseCase = Depends(create_dependency(MoveRobotUseCase))
):
    command = MoveRobotCommand(
        robot_id=robot_id,
        target_x=target_x,
        target_y=target_y,
        target_z=target_z
    )
    result = await use_case.execute(command)
    return result
```

### Ejemplo 2: Integración con Código Existente

```python
# Mantener compatibilidad con código existente
from core.robot.movement_engine import RobotMovementEngine
from core.architecture.application_layer import MoveRobotUseCase

class RobotMovementEngineWrapper:
    """Wrapper que integra nueva arquitectura con código existente."""
    
    def __init__(self):
        self._use_case = None
    
    async def _get_use_case(self):
        if self._use_case is None:
            from core.architecture.di_setup import get_move_robot_use_case
            self._use_case = await get_move_robot_use_case()
        return self._use_case
    
    async def move_to_position(self, robot_id: str, x: float, y: float, z: float):
        """Método compatible con código existente."""
        use_case = await self._get_use_case()
        command = MoveRobotCommand(
            robot_id=robot_id,
            target_x=x,
            target_y=y,
            target_z=z
        )
        result = await use_case.execute(command)
        return result
```

### Ejemplo 3: Chat Controller con Nueva Arquitectura

```python
from core.architecture.di_setup import get_move_robot_use_case
from core.architecture.application_layer import MoveRobotCommand

class ChatRobotControllerV2:
    """Chat controller usando nueva arquitectura."""
    
    async def process_message(self, message: str, robot_id: str):
        """Procesar mensaje de chat."""
        # Parsear comando
        if message.startswith("move to"):
            coords = self._parse_move_command(message)
            
            # Usar use case
            use_case = await get_move_robot_use_case()
            command = MoveRobotCommand(
                robot_id=robot_id,
                target_x=coords['x'],
                target_y=coords['y'],
                target_z=coords['z']
            )
            result = await use_case.execute(command)
            return f"Moving robot to ({coords['x']}, {coords['y']}, {coords['z']})"
        
        elif message == "status":
            from core.architecture.di_setup import get_robot_status_use_case
            from core.architecture.application_layer import GetRobotStatusQuery
            
            use_case = await get_robot_status_use_case()
            query = GetRobotStatusQuery(robot_id=robot_id)
            status = await use_case.execute(query)
            return f"Robot {robot_id}: {'Connected' if status.is_connected else 'Disconnected'}"
    
    def _parse_move_command(self, message: str) -> dict:
        """Parsear comando de movimiento."""
        # Implementación de parsing
        import re
        match = re.search(r'\(([\d.]+),\s*([\d.]+),\s*([\d.]+)\)', message)
        if match:
            return {
                'x': float(match.group(1)),
                'y': float(match.group(2)),
                'z': float(match.group(3))
            }
        raise ValueError("Invalid move command format")
```

### Ejemplo 4: Con Circuit Breaker

```python
from core.architecture.circuit_breaker import circuit_breaker, CircuitBreakerConfig

@circuit_breaker(
    name="robot_driver",
    config=CircuitBreakerConfig(
        failure_threshold=5,
        recovery_timeout=60.0,
        expected_exception=ConnectionError
    )
)
async def call_robot_driver(robot_id: str, command: str):
    """Llamar driver físico con protección de circuit breaker."""
    # Tu código aquí
    pass
```

### Ejemplo 5: Manejo de Errores Mejorado

```python
from core.architecture.error_handling import (
    handle_error,
    ErrorContext,
    ApplicationError,
    ErrorCode
)

async def safe_operation(robot_id: str):
    """Operación con manejo robusto de errores."""
    context = ErrorContext(
        operation="safe_operation",
        robot_id=robot_id,
        request_id="req-123"
    )
    
    try:
        # Tu código aquí
        pass
    except ApplicationError as e:
        # Errores de aplicación ya tienen contexto
        raise
    except Exception as e:
        # Otros errores - agregar contexto
        error_details = handle_error(e, context)
        raise ApplicationError(
            f"Error en operación: {error_details.message}",
            ErrorCode.APPLICATION_VALIDATION_ERROR,
            context=context,
            original_error=e
        )
```

---

## 🔄 Patrones de Migración

### Patrón 1: Feature Flag

```python
USE_NEW_ARCHITECTURE = os.getenv("USE_NEW_ARCHITECTURE", "false").lower() == "true"

if USE_NEW_ARCHITECTURE:
    # Nueva implementación
    use_case = await get_move_robot_use_case()
    result = await use_case.execute(command)
else:
    # Implementación antigua
    engine = RobotMovementEngine()
    result = await engine.move_to_position(...)
```

### Patrón 2: Adapter

```python
class LegacyAdapter:
    """Adapter para código legacy."""
    
    def __init__(self, use_case: MoveRobotUseCase):
        self.use_case = use_case
    
    async def move_to(self, x: float, y: float, z: float):
        """Interfaz legacy compatible."""
        command = MoveRobotCommand(
            robot_id=self.robot_id,
            target_x=x,
            target_y=y,
            target_z=z
        )
        return await self.use_case.execute(command)
```

### Patrón 3: Gradual Migration

```python
# Endpoint híbrido - usa ambos sistemas
@app.post("/api/v1/move/to")
async def move_to_hybrid(request: MoveToRequest):
    """Endpoint que usa ambos sistemas durante migración."""
    try:
        # Intentar nueva arquitectura primero
        use_case = await get_move_robot_use_case()
        command = MoveRobotCommand(...)
        result = await use_case.execute(command)
        return result
    except Exception:
        # Fallback a implementación antigua
        engine = RobotMovementEngine()
        return await engine.move_to_position(...)
```

---

## 📝 Notas Importantes

1. **Compatibilidad**: La nueva arquitectura puede coexistir con código existente
2. **Migración Gradual**: No es necesario migrar todo de una vez
3. **Testing**: Siempre probar después de cada cambio
4. **Documentación**: Documentar cambios durante migración

---

**Versión**: 1.0.0  
**Fecha**: 2025-01-27




