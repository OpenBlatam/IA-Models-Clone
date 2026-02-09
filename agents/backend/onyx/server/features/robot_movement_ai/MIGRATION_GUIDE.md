# Guía de Migración - Integración con Código Existente
## Robot Movement AI v2.0 - Migración Gradual

---

## 🎯 Objetivo

Esta guía te ayudará a integrar gradualmente la nueva arquitectura mejorada con el código existente del sistema Robot Movement AI.

**Estrategia**: Migración gradual, módulo por módulo, sin romper funcionalidad existente.

---

## 📋 Estrategia de Migración

### Fase 1: Setup y Configuración ✅

**Estado**: Completado

- [x] Arquitectura nueva implementada
- [x] Dependency Injection configurado
- [x] Repositorios creados
- [x] Tests escritos

### Fase 2: Integración con API (Actual)

**Objetivo**: Refactorizar `robot_api.py` para usar use cases

**Pasos**:

1. **Mantener compatibilidad hacia atrás**
2. **Agregar nuevos endpoints usando use cases**
3. **Migrar endpoints existentes gradualmente**
4. **Deprecar endpoints antiguos**

### Fase 3: Integración con Chat Controller

**Objetivo**: Refactorizar `chat_controller.py` para usar use cases

### Fase 4: Integración con Movement Engine

**Objetivo**: Refactorizar `movement_engine.py` para usar entidades de dominio

---

## 🔄 Fase 2: Integración con API

### Paso 1: Setup de DI en API

**Archivo**: `api/robot_api.py`

```python
from fastapi import FastAPI, Depends
from core.architecture.di_setup import (
    setup_dependency_injection,
    create_dependency,
    get_move_robot_use_case,
    get_robot_status_use_case
)
from core.architecture.application_layer import (
    MoveRobotUseCase,
    GetRobotStatusUseCase,
    MoveRobotCommand,
    GetRobotStatusQuery
)

app = FastAPI()

@app.on_event("startup")
async def startup():
    """Configurar DI al iniciar."""
    await setup_dependency_injection({
        'repository_type': 'in_memory'  # Cambiar a 'sql' en producción
    })
```

### Paso 2: Refactorizar Endpoint Existente

**Antes**:
```python
@app.post("/api/v1/move/to")
async def move_to(request: MoveToRequest):
    # Lógica directa con RobotMovementEngine
    engine = RobotMovementEngine()
    await engine.move_to_position(...)
```

**Después**:
```python
@app.post("/api/v1/move/to")
async def move_to(
    request: MoveToRequest,
    use_case: MoveRobotUseCase = Depends(create_dependency(MoveRobotUseCase))
):
    # Usar use case
    command = MoveRobotCommand(
        robot_id=request.robot_id,
        target_x=request.x,
        target_y=request.y,
        target_z=request.z
    )
    result = await use_case.execute(command)
    return result
```

### Paso 3: Agregar Nuevos Endpoints

```python
@app.get("/api/v1/robots/{robot_id}/status")
async def get_robot_status(
    robot_id: str,
    use_case: GetRobotStatusUseCase = Depends(create_dependency(GetRobotStatusUseCase))
):
    query = GetRobotStatusQuery(robot_id=robot_id)
    status = await use_case.execute(query)
    return status
```

### Paso 4: Mantener Compatibilidad

```python
# Endpoint antiguo (deprecated pero funcional)
@app.post("/api/v1/move/to/legacy")
async def move_to_legacy(request: MoveToRequest):
    """Endpoint legacy - usar /api/v1/move/to en su lugar."""
    # Implementación antigua
    pass

# Endpoint nuevo (recomendado)
@app.post("/api/v1/move/to")
async def move_to(request: MoveToRequest):
    # Nueva implementación con use cases
    pass
```

---

## 💬 Fase 3: Integración con Chat Controller

### Refactorizar Chat Controller

**Archivo**: `chat/chat_controller.py`

**Antes**:
```python
class ChatRobotController:
    def __init__(self):
        self.engine = RobotMovementEngine()
    
    async def process_message(self, message: str):
        # Lógica directa
        if "move to" in message:
            await self.engine.move_to_position(...)
```

**Después**:
```python
from core.architecture.di_setup import get_move_robot_use_case
from core.architecture.application_layer import MoveRobotCommand

class ChatRobotController:
    def __init__(self):
        # Use cases se resuelven cuando se necesitan
        pass
    
    async def process_message(self, message: str):
        # Parsear mensaje
        if "move to" in message:
            # Extraer coordenadas del mensaje
            coords = self._parse_coordinates(message)
            
            # Usar use case
            use_case = await get_move_robot_use_case()
            command = MoveRobotCommand(
                robot_id=self.current_robot_id,
                target_x=coords['x'],
                target_y=coords['y'],
                target_z=coords['z']
            )
            result = await use_case.execute(command)
            return f"Moving robot to {coords}"
```

---

## ⚙️ Fase 4: Integración con Movement Engine

### Refactorizar Movement Engine

**Archivo**: `core/robot/movement_engine.py`

**Estrategia**: Crear wrapper que use entidades de dominio

```python
from core.architecture.domain_improved import RobotMovement, Position
from core.architecture.application_layer import IMovementRepository

class RobotMovementEngine:
    """Wrapper que usa entidades de dominio."""
    
    def __init__(self, movement_repo: IMovementRepository):
        self.movement_repo = movement_repo
    
    async def move_to_position(self, robot_id: str, x: float, y: float, z: float):
        # Crear entidad de dominio
        target_pos = Position(x=x, y=y, z=z)
        movement = RobotMovement(
            robot_id=robot_id,
            target_position=target_pos
        )
        
        # Iniciar movimiento (lógica de dominio)
        movement.start()
        
        # Persistir
        await self.movement_repo.save(movement)
        
        # Ejecutar movimiento físico (lógica de infraestructura)
        await self._execute_physical_movement(movement)
        
        return movement
```

---

## 🔧 Configuración para Producción

### Paso 1: Configurar Repositorio SQL

```python
import sqlite3  # o asyncpg para PostgreSQL
from core.architecture.di_setup import setup_dependency_injection

# Crear conexión
db = sqlite3.connect("robots.db")

# Configurar DI con SQL
await setup_dependency_injection({
    'repository_type': 'sql',
    'db_connection': db
})
```

### Paso 2: Configurar Cache

```python
await setup_dependency_injection({
    'repository_type': 'sql_with_cache',
    'db_connection': db,
    'cache_config': {
        'ttl': 300,  # 5 minutos
        'max_size': 1000
    }
})
```

### Paso 3: Configurar Circuit Breaker

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
async def call_robot_driver():
    # Llamada a driver físico
    pass
```

---

## 📝 Checklist de Migración

### Pre-Migración

- [x] Nueva arquitectura implementada
- [x] Tests escritos y pasando
- [x] Documentación completa
- [ ] Backup de código existente
- [ ] Branch de migración creado

### Migración API

- [ ] Setup DI en `robot_api.py`
- [ ] Refactorizar endpoint `/api/v1/move/to`
- [ ] Agregar endpoint `/api/v1/robots/{id}/status`
- [ ] Agregar endpoint `/api/v1/robots/{id}/history`
- [ ] Mantener endpoints legacy funcionando
- [ ] Tests de integración para nuevos endpoints

### Migración Chat

- [ ] Refactorizar `ChatRobotController`
- [ ] Integrar con use cases
- [ ] Mantener compatibilidad con comandos existentes
- [ ] Tests para chat controller

### Migración Movement Engine

- [ ] Crear wrapper que use entidades de dominio
- [ ] Migrar lógica a entidades
- [ ] Mantener interfaz pública compatible
- [ ] Tests para movement engine

### Post-Migración

- [ ] Todos los tests pasando
- [ ] Documentación actualizada
- [ ] Endpoints legacy deprecados
- [ ] Monitoreo configurado
- [ ] Performance verificado

---

## 🚨 Manejo de Errores Durante Migración

### Estrategia de Rollback

```python
# Feature flag para nueva arquitectura
USE_NEW_ARCHITECTURE = os.getenv("USE_NEW_ARCHITECTURE", "false").lower() == "true"

@app.post("/api/v1/move/to")
async def move_to(request: MoveToRequest):
    if USE_NEW_ARCHITECTURE:
        # Nueva implementación
        use_case = await get_move_robot_use_case()
        # ...
    else:
        # Implementación antigua
        engine = RobotMovementEngine()
        # ...
```

### Monitoreo

```python
from core.architecture.error_handling import handle_error, ErrorContext

try:
    result = await use_case.execute(command)
except Exception as e:
    context = ErrorContext(
        operation="move_robot",
        robot_id=command.robot_id,
        request_id=request_id
    )
    error_details = handle_error(e, context)
    # Log y manejo apropiado
```

---

## 📊 Métricas de Migración

### Tracking

- **Endpoints migrados**: X/Y
- **Tests pasando**: X/Y
- **Performance**: Comparar antes/después
- **Errores**: Monitorear durante migración

### KPIs

- **Tiempo de respuesta**: Debe mantenerse o mejorar
- **Tasa de errores**: Debe mantenerse o mejorar
- **Cobertura de tests**: Debe aumentar
- **Complejidad**: Debe disminuir

---

## 🎯 Ejemplo Completo de Migración

### Endpoint Completo Migrado

```python
from fastapi import FastAPI, Depends, HTTPException
from core.architecture.di_setup import (
    setup_dependency_injection,
    create_dependency
)
from core.architecture.application_layer import (
    MoveRobotUseCase,
    GetRobotStatusUseCase,
    MoveRobotCommand,
    GetRobotStatusQuery,
    ApplicationError,
    ErrorCode
)
from core.architecture.error_handling import handle_error, ErrorContext

app = FastAPI()

@app.on_event("startup")
async def startup():
    await setup_dependency_injection({
        'repository_type': os.getenv('REPOSITORY_TYPE', 'in_memory')
    })

@app.post("/api/v1/robots/{robot_id}/move")
async def move_robot(
    robot_id: str,
    target_x: float,
    target_y: float,
    target_z: float,
    use_case: MoveRobotUseCase = Depends(create_dependency(MoveRobotUseCase))
):
    """Mover robot usando nueva arquitectura."""
    try:
        command = MoveRobotCommand(
            robot_id=robot_id,
            target_x=target_x,
            target_y=target_y,
            target_z=target_z
        )
        result = await use_case.execute(command)
        return {
            "success": True,
            "movement_id": result.movement_id,
            "status": result.status
        }
    except ApplicationError as e:
        if e.code == ErrorCode.APPLICATION_NOT_FOUND:
            raise HTTPException(status_code=404, detail=str(e))
        elif e.code == ErrorCode.ROBOT_NOT_CONNECTED:
            raise HTTPException(status_code=400, detail=str(e))
        else:
            raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        context = ErrorContext(
            operation="move_robot",
            robot_id=robot_id
        )
        error_details = handle_error(e, context)
        raise HTTPException(
            status_code=500,
            detail=error_details.message
        )

@app.get("/api/v1/robots/{robot_id}/status")
async def get_robot_status(
    robot_id: str,
    use_case: GetRobotStatusUseCase = Depends(create_dependency(GetRobotStatusUseCase))
):
    """Obtener estado del robot."""
    try:
        query = GetRobotStatusQuery(robot_id=robot_id)
        status = await use_case.execute(query)
        return status
    except ApplicationError as e:
        if e.code == ErrorCode.APPLICATION_NOT_FOUND:
            raise HTTPException(status_code=404, detail=str(e))
        raise HTTPException(status_code=500, detail=str(e))
```

---

## 🔍 Testing Durante Migración

### Tests de Compatibilidad

```python
@pytest.mark.asyncio
async def test_backward_compatibility():
    """Verificar que endpoints antiguos siguen funcionando."""
    # Test implementación antigua
    pass

@pytest.mark.asyncio
async def test_new_endpoints():
    """Verificar que nuevos endpoints funcionan."""
    # Test nueva implementación
    pass
```

### Tests de Integración

```python
@pytest.mark.asyncio
async def test_end_to_end_migration():
    """Test completo de migración."""
    # Setup
    await setup_dependency_injection({'repository_type': 'in_memory'})
    
    # Crear robot
    # Mover robot usando nuevo endpoint
    # Verificar estado usando nuevo endpoint
    # Comparar con implementación antigua
    pass
```

---

## 📚 Recursos Adicionales

### Documentación

- [Guía Maestra](./MASTER_ARCHITECTURE_GUIDE.md)
- [Guía de DI](./core/architecture/DI_INTEGRATION_GUIDE.md)
- [Guía de Repositorios](./core/architecture/REPOSITORIES_GUIDE.md)

### Ejemplos

- Ver `core/architecture/di_integration_example.py`
- Ver tests en `tests/`

---

## ✅ Checklist Final

### Antes de Empezar

- [ ] Leer esta guía completa
- [ ] Entender nueva arquitectura
- [ ] Backup de código existente
- [ ] Branch de migración creado

### Durante Migración

- [ ] Migrar un endpoint a la vez
- [ ] Tests pasando después de cada cambio
- [ ] Documentar cambios
- [ ] Code review

### Después de Migración

- [ ] Todos los tests pasando
- [ ] Performance verificado
- [ ] Documentación actualizada
- [ ] Endpoints legacy deprecados
- [ ] Monitoreo configurado

---

## 🎯 Conclusión

La migración debe ser **gradual y segura**. La nueva arquitectura está diseñada para coexistir con el código existente, permitiendo migración módulo por módulo sin romper funcionalidad.

**Principio clave**: Mantener compatibilidad hacia atrás durante la transición.

---

**Versión**: 1.0.0  
**Fecha**: 2025-01-27  
**Estado**: Listo para usar




