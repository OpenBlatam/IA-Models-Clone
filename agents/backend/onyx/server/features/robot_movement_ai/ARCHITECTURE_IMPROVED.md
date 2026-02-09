# Arquitectura Mejorada - Robot Movement AI

## 🎯 Visión General

Esta arquitectura mejorada aplica principios de **Clean Architecture**, **Domain-Driven Design (DDD)**, y **SOLID** para crear un sistema más mantenible, escalable y testeable.

## 📐 Capas de Arquitectura

```
┌─────────────────────────────────────────┐
│   Presentation Layer (API/Controllers)  │
├─────────────────────────────────────────┤
│   Application Layer (Use Cases)         │
├─────────────────────────────────────────┤
│   Domain Layer (Entities & Business)    │
├─────────────────────────────────────────┤
│   Infrastructure Layer (Repos, External)│
└─────────────────────────────────────────┘
```

## 🏗️ Estructura de Capas

### 1. Domain Layer (Núcleo)

**Responsabilidad**: Lógica de negocio pura, entidades y reglas de dominio.

**Componentes**:
- **Entities**: Entidades de dominio con identidad única
- **Value Objects**: Objetos inmutables sin identidad
- **Domain Services**: Servicios de dominio (lógica que no pertenece a una entidad)
- **Domain Events**: Eventos del dominio
- **Repositories (Interfaces)**: Contratos de persistencia

**Principios**:
- ✅ Sin dependencias externas
- ✅ Lógica de negocio pura
- ✅ Inmutabilidad donde sea posible
- ✅ Validación de invariantes

**Ejemplo**:
```python
# Domain Entity
class RobotMovement(Entity):
    def __init__(self, robot_id: str, trajectory: Trajectory):
        self._validate_trajectory(trajectory)
        self.robot_id = robot_id
        self.trajectory = trajectory
        self.status = MovementStatus.PENDING
    
    def execute(self):
        """Lógica de negocio para ejecutar movimiento."""
        if self.status != MovementStatus.PENDING:
            raise DomainError("Movement already executed")
        # ... lógica de negocio
```

### 2. Application Layer

**Responsabilidad**: Orquestación de casos de uso, coordinación entre capas.

**Componentes**:
- **Use Cases**: Casos de uso específicos (Command/Query)
- **DTOs**: Data Transfer Objects para comunicación entre capas
- **Application Services**: Servicios de aplicación que orquestan casos de uso
- **Command/Query Handlers**: Handlers para CQRS pattern

**Principios**:
- ✅ Depende solo de Domain
- ✅ Orquesta llamadas a repositorios y servicios
- ✅ Maneja transacciones
- ✅ Valida inputs

**Ejemplo**:
```python
# Use Case
class MoveRobotUseCase:
    def __init__(
        self,
        robot_repo: IRobotRepository,
        movement_repo: IMovementRepository,
        event_bus: EventBus
    ):
        self.robot_repo = robot_repo
        self.movement_repo = movement_repo
        self.event_bus = event_bus
    
    async def execute(self, command: MoveRobotCommand) -> MovementResult:
        # 1. Validar comando
        # 2. Obtener entidad de dominio
        # 3. Ejecutar lógica de negocio
        # 4. Persistir cambios
        # 5. Emitir eventos
        pass
```

### 3. Infrastructure Layer

**Responsabilidad**: Implementaciones técnicas, acceso a datos, servicios externos.

**Componentes**:
- **Repositories (Implementaciones)**: Implementaciones concretas de repositorios
- **External Services**: Integraciones con servicios externos
- **Persistence**: Acceso a base de datos, cache, etc.
- **Messaging**: Colas de mensajes, eventos

**Principios**:
- ✅ Implementa interfaces del Domain
- ✅ Abstrae detalles técnicos
- ✅ Fácil intercambiar implementaciones

**Ejemplo**:
```python
# Repository Implementation
class SQLRobotRepository(IRobotRepository):
    def __init__(self, db: Database):
        self.db = db
    
    async def find_by_id(self, robot_id: str) -> Optional[Robot]:
        # Implementación con SQL
        pass
```

### 4. Presentation Layer

**Responsabilidad**: Interfaz con el mundo exterior, HTTP, WebSockets, CLI.

**Componentes**:
- **Controllers**: Endpoints HTTP
- **DTOs**: Request/Response models
- **Middleware**: Autenticación, logging, etc.
- **Serializers**: Serialización de datos

**Principios**:
- ✅ Depende solo de Application Layer
- ✅ Valida inputs
- ✅ Maneja errores HTTP
- ✅ Transforma DTOs

**Ejemplo**:
```python
# Controller
@router.post("/api/v1/robots/{robot_id}/move")
async def move_robot(
    robot_id: str,
    request: MoveRobotRequest,
    use_case: MoveRobotUseCase = Depends(get_move_robot_use_case)
):
    command = MoveRobotCommand(
        robot_id=robot_id,
        target=request.target
    )
    result = await use_case.execute(command)
    return MoveRobotResponse.from_domain(result)
```

## 🎨 Patrones de Diseño Aplicados

### 1. Clean Architecture
- Separación clara de capas
- Dependencias apuntan hacia adentro
- Domain es independiente

### 2. Domain-Driven Design (DDD)
- Entidades ricas con lógica de negocio
- Value Objects para conceptos del dominio
- Domain Events para comunicación desacoplada
- Aggregates para consistencia

### 3. CQRS (Command Query Responsibility Segregation)
- Separación de comandos y consultas
- Optimización independiente
- Escalabilidad mejorada

### 4. Repository Pattern
- Abstracción de persistencia
- Fácil testing con mocks
- Intercambiable (SQL, NoSQL, In-Memory)

### 5. Dependency Injection
- Inversión de dependencias
- Fácil testing
- Bajo acoplamiento

### 6. Event-Driven Architecture
- Comunicación desacoplada
- Escalabilidad
- Observabilidad

## 📦 Estructura de Directorios Mejorada

```
robot_movement_ai/
├── domain/                          # Domain Layer
│   ├── entities/
│   │   ├── robot.py
│   │   ├── movement.py
│   │   └── trajectory.py
│   ├── value_objects/
│   │   ├── position.py
│   │   ├── orientation.py
│   │   └── metrics.py
│   ├── services/
│   │   ├── trajectory_optimizer.py
│   │   └── collision_detector.py
│   ├── events/
│   │   ├── movement_started.py
│   │   └── movement_completed.py
│   └── repositories/                # Interfaces
│       ├── robot_repository.py
│       └── movement_repository.py
│
├── application/                     # Application Layer
│   ├── use_cases/
│   │   ├── move_robot.py
│   │   ├── plan_trajectory.py
│   │   └── get_robot_status.py
│   ├── commands/
│   │   ├── move_robot_command.py
│   │   └── plan_trajectory_command.py
│   ├── queries/
│   │   ├── get_robot_status_query.py
│   │   └── get_movement_history_query.py
│   ├── dtos/
│   │   ├── move_robot_dto.py
│   │   └── movement_result_dto.py
│   └── services/
│       └── robot_application_service.py
│
├── infrastructure/                  # Infrastructure Layer
│   ├── persistence/
│   │   ├── repositories/
│   │   │   ├── sql_robot_repository.py
│   │   │   └── sql_movement_repository.py
│   │   └── database.py
│   ├── external/
│   │   ├── ros_bridge.py
│   │   └── robot_drivers/
│   │       ├── kuka_driver.py
│   │       └── abb_driver.py
│   ├── messaging/
│   │   ├── event_bus.py
│   │   └── message_queue.py
│   └── cache/
│       └── redis_cache.py
│
├── presentation/                    # Presentation Layer
│   ├── api/
│   │   ├── controllers/
│   │   │   ├── robot_controller.py
│   │   │   └── movement_controller.py
│   │   ├── middleware/
│   │   │   ├── auth_middleware.py
│   │   │   └── error_handler.py
│   │   └── serializers/
│   │       └── movement_serializer.py
│   ├── websocket/
│   │   └── robot_websocket_handler.py
│   └── cli/
│       └── robot_cli.py
│
└── core/                            # Core Utilities (Shared)
    ├── architecture/                # Patrones arquitectónicos
    │   ├── dependency_injection.py
    │   ├── events.py
    │   └── validation.py
    └── config/
        └── settings.py
```

## 🔄 Flujo de Datos

### Ejemplo: Mover Robot

```
1. HTTP Request → Controller
   ↓
2. Controller → Use Case (MoveRobotUseCase)
   ↓
3. Use Case → Repository (obtener Robot entity)
   ↓
4. Use Case → Domain Service (validar movimiento)
   ↓
5. Robot Entity → execute_movement() (lógica de negocio)
   ↓
6. Use Case → Repository (persistir cambios)
   ↓
7. Use Case → Event Bus (emitir MovementCompleted event)
   ↓
8. Controller → HTTP Response
```

## ✅ Ventajas de la Nueva Arquitectura

1. **Testabilidad**: Fácil mockear dependencias
2. **Mantenibilidad**: Código organizado y claro
3. **Escalabilidad**: Fácil agregar nuevas funcionalidades
4. **Flexibilidad**: Intercambiar implementaciones fácilmente
5. **Separación de Concerns**: Cada capa tiene responsabilidad clara
6. **Type Safety**: Type hints en toda la arquitectura
7. **Domain Focus**: Lógica de negocio en el centro

## 🚀 Migración Gradual

La nueva arquitectura se puede implementar gradualmente:

1. **Fase 1**: Crear estructura de directorios
2. **Fase 2**: Migrar entidades de dominio
3. **Fase 3**: Crear casos de uso para nuevas funcionalidades
4. **Fase 4**: Refactorizar código existente gradualmente
5. **Fase 5**: Migrar completamente

## 📚 Próximos Pasos

1. Implementar entidades de dominio mejoradas
2. Crear casos de uso para operaciones principales
3. Implementar repositorios con base de datos real
4. Agregar validación robusta
5. Implementar manejo de errores centralizado
6. Crear tests unitarios y de integración
7. Documentar APIs con OpenAPI/Swagger




