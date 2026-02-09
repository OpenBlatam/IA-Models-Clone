# Guía de Testing - Arquitectura Mejorada

## 📋 Visión General

Suite completa de tests unitarios para todos los componentes de la arquitectura mejorada, siguiendo las mejores prácticas de testing.

## 🧪 Estructura de Tests

```
tests/
├── conftest.py                          # Configuración pytest
├── test_architecture_domain.py          # Tests Domain Layer
├── test_architecture_application.py    # Tests Application Layer
├── test_architecture_repositories.py    # Tests Infrastructure Repositories
├── test_architecture_circuit_breaker.py # Tests Circuit Breaker
└── test_architecture_di.py             # Tests Dependency Injection
```

## 🚀 Ejecutar Tests

### Todos los Tests

```bash
pytest tests/
```

### Tests Específicos

```bash
# Tests de dominio
pytest tests/test_architecture_domain.py

# Tests de aplicación
pytest tests/test_architecture_application.py

# Tests de repositorios
pytest tests/test_architecture_repositories.py

# Tests de circuit breaker
pytest tests/test_architecture_circuit_breaker.py

# Tests de DI
pytest tests/test_architecture_di.py
```

### Con Cobertura

```bash
pytest tests/ --cov=core.architecture --cov-report=html
```

### Tests Verbosos

```bash
pytest tests/ -v
```

### Tests con Output Detallado

```bash
pytest tests/ -v -s
```

## 📊 Cobertura de Tests

### Domain Layer
- ✅ Value Objects (Position, Orientation)
- ✅ Entidades (Robot, RobotMovement)
- ✅ Validaciones de dominio
- ✅ Domain Events
- ✅ Lógica de negocio

### Application Layer
- ✅ Use Cases (MoveRobot, GetStatus, GetHistory)
- ✅ Commands y Queries
- ✅ Manejo de errores
- ✅ DTOs

### Infrastructure Layer
- ✅ Repositorios In-Memory
- ✅ Operaciones CRUD
- ✅ Indexación
- ✅ Búsquedas

### Circuit Breaker
- ✅ Estados (CLOSED, OPEN, HALF_OPEN)
- ✅ Transiciones de estado
- ✅ Métricas
- ✅ Timeouts
- ✅ Domain Events

### Dependency Injection
- ✅ Setup y configuración
- ✅ Resolución de servicios
- ✅ Lifecycle management
- ✅ Helper functions

## 🎯 Ejemplos de Tests

### Test de Entidad de Dominio

```python
def test_robot_connect():
    robot = Robot(robot_id="robot-1", brand="KUKA", model="KR210")
    robot.connect()
    assert robot.is_connected
```

### Test de Use Case

```python
@pytest.mark.asyncio
async def test_move_robot_success():
    use_case = MoveRobotUseCase(mock_repo, mock_movement_repo)
    command = MoveRobotCommand(robot_id="robot-1", target_x=1.0, target_y=2.0, target_z=3.0)
    result = await use_case.execute(command)
    assert result.movement_id is not None
```

### Test de Repositorio

```python
@pytest.mark.asyncio
async def test_save_and_find_robot():
    repo = InMemoryRobotRepository()
    await repo.initialize()
    
    robot = Robot(robot_id="robot-1", brand="KUKA", model="KR210")
    await repo.save(robot)
    
    found = await repo.find_by_id("robot-1")
    assert found is not None
```

## 🔧 Configuración

### Variables de Entorno para Tests

Los tests usan automáticamente:
- `REPOSITORY_TYPE=in_memory` - Repositorios en memoria para tests rápidos
- `ENABLE_EVENT_BUS=false` - Deshabilitar event bus en tests

### Fixtures Disponibles

- `event_loop`: Event loop para tests async
- `reset_di`: Resetear DI antes de cada test
- `mock_robot_repo`: Mock de repositorio de robots
- `mock_movement_repo`: Mock de repositorio de movimientos

## 📝 Mejores Prácticas

1. **Tests Aislados**: Cada test es independiente
2. **Mocks**: Usar mocks para dependencias externas
3. **Nombres Descriptivos**: Nombres claros que describen qué se testea
4. **Arrange-Act-Assert**: Estructura clara de tests
5. **Async Tests**: Usar `@pytest.mark.asyncio` para tests async

## 🚀 Próximos Pasos

1. Tests de integración
2. Tests de aceptación
3. Tests de performance
4. Tests de carga

---

**Fecha**: 2025-01-27
**Versión**: 1.0.0




