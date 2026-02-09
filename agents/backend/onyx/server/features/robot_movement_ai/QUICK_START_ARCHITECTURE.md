# Quick Start - Nueva Arquitectura
## Robot Movement AI v2.0

## 🚀 Inicio Rápido

### 1. Setup Básico

```python
from core.architecture.di_setup import setup_dependency_injection
from core.architecture.application_layer import MoveRobotCommand, MoveRobotUseCase

# Configurar Dependency Injection
await setup_dependency_injection({
    'repository_type': 'in_memory'  # Para desarrollo
})

# Obtener use case
from core.architecture.di_setup import get_move_robot_use_case
use_case = await get_move_robot_use_case()

# Usar
command = MoveRobotCommand(
    robot_id="robot-1",
    target_x=0.5,
    target_y=0.3,
    target_z=0.2
)
result = await use_case.execute(command)
```

### 2. Con Circuit Breaker

```python
from core.architecture.circuit_breaker import circuit_breaker, CircuitBreakerConfig

@circuit_breaker(
    name="robot_service",
    config=CircuitBreakerConfig(
        failure_threshold=5,
        recovery_timeout=60.0
    )
)
async def call_robot_service():
    # Tu código aquí
    pass
```

### 3. Con Repositorio SQL

```python
import sqlite3
from core.architecture.repository_factory import create_repository_factory

# Crear conexión
db = sqlite3.connect("robots.db")

# Crear factory
factory = create_repository_factory(
    "sql",
    db_connection=db
)

# Usar repositorios
robot_repo = factory.create_robot_repository()
movement_repo = factory.create_movement_repository()
```

## 📚 Documentación Completa

- [Resumen Ejecutivo](./ARCHITECTURE_IMPROVEMENTS_EXECUTIVE_SUMMARY.md)
- [Arquitectura Completa](./ARCHITECTURE_IMPROVED.md)
- [Guías Específicas](./core/architecture/)

## ✅ Checklist de Migración

- [ ] Leer documentación de arquitectura
- [ ] Configurar Dependency Injection
- [ ] Migrar código existente gradualmente
- [ ] Agregar tests para nuevos componentes
- [ ] Configurar repositorios de producción

---

**Versión**: 2.0.0  
**Fecha**: 2025-01-27




