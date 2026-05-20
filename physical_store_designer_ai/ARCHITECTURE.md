# Arquitectura Modular

Este documento describe la arquitectura modular del proyecto Physical Store Designer AI.

## Estructura Modular

### Core Module (`core/`)

Módulo central con componentes reutilizables:

- **models.py**: Modelos de datos Pydantic
- **exceptions.py**: Excepciones personalizadas
- **interfaces.py**: Interfaces/Protocols para servicios
- **factories.py**: Factories para crear instancias
- **validators.py**: Validadores reutilizables
- **decorators.py**: Decoradores útiles
- **dependencies.py**: Dependencies de FastAPI
- **response_models.py**: Modelos de respuesta estándar
- **middleware.py**: Middleware personalizado
- **logging_config.py**: Configuración de logging
- **utils.py**: Utilidades generales

### Services Module (`services/`)

Servicios organizados por responsabilidad:

- **base.py**: Clases base y mixins
- **storage_service.py**: Persistencia de datos
- **chat_service.py**: Chat interactivo
- **store_designer_service.py**: Servicio principal de diseño
- Otros servicios especializados...

### API Module (`api/`)

Rutas organizadas por funcionalidad:

- **main.py**: Aplicación FastAPI principal
- **routes.py**: Rutas principales
- **analysis_routes.py**: Rutas de análisis
- **advanced_routes.py**: Rutas avanzadas
- ... (más rutas organizadas)

## Patrones de Diseño

### Factory Pattern

```python
from ..core.factories import ServiceFactory

# Obtener instancias singleton
storage_service = ServiceFactory.get_storage_service()
chat_service = ServiceFactory.get_chat_service()
```

### Interface Pattern

```python
from ..core.interfaces import IStorageService

class MyStorageService(IStorageService):
    def save_design(self, design):
        # Implementación
        pass
```

### Mixin Pattern

```python
from ..services.base import BaseService, StorageMixin

class MyService(BaseService, StorageMixin):
    def __init__(self):
        super().__init__()
        # Tiene acceso a self.store(), self.retrieve(), etc.
```

## Beneficios de la Arquitectura Modular

1. **Separación de Responsabilidades**: Cada módulo tiene una responsabilidad clara
2. **Reutilización**: Componentes base reutilizables (BaseService, Mixins)
3. **Testabilidad**: Interfaces facilitan testing con mocks
4. **Mantenibilidad**: Código organizado y fácil de encontrar
5. **Extensibilidad**: Fácil agregar nuevos servicios siguiendo interfaces
6. **Consistencia**: Patrones comunes en todo el código

## Uso de Componentes

### Crear un Nuevo Servicio

```python
from ..services.base import BaseService, StorageMixin
from ..core.logging_config import get_logger

class MyNewService(BaseService, StorageMixin):
    def __init__(self):
        super().__init__()
        # Inicialización
    
    def do_something(self):
        self.log_info("Doing something")
        # Lógica del servicio
```

### Usar Validators

```python
from ..core.validators import Validator, validate_and_raise

# Validar antes de usar
validate_and_raise(
    Validator.validate_store_type,
    store_type,
    "Tipo de tienda inválido"
)
```

### Usar Decorators

```python
from ..core.decorators import log_execution_time, retry_on_failure

@log_execution_time
@retry_on_failure(max_retries=3)
async def process_data(data):
    # Tu código
    pass
```

## Organización de Servicios

Los servicios están organizados por categoría:

- **Core Services**: Storage, Chat, Designer
- **Analysis Services**: Competitor, Financial, Location
- **Design Services**: Visualization, Decoration, Marketing
- **ML Services**: Modelos, entrenamiento, evaluación
- **Integration Services**: APIs externas, webhooks

## Próximas Mejoras

- [ ] Organizar servicios en subdirectorios por categoría
- [ ] Crear módulos de dominio específicos
- [ ] Implementar dependency injection
- [ ] Agregar tests unitarios para cada módulo

