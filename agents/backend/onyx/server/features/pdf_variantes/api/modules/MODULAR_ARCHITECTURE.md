# Arquitectura Modular - PDF Variantes API

## Visión General

Arquitectura completamente modular donde cada funcionalidad es un módulo independiente, reutilizable y auto-contenido.

## Estructura Modular

```
api/modules/
├── document/              # Módulo de Documentos
│   ├── __init__.py        # Exports públicos
│   ├── domain.py          # Entidades y lógica de negocio
│   ├── application.py     # Casos de uso
│   ├── infrastructure.py  # Repositorios
│   └── presentation.py   # Controllers y Presenters
│
├── variant/               # Módulo de Variantes
│   ├── __init__.py
│   ├── domain.py
│   ├── application.py
│   ├── infrastructure.py
│   └── presentation.py
│
├── topic/                 # Módulo de Topics
│   ├── __init__.py
│   ├── domain.py
│   ├── application.py
│   ├── infrastructure.py
│   └── presentation.py
│
├── module_registry.py     # Registry de módulos
├── module_factory.py     # Factory para crear módulos
└── module_router.py      # Auto-generación de routers
```

## Principios de Modularidad

### 1. Independencia
Cada módulo es independiente y puede funcionar solo:
- No depende de otros módulos directamente
- Comunicación a través de interfaces
- Puede ser removido sin afectar otros

### 2. Encapsulación
Cada módulo encapsula toda su funcionalidad:
- Domain, Application, Infrastructure, Presentation
- Todo lo relacionado con el módulo está dentro

### 3. Reusabilidad
Los módulos son reutilizables:
- Pueden usarse en otros proyectos
- Pueden combinarse de diferentes formas
- Intercambiables

### 4. Auto-contención
Cada módulo contiene todo lo necesario:
- No requiere configuración externa
- Self-contained
- Puede ser testeado independientemente

## Uso de Módulos

### Registro y Carga

```python
from api.modules.module_registry import register_module, load_module

# Registrar módulo
register_module("document", {
    "path": "api.modules.document",
    "config": {...}
})

# Cargar módulo
load_module("document")
```

### Usar Módulo Completo

```python
from api.modules.document import (
    DocumentEntity,
    DocumentController,
    DocumentRepository,
    UploadDocumentUseCase
)

# Usar directamente
controller = DocumentController(...)
```

### Auto-generación de Routers

```python
from api.modules.module_router import ModuleRouter

router_generator = ModuleRouter()

# Crear router para un módulo
document_router = router_generator.create_router("document")

# Registrar en FastAPI
app.include_router(document_router)
```

### Factory Pattern

```python
from api.modules.module_factory import ModuleFactory

factory = ModuleFactory()

# Crear módulo con configuración
doc_module = factory.create_document_module(
    repository=my_repo,
    config={"auto_process": True}
)

# Crear todos los módulos
all_modules = factory.create_all_modules({
    "document": doc_repo,
    "variant": variant_repo,
    "topic": topic_repo
})
```

## Estructura de un Módulo

### 1. Domain Layer
```python
# modules/document/domain.py
class DocumentEntity:
    # Entidad con lógica de negocio
    def is_owned_by(self, user_id: str) -> bool:
        return self.user_id == user_id

class DocumentFactory:
    # Factory para crear entidades
    @staticmethod
    def create(...):
        pass
```

### 2. Application Layer
```python
# modules/document/application.py
@dataclass
class UploadDocumentCommand:
    user_id: str
    filename: str
    file_content: bytes

class UploadDocumentUseCase(ABC):
    @abstractmethod
    async def execute(self, command: UploadDocumentCommand):
        pass
```

### 3. Infrastructure Layer
```python
# modules/document/infrastructure.py
class DocumentRepository(ABC):
    @abstractmethod
    async def get_by_id(self, id: str):
        pass
```

### 4. Presentation Layer
```python
# modules/document/presentation.py
class DocumentController:
    async def upload(self, request, command):
        document = await self.use_case.execute(command)
        return self.presenter.to_dict(document)

class DocumentPresenter:
    @staticmethod
    def to_dict(document):
        return {...}
```

## Ventajas de la Arquitectura Modular

### ✅ Independencia
- Cada módulo puede desarrollarse por separado
- Equipos pueden trabajar en paralelo
- Deployment independiente posible

### ✅ Reusabilidad
- Módulos pueden usarse en otros proyectos
- Fácil compartir entre aplicaciones
- Librerías reutilizables

### ✅ Testabilidad
- Cada módulo testeable independientemente
- Tests más rápidos y enfocados
- Mocks más simples

### ✅ Mantenibilidad
- Código organizado por funcionalidad
- Fácil localizar código relacionado
- Cambios aislados por módulo

### ✅ Escalabilidad
- Agregar nuevos módulos fácil
- No afecta módulos existentes
- Escalar por módulo

## Ejemplo Completo

### Crear y Usar Módulo

```python
# 1. Importar módulo
from api.modules.document import DocumentController, DocumentFactory

# 2. Crear instancias necesarias (con DI)
repository = DocumentRepository()
use_cases = create_use_cases(repository)
controller = DocumentController(*use_cases)

# 3. Usar en routes
@router.post("/upload")
async def upload(request: Request, user_id: str, file: bytes):
    command = UploadDocumentCommand(
        user_id=user_id,
        filename=file.filename,
        file_content=file.content
    )
    return await controller.upload(request, command)
```

### Auto-registro de Módulos

```python
# En main.py o similar
from api.modules.module_router import ModuleRouter

router_gen = ModuleRouter()
router_gen.register_all_modules(app)
```

## Comparación: Modular vs Monolítico

| Aspecto | Monolítico | Modular |
|---------|------------|---------|
| Organización | Por capas | Por funcionalidad |
| Dependencias | Acopladas | Desacopladas |
| Testing | Complejo | Simple |
| Reusabilidad | Baja | Alta |
| Paralelo Dev | Difícil | Fácil |
| Deployment | Todo junto | Por módulo |

## Próximos Pasos

1. ✅ **Completar módulos restantes** (si hay más)
2. ✅ **Agregar tests por módulo**
3. ✅ **Documentación de cada módulo**
4. ✅ **Plugin system** para módulos externos
5. ✅ **Module marketplace** (si aplica)

## Mejores Prácticas

1. **Cada módulo en su directorio**
2. **Exports claros en __init__.py**
3. **No dependencias entre módulos**
4. **Comunicación vía eventos/interfaces**
5. **Tests independientes por módulo**
6. **Documentación por módulo**

La arquitectura modular está completa y lista para usar. Cada módulo es independiente, testeable y reutilizable.






