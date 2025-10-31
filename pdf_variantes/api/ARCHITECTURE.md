# Arquitectura Limpia - PDF Variantes API

## Visión General

Esta API implementa **Clean Architecture** (Arquitectura Limpia) con separación clara de capas y responsabilidades, siguiendo principios SOLID y mejores prácticas de diseño de software.

## Estructura de Capas

```
api/
├── architecture/
│   ├── domain/              # Capa de Dominio
│   │   └── entities.py      # Entidades de negocio
│   ├── application/          # Capa de Aplicación
│   │   └── use_cases.py     # Casos de uso (CQRS)
│   ├── infrastructure/       # Capa de Infraestructura
│   │   └── repositories.py  # Implementaciones concretas
│   └── presentation/         # Capa de Presentación
│       ├── controllers.py   # Controladores HTTP
│       └── presenters.py     # Formatos de respuesta
```

## Principios Arquitectónicos

### 1. Separación de Responsabilidades (SRP)
- Cada capa tiene una responsabilidad única
- Domain: Lógica de negocio
- Application: Orquestación de casos de uso
- Infrastructure: Implementaciones técnicas
- Presentation: Interfaz HTTP

### 2. Inversión de Dependencias (DIP)
- Las capas internas no dependen de las externas
- Interfaces definen contratos
- Implementaciones en capas externas

### 3. CQRS (Command Query Responsibility Segregation)
- Separación de operaciones de lectura y escritura
- Commands: Operaciones de escritura
- Queries: Operaciones de lectura

## Capas de la Arquitectura

### Domain Layer (Capa de Dominio)
**Responsabilidad**: Lógica de negocio y entidades puras

```python
# Entidades de dominio con lógica de negocio
@dataclass
class DocumentEntity:
    def is_owned_by(self, user_id: str) -> bool:
        # Lógica de negocio en la entidad
        return self.user_id == user_id
```

**Características**:
- Entidades con comportamiento
- Validaciones de negocio
- Invariantes del dominio
- Sin dependencias externas

### Application Layer (Capa de Aplicación)
**Responsabilidad**: Orquestación de casos de uso

```python
# Casos de uso específicos
class UploadDocumentUseCase:
    async def execute(self, command: UploadDocumentCommand) -> DocumentEntity:
        # Orquestación de la lógica
        pass
```

**Características**:
- Casos de uso únicos y específicos
- Commands y Queries separados
- Uso de interfaces (abstracciones)
- Sin detalles de implementación

### Infrastructure Layer (Capa de Infraestructura)
**Responsabilidad**: Implementaciones técnicas

```python
# Implementación concreta del repositorio
class DocumentRepository(Repository):
    async def get_by_id(self, id: str) -> Optional[DocumentEntity]:
        # Implementación específica (DB, API, etc.)
        pass
```

**Características**:
- Implementaciones concretas
- Acceso a bases de datos
- Servicios externos
- Frameworks y librerías

### Presentation Layer (Capa de Presentación)
**Responsabilidad**: Interfaz HTTP y formato de respuestas

```python
# Controlador que delega a casos de uso
class DocumentController:
    async def handle_upload(self, request, user_id, file):
        command = UploadDocumentCommand(...)
        document = await self.upload_use_case.execute(command)
        return self.presenter.present(document)
```

**Características**:
- Controladores HTTP
- Validación de entrada
- Formato de respuestas
- Manejo de errores HTTP

## Flujo de Datos

### 1. Request Flow
```
HTTP Request
    ↓
Controller (Presentation)
    ↓
Use Case (Application)
    ↓
Repository (Infrastructure)
    ↓
Database/External Service
```

### 2. Response Flow
```
Domain Entity
    ↓
Presenter (Presentation)
    ↓
JSON Response
    ↓
HTTP Response
```

## Ventajas de esta Arquitectura

### ✅ Testabilidad
- Cada capa es testeable independientemente
- Mocks y stubs fáciles de implementar
- Tests unitarios sin dependencias externas

### ✅ Mantenibilidad
- Código organizado y claro
- Fácil de entender y modificar
- Cambios aislados por capa

### ✅ Escalabilidad
- Fácil agregar nuevas funcionalidades
- Extensiones sin modificar código existente
- Separación permite escalar por capas

### ✅ Independencia de Frameworks
- Domain y Application independientes
- Fácil cambiar frameworks (FastAPI → Django, etc.)
- Migración gradual posible

## Patrones Implementados

### Repository Pattern
```python
# Abstracción
class Repository(ABC):
    @abstractmethod
    async def get_by_id(self, id: str) -> Optional[Any]:
        pass

# Implementación
class DocumentRepository(Repository):
    async def get_by_id(self, id: str) -> Optional[DocumentEntity]:
        # Implementación concreta
        pass
```

### Use Case Pattern
```python
# Cada caso de uso es una clase
class UploadDocumentUseCase:
    def __init__(self, repository: DocumentRepository):
        self.repository = repository
    
    async def execute(self, command: UploadDocumentCommand):
        # Lógica del caso de uso
        pass
```

### Presenter Pattern
```python
# Formatea entidades para la API
class DocumentPresenter:
    def present(self, entity: DocumentEntity) -> dict:
        return {
            "id": entity.id,
            "filename": entity.filename,
            # ... formateado para API
        }
```

### Dependency Injection
```python
# Container maneja dependencias
container = ApplicationContainer()
controller = container.get_document_controller()
```

## Estructura de Directorios Recomendada

```
pdf_variantes/
├── api/
│   ├── architecture/
│   │   ├── domain/
│   │   │   ├── entities.py
│   │   │   ├── value_objects.py
│   │   │   └── services.py
│   │   ├── application/
│   │   │   ├── commands/
│   │   │   ├── queries/
│   │   │   └── use_cases/
│   │   ├── infrastructure/
│   │   │   ├── repositories/
│   │   │   ├── cache/
│   │   │   └── external_services/
│   │   └── presentation/
│   │       ├── controllers/
│   │       ├── presenters/
│   │       └── validators/
│   ├── routes.py
│   └── main.py
└── tests/
    ├── unit/
    ├── integration/
    └── e2e/
```

## Reglas de la Arquitectura

### Reglas de Dependencia

1. **Domain** → No depende de nadie
2. **Application** → Solo depende de Domain
3. **Infrastructure** → Depende de Domain y Application (interfaces)
4. **Presentation** → Depende de Application (casos de uso)

### Flujo de Dependencias

```
Presentation → Application → Domain ← Infrastructure
```

Las dependencias siempre apuntan hacia adentro (hacia Domain).

## Ejemplo de Implementación

### 1. Definir Entidad de Dominio
```python
@dataclass
class DocumentEntity:
    id: str
    user_id: str
    filename: str
    
    def can_be_deleted_by(self, user_id: str) -> bool:
        return self.user_id == user_id
```

### 2. Definir Caso de Uso
```python
class DeleteDocumentUseCase:
    def __init__(self, repository: DocumentRepository):
        self.repository = repository
    
    async def execute(self, document_id: str, user_id: str):
        document = await self.repository.get_by_id(document_id)
        if not document.can_be_deleted_by(user_id):
            raise PermissionError()
        await self.repository.delete(document_id)
```

### 3. Implementar Repositorio
```python
class DocumentRepository(Repository):
    async def get_by_id(self, id: str):
        # Query a base de datos
        pass
```

### 4. Crear Controlador
```python
class DocumentController:
    def __init__(self, delete_use_case: DeleteDocumentUseCase):
        self.delete_use_case = delete_use_case
    
    async def handle_delete(self, document_id: str, user_id: str):
        await self.delete_use_case.execute(document_id, user_id)
        return {"success": True}
```

## Testing por Capas

### Domain Tests
```python
def test_document_can_be_deleted_by_owner():
    doc = DocumentEntity(id="1", user_id="user1")
    assert doc.can_be_deleted_by("user1")
```

### Application Tests
```python
async def test_delete_document_use_case():
    mock_repo = MockRepository()
    use_case = DeleteDocumentUseCase(mock_repo)
    await use_case.execute("doc1", "user1")
    assert mock_repo.delete_called
```

### Integration Tests
```python
async def test_delete_document_endpoint():
    response = await client.delete("/documents/doc1")
    assert response.status_code == 200
```

## Próximos Pasos

1. **Completar implementaciones** de casos de uso
2. **Agregar tests** por cada capa
3. **Implementar repositories** con bases de datos reales
4. **Agregar validaciones** en cada capa apropiada
5. **Documentar interfaces** y contratos

## Beneficios a Largo Plazo

- ✅ **Código más limpio y mantenible**
- ✅ **Fácil agregar nuevas features**
- ✅ **Tests más simples y rápidos**
- ✅ **Cambios aislados por capa**
- ✅ **Mejor colaboración en equipo**
- ✅ **Escalabilidad mejorada**






