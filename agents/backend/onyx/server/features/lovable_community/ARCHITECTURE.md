# Arquitectura Avanzada de Lovable Community

Este documento describe la arquitectura avanzada del proyecto, implementando patrones de diseño profesionales y principios de Clean Architecture.

## 🏗️ Patrones de Diseño Implementados

### 1. Repository Pattern

**Ubicación:** `repositories/`

**Propósito:** Abstraer el acceso a datos, separando la lógica de negocio de la persistencia.

**Estructura:**
```
repositories/
├── base.py              # BaseRepository con operaciones CRUD comunes
├── chat_repository.py   # ChatRepository con queries especializadas
├── remix_repository.py  # RemixRepository
├── vote_repository.py   # VoteRepository
└── view_repository.py   # ViewRepository
```

**Ejemplo de uso:**
```python
from .repositories import ChatRepository

# En un servicio
chat_repo = ChatRepository(db)
chat = chat_repo.get_by_id(chat_id)
chats = chat_repo.get_public_chats(skip=0, limit=20, sort_by="score")
```

**Beneficios:**
- Separación de responsabilidades
- Fácil testing (mock repositories)
- Cambio de ORM sin afectar servicios
- Queries especializadas organizadas

### 2. Factory Pattern

**Ubicación:** `factories/`

**Propósito:** Centralizar la creación de objetos complejos con dependencias.

**Estructura:**
```
factories/
├── repository_factory.py  # Factory para repositorios
└── service_factory.py     # Factory para servicios
```

**Ejemplo de uso:**
```python
from .factories import ServiceFactory, RepositoryFactory

# Crear factories
repo_factory = RepositoryFactory(db)
service_factory = ServiceFactory(db)

# Obtener instancias
chat_repo = repo_factory.get_chat_repository()
chat_service = service_factory.get_chat_service()
```

**Beneficios:**
- Gestión centralizada de dependencias
- Singleton pattern integrado
- Fácil configuración y testing

### 3. Interface Segregation (Protocols)

**Ubicación:** `interfaces/`

**Propósito:** Definir contratos claros para servicios y repositorios.

**Estructura:**
```
interfaces/
└── __init__.py  # Protocolos para servicios y repositorios
```

**Ejemplo:**
```python
from typing import Protocol

class IChatRepository(Protocol):
    def get_by_id(self, chat_id: str) -> Optional[PublishedChat]:
        ...
    
    def create(self, **kwargs) -> PublishedChat:
        ...
```

**Beneficios:**
- Contratos claros
- Mejor testabilidad
- Desacoplamiento
- Type safety

## 📐 Arquitectura en Capas

```
┌─────────────────────────────────────────┐
│         API Layer (FastAPI)            │
│  - Routes, Endpoints, Middleware       │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│        Service Layer                    │
│  - Business Logic, Orchestration        │
│  - ChatService, RankingService         │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│      Repository Layer                   │
│  - Data Access Abstraction              │
│  - ChatRepository, VoteRepository      │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│        Model Layer (SQLAlchemy)         │
│  - Database Models                      │
│  - PublishedChat, ChatVote, etc.        │
└─────────────────────────────────────────┘
```

## 🔄 Flujo de Datos

### Ejemplo: Publicar un Chat

```
1. API Route recibe request
   ↓
2. Schema validation (Pydantic)
   ↓
3. Service Factory crea ChatService
   ↓
4. ChatService usa ChatRepository
   ↓
5. Repository ejecuta query SQLAlchemy
   ↓
6. Model se persiste en DB
   ↓
7. Response se serializa con Schema
   ↓
8. API retorna respuesta
```

## 🎯 Principios SOLID Aplicados

### Single Responsibility Principle (SRP)
- Cada repositorio maneja un solo modelo
- Cada servicio tiene una responsabilidad clara
- Helpers organizados por funcionalidad

### Open/Closed Principle (OCP)
- BaseRepository extensible sin modificar
- Nuevos repositorios extienden BaseRepository
- Servicios extensibles mediante herencia

### Liskov Substitution Principle (LSP)
- Todos los repositorios son intercambiables
- Protocolos garantizan compatibilidad

### Interface Segregation Principle (ISP)
- Protocolos específicos por funcionalidad
- No se fuerza implementación de métodos innecesarios

### Dependency Inversion Principle (DIP)
- Servicios dependen de abstracciones (repositorios)
- Factories inyectan dependencias
- Fácil testing con mocks

## 🧪 Testing Strategy

### Con Repositorios
```python
# Mock repository para testing
class MockChatRepository:
    def get_by_id(self, chat_id: str):
        return PublishedChat(id=chat_id, ...)

# Test service con mock
def test_chat_service():
    mock_repo = MockChatRepository()
    service = ChatService(repository=mock_repo)
    # Test logic...
```

### Con Factories
```python
# Factory para testing
test_factory = ServiceFactory(mock_db)
service = test_factory.get_chat_service()
```

## 📦 Estructura Completa

```
lovable_community/
├── api/              # API Layer
├── services/         # Service Layer
├── repositories/     # Repository Layer ✨ NUEVO
├── factories/        # Factory Pattern ✨ NUEVO
├── interfaces/       # Protocols ✨ NUEVO
├── models/           # Model Layer
├── schemas/          # Data Transfer Objects
├── helpers/          # Utility Functions
├── validators/       # Validation Logic
├── config/           # Configuration
├── core/             # Core Infrastructure
└── utils/            # General Utilities
```

## 🚀 Próximas Mejoras

1. **Domain Layer**: Entidades de dominio puras
2. **Event System**: Event-driven architecture
3. **CQRS**: Separación de comandos y queries
4. **Unit of Work**: Gestión transaccional
5. **Specification Pattern**: Queries complejas

## 📚 Referencias

- [Clean Architecture](https://blog.cleancoder.com/uncle-bob/2012/08/13/the-clean-architecture.html)
- [Repository Pattern](https://martinfowler.com/eaaCatalog/repository.html)
- [Factory Pattern](https://refactoring.guru/design-patterns/factory-method)
- [SOLID Principles](https://en.wikipedia.org/wiki/SOLID)













