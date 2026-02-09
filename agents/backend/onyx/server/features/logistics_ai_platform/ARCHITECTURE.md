# Architecture Guide

## 🏗️ Arquitectura Modular

La plataforma sigue una arquitectura modular basada en principios de programación funcional y separación de responsabilidades.

## 📐 Capas de la Aplicación

```
┌─────────────────────────────────────────┐
│         API Layer (Routes)              │
│  - Declarative route definitions        │
│  - Request/Response handling             │
│  - Dependency injection                  │
└──────────────┬──────────────────────────┘
               │
┌──────────────▼──────────────────────────┐
│      Handler Layer (Handlers)          │
│  - Request orchestration                │
│  - Cache management                     │
│  - Error handling                       │
└──────────────┬──────────────────────────┘
               │
┌──────────────▼──────────────────────────┐
│      Domain Layer (Pure Functions)     │
│  - Business logic                       │
│  - Validation                           │
│  - No side effects                      │
└──────────────┬──────────────────────────┘
               │
┌──────────────▼──────────────────────────┐
│    Factory Layer (Object Creation)     │
│  - Build domain objects                 │
│  - Pure factory functions               │
└──────────────┬──────────────────────────┘
               │
┌──────────────▼──────────────────────────┐
│  Business Logic Layer (Pure Functions) │
│  - Calculations                         │
│  - Transformations                      │
│  - Algorithms                           │
└──────────────┬──────────────────────────┘
               │
┌──────────────▼──────────────────────────┐
│   Repository Layer (Data Access)        │
│  - Abstract data access                 │
│  - Database operations                  │
└─────────────────────────────────────────┘
```

## 📁 Estructura de Directorios

```
logistics_ai_platform/
├── api/                    # API Routes (HTTP layer)
│   ├── quotes/
│   ├── bookings/
│   ├── shipments/
│   └── containers/
│
├── handlers/               # Request Handlers (orchestration)
│   ├── quote_handlers.py
│   ├── booking_handlers.py
│   └── shipment_handlers.py
│
├── domain/                 # Domain Logic (pure functions)
│   ├── quotes.py
│   ├── bookings.py
│   └── shipments.py
│
├── factories/              # Object Factories (creation)
│   ├── quote_factory.py
│   ├── booking_factory.py
│   └── shipment_factory.py
│
├── business_logic/         # Business Rules (pure functions)
│   ├── quote_logic.py
│   ├── booking_logic.py
│   └── shipment_logic.py
│
├── repositories/           # Data Access (abstraction)
│   ├── quote_repository.py
│   ├── booking_repository.py
│   └── shipment_repository.py
│
├── validators/             # Validation (pure functions)
│   ├── quote_validators.py
│   ├── booking_validators.py
│   └── shipment_validators.py
│
└── utils/                  # Utilities
    ├── cache.py
    ├── decorators.py
    ├── performance.py
    └── async_helpers.py
```

## 🔄 Flujo de Datos

### Ejemplo: Crear Quote

```
1. Route (api/quotes/routes.py)
   ↓
   POST /quotes
   ↓
2. Handler (handlers/quote_handlers.py)
   ↓
   handle_create_quote()
   - Cache check
   - Error handling
   ↓
3. Domain (domain/quotes.py)
   ↓
   create_quote_domain()
   - Validation
   - Business rules
   ↓
4. Factory (factories/quote_factory.py)
   ↓
   build_quote_response()
   - Object creation
   ↓
5. Repository (repositories/quote_repository.py)
   ↓
   save()
   - Data persistence
```

## 🎯 Principios Aplicados

### 1. Separación de Responsabilidades

- **Routes**: Solo definición de endpoints
- **Handlers**: Orquestación y efectos secundarios (cache, logging)
- **Domain**: Lógica de negocio pura
- **Factories**: Creación de objetos
- **Repositories**: Acceso a datos

### 2. Programación Funcional

- **Funciones puras**: Sin efectos secundarios
- **Inmutabilidad**: Pydantic models
- **Composición**: Funciones pequeñas combinables
- **Sin clases**: Preferencia por funciones

### 3. Dependency Injection

- **FastAPI Depends**: Inyección automática
- **Repositorios**: Inyectados directamente
- **Sin servicios globales**: Todo por DI

### 4. RORO Pattern

- **Receive an Object**: Pydantic models como entrada
- **Return an Object**: Pydantic models como salida
- **Sin tuplas**: Objetos estructurados

## 📊 Ventajas de esta Arquitectura

1. **Testabilidad**: Funciones puras fáciles de testear
2. **Mantenibilidad**: Código organizado y claro
3. **Escalabilidad**: Fácil agregar nuevas features
4. **Reutilización**: Funciones puras reutilizables
5. **Debugging**: Fácil rastrear problemas
6. **Performance**: Optimizaciones por capa

## 🔧 Patrones Utilizados

### Factory Pattern
```python
# factories/quote_factory.py
def build_quote_response(request: QuoteRequest) -> QuoteResponse:
    # Pure function to create objects
```

### Repository Pattern
```python
# repositories/quote_repository.py
class QuoteRepository:
    async def save(self, quote: QuoteResponse) -> QuoteResponse
    async def find_by_id(self, quote_id: str) -> Optional[QuoteResponse]
```

### Handler Pattern
```python
# handlers/quote_handlers.py
async def handle_create_quote(
    request: QuoteRequest,
    repository: QuoteRepository
) -> QuoteResponse:
    # Orchestration with side effects
```

### Domain Function Pattern
```python
# domain/quotes.py
async def create_quote_domain(
    request: QuoteRequest,
    repository: QuoteRepository
) -> QuoteResponse:
    # Pure business logic
```

## 🚀 Mejores Prácticas

1. **Routes**: Solo definición, sin lógica
2. **Handlers**: Orquestación y efectos secundarios
3. **Domain**: Lógica de negocio pura
4. **Factories**: Creación de objetos
5. **Validators**: Validación temprana
6. **Repositories**: Solo acceso a datos

## 📝 Ejemplo Completo

### Route
```python
@router.post("", response_model=QuoteResponse, status_code=201)
async def create_quote(
    request: QuoteRequest,
    repository: QuoteRepository = Depends(get_quote_repository)
) -> QuoteResponse:
    return await handle_create_quote(request, repository)
```

### Handler
```python
async def handle_create_quote(
    request: QuoteRequest,
    repository: QuoteRepository
) -> QuoteResponse:
    quote = await create_quote_domain(request, repository)
    await cache_service.set(f"quote:{quote.quote_id}", quote.model_dump(), ttl=3600)
    return quote
```

### Domain
```python
async def create_quote_domain(
    request: QuoteRequest,
    repository: QuoteRepository
) -> QuoteResponse:
    validate_quote_request(request)
    quote = build_quote_response(request)
    await repository.save(quote)
    return quote
```

### Factory
```python
def build_quote_response(request: QuoteRequest) -> QuoteResponse:
    return QuoteResponse(
        quote_id=generate_quote_id(),
        request_id=generate_request_id(),
        # ... más campos
    )
```








