# Mejoras de Arquitectura

Este documento describe las mejoras arquitectónicas implementadas siguiendo principios de Clean Architecture y SOLID.

## Estructura de Capas

### 1. Domain Layer (`domain/`)
**Responsabilidad**: Lógica de negocio pura, independiente de frameworks

- **`use_cases/`**: Casos de uso que encapsulan la lógica de negocio
  - `create_visualization.py`: Crear visualización
  - `get_visualization.py`: Obtener visualización

**Principios**:
- No depende de frameworks externos
- Define interfaces (protocols) para dependencias
- Contiene la lógica de negocio pura

### 2. Infrastructure Layer (`infrastructure/`)
**Responsabilidad**: Implementaciones concretas de interfaces

- **`repositories/`**: Implementaciones de repositorios
  - `storage_repository.py`: Almacenamiento de archivos
  - `cache_repository.py`: Caché de archivos

- **`adapters/`**: Adaptadores para servicios existentes
  - `image_processor_adapter.py`: Adaptador para procesador de imágenes
  - `ai_processor_adapter.py`: Adaptador para procesador AI
  - `metrics_adapter.py`: Adaptador para métricas

**Principios**:
- Implementa interfaces definidas en Domain
- Puede depender de frameworks externos
- Aislado de la lógica de negocio

### 3. Core Layer (`core/`)
**Responsabilidad**: Interfaces, excepciones, constantes

- **`interfaces.py`**: Protocolos/Interfaces para inyección de dependencias
  - `IImageProcessor`: Interface para procesamiento de imágenes
  - `IAProcessor`: Interface para procesamiento AI
  - `IStorageRepository`: Interface para almacenamiento
  - `ICacheRepository`: Interface para caché
  - `IMetricsCollector`: Interface para métricas

**Principios**:
- Define contratos (interfaces)
- No contiene implementaciones
- Independiente de frameworks

### 4. Application Layer (`services/`)
**Responsabilidad**: Facade/Orquestación de casos de uso

- **`visualization_service.py`**: Servicio que orquesta casos de uso
  - Facade pattern
  - Inyección de dependencias
  - Delegación a use cases

**Principios**:
- Orquesta casos de uso
- No contiene lógica de negocio
- Facilita testing y mantenimiento

### 5. Presentation Layer (`api/`)
**Responsabilidad**: Endpoints HTTP, validación de entrada

- **`routes/`**: Endpoints de la API
- **`schemas/`**: Modelos Pydantic para validación

**Principios**:
- Solo maneja HTTP
- Valida entrada
- Delega a servicios

## Patrones de Diseño Implementados

### 1. Repository Pattern
- **Interfaces**: `IStorageRepository`, `ICacheRepository`
- **Implementaciones**: `FileStorageRepository`, `FileCacheRepository`
- **Beneficio**: Fácil cambiar implementación (ej: S3, Redis)

### 2. Use Case Pattern
- **Casos de uso**: `CreateVisualizationUseCase`, `GetVisualizationUseCase`
- **Beneficio**: Lógica de negocio encapsulada y testeable

### 3. Adapter Pattern
- **Adaptadores**: `ImageProcessorAdapter`, `AIProcessorAdapter`, `MetricsCollectorAdapter`
- **Beneficio**: Integra código existente con nuevas interfaces

### 4. Factory Pattern
- **Factories**: `create_image_processor()`, `create_storage_repository()`, etc.
- **Beneficio**: Centraliza creación de instancias

### 5. Dependency Injection
- **Interfaces**: Todas las dependencias se inyectan vía interfaces
- **Beneficio**: Fácil testing, bajo acoplamiento

## Principios SOLID

### Single Responsibility Principle (SRP)
- Cada clase tiene una única responsabilidad
- Use cases: solo lógica de negocio
- Repositories: solo acceso a datos
- Services: solo orquestación

### Open/Closed Principle (OCP)
- Abierto para extensión, cerrado para modificación
- Nuevas implementaciones: crear nuevos repositorios/adapters
- No modificar código existente

### Liskov Substitution Principle (LSP)
- Implementaciones pueden sustituir interfaces
- Cualquier `IStorageRepository` funciona igual

### Interface Segregation Principle (ISP)
- Interfaces pequeñas y específicas
- `IImageProcessor` solo métodos de imágenes
- `IAProcessor` solo métodos de AI

### Dependency Inversion Principle (DIP)
- Dependencias de abstracciones, no implementaciones
- Use cases dependen de interfaces
- Implementaciones en infrastructure

## Flujo de Datos

```
Request (API)
  ↓
Route Handler (api/routes/)
  ↓
Service (services/) - Facade
  ↓
Use Case (domain/use_cases/) - Business Logic
  ↓
Repository (infrastructure/repositories/) - Data Access
  ↓
Storage/External Service
```

## Beneficios

1. **Testabilidad**: Fácil mockear interfaces
2. **Mantenibilidad**: Código organizado por responsabilidades
3. **Escalabilidad**: Fácil agregar nuevas features
4. **Flexibilidad**: Cambiar implementaciones sin afectar lógica
5. **Claridad**: Separación clara de concerns

## Migración

El código antiguo sigue funcionando gracias a:
- Adapters que implementan interfaces
- Factories que crean instancias compatibles
- Backward compatibility en servicios

## Próximos Pasos

1. Agregar más use cases (comparison, batch)
2. Implementar repositorios alternativos (S3, Redis)
3. Agregar unit tests para use cases
4. Documentar interfaces con docstrings
5. Agregar validación de dominio

