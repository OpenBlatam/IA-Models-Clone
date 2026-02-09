# Índice de Documentación - Robot Movement AI
## Arquitectura Mejorada v2.0

---

## 📋 Documentos Principales

### 🎯 Resúmenes Ejecutivos
1. **[Resumen Ejecutivo de Mejoras](./ARCHITECTURE_IMPROVEMENTS_EXECUTIVE_SUMMARY.md)**
   - Visión general de todas las mejoras
   - Métricas de impacto
   - Valor de negocio
   - Próximos pasos

2. **[Quick Start - Nueva Arquitectura](./QUICK_START_ARCHITECTURE.md)**
   - Inicio rápido
   - Ejemplos básicos
   - Checklist de migración

---

## 🏗️ Arquitectura

### Documentación de Arquitectura
1. **[Arquitectura Mejorada](./ARCHITECTURE_IMPROVED.md)**
   - Clean Architecture + DDD
   - Capas y responsabilidades
   - Patrones de diseño
   - Estructura de directorios

2. **[Resumen de Mejoras Arquitectónicas](./ARCHITECTURE_IMPROVEMENTS_SUMMARY.md)**
   - Detalles técnicos
   - Componentes implementados
   - Ventajas y beneficios

---

## 📚 Guías por Componente

### Domain Layer
- **Archivo**: `core/architecture/domain_improved.py`
- **Tests**: `tests/test_architecture_domain.py`
- **Conceptos**: Entidades, Value Objects, Domain Events

### Application Layer
- **Archivo**: `core/architecture/application_layer.py`
- **Tests**: `tests/test_architecture_application.py`
- **Conceptos**: Use Cases, Commands, Queries, DTOs

### Infrastructure Layer
- **[Guía de Repositorios](./core/architecture/REPOSITORIES_GUIDE.md)**
- **Archivos**: 
  - `core/architecture/infrastructure_repositories.py`
  - `core/architecture/repository_factory.py`
- **Tests**: `tests/test_architecture_repositories.py`

### Dependency Injection
- **[Guía de Integración DI](./core/architecture/DI_INTEGRATION_GUIDE.md)**
- **Archivos**:
  - `core/architecture/di_setup.py`
  - `core/architecture/dependency_injection.py`
  - `core/architecture/di_integration_example.py`
- **Tests**: `tests/test_architecture_di.py`

### Circuit Breaker
- **[Guía de Circuit Breaker](./core/architecture/CIRCUIT_BREAKER_GUIDE.md)**
- **[Mejoras del Circuit Breaker](./core/architecture/CIRCUIT_BREAKER_IMPROVEMENTS.md)**
- **Archivo**: `core/architecture/circuit_breaker.py`
- **Tests**: `tests/test_architecture_circuit_breaker.py`

### Error Handling
- **Archivo**: `core/architecture/error_handling.py`
- **Conceptos**: Jerarquía de excepciones, Error codes, Context

---

## 🧪 Testing

- **[Guía de Testing](./tests/README_TESTS.md)**
- **Tests Disponibles**:
  - `tests/test_architecture_domain.py`
  - `tests/test_architecture_application.py`
  - `tests/test_architecture_repositories.py`
  - `tests/test_architecture_circuit_breaker.py`
  - `tests/test_architecture_di.py`

---

## 📖 Documentación Original

### Arquitectura Original
- `ARCHITECTURE.md` - Arquitectura original
- `ARCHITECTURE_SUMMARY.md` - Resumen original
- `INTERNAL_ARCHITECTURE.md` - Arquitectura interna modular

### Otros Documentos
- `README.md` - Documentación principal
- `QUICK_START.md` - Inicio rápido original
- `STRUCTURE.md` - Estructura del proyecto
- `IMPROVEMENTS_2025.md` - Mejoras planificadas

---

## 🗺️ Ruta de Aprendizaje Recomendada

### Para Nuevos Desarrolladores

1. **Empezar aquí**:
   - [Resumen Ejecutivo](./ARCHITECTURE_IMPROVEMENTS_EXECUTIVE_SUMMARY.md)
   - [Quick Start](./QUICK_START_ARCHITECTURE.md)

2. **Entender la Arquitectura**:
   - [Arquitectura Mejorada](./ARCHITECTURE_IMPROVED.md)
   - [Resumen de Mejoras](./ARCHITECTURE_IMPROVEMENTS_SUMMARY.md)

3. **Aprender Componentes Específicos**:
   - [Guía de Repositorios](./core/architecture/REPOSITORIES_GUIDE.md)
   - [Guía de DI](./core/architecture/DI_INTEGRATION_GUIDE.md)
   - [Guía de Circuit Breaker](./core/architecture/CIRCUIT_BREAKER_GUIDE.md)

4. **Practicar con Tests**:
   - [Guía de Testing](./tests/README_TESTS.md)
   - Revisar tests existentes

### Para Arquitectos

1. [Arquitectura Mejorada](./ARCHITECTURE_IMPROVED.md)
2. [Resumen Ejecutivo](./ARCHITECTURE_IMPROVEMENTS_EXECUTIVE_SUMMARY.md)
3. Revisar código fuente de componentes clave
4. Analizar decisiones de diseño

### Para QA/Testing

1. [Guía de Testing](./tests/README_TESTS.md)
2. Revisar tests existentes
3. Entender estructura de tests
4. Agregar nuevos tests según necesidad

---

## 🔍 Búsqueda Rápida

### Por Tema

**Arquitectura**:
- Clean Architecture
- Domain-Driven Design
- CQRS Pattern
- Repository Pattern

**Componentes**:
- Domain Layer
- Application Layer
- Infrastructure Layer
- Dependency Injection
- Circuit Breaker
- Error Handling

**Guías Prácticas**:
- Setup y configuración
- Ejemplos de código
- Testing
- Migración

---

## 📝 Notas

- Todos los documentos están en Markdown
- Los ejemplos de código están probados
- La documentación se actualiza con cada cambio importante
- Para preguntas, revisar primero la documentación relevante

---

**Última actualización**: 2025-01-27  
**Versión**: 2.0.0




