# Modularización del Physical Store Designer AI

## Resumen

Este documento describe la estructura modular mejorada del proyecto.

## Estructura de Servicios

Los servicios están organizados en subdirectorios por categoría:

```
services/
├── __init__.py           # Exports principales y compatibilidad hacia atrás
├── base.py                # Clases base (BaseService, StorageMixin, etc.)
├── core/                  # Servicios core del negocio
│   ├── __init__.py
│   ├── storage_service.py
│   ├── chat_service.py
│   └── store_designer_service.py
├── ml/                    # Servicios de Machine Learning
│   ├── __init__.py
│   └── [30+ servicios ML]
├── analysis/              # Servicios de análisis
│   ├── __init__.py
│   └── [10+ servicios de análisis]
├── integration/           # Servicios de integración
│   ├── __init__.py
│   └── [4+ servicios de integración]
└── business/              # Servicios de negocio
    ├── __init__.py
    └── [2+ servicios de negocio]
```

## Uso

### Importación Directa (Compatibilidad hacia atrás)

```python
from services import StorageService, ChatService
from services.storage_service import StorageService
```

### Importación por Categoría

```python
from services import core_services, ml_services, analysis_services

# Usar servicios core
storage = core_services.StorageService()

# Usar servicios ML
validator = ml_services.AdvancedValidationService()
```

### Funciones de Conveniencia

```python
from services import get_storage_service, get_chat_service

storage = get_storage_service()
chat = get_chat_service()
```

## Estructura de Core

El módulo `core/` está organizado en:

```
core/
├── __init__.py            # Exports principales
├── README.md              # Documentación del módulo
├── models.py              # Modelos Pydantic
├── exceptions.py           # Excepciones
├── service_base.py         # Clases base
├── circuit_breaker/        # Circuit breaker modular
│   ├── __init__.py
│   ├── README.md
│   └── [10 módulos]
├── middleware/            # Middleware
│   └── [6 middlewares]
└── utils/                  # Utilidades
    └── [8 módulos de utilidades]
```

## Beneficios

1. **Organización Clara**: Servicios agrupados por responsabilidad
2. **Compatibilidad**: Imports existentes siguen funcionando
3. **Escalabilidad**: Fácil agregar nuevos servicios en la categoría correcta
4. **Mantenibilidad**: Código más fácil de encontrar y mantener
5. **Documentación**: READMEs en cada módulo principal

## Próximos Pasos

- [ ] Mover servicios físicamente a subdirectorios (opcional)
- [ ] Agregar tests por categoría
- [ ] Documentar cada categoría de servicios
- [ ] Crear diagramas de dependencias


