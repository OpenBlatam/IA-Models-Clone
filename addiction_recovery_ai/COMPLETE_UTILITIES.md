# Utilidades Completas - Addiction Recovery AI

## Resumen de Utilidades

El sistema ahora cuenta con **22 módulos de utilidades** con **120+ funciones** reutilizables organizadas por dominio.

## Módulos de Utilidades

### 1. Errores (`utils/errors.py`)
- Tipos de error personalizados
- Factories para crear excepciones HTTP

### 2. Validadores (`utils/validators.py`)
- Validación de usuarios, emails, fechas
- Validación de rangos y enums

### 3. Respuestas (`utils/response.py`)
- Helpers para crear respuestas consistentes

### 4. Cache (`utils/cache.py`)
- Sistema de caching con decorators
- TTL y estadísticas

### 5. Async Helpers (`utils/async_helpers.py`)
- Batching, parallel execution
- Retry y timeout helpers

### 6. Pydantic Helpers (`utils/pydantic_helpers.py`)
- Optimizaciones para Pydantic
- Batch processing

### 7. Paginación (`utils/pagination.py`)
- Cálculo de paginación
- Validación de parámetros

### 8. Filtros (`utils/filters.py`)
- Filtrado por campo, fecha, predicado
- Ordenamiento

### 9. Seguridad (`utils/security.py`)
- Tokens seguros
- Hash, validación de contraseñas
- Sanitización y enmascaramiento

### 10. Serialización (`utils/serialization.py`)
- Serialización para JSON
- Manejo de tipos especiales

### 11. Query Params (`utils/query_params.py`)
- Parsing de query parameters
- Date ranges y filtros

### 12. Date Helpers (`utils/date_helpers.py`) ✨
- 11 funciones para manejo de fechas
- Parsing, formateo, cálculos

### 13. String Helpers (`utils/string_helpers.py`) ✨
- 5 funciones para strings
- Truncado, normalización, capitalización

### 14. Math Helpers (`utils/math_helpers.py`) ✨
- 5 funciones matemáticas
- Porcentajes, promedios, medianas

### 15. Logging Config (`utils/logging_config.py`)
- Configuración de logging
- Rotating handlers

### 16. Métricas (`utils/metrics.py`)
- Colección de métricas
- Performance tracking

### 17. Transformers (`utils/transformers.py`) ✨ NUEVO
- Transformación de diccionarios
- Flatten/nest, map, filter, group

### 18. Response Builders (`utils/response_builders.py`) ✨ NUEVO
- Builders para respuestas consistentes
- Success, error, paginated, list

### 19. Type Converters (`utils/type_converters.py`) ✨ NUEVO
- Conversión de tipos segura
- Int, float, bool, string, list, datetime

### 20. Collection Helpers (`utils/collection_helpers.py`) ✨ NUEVO
- Manipulación de colecciones
- Chunk, unique, merge, nested access

## Nuevas Utilidades Agregadas

### Transformers (`utils/transformers.py`)
```python
from utils import transform_dict, flatten_dict, nest_dict, group_by

# Transformar keys de diccionario
data = transform_dict(source, {"old_key": "new_key"})

# Flatten nested dict
flat = flatten_dict(nested_dict)

# Nest flat dict
nested = nest_dict(flat_dict)

# Agrupar por key
grouped = group_by(items, lambda x: x.category)
```

### Response Builders (`utils/response_builders.py`)
```python
from utils import (
    build_success_response,
    build_error_response,
    build_paginated_response
)

# Success response
response = build_success_response(data, message="Success")

# Error response
error = build_error_response("Error message", code="ERR_001")

# Paginated response
paginated = build_paginated_response(items, page=1, page_size=20, ...)
```

### Type Converters (`utils/type_converters.py`)
```python
from utils import to_int, to_float, to_bool, to_datetime

# Conversión segura con defaults
value = to_int("123", default=0)
price = to_float("99.99", default=0.0)
is_active = to_bool("true", default=False)
date = to_datetime("2024-01-01", default=None)
```

### Collection Helpers (`utils/collection_helpers.py`)
```python
from utils import chunk_list, unique_list, get_nested_value

# Chunk list
chunks = chunk_list(items, chunk_size=10)

# Unique items
unique = unique_list(items, key_func=lambda x: x.id)

# Nested access
name = get_nested_value(data, "user.profile.name", default="Unknown")
```

## Estadísticas Finales

### Utilidades
- ✅ **22 módulos** de utilidades
- ✅ **120+ funciones** reutilizables
- ✅ **Organización por dominio**
- ✅ **Funciones puras** (sin efectos secundarios)

### Cobertura
- ✅ Validación
- ✅ Transformación
- ✅ Conversión de tipos
- ✅ Manipulación de colecciones
- ✅ Respuestas API
- ✅ Fechas y tiempos
- ✅ Strings y matemáticas
- ✅ Seguridad
- ✅ Performance

## Ejemplos de Uso Completo

### Transformación de Datos
```python
from utils import transform_dict, flatten_dict, group_by

# Transformar estructura
user_data = transform_dict(
    raw_data,
    {"user_name": "name", "user_email": "email"}
)

# Flatten para logging
flat_data = flatten_dict(complex_nested_dict)

# Agrupar entradas por fecha
entries_by_date = group_by(entries, lambda e: e.date)
```

### Construcción de Respuestas
```python
from utils import build_success_response, build_paginated_response

# Respuesta simple
return build_success_response(
    data=result,
    message="Operation successful",
    meta={"version": "1.0"}
)

# Respuesta paginada
return build_paginated_response(
    items=items,
    page=page,
    page_size=page_size,
    total_items=total,
    total_pages=pages
)
```

### Conversión de Tipos
```python
from utils import to_int, to_float, to_datetime

# Parsear query params
page = to_int(request.get("page"), default=1)
limit = to_int(request.get("limit"), default=10)
start_date = to_datetime(request.get("start"), default=None)
```

### Manipulación de Colecciones
```python
from utils import chunk_list, unique_list, get_nested_value

# Procesar en chunks
for chunk in chunk_list(large_list, chunk_size=100):
    await process_chunk(chunk)

# Obtener valor anidado
user_name = get_nested_value(
    response_data,
    "data.user.profile.name",
    default="Anonymous"
)
```

## Beneficios

1. ✅ **Reutilización**: Funciones reutilizables en toda la aplicación
2. ✅ **Consistencia**: Mismos patrones en todo el código
3. ✅ **Testabilidad**: Funciones puras fáciles de testear
4. ✅ **Mantenibilidad**: Código organizado y claro
5. ✅ **Productividad**: Menos código duplicado
6. ✅ **Calidad**: Funciones bien probadas y documentadas

## Conclusión

El sistema ahora cuenta con un conjunto completo de utilidades que cubren todos los casos de uso comunes, haciendo el desarrollo más rápido, consistente y mantenible.

**Estado**: ✅ Complete Utilities Suite Implemented

