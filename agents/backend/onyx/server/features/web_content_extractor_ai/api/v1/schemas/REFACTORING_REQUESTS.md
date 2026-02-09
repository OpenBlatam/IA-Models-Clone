# Refactorización de requests.py

## 📋 Resumen

Refactorización de `requests.py` aplicando principios SOLID, DRY y mejores prácticas, eliminando duplicación de código y usando constantes centralizadas.

## 🎯 Mejoras Aplicadas

### 1. **Eliminación de Comentarios de Prueba**

**Antes:**
```python
mantenlo simple
modular
mas modualr
refactor
mas
refactor
mas
"""
Schemas de requests para la API
"""
```

**Después:**
```python
"""
Schemas de requests para la API
"""
```

**Beneficios:**
- ✅ Código limpio y profesional
- ✅ Sin comentarios de desarrollo

### 2. **Clase Base para Eliminar Duplicación (DRY)**

**Antes:**
```python
class ExtractContentRequest(BaseModel):
    # ... otros campos ...
    use_javascript: Optional[bool] = Field(
        default=False,
        description="Si True, usa Playwright para renderizar JavaScript (más lento pero más completo)"
    )
    extract_strategy: Optional[str] = Field(
        default="auto",
        description="Estrategia de extracción: 'auto', 'trafilatura', 'readability', 'newspaper', 'beautifulsoup'"
    )

class BatchExtractRequest(BaseModel):
    # ... otros campos ...
    use_javascript: Optional[bool] = Field(
        default=False,
        description="Si True, usa Playwright para renderizar JavaScript"
    )
    extract_strategy: Optional[str] = Field(
        default="auto",
        description="Estrategia de extracción"
    )
```

**Después:**
```python
class BaseExtractionRequest(BaseModel):
    """
    Clase base para requests de extracción.
    Contiene campos comunes compartidos entre ExtractContentRequest y BatchExtractRequest.
    """
    use_javascript: Optional[bool] = Field(
        default=False,
        description="Si True, usa Playwright para renderizar JavaScript (más lento pero más completo)"
    )
    extract_strategy: Optional[str] = Field(
        default=DEFAULT_EXTRACT_STRATEGY,
        description=f"Estrategia de extracción. Opciones: {', '.join(EXTRACT_STRATEGIES)}"
    )

    @field_validator('extract_strategy')
    @classmethod
    def validate_extract_strategy(cls, v: Optional[str]) -> Optional[str]:
        """Valida que la estrategia de extracción sea válida."""
        if v is not None and v not in EXTRACT_STRATEGIES:
            raise ValueError(
                f"extract_strategy debe ser uno de: {', '.join(EXTRACT_STRATEGIES)}"
            )
        return v

class ExtractContentRequest(BaseExtractionRequest):
    # ... solo campos específicos ...

class BatchExtractRequest(BaseExtractionRequest):
    # ... solo campos específicos ...
```

**Beneficios:**
- ✅ Eliminación de duplicación (DRY)
- ✅ Validación centralizada
- ✅ Más fácil de mantener
- ✅ Cambios en campos comunes se hacen en un solo lugar

### 3. **Uso de Constantes en Lugar de Valores Hardcodeados**

**Antes:**
```python
model: Optional[str] = Field(
    default="anthropic/claude-3.5-sonnet",  # Hardcoded
    description="Modelo de OpenRouter a usar"
)
max_tokens: Optional[int] = Field(
    default=4000,  # Hardcoded
    ge=100,  # Hardcoded
    le=8000,  # Hardcoded
    description="Máximo de tokens en la respuesta"
)
extract_strategy: Optional[str] = Field(
    default="auto",  # Hardcoded
    description="Estrategia de extracción: 'auto', 'trafilatura', 'readability', 'newspaper', 'beautifulsoup'"
)
urls: List[HttpUrl] = Field(
    ...,
    description="Lista de URLs a extraer",
    min_items=1,  # Hardcoded
    max_items=50  # Hardcoded
)
max_concurrent: Optional[int] = Field(
    default=5,  # Hardcoded
    ge=1,  # Hardcoded
    le=20,  # Hardcoded
    description="Máximo de requests concurrentes"
)
```

**Después:**
```python
from .constants import (
    DEFAULT_MODEL,
    DEFAULT_MAX_TOKENS,
    MIN_MAX_TOKENS,
    MAX_MAX_TOKENS,
    EXTRACT_STRATEGIES,
    DEFAULT_EXTRACT_STRATEGY,
    DEFAULT_MAX_CONCURRENT,
    MIN_MAX_CONCURRENT,
    MAX_MAX_CONCURRENT,
    MAX_BATCH_URLS,
    MIN_BATCH_URLS,
)

model: Optional[str] = Field(
    default=DEFAULT_MODEL,  # Constante
    description="Modelo de OpenRouter a usar"
)
max_tokens: Optional[int] = Field(
    default=DEFAULT_MAX_TOKENS,  # Constante
    ge=MIN_MAX_TOKENS,  # Constante
    le=MAX_MAX_TOKENS,  # Constante
    description=f"Máximo de tokens en la respuesta (entre {MIN_MAX_TOKENS} y {MAX_MAX_TOKENS})"
)
extract_strategy: Optional[str] = Field(
    default=DEFAULT_EXTRACT_STRATEGY,  # Constante
    description=f"Estrategia de extracción. Opciones: {', '.join(EXTRACT_STRATEGIES)}"
)
urls: List[HttpUrl] = Field(
    ...,
    description="Lista de URLs a extraer",
    min_length=MIN_BATCH_URLS,  # Constante
    max_length=MAX_BATCH_URLS  # Constante
)
max_concurrent: Optional[int] = Field(
    default=DEFAULT_MAX_CONCURRENT,  # Constante
    ge=MIN_MAX_CONCURRENT,  # Constante
    le=MAX_MAX_CONCURRENT,  # Constante
    description=f"Máximo de requests concurrentes (entre {MIN_MAX_CONCURRENT} y {MAX_MAX_CONCURRENT})"
)
```

**Beneficios:**
- ✅ Valores centralizados
- ✅ Fácil de modificar
- ✅ Consistencia en toda la aplicación
- ✅ Descripciones más informativas con valores dinámicos

### 4. **Validación Mejorada**

**Antes:**
```python
extract_strategy: Optional[str] = Field(
    default="auto",
    description="Estrategia de extracción: 'auto', 'trafilatura', 'readability', 'newspaper', 'beautifulsoup'"
)
# Sin validación explícita
```

**Después:**
```python
extract_strategy: Optional[str] = Field(
    default=DEFAULT_EXTRACT_STRATEGY,
    description=f"Estrategia de extracción. Opciones: {', '.join(EXTRACT_STRATEGIES)}"
)

@field_validator('extract_strategy')
@classmethod
def validate_extract_strategy(cls, v: Optional[str]) -> Optional[str]:
    """Valida que la estrategia de extracción sea válida."""
    if v is not None and v not in EXTRACT_STRATEGIES:
        raise ValueError(
            f"extract_strategy debe ser uno de: {', '.join(EXTRACT_STRATEGIES)}"
        )
    return v
```

**Beneficios:**
- ✅ Validación explícita
- ✅ Mensajes de error claros
- ✅ Prevención de valores inválidos

### 5. **Mejora en Descripciones de Campos**

**Antes:**
```python
description="Máximo de tokens en la respuesta"
description="Máximo de requests concurrentes"
```

**Después:**
```python
description=f"Máximo de tokens en la respuesta (entre {MIN_MAX_TOKENS} y {MAX_MAX_TOKENS})"
description=f"Máximo de requests concurrentes (entre {MIN_MAX_CONCURRENT} y {MAX_MAX_CONCURRENT})"
```

**Beneficios:**
- ✅ Descripciones más informativas
- ✅ Valores dinámicos basados en constantes
- ✅ Mejor documentación automática

## 📊 Métricas

| Aspecto | Antes | Después | Mejora |
|---------|-------|---------|--------|
| **Líneas de código** | 57 | 75 | **+18** (pero más mantenible) |
| **Clases** | 2 | 3 | **+1** (clase base) |
| **Duplicación de campos** | 2 campos duplicados | 0 | **-100%** |
| **Valores hardcodeados** | 10+ | 0 | **-100%** |
| **Validaciones** | 0 | 1 | **+1** |
| **Uso de constantes** | 0% | 100% | **+100%** |

## ✅ Principios Aplicados

### DRY (Don't Repeat Yourself)
- ✅ Campos comunes extraídos a clase base
- ✅ Constantes centralizadas

### Single Responsibility Principle (SRP)
- ✅ `BaseExtractionRequest` - Solo campos comunes
- ✅ `ExtractContentRequest` - Solo campos específicos de extracción única
- ✅ `BatchExtractRequest` - Solo campos específicos de batch

### Open/Closed Principle (OCP)
- ✅ Fácil extender con nuevas clases que hereden de `BaseExtractionRequest`

### Don't Repeat Yourself (DRY)
- ✅ Validación centralizada
- ✅ Constantes reutilizables

### Clean Code
- ✅ Constantes en lugar de valores hardcodeados
- ✅ Validación explícita
- ✅ Descripciones informativas

## 🎯 Estado Final

✅ **Comentarios de Prueba Eliminados**  
✅ **Clase Base Creada**  
✅ **Duplicación Eliminada**  
✅ **Constantes Implementadas**  
✅ **Validación Mejorada**  
✅ **Código Más Limpio y Mantenible**  

## 📝 Archivos Modificados

1. **`requests.py`**
   - ✅ Eliminados comentarios de prueba
   - ✅ Creada clase base `BaseExtractionRequest`
   - ✅ `ExtractContentRequest` ahora hereda de `BaseExtractionRequest`
   - ✅ `BatchExtractRequest` ahora hereda de `BaseExtractionRequest`
   - ✅ Uso de constantes de `constants.py`
   - ✅ Validación de `extract_strategy` agregada
   - ✅ Descripciones mejoradas con valores dinámicos

## ✨ Conclusión

El código está ahora completamente refactorizado, eliminando duplicación y usando constantes centralizadas. La clase base facilita el mantenimiento y la extensión futura.

**Refactorización completa.** 🎉

