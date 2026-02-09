# Fase 33: Refactorización de Web Content Extractor AI - OpenRouter Client

## Resumen

Esta fase refactoriza el `OpenRouterClient` en `web_content_extractor_ai` para eliminar duplicación en el manejo de errores y la construcción de prompts.

## Problemas Identificados

### 1. Manejo de Errores Duplicado
- **Ubicación**: `infrastructure/openrouter/client.py`
- **Problema**: El método `extract_content()` tiene ~30 líneas de manejo de errores duplicado para diferentes tipos de excepciones HTTP.
- **Impacto**: Código repetitivo, difícil de mantener, inconsistente.

### 2. Construcción de Prompts Inline
- **Ubicación**: `infrastructure/openrouter/client.py`
- **Problema**: El prompt de extracción está construido directamente en el método (~20 líneas).
- **Impacto**: Difícil de reutilizar, difícil de testear, viola SRP.

### 3. Lógica de Truncamiento Inline
- **Ubicación**: `infrastructure/openrouter/client.py`
- **Problema**: La lógica de truncamiento de contenido está inline.
- **Impacto**: No reutilizable, difícil de ajustar.

## Soluciones Implementadas

### 1. Creación de `error_handlers.py` ✅

**Ubicación**: Nuevo archivo `infrastructure/openrouter/error_handlers.py`

**Función**: `handle_openrouter_error()`
- Centraliza el manejo de errores HTTP, timeout y genéricos
- Proporciona mensajes de error consistentes
- Maneja la extracción de mensajes de error de respuestas JSON

**Antes**:
```python
except httpx.HTTPStatusError as e:
    error_msg = f"Error de API OpenRouter: {e.response.status_code}"
    try:
        error_data = e.response.json()
        error_detail = error_data.get("error", {})
        error_msg = error_detail.get("message", error_msg)
    except Exception:
        pass
    
    logger.error(f"Error OpenRouter ({e.response.status_code}): {error_msg}")
    raise Exception(f"Error al procesar con OpenRouter: {error_msg}")
except httpx.TimeoutException:
    logger.error(f"Timeout al llamar OpenRouter después de {self.timeout}s")
    raise Exception(f"Timeout al procesar con OpenRouter")
```

**Después**:
```python
except (httpx.HTTPStatusError, httpx.TimeoutException) as e:
    raise handle_openrouter_error(e, "extract content", timeout=self.timeout)
```

### 2. Creación de `prompt_builder.py` ✅

**Ubicación**: Nuevo archivo `infrastructure/openrouter/prompt_builder.py`

**Funciones**:
- `build_extraction_prompt()`: Construye el prompt de extracción
- `truncate_content()`: Trunca contenido si excede el límite

**Antes**:
```python
content_preview = web_content[:50000] if len(web_content) > 50000 else web_content

prompt = f"""Extrae toda la información relevante de esta página web.

URL: {url}

Contenido:
{content_preview}

Extrae y estructura la siguiente información:
1. Título principal
2. Descripción/resumen
...
"""
```

**Después**:
```python
content_preview = truncate_content(web_content)
prompt = build_extraction_prompt(url, content_preview)
```

### 3. Refactorización de `client.py` ✅

**Cambios**:
- Importa y usa `handle_openrouter_error()` para manejo de errores
- Importa y usa `build_extraction_prompt()` y `truncate_content()` para construcción de prompts
- Elimina ~50 líneas de código duplicado

## Métricas

### Reducción de Código
- **Líneas eliminadas**: ~50 líneas de código duplicado
- **Archivos nuevos**: 2 archivos de helpers
- **Métodos refactorizados**: 1 método (`extract_content`)

### Mejoras de Mantenibilidad
- **Consistencia**: Manejo de errores centralizado
- **Reutilización**: Prompts y truncamiento pueden ser reutilizados
- **Testabilidad**: Helpers pueden ser probados independientemente
- **SRP**: Cada módulo tiene una responsabilidad única

## Principios Aplicados

1. **DRY (Don't Repeat Yourself)**: Eliminación de código duplicado
2. **Single Responsibility Principle**: Cada helper tiene una responsabilidad única
3. **Separation of Concerns**: Separación de lógica de errores, prompts y cliente
4. **Mantenibilidad**: Cambios futuros solo requieren modificar un lugar

## Archivos Modificados/Creados

1. **`infrastructure/openrouter/error_handlers.py`** (NUEVO): Manejo centralizado de errores
2. **`infrastructure/openrouter/prompt_builder.py`** (NUEVO): Construcción de prompts
3. **`infrastructure/openrouter/client.py`**: Refactorizado para usar los nuevos helpers

## Compatibilidad

- ✅ **Backward Compatible**: La interfaz pública de `OpenRouterClient` no cambia
- ✅ **Sin Breaking Changes**: Los cambios son internos
- ✅ **Mismo comportamiento**: El comportamiento externo es idéntico

## Estado Final

- ✅ Manejo de errores centralizado
- ✅ Construcción de prompts extraída
- ✅ Lógica de truncamiento reutilizable
- ✅ Código más limpio y mantenible

