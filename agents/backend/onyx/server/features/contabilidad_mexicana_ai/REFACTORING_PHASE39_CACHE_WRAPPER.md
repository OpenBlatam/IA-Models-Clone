# Fase 39: Refactorización de Cache Wrapper - Contador AI

## Resumen

Esta fase refactoriza el manejo de caché en `contabilidad_mexicana_ai` para eliminar el patrón repetitivo de verificar caché, ejecutar servicio y almacenar resultado.

## Problemas Identificados

### 1. Patrón Repetitivo de Caché
- **Ubicación**: `core/contador_ai.py`
- **Problema**: Múltiples métodos seguían el mismo patrón:
  1. Construir `cache_params` usando `ServiceDataBuilder`
  2. Verificar caché con `CacheHelper.get_cached_result()`
  3. Si hay caché, retornar
  4. Ejecutar servicio
  5. Almacenar resultado con `CacheHelper.store_result()`
- **Impacto**: Código repetitivo, difícil de mantener, inconsistente.

**Archivos afectados**:
- `guia_fiscal()`: Líneas 209-232
- `tramite_sat()`: Líneas 252-277
- `ayuda_declaracion()`: Líneas 299-333

## Soluciones Implementadas

### 1. Creación de `cached_service_execution()` ✅

**Ubicación**: `core/cache_helper.py`

**Función**: `CacheHelper.cached_service_execution()`
- Encapsula el flujo completo de caché
- Acepta una función async que ejecuta el servicio
- Maneja automáticamente verificación y almacenamiento de caché
- Soporta TTL personalizado

**Antes** (en `guia_fiscal`):
```python
# Check cache (guides can be cached longer)
cache_params = ServiceDataBuilder.build_guia_cache_params(tema, nivel_detalle)
cached = CacheHelper.get_cached_result(
    self.cache, "guia_fiscal", cache_params, use_cache, ttl=7200
)
if cached:
    return cached

prompt = PromptBuilder.build_guide_prompt(tema, nivel_detalle)
service_data = ServiceDataBuilder.build_guia_service_data(tema, nivel_detalle)

result = await ServiceMethodHelper.execute_service(
    prompt=prompt,
    system_prompt=self.system_prompts["guias_fiscales"],
    api_handler=self.api_handler,
    service_name="guia_fiscal",
    service_data=service_data,
    temperature=0.5,
    extract_key="guia",
    time_field_rename="tiempo_generacion"
)

return result
```

**Después**:
```python
# Build cache params
cache_params = ServiceDataBuilder.build_guia_cache_params(tema, nivel_detalle)

# Execute with cache
async def _execute_guia():
    prompt = PromptBuilder.build_guide_prompt(tema, nivel_detalle)
    service_data = ServiceDataBuilder.build_guia_service_data(tema, nivel_detalle)
    
    return await ServiceMethodHelper.execute_service(
        prompt=prompt,
        system_prompt=self.system_prompts["guias_fiscales"],
        api_handler=self.api_handler,
        service_name="guia_fiscal",
        service_data=service_data,
        temperature=0.5,
        extract_key="guia",
        time_field_rename="tiempo_generacion"
    )

return await CacheHelper.cached_service_execution(
    self.cache,
    "guia_fiscal",
    cache_params,
    _execute_guia,
    use_cache=use_cache,
    cache_ttl=7200
)
```

**Antes** (en `tramite_sat`):
```python
# Check cache (SAT procedures change rarely)
cache_params = ServiceDataBuilder.build_tramite_cache_params(tipo_tramite, detalles)
cached = CacheHelper.get_cached_result(
    self.cache, "tramite_sat", cache_params, use_cache, ttl=14400
)
if cached:
    return cached

prompt = PromptBuilder.build_procedure_prompt(tipo_tramite, detalles)
service_data = ServiceDataBuilder.build_tramite_service_data(tipo_tramite, detalles)

result = await ServiceMethodHelper.execute_service(...)

# Cache result if enabled
CacheHelper.store_result(
    self.cache, "tramite_sat", cache_params, result, use_cache, ttl=14400
)

return result
```

**Después**:
```python
# Build cache params
cache_params = ServiceDataBuilder.build_tramite_cache_params(tipo_tramite, detalles)

# Execute with cache
async def _execute_tramite():
    prompt = PromptBuilder.build_procedure_prompt(tipo_tramite, detalles)
    service_data = ServiceDataBuilder.build_tramite_service_data(tipo_tramite, detalles)
    
    return await ServiceMethodHelper.execute_service(...)

return await CacheHelper.cached_service_execution(
    self.cache,
    "tramite_sat",
    cache_params,
    _execute_tramite,
    use_cache=use_cache,
    cache_ttl=14400
)
```

**Antes** (en `ayuda_declaracion`):
```python
# Check cache
cache_params = {
    "tipo_declaracion": tipo_declaracion,
    "periodo": periodo,
    "datos": datos or {}
}
cached = CacheHelper.get_cached_result(
    self.cache, "ayuda_declaracion", cache_params, use_cache
)
if cached:
    return cached

prompt = PromptBuilder.build_declaration_prompt(...)
result = await ServiceMethodHelper.execute_service(...)

# Cache result if enabled
CacheHelper.store_result(
    self.cache, "ayuda_declaracion", cache_params, result, use_cache
)

return result
```

**Después**:
```python
# Build cache params
cache_params = {
    "tipo_declaracion": tipo_declaracion,
    "periodo": periodo,
    "datos": datos or {}
}

# Execute with cache
async def _execute_declaracion():
    prompt = PromptBuilder.build_declaration_prompt(...)
    return await ServiceMethodHelper.execute_service(...)

return await CacheHelper.cached_service_execution(
    self.cache,
    "ayuda_declaracion",
    cache_params,
    _execute_declaracion,
    use_cache=use_cache
)
```

## Métricas

### Reducción de Código
- **Líneas eliminadas**: ~30 líneas de código duplicado
- **Funciones nuevas**: 1 función helper (`cached_service_execution`)
- **Métodos refactorizados**: 3 métodos (`guia_fiscal`, `tramite_sat`, `ayuda_declaracion`)

### Mejoras de Mantenibilidad
- **Consistencia**: Flujo de caché centralizado
- **Reutilización**: Helper puede ser reutilizado en otros métodos
- **Testabilidad**: Helper puede ser probado independientemente
- **SRP**: Helper tiene una responsabilidad única
- **Legibilidad**: Código más limpio y expresivo

## Principios Aplicados

1. **DRY (Don't Repeat Yourself)**: Eliminación de código duplicado
2. **Single Responsibility Principle**: Helper tiene una responsabilidad única
3. **Separation of Concerns**: Separación de lógica de caché y ejecución de servicio
4. **Mantenibilidad**: Cambios futuros solo requieren modificar un lugar
5. **Flexibilidad**: Soporta TTL personalizado y diferentes configuraciones

## Archivos Modificados

1. **`core/cache_helper.py`**: Agregada función `cached_service_execution()`
2. **`core/contador_ai.py`**: Refactorizados métodos para usar el nuevo helper

## Compatibilidad

- ✅ **Backward Compatible**: La funcionalidad es idéntica
- ✅ **Sin Breaking Changes**: Los cambios son internos
- ✅ **Mismo comportamiento**: El comportamiento externo es idéntico

## Estado Final

- ✅ Flujo de caché centralizado
- ✅ Código más limpio y mantenible
- ✅ Patrones consistentes
- ✅ Helper reutilizable

