# Fase 34: Refactorización de Contabilidad Mexicana AI SAM3

## Resumen

Esta fase refactoriza el módulo `contabilidad_mexicana_ai_sam3` para eliminar duplicación en la API y en los handlers de servicios.

## Problemas Identificados

### 1. Duplicación en API Endpoints
- **Ubicación**: `api/contador_sam3_api.py`
- **Problema**: Todos los endpoints tienen el mismo patrón:
  - Verificación `if not _agent: raise HTTPException`
  - Retorno del mismo formato `{"task_id": task_id, "status": "submitted"}`
  - Manejo de errores duplicado en endpoints de task status/result
- **Impacto**: ~50 líneas de código duplicado, difícil de mantener.

### 2. Duplicación en Service Handlers
- **Ubicación**: `core/contador_sam3_agent.py` - métodos `_handle_*`
- **Problema**: Los 5 métodos `_handle_*` tienen el mismo patrón:
  - Construir prompt usando PromptBuilder
  - Optimizar con TruthGPT
  - Crear mensajes con system prompt
  - Llamar a OpenRouter
  - Retornar resultado con formato similar
- **Impacto**: ~150 líneas de código duplicado, violación de DRY.

## Soluciones Implementadas

### 1. Creación de `api_helpers.py` ✅

**Ubicación**: Nuevo archivo `api/api_helpers.py`

**Funciones**:
- `require_agent()`: Verifica que el agent esté inicializado
- `create_task_response()`: Crea respuesta estándar para tareas
- `handle_task_operation()`: Maneja operaciones con error handling consistente

**Antes** (repetido 5 veces):
```python
@app.post("/calcular-impuestos")
async def calcular_impuestos(request: CalcularImpuestosRequest):
    if not _agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    task_id = await _agent.calcular_impuestos(...)
    
    return {"task_id": task_id, "status": "submitted"}
```

**Después**:
```python
@app.post("/calcular-impuestos")
async def calcular_impuestos(request: CalcularImpuestosRequest):
    agent = require_agent(_agent)
    
    task_id = await agent.calcular_impuestos(...)
    
    return create_task_response(task_id)
```

### 2. Creación de `service_handler.py` ✅

**Ubicación**: Nuevo archivo `core/service_handler.py`

**Clase**: `ServiceHandler`
- Centraliza el patrón común de los métodos `_handle_*`
- Método `handle_service_request()` que encapsula:
  - Construcción de prompts
  - Optimización con TruthGPT
  - Creación de mensajes
  - Llamada a OpenRouter
  - Formato de respuesta

**Antes** (repetido 5 veces, ~30 líneas cada uno):
```python
async def _handle_calcular_impuestos(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
    regimen = parameters.get("regimen")
    tipo_impuesto = parameters.get("tipo_impuesto")
    datos = parameters.get("datos", {})
    
    prompt = PromptBuilder.build_calculation_prompt(regimen, tipo_impuesto, datos)
    
    # Optimize query with TruthGPT
    optimized_prompt = await self.truthgpt_client.optimize_query(prompt)
    
    messages = [
        create_message("system", self.system_prompts["calculo_impuestos"]),
        create_message("user", optimized_prompt)
    ]
    
    response = await self.openrouter_client.chat_completion(
        model=self.config.openrouter.model,
        messages=messages,
        temperature=0.3,
        max_tokens=4000,
    )
    
    return {
        "resultado": response["response"],
        "tokens_used": response["tokens_used"],
        "model": response["model"],
        "tiempo_calculo": datetime.now().isoformat(),
    }
```

**Después**:
```python
async def _handle_calcular_impuestos(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
    regimen = parameters.get("regimen")
    tipo_impuesto = parameters.get("tipo_impuesto")
    datos = parameters.get("datos", {})
    
    return await self.service_handler.handle_service_request(
        service_type="calcular_impuestos",
        prompt_builder_method=PromptBuilder.build_calculation_prompt,
        system_prompt_key="calculo_impuestos",
        response_key="resultado",
        temperature=0.3,
        include_timestamp=True,
        timestamp_key="tiempo_calculo",
        regimen=regimen,
        tipo_impuesto=tipo_impuesto,
        datos=datos
    )
```

## Métricas

### Reducción de Código
- **Líneas eliminadas**: ~200 líneas de código duplicado
- **Archivos nuevos**: 2 archivos de helpers
- **Endpoints refactorizados**: 7 endpoints
- **Service handlers refactorizados**: 5 métodos

### Mejoras de Mantenibilidad
- **Consistencia**: Todos los endpoints y handlers siguen el mismo patrón
- **Reutilización**: Helpers pueden ser reutilizados en futuros endpoints
- **Testabilidad**: Helpers pueden ser probados independientemente
- **SRP**: Cada helper tiene una responsabilidad única

## Principios Aplicados

1. **DRY (Don't Repeat Yourself)**: Eliminación de código duplicado
2. **Single Responsibility Principle**: Cada helper tiene una responsabilidad única
3. **Separation of Concerns**: Separación de lógica de API, servicios y handlers
4. **Mantenibilidad**: Cambios futuros solo requieren modificar un lugar

## Archivos Modificados/Creados

1. **`api/api_helpers.py`** (NUEVO): Helpers para API endpoints
2. **`core/service_handler.py`** (NUEVO): Handler centralizado para servicios
3. **`api/contador_sam3_api.py`**: Refactorizado para usar `api_helpers`
4. **`core/contador_sam3_agent.py`**: Refactorizado para usar `ServiceHandler`

## Compatibilidad

- ✅ **Backward Compatible**: Todas las interfaces públicas mantienen su formato
- ✅ **Sin Breaking Changes**: Los cambios son internos
- ✅ **Mismo comportamiento**: El comportamiento externo es idéntico

## Estado Final

- ✅ API endpoints consolidados
- ✅ Service handlers centralizados
- ✅ Código más limpio y mantenible
- ✅ Patrones consistentes en todo el módulo

