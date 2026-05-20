# Refactorización V26: Service Method Helper

## 📋 Resumen

Refactorización del módulo `contabilidad_mexicana_ai` para centralizar el patrón común de ejecución de servicios en `ServiceMethodHelper`, eliminando duplicación en la construcción de mensajes, llamadas API y formateo de respuestas.

---

## 🔍 Problemas Identificados

### 1. Patrón Repetitivo en Métodos de Servicio

**Problema**: Todos los métodos de servicio seguían el mismo patrón:
1. Construir prompt (usando `PromptBuilder`)
2. Construir mensajes (usando `MessageBuilder.build_messages`)
3. Llamar API (usando `api_handler.call_with_metrics`)
4. Renombrar campo de tiempo (usando `ResponseFormatter.rename_time_field`)

**Impacto**:
- ❌ Código duplicado en 6 métodos
- ❌ Difícil mantener (cambios requieren modificar múltiples lugares)
- ❌ Inconsistencia potencial

**Ejemplo de duplicación**:
```python
# ❌ Patrón repetido en múltiples métodos
prompt = PromptBuilder.build_advice_prompt(pregunta, contexto)

messages = MessageBuilder.build_messages(
    system_prompt=self.system_prompts["asesoria_fiscal"],
    user_prompt=prompt
)

service_data = {"pregunta": pregunta}

result = await self.api_handler.call_with_metrics(
    messages=messages,
    service_name="asesoria_fiscal",
    service_data=service_data,
    extract_key="asesoria"
)
```

### 2. Prompt Hardcodeado en `comparar_regimenes`

**Problema**: El método `comparar_regimenes` tenía un prompt hardcodeado en lugar de usar `PromptBuilder`, siendo inconsistente con el resto de los métodos.

**Impacto**:
- ❌ Inconsistencia con otros métodos
- ❌ Difícil mantener
- ❌ No reutiliza código existente

### 3. `ServiceMethodHelper` Existente pero No Utilizado

**Problema**: Ya existía un archivo `service_method_helper.py` pero no se estaba usando en ningún método.

**Impacto**:
- ❌ Código muerto
- ❌ Oportunidad perdida de reducir duplicación

---

## ✅ Soluciones Implementadas

### 1. Mejora de `ServiceMethodHelper`

**Archivo**: `core/service_method_helper.py`

**Cambios**:
- Simplificado para aceptar prompt ya construido (en lugar de método de construcción)
- Integrado con `ResponseFormatter` para renombrar campos de tiempo
- API más clara y fácil de usar

**Antes**:
```python
async def execute_service(
    prompt_builder_method: Callable[..., str],
    system_prompt_key: str,
    # ... muchos parámetros
    *prompt_args,
    **prompt_kwargs
) -> Dict[str, Any]:
    prompt = prompt_builder_method(*prompt_args, **prompt_kwargs)
    # ...
```

**Después**:
```python
async def execute_service(
    prompt: str,  # Prompt ya construido
    system_prompt: str,  # System prompt directo
    api_handler: APIHandler,
    service_name: str,
    service_data: Dict[str, Any],
    extract_key: str = "resultado",
    temperature: Optional[float] = None,
    time_field_rename: Optional[str] = None
) -> Dict[str, Any]:
    # Construye mensajes, llama API, formatea respuesta
```

**Beneficios**:
- ✅ API más simple y clara
- ✅ Menos parámetros
- ✅ Más flexible (permite modificar prompt antes de pasar)

### 2. Agregado `build_comparison_prompt` a `PromptBuilder`

**Archivo**: `core/prompt_builder.py`

**Cambio**: Agregado método estático `build_comparison_prompt()` para construir prompts de comparación de regímenes.

**Antes** (`comparar_regimenes`):
```python
prompt = f"""Compara los siguientes regímenes fiscales para un contribuyente:

Regímenes a comparar: {', '.join(regimenes)}

Datos del contribuyente:
{PromptBuilder._format_data(datos)}

Proporciona:
1. Comparación de carga fiscal para cada régimen
# ... más texto hardcodeado
"""
```

**Después**:
```python
prompt = PromptBuilder.build_comparison_prompt(regimenes, datos)
```

**Beneficios**:
- ✅ Consistencia con otros métodos
- ✅ Reutiliza `_format_data`
- ✅ Fácil mantener y modificar

### 3. Refactorización de Métodos de Servicio

**Archivo**: `core/contador_ai.py`

**Cambios**: Todos los métodos ahora usan `ServiceMethodHelper.execute_service()` en lugar de construir mensajes y llamar API manualmente.

**Antes** (`asesoria_fiscal`):
```python
prompt = PromptBuilder.build_advice_prompt(pregunta, contexto)

messages = MessageBuilder.build_messages(
    system_prompt=self.system_prompts["asesoria_fiscal"],
    user_prompt=prompt
)

service_data = {"pregunta": pregunta}

result = await self.api_handler.call_with_metrics(
    messages=messages,
    service_name="asesoria_fiscal",
    service_data=service_data,
    extract_key="asesoria"
)
```

**Después**:
```python
prompt = PromptBuilder.build_advice_prompt(pregunta, contexto)

service_data = {"pregunta": pregunta}

result = await ServiceMethodHelper.execute_service(
    prompt=prompt,
    system_prompt=self.system_prompts["asesoria_fiscal"],
    api_handler=self.api_handler,
    service_name="asesoria_fiscal",
    service_data=service_data,
    extract_key="asesoria"
)
```

**Métodos Refactorizados**:
- ✅ `calcular_impuestos()` - Usa `ServiceMethodHelper` con `time_field_rename="tiempo_calculo"`
- ✅ `asesoria_fiscal()` - Usa `ServiceMethodHelper`
- ✅ `guia_fiscal()` - Usa `ServiceMethodHelper` con `temperature=0.5` y `time_field_rename="tiempo_generacion"`
- ✅ `tramite_sat()` - Usa `ServiceMethodHelper`
- ✅ `ayuda_declaracion()` - Usa `ServiceMethodHelper`
- ✅ `comparar_regimenes()` - Usa `ServiceMethodHelper` y `PromptBuilder.build_comparison_prompt()`

**Beneficios**:
- ✅ Reducción significativa de código duplicado
- ✅ Consistencia garantizada
- ✅ Fácil mantener (cambios en un solo lugar)
- ✅ Métodos más cortos y legibles

---

## 📊 Métricas de Mejora

### Reducción de Duplicación

| Aspecto | Antes | Después | Mejora |
|---------|-------|---------|--------|
| Líneas por método (prompt → API) | ~12 líneas | ~8 líneas | ✅ **-33%** |
| Construcción de mensajes duplicada | 6 métodos | 0 métodos | ✅ **-100%** |
| Llamadas API duplicadas | 6 métodos | 0 métodos | ✅ **-100%** |
| Renombrado de campos duplicado | 2 métodos | 0 métodos | ✅ **-100%** |

### Consistencia

| Método | PromptBuilder | ServiceMethodHelper | ResponseFormatter |
|--------|---------------|---------------------|-------------------|
| `calcular_impuestos` | ✅ | ✅ | ✅ (vía helper) |
| `asesoria_fiscal` | ✅ | ✅ | ✅ (vía helper) |
| `guia_fiscal` | ✅ | ✅ | ✅ (vía helper) |
| `tramite_sat` | ✅ | ✅ | ✅ (vía helper) |
| `ayuda_declaracion` | ✅ | ✅ | ✅ (vía helper) |
| `comparar_regimenes` | ✅ Agregado | ✅ | ✅ (vía helper) |

---

## 🎯 Principios Aplicados

### 1. DRY (Don't Repeat Yourself)

**Aplicación**:
- ✅ Patrón común centralizado en `ServiceMethodHelper`
- ✅ Sin duplicación de construcción de mensajes
- ✅ Sin duplicación de llamadas API
- ✅ Sin duplicación de formateo de respuestas

**Beneficios**:
- ✅ Single source of truth
- ✅ Fácil mantener
- ✅ Consistencia garantizada

### 2. Single Responsibility Principle (SRP)

**Aplicación**:
- ✅ `ServiceMethodHelper`: Solo ejecuta el patrón común de servicios
- ✅ `PromptBuilder`: Solo construye prompts
- ✅ `MessageBuilder`: Solo construye mensajes
- ✅ `APIHandler`: Solo maneja llamadas API
- ✅ `ResponseFormatter`: Solo formatea respuestas
- ✅ Métodos de servicio: Solo orquestan la lógica de negocio

**Beneficios**:
- ✅ Responsabilidades claras
- ✅ Fácil testear
- ✅ Fácil modificar

### 3. Open/Closed Principle

**Aplicación**:
- ✅ `ServiceMethodHelper` es extensible (puede agregar nuevos parámetros sin romper código existente)
- ✅ Los métodos de servicio están cerrados para modificación pero abiertos para extensión

**Beneficios**:
- ✅ Fácil agregar nuevas funcionalidades
- ✅ No rompe código existente

---

## 📝 Archivos Modificados

### Archivos Modificados

1. **`core/service_method_helper.py`**
   - Simplificado para aceptar prompt ya construido
   - Integrado con `ResponseFormatter`
   - API más clara y fácil de usar

2. **`core/prompt_builder.py`**
   - Agregado método `build_comparison_prompt()` para comparación de regímenes

3. **`core/contador_ai.py`**
   - Importado `ServiceMethodHelper`
   - Refactorizado `calcular_impuestos()` para usar `ServiceMethodHelper`
   - Refactorizado `asesoria_fiscal()` para usar `ServiceMethodHelper`
   - Refactorizado `guia_fiscal()` para usar `ServiceMethodHelper`
   - Refactorizado `tramite_sat()` para usar `ServiceMethodHelper`
   - Refactorizado `ayuda_declaracion()` para usar `ServiceMethodHelper`
   - Refactorizado `comparar_regimenes()` para usar `ServiceMethodHelper` y `PromptBuilder.build_comparison_prompt()`

---

## 🧪 Testing

### Casos de Prueba Sugeridos

1. **ServiceMethodHelper.execute_service()**
   - ✅ Construye mensajes correctamente
   - ✅ Llama API handler con parámetros correctos
   - ✅ Renombra campo de tiempo si se especifica
   - ✅ Maneja temperatura opcional
   - ✅ Maneja extract_key correctamente

2. **PromptBuilder.build_comparison_prompt()**
   - ✅ Construye prompt correctamente
   - ✅ Formatea datos correctamente
   - ✅ Incluye todos los regímenes

3. **Métodos de Servicio**
   - ✅ Usan `ServiceMethodHelper` correctamente
   - ✅ Pasan parámetros correctos
   - ✅ Manejan campos de tiempo correctamente

---

## 📈 Beneficios Finales

### Mantenibilidad
- ✅ Cambios en patrón común en un solo lugar
- ✅ Fácil agregar nuevos servicios
- ✅ Código más limpio y legible

### Consistencia
- ✅ Todos los métodos usan el mismo patrón
- ✅ Comportamiento predecible
- ✅ Fácil entender

### Testabilidad
- ✅ `ServiceMethodHelper` puede ser testeado independientemente
- ✅ Métodos de servicio más fáciles de testear (mock de `ServiceMethodHelper`)
- ✅ Menos acoplamiento

### Extensibilidad
- ✅ Fácil agregar nuevas funcionalidades al patrón común
- ✅ Fácil modificar comportamiento sin afectar servicios individuales

---

## 🔄 Próximos Pasos Sugeridos

1. **Agregar Validación de Parámetros**
   - Considerar agregar validación de parámetros en `ServiceMethodHelper`
   - Facilitar debugging y errores más claros

2. **Agregar Logging**
   - Considerar agregar logging en `ServiceMethodHelper`
   - Facilitar debugging y monitoreo

3. **Agregar Métricas**
   - Considerar agregar métricas en `ServiceMethodHelper`
   - Facilitar monitoreo de rendimiento

4. **Unificar Manejo de Calculadoras**
   - `calcular_impuestos` y `comparar_regimenes` tienen lógica similar para calculadoras especializadas
   - Considerar extraer a un helper común

---

## ✅ Checklist de Refactorización

- [x] Mejorar `ServiceMethodHelper` con API más simple
- [x] Agregar `build_comparison_prompt()` a `PromptBuilder`
- [x] Refactorizar `calcular_impuestos()` para usar `ServiceMethodHelper`
- [x] Refactorizar `asesoria_fiscal()` para usar `ServiceMethodHelper`
- [x] Refactorizar `guia_fiscal()` para usar `ServiceMethodHelper`
- [x] Refactorizar `tramite_sat()` para usar `ServiceMethodHelper`
- [x] Refactorizar `ayuda_declaracion()` para usar `ServiceMethodHelper`
- [x] Refactorizar `comparar_regimenes()` para usar `ServiceMethodHelper` y `PromptBuilder`
- [x] Verificar que no hay errores de linter
- [x] Documentar cambios

---

**Refactorización completada**: ✅ V26 - Service Method Helper

