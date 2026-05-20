# Análisis de Refactorización - ContadorSAM3Agent

## 📋 Resumen

Análisis del código actual para identificar problemas y oportunidades de mejora aplicando principios SOLID y DRY.

---

## 🔍 Problemas Identificados

### Problema 1: Duplicación Masiva en Métodos `_handle_*`

**Problema**: Los 5 métodos `_handle_*` tienen código casi idéntico

**Patrón Duplicado**:
```python
async def _handle_<servicio>(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
    # 1. Extraer parámetros
    param1 = parameters.get("param1")
    param2 = parameters.get("param2")
    
    # 2. Construir prompt
    prompt = PromptBuilder.build_<tipo>_prompt(param1, param2)
    
    # 3. Optimizar con TruthGPT
    optimized_prompt = await self.truthgpt_client.optimize_query(prompt)
    
    # 4. Crear mensajes
    messages = [
        create_message("system", self.system_prompts["<tipo>"]),
        create_message("user", optimized_prompt)
    ]
    
    # 5. Llamar API
    response = await self.openrouter_client.chat_completion(
        model=self.config.openrouter.model,
        messages=messages,
        temperature=0.3,  # o 0.5
        max_tokens=4000,
    )
    
    # 6. Retornar resultado
    return {
        "<clave>": response["response"],
        "tokens_used": response["tokens_used"],
        "model": response["model"],
        # ... campos adicionales opcionales
    }
```

**Impacto**:
- ❌ ~30 líneas duplicadas × 5 métodos = ~150 líneas duplicadas
- ❌ Difícil mantener (cambios requieren modificar 5 lugares)
- ❌ Fácil olvidar actualizar uno

---

### Problema 2: Routing Manual con if/elif

**Problema**: `_process_task` usa if/elif largo para routing

```python
if service_type == "calcular_impuestos":
    result = await self._handle_calcular_impuestos(parameters)
elif service_type == "asesoria_fiscal":
    result = await self._handle_asesoria_fiscal(parameters)
elif service_type == "guia_fiscal":
    result = await self._handle_guia_fiscal(parameters)
elif service_type == "tramite_sat":
    result = await self._handle_tramite_sat(parameters)
elif service_type == "ayuda_declaracion":
    result = await self._handle_ayuda_declaracion(parameters)
else:
    raise ValueError(f"Unknown service type: {service_type}")
```

**Impacto**:
- ❌ Difícil agregar nuevos servicios
- ❌ Código verboso
- ❌ No escalable

---

### Problema 3: Duplicación en Métodos Públicos

**Problema**: Los métodos públicos tienen el mismo patrón

```python
async def <servicio>(self, ...):
    task_id = await self.task_manager.create_task(
        service_type="<servicio>",
        parameters={...},
        priority=priority,
    )
    return task_id
```

**Impacto**:
- ❌ Código repetitivo
- ❌ Difícil mantener

---

### Problema 4: Configuración Hardcodeada

**Problema**: Temperature y max_tokens hardcodeados en cada método

```python
temperature=0.3,  # o 0.5
max_tokens=4000,
```

**Impacto**:
- ❌ No configurable
- ❌ Difícil ajustar por servicio

---

## ✅ Soluciones Propuestas

### Solución 1: Método Helper Común

**Extraer método común `_execute_service_call`**:
- Construye prompt
- Optimiza con TruthGPT
- Crea mensajes
- Llama API
- Retorna resultado

---

### Solución 2: Diccionario de Handlers

**Usar diccionario en lugar de if/elif**:
- Más escalable
- Más fácil agregar nuevos servicios
- Código más limpio

---

### Solución 3: Helper para Crear Tareas

**Extraer método helper para crear tareas**:
- Reduce duplicación
- Consistencia garantizada

---

### Solución 4: Configuración por Servicio

**Agregar configuración de temperatura por servicio**:
- Más flexible
- Mejor control

---

## 📊 Métricas Esperadas

| Aspecto | Antes | Después | Mejora |
|---------|-------|---------|--------|
| Líneas duplicadas | ~150 | ~30 | ✅ **-80%** |
| Métodos `_handle_*` | 5 métodos largos | 1 método común + config | ✅ **-80%** |
| Routing | if/elif largo | Diccionario | ✅ **+100% escalable** |
| Métodos públicos | 5 métodos repetitivos | 1 helper + 5 métodos simples | ✅ **-60%** |

---

**Estado**: ✅ Análisis Completo - Listo para Refactorización

