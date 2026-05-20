# Ejemplos de Código - Refactorización ContadorAI

## 📋 Resumen

Este documento proporciona ejemplos detallados de código antes/después de la refactorización, con explicaciones de cada cambio.

---

## 🔄 Comparaciones Antes/Después

### Ejemplo 1: `calcular_impuestos()`

#### ❌ ANTES: Código con Duplicación y Responsabilidades Mezcladas

```python
async def calcular_impuestos(
    self,
    regimen: str,
    tipo_impuesto: str,
    datos: Dict[str, Any],
    use_cache: bool = True
) -> Dict[str, Any]:
    """
    Calcular impuestos para un régimen fiscal específico.
    """
    # ❌ Construcción manual de prompt
    datos_str = ""
    for key, value in datos.items():
        datos_str += f"- {key}: {value}\n"
    
    prompt = f"""Calcula el {tipo_impuesto} para un contribuyente en régimen {regimen}.

Datos proporcionados:
{datos_str}

Proporciona:
1. Base de cálculo detallada
2. Tasa aplicable según el régimen
3. Cálculo paso a paso con fórmula
4. Resultado final
5. Fechas de pago y obligaciones
6. Recomendaciones si aplica"""
    
    # ❌ Construcción manual de mensajes
    messages = [
        {"role": "system", "content": self.system_prompts["calculo_impuestos"]},
        {"role": "user", "content": prompt}
    ]
    
    # ❌ Timing manual
    start_time = time.time()
    
    try:
        # ❌ Llamada directa al cliente
        response = await self.client.generate_completion(
            messages=messages,
            model=self.config.openrouter.default_model,
            temperature=self.config.openrouter.temperature,
            max_tokens=self.config.openrouter.max_tokens
        )
        
        # ❌ Extracción manual de contenido
        content = response["choices"][0]["message"]["content"]
        response_time = time.time() - start_time
        
        # ❌ Construcción manual de respuesta
        return {
            "success": True,
            "regimen": regimen,
            "tipo_impuesto": tipo_impuesto,
            "datos_entrada": datos,
            "resultado": content,
            "tiempo_respuesta": response_time,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        # ❌ Manejo manual de errores
        logger.error(f"Error calculating taxes: {e}")
        return {
            "success": False,
            "error": str(e),
            "regimen": regimen,
            "tipo_impuesto": tipo_impuesto
        }
```

**Problemas:**
- ❌ ~60 líneas de código
- ❌ Múltiples responsabilidades
- ❌ Código duplicado (timing, error handling, extracción)
- ❌ Difícil testear
- ❌ Difícil mantener

---

#### ✅ DESPUÉS: Código Refactorizado con Helpers

```python
async def calcular_impuestos(
    self,
    regimen: str,
    tipo_impuesto: str,
    datos: Dict[str, Any],
    use_cache: bool = True
) -> Dict[str, Any]:
    """
    Calcular impuestos para un régimen fiscal específico.
    
    Args:
        regimen: Régimen fiscal (RESICO, PFAE, Sueldos y Salarios, etc.)
        tipo_impuesto: Tipo de impuesto (ISR, IVA, IEPS)
        datos: Datos necesarios para el cálculo (ingresos, gastos, etc.)
        use_cache: Whether to use cache
    
    Returns:
        Response dictionary with calculation and breakdown
    """
    # ✅ PromptBuilder: Solo construye prompts
    prompt = PromptBuilder.build_calculation_prompt(regimen, tipo_impuesto, datos)
    
    # ✅ MessageBuilder: Solo construye mensajes
    messages = MessageBuilder.build_messages(
        system_prompt=self.system_prompts["calculo_impuestos"],
        user_prompt=prompt
    )
    
    # ✅ Service data completo
    service_data = {
        "regimen": regimen,
        "tipo_impuesto": tipo_impuesto,
        "datos_entrada": datos,
    }
    
    # ✅ APIHandler: Maneja timing, error handling, extracción automáticamente
    result = await self.api_handler.call_with_metrics(
        messages=messages,
        service_name="calcular_impuestos",
        service_data=service_data,
        extract_key="resultado"
    )
    
    # ✅ Renombrar para consistencia
    if result.get("tiempo_respuesta"):
        result["tiempo_calculo"] = result.pop("tiempo_respuesta")
    
    return result
```

**Beneficios:**
- ✅ ~30 líneas de código (-50%)
- ✅ Responsabilidades separadas
- ✅ Sin duplicación
- ✅ Fácil testear
- ✅ Fácil mantener

---

### Ejemplo 2: `tramite_sat()`

#### ❌ ANTES: Código con Duplicación

```python
async def tramite_sat(
    self,
    tipo_tramite: str,
    detalles: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Obtener información sobre un trámite del SAT.
    """
    # ❌ Timing manual
    start_time = time.time()
    
    # ❌ Formateo manual de datos
    detalles_str = ""
    if detalles:
        for key, value in detalles.items():
            detalles_str += f"- {key}: {value}\n"
        detalles_str = f"\n\nDetalles específicos:\n{detalles_str}"
    
    # ❌ Prompt inline
    prompt = f"""Proporciona información completa sobre el trámite: {tipo_tramite}{detalles_str}

Incluye:
1. Descripción del trámite
2. Requisitos necesarios
3. Documentación requerida
4. Pasos detallados del proceso
5. Tiempos de respuesta estimados
6. Costos si aplica
7. Enlaces oficiales relevantes
8. Consejos y recomendaciones"""
    
    # ❌ Construcción manual de mensajes
    messages = [
        {"role": "system", "content": self.system_prompts["tramites_sat"]},
        {"role": "user", "content": prompt}
    ]
    
    try:
        # ❌ Llamada directa
        response = await self.client.generate_completion(
            messages=messages,
            model=self.config.openrouter.default_model,
            temperature=self.config.openrouter.temperature,
            max_tokens=self.config.openrouter.max_tokens
        )
        
        # ❌ Extracción manual
        content = response["choices"][0]["message"]["content"]
        response_time = time.time() - start_time
        
        # ❌ Construcción manual de respuesta
        return {
            "success": True,
            "tipo_tramite": tipo_tramite,
            "informacion": content,
            "tiempo_respuesta": response_time,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        # ❌ Manejo manual de errores
        logger.error(f"Error getting SAT procedure info: {e}")
        return {
            "success": False,
            "error": str(e),
            "tipo_tramite": tipo_tramite
        }
```

**Problemas:**
- ❌ ~50 líneas de código
- ❌ Duplicación con otros métodos
- ❌ Service data incompleto (falta detalles)

---

#### ✅ DESPUÉS: Código Refactorizado

```python
async def tramite_sat(
    self,
    tipo_tramite: str,
    detalles: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Obtener información sobre un trámite del SAT.
    
    Args:
        tipo_tramite: Tipo de trámite (ej: "Alta en RFC", "Renovación e.firma")
        detalles: Detalles adicionales del trámite
    
    Returns:
        Response dictionary with procedure information
    """
    # ✅ PromptBuilder: Construye prompt
    prompt = PromptBuilder.build_procedure_prompt(tipo_tramite, detalles)
    
    # ✅ MessageBuilder: Construye mensajes
    messages = MessageBuilder.build_messages(
        system_prompt=self.system_prompts["tramites_sat"],
        user_prompt=prompt
    )
    
    # ✅ Service data completo (incluye detalles)
    service_data = {
        "tipo_tramite": tipo_tramite,
        "detalles": detalles or {}
    }
    
    # ✅ APIHandler: Maneja todo automáticamente
    return await self.api_handler.call_with_metrics(
        messages=messages,
        service_name="tramite_sat",
        service_data=service_data,
        extract_key="informacion"
    )
```

**Beneficios:**
- ✅ ~25 líneas de código (-50%)
- ✅ Sin duplicación
- ✅ Service data completo
- ✅ Consistente con otros métodos

---

### Ejemplo 3: `_build_system_prompts()`

#### ❌ ANTES: Prompts Inline

```python
def _build_system_prompts(self) -> Dict[str, str]:
    """Build system prompts for different services."""
    # ❌ Base prompt inline
    base_prompt = """Eres un contador público certificado experto en legislación fiscal mexicana.
Tienes conocimiento profundo del SAT, regímenes fiscales (RESICO, PFAE, Sueldos y Salarios, etc.),
cálculo de impuestos (ISR, IVA, IEPS), trámites fiscales, declaraciones y cumplimiento fiscal.

Siempre proporcionas información precisa, actualizada y basada en la legislación mexicana vigente.
Cuando realices cálculos, muestra las fórmulas y el proceso paso a paso.
Incluye referencias a artículos de la LISR, LIVA, CFF cuando sea relevante.

Responde en español mexicano, de forma profesional pero accesible."""
    
    # ❌ Especializaciones inline
    return {
        "default": base_prompt,
        "calculo_impuestos": base_prompt + """
Especialízate en cálculos precisos de impuestos. Siempre muestra:
1. Base de cálculo
2. Tasa aplicable
3. Fórmula utilizada
4. Resultado con desglose
5. Fechas de pago y obligaciones relacionadas""",
        
        "asesoria_fiscal": base_prompt + """
Proporciona asesoría fiscal personalizada. Analiza la situación del contribuyente,
recomienda el régimen fiscal más adecuado, identifica deducciones aplicables,
y sugiere estrategias de optimización fiscal dentro del marco legal.""",
        
        # ... más prompts inline
    }
```

**Problemas:**
- ❌ ~40 líneas de código
- ❌ Duplicación con SystemPromptsBuilder
- ❌ Difícil mantener

---

#### ✅ DESPUÉS: Reutiliza SystemPromptsBuilder

```python
def _build_system_prompts(self) -> Dict[str, str]:
    """
    Build system prompts for different services.
    
    Uses SystemPromptsBuilder to centralize prompt construction logic.
    """
    # ✅ Reutiliza SystemPromptsBuilder existente
    return SystemPromptsBuilder.build_all_prompts()
```

**Beneficios:**
- ✅ ~3 líneas de código (-92%)
- ✅ Reutiliza código existente
- ✅ Single source of truth
- ✅ Fácil mantener

---

## 📊 Comparación de Métricas

### Reducción de Código

| Método | Antes | Después | Reducción |
|--------|-------|---------|-----------|
| `calcular_impuestos` | ~60 líneas | ~30 líneas | ✅ **-50%** |
| `tramite_sat` | ~50 líneas | ~25 líneas | ✅ **-50%** |
| `ayuda_declaracion` | ~50 líneas | ~25 líneas | ✅ **-50%** |
| `_build_system_prompts` | ~40 líneas | ~3 líneas | ✅ **-92%** |

### Eliminación de Duplicación

| Patrón | Antes | Después | Mejora |
|--------|-------|---------|--------|
| Construcción de mensajes | 5 veces | 0 (MessageBuilder) | ✅ **-100%** |
| Timing manual | 2 veces | 0 (APIHandler) | ✅ **-100%** |
| Error handling manual | 2 veces | 0 (APIHandler) | ✅ **-100%** |
| Extracción de contenido | 2 veces | 0 (APIHandler) | ✅ **-100%** |
| System prompts | 1 vez inline | 0 (SystemPromptsBuilder) | ✅ **-100%** |

---

## 🎯 Explicación de Cambios

### Cambio 1: Uso de PromptBuilder

**Razón**: Eliminar duplicación en construcción de prompts

**Antes**: Prompts construidos inline en cada método
**Después**: Prompts construidos por PromptBuilder

**Beneficio**: Single source of truth para prompts

---

### Cambio 2: Uso de MessageBuilder

**Razón**: Eliminar duplicación en construcción de mensajes

**Antes**: Mensajes construidos manualmente en cada método
**Después**: Mensajes construidos por MessageBuilder

**Beneficio**: Consistencia y reutilización

---

### Cambio 3: Uso de APIHandler

**Razón**: Centralizar timing, error handling y extracción

**Antes**: Timing, error handling y extracción manuales
**Después**: Todo manejado por APIHandler

**Beneficio**: Sin duplicación, manejo consistente

---

### Cambio 4: Uso de SystemPromptsBuilder

**Razón**: Reutilizar código existente

**Antes**: System prompts construidos inline
**Después**: System prompts de SystemPromptsBuilder

**Beneficio**: Single source of truth, fácil mantener

---

### Cambio 5: Service Data Completo

**Razón**: Información completa en respuestas

**Antes**: Service data incompleto (faltaban detalles/datos)
**Después**: Service data completo

**Beneficio**: Mejor trazabilidad, información completa

---

## ✅ Resumen

### Mejoras Cuantitativas

- ✅ **Reducción de código**: ~50% en métodos principales
- ✅ **Eliminación de duplicación**: 100% en patrones comunes
- ✅ **Consistencia**: 100% en todos los métodos

### Mejoras Cualitativas

- ✅ **Mantenibilidad**: Código más fácil de mantener
- ✅ **Testabilidad**: Código más fácil de testear
- ✅ **Extensibilidad**: Fácil agregar nuevos servicios
- ✅ **Legibilidad**: Código más claro y predecible

---

**🎊🎊🎊 Refactorización Completa. Código Optimizado y Consistente. 🎊🎊🎊**

