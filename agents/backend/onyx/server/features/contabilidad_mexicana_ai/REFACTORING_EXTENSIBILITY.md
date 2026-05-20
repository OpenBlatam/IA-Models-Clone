# Guía de Extensibilidad - ContadorAI

## 📋 Resumen

Esta guía explica cómo extender el código refactorizado para agregar nuevas funcionalidades sin modificar código existente, siguiendo el principio Open/Closed.

---

## 🎯 Principios de Extensibilidad

### 1. Open/Closed Principle

**Regla**: El código debe estar **abierto para extensión** pero **cerrado para modificación**.

**Aplicación en el código refactorizado:**
- ✅ Agregar nuevos servicios sin modificar código existente
- ✅ Agregar nuevos tipos de prompts sin modificar código existente
- ✅ Agregar nuevos system prompts sin modificar código existente

---

## 🔧 Cómo Agregar Nuevo Servicio

### Paso 1: Agregar Método en PromptBuilder

**Archivo**: `core/prompt_builder.py`

```python
@staticmethod
def build_consulta_legal_prompt(
    pregunta: str,
    contexto: Optional[Dict[str, Any]] = None
) -> str:
    """
    ✅ Nuevo método para construir prompts de consulta legal.
    Sin modificar métodos existentes.
    """
    context_str = ""
    if contexto:
        context_str = f"\n\nContexto adicional:\n{PromptBuilder._format_data(contexto)}"
    
    return f"""Proporciona consulta legal sobre la siguiente situación:

{pregunta}{context_str}

Proporciona:
1. Análisis legal de la situación
2. Marco legal aplicable
3. Opciones legales disponibles
4. Recomendaciones específicas
5. Consideraciones importantes"""
```

---

### Paso 2: Agregar System Prompt en SystemPromptsBuilder

**Archivo**: `core/system_prompts_builder.py`

```python
@staticmethod
def _get_legal_consultation_specialization() -> str:
    """✅ Nuevo método para especialización legal."""
    return """
Proporciona consulta legal especializada. Analiza la situación desde el punto de vista legal,
identifica el marco legal aplicable, y proporciona recomendaciones basadas en la legislación vigente."""

@staticmethod
def build_all_prompts() -> Dict[str, str]:
    base_prompt = SystemPromptsBuilder._build_base_prompt()
    
    return {
        # ... prompts existentes ...
        "consulta_legal": base_prompt + SystemPromptsBuilder._get_legal_consultation_specialization(),  # ✅ Solo agregar
    }
```

---

### Paso 3: Agregar Método en ContadorAI

**Archivo**: `core/contador_ai.py`

```python
async def consulta_legal(
    self,
    pregunta: str,
    contexto: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    ✅ Nuevo servicio sin modificar código existente.
    
    Proporciona consulta legal sobre situaciones fiscales.
    
    Args:
        pregunta: Pregunta o situación legal
        contexto: Contexto adicional
    
    Returns:
        Response dictionary with legal consultation
    """
    # ✅ Usa PromptBuilder existente
    prompt = PromptBuilder.build_consulta_legal_prompt(pregunta, contexto)
    
    # ✅ Usa MessageBuilder existente
    messages = MessageBuilder.build_messages(
        system_prompt=self.system_prompts["consulta_legal"],
        user_prompt=prompt
    )
    
    # ✅ Service data completo
    service_data = {
        "pregunta": pregunta,
        "contexto": contexto or {}
    }
    
    # ✅ Usa APIHandler existente
    return await self.api_handler.call_with_metrics(
        messages=messages,
        service_name="consulta_legal",
        service_data=service_data,
        extract_key="consulta"
    )
```

**Beneficios:**
- ✅ No modifica código existente
- ✅ Usa helpers existentes
- ✅ Sigue el mismo patrón
- ✅ Fácil testear

---

## 🔧 Cómo Agregar Nuevo Tipo de Prompt

### Paso 1: Agregar Método en PromptBuilder

**Archivo**: `core/prompt_builder.py`

```python
@staticmethod
def build_analisis_fiscal_prompt(
    datos_fiscales: Dict[str, Any],
    periodo: str
) -> str:
    """
    ✅ Nuevo tipo de prompt sin modificar código existente.
    """
    datos_str = PromptBuilder._format_data(datos_fiscales)
    
    return f"""Analiza la situación fiscal para el período {periodo}.

Datos fiscales:
{datos_str}

Proporciona:
1. Análisis de la situación fiscal
2. Identificación de oportunidades
3. Recomendaciones específicas
4. Plan de acción"""
```

---

### Paso 2: Usar en Nuevo Servicio

**Archivo**: `core/contador_ai.py`

```python
async def analisis_fiscal(self, datos_fiscales, periodo):
    # ✅ Usa nuevo método de PromptBuilder
    prompt = PromptBuilder.build_analisis_fiscal_prompt(datos_fiscales, periodo)
    
    # ... resto del código usando helpers existentes ...
```

---

## 🔧 Cómo Agregar Nuevo System Prompt

### Paso 1: Agregar Especialización

**Archivo**: `core/system_prompts_builder.py`

```python
@staticmethod
def _get_analisis_specialization() -> str:
    """✅ Nueva especialización sin modificar código existente."""
    return """
Especialízate en análisis fiscal profundo. Proporciona análisis detallados,
identifica patrones, y sugiere estrategias de optimización fiscal."""

@staticmethod
def build_all_prompts() -> Dict[str, str]:
    base_prompt = SystemPromptsBuilder._build_base_prompt()
    
    return {
        # ... prompts existentes ...
        "analisis_fiscal": base_prompt + SystemPromptsBuilder._get_analisis_specialization(),  # ✅ Solo agregar
    }
```

---

## 📊 Ejemplo Completo: Agregar Servicio de Análisis Fiscal

### Paso 1: PromptBuilder

```python
# core/prompt_builder.py
@staticmethod
def build_analisis_fiscal_prompt(
    datos_fiscales: Dict[str, Any],
    periodo: str
) -> str:
    """Build prompt for fiscal analysis."""
    datos_str = PromptBuilder._format_data(datos_fiscales)
    
    return f"""Analiza la situación fiscal para el período {periodo}.

Datos fiscales:
{datos_str}

Proporciona:
1. Análisis de la situación fiscal
2. Identificación de oportunidades
3. Recomendaciones específicas
4. Plan de acción"""
```

### Paso 2: SystemPromptsBuilder

```python
# core/system_prompts_builder.py
@staticmethod
def _get_analisis_specialization() -> str:
    """Get specialization for fiscal analysis."""
    return """
Especialízate en análisis fiscal profundo. Proporciona análisis detallados,
identifica patrones, y sugiere estrategias de optimización fiscal."""

@staticmethod
def build_all_prompts() -> Dict[str, str]:
    base_prompt = SystemPromptsBuilder._build_base_prompt()
    
    return {
        # ... existentes ...
        "analisis_fiscal": base_prompt + SystemPromptsBuilder._get_analisis_specialization(),
    }
```

### Paso 3: ContadorAI

```python
# core/contador_ai.py
async def analisis_fiscal(
    self,
    datos_fiscales: Dict[str, Any],
    periodo: str
) -> Dict[str, Any]:
    """
    Realizar análisis fiscal profundo.
    
    Args:
        datos_fiscales: Datos fiscales a analizar
        periodo: Período fiscal
    
    Returns:
        Response dictionary with fiscal analysis
    """
    # ✅ Usa helpers existentes
    prompt = PromptBuilder.build_analisis_fiscal_prompt(datos_fiscales, periodo)
    
    messages = MessageBuilder.build_messages(
        system_prompt=self.system_prompts["analisis_fiscal"],
        user_prompt=prompt
    )
    
    service_data = {
        "datos_fiscales": datos_fiscales,
        "periodo": periodo
    }
    
    return await self.api_handler.call_with_metrics(
        messages=messages,
        service_name="analisis_fiscal",
        service_data=service_data,
        extract_key="analisis"
    )
```

**✅ Listo! Nuevo servicio agregado sin modificar código existente**

---

## ✅ Checklist de Extensión

### Para Agregar Nuevo Servicio:
- [ ] Agregar método en `PromptBuilder` para construir prompt
- [ ] Agregar especialización en `SystemPromptsBuilder`
- [ ] Agregar método en `ContadorAI` usando helpers existentes
- [ ] Agregar tests
- [ ] Actualizar documentación

### Para Agregar Nuevo Tipo de Prompt:
- [ ] Agregar método en `PromptBuilder`
- [ ] Usar en servicio existente o nuevo
- [ ] Agregar tests
- [ ] Actualizar documentación

### Para Agregar Nuevo System Prompt:
- [ ] Agregar especialización en `SystemPromptsBuilder`
- [ ] Agregar al diccionario en `build_all_prompts()`
- [ ] Usar en servicio
- [ ] Agregar tests

---

## 🎯 Mejores Prácticas para Extensión

### 1. Seguir el Patrón Existente

**✅ Bueno:**
```python
# Sigue el mismo patrón que otros métodos
async def nuevo_servicio(self, ...):
    prompt = PromptBuilder.build_nuevo_prompt(...)  # ✅ Mismo patrón
    messages = MessageBuilder.build_messages(...)    # ✅ Mismo patrón
    return await self.api_handler.call_with_metrics(...)  # ✅ Mismo patrón
```

**❌ Malo:**
```python
# No sigue el patrón
async def nuevo_servicio(self, ...):
    # ❌ Construcción manual
    messages = [{"role": "system", ...}, {"role": "user", ...}]
    # ❌ Llamada directa
    response = await self.client.generate_completion(...)
```

---

### 2. Usar Helpers Existentes

**✅ Bueno:**
```python
# Usa helpers existentes
prompt = PromptBuilder.build_nuevo_prompt(...)
messages = MessageBuilder.build_messages(...)
```

**❌ Malo:**
```python
# No usa helpers
prompt = f"""Nuevo prompt..."""  # ❌ Construcción manual
messages = [{"role": "system", ...}]  # ❌ Construcción manual
```

---

### 3. Service Data Completo

**✅ Bueno:**
```python
service_data = {
    "parametro1": valor1,
    "parametro2": valor2,
    "contexto": contexto or {}  # ✅ Incluye todo
}
```

**❌ Malo:**
```python
service_data = {"parametro1": valor1}  # ❌ Faltan datos
```

---

## 🚀 Ejemplo Completo: Agregar Servicio de Consulta Legal

### Paso 1: PromptBuilder

```python
# core/prompt_builder.py
@staticmethod
def build_consulta_legal_prompt(pregunta, contexto=None):
    """Build prompt for legal consultation."""
    # ... implementación ...
```

### Paso 2: SystemPromptsBuilder

```python
# core/system_prompts_builder.py
@staticmethod
def _get_legal_consultation_specialization():
    """Get specialization for legal consultation."""
    return """..."""

@staticmethod
def build_all_prompts():
    return {
        # ... existentes ...
        "consulta_legal": base_prompt + SystemPromptsBuilder._get_legal_consultation_specialization(),
    }
```

### Paso 3: ContadorAI

```python
# core/contador_ai.py
async def consulta_legal(self, pregunta, contexto=None):
    prompt = PromptBuilder.build_consulta_legal_prompt(pregunta, contexto)
    messages = MessageBuilder.build_messages(
        system_prompt=self.system_prompts["consulta_legal"],
        user_prompt=prompt
    )
    service_data = {"pregunta": pregunta, "contexto": contexto or {}}
    return await self.api_handler.call_with_metrics(
        messages=messages,
        service_name="consulta_legal",
        service_data=service_data,
        extract_key="consulta"
    )
```

---

## ✅ Resumen

### Ventajas de la Arquitectura Refactorizada para Extensión

1. ✅ **Fácil Agregar Nuevos Servicios**: Solo agregar método siguiendo patrón
2. ✅ **Fácil Agregar Nuevos Prompts**: Solo agregar método en PromptBuilder
3. ✅ **Fácil Agregar Nuevos System Prompts**: Solo agregar especialización
4. ✅ **Sin Modificar Código Existente**: Principio Open/Closed
5. ✅ **Reutiliza Helpers**: Usa componentes existentes
6. ✅ **Testeable**: Fácil agregar tests para nuevas funcionalidades

---

**🎊🎊🎊 Guía de Extensibilidad Completa. Código Listo para Crecimiento. 🎊🎊🎊**

