# Guía de Extensibilidad - ContadorSAM3Agent

## 📋 Resumen

Esta guía explica cómo extender el código refactorizado para agregar nuevas funcionalidades sin modificar código existente, siguiendo el principio Open/Closed.

---

## 🎯 Principios de Extensibilidad

### 1. Open/Closed Principle

**Regla**: El código debe estar **abierto para extensión** pero **cerrado para modificación**.

**Aplicación en el código refactorizado:**
- ✅ Agregar nuevos servicios sin modificar código existente
- ✅ Agregar nuevos tipos de prompts sin modificar código existente
- ✅ Agregar nuevos endpoints sin modificar código existente

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

### Paso 3: Agregar Handler en ContadorSAM3Agent

**Archivo**: `core/contador_sam3_agent.py`

```python
async def _handle_consulta_legal(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
    """
    ✅ Nuevo handler sin modificar código existente.
    
    Handle legal consultation request.
    """
    pregunta = parameters.get("pregunta")
    contexto = parameters.get("contexto")
    
    # ✅ Usa PromptBuilder existente
    prompt = PromptBuilder.build_consulta_legal_prompt(pregunta, contexto)
    
    # ✅ Usa método común existente
    return await self._execute_service_call(
        prompt=prompt,
        system_prompt_key="consulta_legal",
        temperature=0.5,
        response_key="consulta"
    )
```

---

### Paso 4: Agregar al Diccionario de Handlers

**Archivo**: `core/contador_sam3_agent.py`

```python
def _get_service_handler(self, service_type: str):
    handlers = {
        # ... existentes ...
        "consulta_legal": self._handle_consulta_legal,  # ✅ Solo agregar
    }
    return handlers.get(service_type)
```

---

### Paso 5: Agregar Método Público

**Archivo**: `core/contador_sam3_agent.py`

```python
async def consulta_legal(
    self,
    pregunta: str,
    contexto: Optional[Dict[str, Any]] = None,
    priority: int = 0,
) -> str:
    """
    ✅ Nuevo servicio sin modificar código existente.
    
    Submit legal consultation task.
    
    Args:
        pregunta: Legal question
        contexto: Additional context
        priority: Task priority
        
    Returns:
        Task ID
    """
    # ✅ Usa helper existente
    return await self._create_service_task(
        service_type="consulta_legal",
        parameters={
            "pregunta": pregunta,
            "contexto": contexto,
        },
        priority=priority,
    )
```

**Beneficios:**
- ✅ No modifica código existente
- ✅ Usa helpers existentes
- ✅ Sigue el mismo patrón
- ✅ Fácil testear

---

## 🔧 Cómo Agregar Nuevo Endpoint API

### Paso 1: Agregar Request Model

**Archivo**: `api/contador_sam3_api.py`

```python
class ConsultaLegalRequest(BaseModel):
    """✅ Nuevo request model."""
    pregunta: str
    contexto: Optional[Dict[str, Any]] = None
    priority: int = 0
```

---

### Paso 2: Agregar Endpoint

**Archivo**: `api/contador_sam3_api.py`

```python
@app.post("/consulta-legal")
async def consulta_legal(request: ConsultaLegalRequest):
    """
    ✅ Nuevo endpoint sin modificar código existente.
    
    Get legal consultation.
    """
    # ✅ Usa helpers existentes
    agent = require_agent(_agent)
    
    task_id = await agent.consulta_legal(
        pregunta=request.pregunta,
        contexto=request.contexto,
        priority=request.priority
    )
    
    # ✅ Usa ResponseBuilder existente
    return ResponseBuilder.task_submitted(task_id)
```

**Beneficios:**
- ✅ No modifica código existente
- ✅ Usa helpers existentes
- ✅ Consistente con otros endpoints

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

**Archivo**: `core/contador_sam3_agent.py`

```python
async def _handle_analisis_fiscal(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
    # ✅ Usa nuevo método de PromptBuilder
    prompt = PromptBuilder.build_analisis_fiscal_prompt(
        parameters.get("datos_fiscales"),
        parameters.get("periodo")
    )
    
    # ... resto del código usando helpers existentes ...
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

### Paso 3: ContadorSAM3Agent - Handler

```python
# core/contador_sam3_agent.py
async def _handle_analisis_fiscal(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
    """Handle fiscal analysis request."""
    datos_fiscales = parameters.get("datos_fiscales")
    periodo = parameters.get("periodo")
    
    # ✅ Usa helpers existentes
    prompt = PromptBuilder.build_analisis_fiscal_prompt(datos_fiscales, periodo)
    
    return await self._execute_service_call(
        prompt=prompt,
        system_prompt_key="analisis_fiscal",
        temperature=0.5,
        response_key="analisis"
    )
```

### Paso 4: ContadorSAM3Agent - Diccionario

```python
# core/contador_sam3_agent.py
def _get_service_handler(self, service_type: str):
    handlers = {
        # ... existentes ...
        "analisis_fiscal": self._handle_analisis_fiscal,  # ✅ Solo agregar
    }
    return handlers.get(service_type)
```

### Paso 5: ContadorSAM3Agent - Método Público

```python
# core/contador_sam3_agent.py
async def analisis_fiscal(
    self,
    datos_fiscales: Dict[str, Any],
    periodo: str,
    priority: int = 0,
) -> str:
    """
    Submit fiscal analysis task.
    
    Args:
        datos_fiscales: Fiscal data to analyze
        periodo: Fiscal period
        priority: Task priority
        
    Returns:
        Task ID
    """
    # ✅ Usa helper existente
    return await self._create_service_task(
        service_type="analisis_fiscal",
        parameters={
            "datos_fiscales": datos_fiscales,
            "periodo": periodo,
        },
        priority=priority,
    )
```

### Paso 6: API - Request Model

```python
# api/contador_sam3_api.py
class AnalisisFiscalRequest(BaseModel):
    """Request model for fiscal analysis."""
    datos_fiscales: Dict[str, Any]
    periodo: str
    priority: int = 0
```

### Paso 7: API - Endpoint

```python
# api/contador_sam3_api.py
@app.post("/analisis-fiscal")
async def analisis_fiscal(request: AnalisisFiscalRequest):
    """Get fiscal analysis."""
    # ✅ Usa helpers existentes
    agent = require_agent(_agent)
    
    task_id = await agent.analisis_fiscal(
        datos_fiscales=request.datos_fiscales,
        periodo=request.periodo,
        priority=request.priority
    )
    
    # ✅ Usa ResponseBuilder existente
    return ResponseBuilder.task_submitted(task_id)
```

**✅ Listo! Nuevo servicio agregado sin modificar código existente**

---

## ✅ Checklist de Extensión

### Para Agregar Nuevo Servicio:
- [ ] Agregar método en `PromptBuilder` para construir prompt
- [ ] Agregar especialización en `SystemPromptsBuilder`
- [ ] Agregar handler en `ContadorSAM3Agent` usando `_execute_service_call`
- [ ] Agregar entrada al diccionario en `_get_service_handler`
- [ ] Agregar método público usando `_create_service_task`
- [ ] Agregar request model en API
- [ ] Agregar endpoint usando helpers existentes
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
    return await self._create_service_task(...)  # ✅ Mismo patrón

async def _handle_nuevo_servicio(self, parameters):
    prompt = PromptBuilder.build_nuevo_prompt(...)  # ✅ Mismo patrón
    return await self._execute_service_call(...)  # ✅ Mismo patrón
```

**❌ Malo:**
```python
# No sigue el patrón
async def nuevo_servicio(self, ...):
    # ❌ Construcción manual
    messages = [{"role": "system", ...}, {"role": "user", ...}]
    # ❌ Llamada directa
    response = await self.openrouter_client.chat_completion(...)
```

---

### 2. Usar Helpers Existentes

**✅ Bueno:**
```python
# Usa helpers existentes
agent = require_agent(_agent)  # ✅ Helper
return ResponseBuilder.task_submitted(task_id)  # ✅ Helper
```

**❌ Malo:**
```python
# No usa helpers
if not _agent:  # ❌ Validación manual
    raise HTTPException(...)
return {"task_id": task_id, "status": "submitted"}  # ❌ Respuesta manual
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

### Paso 3: ContadorSAM3Agent

```python
# core/contador_sam3_agent.py
async def consulta_legal(self, pregunta, contexto=None, priority=0):
    return await self._create_service_task(
        service_type="consulta_legal",
        parameters={"pregunta": pregunta, "contexto": contexto},
        priority=priority,
    )

async def _handle_consulta_legal(self, parameters):
    prompt = PromptBuilder.build_consulta_legal_prompt(
        parameters.get("pregunta"),
        parameters.get("contexto")
    )
    return await self._execute_service_call(
        prompt=prompt,
        system_prompt_key="consulta_legal",
        temperature=0.5,
        response_key="consulta"
    )

def _get_service_handler(self, service_type):
    handlers = {
        # ... existentes ...
        "consulta_legal": self._handle_consulta_legal,
    }
    return handlers.get(service_type)
```

### Paso 4: API

```python
# api/contador_sam3_api.py
class ConsultaLegalRequest(BaseModel):
    pregunta: str
    contexto: Optional[Dict[str, Any]] = None
    priority: int = 0

@app.post("/consulta-legal")
async def consulta_legal(request: ConsultaLegalRequest):
    agent = require_agent(_agent)
    task_id = await agent.consulta_legal(
        pregunta=request.pregunta,
        contexto=request.contexto,
        priority=request.priority
    )
    return ResponseBuilder.task_submitted(task_id)
```

---

## ✅ Resumen

### Ventajas de la Arquitectura Refactorizada para Extensión

1. ✅ **Fácil Agregar Nuevos Servicios**: Solo agregar método siguiendo patrón
2. ✅ **Fácil Agregar Nuevos Prompts**: Solo agregar método en PromptBuilder
3. ✅ **Fácil Agregar Nuevos System Prompts**: Solo agregar especialización
4. ✅ **Fácil Agregar Nuevos Endpoints**: Solo agregar endpoint usando helpers
5. ✅ **Sin Modificar Código Existente**: Principio Open/Closed
6. ✅ **Reutiliza Helpers**: Usa componentes existentes
7. ✅ **Testeable**: Fácil agregar tests para nuevas funcionalidades

---

**🎊🎊🎊 Guía de Extensibilidad Completa. Código Listo para Crecimiento. 🎊🎊🎊**

