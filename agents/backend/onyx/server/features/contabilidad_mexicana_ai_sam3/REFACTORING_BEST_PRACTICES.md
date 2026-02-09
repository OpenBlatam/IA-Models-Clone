# Guía de Mejores Prácticas Aplicadas - ContadorSAM3Agent

## 📋 Resumen

Esta guía detalla todas las mejores prácticas aplicadas durante la refactorización del módulo `contabilidad_mexicana_ai_sam3`, con ejemplos concretos y explicaciones.

---

## 🎯 Principios SOLID Aplicados

### 1. Single Responsibility Principle (SRP)

#### ✅ Aplicación: Separación de Responsabilidades

**Ejemplo: Ejecución de Servicios**

**❌ ANTES: Responsabilidad Mezclada**

```python
async def _handle_calcular_impuestos(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
    # ❌ Extrae parámetros
    regimen = parameters.get("regimen")
    tipo_impuesto = parameters.get("tipo_impuesto")
    datos = parameters.get("datos", {})
    
    # ❌ Construye prompt
    prompt = PromptBuilder.build_calculation_prompt(regimen, tipo_impuesto, datos)
    
    # ❌ Optimiza con TruthGPT
    optimized_prompt = await self.truthgpt_client.optimize_query(prompt)
    
    # ❌ Crea mensajes
    messages = [
        create_message("system", self.system_prompts["calculo_impuestos"]),
        create_message("user", optimized_prompt)
    ]
    
    # ❌ Llama API
    response = await self.openrouter_client.chat_completion(
        model=self.config.openrouter.model,
        messages=messages,
        temperature=0.3,
        max_tokens=4000,
    )
    
    # ❌ Formatea respuesta
    return {
        "resultado": response["response"],
        "tokens_used": response["tokens_used"],
        "model": response["model"],
        "tiempo_calculo": datetime.now().isoformat(),
    }
```

**Problemas:**
- ❌ Múltiples responsabilidades en un método
- ❌ Difícil testear cada parte
- ❌ Código duplicado en otros métodos

**✅ DESPUÉS: Responsabilidades Separadas**

```python
async def _execute_service_call(
    self,
    prompt: str,
    system_prompt_key: str,
    temperature: float = 0.3,
    max_tokens: int = 4000,
    response_key: str = "resultado",
    additional_fields: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    ✅ Single Responsibility: Execute service calls with common pattern.
    """
    # ✅ Solo optimiza prompt
    optimized_prompt = await self.truthgpt_client.optimize_query(prompt)
    
    # ✅ Solo crea mensajes
    messages = [
        create_message("system", self.system_prompts[system_prompt_key]),
        create_message("user", optimized_prompt)
    ]
    
    # ✅ Solo llama API
    response = await self.openrouter_client.chat_completion(
        model=self.config.openrouter.model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    
    # ✅ Solo formatea respuesta
    result = {
        response_key: response["response"],
        "tokens_used": response["tokens_used"],
        "model": response["model"],
    }
    
    if additional_fields:
        result.update(additional_fields)
    
    return result

async def _handle_calcular_impuestos(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
    """✅ Solo extrae parámetros y delega"""
    regimen = parameters.get("regimen")
    tipo_impuesto = parameters.get("tipo_impuesto")
    datos = parameters.get("datos", {})
    
    prompt = PromptBuilder.build_calculation_prompt(regimen, tipo_impuesto, datos)
    
    # ✅ Delega a método común
    return await self._execute_service_call(
        prompt=prompt,
        system_prompt_key="calculo_impuestos",
        temperature=0.3,
        response_key="resultado",
        additional_fields={"tiempo_calculo": datetime.now().isoformat()}
    )
```

**Beneficios:**
- ✅ Cada componente tiene una responsabilidad única
- ✅ Fácil testear cada componente independientemente
- ✅ Fácil modificar un componente sin afectar otros
- ✅ Código reutilizable

---

### 2. DRY (Don't Repeat Yourself)

#### ✅ Aplicación: Eliminación de Duplicación

**Ejemplo: Routing de Servicios**

**❌ ANTES: Duplicación en 5 Métodos**

```python
# ❌ Duplicado en _handle_calcular_impuestos
optimized_prompt = await self.truthgpt_client.optimize_query(prompt)
messages = [
    create_message("system", self.system_prompts["calculo_impuestos"]),
    create_message("user", optimized_prompt)
]
response = await self.openrouter_client.chat_completion(...)
return {...}

# ❌ Duplicado en _handle_asesoria_fiscal
optimized_prompt = await self.truthgpt_client.optimize_query(prompt)
messages = [
    create_message("system", self.system_prompts["asesoria_fiscal"]),
    create_message("user", optimized_prompt)
]
response = await self.openrouter_client.chat_completion(...)
return {...}

# ❌ Duplicado en _handle_guia_fiscal
# ... mismo código ...

# ❌ Duplicado en _handle_tramite_sat
# ... mismo código ...

# ❌ Duplicado en _handle_ayuda_declaracion
# ... mismo código ...
```

**Problemas:**
- ❌ Código duplicado 5 veces
- ❌ Cambios requieren modificar 5 lugares
- ❌ Fácil olvidar uno

**✅ DESPUÉS: Single Source of Truth**

```python
# ✅ Método común: Single source of truth
async def _execute_service_call(self, prompt, system_prompt_key, ...):
    optimized_prompt = await self.truthgpt_client.optimize_query(prompt)
    messages = [
        create_message("system", self.system_prompts[system_prompt_key]),
        create_message("user", optimized_prompt)
    ]
    response = await self.openrouter_client.chat_completion(...)
    return {...}

# ✅ Usado en todos los métodos (sin duplicación)
async def _handle_calcular_impuestos(self, parameters):
    prompt = PromptBuilder.build_calculation_prompt(...)
    return await self._execute_service_call(...)

async def _handle_asesoria_fiscal(self, parameters):
    prompt = PromptBuilder.build_advice_prompt(...)
    return await self._execute_service_call(...)
```

**Beneficios:**
- ✅ Single source of truth
- ✅ Cambios en un solo lugar
- ✅ Consistencia garantizada
- ✅ Fácil extender

---

### 3. Open/Closed Principle

#### ✅ Aplicación: Extensible Sin Modificar

**Ejemplo: Agregar Nuevo Tipo de Servicio**

**✅ Fácil Agregar Sin Modificar Código Existente**

```python
# ✅ Agregar nuevo método sin modificar código existente
async def consulta_legal(self, pregunta: str) -> str:
    # ✅ Usa los mismos helpers existentes
    return await self._create_service_task(
        service_type="consulta_legal",
        parameters={"pregunta": pregunta},
        priority=priority,
    )

# ✅ Agregar handler sin modificar código existente
async def _handle_consulta_legal(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
    pregunta = parameters.get("pregunta")
    prompt = PromptBuilder.build_legal_prompt(pregunta)  # ✅ Nuevo método en PromptBuilder
    
    return await self._execute_service_call(
        prompt=prompt,
        system_prompt_key="consulta_legal",  # ✅ Nuevo prompt
        temperature=0.5,
        response_key="consulta"
    )

# ✅ Solo agregar entrada al diccionario
def _get_service_handler(self, service_type: str):
    handlers = {
        # ... existentes ...
        "consulta_legal": self._handle_consulta_legal,  # ✅ Solo agregar
    }
    return handlers.get(service_type)
```

**Beneficios:**
- ✅ No modifica código existente
- ✅ Usa helpers existentes
- ✅ Fácil agregar nuevos servicios

---

## 🎯 Mejores Prácticas de Código

### 1. Consistencia en Patrones

#### ✅ Aplicación: Mismo Patrón en Todos los Métodos

**Patrón Unificado para Handlers**:

```python
async def _handle_<servicio>(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
    # 1. Extraer parámetros
    param1 = parameters.get("param1")
    param2 = parameters.get("param2")
    
    # 2. Construir prompt (PromptBuilder)
    prompt = PromptBuilder.build_<tipo>_prompt(param1, param2)
    
    # 3. Ejecutar servicio común (_execute_service_call)
    return await self._execute_service_call(
        prompt=prompt,
        system_prompt_key="<tipo>",
        temperature=0.3,  # o 0.5
        response_key="<clave>",
        additional_fields={...}  # opcional
    )
```

**Patrón Unificado para Métodos Públicos**:

```python
async def <servicio>(self, ...):
    # ✅ Usa helper común
    return await self._create_service_task(
        service_type="<servicio>",
        parameters={...},
        priority=priority,
    )
```

**Beneficios:**
- ✅ Código predecible
- ✅ Fácil entender
- ✅ Fácil mantener

---

### 2. Reutilización de Código

#### ✅ Aplicación: Helpers Reutilizables

**Ejemplo: API Helpers**

**❌ ANTES: Duplicación**

```python
@app.get("/task/{task_id}/status")
async def get_task_status(task_id: str):
    if not _agent:  # ❌ Duplicado
        raise HTTPException(status_code=503, detail="Agent not initialized")
    try:
        status = await _agent.get_task_status(task_id)
        return status
    except ValueError as e:  # ❌ Duplicado
        raise HTTPException(status_code=404, detail=str(e))

@app.get("/task/{task_id}/result")
async def get_task_result(task_id: str):
    if not _agent:  # ❌ Duplicado
        raise HTTPException(status_code=503, detail="Agent not initialized")
    try:
        result = await _agent.get_task_result(task_id)
        return result
    except ValueError as e:  # ❌ Duplicado
        raise HTTPException(status_code=404, detail=str(e))
```

**✅ DESPUÉS: Reutilización**

```python
# ✅ Helper reutilizable
def require_agent(agent: Optional[ContadorSAM3Agent]) -> ContadorSAM3Agent:
    if not agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    return agent

# ✅ Decorador reutilizable
@handle_task_errors
async def get_task_status(task_id: str):
    agent = require_agent(_agent)  # ✅ Usa helper
    status = await agent.get_task_status(task_id)
    return status

@handle_task_errors
async def get_task_result(task_id: str):
    agent = require_agent(_agent)  # ✅ Usa helper
    result = await agent.get_task_result(task_id)
    return result
```

**Beneficios:**
- ✅ Reutiliza código existente
- ✅ Single source of truth
- ✅ Fácil mantener

---

### 3. Response Builder Consistente

#### ✅ Aplicación: Formato de Respuestas Estandarizado

**Antes**:
```python
return {"task_id": task_id, "status": "submitted"}  # ❌ Inconsistente
return {"status": "healthy", "agent_running": ...}  # ❌ Inconsistente
```

**Después**:
```python
return ResponseBuilder.task_submitted(task_id)  # ✅ Consistente
return ResponseBuilder.health_check(agent_running)  # ✅ Consistente
```

**Beneficios:**
- ✅ Formato consistente
- ✅ Fácil modificar
- ✅ Single source of truth

---

## 🎯 Patrones de Diseño Aplicados

### 1. Template Method Pattern

#### ✅ Aplicación: `_execute_service_call`

```python
# ✅ Template method que define el flujo común
async def _execute_service_call(self, prompt, system_prompt_key, ...):
    # 1. Optimizar prompt
    optimized_prompt = await self.truthgpt_client.optimize_query(prompt)
    
    # 2. Crear mensajes
    messages = [...]
    
    # 3. Llamar API
    response = await self.openrouter_client.chat_completion(...)
    
    # 4. Formatear respuesta
    result = {...}
    
    return result
```

**Beneficios:**
- ✅ Flujo común definido
- ✅ Fácil extender
- ✅ Consistencia garantizada

---

### 2. Strategy Pattern

#### ✅ Aplicación: Diccionario de Handlers

```python
# ✅ Strategy pattern para routing
def _get_service_handler(self, service_type: str):
    handlers = {
        "calcular_impuestos": self._handle_calcular_impuestos,
        "asesoria_fiscal": self._handle_asesoria_fiscal,
        # ... más handlers ...
    }
    return handlers.get(service_type)
```

**Beneficios:**
- ✅ Fácil agregar nuevas estrategias
- ✅ Sin if/elif largo
- ✅ Escalable

---

### 3. Builder Pattern

#### ✅ Aplicación: ResponseBuilder

```python
# ✅ Builder pattern para respuestas
ResponseBuilder.task_submitted(task_id)
ResponseBuilder.health_check(agent_running)
```

**Beneficios:**
- ✅ Construcción flexible
- ✅ Código legible
- ✅ Fácil extender

---

## 🎯 Convenciones de Código Aplicadas

### 1. Naming Conventions

#### ✅ Clases: PascalCase

```python
# ✅ Correcto
class ContadorSAM3Agent:
    ...

class ResponseBuilder:
    ...

class TaskManager:
    ...
```

#### ✅ Métodos: snake_case

```python
# ✅ Correcto
async def calcular_impuestos(self, ...):
    ...

async def _handle_asesoria_fiscal(self, ...):
    ...
```

#### ✅ Métodos Privados: `_` prefix

```python
# ✅ Correcto
async def _execute_service_call(self, ...):
    ...

def _get_service_handler(self, ...):
    ...
```

---

### 2. Organización de Código

#### ✅ Imports Organizados

```python
# ✅ Standard library
import asyncio
import logging
from typing import Dict, Any, Optional

# ✅ Third-party
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ✅ Local - Core
from ..core.contador_sam3_agent import ContadorSAM3Agent

# ✅ Local - Config
from ..config.contador_sam3_config import ContadorSAM3Config

# ✅ Local - Helpers
from .api_helpers import require_agent
from .error_handlers import handle_task_errors
from .response_builder import ResponseBuilder
```

#### ✅ Métodos Organizados

```python
class ContadorSAM3Agent:
    # ✅ 1. __init__ y inicialización
    def __init__(self, ...):
        ...
    
    # ✅ 2. Métodos de ciclo de vida
    async def start(self):
        ...
    
    async def stop(self):
        ...
    
    # ✅ 3. Métodos privados de procesamiento
    async def _process_task(self, ...):
        ...
    
    def _get_service_handler(self, ...):
        ...
    
    async def _execute_service_call(self, ...):
        ...
    
    # ✅ 4. Handlers específicos
    async def _handle_calcular_impuestos(self, ...):
        ...
    
    # ✅ 5. Métodos públicos
    async def calcular_impuestos(self, ...):
        ...
    
    # ✅ 6. Métodos de utilidad
    async def get_task_status(self, ...):
        ...
```

---

### 3. Documentación

#### ✅ Docstrings Completos

```python
async def calcular_impuestos(
    self,
    regimen: str,
    tipo_impuesto: str,
    datos: Dict[str, Any],
    priority: int = 0,
) -> str:
    """
    Submit tax calculation task.
    
    Args:
        regimen: Fiscal regime
        tipo_impuesto: Tax type
        datos: Input data
        priority: Task priority
        
    Returns:
        Task ID
    """
    # ...
```

**Beneficios:**
- ✅ Documentación clara
- ✅ Fácil entender
- ✅ Mejor IDE support

---

## ✅ Resumen de Mejores Prácticas

### Principios SOLID
- ✅ **SRP**: Cada componente una responsabilidad
- ✅ **OCP**: Extensible sin modificar
- ✅ **LSP**: Interfaces consistentes
- ✅ **ISP**: Interfaces pequeñas
- ✅ **DIP**: Dependencias invertidas

### DRY
- ✅ **Don't Repeat Yourself**: Sin duplicación
- ✅ **Single Source of Truth**: Helpers centralizados

### Código
- ✅ **Type hints**: Completos
- ✅ **Docstrings**: Descriptivos
- ✅ **Naming**: Consistente
- ✅ **Organización**: Clara

### Patrones
- ✅ **Template Method**: `_execute_service_call`
- ✅ **Strategy**: Diccionario de handlers
- ✅ **Builder**: ResponseBuilder

### Convenciones
- ✅ **Naming**: Consistente
- ✅ **Organización**: Lógica
- ✅ **Documentación**: Completa

---

**🎊🎊🎊 Mejores Prácticas Completamente Aplicadas. Código de Calidad Profesional. 🎊🎊🎊**

