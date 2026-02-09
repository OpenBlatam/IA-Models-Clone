# Ejemplos Prácticos de Uso - ContadorSAM3Agent Refactorizado

## 📋 Resumen

Ejemplos prácticos de cómo usar el código refactorizado en diferentes escenarios.

---

## 🎯 Ejemplos de Uso

### Ejemplo 1: Uso Básico del Agente

```python
from contabilidad_mexicana_ai_sam3.core.contador_sam3_agent import ContadorSAM3Agent
from contabilidad_mexicana_ai_sam3.config.contador_sam3_config import ContadorSAM3Config

# Inicializar agente
config = ContadorSAM3Config()
agent = ContadorSAM3Agent(config=config)

# Iniciar agente (modo 24/7)
await agent.start()

# Enviar tarea de cálculo
task_id = await agent.calcular_impuestos(
    regimen="RESICO",
    tipo_impuesto="ISR",
    datos={"ingresos": 100000, "gastos": 20000},
    priority=1
)

# Verificar estado
status = await agent.get_task_status(task_id)
print(f"Task status: {status['status']}")

# Obtener resultado cuando esté listo
result = await agent.get_task_result(task_id)
print(f"Result: {result}")

# Detener agente
await agent.stop()
```

---

### Ejemplo 2: Uso de ServiceHandler Directamente

```python
from contabilidad_mexicana_ai_sam3.core.service_handler import ServiceHandler
from contabilidad_mexicana_ai_sam3.core.system_prompts_builder import SystemPromptsBuilder
from contabilidad_mexicana_ai_sam3.infrastructure.openrouter_client import OpenRouterClient
from contabilidad_mexicana_ai_sam3.infrastructure.truthgpt_client import TruthGPTClient

# Inicializar componentes
openrouter_client = OpenRouterClient(api_key="...")
truthgpt_client = TruthGPTClient(config={})
system_prompts = SystemPromptsBuilder.build_all_prompts()

# Crear service handler
service_handler = ServiceHandler(
    openrouter_client=openrouter_client,
    truthgpt_client=truthgpt_client,
    system_prompts=system_prompts,
    config=config
)

# Usar directamente
parameters = {
    "regimen": "RESICO",
    "tipo_impuesto": "ISR",
    "datos": {"ingresos": 100000}
}

result = await service_handler.handle_calcular_impuestos(parameters)
print(f"Calculation result: {result['resultado']}")
```

---

### Ejemplo 3: Uso de TaskCreator

```python
from contabilidad_mexicana_ai_sam3.core.task_creator import TaskCreator
from contabilidad_mexicana_ai_sam3.core.task_manager import TaskManager

# Inicializar task manager
task_manager = TaskManager()

# Crear tarea usando TaskCreator
task_id = await TaskCreator.create_calcular_impuestos_task(
    task_manager=task_manager,
    regimen="RESICO",
    tipo_impuesto="ISR",
    datos={"ingresos": 100000},
    priority=1
)

print(f"Created task: {task_id}")

# O crear tarea genérica
task_id = await TaskCreator.create_task(
    task_manager=task_manager,
    service_type="asesoria_fiscal",
    parameters={"pregunta": "¿Cómo funciona RESICO?"},
    priority=0
)
```

---

### Ejemplo 4: Uso de API REST

```python
import httpx

# Cliente HTTP
client = httpx.AsyncClient(base_url="http://localhost:8000")

# Enviar tarea de cálculo
response = await client.post(
    "/calcular-impuestos",
    json={
        "regimen": "RESICO",
        "tipo_impuesto": "ISR",
        "datos": {"ingresos": 100000},
        "priority": 1
    }
)

task_data = response.json()
task_id = task_data["task_id"]
print(f"Task submitted: {task_id}")

# Verificar estado
status_response = await client.get(f"/task/{task_id}/status")
status = status_response.json()
print(f"Status: {status['status']}")

# Obtener resultado
result_response = await client.get(f"/task/{task_id}/result")
result = result_response.json()
print(f"Result: {result}")
```

---

### Ejemplo 5: Agregar Nuevo Servicio

```python
# 1. Agregar método en PromptBuilder
# core/prompt_builder.py
@staticmethod
def build_consulta_legal_prompt(pregunta: str) -> str:
    return f"""Consulta legal: {pregunta}"""

# 2. Agregar especialización en SystemPromptsBuilder
# core/system_prompts_builder.py
@staticmethod
def _get_legal_specialization() -> str:
    return "Especialización legal..."

# 3. Agregar handler en ServiceHandler
# core/service_handler.py
async def handle_consulta_legal(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
    pregunta = parameters.get("pregunta")
    return await self.handle_service_request(
        service_type="consulta_legal",
        prompt_builder_method=PromptBuilder.build_consulta_legal_prompt,
        system_prompt_key="consulta_legal",
        response_key="consulta",
        pregunta=pregunta
    )

# 4. Agregar al handler map en ContadorSAM3Agent
def _get_service_handler(self, service_type: str):
    handler_map = {
        # ... existentes ...
        "consulta_legal": self.service_handler.handle_consulta_legal,  # ✅ Agregar
    }
    return handler_map.get(service_type)

# 5. Agregar método público
async def consulta_legal(self, pregunta: str, priority: int = 0) -> str:
    return await TaskCreator.create_task(
        self.task_manager,
        service_type="consulta_legal",
        parameters={"pregunta": pregunta},
        priority=priority
    )

# 6. Usar
task_id = await agent.consulta_legal("¿Cuál es el marco legal de RESICO?")
```

---

### Ejemplo 6: Manejo de Errores

```python
from contabilidad_mexicana_ai_sam3.api.error_handlers import handle_task_errors
from contabilidad_mexicana_ai_sam3.api.api_helpers import require_agent

# En endpoint API
@app.get("/task/{task_id}/status")
@handle_task_errors  # ✅ Maneja errores automáticamente
async def get_task_status(task_id: str):
    agent = require_agent(_agent)  # ✅ Valida agente
    return await agent.get_task_status(task_id)

# En código core
try:
    result = await service_handler.handle_calcular_impuestos(parameters)
except Exception as e:
    logger.error(f"Error: {e}")
    # Manejar error
```

---

### Ejemplo 7: Configuración Personalizada

```python
from contabilidad_mexicana_ai_sam3.config.contador_sam3_config import ContadorSAM3Config

# Configuración personalizada
config = ContadorSAM3Config()
config.openrouter.model = "anthropic/claude-3.5-sonnet"
config.openrouter.temperature = 0.5
config.openrouter.max_tokens = 8000

# Usar configuración
agent = ContadorSAM3Agent(
    config=config,
    max_parallel_tasks=20,  # Más workers
    output_dir="custom_output",
    debug=True
)
```

---

### Ejemplo 8: Monitoreo y Estadísticas

```python
# Obtener estadísticas del executor
stats = agent.parallel_executor.get_stats()
print(f"Total tasks: {stats['total_tasks']}")
print(f"Completed: {stats['completed_tasks']}")
print(f"Failed: {stats['failed_tasks']}")
print(f"Queue size: {stats['queue_size']}")

# Obtener analytics de TruthGPT
analytics = await agent.truthgpt_client.get_analytics()
print(f"TruthGPT analytics: {analytics}")
```

---

## 🎯 Patrones de Uso Comunes

### Patrón 1: Procesamiento Asíncrono

```python
# Enviar múltiples tareas
task_ids = []
for datos in lista_datos:
    task_id = await agent.calcular_impuestos(
        regimen="RESICO",
        tipo_impuesto="ISR",
        datos=datos
    )
    task_ids.append(task_id)

# Esperar resultados
results = []
for task_id in task_ids:
    while True:
        status = await agent.get_task_status(task_id)
        if status['status'] == 'completed':
            result = await agent.get_task_result(task_id)
            results.append(result)
            break
        await asyncio.sleep(1)
```

---

### Patrón 2: Uso con Context Manager

```python
async with ContadorSAM3Agent(config=config) as agent:
    await agent.start()
    
    task_id = await agent.calcular_impuestos(...)
    result = await agent.get_task_result(task_id)
    
    # Agent se cierra automáticamente
```

---

### Patrón 3: Validación y Manejo de Errores

```python
from contabilidad_mexicana_ai_sam3.core.task_validator import TaskValidator

# Validar tarea antes de procesar
try:
    task = TaskValidator.validate_task_exists(task_manager, task_id)
    # Procesar tarea
except ValueError as e:
    logger.error(f"Task validation failed: {e}")
    # Manejar error
```

---

## ✅ Resumen de Ejemplos

### Casos de Uso Cubiertos

1. ✅ **Uso básico del agente**
2. ✅ **Uso directo de ServiceHandler**
3. ✅ **Uso de TaskCreator**
4. ✅ **Uso de API REST**
5. ✅ **Agregar nuevo servicio**
6. ✅ **Manejo de errores**
7. ✅ **Configuración personalizada**
8. ✅ **Monitoreo y estadísticas**

### Patrones Cubiertos

1. ✅ **Procesamiento asíncrono**
2. ✅ **Context managers**
3. ✅ **Validación y errores**

---

**Fecha**: 2024  
**Versión**: 1.0.0  
**Estado**: ✅ Ejemplos Completos

