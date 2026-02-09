---
description: Reglas de desarrollo para Antigravity basadas en la arquitectura autonomous_sam3_agent
---

# 🚀 Reglas de Antigravity - Basadas en Autonomous SAM3 Agent

## 1. Arquitectura Modular

### Estructura de Directorios Requerida
```
proyecto/
├── __init__.py
├── main.py                    # Punto de entrada principal
├── config.yaml                # Configuración centralizada
├── core/                      # Lógica principal
│   ├── __init__.py
│   ├── agent_core.py          # Núcleo del agente
│   ├── task_manager.py        # Gestión de tareas
│   ├── parallel_executor.py   # Ejecución paralela
│   ├── validators.py          # Validaciones
│   ├── helpers.py             # Funciones auxiliares
│   ├── cache.py               # Sistema de caché
│   ├── rate_limiter.py        # Control de rate limiting
│   └── metrics.py             # Métricas y monitoreo
├── infrastructure/            # Clientes externos
│   ├── __init__.py
│   ├── openrouter_client.py   # Cliente API LLM
│   ├── retry_helpers.py       # Lógica de reintentos
│   └── *_client.py            # Otros clientes
└── system_prompts/            # Prompts del sistema
    ├── system_prompt.txt
    └── system_prompt_*.txt
```

---

## 2. Reglas de Código

### 2.1 Async/Await Obligatorio
```python
# ✅ CORRECTO
async def process_task(self, task: Dict[str, Any]) -> Dict:
    result = await self.client.call_api(task)
    return result

# ❌ INCORRECTO
def process_task(self, task):
    result = self.client.call_api(task)
    return result
```

### 2.2 Type Hints Siempre
```python
# ✅ CORRECTO
def submit_task(
    self,
    image_path: str,
    text_prompt: str,
    priority: int = 0,
) -> str:
    pass

# ❌ INCORRECTO
def submit_task(self, image_path, text_prompt, priority=0):
    pass
```

### 2.3 Docstrings Descriptivos
```python
# ✅ CORRECTO
async def _agent_inference(
    self,
    image_path: str,
    initial_text_prompt: str,
    task_id: str,
) -> Dict[str, Any]:
    """
    Run SAM3 agent inference with OpenRouter LLM.
    
    Based on sam3.agent.agent_core.agent_inference but adapted for
    async operation and OpenRouter integration.
    
    Args:
        image_path: Path to input image
        initial_text_prompt: Text prompt for segmentation
        task_id: Unique task identifier
        
    Returns:
        Dictionary with inference results
        
    Raises:
        ValidationError: If inputs are invalid
    """
    pass
```

---

## 3. Patrones de Diseño Obligatorios

### 3.1 Configuración Centralizada (config.yaml)
```yaml
# SIEMPRE usar este formato
api:
  api_key: ${API_KEY}  # Variables de entorno
  model: "default-model"
  timeout: 60.0
  max_retries: 3
  retry_delay: 1.0

agent:
  max_parallel_tasks: 10
  output_dir: "output"
  debug: false

logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: "agent.log"
```

### 3.2 Clase Agent Principal
```python
class AutonomousAgent:
    """
    Patrón obligatorio para agentes autónomos.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        max_parallel_tasks: int = 10,
        output_dir: str = "output",
        debug: bool = False,
    ):
        # 1. Configurar logging
        self._setup_logging()
        
        # 2. Inicializar componentes
        self.task_manager = TaskManager()
        self.parallel_executor = ParallelExecutor(max_workers=max_parallel_tasks)
        self.api_client = APIClient(api_key)
        
        # 3. Estado
        self._running = False
        
    async def start(self) -> None:
        """Start the autonomous agent in continuous operation mode."""
        pass
        
    async def stop(self) -> None:
        """Stop the autonomous agent."""
        pass
        
    async def submit_task(self, **kwargs) -> str:
        """Submit a new task to the agent."""
        pass
        
    def get_task_status(self, task_id: str) -> Dict:
        """Get status of a task."""
        pass
        
    def get_stats(self) -> Dict:
        """Get agent statistics."""
        pass
        
    async def health_check(self) -> Dict:
        """Perform health check on agent components."""
        pass
```

### 3.3 Task Manager
```python
class TaskManager:
    """
    Gestión de cola de tareas con prioridades.
    """
    
    def __init__(self, max_pending_tasks: int = 1000):
        self._pending_queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self._tasks: Dict[str, Dict] = {}
        self._results: Dict[str, Any] = {}
        
    async def add_task(self, task: Dict, priority: int = 0) -> str:
        """Add task to queue with priority."""
        pass
        
    async def get_next_task(self) -> Optional[Dict]:
        """Get next task from queue."""
        pass
        
    def update_status(self, task_id: str, status: str) -> None:
        """Update task status."""
        pass
        
    def store_result(self, task_id: str, result: Any) -> None:
        """Store task result."""
        pass
```

### 3.4 Parallel Executor
```python
class ParallelExecutor:
    """
    Ejecutor paralelo con worker pool.
    """
    
    def __init__(self, max_workers: int = 10):
        self.max_workers = max_workers
        self._active_workers = 0
        self._total_tasks = 0
        self._completed_tasks = 0
        self._failed_tasks = 0
        
    async def execute(
        self, 
        task: Dict,
        processor: Callable,
    ) -> Any:
        """Execute task with worker pool management."""
        pass
        
    def get_stats(self) -> Dict:
        """Get executor statistics."""
        return {
            "max_workers": self.max_workers,
            "active_workers": self._active_workers,
            "total_tasks": self._total_tasks,
            "completed_tasks": self._completed_tasks,
            "failed_tasks": self._failed_tasks,
        }
```

---

## 4. Infrastructure Clients

### 4.1 Patrón de Cliente API
```python
class APIClient:
    """
    Cliente para APIs externas.
    """
    
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.example.com",
        timeout: float = 60.0,
        max_retries: int = 3,
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        self._client: Optional[httpx.AsyncClient] = None
        
    async def __aenter__(self):
        self._client = httpx.AsyncClient(timeout=self.timeout)
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._client:
            await self._client.aclose()
            
    @retry_with_exponential_backoff
    async def call(self, endpoint: str, payload: Dict) -> Dict:
        """Make API call with retry logic."""
        pass
```

### 4.2 Retry Helpers
```python
# SIEMPRE implementar retry con backoff exponencial
import asyncio
from functools import wraps

def retry_with_exponential_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            delay = base_delay
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    await asyncio.sleep(delay)
                    delay = min(delay * 2, max_delay)
        return wrapper
    return decorator
```

---

## 5. System Prompts

### 5.1 Estructura de Prompts
```text
# system_prompts/system_prompt.txt

You are a helpful [ROLE] assistant capable of [CAPABILITIES].

[CONTEXT ABOUT WHAT USER PROVIDES]

[DETAILED INSTRUCTIONS - NUMBERED LIST]
1. Rule one with examples
2. Rule two with examples
...

Available tools:

[TOOL_NAME]: [DESCRIPTION]
Use cases: "[WHEN TO USE]"
Parameters: {JSON_SCHEMA}
Return type: [RETURN DESCRIPTION]
Important rules for using [TOOL_NAME]:
1. [Rule 1]
2. [Rule 2]

Steps for Each Turn:
1. Analyze the input carefully.
2. Think about what tool to call.
3. Call exactly one tool using: <tool>{"name": "tool_name", "parameters": {...}}</tool>
4. Stop immediately after calling the tool.
```

### 5.2 Formato de Tool Calls
```xml
<tool>{"name": "tool_name", "parameters": {"param1": "value1"}}</tool>
```

### 5.3 Formato de Respuesta Estructurada
```xml
<think>Analyze and reason step-by-step</think>
<verdict>Accept</verdict> or <verdict>Reject</verdict>
```

---

## 6. Error Handling

### 6.1 Excepciones Personalizadas
```python
class AgentError(Exception):
    """Base exception for agent errors."""
    pass

class ValidationError(AgentError):
    """Raised when validation fails."""
    pass

class TimeoutError(AgentError):
    """Raised when operation times out."""
    pass

class RateLimitError(AgentError):
    """Raised when rate limit is exceeded."""
    pass
```

### 6.2 Try/Except Pattern
```python
async def _process_task(self, task: Dict) -> Dict:
    try:
        result = await self._execute(task)
        self.task_manager.update_status(task["id"], "completed")
        return result
    except ValidationError as e:
        logger.warning(f"Validation failed: {e}")
        self.task_manager.update_status(task["id"], "failed")
        return {"error": str(e)}
    except TimeoutError as e:
        logger.error(f"Task timed out: {e}")
        self.task_manager.update_status(task["id"], "timeout")
        return {"error": "timeout"}
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        self.task_manager.update_status(task["id"], "error")
        raise
```

---

## 7. Logging

### 7.1 Logger Setup
```python
import logging

logger = logging.getLogger(__name__)

# En __init__ de la clase principal
def _setup_logging(self):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("agent.log"),
            logging.StreamHandler(),
        ]
    )
```

### 7.2 Log Levels
```python
# DEBUG: Información detallada para debugging
logger.debug(f"Processing task {task_id} with params: {params}")

# INFO: Confirmación de que las cosas funcionan
logger.info(f"Task {task_id} completed successfully")

# WARNING: Algo inesperado pero no crítico
logger.warning(f"Retry attempt {attempt} for task {task_id}")

# ERROR: Problema serio que previene funcionalidad
logger.error(f"Failed to process task {task_id}: {error}")

# CRITICAL: Error muy serio que puede detener el programa
logger.critical(f"Agent failed to start: {error}")
```

---

## 8. Testing

### 8.1 Verificación de Setup
```python
# verify_setup.py - SIEMPRE incluir

def check_imports() -> bool:
    """Verificar que todos los imports funcionan."""
    pass

def check_structure() -> bool:
    """Verificar estructura de archivos."""
    pass

def check_api_key() -> bool:
    """Verificar configuración de API."""
    pass

def check_cuda() -> bool:
    """Verificar disponibilidad de GPU."""
    pass

def main() -> int:
    """Ejecutar todas las verificaciones."""
    results = {
        "imports": check_imports(),
        "structure": check_structure(),
        "api_key": check_api_key(),
        "cuda": check_cuda(),
    }
    
    # Print summary
    for check, result in results.items():
        status = "✅ OK" if result else "⚠️ ADVERTENCIA"
        print(f"  {check}: {status}")
    
    return 0 if all(results.values()) else 1

if __name__ == "__main__":
    sys.exit(main())
```

---

## 9. Ejecución Continua 24/7

### 9.1 Patrón de Main Loop
```python
# main.py
import asyncio
import signal
from core.agent_core import AutonomousAgent

async def main():
    agent = AutonomousAgent()
    
    # Handle graceful shutdown
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(agent.stop()))
    
    try:
        await agent.start()
    except KeyboardInterrupt:
        await agent.stop()
    except Exception as e:
        logger.exception(f"Agent crashed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
```

---

## 10. Resumen de Reglas Críticas

| Regla | Descripción |
|-------|-------------|
| 🔄 **Async First** | Todo async/await, nunca blocking |
| 📝 **Type Hints** | Siempre especificar tipos |
| 📁 **Modular** | core/, infrastructure/, system_prompts/ |
| ⚙️ **Config Central** | config.yaml para toda configuración |
| 🔁 **Retry Logic** | Exponential backoff en APIs |
| 📊 **Metrics** | Estadísticas de tareas/workers |
| 🏥 **Health Check** | Verificación de componentes |
| 📋 **Logging** | Logs estructurados y niveles apropiados |
| ⚡ **Parallel** | Worker pool para tareas |
| 🛡️ **Validation** | Validar inputs antes de procesar |
