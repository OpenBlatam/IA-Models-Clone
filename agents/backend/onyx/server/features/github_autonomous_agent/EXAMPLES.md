# Ejemplos de Uso - GitHub Autonomous Agent

## 📚 Índice

- [Cliente API](#cliente-api)
- [WebSocket](#websocket)
- [Sistema de Eventos](#sistema-de-eventos)
- [LLM Service](#llm-service)
- [Stats y Métricas](#stats-y-métricas)

## 🔌 Cliente API

### Uso Básico

```python
from api.client import APIClient

async def main():
    async with APIClient(base_url="http://localhost:8030") as client:
        # Health check
        health = await client.health_check()
        print(f"Status: {health['status']}")
        
        # Crear tarea
        task = await client.create_task(
            repository_owner="owner",
            repository_name="repo",
            instruction="create file: README.md with project description"
        )
        print(f"Task created: {task['id']}")
        
        # Monitorear tarea
        while True:
            task_status = await client.get_task(task['id'])
            print(f"Status: {task_status['status']}")
            if task_status['status'] in ['completed', 'failed']:
                break
            await asyncio.sleep(2)

import asyncio
asyncio.run(main())
```

### Control del Agente

```python
async with APIClient() as client:
    # Iniciar agente
    await client.start_agent()
    
    # Verificar estado
    status = await client.get_agent_status()
    print(f"Agent running: {status['is_running']}")
    
    # Detener agente
    await client.stop_agent()
```

### Trabajar con Repositorios

```python
async with APIClient() as client:
    # Obtener información de repositorio
    repo_info = await client.get_repository_info("owner", "repo")
    print(f"Stars: {repo_info['stars']}")
    print(f"Language: {repo_info['language']}")
    
    # Listar repositorios
    repos = await client.list_repositories(owner="owner", limit=10)
    for repo in repos['repositories']:
        print(f"- {repo['full_name']}")
```

## 🔌 WebSocket

### Conexión Básica

```python
import asyncio
import websockets
import json

async def websocket_client():
    uri = "ws://localhost:8030/ws/tasks"
    
    async with websockets.connect(uri) as websocket:
        # Suscribirse a actualizaciones de tareas
        await websocket.send(json.dumps({
            "type": "subscribe",
            "data": {"type": "tasks"}
        }))
        
        # Escuchar mensajes
        async for message in websocket:
            data = json.loads(message)
            if data['type'] == 'task_update':
                task = data['data']
                print(f"Task {task['id']} updated: {data['status']}")

asyncio.run(websocket_client())
```

### React Hook

```typescript
import { useEffect, useState } from 'react';

function useTaskUpdates() {
  const [tasks, setTasks] = useState([]);
  const [connected, setConnected] = useState(false);
  
  useEffect(() => {
    const ws = new WebSocket('ws://localhost:8030/ws/tasks');
    
    ws.onopen = () => {
      setConnected(true);
      ws.send(JSON.stringify({
        type: 'subscribe',
        data: { type: 'tasks' }
      }));
    };
    
    ws.onmessage = (event) => {
      const msg = JSON.parse(event.data);
      if (msg.type === 'task_update') {
        setTasks(prev => {
          const index = prev.findIndex(t => t.id === msg.data.id);
          if (index >= 0) {
            const updated = [...prev];
            updated[index] = msg.data;
            return updated;
          }
          return [...prev, msg.data];
        });
      }
    };
    
    ws.onerror = () => setConnected(false);
    ws.onclose = () => setConnected(false);
    
    return () => ws.close();
  }, []);
  
  return { tasks, connected };
}
```

## 📡 Sistema de Eventos

### Suscribirse a Eventos

```python
from core.events import get_event_bus, EventType, Event

async def handle_task_events(event: Event):
    """Manejar eventos de tareas."""
    if event.event_type == EventType.TASK_COMPLETED:
        task = event.data['task']
        print(f"✅ Task {task['id']} completed!")
        
        # Enviar notificación, actualizar UI, etc.
        await send_notification(f"Task {task['id']} completed")
        
    elif event.event_type == EventType.TASK_FAILED:
        task = event.data['task']
        print(f"❌ Task {task['id']} failed: {task.get('error')}")
        
        # Alertar, loggear, etc.
        await send_alert(f"Task {task['id']} failed")

# Suscribirse
event_bus = get_event_bus()
event_bus.subscribe(EventType.TASK_COMPLETED, handle_task_events)
event_bus.subscribe(EventType.TASK_FAILED, handle_task_events)
```

### Publicar Eventos

```python
from core.events import publish_task_event, publish_agent_event, EventType

# Publicar evento de tarea
await publish_task_event(
    EventType.TASK_COMPLETED,
    task_dict,
    source="task_processor"
)

# Publicar evento de agente
await publish_agent_event(
    EventType.AGENT_STARTED,
    agent_state,
    source="agent_routes"
)
```

### Obtener Historial

```python
from core.events import get_event_bus, EventType

event_bus = get_event_bus()

# Últimos 100 eventos
recent_events = event_bus.get_history(limit=100)

# Eventos de un tipo específico
task_events = event_bus.get_history(
    event_type=EventType.TASK_COMPLETED,
    limit=50
)

for event in task_events:
    print(f"{event.timestamp}: {event.event_type.value}")
```

## 🤖 LLM Service

### Análisis de Código

```python
from core.services import LLMService
from config.di_setup import get_service

llm = get_service("llm_service")

# Análisis general
response = await llm.analyze_code(
    code="""
def calculate_total(items):
    total = 0
    for item in items:
        total += item.price
    return total
    """,
    language="python",
    analysis_type="general"
)
print(response.content)

# Análisis de seguridad
security_analysis = await llm.analyze_code(
    code=code,
    language="python",
    analysis_type="security"
)

# Análisis de performance
perf_analysis = await llm.analyze_code(
    code=code,
    language="python",
    analysis_type="performance"
)
```

### Generar Instrucciones

```python
# Convertir descripción en instrucción estructurada
response = await llm.generate_instruction(
    description="Crear un archivo README.md con información del proyecto",
    context="Proyecto: GitHub Agent\nLenguaje: Python"
)

instruction = response.content
# Output: "create file: README.md"
```

### Generar en Paralelo

```python
# Comparar respuestas de múltiples modelos
responses = await llm.generate_parallel(
    prompt="Explica qué es async/await en Python",
    models=["openai/gpt-4o-mini", "anthropic/claude-3.5-sonnet"]
)

for model, response in responses.items():
    print(f"\n{model}:")
    print(response.content)
```

## 📊 Stats y Métricas

### Obtener Estadísticas

```python
from api.client import APIClient

async with APIClient() as client:
    # Resumen general
    overview = await client.get_stats_overview()
    print(f"Total tasks: {overview['tasks']['total']}")
    print(f"By status: {overview['tasks']['by_status']}")
    
    # Resumen de tareas
    summary = await client.get_tasks_summary()
    print(f"Recent tasks: {summary['recent_tasks']}")
    
    # Rendimiento
    performance = await client.get_performance_stats()
    print(f"Cache hit rate: {performance['cache']['hit_rate']}%")
```

### Métricas de Servicios

```python
from config.di_setup import get_service

# Cache stats
cache = get_service("cache_service")
cache_stats = cache.get_stats()
print(f"Cache hits: {cache_stats['hits']}")
print(f"Hit rate: {cache_stats['hit_rate']}%")

# Metrics
metrics = get_service("metrics_service")
all_metrics = metrics.get_metrics()
print(f"Metrics: {all_metrics}")

# Rate limit
rate_limit = get_service("rate_limit_service")
stats = rate_limit.get_stats("token_123")
print(f"Remaining: {stats['remaining']}")
```

## 🔄 Flujo Completo

### Crear y Monitorear Tarea

```python
import asyncio
from api.client import APIClient
import websockets
import json

async def create_and_monitor_task():
    # Crear tarea
    async with APIClient() as client:
        task = await client.create_task(
            repository_owner="owner",
            repository_name="repo",
            instruction="create file: test.py with hello world"
        )
        task_id = task['id']
        print(f"Task created: {task_id}")
    
    # Monitorear vía WebSocket
    uri = "ws://localhost:8030/ws/tasks"
    async with websockets.connect(uri) as ws:
        while True:
            message = await ws.recv()
            data = json.loads(message)
            
            if data['type'] == 'task_update':
                task = data['data']
                if task['id'] == task_id:
                    print(f"Task status: {task['status']}")
                    if task['status'] in ['completed', 'failed']:
                        break

asyncio.run(create_and_monitor_task())
```

### Dashboard Completo

```python
from api.client import APIClient
from core.events import get_event_bus, EventType

async def dashboard():
    client = APIClient()
    
    # Cargar datos iniciales
    overview = await client.get_stats_overview()
    tasks = await client.list_tasks(limit=20)
    
    print("=== Dashboard ===")
    print(f"Agent: {'🟢 Running' if overview['agent']['is_running'] else '🔴 Stopped'}")
    print(f"Total Tasks: {overview['tasks']['total']}")
    print(f"Cache Hit Rate: {overview['cache']['hit_rate']}%")
    
    # Suscribirse a eventos
    event_bus = get_event_bus()
    
    async def on_update(event):
        print(f"Event: {event.event_type.value}")
        # Actualizar dashboard
    
    event_bus.subscribe(EventType.TASK_UPDATED, on_update)
    event_bus.subscribe(EventType.AGENT_STATUS_CHANGED, on_update)
    
    # Mantener dashboard activo
    await asyncio.sleep(60)

asyncio.run(dashboard())
```

## 🎯 Casos de Uso Avanzados

### Pipeline Automático

```python
async def automated_pipeline():
    """Pipeline automático de tareas."""
    client = APIClient()
    
    tasks = [
        ("owner/repo1", "create file: README.md"),
        ("owner/repo2", "update file: setup.py"),
        ("owner/repo3", "create branch: feature/new-feature")
    ]
    
    created_tasks = []
    for repo, instruction in tasks:
        owner, repo_name = repo.split("/")
        task = await client.create_task(owner, repo_name, instruction)
        created_tasks.append(task['id'])
    
    # Monitorear todas
    while True:
        all_completed = True
        for task_id in created_tasks:
            task = await client.get_task(task_id)
            if task['status'] not in ['completed', 'failed']:
                all_completed = False
                break
        
        if all_completed:
            break
        await asyncio.sleep(5)
    
    print("All tasks completed!")

asyncio.run(automated_pipeline())
```

### Análisis Automático de PRs

```python
async def analyze_pr_code(code_diff: str):
    """Analizar código de PR automáticamente."""
    llm = get_service("llm_service")
    
    # Análisis múltiple
    analyses = await asyncio.gather(
        llm.analyze_code(code_diff, analysis_type="security"),
        llm.analyze_code(code_diff, analysis_type="performance"),
        llm.analyze_code(code_diff, analysis_type="bugs")
    )
    
    return {
        "security": analyses[0].content,
        "performance": analyses[1].content,
        "bugs": analyses[2].content
    }
```

---

**Versión**: 1.0  
**Última actualización**: Enero 2025



