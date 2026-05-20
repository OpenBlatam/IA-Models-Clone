# Autonomous SAM3 Agent

Agente autónomo basado en la arquitectura SAM3 que ejecuta 24/7 en paralelo utilizando OpenRouter para procesamiento de lenguaje natural.

## Características

- ✅ **Ejecución continua 24/7**: El agente opera de forma continua sin interrupciones
- ✅ **Procesamiento paralelo**: Múltiples tareas se ejecutan simultáneamente
- ✅ **Integración OpenRouter**: Utiliza OpenRouter API para acceso a múltiples modelos LLM
- ✅ **Arquitectura SAM3**: Basado en la arquitectura de Segment Anything Model 3
- ✅ **Gestión de tareas**: Sistema de cola con prioridades
- ✅ **Monitoreo y logging**: Registro completo de operaciones
- ✅ **Resiliencia**: Recuperación automática de errores

## Estructura del Proyecto

```
autonomous_sam3_agent/
├── __init__.py
├── main.py                 # Punto de entrada principal
├── config.yaml            # Configuración del agente
├── README.md              # Este archivo
├── core/                  # Componentes principales
│   ├── __init__.py
│   ├── agent_core.py      # Núcleo del agente autónomo
│   ├── task_manager.py    # Gestión de tareas
│   └── parallel_executor.py  # Ejecutor paralelo
├── infrastructure/        # Infraestructura
│   ├── __init__.py
│   ├── openrouter_client.py  # Cliente OpenRouter
│   └── sam3_client.py    # Cliente SAM3
└── system_prompts/        # Prompts del sistema
    ├── system_prompt.txt
    └── system_prompt_iterative_checking.txt
```

## Requisitos

### Dependencias Python

```bash
pip install asyncio httpx pyyaml torch torchvision pillow
```

### Dependencias SAM3

El agente requiere acceso a la carpeta `sam3-main` en el mismo directorio padre:

```
TruthGPT-main/
├── sam3-main/          # Requerido
└── autonomous_sam3_agent/
```

### Variables de Entorno

```bash
export OPENROUTER_API_KEY="tu-api-key-aqui"
```

## Instalación

1. **Clonar o asegurar que sam3-main está disponible**:
   ```bash
   # Asegúrate de que sam3-main está en el directorio correcto
   ls ../sam3-main/
   ```

2. **Instalar dependencias**:
   ```bash
   pip install -r requirements.txt  # Si existe
   # O instalar manualmente:
   pip install httpx pyyaml torch torchvision pillow
   ```

3. **Configurar API Key**:
   ```bash
   export OPENROUTER_API_KEY="tu-api-key"
   ```

## Uso

### Ejecución Básica

```bash
python main.py
```

El agente iniciará en modo 24/7 y comenzará a procesar tareas.

### Configuración

Edita `config.yaml` para personalizar:

```yaml
openrouter:
  model: "anthropic/claude-3.5-sonnet"  # Modelo a usar

agent:
  max_parallel_tasks: 10  # Número máximo de tareas paralelas
  output_dir: "autonomous_agent_output"
```

### Uso Programático

```python
import asyncio
from core.agent_core import AutonomousSAM3Agent

async def example():
    # Inicializar agente
    agent = AutonomousSAM3Agent(
        openrouter_api_key="tu-api-key",
        max_parallel_tasks=10,
    )
    
    # Iniciar agente en background
    agent_task = asyncio.create_task(agent.start())
    
    # Enviar tarea
    task_id = await agent.submit_task(
        image_path="path/to/image.jpg",
        text_prompt="a person holding a cup",
        priority=5,
    )
    
    # Verificar estado
    status = await agent.get_task_status(task_id)
    print(f"Task status: {status['status']}")
    
    # Obtener resultado cuando esté completo
    result = await agent.get_task_result(task_id)
    if result:
        print(f"Found {len(result['pred_masks'])} masks")

asyncio.run(example())
```

## Arquitectura

### Componentes Principales

1. **AutonomousSAM3Agent**: Núcleo del agente que coordina todos los componentes
2. **TaskManager**: Gestiona la cola de tareas con prioridades
3. **ParallelExecutor**: Ejecuta tareas en paralelo con worker pool
4. **OpenRouterClient**: Cliente para OpenRouter API con retry y pooling
5. **SAM3Client**: Cliente para modelo SAM3 con soporte async

### Flujo de Procesamiento

1. **Recepción de Tarea**: El agente recibe una tarea con imagen y prompt
2. **Cola de Tareas**: La tarea se añade a la cola con prioridad
3. **Procesamiento Paralelo**: Un worker toma la tarea y la procesa
4. **Inferencia SAM3**: Se ejecuta la inferencia del modelo SAM3
5. **LLM con OpenRouter**: Se usa OpenRouter para razonamiento y decisiones
6. **Resultado**: Se guarda el resultado y se actualiza el estado

## Monitoreo

### Logs

Los logs se guardan en:
- `autonomous_agent.log`: Archivo de log principal
- Consola: Salida en tiempo real

### Estadísticas

```python
# Obtener estadísticas del ejecutor paralelo
stats = agent.parallel_executor.get_stats()
print(f"Total tasks: {stats['total_tasks']}")
print(f"Completed: {stats['completed_tasks']}")
print(f"Failed: {stats['failed_tasks']}")
```

## Troubleshooting

### Error: "SAM3 components not available"

Asegúrate de que `sam3-main` está en el directorio correcto y que los imports funcionan.

### Error: "OpenRouter API key not configured"

Configura la variable de entorno:
```bash
export OPENROUTER_API_KEY="tu-api-key"
```

### Error: CUDA out of memory

Reduce el número de workers paralelos en `config.yaml`:
```yaml
agent:
  max_parallel_tasks: 5  # Reducir de 10 a 5
```

## Desarrollo

### Estructura de Código

- **core/**: Lógica principal del agente
- **infrastructure/**: Clientes y servicios externos
- **system_prompts/**: Prompts para el LLM

### Testing

```python
# Ejemplo de test
import asyncio
from core.agent_core import AutonomousSAM3Agent

async def test_agent():
    agent = AutonomousSAM3Agent(
        max_parallel_tasks=2,
        debug=True,
    )
    
    task_id = await agent.submit_task(
        image_path="test_image.jpg",
        text_prompt="test prompt",
    )
    
    # Esperar resultado
    await asyncio.sleep(30)
    result = await agent.get_task_result(task_id)
    print(result)

asyncio.run(test_agent())
```

## Licencia

Basado en SAM3 (Meta Platforms, Inc.) y adaptado para uso autónomo.

## Contribuciones

Este agente está basado en la arquitectura de `sam3-main` y adaptado para ejecución autónoma 24/7 con OpenRouter.
