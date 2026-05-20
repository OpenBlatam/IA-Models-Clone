# Unified AI Model

API de IA unificada lista para frontend. Usa DeepSeek por defecto vía OpenRouter.

## 🚀 Quick Start

### 1. Configurar API Key

**Opción A: DeepSeek (Recomendado)**
```bash
# Windows
set DEEPSEEK_API_KEY=sk-c5d0a8db81d14950ae13ec8434c6bfad

# Linux/Mac
export DEEPSEEK_API_KEY=sk-c5d0a8db81d14950ae13ec8434c6bfad
```

**Opción B: OpenRouter (múltiples modelos)**
```bash
set OPENROUTER_API_KEY=sk-or-v1-tu-api-key
```

### 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 3. Iniciar servidor

```bash
# Windows
start.bat

# Linux/Mac
./start.sh

# O directamente
python run.py
```

El servidor inicia en `http://localhost:8050`

## 📚 Endpoints para Frontend

| Endpoint | Método | Descripción |
|----------|--------|-------------|
| `/api/v1/chat` | POST | Chat con IA |
| `/api/v1/chat/stream` | POST | Chat con streaming |
| `/api/v1/generate` | POST | Generación de texto |
| `/api/v1/agents` | POST | Crear agente continuo |
| `/api/v1/health` | GET | Health check |
| `/docs` | GET | Documentación interactiva |

## 🔧 Características

- **Chat con Memoria**: Conversaciones con historial y contexto
- **Streaming**: Respuestas token por token en tiempo real
- **Agentes Continuos**: Procesamiento 24/7 hasta que el usuario pause
- **Procesamiento Paralelo**: Múltiples tareas concurrentes
- **DeepSeek por defecto**: Modelo económico y potente

## ⚙️ Variables de Entorno

```bash
# API Key (una de las dos)
DEEPSEEK_API_KEY=sk-xxx        # Recomendado - API directa
OPENROUTER_API_KEY=sk-or-v1-xxx  # Alternativa - múltiples modelos

# Opcional (con valores por defecto)
UNIFIED_AI_DEFAULT_MODEL=deepseek-chat
UNIFIED_AI_HOST=0.0.0.0
UNIFIED_AI_PORT=8050
UNIFIED_AI_CORS_ORIGINS=http://localhost:3000
UNIFIED_AI_DEBUG=false
```

## 🎯 Uso desde Frontend

### Chat Simple (JavaScript/TypeScript)

```javascript
// Chat básico
async function chat(message) {
  const response = await fetch('http://localhost:8050/api/v1/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message })
  });
  const data = await response.json();
  return data.message.content;
}

// Uso
const respuesta = await chat('Hola, ¿cómo estás?');
console.log(respuesta);
```

### Chat con Streaming (React)

```javascript
async function chatStream(message, onChunk) {
  const response = await fetch('http://localhost:8050/api/v1/chat/stream', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message, stream: true })
  });

  const reader = response.body.getReader();
  const decoder = new TextDecoder();

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    
    const text = decoder.decode(value);
    const lines = text.split('\n').filter(line => line.startsWith('data: '));
    
    for (const line of lines) {
      const data = line.slice(6);
      if (data === '[DONE]') return;
      
      try {
        const parsed = JSON.parse(data);
        onChunk(parsed.chunk);
      } catch {}
    }
  }
}

// Uso en React
const [response, setResponse] = useState('');

await chatStream('Explica qué es JavaScript', (chunk) => {
  setResponse(prev => prev + chunk);
});
```

### Agente Continuo (Frontend Control)

```javascript
// Crear agente
const agentRes = await fetch('http://localhost:8050/api/v1/agents', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ name: 'MiAgente' })
});
const { agent_id } = await agentRes.json();

// Enviar tarea
await fetch(`http://localhost:8050/api/v1/agents/${agent_id}/tasks`, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    description: 'Analiza tendencias de IA',
    priority: 8
  })
});

// Pausar agente
await fetch(`http://localhost:8050/api/v1/agents/${agent_id}/pause`, { method: 'POST' });

// Reanudar
await fetch(`http://localhost:8050/api/v1/agents/${agent_id}/resume`, { method: 'POST' });

// Detener
await fetch(`http://localhost:8050/api/v1/agents/${agent_id}/stop`, { method: 'POST' });
```

### curl Examples

```bash
# Chat
curl -X POST http://localhost:8050/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hola!"}'

# Crear agente
curl -X POST http://localhost:8050/api/v1/agents \
  -H "Content-Type: application/json" \
  -d '{"name": "TestAgent"}'

# Health check
curl http://localhost:8050/api/v1/health
```

### Documentación Interactiva

- **Swagger UI**: http://localhost:8050/docs
- **ReDoc**: http://localhost:8050/redoc

## 🏗️ Arquitectura

```
unified_ai_model/
├── __init__.py           # Package init
├── config.py             # Configuración
├── main.py               # Aplicación FastAPI
├── requirements.txt      # Dependencias
├── README.md             # Documentación
├── api/
│   ├── __init__.py
│   ├── routes.py         # Endpoints FastAPI
│   └── schemas/
│       └── __init__.py   # Pydantic schemas
└── core/
    ├── __init__.py
    ├── llm_client.py     # Cliente OpenRouter
    ├── llm_service.py    # Servicio LLM con caché
    ├── chat_service.py   # Servicio de chat
    └── performance_monitor.py  # Métricas
```

## 🔧 Modelos Soportados

El sistema soporta todos los modelos disponibles en OpenRouter:

| Provider | Modelo | ID |
|----------|--------|-----|
| DeepSeek | DeepSeek Chat | `deepseek/deepseek-chat` |
| OpenAI | GPT-4o | `openai/gpt-4o` |
| OpenAI | GPT-4o Mini | `openai/gpt-4o-mini` |
| Anthropic | Claude 3.5 Sonnet | `anthropic/claude-3.5-sonnet` |
| Google | Gemini Pro 1.5 | `google/gemini-pro-1.5` |
| Meta | Llama 3 70B | `meta-llama/llama-3-70b-instruct` |
| Mistral | Mistral Large | `mistralai/mistral-large` |

Ver la lista completa en: https://openrouter.ai/models

## 📊 Monitoreo

### Estadísticas

```bash
# Obtener estadísticas
curl http://localhost:8050/api/v1/stats

# Respuesta ejemplo:
{
  "uptime_seconds": 3600.5,
  "requests": {
    "total": 150,
    "successful": 145,
    "failed": 5,
    "error_rate": 3.33
  },
  "cache": {
    "hits": 50,
    "misses": 100,
    "hit_rate": 33.33
  },
  "latency": {
    "mean": 1250.5,
    "p95": 2500.0,
    "p99": 3000.0
  }
}
```

### Health Check

```bash
curl http://localhost:8050/api/v1/health
```

## 🛡️ Características de Resiliencia

- **Circuit Breaker**: Previene cascadas de fallos
- **Rate Limiting**: Control de tasa de solicitudes
- **Retry con Backoff**: Reintentos automáticos con espera exponencial
- **Caché Inteligente**: Reduce llamadas duplicadas a la API
- **Connection Pooling**: Conexiones HTTP eficientes

## 🔗 Integración con Frontend

Este backend está diseñado para ser consumido por cualquier frontend. Ejemplo con JavaScript:

```javascript
// Chat simple
const response = await fetch('http://localhost:8050/api/v1/chat', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    message: 'Hola!',
    conversation_id: 'my-conversation'
  })
});
const data = await response.json();
console.log(data.message.content);

// Streaming
const eventSource = new EventSource('http://localhost:8050/api/v1/chat/stream');
eventSource.onmessage = (event) => {
  if (event.data === '[DONE]') {
    eventSource.close();
    return;
  }
  const chunk = JSON.parse(event.data);
  console.log(chunk.chunk);
};
```

## 🤖 Agente Continuo

El sistema incluye un agente autónomo que corre indefinidamente hasta que el usuario lo pause o detenga.

### Crear y Usar Agente

```bash
# Crear un agente continuo (corre hasta que lo detengas)
curl -X POST http://localhost:8050/api/v1/agents \
  -H "Content-Type: application/json" \
  -d '{
    "name": "MiAgente",
    "system_prompt": "Eres un asistente experto que procesa tareas continuamente"
  }'

# Enviar tareas al agente
curl -X POST http://localhost:8050/api/v1/agents/{agent_id}/tasks \
  -H "Content-Type: application/json" \
  -d '{
    "description": "Analiza las tendencias de IA para 2025",
    "priority": 8
  }'

# Ver estado del agente
curl http://localhost:8050/api/v1/agents/{agent_id}

# Ver tareas del agente
curl http://localhost:8050/api/v1/agents/{agent_id}/tasks

# Pausar agente (deja de procesar nuevas tareas)
curl -X POST http://localhost:8050/api/v1/agents/{agent_id}/pause

# Reanudar agente
curl -X POST http://localhost:8050/api/v1/agents/{agent_id}/resume

# Detener agente
curl -X POST http://localhost:8050/api/v1/agents/{agent_id}/stop

# Detener todos los agentes
curl -X POST http://localhost:8050/api/v1/agents/stop-all
```

### Uso con JavaScript

```javascript
// Crear agente
const createResponse = await fetch('http://localhost:8050/api/v1/agents', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    name: 'MiAgente',
    loop_interval: 1.0
  })
});
const { agent_id } = await createResponse.json();

// Enviar tarea
await fetch(`http://localhost:8050/api/v1/agents/${agent_id}/tasks`, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    description: 'Genera un reporte de ventas',
    priority: 9
  })
});

// Verificar estado cada 5 segundos
const checkStatus = async () => {
  const res = await fetch(`http://localhost:8050/api/v1/agents/${agent_id}`);
  const status = await res.json();
  console.log('Estado:', status.status);
  console.log('Tareas completadas:', status.metrics.tasks_completed);
};

setInterval(checkStatus, 5000);

// Pausar cuando el usuario lo solicite
document.getElementById('pauseBtn').onclick = async () => {
  await fetch(`http://localhost:8050/api/v1/agents/${agent_id}/pause`, { method: 'POST' });
};

// Reanudar
document.getElementById('resumeBtn').onclick = async () => {
  await fetch(`http://localhost:8050/api/v1/agents/${agent_id}/resume`, { method: 'POST' });
};

// Detener completamente
document.getElementById('stopBtn').onclick = async () => {
  await fetch(`http://localhost:8050/api/v1/agents/${agent_id}/stop`, { method: 'POST' });
};
```

### Características del Agente

- **Ejecución continua 24/7**: El agente corre hasta que el usuario lo detenga
- **Cola de tareas con prioridad**: Heap-based priority queue (1-10, mayor = más urgente)
- **Procesamiento paralelo**: Worker pool para ejecución concurrente de tareas
- **Batch processing**: Procesa múltiples tareas en lotes optimizados
- **Modo idle inteligente**: Espera más tiempo cuando no hay tareas
- **Pausar/Reanudar**: Control total del usuario sobre la ejecución
- **Callbacks para eventos**: Notificaciones de tareas completadas/errores
- **Métricas en tiempo real**: Estadísticas de rendimiento y uso

### Procesamiento Paralelo

```bash
# Crear agente con configuración de paralelismo
curl -X POST http://localhost:8050/api/v1/agents \
  -H "Content-Type: application/json" \
  -d '{
    "name": "ParallelAgent",
    "enable_parallel": true,
    "num_workers": 5,
    "batch_size": 16,
    "max_concurrent_tasks": 10
  }'

# Enviar múltiples tareas en batch (más eficiente)
curl -X POST http://localhost:8050/api/v1/agents/{agent_id}/tasks/batch \
  -H "Content-Type: application/json" \
  -d '{
    "tasks": [
      {"description": "Tarea 1", "priority": 8},
      {"description": "Tarea 2", "priority": 5},
      {"description": "Tarea 3", "priority": 10}
    ]
  }'
```

### JavaScript - Batch Processing

```javascript
// Crear agente con paralelismo
const agent = await fetch('http://localhost:8050/api/v1/agents', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    name: 'ParallelAgent',
    enable_parallel: true,
    num_workers: 5,
    batch_size: 16
  })
}).then(r => r.json());

// Enviar batch de tareas
const batchResult = await fetch(`http://localhost:8050/api/v1/agents/${agent.agent_id}/tasks/batch`, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    tasks: [
      { description: 'Analiza tendencias de IA', priority: 9 },
      { description: 'Genera reporte de ventas', priority: 7 },
      { description: 'Revisa código del proyecto', priority: 5 }
    ]
  })
}).then(r => r.json());

console.log(`Submitted ${batchResult.submitted} tasks`);
```

## 📝 Notas

- El servidor no tiene autenticación por defecto. Para producción, agrega middleware de autenticación.
- La caché es en memoria. Para producción distribuida, considera usar Redis.
- Los logs se escriben a stdout. Configura un handler de archivos si es necesario.
- Los agentes continuos corren hasta que el usuario los detenga explícitamente.

## 📄 Licencia

Este proyecto es parte de Onyx.

## ☁️ AWS Deployment (EC2)

El proyecto incluye infraestructura para despliegue automático a EC2 cuando se hace push a `main`.

### Estructura de Archivos AWS

```
aws/
├── Dockerfile           # Container Docker para la API
├── docker-compose.yml   # Orquestación de servicios
├── deploy.sh           # Script de despliegue (corre en EC2)
├── setup_ec2.sh        # Setup inicial de EC2
├── nginx.conf          # Reverse proxy con SSL/CORS
├── env.template        # Template de variables de entorno
└── ssl/                # Certificados SSL
    └── README.md       # Instrucciones SSL
```

### 1. Setup Inicial de EC2

```bash
# En tu EC2 (Amazon Linux 2 o Ubuntu)
curl -fsSL https://raw.githubusercontent.com/OpenBlatam/onyx/main/backend/unified_ai_model/aws/setup_ec2.sh | bash
```

O manualmente:
```bash
# Clonar repositorio
git clone https://github.com/OpenBlatam/onyx.git
cd onyx/backend/unified_ai_model/aws

# Crear archivo .env
cp env.template .env
nano .env  # Editar con tus API keys

# Iniciar servicios
docker-compose up -d
```

### 2. Configurar GitHub Secrets

Para el CI/CD automático, configura estos secrets en GitHub:

| Secret | Descripción |
|--------|-------------|
| `EC2_HOST` | IP pública o hostname de tu EC2 |
| `EC2_USER` | Usuario SSH (ej: `ec2-user`) |
| `EC2_SSH_KEY` | Clave privada SSH |
| `DEEPSEEK_API_KEY` | API key de DeepSeek |

### 3. CI/CD Automático

Cada push a `main` que modifique `backend/unified_ai_model/` disparará el despliegue automático.

Ver workflow: `.github/workflows/deploy-unified-ai-model.yml`

### 4. Integración con Frontend

```javascript
const API_URL = process.env.NEXT_PUBLIC_UNIFIED_AI_URL || 'https://tu-ec2.com';

const response = await fetch(`${API_URL}/api/v1/chat`, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ message: 'Hola' })
});
```

### 5. Verificar Despliegue

```bash
curl http://your-ec2-ip:8050/api/v1/health
```



