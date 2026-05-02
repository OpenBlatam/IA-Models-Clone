# Guía de Integración Frontend-Backend

## 📋 Resumen

Esta guía explica cómo integrar el frontend (Next.js/React) con el backend (FastAPI) del GitHub Autonomous Agent.

## 🚀 Cliente API

### Instalación

El cliente API está disponible en `api/client.py` y puede ser usado desde Python o adaptado para TypeScript/JavaScript.

### Uso en Python

```python
from api.client import APIClient

async def main():
    async with APIClient(base_url="http://localhost:8030") as client:
        # Obtener estado del agente
        status = await client.get_agent_status()
        print(f"Agent running: {status['is_running']}")
        
        # Crear tarea
        task = await client.create_task(
            repository_owner="owner",
            repository_name="repo",
            instruction="create file: test.py"
        )
        print(f"Task created: {task['id']}")
        
        # Listar tareas
        tasks = await client.list_tasks(limit=10)
        print(f"Total tasks: {tasks['total']}")

# Ejecutar
import asyncio
asyncio.run(main())
```

### Adaptación para TypeScript/JavaScript

```typescript
// lib/api-client.ts
class APIClient {
  private baseURL: string;
  
  constructor(baseURL: string = 'http://localhost:8030') {
    this.baseURL = baseURL;
  }
  
  private async request(
    method: string,
    endpoint: string,
    data?: any
  ): Promise<any> {
    const response = await fetch(`${this.baseURL}${endpoint}`, {
      method,
      headers: {
        'Content-Type': 'application/json',
      },
      body: data ? JSON.stringify(data) : undefined,
    });
    
    if (!response.ok) {
      throw new Error(`API error: ${response.statusText}`);
    }
    
    return response.json();
  }
  
  async getAgentStatus() {
    return this.request('GET', '/api/v1/agent/status');
  }
  
  async createTask(
    repositoryOwner: string,
    repositoryName: string,
    instruction: string,
    metadata?: Record<string, any>
  ) {
    return this.request('POST', '/api/v1/tasks/', {
      repository_owner: repositoryOwner,
      repository_name: repositoryName,
      instruction,
      metadata: metadata || {},
    });
  }
  
  async listTasks(params?: {
    status?: string;
    repository?: string;
    limit?: number;
    offset?: number;
  }) {
    const query = new URLSearchParams(params as any).toString();
    return this.request('GET', `/api/v1/tasks/?${query}`);
  }
  
  async generateLLMResponse(
    prompt: string,
    model?: string,
    systemPrompt?: string
  ) {
    return this.request('POST', '/api/v1/llm/generate', {
      prompt,
      model,
      system_prompt: systemPrompt,
    });
  }
}

export const apiClient = new APIClient();
```

## 🔌 WebSocket para Actualizaciones en Tiempo Real

### Endpoints WebSocket

- `ws://localhost:8030/ws` - WebSocket general
- `ws://localhost:8030/ws/tasks` - Solo actualizaciones de tareas
- `ws://localhost:8030/ws/agent` - Solo actualizaciones del agente

### Uso en TypeScript/React

```typescript
// hooks/useWebSocket.ts
import { useEffect, useState, useRef } from 'react';

export function useWebSocket(url: string) {
  const [messages, setMessages] = useState<any[]>([]);
  const [connected, setConnected] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);
  
  useEffect(() => {
    const ws = new WebSocket(url);
    wsRef.current = ws;
    
    ws.onopen = () => {
      setConnected(true);
      // Suscribirse a actualizaciones de tareas
      ws.send(JSON.stringify({
        type: 'subscribe',
        data: { type: 'tasks' }
      }));
    };
    
    ws.onmessage = (event) => {
      const message = JSON.parse(event.data);
      setMessages(prev => [...prev, message]);
    };
    
    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
    };
    
    ws.onclose = () => {
      setConnected(false);
    };
    
    // Heartbeat
    const heartbeat = setInterval(() => {
      if (ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: 'ping' }));
      }
    }, 30000);
    
    return () => {
      clearInterval(heartbeat);
      ws.close();
    };
  }, [url]);
  
  return { messages, connected };
}

// Uso en componente
function TaskList() {
  const { messages, connected } = useWebSocket('ws://localhost:8030/ws/tasks');
  const [tasks, setTasks] = useState([]);
  
  useEffect(() => {
    // Procesar mensajes de actualización
    messages.forEach(msg => {
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
    });
  }, [messages]);
  
  return (
    <div>
      {connected ? '🟢 Conectado' : '🔴 Desconectado'}
      {/* Renderizar tareas */}
    </div>
  );
}
```

## 📝 Tipos TypeScript

Los tipos están definidos en `api/types.py` y pueden ser convertidos a TypeScript:

```typescript
// types/api.ts
export interface Task {
  id: string;
  repository_owner: string;
  repository_name: string;
  instruction: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  created_at: string;
  updated_at: string;
  metadata?: Record<string, any>;
  result?: Record<string, any>;
  error?: string;
}

export interface AgentStatus {
  is_running: boolean;
  current_task_id?: string;
  last_activity?: string;
  metadata: Record<string, any>;
}

export interface LLMResponse {
  model: string;
  content: string;
  usage?: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
  finish_reason?: string;
  error?: string;
  latency_ms?: number;
}

export interface WebSocketMessage {
  type: string;
  data: any;
  timestamp: string;
}
```

## 🔄 Flujo de Integración

### 1. Inicialización

```typescript
// app/layout.tsx o _app.tsx
import { apiClient } from '@/lib/api-client';

// Verificar conexión al iniciar
useEffect(() => {
  apiClient.healthCheck()
    .then(health => {
      console.log('Backend status:', health.status);
    })
    .catch(err => {
      console.error('Backend not available:', err);
    });
}, []);
```

### 2. Crear Tarea

```typescript
async function handleCreateTask() {
  try {
    const task = await apiClient.createTask(
      selectedRepo.owner,
      selectedRepo.name,
      instruction
    );
    
    // Actualizar UI
    setTasks(prev => [...prev, task]);
    
    // Notificación
    toast.success(`Tarea ${task.id} creada`);
  } catch (error) {
    toast.error('Error al crear tarea');
  }
}
```

### 3. Monitorear Tareas

```typescript
function TaskMonitor() {
  const { messages } = useWebSocket('ws://localhost:8030/ws/tasks');
  const [tasks, setTasks] = useState<Task[]>([]);
  
  // Cargar tareas iniciales
  useEffect(() => {
    apiClient.listTasks().then(data => setTasks(data.tasks));
  }, []);
  
  // Actualizar con WebSocket
  useEffect(() => {
    messages.forEach(msg => {
      if (msg.type === 'task_update') {
        setTasks(prev => updateTask(prev, msg.data));
      }
    });
  }, [messages]);
  
  return (
    <div>
      {tasks.map(task => (
        <TaskCard key={task.id} task={task} />
      ))}
    </div>
  );
}
```

### 4. Control del Agente

```typescript
async function handleStartAgent() {
  try {
    await apiClient.startAgent();
    toast.success('Agente iniciado');
  } catch (error) {
    toast.error('Error al iniciar agente');
  }
}

async function handleStopAgent() {
  try {
    await apiClient.stopAgent();
    toast.success('Agente detenido');
  } catch (error) {
    toast.error('Error al detener agente');
  }
}
```

## 🔐 Autenticación

Si necesitas autenticación, puedes agregar tokens:

```typescript
class APIClient {
  private token?: string;
  
  setToken(token: string) {
    this.token = token;
  }
  
  private async request(method: string, endpoint: string, data?: any) {
    const headers: HeadersInit = {
      'Content-Type': 'application/json',
    };
    
    if (this.token) {
      headers['Authorization'] = `Bearer ${this.token}`;
    }
    
    // ... resto del código
  }
}
```

## 📊 Ejemplo Completo

```typescript
// components/GitHubAgent.tsx
'use client';

import { useState, useEffect } from 'react';
import { apiClient } from '@/lib/api-client';
import { useWebSocket } from '@/hooks/useWebSocket';
import type { Task, AgentStatus } from '@/types/api';

export default function GitHubAgent() {
  const [agentStatus, setAgentStatus] = useState<AgentStatus | null>(null);
  const [tasks, setTasks] = useState<Task[]>([]);
  const { messages, connected } = useWebSocket('ws://localhost:8030/ws');
  
  // Cargar estado inicial
  useEffect(() => {
    loadInitialData();
  }, []);
  
  // Procesar mensajes WebSocket
  useEffect(() => {
    messages.forEach(msg => {
      if (msg.type === 'agent_status') {
        setAgentStatus(msg.data);
      } else if (msg.type === 'task_update') {
        setTasks(prev => updateTask(prev, msg.data));
      }
    });
  }, [messages]);
  
  async function loadInitialData() {
    try {
      const [status, tasksData] = await Promise.all([
        apiClient.getAgentStatus(),
        apiClient.listTasks()
      ]);
      setAgentStatus(status);
      setTasks(tasksData.tasks);
    } catch (error) {
      console.error('Error loading data:', error);
    }
  }
  
  async function handleCreateTask() {
    // ... crear tarea
  }
  
  return (
    <div>
      <div className="status">
        {connected ? '🟢 Conectado' : '🔴 Desconectado'}
        {agentStatus?.is_running ? '🟢 Agente Activo' : '🔴 Agente Inactivo'}
      </div>
      
      <button onClick={() => apiClient.startAgent()}>
        Iniciar Agente
      </button>
      
      <TaskList tasks={tasks} />
    </div>
  );
}
```

## 🛠️ Configuración

### Variables de Entorno

```env
# Frontend (.env.local)
NEXT_PUBLIC_API_URL=http://localhost:8030
NEXT_PUBLIC_WS_URL=ws://localhost:8030

# Backend (.env)
CORS_ORIGINS=http://localhost:3000,http://localhost:3001
```

### CORS

El backend ya está configurado para aceptar requests del frontend. Asegúrate de que `CORS_ORIGINS` incluya la URL de tu frontend.

## 📚 Referencias

- [FastAPI WebSockets](https://fastapi.tiangolo.com/advanced/websockets/)
- [React WebSocket Hooks](https://github.com/robtaussig/react-use-websocket)
- [TypeScript API Client Patterns](https://www.typescriptlang.org/docs/handbook/declaration-files/templates/module-d-ts.html)

---

**Versión**: 1.0  
**Última actualización**: Enero 2025



