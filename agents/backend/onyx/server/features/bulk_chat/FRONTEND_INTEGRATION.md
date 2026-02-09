# Integración Frontend - Bulk Chat API

## 🚀 Guía de Integración para Frontend

Esta API está **100% lista** para usar desde el frontend. Incluye CORS habilitado y endpoints REST + WebSocket.

---

## 📡 Endpoints Principales

### Base URL
```
http://localhost:8006
```

### 1. Crear Sesión de Chat

```typescript
// POST /api/v1/chat/sessions
const createSession = async (initialMessage?: string) => {
  const response = await fetch('http://localhost:8006/api/v1/chat/sessions', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      initial_message: initialMessage || 'Hola',
      auto_continue: true,
      response_interval: 2.0
    })
  });
  
  return await response.json();
  // Response: { session_id, state, is_paused, message_count, auto_continue }
};
```

### 2. Enviar Mensaje

```typescript
// POST /api/v1/chat/sessions/{session_id}/messages
const sendMessage = async (sessionId: string, message: string) => {
  const response = await fetch(
    `http://localhost:8006/api/v1/chat/sessions/${sessionId}/messages`,
    {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        message: message
      })
    }
  );
  
  return await response.json();
};
```

### 3. Obtener Mensajes

```typescript
// GET /api/v1/chat/sessions/{session_id}/messages
const getMessages = async (sessionId: string, limit: number = 50) => {
  const response = await fetch(
    `http://localhost:8006/api/v1/chat/sessions/${sessionId}/messages?limit=${limit}`
  );
  
  const data = await response.json();
  return data.messages; // Array de mensajes
};
```

### 4. Control de Sesión

```typescript
// Pausar
const pauseSession = async (sessionId: string, reason?: string) => {
  await fetch(
    `http://localhost:8006/api/v1/chat/sessions/${sessionId}/pause?reason=${reason || ''}`,
    { method: 'POST' }
  );
};

// Reanudar
const resumeSession = async (sessionId: string) => {
  await fetch(
    `http://localhost:8006/api/v1/chat/sessions/${sessionId}/resume`,
    { method: 'POST' }
  );
};

// Detener
const stopSession = async (sessionId: string) => {
  await fetch(
    `http://localhost:8006/api/v1/chat/sessions/${sessionId}/stop`,
    { method: 'POST' }
  );
};
```

### 5. Obtener Estado de Sesión

```typescript
// GET /api/v1/chat/sessions/{session_id}
const getSession = async (sessionId: string) => {
  const response = await fetch(
    `http://localhost:8006/api/v1/chat/sessions/${sessionId}`
  );
  return await response.json();
};
```

---

## 🔌 WebSocket (Tiempo Real)

### Conexión WebSocket

```typescript
const connectWebSocket = (sessionId: string) => {
  const ws = new WebSocket(`ws://localhost:8006/ws/chat/${sessionId}`);
  
  ws.onopen = () => {
    console.log('WebSocket conectado');
  };
  
  ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    
    switch (data.type) {
      case 'chunk':
        // Chunk de respuesta en streaming
        console.log('Chunk:', data.data);
        break;
        
      case 'message':
        // Mensaje completo
        console.log('Mensaje:', data.content);
        break;
        
      case 'session_state':
        // Estado de sesión actualizado
        console.log('Estado:', data.data);
        break;
        
      case 'paused':
        console.log('Sesión pausada');
        break;
        
      case 'resumed':
        console.log('Sesión reanudada');
        break;
        
      case 'error':
        console.error('Error:', data.message);
        break;
    }
  };
  
  ws.onerror = (error) => {
    console.error('WebSocket error:', error);
  };
  
  ws.onclose = () => {
    console.log('WebSocket desconectado');
  };
  
  // Enviar mensaje
  const sendMessage = (content: string) => {
    ws.send(JSON.stringify({
      type: 'message',
      content: content
    }));
  };
  
  // Pausar
  const pause = () => {
    ws.send(JSON.stringify({ type: 'pause' }));
  };
  
  // Reanudar
  const resume = () => {
    ws.send(JSON.stringify({ type: 'resume' }));
  };
  
  // Detener
  const stop = () => {
    ws.send(JSON.stringify({ type: 'stop' }));
  };
  
  return { sendMessage, pause, resume, stop };
};
```

---

## ⚛️ Ejemplo Completo con React

```typescript
import React, { useState, useEffect, useRef } from 'react';

const API_BASE = 'http://localhost:8006';

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: string;
}

export const ChatInterface: React.FC = () => {
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isConnected, setIsConnected] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);

  // Crear sesión al montar
  useEffect(() => {
    const createSession = async () => {
      try {
        const response = await fetch(`${API_BASE}/api/v1/chat/sessions`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            initial_message: 'Hola',
            auto_continue: true
          })
        });
        
        const data = await response.json();
        setSessionId(data.session_id);
        
        // Conectar WebSocket
        connectWebSocket(data.session_id);
        
        // Cargar mensajes iniciales
        loadMessages(data.session_id);
      } catch (error) {
        console.error('Error creando sesión:', error);
      }
    };
    
    createSession();
  }, []);

  // Conectar WebSocket
  const connectWebSocket = (sid: string) => {
    const ws = new WebSocket(`ws://localhost:8006/ws/chat/${sid}`);
    
    ws.onopen = () => {
      setIsConnected(true);
      console.log('WebSocket conectado');
    };
    
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      
      if (data.type === 'message' && data.role === 'assistant') {
        setMessages(prev => [...prev, {
          id: Date.now().toString(),
          role: 'assistant',
          content: data.content,
          timestamp: data.timestamp
        }]);
      } else if (data.type === 'chunk') {
        // Actualizar último mensaje con chunk
        setMessages(prev => {
          const updated = [...prev];
          const lastMsg = updated[updated.length - 1];
          if (lastMsg && lastMsg.role === 'assistant') {
            lastMsg.content += data.data;
          } else {
            updated.push({
              id: Date.now().toString(),
              role: 'assistant',
              content: data.data,
              timestamp: new Date().toISOString()
            });
          }
          return updated;
        });
      }
    };
    
    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
      setIsConnected(false);
    };
    
    ws.onclose = () => {
      setIsConnected(false);
      console.log('WebSocket desconectado');
    };
    
    wsRef.current = ws;
  };

  // Cargar mensajes
  const loadMessages = async (sid: string) => {
    try {
      const response = await fetch(
        `${API_BASE}/api/v1/chat/sessions/${sid}/messages`
      );
      const data = await response.json();
      setMessages(data.messages);
    } catch (error) {
      console.error('Error cargando mensajes:', error);
    }
  };

  // Enviar mensaje
  const sendMessage = async () => {
    if (!input.trim() || !sessionId) return;
    
    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: input,
      timestamp: new Date().toISOString()
    };
    
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    
    // Enviar por WebSocket
    if (wsRef.current && isConnected) {
      wsRef.current.send(JSON.stringify({
        type: 'message',
        content: userMessage.content
      }));
    } else {
      // Fallback a REST
      try {
        await fetch(
          `${API_BASE}/api/v1/chat/sessions/${sessionId}/messages`,
          {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: userMessage.content })
          }
        );
        
        // Recargar mensajes después de un momento
        setTimeout(() => loadMessages(sessionId), 1000);
      } catch (error) {
        console.error('Error enviando mensaje:', error);
      }
    }
  };

  // Pausar
  const pause = () => {
    if (wsRef.current && isConnected) {
      wsRef.current.send(JSON.stringify({ type: 'pause' }));
    } else if (sessionId) {
      fetch(`${API_BASE}/api/v1/chat/sessions/${sessionId}/pause`, {
        method: 'POST'
      });
    }
  };

  // Reanudar
  const resume = () => {
    if (wsRef.current && isConnected) {
      wsRef.current.send(JSON.stringify({ type: 'resume' }));
    } else if (sessionId) {
      fetch(`${API_BASE}/api/v1/chat/sessions/${sessionId}/resume`, {
        method: 'POST'
      });
    }
  };

  return (
    <div className="chat-container">
      <div className="chat-header">
        <h2>Chat</h2>
        <div className="status">
          {isConnected ? '🟢 Conectado' : '🔴 Desconectado'}
        </div>
      </div>
      
      <div className="messages">
        {messages.map(msg => (
          <div key={msg.id} className={`message ${msg.role}`}>
            <strong>{msg.role === 'user' ? 'Tú' : 'Asistente'}:</strong>
            <p>{msg.content}</p>
          </div>
        ))}
      </div>
      
      <div className="input-area">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyPress={(e) => e.key === 'Enter' && sendMessage()}
          placeholder="Escribe un mensaje..."
        />
        <button onClick={sendMessage}>Enviar</button>
        <button onClick={pause}>Pausar</button>
        <button onClick={resume}>Reanudar</button>
      </div>
    </div>
  );
};
```

---

## 🎨 Ejemplo con Fetch API (Vanilla JS)

```javascript
class BulkChatClient {
  constructor(baseUrl = 'http://localhost:8006') {
    this.baseUrl = baseUrl;
    this.sessionId = null;
    this.ws = null;
  }

  async createSession(initialMessage = 'Hola', autoContinue = true) {
    const response = await fetch(`${this.baseUrl}/api/v1/chat/sessions`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        initial_message: initialMessage,
        auto_continue: autoContinue
      })
    });
    
    const data = await response.json();
    this.sessionId = data.session_id;
    return data;
  }

  async sendMessage(message) {
    if (!this.sessionId) throw new Error('No session created');
    
    return fetch(
      `${this.baseUrl}/api/v1/chat/sessions/${this.sessionId}/messages`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message })
      }
    );
  }

  async getMessages(limit = 50) {
    if (!this.sessionId) throw new Error('No session created');
    
    const response = await fetch(
      `${this.baseUrl}/api/v1/chat/sessions/${this.sessionId}/messages?limit=${limit}`
    );
    return await response.json();
  }

  async pause(reason = '') {
    if (!this.sessionId) throw new Error('No session created');
    return fetch(
      `${this.baseUrl}/api/v1/chat/sessions/${this.sessionId}/pause?reason=${reason}`,
      { method: 'POST' }
    );
  }

  async resume() {
    if (!this.sessionId) throw new Error('No session created');
    return fetch(
      `${this.baseUrl}/api/v1/chat/sessions/${this.sessionId}/resume`,
      { method: 'POST' }
    );
  }

  async stop() {
    if (!this.sessionId) throw new Error('No session created');
    return fetch(
      `${this.baseUrl}/api/v1/chat/sessions/${this.sessionId}/stop`,
      { method: 'POST' }
    );
  }

  connectWebSocket(onMessage, onError) {
    if (!this.sessionId) throw new Error('No session created');
    
    this.ws = new WebSocket(`ws://localhost:8006/ws/chat/${this.sessionId}`);
    
    this.ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      onMessage(data);
    };
    
    this.ws.onerror = onError;
    
    this.ws.onopen = () => console.log('WebSocket conectado');
    this.ws.onclose = () => console.log('WebSocket desconectado');
    
    return this.ws;
  }

  sendWebSocketMessage(type, content) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ type, content }));
    }
  }
}

// Uso
const client = new BulkChatClient();
await client.createSession('Hola, ¿cómo estás?');
client.connectWebSocket(
  (data) => console.log('Mensaje recibido:', data),
  (error) => console.error('Error:', error)
);
```

---

## 🔒 CORS

La API tiene CORS habilitado por defecto. Si necesitas configurar dominios específicos, revisa la configuración en `chat_api.py`.

---

## 📊 Health Check

```typescript
const checkHealth = async () => {
  const response = await fetch('http://localhost:8006/health');
  const data = await response.json();
  console.log('API Status:', data);
  // Response: { status: "healthy", service: "bulk_chat", active_sessions: 0 }
};
```

---

## 🎯 Endpoints Adicionales Útiles

### Métricas
```typescript
// GET /api/v1/chat/metrics
const getMetrics = async () => {
  const response = await fetch('http://localhost:8006/api/v1/chat/metrics');
  return await response.json();
};
```

### Exportar Sesión
```typescript
// GET /api/v1/chat/sessions/{session_id}/export/{format}
const exportSession = async (sessionId: string, format: 'json' | 'txt' = 'json') => {
  const response = await fetch(
    `http://localhost:8006/api/v1/chat/sessions/${sessionId}/export/${format}`
  );
  return await response.json();
};
```

### Análisis de Sesión
```typescript
// GET /api/v1/chat/sessions/{session_id}/analyze
const analyzeSession = async (sessionId: string) => {
  const response = await fetch(
    `http://localhost:8006/api/v1/chat/sessions/${sessionId}/analyze`
  );
  return await response.json();
};
```

---

## ✅ Checklist de Integración

- [x] API REST lista
- [x] WebSocket funcionando
- [x] CORS habilitado
- [x] Documentación en `/docs`
- [x] Health check disponible
- [x] Endpoints de control (pause/resume/stop)
- [x] Streaming de mensajes
- [x] Manejo de errores

---

## 🚀 Iniciar el Servidor

```bash
# Desde el directorio bulk_chat
python -m bulk_chat.main --port 8006

# O con script
python start.py
```

Una vez iniciado, la API estará disponible en `http://localhost:8006`

---

## 📚 Documentación Completa

- Swagger UI: `http://localhost:8006/docs`
- ReDoc: `http://localhost:8006/redoc`
- Dashboard: `http://localhost:8006/dashboard`

---

¡La API está lista para usar! 🎉











