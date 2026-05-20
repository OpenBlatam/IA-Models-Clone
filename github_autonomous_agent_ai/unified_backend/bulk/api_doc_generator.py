"""
Generador Automático de Documentación de API
Crea documentación completa basada en el código y endpoints
"""

import json
import requests
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

BASE_URL = "http://localhost:8000"

def fetch_openapi_schema() -> Dict[str, Any]:
    """Obtiene el schema OpenAPI."""
    try:
        response = requests.get(f"{BASE_URL}/api/openapi.json", timeout=10)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return {}

def generate_api_documentation(output_file: str = "API_DOCUMENTATION.md"):
    """Genera documentación completa de la API."""
    
    schema = fetch_openapi_schema()
    
    if not schema:
        print("⚠️ No se pudo obtener el schema OpenAPI, generando documentación básica")
    
    doc = f"""# 📚 Documentación Completa de la API BUL

**Versión:** {schema.get('info', {}).get('version', '1.0.0') if schema else '1.0.0'}  
**Generado:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Base URL:** `{BASE_URL}`

---

## 📋 Tabla de Contenidos

1. [Información General](#información-general)
2. [Endpoints del Sistema](#endpoints-del-sistema)
3. [Endpoints de Documentos](#endpoints-de-documentos)
4. [Endpoints de Tareas](#endpoints-de-tareas)
5. [WebSocket](#websocket)
6. [Modelos de Datos](#modelos-de-datos)
7. [Ejemplos de Uso](#ejemplos-de-uso)
8. [Códigos de Estado](#códigos-de-estado)
9. [Manejo de Errores](#manejo-de-errores)

---

## 🌐 Información General

### Descripción
API para generación de documentos con IA, lista para consumo desde frontend TypeScript.

### Características
- ✅ Generación de documentos con IA
- ✅ WebSocket para actualizaciones en tiempo real
- ✅ Rate limiting (10 req/min)
- ✅ Validaciones robustas
- ✅ CORS configurado
- ✅ Documentación OpenAPI

### Autenticación
Actualmente no se requiere autenticación. En producción, se recomienda implementar JWT.

---

## 🔧 Endpoints del Sistema

### `GET /`
Información básica del sistema.

**Respuesta:**
```json
{{
  "message": "BUL API - Frontend Ready",
  "version": "1.0.0",
  "status": "operational",
  "timestamp": "2024-01-15T10:30:00",
  "docs": "/api/docs",
  "health": "/api/health"
}}
```

### `GET /api/health`
Health check del sistema.

**Respuesta:**
```json
{{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00",
  "uptime": "2:30:15",
  "active_tasks": 3,
  "total_requests": 150,
  "version": "1.0.0"
}}
```

### `GET /api/stats`
Estadísticas del sistema.

**Respuesta:**
```json
{{
  "total_requests": 150,
  "active_tasks": 2,
  "completed_tasks": 140,
  "success_rate": 0.933,
  "average_processing_time": 4.5,
  "uptime": "2:30:15"
}}
```

---

## 📄 Endpoints de Documentos

### `POST /api/documents/generate`
Genera un nuevo documento.

**Request:**
```json
{{
  "query": "Crear un plan de marketing digital",
  "business_area": "marketing",
  "document_type": "strategy",
  "priority": 1,
  "metadata": {{"industry": "technology"}}
}}
```

**Campos:**
- `query` (string, requerido): Consulta de negocio (10-5000 caracteres)
- `business_area` (string, opcional): Área de negocio
- `document_type` (string, opcional): Tipo de documento
- `priority` (integer, opcional): Prioridad 1-5 (default: 1)
- `metadata` (object, opcional): Metadatos adicionales

**Respuesta:**
```json
{{
  "task_id": "task_20240115_103045_a1b2c3d4",
  "status": "queued",
  "message": "Generación de documento iniciada",
  "estimated_time": 60,
  "queue_position": 1,
  "created_at": "2024-01-15T10:30:45"
}}
```

**Rate Limit:** 10 solicitudes por minuto por IP

### `GET /api/tasks/{{task_id}}/document`
Obtiene el documento generado.

**Respuesta:**
```json
{{
  "task_id": "task_20240115_103045_a1b2c3d4",
  "document": {{
    "content": "# Documento Generado\\n\\n...",
    "format": "markdown",
    "word_count": 1250,
    "generated_at": "2024-01-15T10:31:30"
  }},
  "metadata": {{...}},
  "created_at": "2024-01-15T10:30:45",
  "completed_at": "2024-01-15T10:31:30"
}}
```

### `GET /api/documents`
Lista documentos generados.

**Query Parameters:**
- `limit` (integer, default: 50): Límite de resultados (1-100)
- `offset` (integer, default: 0): Offset para paginación

**Respuesta:**
```json
{{
  "documents": [...],
  "total": 20,
  "limit": 50,
  "offset": 0,
  "has_more": false
}}
```

---

## 📋 Endpoints de Tareas

### `GET /api/tasks/{{task_id}}/status`
Obtiene el estado de una tarea.

**Respuesta:**
```json
{{
  "task_id": "task_20240115_103045_a1b2c3d4",
  "status": "completed",
  "progress": 100,
  "result": {{...}},
  "error": null,
  "created_at": "2024-01-15T10:30:45",
  "updated_at": "2024-01-15T10:31:30",
  "processing_time": 45.2
}}
```

**Estados posibles:**
- `queued` - En cola
- `processing` - Procesando
- `completed` - Completada
- `failed` - Fallida
- `cancelled` - Cancelada

### `GET /api/tasks`
Lista tareas.

**Query Parameters:**
- `status` (string, opcional): Filtrar por estado
- `user_id` (string, opcional): Filtrar por usuario
- `limit` (integer, default: 50): Límite de resultados
- `offset` (integer, default: 0): Offset para paginación

### `DELETE /api/tasks/{{task_id}}`
Elimina una tarea.

### `POST /api/tasks/{{task_id}}/cancel`
Cancela una tarea en ejecución.

---

## 🔌 WebSocket

### `WS /api/ws/{{task_id}}`
WebSocket para actualizaciones de una tarea específica.

**Mensajes recibidos:**
```json
{{
  "type": "task_update",
  "task_id": "task_123",
  "data": {{
    "status": "processing",
    "progress": 50
  }},
  "timestamp": "2024-01-15T10:31:00"
}}
```

**Tipos de mensajes:**
- `initial_state` - Estado inicial
- `task_update` - Actualización de progreso
- `error` - Error ocurrido
- `pong` - Respuesta a ping

### `WS /api/ws`
WebSocket para todas las actualizaciones.

---

## 📊 Modelos de Datos

### DocumentRequest
```typescript
{{
  query: string;              // 10-5000 caracteres
  business_area?: string;
  document_type?: string;
  priority?: number;          // 1-5
  metadata?: Record<string, any>;
  user_id?: string;
  session_id?: string;
}}
```

### DocumentResponse
```typescript
{{
  task_id: string;
  status: 'queued' | 'processing' | 'completed' | 'failed' | 'cancelled';
  message: string;
  estimated_time?: number;
  queue_position?: number;
  created_at: string;
}}
```

### TaskStatus
```typescript
{{
  task_id: string;
  status: string;
  progress: number;           // 0-100
  result?: {{
    content: string;
    format: string;
    word_count: number;
    generated_at: string;
  }};
  error?: string;
  created_at: string;
  updated_at: string;
  processing_time?: number;
}}
```

---

## 💡 Ejemplos de Uso

### Ejemplo 1: Generar Documento Básico

```typescript
import {{ createBULApiClient }} from './api/bul-api-client';

const client = createBULApiClient({{
  baseUrl: 'http://localhost:8000'
}});

const response = await client.generateDocument({{
  query: 'Crear un plan de marketing digital',
  priority: 1
}});

console.log('Task ID:', response.task_id);
```

### Ejemplo 2: Generar y Esperar con WebSocket

```typescript
const document = await client.generateDocumentAndWait(
  {{
    query: 'Estrategia de ventas B2B',
    business_area: 'sales',
    document_type: 'strategy'
  }},
  {{
    onProgress: (status) => {{
      console.log(`Progreso: ${{status.progress}}%`);
    }}
  }}
);

console.log('Documento:', document.document.content);
```

### Ejemplo 3: Usar WebSocket Manualmente

```typescript
const ws = await client.connectTaskWebSocket(taskId, (message) => {{
  if (message.type === 'task_update') {{
    console.log('Actualización:', message.data);
  }}
}});
```

### Ejemplo 4: Listar y Filtrar Tareas

```typescript
// Listar tareas completadas
const tasks = await client.listTasks({{
  status: 'completed',
  limit: 10,
  offset: 0
}});

// Listar documentos
const documents = await client.listDocuments(10, 0);
```

---

## 🔢 Códigos de Estado

- `200 OK` - Solicitud exitosa
- `400 Bad Request` - Solicitud inválida
- `404 Not Found` - Recurso no encontrado
- `429 Too Many Requests` - Rate limit excedido
- `500 Internal Server Error` - Error del servidor

---

## ⚠️ Manejo de Errores

### Errores Comunes

**Query muy corta:**
```json
{{
  "detail": "La consulta debe tener al menos 10 caracteres"
}}
```

**Query muy larga:**
```json
{{
  "detail": "La consulta no puede exceder 5000 caracteres"
}}
```

**Rate limit excedido:**
```json
{{
  "detail": "Rate limit exceeded"
}}
```

**Tarea no encontrada:**
```json
{{
  "detail": "Tarea no encontrada"
}}
```

---

## 🔗 Enlaces Útiles

- **Swagger UI:** http://localhost:8000/api/docs
- **ReDoc:** http://localhost:8000/api/redoc
- **OpenAPI JSON:** http://localhost:8000/api/openapi.json

---

## 📝 Notas

- Todos los timestamps están en formato ISO 8601
- Los documentos se generan en formato Markdown
- El tiempo promedio de procesamiento es 30-60 segundos
- La API soporta CORS desde cualquier origen (configurar en producción)

---

**Documentación generada automáticamente**  
**Última actualización:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(doc)
    
    print(f"✅ Documentación generada: {output_file}")
    return output_file

if __name__ == "__main__":
    generate_api_documentation()
































