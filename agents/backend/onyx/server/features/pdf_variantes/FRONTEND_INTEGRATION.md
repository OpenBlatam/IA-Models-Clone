# Frontend Integration Guide

## API Lista para Frontend

La API está completamente configurada y lista para integrarse con cualquier frontend (React, Vue, Angular, etc.).

## Características Implementadas

### ✅ CORS Configurado
- Permite conexiones desde los puertos de desarrollo más comunes
- Configuración flexible según entorno (desarrollo/producción)
- Headers expuestos para mejor integración

### ✅ Autenticación Flexible
- **Modo Desarrollo**: Permite acceso sin autenticación (para desarrollo rápido)
- **Modo Producción**: Requiere JWT Bearer tokens
- Headers personalizados para identificación de usuario

### ✅ Respuestas Frontend-Friendly
- Formato estándar de respuestas con `success`, `data`, `error`
- Headers informativos (`X-Request-ID`, `X-Response-Time`)
- Errores detallados en modo desarrollo

### ✅ WebSocket Support
- Conexiones WebSocket para colaboración en tiempo real
- Configuración flexible de CORS para WebSockets

### ✅ Rate Limiting
- Límites configurables por IP
- Prevención de abuso

## Endpoints Principales

### Health Check
```
GET /health
GET /api/v1/health/status
```

### Documentos PDF
```
POST   /api/v1/pdf/upload          # Subir PDF
GET    /api/v1/pdf/documents        # Listar documentos
GET    /api/v1/pdf/documents/{id}    # Obtener documento
DELETE /api/v1/pdf/documents/{id}    # Eliminar documento
```

### Variantes
```
POST /api/v1/variants/generate                # Generar variantes
GET  /api/v1/variants/documents/{id}/variants  # Listar variantes
GET  /api/v1/variants/{variant_id}             # Obtener variante
POST /api/v1/variants/stop                     # Detener generación
```

### Topics
```
POST /api/v1/topics/extract                    # Extraer topics
GET  /api/v1/topics/documents/{id}/topics      # Listar topics
```

### Brainstorming
```
POST /api/v1/brainstorm/generate              # Generar ideas
GET  /api/v1/brainstorm/documents/{id}/ideas   # Listar ideas
```

### Export
```
POST /api/v1/export/export              # Exportar contenido
GET  /api/v1/export/download/{file_id}  # Descargar archivo
```

### WebSocket
```
WS /api/v1/collaboration/ws/{document_id}?user_id={user_id}
```

## Ejemplo de Uso con TypeScript/React

### Configuración del Cliente

```typescript
// api/client.ts
const API_BASE_URL = 'http://localhost:8000';

interface ApiResponse<T> {
  success: boolean;
  data?: T;
  error?: {
    message: string;
    status_code: number;
    type: string;
  };
}

async function apiRequest<T>(
  endpoint: string,
  options: RequestInit = {}
): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${endpoint}`, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      'X-User-ID': 'your-user-id', // En desarrollo
      // En producción: 'Authorization': 'Bearer YOUR_TOKEN'
      ...options.headers,
    },
  });

  const data: ApiResponse<T> = await response.json();

  if (!response.ok || !data.success) {
    throw new Error(data.error?.message || 'Request failed');
  }

  return data.data!;
}

// Subir PDF
export async function uploadPDF(file: File) {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('auto_process', 'true');

  const response = await fetch(`${API_BASE_URL}/api/v1/pdf/upload`, {
    method: 'POST',
    body: formData,
    headers: {
      'X-User-ID': 'your-user-id',
    },
  });

  const data = await response.json();
  return data;
}

// Listar documentos
export async function getDocuments() {
  return apiRequest<any[]>('/api/v1/pdf/documents');
}
```

### Ejemplo de Componente React

```typescript
// components/DocumentUpload.tsx
import { useState } from 'react';
import { uploadPDF } from '../api/client';

export function DocumentUpload() {
  const [uploading, setUploading] = useState(false);

  async function handleUpload(file: File) {
    setUploading(true);
    try {
      const result = await uploadPDF(file);
      console.log('Upload success:', result);
    } catch (error) {
      console.error('Upload error:', error);
    } finally {
      setUploading(false);
    }
  }

  return (
    <input
      type="file"
      accept=".pdf"
      onChange={(e) => {
        const file = e.target.files?.[0];
        if (file) handleUpload(file);
      }}
      disabled={uploading}
    />
  );
}
```

## Variables de Entorno

Para desarrollo local, crea un archivo `.env`:

```env
ENVIRONMENT=development
DEBUG=true
SECRET_KEY=your-secret-key-here
DATABASE_URL=postgresql://user:pass@localhost/db
REDIS_URL=redis://localhost:6379/0
```

## Modo Desarrollo vs Producción

### Desarrollo
- Autenticación opcional (usa header `X-User-ID`)
- CORS abierto
- Errores detallados con stack traces
- Logging verboso

### Producción
- Autenticación requerida (JWT Bearer tokens)
- CORS configurado según `CORS_ORIGINS`
- Errores genéricos (sin detalles internos)
- Logging optimizado

## Headers Importantes

- `X-User-ID`: Identificador de usuario (desarrollo)
- `Authorization`: Bearer token (producción)
- `X-Request-ID`: ID de la request (para tracking)
- `X-Response-Time`: Tiempo de procesamiento

## Respuestas de Error

Todas las respuestas de error siguen este formato:

```json
{
  "error": {
    "message": "Error description",
    "status_code": 400,
    "type": "HTTPException"
  },
  "success": false,
  "data": null
}
```

## Respuestas Exitosas

```json
{
  "success": true,
  "data": {
    // ... datos de la respuesta
  },
  "error": null
}
```

## WebSocket Example

```typescript
const ws = new WebSocket(
  'ws://localhost:8000/api/v1/collaboration/ws/document123?user_id=user456'
);

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('WebSocket message:', data);
};

ws.send(JSON.stringify({
  type: 'annotation',
  data: { /* ... */ }
}));
```

## Documentación Interactiva

Una vez que el servidor esté corriendo, accede a:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Testing

Prueba rápida con curl:

```bash
# Health check
curl http://localhost:8000/health

# Listar documentos (desarrollo)
curl -H "X-User-ID: test-user" http://localhost:8000/api/v1/pdf/documents

# Con autenticación (producción)
curl -H "Authorization: Bearer YOUR_TOKEN" http://localhost:8000/api/v1/pdf/documents
```

## Notas Importantes

1. En desarrollo, puedes omitir la autenticación usando el header `X-User-ID`
2. Todos los endpoints retornan JSON consistente
3. Los IDs de request permiten tracking de requests en los logs
4. Los errores incluyen información útil para debugging en desarrollo






