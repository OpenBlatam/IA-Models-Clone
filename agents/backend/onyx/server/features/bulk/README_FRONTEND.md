# BUL API - Frontend Ready

API lista para consumo desde frontend TypeScript con todas las configuraciones necesarias.

## 🚀 Inicio Rápido

### 1. Iniciar el servidor API

```bash
# Desde el directorio bulk
python api_frontend_ready.py --host 0.0.0.0 --port 8000
```

O con opciones adicionales:

```bash
python api_frontend_ready.py --host 0.0.0.0 --port 8000 --reload
```

El servidor estará disponible en:
- API: `http://localhost:8000`
- Documentación: `http://localhost:8000/api/docs`
- ReDoc: `http://localhost:8000/api/redoc`

### 2. Instalar tipos TypeScript en el frontend

Copia los archivos TypeScript a tu proyecto frontend:

```
frontend/src/
├── api/
│   ├── bul-api-client.ts
│   └── frontend_types.ts
```

### 3. Usar el cliente en tu frontend

```typescript
import { createBULApiClient } from './api/bul-api-client';

// Crear instancia del cliente
const apiClient = createBULApiClient({
  baseUrl: 'http://localhost:8000',
  timeout: 30000
});

// Generar un documento
const response = await apiClient.generateDocument({
  query: 'Crear una estrategia de marketing para un nuevo restaurante',
  business_area: 'marketing',
  document_type: 'strategy',
  priority: 1
});

console.log('Task ID:', response.task_id);

// Esperar a que se complete
const document = await apiClient.generateDocumentAndWait(
  {
    query: 'Crear una estrategia de marketing para un nuevo restaurante',
    business_area: 'marketing',
    document_type: 'strategy',
    priority: 1
  },
  {
    onProgress: (status) => {
      console.log(`Progreso: ${status.progress}%`);
    }
  }
);

console.log('Documento generado:', document.document.content);
```

## 📚 Endpoints Disponibles

### Sistema

- `GET /` - Información del sistema
- `GET /api/health` - Health check
- `GET /api/stats` - Estadísticas del sistema

### Documentos

- `POST /api/documents/generate` - Generar documento
- `GET /api/tasks/{task_id}/document` - Obtener documento generado
- `GET /api/documents` - Listar documentos

### Tareas

- `GET /api/tasks/{task_id}/status` - Estado de tarea
- `GET /api/tasks` - Listar tareas
- `DELETE /api/tasks/{task_id}` - Eliminar tarea
- `POST /api/tasks/{task_id}/cancel` - Cancelar tarea

## 🔧 Configuración

### CORS

La API está configurada con CORS para permitir conexiones desde cualquier origen. En producción, deberías especificar los dominios permitidos:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://tu-dominio.com", "https://www.tu-dominio.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### Variables de Entorno

Crea un archivo `.env` si necesitas configuración adicional:

```env
API_HOST=0.0.0.0
API_PORT=8000
CORS_ORIGINS=http://localhost:3000,http://localhost:5173
```

## 📝 Ejemplos de Uso

### Ejemplo 1: Generar documento básico

```typescript
import { createBULApiClient } from './api/bul-api-client';

const client = createBULApiClient({
  baseUrl: 'http://localhost:8000'
});

async function generateBasicDocument() {
  try {
    const response = await client.generateDocument({
      query: 'Crear un plan de negocios para una startup tecnológica',
      priority: 1
    });

    console.log('Documento en proceso:', response.task_id);
    
    // Polling manual
    let status = await client.getTaskStatus(response.task_id);
    while (status.status === 'queued' || status.status === 'processing') {
      await new Promise(resolve => setTimeout(resolve, 2000));
      status = await client.getTaskStatus(response.task_id);
      console.log(`Progreso: ${status.progress}%`);
    }

    if (status.status === 'completed') {
      const document = await client.getTaskDocument(response.task_id);
      console.log('Documento:', document.document.content);
    }
  } catch (error) {
    console.error('Error:', error);
  }
}
```

### Ejemplo 2: Generar con polling automático

```typescript
async function generateWithAutoPolling() {
  try {
    const document = await client.generateDocumentAndWait(
      {
        query: 'Estrategia de ventas para B2B',
        business_area: 'sales',
        document_type: 'strategy',
        priority: 2
      },
      {
        pollingInterval: 2000, // Poll cada 2 segundos
        onProgress: (status) => {
          console.log(`Progreso: ${status.progress}% - Estado: ${status.status}`);
        }
      }
    );

    console.log('Documento completo:', document.document.content);
  } catch (error) {
    console.error('Error:', error);
  }
}
```

### Ejemplo 3: Listar tareas y documentos

```typescript
async function listItems() {
  // Listar tareas
  const tasks = await client.listTasks({
    status: 'completed',
    limit: 10,
    offset: 0
  });

  console.log(`Total tareas: ${tasks.total}`);
  tasks.tasks.forEach(task => {
    console.log(`- ${task.task_id}: ${task.status} - ${task.query_preview}`);
  });

  // Listar documentos
  const documents = await client.listDocuments(10, 0);
  console.log(`Total documentos: ${documents.total}`);
  documents.documents.forEach(doc => {
    console.log(`- ${doc.task_id}: ${doc.query_preview}`);
  });
}
```

### Ejemplo 4: React Hook

```typescript
import { useState, useEffect } from 'react';
import { createBULApiClient, TaskStatus } from './api/bul-api-client';

const client = createBULApiClient({
  baseUrl: 'http://localhost:8000'
});

export function useDocumentGeneration() {
  const [status, setStatus] = useState<TaskStatus | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const generateDocument = async (query: string) => {
    setLoading(true);
    setError(null);

    try {
      const response = await client.generateDocument({ query });
      
      // Polling
      const finalStatus = await client.waitForTaskCompletion(
        response.task_id,
        {
          onProgress: setStatus
        }
      );

      if (finalStatus.status === 'completed') {
        const document = await client.getTaskDocument(response.task_id);
        setStatus(finalStatus);
        return document;
      } else {
        throw new Error(finalStatus.error || 'Generation failed');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
      throw err;
    } finally {
      setLoading(false);
    }
  };

  return { generateDocument, status, loading, error };
}
```

## 🛡️ Manejo de Errores

El cliente maneja automáticamente:
- Timeouts de red
- Errores HTTP
- Errores de parsing JSON

Ejemplo de manejo:

```typescript
try {
  const document = await client.generateDocumentAndWait({
    query: 'Mi consulta'
  });
} catch (error) {
  if (error instanceof Error) {
    console.error('Error:', error.message);
    
    if (error.message.includes('timeout')) {
      // Manejar timeout
    } else if (error.message.includes('HTTP')) {
      // Manejar error HTTP
    }
  }
}
```

## 🔍 Verificación

### Verificar que la API está funcionando

```bash
# Health check
curl http://localhost:8000/api/health

# Generar documento de prueba
curl -X POST http://localhost:8000/api/documents/generate \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Crear un documento de prueba",
    "priority": 1
  }'
```

## 📦 Dependencias del Frontend

Si usas el cliente TypeScript, necesitarás:

```json
{
  "devDependencies": {
    "typescript": "^5.0.0"
  }
}
```

No se requieren dependencias adicionales ya que el cliente usa la API Fetch nativa del navegador.

## 🚀 Producción

Para producción:

1. Configura CORS con dominios específicos
2. Añade autenticación si es necesario
3. Configura rate limiting
4. Usa HTTPS
5. Configura variables de entorno

## 📖 Documentación Completa

La documentación completa de la API está disponible en:
- Swagger UI: `http://localhost:8000/api/docs`
- ReDoc: `http://localhost:8000/api/redoc`
- OpenAPI JSON: `http://localhost:8000/api/openapi.json`

## 🆘 Soporte

Para problemas o preguntas:
1. Revisa los logs del servidor: `bul_api.log`
2. Verifica el health check: `GET /api/health`
3. Revisa la documentación en `/api/docs`
































