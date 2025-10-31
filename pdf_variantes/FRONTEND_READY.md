# API Lista para Frontend TypeScript ✅

La API de PDF Variantes está completamente configurada y lista para ser utilizada desde un frontend TypeScript.

## ✅ Mejoras Implementadas

### 1. CORS Configurado
- ✅ CORS middleware configurado con orígenes permitidos para desarrollo
- ✅ Soporte para puertos comunes: 3000, 3001, 5173 (Vite), 4200 (Angular)
- ✅ Headers expuestos: Content-Disposition, Content-Type, X-Total-Count
- ✅ Métodos permitidos: GET, POST, PUT, DELETE, PATCH, OPTIONS
- ✅ Credentials habilitadas para autenticación

### 2. Routers Completos
- ✅ `/api/v1/pdf` - Operaciones PDF
- ✅ `/api/v1/variants` - Generación de variantes
- ✅ `/api/v1/topics` - Extracción de temas
- ✅ `/api/v1/brainstorm` - Generación de ideas
- ✅ `/api/v1/collaboration` - Colaboración en tiempo real
- ✅ `/api/v1/export` - Exportación de contenido
- ✅ `/api/v1/analytics` - Analytics y reportes
- ✅ `/api/v1/search` - Búsqueda de contenido
- ✅ `/api/v1/batch` - Procesamiento por lotes
- ✅ `/api/v1/health` - Estado del sistema

### 3. SDK TypeScript Completo
- ✅ **types.ts** - Todos los tipos TypeScript generados desde modelos Pydantic
- ✅ **api-client.ts** - Cliente API completo con todas las funciones
- ✅ **config.ts** - Configuración para diferentes entornos
- ✅ **index.ts** - Punto de entrada principal
- ✅ **README.md** - Documentación completa con ejemplos

### 4. Documentación OpenAPI
- ✅ `/docs` - Swagger UI interactivo
- ✅ `/redoc` - ReDoc documentation
- ✅ `/openapi.json` - Especificación OpenAPI completa

## 📁 Estructura de Archivos TypeScript

```
typescript/
├── types.ts          # Todos los tipos TypeScript
├── api-client.ts     # Cliente API completo
├── config.ts         # Configuración
├── index.ts          # Exportaciones principales
└── README.md         # Documentación completa
```

## 🚀 Uso Rápido

### Instalación en Frontend

1. Copia la carpeta `typescript/` a tu proyecto:

```bash
cp -r typescript/ src/lib/pdf-variantes-api/
```

2. Instala el cliente en tu código:

```typescript
import { createClient } from './lib/pdf-variantes-api';

const apiClient = createClient({
  baseURL: 'http://localhost:8000',
  apiKey: 'your-api-key', // Opcional
});
```

3. Usa el cliente:

```typescript
// Subir PDF
const response = await apiClient.uploadPDF(file);

// Generar variantes
const variants = await apiClient.generateVariants({
  document_id: 'doc-123',
  number_of_variants: 10,
});

// Extraer temas
const topics = await apiClient.extractTopics({
  document_id: 'doc-123',
  min_relevance: 0.7,
});
```

## 📝 Endpoints Disponibles

### PDF Operations
```
POST   /api/v1/pdf/upload
GET    /api/v1/pdf/documents
GET    /api/v1/pdf/documents/{document_id}
DELETE /api/v1/pdf/documents/{document_id}
```

### Variant Operations
```
POST   /api/v1/variants/generate
GET    /api/v1/variants/documents/{document_id}/variants
GET    /api/v1/variants/variants/{variant_id}
POST   /api/v1/variants/stop
```

### Topic Operations
```
POST   /api/v1/topics/extract
GET    /api/v1/topics/documents/{document_id}/topics
```

### Brainstorm Operations
```
POST   /api/v1/brainstorm/generate
GET    /api/v1/brainstorm/documents/{document_id}/ideas
```

### Search & Batch
```
POST   /api/v1/search/search
POST   /api/v1/batch/process
```

### Export
```
POST   /api/v1/export/export
GET    /api/v1/export/download/{file_id}
```

### Analytics
```
GET    /api/v1/analytics/dashboard
GET    /api/v1/analytics/reports
```

### Health
```
GET    /api/v1/health/status
GET    /health
```

## 🔧 Configuración CORS

Para agregar orígenes personalizados, configura la variable de entorno o ajusta `api/main.py`:

```python
cors_origins = [
    "http://localhost:3000",
    "http://localhost:5173",
    "https://tu-dominio.com",
]
```

## 🔐 Autenticación

La API soporta autenticación por Bearer Token:

```typescript
const apiClient = createClient({
  baseURL: 'http://localhost:8000',
  apiKey: 'tu-token-aqui',
});
```

El token se incluye automáticamente en el header `Authorization: Bearer {token}`.

## 📊 Manejo de Errores

Todas las respuestas siguen el formato estándar:

```typescript
interface APIResponse<T> {
  data?: T;
  error?: {
    message: string;
    code?: string | number;
    details?: Record<string, unknown>;
  };
  status: number;
}
```

Ejemplo de uso:

```typescript
const response = await apiClient.getDocument(docId);

if (response.error) {
  console.error('Error:', response.error.message);
  // response.error.code contiene el código HTTP
  // response.error.details contiene información adicional
  return;
}

// Usar datos
const document = response.data;
```

## 🌐 WebSocket (Colaboración)

Para colaboración en tiempo real:

```typescript
const ws = new WebSocket(
  `ws://localhost:8000/api/v1/collaboration/ws/${documentId}?user_id=${userId}`
);

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  // Manejar mensajes de colaboración
};
```

## ✅ Checklist de Integración

- [x] CORS configurado para frontend
- [x] Todos los routers incluidos
- [x] Tipos TypeScript generados
- [x] Cliente API completo
- [x] Documentación OpenAPI
- [x] Manejo de errores consistente
- [x] Soporte para autenticación
- [x] Ejemplos de uso
- [x] README completo

## 🔗 Enlaces Útiles

- **Documentación Swagger**: http://localhost:8000/docs
- **Especificación OpenAPI**: http://localhost:8000/openapi.json
- **Health Check**: http://localhost:8000/health
- **ReDoc**: http://localhost:8000/redoc

## 📦 Variables de Entorno Recomendadas

```env
# Frontend (Next.js/Vite)
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_API_KEY=your-api-key

# O para Vite
VITE_API_URL=http://localhost:8000
VITE_API_KEY=your-api-key
```

## 🎯 Próximos Pasos

1. Copiar la carpeta `typescript/` a tu proyecto frontend
2. Configurar las variables de entorno
3. Instalar el cliente en tu código
4. Comenzar a usar los endpoints!

## 📞 Soporte

Para más información, consulta:
- `typescript/README.md` - Documentación del SDK
- `/docs` - Documentación interactiva de la API
- Los archivos de ejemplo en el código fuente

---

**Estado**: ✅ **API LISTA PARA PRODUCCIÓN CON FRONTEND TYPESCRIPT**






