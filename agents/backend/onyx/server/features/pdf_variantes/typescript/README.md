# PDF Variantes API - TypeScript Client

SDK completo para integrar la API de PDF Variantes en aplicaciones TypeScript/JavaScript.

## Instalación

### Opción 1: Copiar archivos directamente
Copia los archivos de esta carpeta a tu proyecto frontend:

```bash
# React/Vue/Next.js/etc
cp -r typescript/ src/lib/pdf-variantes-api/
```

### Opción 2: Instalar como módulo local
Si usas un monorepo, puedes importar directamente desde esta carpeta.

## Uso Básico

### 1. Configurar el cliente

```typescript
import { createClient } from './lib/pdf-variantes-api';

const apiClient = createClient({
  baseURL: 'http://localhost:8000',
  apiKey: 'your-api-key', // Opcional
  timeout: 30000,
});
```

### 2. Subir un PDF

```typescript
import { apiClient } from './lib/pdf-variantes-api';

async function uploadPDF(file: File) {
  const response = await apiClient.uploadPDF(file, {
    auto_process: true,
    extract_text: true,
  });

  if (response.error) {
    console.error('Error:', response.error.message);
    return;
  }

  console.log('Document uploaded:', response.data?.document);
}
```

### 3. Generar Variantes

```typescript
async function generateVariants(documentId: string) {
  const response = await apiClient.generateVariants({
    document_id: documentId,
    number_of_variants: 10,
    continuous_generation: true,
  });

  if (response.error) {
    console.error('Error:', response.error.message);
    return;
  }

  console.log('Variants generated:', response.data?.variants);
}
```

### 4. Extraer Temas

```typescript
async function extractTopics(documentId: string) {
  const response = await apiClient.extractTopics({
    document_id: documentId,
    min_relevance: 0.7,
    max_topics: 20,
  });

  if (response.error) {
    console.error('Error:', response.error.message);
    return;
  }

  console.log('Topics:', response.data?.topics);
}
```

## Ejemplos Completos

### React Hook Example

```typescript
import { useState } from 'react';
import { apiClient } from './lib/pdf-variantes-api';
import type { PDFDocument } from './lib/pdf-variantes-api';

export function usePDFUpload() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [document, setDocument] = useState<PDFDocument | null>(null);

  const upload = async (file: File) => {
    setLoading(true);
    setError(null);

    const response = await apiClient.uploadPDF(file);

    if (response.error) {
      setError(response.error.message);
      setLoading(false);
      return;
    }

    setDocument(response.data?.document || null);
    setLoading(false);
  };

  return { upload, loading, error, document };
}
```

### Vue Composable Example

```typescript
import { ref } from 'vue';
import { apiClient } from './lib/pdf-variantes-api';
import type { PDFVariant } from './lib/pdf-variantes-api';

export function useVariants(documentId: string) {
  const variants = ref<PDFVariant[]>([]);
  const loading = ref(false);
  const error = ref<string | null>(null);

  const fetchVariants = async () => {
    loading.value = true;
    error.value = null;

    const response = await apiClient.listVariants(documentId);

    if (response.error) {
      error.value = response.error.message;
      loading.value = false;
      return;
    }

    variants.value = response.data || [];
    loading.value = false;
  };

  const generate = async (count: number = 10) => {
    loading.value = true;
    const response = await apiClient.generateVariants({
      document_id: documentId,
      number_of_variants: count,
    });

    if (response.error) {
      error.value = response.error.message;
      loading.value = false;
      return;
    }

    await fetchVariants();
  };

  return {
    variants,
    loading,
    error,
    fetchVariants,
    generate,
  };
}
```

## Todas las Funciones Disponibles

### PDF Operations
- `uploadPDF(file, options)` - Subir PDF
- `listDocuments(params)` - Listar documentos
- `getDocument(documentId)` - Obtener documento
- `deleteDocument(documentId)` - Eliminar documento

### Variant Operations
- `generateVariants(request)` - Generar variantes
- `listVariants(documentId, params)` - Listar variantes
- `getVariant(variantId)` - Obtener variante
- `stopGeneration(request)` - Detener generación

### Topic Operations
- `extractTopics(request)` - Extraer temas
- `listTopics(documentId, minRelevance)` - Listar temas

### Brainstorm Operations
- `generateBrainstormIdeas(request)` - Generar ideas
- `listBrainstormIdeas(documentId, category)` - Listar ideas

### Search & Batch
- `search(request)` - Buscar contenido
- `batchProcess(request)` - Procesamiento en lote

### Export
- `export(request)` - Exportar contenido
- `downloadFile(fileId)` - Descargar archivo exportado

### Analytics & Health
- `getDashboard()` - Dashboard de analytics
- `getAnalyticsReport(startDate, endDate)` - Reporte de analytics
- `getHealth()` - Estado del sistema
- `healthCheck()` - Health check simple

## Manejo de Errores

Todas las funciones retornan un objeto `APIResponse<T>`:

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

Siempre verifica `response.error` antes de usar `response.data`:

```typescript
const response = await apiClient.getDocument(documentId);

if (response.error) {
  // Manejar error
  console.error('Error:', response.error.message);
  return;
}

// Usar datos
const document = response.data;
```

## Variables de Entorno

Configura estas variables en tu proyecto:

```env
# Next.js
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_API_KEY=your-api-key

# Vite
VITE_API_URL=http://localhost:8000
VITE_API_KEY=your-api-key
```

## Tipos TypeScript

Todos los tipos están exportados desde `types.ts`:

```typescript
import type {
  PDFDocument,
  PDFVariant,
  TopicItem,
  VariantGenerateRequest,
  // ... más tipos
} from './lib/pdf-variantes-api';
```

## WebSocket (Colaboración)

Para WebSocket, necesitarás usar una librería externa como `ws` o `socket.io-client`:

```typescript
const ws = new WebSocket('ws://localhost:8000/api/v1/collaboration/ws/{document_id}?user_id={user_id}');

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  // Manejar mensajes
};
```

## Soporte y Documentación

- API Docs: `http://localhost:8000/docs`
- OpenAPI Spec: `http://localhost:8000/openapi.json`
- Health Check: `http://localhost:8000/health`






