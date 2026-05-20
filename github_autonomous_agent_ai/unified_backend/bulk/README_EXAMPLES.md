# 📚 Ejemplos de Uso - API BUL

## 🎯 Ejemplos Disponibles

### Python
**Archivo:** `examples/python_example.py`

```python
from bul_api_client import create_bul_client, DocumentRequest

client = create_bul_client(base_url="http://localhost:8000")

request = DocumentRequest(
    query="Crear un plan de marketing digital",
    business_area="marketing",
    document_type="strategy"
)

document = client.generate_document_and_wait(request)
print(document['document']['title'])
```

**Ejecutar:**
```bash
python examples/python_example.py
```

### JavaScript/Node.js
**Archivo:** `examples/javascript_example.js`

```javascript
const { createBULApiClient } = require('./bul-api-client.js');

const client = createBULApiClient({
    baseUrl: 'http://localhost:8000'
});

const document = await client.generateDocumentAndWait({
    query: 'Crear un plan de marketing digital',
    business_area: 'marketing'
});
```

**Ejecutar:**
```bash
node examples/javascript_example.js
```

### TypeScript
**Archivo:** `examples/typescript_example.ts`

```typescript
import { createBULApiClient } from '../bul-api-client';
import { DocumentRequest } from '../frontend_types';

const client = createBULApiClient({
    baseUrl: 'http://localhost:8000'
});

const request: DocumentRequest = {
    query: 'Crear un plan de marketing digital',
    business_area: 'marketing'
};

const document = await client.generateDocumentAndWait(request);
```

**Ejecutar:**
```bash
ts-node examples/typescript_example.ts
```

## 🔧 Uso Básico

### 1. Verificar Salud
```python
# Python
health = client.get_health()

# JavaScript
const health = await client.getHealth();
```

### 2. Generar Documento

**Python:**
```python
request = DocumentRequest(
    query="Tu consulta aquí",
    business_area="marketing",
    document_type="strategy"
)

document = client.generate_document_and_wait(request)
```

**JavaScript:**
```javascript
const document = await client.generateDocumentAndWait({
    query: 'Tu consulta aquí',
    business_area: 'marketing',
    document_type: 'strategy'
});
```

### 3. Con WebSocket (Tiempo Real)

**Python:**
```python
def on_progress(status):
    print(f"Progreso: {status.progress}%")

document = client.generate_document_and_wait(
    request,
    use_websocket=True,
    on_progress=on_progress
)
```

**JavaScript:**
```javascript
const document = await client.generateDocumentAndWait(request, {
    useWebSocket: true,
    onProgress: (status) => {
        console.log(`Progreso: ${status.progress}%`);
    }
});
```

### 4. Listar Documentos

**Python:**
```python
documents = client.list_documents(limit=10, offset=0)
```

**JavaScript:**
```javascript
const documents = await client.listDocuments(10, 0);
```

### 5. Obtener Estado de Tarea

**Python:**
```python
status = client.get_task_status(task_id)
```

**JavaScript:**
```javascript
const status = await client.getTaskStatus(taskId);
```

## 📊 Casos de Uso Avanzados

### Procesamiento en Lote

**Python:**
```python
requests = [
    DocumentRequest(query="Consulta 1"),
    DocumentRequest(query="Consulta 2"),
    DocumentRequest(query="Consulta 3")
]

for request in requests:
    document = client.generate_document_and_wait(request)
    print(f"Documento generado: {document['document']['title']}")
```

### Manejo de Errores

**Python:**
```python
try:
    document = client.generate_document_and_wait(request)
except Exception as e:
    print(f"Error: {e}")
    # Retry logic here
```

**JavaScript:**
```javascript
try {
    const document = await client.generateDocumentAndWait(request);
} catch (error) {
    console.error('Error:', error.message);
    // Retry logic here
}
```

### Polling Personalizado

**Python:**
```python
status = client.wait_for_task_completion(
    task_id,
    interval=5,  # 5 segundos
    max_attempts=60,  # 5 minutos máximo
    on_progress=lambda s: print(f"Status: {s.status}")
)
```

## 🎯 Integraciones

### React/Next.js

```typescript
import { createBULApiClient } from '@/lib/bul-api-client';

export function useBULDocument() {
    const client = createBULApiClient({
        baseUrl: process.env.NEXT_PUBLIC_API_URL
    });
    
    const generateDocument = async (query: string) => {
        return await client.generateDocumentAndWait({
            query,
            business_area: 'marketing'
        });
    };
    
    return { generateDocument };
}
```

### Vue.js

```javascript
import { createBULApiClient } from './bul-api-client';

export default {
    data() {
        return {
            client: createBULApiClient({
                baseUrl: 'http://localhost:8000'
            })
        };
    },
    methods: {
        async generateDocument(query) {
            return await this.client.generateDocumentAndWait({
                query,
                business_area: 'marketing'
            });
        }
    }
};
```

## 📝 Notas

- Todos los ejemplos asumen que la API está corriendo en `http://localhost:8000`
- WebSocket requiere que el servidor esté disponible
- Los timeouts por defecto son 30 segundos (configurable)
- El polling por defecto es cada 2 segundos

---

**Más información:** Ver `README_SDK_COMPLETE.md`
































