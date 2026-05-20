# 📦 SDK Completo - API BUL

## 🎯 SDKs Disponibles

### TypeScript/JavaScript
- **bul-api-client.ts** - Cliente TypeScript completo
- **bul-api-client.js** - Cliente JavaScript (compatible Node.js y navegador)
- **frontend_types.ts** - Tipos TypeScript

### Python (Próximamente)
- Cliente Python con requests
- Async/await support

### Otros Lenguajes
- Postman Collection disponible
- OpenAPI spec para generación automática

## 📚 Instalación

### TypeScript/JavaScript

**NPM (próximamente):**
```bash
npm install bul-api-client
```

**Manual:**
```bash
# Copiar archivos
cp bul-api-client.ts frontend/src/api/
cp bul-api-client.js frontend/src/api/
cp frontend_types.ts frontend/src/api/
```

**Browser:**
```html
<script src="bul-api-client.js"></script>
<script>
  const client = createBULApiClient({
    baseUrl: 'http://localhost:8000'
  });
</script>
```

**Node.js:**
```javascript
const { createBULApiClient } = require('./bul-api-client.js');

const client = createBULApiClient({
  baseUrl: 'http://localhost:8000'
});
```

## 🚀 Uso

### TypeScript
```typescript
import { createBULApiClient } from './api/bul-api-client';

const client = createBULApiClient({
  baseUrl: 'http://localhost:8000'
});

const document = await client.generateDocumentAndWait({
  query: 'Crear un plan de marketing'
});
```

### JavaScript
```javascript
const client = createBULApiClient({
  baseUrl: 'http://localhost:8000'
});

const document = await client.generateDocumentAndWait({
  query: 'Crear un plan de marketing'
});
```

## 📋 Postman Collection

Importa `postman_collection.json` en Postman para:
- Probar todos los endpoints
- Ejemplos de requests
- Variables de entorno
- Tests automatizados

## 🔧 Características del SDK

### TypeScript
- ✅ Tipado completo
- ✅ Autocompletado
- ✅ Validación de tipos
- ✅ Documentación inline

### JavaScript
- ✅ Compatible Node.js
- ✅ Compatible navegador
- ✅ Sin dependencias
- ✅ WebSocket support

### Ambos
- ✅ WebSocket automático
- ✅ Polling con fallback
- ✅ Manejo de errores
- ✅ Timeouts configurables

## 📊 Métricas Prometheus

### Endpoint
```
GET /metrics
```

### Métricas Disponibles
- `bul_requests_total` - Total de requests
- `bul_request_duration_seconds` - Duración de requests
- `bul_active_tasks` - Tareas activas
- `bul_document_generation_seconds` - Tiempo de generación
- `bul_cache_hits_total` - Cache hits
- `bul_cache_misses_total` - Cache misses
- `bul_errors_total` - Errores

### Uso
```bash
# Iniciar servidor de métricas
python prometheus_metrics.py --port 9090

# Acceder a métricas
curl http://localhost:9090/metrics
```

## 🔍 Health Check Avanzado

```bash
python health_check_advanced.py
```

**Verifica:**
- Salud de la API
- Almacenamiento
- Espacio en disco
- Uso de memoria
- Disponibilidad de endpoints

## 📦 Archivos del SDK

### TypeScript
- `bul-api-client.ts` - Cliente completo
- `frontend_types.ts` - Tipos

### JavaScript
- `bul-api-client.js` - Cliente compatible

### Postman
- `postman_collection.json` - Collection completa

### Package
- `package.json` - Configuración NPM

## 🎯 Ejemplos por Lenguaje

### TypeScript
```typescript
import { createBULApiClient } from './api/bul-api-client';

const client = createBULApiClient({
  baseUrl: 'http://localhost:8000'
});

// Generar documento
const doc = await client.generateDocumentAndWait({
  query: 'Plan de marketing',
  business_area: 'marketing'
});
```

### JavaScript
```javascript
const client = createBULApiClient({
  baseUrl: 'http://localhost:8000'
});

// Con async/await
async function generateDoc() {
  const doc = await client.generateDocumentAndWait({
    query: 'Plan de marketing'
  });
  console.log(doc.document.content);
}
```

### Postman
1. Importar `postman_collection.json`
2. Configurar variable `base_url`
3. Ejecutar requests

## 🔗 Integraciones

### Prometheus
```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'bul-api'
    static_configs:
      - targets: ['localhost:9090']
```

### Grafana
Usar métricas Prometheus para dashboards

---

**Estado**: ✅ **SDK Completo Disponible**
































