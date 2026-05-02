# 🚀 BUL API - Guía Completa

## 📋 Tabla de Contenidos

1. [Características](#características)
2. [Instalación](#instalación)
3. [SDKs](#sdks)
4. [Docker & Kubernetes](#docker--kubernetes)
5. [Ejemplos](#ejemplos)
6. [Documentación](#documentación)
7. [Monitoreo](#monitoreo)

## ✨ Características

### Core
- ✅ Generación de documentos con IA
- ✅ WebSocket para actualizaciones en tiempo real
- ✅ Rate limiting (10 req/min)
- ✅ Validaciones robustas
- ✅ Manejo de errores mejorado

### SDKs
- ✅ **TypeScript** - Tipado completo
- ✅ **JavaScript** - Universal (Node.js/Browser)
- ✅ **Python** - Cliente completo

### DevOps
- ✅ **Docker** - Dockerfile y docker-compose
- ✅ **Kubernetes** - Manifests completos
- ✅ **Prometheus** - Métricas integradas
- ✅ **Health Checks** - Verificación avanzada

### Desarrollo
- ✅ **Logging Estructurado** - JSON logs
- ✅ **OpenAPI** - Documentación completa
- ✅ **Ejemplos** - Múltiples lenguajes
- ✅ **Testing Suite** - Pruebas completas

## 📦 Instalación

### Local

```bash
# Instalar dependencias
pip install -r requirements.txt

# Iniciar servidor
python api_frontend_ready.py
```

### Docker

```bash
# Construir
docker build -t bul-api:latest .

# Ejecutar
docker run -p 8000:8000 bul-api:latest
```

### Docker Compose

```bash
# Iniciar API
docker-compose up -d

# Con monitoreo
docker-compose --profile monitoring up -d
```

### Kubernetes

```bash
# Aplicar manifests
kubectl apply -f k8s/

# Verificar
kubectl get pods
```

## 📚 SDKs

### TypeScript

```typescript
import { createBULApiClient } from './bul-api-client';

const client = createBULApiClient({
    baseUrl: 'http://localhost:8000'
});

const document = await client.generateDocumentAndWait({
    query: 'Plan de marketing'
});
```

### JavaScript

```javascript
const { createBULApiClient } = require('./bul-api-client.js');

const client = createBULApiClient({
    baseUrl: 'http://localhost:8000'
});

const document = await client.generateDocumentAndWait({
    query: 'Plan de marketing'
});
```

### Python

```python
from bul_api_client import create_bul_client, DocumentRequest

client = create_bul_client(base_url="http://localhost:8000")

request = DocumentRequest(
    query="Plan de marketing",
    business_area="marketing"
)

document = client.generate_document_and_wait(request)
```

## 🐳 Docker & Kubernetes

Ver [README_DOCKER_K8S.md](README_DOCKER_K8S.md) para detalles completos.

## 📖 Ejemplos

Ver [README_EXAMPLES.md](README_EXAMPLES.md) para ejemplos detallados.

### Quick Start

**Python:**
```bash
python examples/python_example.py
```

**JavaScript:**
```bash
node examples/javascript_example.js
```

**TypeScript:**
```bash
ts-node examples/typescript_example.ts
```

## 📚 Documentación

### Swagger UI
```
http://localhost:8000/api/docs
```

### ReDoc
```
http://localhost:8000/api/redoc
```

### OpenAPI Schema
```
http://localhost:8000/api/openapi.json
```

### Postman Collection
Importar `postman_collection.json` en Postman

## 📊 Monitoreo

### Métricas Prometheus
```
http://localhost:8000/metrics
```

### Health Check
```
http://localhost:8000/api/health
```

### Health Check Avanzado
```bash
python health_check_advanced.py
```

### Grafana (con Docker Compose)
```
http://localhost:3000
```

## 🔧 Configuración

### Variables de Entorno

```bash
LOG_LEVEL=INFO
PYTHONUNBUFFERED=1
```

### Rate Limiting
- Default: 10 requests/minute
- Configurable en código

### Timeouts
- Request timeout: 30s (configurable en SDK)
- Task timeout: 5 minutos (configurable)

## 🧪 Testing

```bash
# Ejecutar todas las pruebas
./run_all_tests.bat  # Windows
./run_all_tests.sh   # Linux/Mac

# Pruebas individuales
python test_api_responses.py
python test_api_advanced.py
python test_security.py
```

## 📝 Archivos Importantes

### SDKs
- `bul-api-client.ts` - TypeScript
- `bul-api-client.js` - JavaScript
- `bul-api-client.py` - Python
- `frontend_types.ts` - Tipos TypeScript

### Docker
- `Dockerfile` - Imagen Docker
- `docker-compose.yml` - Orquestación
- `.dockerignore` - Exclusiones

### Kubernetes
- `k8s/deployment.yaml` - Deployment
- `k8s/service.yaml` - Service
- `k8s/ingress.yaml` - Ingress

### Documentación
- `README_DOCKER_K8S.md` - Docker/K8s
- `README_EXAMPLES.md` - Ejemplos
- `README_SDK_COMPLETE.md` - SDKs
- `README_MEJORAS_ULTIMAS.md` - Mejoras

### Testing
- `test_api_responses.py` - Pruebas básicas
- `test_api_advanced.py` - Pruebas avanzadas
- `test_security.py` - Pruebas de seguridad
- `health_check_advanced.py` - Health check

## 🚀 Quick Start Completo

1. **Instalar dependencias**
   ```bash
   pip install -r requirements.txt
   ```

2. **Iniciar API**
   ```bash
   python api_frontend_ready.py
   ```

3. **Verificar salud**
   ```bash
   curl http://localhost:8000/api/health
   ```

4. **Generar documento (Python)**
   ```bash
   python examples/python_example.py
   ```

5. **Ver documentación**
   ```
   http://localhost:8000/api/docs
   ```

## 📞 Soporte

### Documentación
- Swagger UI: `/api/docs`
- ReDoc: `/api/redoc`
- OpenAPI: `/api/openapi.json`

### Logs
- Consola: stdout
- Archivo: `bul_api.log`
- Estructurado: `logs/app.json` (si se configura)

### Métricas
- Prometheus: `/metrics`
- Health: `/api/health`
- Stats: `/api/stats`

---

**Versión**: 1.0.0  
**Estado**: ✅ **Producción Ready**
































