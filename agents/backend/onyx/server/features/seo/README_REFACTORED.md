# 🚀 Servicio SEO Ultra-Optimizado - Versión Refactorizada

## 📋 Descripción

Servicio SEO ultra-optimizado con arquitectura modular refactorizada. Proporciona análisis SEO completo con máxima eficiencia, utilizando las librerías más rápidas disponibles y una arquitectura limpia y mantenible.

## ✨ Características Principales

### 🏗️ Arquitectura Modular
- **Separación de responsabilidades**: Cada componente tiene una función específica
- **Inyección de dependencias**: Configuración flexible y testing facilitado
- **Interfaces abstractas**: Contratos claros entre componentes
- **Patrón Factory**: Creación dinámica de componentes

### ⚡ Rendimiento Ultra-Optimizado
- **Selectolax parser**: 3x más rápido que BeautifulSoup
- **OrJSON serialization**: 3-5x más rápido que json estándar
- **Zstandard compression**: 70-80% de compresión en caché
- **HTTPX con connection pooling**: Conexiones optimizadas
- **AsyncIO throttling**: Control de concurrencia inteligente

### 🔧 Componentes Modulares

#### Core Components
- **HTML Parsers**: SelectolaxUltraParser, LXMLFallbackParser
- **HTTP Clients**: UltraFastHTTPClient con connection pooling
- **Cache Managers**: UltraOptimizedCacheManager, MultiLevelCacheManager
- **SEO Analyzers**: UltraFastSEOAnalyzer, RuleBasedAnalyzer
- **Performance Tracking**: Métricas detalladas en tiempo real

#### Services
- **SEOService**: Servicio principal con orquestación
- **SeleniumService**: Integración con Selenium para JavaScript
- **BatchProcessingService**: Procesamiento en lote optimizado

## 🚀 Instalación

### Requisitos Previos

```bash
# Python 3.8+
python --version

# Chrome/Chromium para Selenium
# ChromeDriver será descargado automáticamente
```

### Instalación de Dependencias

```bash
# Instalar dependencias ultra-optimizadas
pip install -r requirements.ultra_optimized.txt

# O instalar dependencias básicas
pip install -r requirements.txt
```

### Configuración Rápida

```bash
# Ejecutar script de configuración
python setup.py

# O configurar manualmente
export OPENAI_API_KEY="your-api-key"
export SEO_CACHE_SIZE=2000
export SEO_RATE_LIMIT=100
```

## 📖 Uso Básico

### Análisis SEO Simple

```python
from seo.services.seo_service import UltraOptimizedSEOService
from seo.models import SEOScrapeRequest

# Crear servicio con configuración por defecto
service = UltraOptimizedSEOService()

# Realizar análisis
request = SEOScrapeRequest(url="https://example.com")
response = await service.scrape(request)

print(f"Score SEO: {response.data['analysis']['score']}")
print(f"Recomendaciones: {response.data['analysis']['recommendations']}")
```

### Configuración Avanzada

```python
# Configuración personalizada
config = {
    'parser': {
        'type': 'selectolax',
        'fallback': 'lxml'
    },
    'http_client': {
        'type': 'ultra_fast',
        'rate_limit': 200,
        'max_connections': 100,
        'timeout': 15.0
    },
    'cache': {
        'type': 'multi_level',
        'l1_maxsize': 1000,
        'l2_enabled': True,
        'compression_level': 3
    },
    'analyzer': {
        'type': 'ultra_fast',
        'model': 'gpt-3.5-turbo',
        'temperature': 0.1,
        'max_tokens': 1000
    },
    'performance': {
        'enable_tracemalloc': True,
        'metrics_window': 1000
    }
}

service = UltraOptimizedSEOService(config)
```

### Procesamiento en Lote

```python
# Análisis de múltiples URLs
urls = [
    "https://example1.com",
    "https://example2.com",
    "https://example3.com"
]

requests = [SEOScrapeRequest(url=url) for url in urls]
responses = await service.batch_scrape(requests, max_concurrent=10)

for response in responses:
    print(f"{response.url}: Score {response.data['analysis']['score']}")
```

## 🔧 Configuración Detallada

### Parsers HTML

```python
# Selectolax (más rápido)
parser_config = {
    'type': 'selectolax',
    'max_images': 15,
    'max_links': 30
}

# LXML (fallback)
parser_config = {
    'type': 'lxml',
    'xpath_optimization': True
}

# Auto-detect
parser_config = {
    'type': 'auto'  # Intenta selectolax, fallback a lxml
}
```

### HTTP Client

```python
http_config = {
    'type': 'ultra_fast',
    'rate_limit': 100,        # Requests por minuto
    'period': 60,             # Período en segundos
    'max_keepalive': 20,      # Conexiones persistentes
    'max_connections': 100,   # Conexiones totales
    'timeout': 10.0,          # Timeout general
    'connect_timeout': 5.0,   # Timeout de conexión
    'http2': True,            # Usar HTTP/2
    'follow_redirects': True  # Seguir redirecciones
}
```

### Cache Manager

```python
# Caché simple optimizado
cache_config = {
    'type': 'ultra_optimized',
    'maxsize': 2000,          # Elementos máximos
    'ttl': 3600,              # TTL en segundos
    'compression_level': 3    # Nivel de compresión Zstandard
}

# Caché multi-nivel
cache_config = {
    'type': 'multi_level',
    'l1_maxsize': 1000,       # Caché L1 (memoria)
    'l1_ttl': 300,            # TTL L1 (5 minutos)
    'l2_enabled': True,       # Habilitar L2 (disco)
    'l2_directory': './cache', # Directorio L2
    'l2_size_limit': 100      # Límite L2 en MB
}
```

### SEO Analyzer

```python
# LangChain analyzer
analyzer_config = {
    'type': 'ultra_fast',
    'api_key': 'your-openai-key',
    'model': 'gpt-3.5-turbo',
    'temperature': 0.1,
    'max_tokens': 1000,
    'timeout': 30
}

# Rule-based analyzer
analyzer_config = {
    'type': 'rule_based',
    'rules': {
        'title': {'min_length': 30, 'max_length': 60},
        'meta_description': {'min_length': 120, 'max_length': 160},
        'content': {'min_length': 300}
    }
}
```

## 📊 API Reference

### SEOScrapeRequest

```python
@dataclass
class SEOScrapeRequest:
    url: str
    options: Optional[Dict[str, Any]] = None
    force_refresh: bool = False
```

### SEOScrapeResponse

```python
@dataclass
class SEOScrapeResponse:
    url: str
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    metrics: Optional[SEOMetrics] = None
```

### SEOMetrics

```python
@dataclass
class SEOMetrics:
    load_time: float
    memory_usage: float
    cache_hit: bool
    processing_time: float
    elements_extracted: int
    compression_ratio: float
    network_latency: float
```

## 🔍 Monitoreo y Métricas

### Métricas de Rendimiento

```python
# Obtener estadísticas completas
stats = service.get_comprehensive_stats()

print(f"Success Rate: {stats['performance']['summary']['success_rate']:.2%}")
print(f"Avg Response Time: {stats['performance']['summary']['avg_response_time']:.3f}s")
print(f"Cache Hit Rate: {stats['cache']['hit_rate']:.2%}")
print(f"Memory Usage: {stats['system']['process']['memory_usage_mb']:.2f}MB")
```

### Métricas del Sistema

```python
# Métricas del sistema en tiempo real
system_metrics = service.get_system_metrics()

print(f"CPU Usage: {system_metrics['system']['cpu_percent']:.1f}%")
print(f"Memory Usage: {system_metrics['system']['memory_percent']:.1f}%")
print(f"Disk Usage: {system_metrics['system']['disk_usage_percent']:.1f}%")
```

### Análisis de Tendencias

```python
# Obtener métricas históricas
tracker = service.performance_tracker
processing_times = tracker.get_historical_metrics('processing_times', 100)

if processing_times:
    avg_time = statistics.mean(processing_times)
    max_time = max(processing_times)
    min_time = min(processing_times)
    
    print(f"Avg Processing Time: {avg_time:.3f}s")
    print(f"Max Processing Time: {max_time:.3f}s")
    print(f"Min Processing Time: {min_time:.3f}s")
```

## 🧪 Testing

### Tests Unitarios

```bash
# Ejecutar todos los tests
pytest tests/

# Tests específicos
pytest tests/test_core.py
pytest tests/test_services.py
pytest tests/test_api.py

# Tests con coverage
pytest --cov=seo tests/
```

### Tests de Performance

```bash
# Ejecutar benchmarks
python test_ultra_optimized.py

# Tests de carga
python -m pytest tests/test_performance.py -v
```

### Tests de Integración

```python
# Test de integración completo
async def test_full_integration():
    service = UltraOptimizedSEOService()
    
    request = SEOScrapeRequest(url="https://example.com")
    response = await service.scrape(request)
    
    assert response.success
    assert response.data is not None
    assert 'analysis' in response.data
    assert 'score' in response.data['analysis']
    
    await service.close()
```

## 🚀 Deployment

### Docker

```dockerfile
# Usar imagen optimizada
FROM python:3.11-slim

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    chromium \
    chromium-driver \
    && rm -rf /var/lib/apt/lists/*

# Copiar código
COPY . /app
WORKDIR /app

# Instalar dependencias Python
RUN pip install -r requirements.ultra_optimized.txt

# Exponer puerto
EXPOSE 8000

# Comando de inicio
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Docker Compose

```yaml
version: '3.8'
services:
  seo-service:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - SEO_CACHE_SIZE=2000
      - SEO_RATE_LIMIT=100
    volumes:
      - ./cache:/app/cache
    restart: unless-stopped
```

### Kubernetes

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: seo-service
spec:
  replicas: 3
  selector:
    matchLabels:
      app: seo-service
  template:
    metadata:
      labels:
        app: seo-service
    spec:
      containers:
      - name: seo-service
        image: seo-service:latest
        ports:
        - containerPort: 8000
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: seo-secrets
              key: openai-api-key
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
```

## 🔧 Troubleshooting

### Problemas Comunes

#### Error de Memoria
```python
# Reducir tamaño de caché
config = {
    'cache': {
        'maxsize': 500,  # Reducir de 2000 a 500
        'compression_level': 1  # Compresión más rápida
    }
}
```

#### Timeouts
```python
# Aumentar timeouts
config = {
    'http_client': {
        'timeout': 30.0,  # Aumentar de 10 a 30 segundos
        'connect_timeout': 10.0
    }
}
```

#### Rate Limiting
```python
# Ajustar rate limiting
config = {
    'http_client': {
        'rate_limit': 50,  # Reducir de 100 a 50
        'period': 60
    }
}
```

### Logs y Debugging

```python
# Habilitar logging detallado
import logging
logging.basicConfig(level=logging.DEBUG)

# O usar loguru
from loguru import logger
logger.add("debug.log", level="DEBUG", rotation="10 MB")
```

## 📈 Performance Benchmarks

### Comparación de Parsers

| Parser | Tiempo (ms) | Memoria (MB) | Precisión |
|--------|-------------|--------------|-----------|
| Selectolax | 45 | 2.1 | 98% |
| LXML | 85 | 3.2 | 99% |
| BeautifulSoup | 150 | 4.5 | 95% |

### Comparación de Serialización

| Método | Tiempo (ms) | Tamaño (KB) |
|--------|-------------|-------------|
| OrJSON | 8 | 45 |
| UJSON | 12 | 48 |
| JSON | 25 | 52 |

### Comparación de Compresión

| Método | Ratio | Tiempo (ms) |
|--------|-------|-------------|
| Zstandard | 75% | 15 |
| LZ4 | 65% | 8 |
| Gzip | 60% | 25 |

## 🤝 Contribución

### Estructura del Proyecto

```
seo/
├── core/           # Componentes fundamentales
├── services/       # Servicios principales
├── api/           # Capa de API
├── utils/         # Utilidades
├── tests/         # Tests
└── docs/          # Documentación
```

### Guías de Contribución

1. **Fork** el repositorio
2. **Crea** una rama para tu feature (`git checkout -b feature/amazing-feature`)
3. **Commit** tus cambios (`git commit -m 'Add amazing feature'`)
4. **Push** a la rama (`git push origin feature/amazing-feature`)
5. **Abre** un Pull Request

### Estándares de Código

```bash
# Formatear código
black seo/
isort seo/

# Linting
flake8 seo/
mypy seo/

# Tests
pytest tests/ --cov=seo
```

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo `LICENSE` para más detalles.

## 🆘 Soporte

- **Documentación**: [docs/](docs/)
- **Issues**: [GitHub Issues](https://github.com/your-repo/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-repo/discussions)
- **Email**: support@example.com

## 🙏 Agradecimientos

- **Selectolax**: Parser HTML ultra-rápido
- **OrJSON**: Serialización JSON de máxima velocidad
- **Zstandard**: Compresión eficiente
- **HTTPX**: Cliente HTTP moderno
- **LangChain**: Framework de IA
- **FastAPI**: Framework web moderno 