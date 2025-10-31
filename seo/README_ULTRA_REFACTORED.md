# SEO Service - Ultra-Optimized & Refactored

## 🚀 Servicio SEO Ultra-Optimizado con Arquitectura Limpia

Este servicio SEO ha sido completamente refactorizado siguiendo los principios de **Clean Architecture** y **Domain-Driven Design**, utilizando las librerías más rápidas disponibles y optimizaciones de rendimiento de nivel empresarial.

## 📋 Tabla de Contenidos

- [Características](#-características)
- [Arquitectura](#-arquitectura)
- [Instalación](#-instalación)
- [Configuración](#-configuración)
- [Uso](#-uso)
- [API](#-api)
- [Testing](#-testing)
- [Despliegue](#-despliegue)
- [Monitoreo](#-monitoreo)
- [Troubleshooting](#-troubleshooting)

## ✨ Características

### 🏗️ Arquitectura Limpia
- **Separación de responsabilidades** clara
- **Inversión de dependencias** con interfaces
- **Inyección de dependencias** automática
- **Testing** simplificado y completo

### ⚡ Ultra-Optimización
- **Selectolax** para parsing HTML ultra-rápido
- **Orjson** para serialización JSON más rápida
- **Zstandard** para compresión de datos
- **Httpx** con HTTP/2 y connection pooling
- **Cache multi-nivel** (memoria + Redis)

### 🤖 IA y ML
- **OpenAI GPT-4** para análisis avanzado
- **LangChain** para prompts estructurados
- **Transformers** para análisis local
- **Sentiment analysis** automático

### 📊 Monitoreo Avanzado
- **Prometheus** para métricas
- **Grafana** para visualización
- **Structured logging** con Loguru
- **Health checks** automáticos

## 🏛️ Arquitectura

```
┌─────────────────────────────────────────────────────────────┐
│                        API Layer                            │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐           │
│  │   Routes    │ │ Middleware  │ │ Validation  │           │
│  └─────────────┘ └─────────────┘ └─────────────┘           │
└─────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────┐
│                    Services Layer                           │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐           │
│  │ SEO Service │ │Batch Service│ │Selenium Svc │           │
│  └─────────────┘ └─────────────┘ └─────────────┘           │
└─────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────┐
│                     Core Layer                              │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐           │
│  │   Parser    │ │    Cache    │ │ HTTP Client │           │
│  └─────────────┘ └─────────────┘ └─────────────┘           │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐           │
│  │  Analyzer   │ │  Interfaces │ │   Models    │           │
│  └─────────────┘ └─────────────┘ └─────────────┘           │
└─────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────┐
│                 Infrastructure Layer                        │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐           │
│  │   Redis     │ │   Database  │ │   External  │           │
│  └─────────────┘ └─────────────┘ └─────────────┘           │
└─────────────────────────────────────────────────────────────┘
```

### Principios de Diseño

1. **Dependency Inversion**: Las capas superiores no dependen de las inferiores
2. **Single Responsibility**: Cada clase tiene una responsabilidad única
3. **Open/Closed**: Abierto para extensión, cerrado para modificación
4. **Interface Segregation**: Interfaces pequeñas y específicas
5. **Dependency Injection**: Dependencias inyectadas externamente

## 🛠️ Instalación

### Requisitos Previos

```bash
# Python 3.11+
python --version

# Docker y Docker Compose
docker --version
docker-compose --version

# Redis (opcional, para cache distribuido)
redis-server --version
```

### Instalación Rápida

```bash
# Clonar repositorio
git clone <repository-url>
cd seo-service

# Instalar dependencias
pip install -r requirements.ultra_optimized_v3.txt

# Configurar variables de entorno
cp .env.example .env
# Editar .env con tus configuraciones

# Ejecutar tests
python -m pytest tests/

# Iniciar servicio
python main.py
```

### Instalación con Docker

```bash
# Construir imagen
docker build -f Dockerfile.production -t seo-service:latest .

# Ejecutar con Docker Compose
docker-compose -f docker-compose.production.yml up -d

# Verificar estado
docker-compose -f docker-compose.production.yml ps
```

## ⚙️ Configuración

### Configuración Básica

```yaml
# config/production.yml
app:
  name: "SEO Service Ultra-Optimized"
  version: "2.0.0"
  environment: "production"
  debug: false

server:
  host: "0.0.0.0"
  port: 8000
  workers: 4
  timeout: 30

parser:
  type: "selectolax"
  fallback: "lxml"
  timeout: 10.0
  max_size: "50MB"
  enable_compression: true

cache:
  type: "ultra_optimized"
  size: 5000
  ttl: 7200
  compression_level: 3
  redis_url: "redis://localhost:6379/0"

http_client:
  rate_limit: 200
  max_connections: 200
  timeout: 15.0
  enable_http2: true
  retry_attempts: 3

analyzer:
  openai_api_key: "${OPENAI_API_KEY}"
  openai_model: "gpt-4-turbo-preview"
  openai_temperature: 0.1
  openai_max_tokens: 4000
  enable_keyword_analysis: true
  enable_sentiment_analysis: true
  enable_readability_analysis: true

selenium:
  headless: true
  timeout: 30
  window_size: "1920x1080"
  user_agent: "SEO-Bot/2.0"
```

### Variables de Entorno

```bash
# .env
# OpenAI
OPENAI_API_KEY=your_openai_api_key_here

# Redis
REDIS_URL=redis://localhost:6379/0
REDIS_PASSWORD=your_redis_password

# Database
DATABASE_URL=postgresql://user:pass@localhost:5432/seo_db

# Monitoring
PROMETHEUS_PORT=9090
GRAFANA_PORT=3000

# Security
SECRET_KEY=your_secret_key_here
API_KEY=your_api_key_here
```

## 🚀 Uso

### Uso Básico

```python
from services.seo_service_factory import get_seo_service

# Obtener servicio
seo_service = get_seo_service()

# Analizar URL
result = await seo_service.scrape("https://example.com")
print(f"Score SEO: {result.analysis.score}")
print(f"Recomendaciones: {result.analysis.recommendations}")
```

### Uso Avanzado

```python
from services.seo_service_factory import SEOServiceFactory

# Crear factory con configuración personalizada
config = {
    'analyzer': {
        'openai_model': 'gpt-4',
        'enable_sentiment_analysis': True
    },
    'cache': {
        'ttl': 3600,
        'compression_level': 5
    }
}

factory = SEOServiceFactory(config)
seo_service = factory.get_seo_service()

# Análisis en lote
urls = ["https://example1.com", "https://example2.com"]
results = await seo_service.batch_analyze(urls)

# Análisis comparativo
comparison = await seo_service.compare_urls(urls)
```

### Uso con Selenium

```python
# Para páginas con JavaScript
result = await seo_service.scrape_with_selenium(
    "https://spa.example.com",
    wait_for_element=".content",
    timeout=30
)
```

## 📡 API

### Endpoints Principales

#### POST /api/v2/seo/analyze
Analiza una URL para SEO.

```bash
curl -X POST "http://localhost:8000/api/v2/seo/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://example.com",
    "options": {
      "include_keywords": true,
      "include_sentiment": true,
      "use_selenium": false
    }
  }'
```

**Respuesta:**
```json
{
  "success": true,
  "data": {
    "url": "https://example.com",
    "title": "Example Domain",
    "meta_description": "This domain is for use in illustrative examples...",
    "analysis": {
      "score": 85.5,
      "technical_score": 90.0,
      "content_score": 82.0,
      "user_experience_score": 88.0,
      "recommendations": [
        "Mejorar la densidad de keywords",
        "Optimizar meta descripción"
      ],
      "issues": [
        "Falta de headers H2"
      ],
      "strengths": [
        "Título optimizado",
        "Meta descripción clara"
      ]
    },
    "keywords": {
      "primary": ["example", "domain"],
      "density": {"example": 2.5, "domain": 1.8}
    },
    "processing_time": 0.85
  }
}
```

#### POST /api/v2/seo/batch
Análisis en lote de múltiples URLs.

```bash
curl -X POST "http://localhost:8000/api/v2/seo/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "urls": [
      "https://example1.com",
      "https://example2.com",
      "https://example3.com"
    ],
    "options": {
      "max_concurrent": 5,
      "timeout": 30
    }
  }'
```

#### POST /api/v2/seo/compare
Compara múltiples URLs.

```bash
curl -X POST "http://localhost:8000/api/v2/seo/compare" \
  -H "Content-Type: application/json" \
  -d '{
    "urls": [
      "https://competitor1.com",
      "https://competitor2.com",
      "https://mysite.com"
    ]
  }'
```

#### GET /api/v2/health
Estado de salud del servicio.

```bash
curl "http://localhost:8000/api/v2/health"
```

#### GET /api/v2/metrics
Métricas de rendimiento.

```bash
curl "http://localhost:8000/api/v2/metrics"
```

### Middleware

#### Rate Limiting
```python
# Configuración automática
RATE_LIMIT = "200/minute"
RATE_LIMIT_BY_IP = "50/minute"
```

#### CORS
```python
# Configuración automática
CORS_ORIGINS = ["https://yourdomain.com"]
CORS_METHODS = ["GET", "POST", "PUT", "DELETE"]
```

#### Authentication
```python
# API Key authentication
API_KEY_HEADER = "X-API-Key"
```

## 🧪 Testing

### Tests Unitarios

```bash
# Ejecutar todos los tests
python -m pytest tests/

# Tests específicos
python -m pytest tests/test_parser.py
python -m pytest tests/test_cache.py
python -m pytest tests/test_analyzer.py

# Con coverage
python -m pytest --cov=core --cov=services --cov=api tests/
```

### Tests de Integración

```bash
# Tests de integración
python -m pytest tests/integration/

# Tests de API
python -m pytest tests/api/

# Tests de rendimiento
python -m pytest tests/performance/
```

### Tests de Rendimiento

```bash
# Benchmark del parser
python -m pytest tests/performance/test_parser_benchmark.py

# Benchmark del cache
python -m pytest tests/performance/test_cache_benchmark.py

# Benchmark completo
python -m pytest tests/performance/test_full_benchmark.py
```

### Ejemplo de Test

```python
# tests/test_seo_service.py
import pytest
from services.seo_service_factory import get_seo_service

@pytest.mark.asyncio
async def test_seo_analysis():
    service = get_seo_service()
    
    result = await service.scrape("https://example.com")
    
    assert result.success
    assert result.data.title
    assert result.analysis.score >= 0
    assert result.analysis.score <= 100
    assert len(result.analysis.recommendations) >= 0
```

## 🚀 Despliegue

### Despliegue con Docker

```bash
# Construir imagen de producción
docker build -f Dockerfile.production -t seo-service:latest .

# Ejecutar stack completo
docker-compose -f docker-compose.production.yml up -d

# Verificar servicios
docker-compose -f docker-compose.production.yml ps

# Ver logs
docker-compose -f docker-compose.production.yml logs -f seo-api
```

### Despliegue con Kubernetes

```bash
# Aplicar configuración
kubectl apply -f k8s/

# Verificar pods
kubectl get pods -l app=seo-service

# Escalar
kubectl scale deployment seo-service --replicas=5
```

### Despliegue Automatizado

```bash
# Script de despliegue
./scripts/deploy_ultra_optimized.sh

# Verificar despliegue
./scripts/health_check.sh
```

## 📊 Monitoreo

### Métricas Prometheus

```bash
# Acceder a métricas
curl http://localhost:9090/metrics

# Métricas específicas
curl http://localhost:9090/metrics | grep seo_
```

### Dashboard Grafana

```bash
# Acceder a Grafana
http://localhost:3000

# Credenciales por defecto
# Usuario: admin
# Contraseña: admin
```

### Logs Estructurados

```python
# Configuración de logging
import loguru

loguru.logger.add(
    "logs/seo-service.log",
    rotation="100 MB",
    retention="30 days",
    compression="zip",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}"
)
```

### Health Checks

```bash
# Health check básico
curl http://localhost:8000/health

# Health check detallado
curl http://localhost:8000/health/detailed

# Health check de componentes
curl http://localhost:8000/health/components
```

## 🔧 Troubleshooting

### Problemas Comunes

#### 1. Error de OpenAI API
```bash
# Verificar API key
echo $OPENAI_API_KEY

# Verificar límites de rate
curl -H "Authorization: Bearer $OPENAI_API_KEY" \
  https://api.openai.com/v1/usage
```

#### 2. Error de Redis
```bash
# Verificar conexión Redis
redis-cli ping

# Verificar memoria Redis
redis-cli info memory
```

#### 3. Error de Selenium
```bash
# Verificar Chrome/ChromeDriver
google-chrome --version
chromedriver --version

# Verificar permisos
ls -la /usr/bin/chromedriver
```

#### 4. Error de Rendimiento
```bash
# Verificar métricas
curl http://localhost:8000/metrics

# Verificar logs
tail -f logs/seo-service.log

# Verificar recursos del sistema
htop
```

### Debugging

#### Modo Debug
```bash
# Habilitar debug
export DEBUG=true
export LOG_LEVEL=DEBUG

# Reiniciar servicio
docker-compose restart seo-api
```

#### Profiling
```python
# Profiling con cProfile
python -m cProfile -o profile.stats main.py

# Analizar resultados
python -c "
import pstats
p = pstats.Stats('profile.stats')
p.sort_stats('cumulative').print_stats(20)
"
```

### Recuperación

#### Restart Automático
```bash
# Configurar restart automático
docker-compose -f docker-compose.production.yml up -d --restart unless-stopped
```

#### Rollback
```bash
# Rollback a versión anterior
docker-compose -f docker-compose.production.yml down
docker tag seo-service:previous seo-service:latest
docker-compose -f docker-compose.production.yml up -d
```

## 📚 Documentación Adicional

- [REFACTOR_ULTRA_OPTIMIZED.md](REFACTOR_ULTRA_OPTIMIZED.md) - Detalles del refactor
- [ULTRA_OPTIMIZATION_SUMMARY.md](ULTRA_OPTIMIZATION_SUMMARY.md) - Resumen de optimizaciones
- [PRODUCTION_ULTRA_OPTIMIZED.md](PRODUCTION_ULTRA_OPTIMIZED.md) - Guía de producción
- [ULTRA_OPTIMIZATION_LIBRARIES.md](ULTRA_OPTIMIZATION_LIBRARIES.md) - Librerías utilizadas

## 🤝 Contribución

1. Fork el proyecto
2. Crear feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push al branch (`git push origin feature/AmazingFeature`)
5. Abrir Pull Request

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo [LICENSE](LICENSE) para más detalles.

## 🆘 Soporte

- **Issues**: [GitHub Issues](https://github.com/your-repo/issues)
- **Documentación**: [Wiki](https://github.com/your-repo/wiki)
- **Email**: support@yourdomain.com

---

**SEO Service Ultra-Optimized v2.0.0** - Arquitectura limpia, rendimiento máximo, código de calidad empresarial. 