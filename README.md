# Blatam Academy - Sistema Integrado Completo

## 🚀 Descripción General

El **Sistema Integrado Blatam Academy** es una plataforma completa que integra múltiples servicios de IA y automatización empresarial en una arquitectura unificada y escalable.

## 🏗️ Arquitectura del Sistema

### Servicios Integrados

1. **Integration System** (Puerta de enlace principal)
   - Puerto: 8000
   - Función: Orquestador central y API Gateway
   - Características: Enrutamiento, autenticación, monitoreo

2. **Content Redundancy Detector**
   - Puerto: 8001
   - Función: Detección de redundancia en contenido
   - Características: Análisis de similitud, evaluación de calidad

3. **BUL (Business Unlimited)**
   - Puerto: 8002
   - Función: Generación de documentos empresariales con IA
   - Características: Plantillas, automatización, múltiples formatos

4. **Gamma App**
   - Puerto: 8003
   - Función: Generación de contenido con IA
   - Características: Presentaciones, documentos, páginas web

5. **Business Agents**
   - Puerto: 8004
   - Función: Agentes de negocio automatizados
   - Características: Workflows, coordinación, automatización

6. **Export IA**
   - Puerto: 8005
   - Función: Exportación avanzada y análisis
   - Características: Múltiples formatos, analytics, validación

### Infraestructura

- **PostgreSQL**: Base de datos principal
- **Redis**: Cache y sesiones
- **Nginx**: Proxy reverso y balanceador de carga
- **Prometheus**: Métricas y monitoreo
- **Grafana**: Dashboards y visualización
- **ELK Stack**: Logs y análisis

## 🚀 Instalación y Configuración

### Prerrequisitos

- Docker y Docker Compose
- Python 3.8+
- Git
- 16GB+ RAM recomendado (8GB mínimo)
- GPU NVIDIA (opcional, recomendado para KV Cache y modelos de IA)

### Instalación Rápida

```bash
# Clonar el repositorio
git clone <repository-url>
cd blatam-academy/agents/backend/onyx/server/features

# Iniciar el sistema completo
python start_system.py start

# Ver estado
python start_system.py status
```

> 📖 **Guías Disponibles**: 
> - **[Guía de Inicio Rápido](QUICK_START_GUIDE.md)** - Para empezar en 5 minutos
> - **[Guía de Arquitectura](ARCHITECTURE_GUIDE.md)** - Arquitectura completa
> - **[Índice de Documentación](DOCUMENTATION_INDEX.md)** - Navegación completa

### Instalación Manual

```bash
# 1. Crear archivo de entorno
cp .env.example .env
# Editar .env con tus configuraciones

# 2. Construir y ejecutar servicios
docker-compose up -d --build

# 3. Verificar estado
python start_system.py status
```

## 📖 Uso del Sistema

### Acceso a Servicios

- **Sistema Principal**: http://localhost:8000
- **Documentación API**: http://localhost:8000/docs
- **Monitoreo**: http://localhost:3000 (Grafana)
- **Métricas**: http://localhost:9090 (Prometheus)
- **Logs**: http://localhost:5601 (Kibana)

### API Gateway

El sistema principal actúa como API Gateway unificado:

```bash
# Ejemplo de uso
curl -X POST "http://localhost:8000/api/v1/gateway/route" \
  -H "Content-Type: application/json" \
  -d '{
    "target_system": "content_redundancy",
    "endpoint": "/analyze",
    "method": "POST",
    "data": {"content": "Texto a analizar"}
  }'
```

### Servicios Individuales

Cada servicio también es accesible directamente:

```bash
# Content Redundancy Detector
curl -X POST "http://localhost:8001/analyze" \
  -H "Content-Type: application/json" \
  -d '{"content": "Texto a analizar"}'

# BUL - Generación de documentos
curl -X POST "http://localhost:8002/documents/generate" \
  -H "Content-Type: application/json" \
  -d '{"query": "Estrategia de marketing", "business_area": "marketing"}'

# Gamma App - Generación de contenido
curl -X POST "http://localhost:8003/api/v1/content/generate" \
  -H "Content-Type: application/json" \
  -d '{"content_type": "presentation", "topic": "El Futuro de la IA"}'

# Business Agents - Ejecutar agente
curl -X POST "http://localhost:8004/business-agents/marketing_001/capabilities/campaign_planning/execute" \
  -H "Content-Type: application/json" \
  -d '{"inputs": {"target_audience": "empresas", "budget": 10000}}'

# Export IA - Exportar documento
curl -X POST "http://localhost:8005/api/v1/exports/generate" \
  -H "Content-Type: application/json" \
  -d '{"content": {...}, "format": "pdf"}'
```

## 🔧 Configuración

### Variables de Entorno

```env
# Configuración de la aplicación
APP_NAME=Blatam Academy Integration System
DEBUG=false
HOST=0.0.0.0
PORT=8000

# Seguridad
SECRET_KEY=tu-clave-secreta-segura
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Base de datos
DATABASE_URL=postgresql://postgres:password@localhost:5432/blatam_academy

# Redis
REDIS_URL=redis://localhost:6379

# Endpoints de servicios
CONTENT_REDUNDANCY_ENDPOINT=http://localhost:8001
BUL_ENDPOINT=http://localhost:8002
GAMMA_APP_ENDPOINT=http://localhost:8003
BUSINESS_AGENTS_ENDPOINT=http://localhost:8004
EXPORT_IA_ENDPOINT=http://localhost:8005

# API Keys (opcional)
OPENAI_API_KEY=tu-openai-key
ANTHROPIC_API_KEY=tu-anthropic-key
OPENROUTER_API_KEY=tu-openrouter-key
```

### Configuración de Servicios

Cada servicio tiene su propia configuración en `config/settings.py`:

```python
# Ejemplo de configuración personalizada
system_configs = {
    "content_redundancy": {
        "max_content_length": 10485760,
        "min_content_length": 10
    },
    "bul": {
        "max_concurrent_tasks": 5,
        "template_directory": "/app/templates"
    }
}
```

## 📊 Monitoreo y Métricas

### Health Checks

```bash
# Verificar estado del sistema completo
curl http://localhost:8000/health

# Verificar servicio específico
curl http://localhost:8001/health
```

### Métricas

- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/admin)
- **Métricas personalizadas**: http://localhost:8000/api/v1/metrics

### Logs

- **Kibana**: http://localhost:5601
- **Logs en tiempo real**: `docker-compose logs -f [service-name]`

## 🛠️ Desarrollo

### Estructura del Proyecto

```
features/
├── integration_system/          # Sistema principal de integración
│   ├── core/                   # Lógica central
│   ├── api/                    # Endpoints
│   ├── config/                 # Configuración
│   └── middleware/             # Middleware
├── content_redundancy_detector/ # Detector de redundancia de contenido
├── bulk/                       # BUL - Generación de documentos empresariales
├── bulk_truthgpt/              # Sistema Bulk TruthGPT
├── gamma_app/                  # Gamma App - Generación de contenido
├── business_agents/            # Agentes de negocio automatizados
├── export_ia/                  # Export IA - Exportación avanzada
├── ads/                        # Sistema de generación de anuncios con IA
├── advanced_ai_models/         # Modelos avanzados de IA
├── ai_document_classifier/     # Clasificador de documentos con IA
├── ai_document_processor/      # Procesador de documentos con IA
├── ai_history_comparison/      # Comparación de historiales con IA
├── ai_integration_system/      # Sistema de integración de IA
├── ai_video/                   # Procesamiento de video con IA
├── blatam_ai/                  # Motor Blatam AI
├── blaze_ai/                   # Sistema Blaze AI
├── blog_posts/                 # Generación de posts para blog
├── brand_voice/                # Sistema de voz de marca
├── copywriting/                # Generación de copywriting
├── document_set/                # Gestión de conjuntos de documentos
├── document_workflow_chain/    # Cadena de flujo de trabajo documental
├── email_sequence/             # Secuencias de email automatizadas
├── enterprise/                 # Módulos empresariales
├── facebook_posts/             # Generación de posts para Facebook
├── folder/                     # Gestión de carpetas
├── heygen_ai/                  # Integración HeyGen AI
├── image_process/              # Procesamiento de imágenes
├── instagram_captions/         # Generación de captions para Instagram
├── input_prompt/               # Gestión de prompts de entrada
├── key_messages/               # Gestión de mensajes clave
├── linkedin_posts/             # Generación de posts para LinkedIn
├── notebooklm_ai/              # Integración NotebookLM AI
├── notifications/              # Sistema de notificaciones
├── os_content/                 # Contenido del sistema operativo
├── password/                   # Gestión de contraseñas
├── pdf_variantes/              # Procesamiento de variantes de PDF
├── persona/                    # Gestión de personas/perfiles
├── product_descriptions/       # Generación de descripciones de productos
├── professional_documents/     # Generación de documentos profesionales
├── seo/                        # Optimización SEO
├── tool/                       # Herramientas generales
├── ultra_extreme_v18/          # Versión ultra extrema v18
├── utils/                      # Utilidades compartidas
├── video-OpusClip/             # Procesamiento de video OpusClip
├── voice_coaching_ai/           # Coaching de voz con IA
├── Frontier-Model-run/         # Ejecución de modelos frontier
├── content_modules/             # Módulos de contenido
├── core/                       # Núcleo del sistema
├── docs/                       # Documentación
├── integrated/                 # Módulos integrados
├── microservices_framework/    # Framework de microservicios
├── nginx/                      # Configuración Nginx
├── production/                 # Configuraciones de producción
├── docker-compose.yml          # Orquestación Docker
├── docker-compose-all.yml     # Orquestación Docker completa
└── start_system.py             # Script de inicio
```

### Servicios y Módulos Principales

- **ads/**: Sistema avanzado para generación de anuncios usando modelos de difusión y transformers
- **advanced_ai_models/**: Modelos de IA avanzados con capacidades de inferencia y entrenamiento
- **ai_document_processor/**: Procesador completo de documentos con capacidades NLP y visión
- **ai_history_comparison/**: Sistema de comparación y análisis de historiales con IA
- **ai_video/**: Sistema completo de procesamiento de video con optimizaciones avanzadas
- **blatam_ai/**: Motor principal de IA con soporte para transformers, LLMs y fine-tuning
- **blaze_ai/**: Sistema Blaze AI con arquitectura modular y optimizaciones de rendimiento
- **blog_posts/**: Generación avanzada de posts para blogs con múltiples modelos
- **bulk/**: Sistema BUL para generación masiva de documentos empresariales con Ultra Adaptive KV Cache Engine
- **business_agents/**: Agentes de negocio con capacidades NLP y ML avanzadas
- **export_ia/**: Sistema de exportación con múltiples formatos y optimizaciones
- **gamma_app/**: Aplicación Gamma para generación de contenido multimedia
- **integration_system/**: Sistema principal de integración y API Gateway

### Componentes Técnicos Avanzados

#### Ultra Adaptive KV Cache Engine

El sistema **bulk/** incluye un **Ultra Adaptive KV Cache Engine** de nivel empresarial que proporciona:

- ✅ **Multi-GPU Support**: Detección automática y balanceo inteligente de carga
- ✅ **Adaptive Caching**: Múltiples políticas de evicción (LRU, LFU, FIFO, Adaptive)
- ✅ **Persistence**: Persistencia de caché en disco y checkpointing automático
- ✅ **Performance Monitoring**: Latencias P50, P95, P99, seguimiento de throughput
- ✅ **Security**: Sanitización de requests, rate limiting, control de acceso
- ✅ **Real-time Monitoring**: Dashboard en tiempo real con métricas y alertas
- ✅ **Self-Healing**: Recuperación automática de errores
- ✅ **Advanced Features**: Prefetching, deduplicación, streaming, priority queue

Para más información, consulta: [`bulk/core/README_ULTRA_ADAPTIVE_KV_CACHE.md`](bulk/core/README_ULTRA_ADAPTIVE_KV_CACHE.md)

### Agregar Nuevo Servicio

1. Crear directorio del servicio
2. Implementar API FastAPI
3. Agregar al `docker-compose.yml`
4. Configurar en `integration_system`
5. Actualizar `nginx.conf`

### Testing

```bash
# Tests unitarios
pytest tests/

# Tests de integración
pytest tests/integration/

# Tests de carga
pytest tests/load/
```

## 🚀 Despliegue

### Desarrollo

```bash
python start_system.py start
```

### Producción

```bash
# Configurar variables de entorno de producción
export ENVIRONMENT=production
export DEBUG=false
export SECRET_KEY=clave-super-segura

# Iniciar con configuración de producción
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

### Escalabilidad

```bash
# Escalar servicios específicos
docker-compose up -d --scale business-agents=3
docker-compose up -d --scale gamma-app=2
```

## 🔒 Seguridad

### Autenticación

- JWT tokens
- Rate limiting
- CORS configurado
- Headers de seguridad

### HTTPS

```bash
# Configurar SSL
cp ssl/cert.pem nginx/ssl/
cp ssl/key.pem nginx/ssl/
```

### Firewall

```bash
# Configurar firewall
ufw allow 80
ufw allow 443
ufw allow 8000:8005
```

## 📈 Performance

### Optimizaciones

- **Ultra Adaptive KV Cache**: Sistema de caché de alto rendimiento con multi-GPU
- **Cache Redis**: Caché distribuido para sesiones y datos frecuentes
- **Compresión Gzip**: Compresión de respuestas HTTP
- **Balanceador de carga Nginx**: Distribución inteligente de carga
- **Pool de conexiones DB**: Gestión optimizada de conexiones a base de datos
- **Async/await**: Operaciones asíncronas en toda la arquitectura
- **Batch Processing**: Procesamiento por lotes optimizado
- **Request Prefetching**: Prefetching inteligente basado en patrones

### Benchmarks

- **Throughput General**: 1000+ requests/segundo
- **KV Cache (Cached)**: 
  - P50: <100ms
  - P95: <500ms
  - P99: <1s
- **KV Cache (Uncached)**: 1-5s
- **Batch Processing**: 100-500 req/s
- **Concurrent Requests**: 50-200 req/s
- **Disponibilidad**: 99.9%
- **Escalabilidad**: Horizontal con auto-scaling

## 🆘 Troubleshooting

> 📖 **Guía Completa de Troubleshooting**: Para troubleshooting detallado, consulta [`TROUBLESHOOTING_GUIDE.md`](TROUBLESHOOTING_GUIDE.md)

### Problemas Comunes

1. **Servicio no inicia**
   ```bash
   docker-compose logs [service-name]
   docker-compose restart [service-name]
   
   # Verificar recursos
   docker stats
   ```

2. **Error de conexión a DB**
   ```bash
   docker-compose restart postgres
   docker-compose exec postgres psql -U postgres -c "SELECT 1;"
   
   # Verificar variables de entorno
   docker-compose exec postgres env | grep DATABASE
   ```

3. **Memoria insuficiente**
   ```bash
   docker system prune -a
   docker-compose down && docker-compose up -d
   
   # Limpiar caché del KV Cache
   python bulk/core/ultra_adaptive_kv_cache_cli.py clear-cache
   ```

4. **KV Cache - Alto uso de memoria**
   ```python
   from bulk.core.ultra_adaptive_kv_cache_config_manager import ConfigManager
   
   # Reducir tamaño de caché
   config_manager = ConfigManager(engine)
   await config_manager.update_config('cache_size', 8192)
   
   # O usar preset memory_efficient
   from bulk.core.ultra_adaptive_kv_cache_config_manager import ConfigPreset
   ConfigPreset.apply_preset(engine, 'memory_efficient')
   ```

5. **KV Cache - Bajo rendimiento**
   ```python
   # Verificar estadísticas
   stats = engine.get_stats()
   print(f"Hit rate: {stats['hit_rate']}")
   
   # Aumentar workers
   await config_manager.update_config('num_workers', 16)
   
   # Habilitar prefetching
   await config_manager.update_config('enable_prefetch', True)
   ```

6. **Servicios lentos**
   ```bash
   # Verificar logs
   docker-compose logs --tail=100 [service-name]
   
   # Verificar métricas
   curl http://localhost:9090/metrics | grep [service-name]
   
   # Reiniciar con más recursos
   docker-compose up -d --scale [service-name]=2
   ```

7. **Error de GPU en KV Cache**
   ```python
   # Verificar disponibilidad
   import torch
   print(f"CUDA available: {torch.cuda.is_available()}")
   print(f"GPU count: {torch.cuda.device_count()}")
   
   # Usar CPU si no hay GPU
   config.use_cuda = False
   ```

### Logs y Debugging

```bash
# Ver logs en tiempo real
docker-compose logs -f

# Debug de servicio específico
docker-compose exec [service-name] bash

# Verificar recursos
docker stats

# Logs del KV Cache
python bulk/core/ultra_adaptive_kv_cache_cli.py monitor --verbose

# Profiling del KV Cache
python bulk/core/ultra_adaptive_kv_cache_cli.py benchmark --duration 60
```

### Herramientas de Diagnóstico

```python
# Diagnóstico completo del KV Cache
from bulk.core.ultra_adaptive_kv_cache_health_checker import HealthChecker

health_checker = HealthChecker(engine)
diagnostic = await health_checker.run_full_diagnostic()
print(diagnostic)

# Analytics detallados
from bulk.core.ultra_adaptive_kv_cache_analytics import Analytics

analytics = Analytics(engine)
report = analytics.generate_detailed_report()
print(report)
```

## 📞 Soporte

### Documentación

- **API Docs**: http://localhost:8000/docs
- **Swagger UI**: http://localhost:8000/redoc
- **Health Status**: http://localhost:8000/health

### Guías Disponibles

- 📖 **[Guía de Inicio Rápido](QUICK_START_GUIDE.md)**: Para empezar rápidamente
- 🏗️ **[Guía de Arquitectura](ARCHITECTURE_GUIDE.md)**: Arquitectura completa del sistema
- 🚀 **[Guía de Uso Avanzado BUL](bulk/ADVANCED_USAGE_GUIDE.md)**: Uso avanzado del sistema BUL
- 🔧 **[Guía de Troubleshooting](TROUBLESHOOTING_GUIDE.md)**: Solución de problemas detallada
- 🎯 **[Mejores Prácticas](BEST_PRACTICES.md)**: Mejores prácticas y patrones
- 💼 **[Casos de Uso Reales](bulk/USE_CASES.md)**: Ejemplos prácticos de implementación
- 💡 **[Ejemplos Prácticos](bulk/EXAMPLES.md)**: Colección completa de ejemplos de código
- 📡 **[Referencia de API](API_REFERENCE.md)**: Documentación completa de APIs
- 📚 **[Índice de Documentación](DOCUMENTATION_INDEX.md)**: Navegación completa de toda la documentación
- 📊 **[Resumen Ejecutivo](SUMMARY.md)**: Resumen completo y estadísticas del sistema

### Documentación Adicional

- **BUL KV Cache**: [`bulk/core/README_ULTRA_ADAPTIVE_KV_CACHE.md`](bulk/core/README_ULTRA_ADAPTIVE_KV_CACHE.md)
- **BUL KV Cache Features**: [`bulk/core/ULTRA_ADAPTIVE_KV_CACHE_COMPLETE_FEATURES.md`](bulk/core/ULTRA_ADAPTIVE_KV_CACHE_COMPLETE_FEATURES.md)
- **BUL README**: [`bulk/README.md`](bulk/README.md)
- **Content Redundancy**: [`content_redundancy_detector/README.md`](content_redundancy_detector/README.md)
- **Business Agents**: [`business_agents/README.md`](business_agents/README.md)
- **Export IA**: [`export_ia/README.md`](export_ia/README.md)

### Herramientas CLI

```bash
# Monitorear KV Cache
python bulk/core/ultra_adaptive_kv_cache_cli.py monitor --dashboard

# Estadísticas del sistema
python bulk/core/ultra_adaptive_kv_cache_cli.py stats

# Health check completo
python bulk/core/ultra_adaptive_kv_cache_cli.py health
```

### Contacto

- **Issues**: GitHub Issues
- **Documentación**: Wiki del proyecto
- **Comunidad**: Discord/Slack

## 🤝 Contribución

¡Las contribuciones son bienvenidas! Por favor consulta nuestra [Guía de Contribución](CONTRIBUTING.md) para más detalles.

### Cómo Contribuir

1. Fork el repositorio
2. Crea un branch para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push al branch (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📝 Changelog

Ver [CHANGELOG.md](CHANGELOG.md) para historial de cambios.

## 📄 Licencia

Este proyecto está licenciado bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para detalles.

## 🙏 Agradecimientos

- Todo el equipo de desarrollo de Blatam Academy
- Contribuidores de código abierto
- Comunidad de usuarios

---

**Blatam Academy** - Transformando la automatización empresarial con IA 🚀

**Sistema Completo con:**
- ✅ 40+ módulos documentados
- ✅ Ultra Adaptive KV Cache Engine
- ✅ Documentación completa
- ✅ Guías técnicas detalladas
- ✅ Ejemplos prácticos
- ✅ Casos de uso reales
#   I A - M o d e l s - C l o n e 
 
 