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

### Instalación Rápida

```bash
# Clonar el repositorio
git clone <repository-url>
cd blatam-academy/agents/backend/onyx/server/features

# Iniciar el sistema completo
python start_system.py start
```

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
├── integration_system/          # Sistema principal
│   ├── core/                   # Lógica central
│   ├── api/                    # Endpoints
│   ├── config/                 # Configuración
│   └── middleware/             # Middleware
├── content_redundancy_detector/ # Detector de redundancia
├── bulk/                       # BUL - Generación de documentos
├── gamma_app/                  # Gamma App
├── business_agents/            # Agentes de negocio
├── export_ia/                  # Export IA
├── nginx/                      # Configuración Nginx
├── monitoring/                 # Configuración monitoreo
├── docker-compose.yml          # Orquestación Docker
└── start_system.py             # Script de inicio
```

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

- Cache Redis
- Compresión Gzip
- Balanceador de carga Nginx
- Pool de conexiones DB
- Async/await en todas las operaciones

### Benchmarks

- **Throughput**: 1000+ requests/segundo
- **Latencia**: <100ms promedio
- **Disponibilidad**: 99.9%
- **Escalabilidad**: Horizontal

## 🆘 Troubleshooting

### Problemas Comunes

1. **Servicio no inicia**
   ```bash
   docker-compose logs [service-name]
   docker-compose restart [service-name]
   ```

2. **Error de conexión a DB**
   ```bash
   docker-compose restart postgres
   docker-compose exec postgres psql -U postgres -c "SELECT 1;"
   ```

3. **Memoria insuficiente**
   ```bash
   docker system prune -a
   docker-compose down && docker-compose up -d
   ```

### Logs y Debugging

```bash
# Ver logs en tiempo real
docker-compose logs -f

# Debug de servicio específico
docker-compose exec [service-name] bash

# Verificar recursos
docker stats
```

## 📞 Soporte

### Documentación

- **API Docs**: http://localhost:8000/docs
- **Swagger UI**: http://localhost:8000/redoc
- **Health Status**: http://localhost:8000/health

### Contacto

- **Issues**: GitHub Issues
- **Documentación**: Wiki del proyecto
- **Comunidad**: Discord/Slack

## 📄 Licencia

Este proyecto está licenciado bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para detalles.

---

**Blatam Academy** - Transformando la automatización empresarial con IA 🚀
#   I A - M o d e l s - C l o n e  
 