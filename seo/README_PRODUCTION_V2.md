# Ultra-Optimized SEO Service v2.0 - Production Guide

## 🚀 Overview

Ultra-Optimized SEO Service v2.0 es una API de análisis SEO de alto rendimiento diseñada para entornos de producción con las siguientes características:

- **Arquitectura modular** con separación clara de responsabilidades
- **Librerías ultra-optimizadas** para máximo rendimiento
- **Multi-level caching** (memoria, Redis, disco)
- **HTTP/2 support** con connection pooling
- **Rate limiting** y circuit breaker
- **Métricas completas** con Prometheus y Grafana
- **Logging estructurado** con Elasticsearch y Kibana
- **SSL/TLS** con certificados automáticos
- **Health checks** y auto-recovery
- **Docker Compose** para despliegue completo

## 📋 Requisitos del Sistema

### Mínimos
- **CPU**: 2 cores
- **RAM**: 4GB
- **Disco**: 20GB libre
- **OS**: Linux (Ubuntu 20.04+ recomendado)

### Recomendados
- **CPU**: 4+ cores
- **RAM**: 8GB+
- **Disco**: 50GB+ SSD
- **Red**: 100Mbps+

### Software Requerido
- Docker 20.10+
- Docker Compose 2.0+
- Git

## 🛠️ Instalación Rápida

### 1. Clonar Repositorio
```bash
git clone <repository-url>
cd seo-service
```

### 2. Desplegar Automáticamente
```bash
chmod +x deploy_production_v2.sh
./deploy_production_v2.sh
```

### 3. Verificar Despliegue
```bash
curl http://localhost:8000/health
```

## 🔧 Configuración Manual

### 1. Configurar Variables de Entorno
```bash
cp .env.production.example .env.production
# Editar .env.production con tus valores
```

### 2. Construir y Desplegar
```bash
# Construir imágenes
docker-compose -f docker-compose.production_v2.yml build

# Desplegar servicios
docker-compose -f docker-compose.production_v2.yml up -d

# Verificar estado
docker-compose -f docker-compose.production_v2.yml ps
```

## 📊 Monitoreo y Métricas

### Grafana Dashboard
- **URL**: http://localhost:3000
- **Usuario**: admin
- **Contraseña**: Generada automáticamente

### Métricas Disponibles
- **Request Rate**: Requests por segundo
- **Response Time**: Tiempo de respuesta promedio
- **Error Rate**: Tasa de errores
- **Cache Hit Ratio**: Efectividad del cache
- **Memory Usage**: Uso de memoria
- **CPU Usage**: Uso de CPU
- **Active Connections**: Conexiones activas

### Prometheus
- **URL**: http://localhost:9090
- **Métricas**: SEO analysis, HTTP requests, cache performance

### Kibana
- **URL**: http://localhost:5601
- **Logs**: Logs estructurados de todos los servicios

## 🔌 API Endpoints

### Análisis SEO Individual
```bash
curl -X POST "http://localhost:8000/analyze" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://example.com",
    "force_refresh": false,
    "include_recommendations": true
  }'
```

### Análisis en Lote
```bash
curl -X POST "http://localhost:8000/analyze/batch" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "urls": [
      "https://example1.com",
      "https://example2.com"
    ],
    "max_concurrent": 5
  }'
```

### Health Check
```bash
curl http://localhost:8000/health
```

### Métricas
```bash
curl http://localhost:8000/metrics
```

### Estadísticas
```bash
curl -H "Authorization: Bearer YOUR_TOKEN" \
  http://localhost:8000/stats
```

## 🔐 Seguridad

### Autenticación
- **JWT Tokens**: Autenticación basada en tokens
- **Rate Limiting**: 200 requests/minuto por IP
- **CORS**: Configurable por dominio

### SSL/TLS
- **Certificados**: Auto-generados o personalizados
- **HTTPS**: Forzado en producción
- **Security Headers**: HSTS, CSP, etc.

### Variables de Entorno Sensibles
```bash
JWT_SECRET_KEY=your-secret-key
POSTGRES_PASSWORD=your-db-password
GRAFANA_PASSWORD=your-grafana-password
```

## 📈 Optimizaciones de Rendimiento

### Cache Multi-Nivel
1. **Memoria**: LRU cache con TTL
2. **Redis**: Cache distribuido
3. **Disco**: Persistencia de datos

### HTTP Client Optimizado
- **HTTP/2**: Soporte nativo
- **Connection Pooling**: Reutilización de conexiones
- **Keep-Alive**: Conexiones persistentes
- **Compression**: Gzip/Brotli automático

### Parser Ultra-Optimizado
- **Selectolax**: Parser más rápido
- **LXML**: Fallback robusto
- **Compression**: Zstandard para datos grandes

### Base de Datos
- **PostgreSQL**: Configuración optimizada
- **Connection Pooling**: PgBouncer
- **Indexing**: Índices automáticos

## 🔄 Operaciones de Mantenimiento

### Backup
```bash
# Backup de base de datos
docker-compose -f docker-compose.production_v2.yml exec postgres \
  pg_dump -U seo_user seo_service > backup_$(date +%Y%m%d_%H%M%S).sql

# Backup de logs
tar -czf logs_backup_$(date +%Y%m%d_%H%M%S).tar.gz logs/
```

### Logs
```bash
# Ver logs en tiempo real
docker-compose -f docker-compose.production_v2.yml logs -f

# Ver logs de servicio específico
docker-compose -f docker-compose.production_v2.yml logs -f seo-service
```

### Actualización
```bash
# Actualizar código
git pull origin main

# Reconstruir y desplegar
docker-compose -f docker-compose.production_v2.yml down
docker-compose -f docker-compose.production_v2.yml build --no-cache
docker-compose -f docker-compose.production_v2.yml up -d
```

### Escalado
```bash
# Escalar servicio SEO
docker-compose -f docker-compose.production_v2.yml up -d --scale seo-service=3

# Escalar Redis
docker-compose -f docker-compose.production_v2.yml up -d --scale redis=2
```

## 🚨 Troubleshooting

### Problemas Comunes

#### Servicio no inicia
```bash
# Verificar logs
docker-compose -f docker-compose.production_v2.yml logs seo-service

# Verificar recursos
docker stats

# Verificar puertos
netstat -tulpn | grep :8000
```

#### Performance lenta
```bash
# Verificar métricas
curl http://localhost:8000/metrics

# Verificar cache
curl -H "Authorization: Bearer YOUR_TOKEN" http://localhost:8000/stats

# Verificar base de datos
docker-compose -f docker-compose.production_v2.yml exec postgres \
  psql -U seo_user -d seo_service -c "SELECT * FROM pg_stat_activity;"
```

#### Errores de memoria
```bash
# Aumentar límites de memoria
docker-compose -f docker-compose.production_v2.yml down
export COMPOSE_DOCKER_CLI_BUILD=1
export DOCKER_BUILDKIT=1
docker-compose -f docker-compose.production_v2.yml up -d
```

### Logs de Debug
```bash
# Habilitar debug
export DEBUG=true
docker-compose -f docker-compose.production_v2.yml up -d

# Ver logs detallados
docker-compose -f docker-compose.production_v2.yml logs -f --tail=100
```

## 📚 Documentación Adicional

### Arquitectura
- [Arquitectura del Sistema](ARCHITECTURE.md)
- [Optimizaciones de Rendimiento](PERFORMANCE.md)
- [Guía de Seguridad](SECURITY.md)

### Desarrollo
- [Guía de Desarrollo](DEVELOPMENT.md)
- [API Reference](API_REFERENCE.md)
- [Testing Guide](TESTING.md)

### Operaciones
- [Guía de Monitoreo](MONITORING.md)
- [Guía de Backup](BACKUP.md)
- [Guía de Escalado](SCALING.md)

## 🤝 Soporte

### Comunidad
- **Issues**: GitHub Issues
- **Discussions**: GitHub Discussions
- **Wiki**: Documentación colaborativa

### Contacto
- **Email**: support@seo-service.com
- **Slack**: #seo-service-support
- **Documentación**: https://docs.seo-service.com

## 📄 Licencia

Este proyecto está licenciado bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para detalles.

---

**Versión**: 2.0.0  
**Última actualización**: 2024  
**Mantenido por**: SEO Service Team 