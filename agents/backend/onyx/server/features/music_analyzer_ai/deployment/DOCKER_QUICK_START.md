# Docker Quick Start Guide

## 🚀 Inicio Rápido (5 minutos)

### Opción 1: Desarrollo

```bash
# 1. Construir imagen
docker build -f deployment/Dockerfile -t music-analyzer-ai:latest ..

# 2. Iniciar servicios
docker-compose -f deployment/docker-compose.dev.yml up -d

# 3. Verificar
curl http://localhost:8010/health
```

### Opción 2: Producción

```bash
# 1. Crear archivo .env
cat > .env << EOF
ENVIRONMENT=production
SPOTIFY_CLIENT_ID=tu_client_id
SPOTIFY_CLIENT_SECRET=tu_client_secret
POSTGRES_PASSWORD=password_seguro
REDIS_PASSWORD=password_seguro
EOF

# 2. Construir e iniciar
docker-compose -f deployment/docker-compose.prod.yml up -d --build
```

### Opción 3: Usando Make (recomendado)

```bash
# Desarrollo
make dev

# Producción
make prod

# Ver logs
make logs

# Detener
make down
```

## 📋 Comandos Útiles

### Gestión de Contenedores

```bash
# Ver contenedores en ejecución
docker ps

# Ver logs
docker-compose logs -f music-analyzer-ai

# Detener servicios
docker-compose down

# Reiniciar un servicio
docker-compose restart music-analyzer-ai

# Ejecutar comando en contenedor
docker-compose exec music-analyzer-ai bash
```

### Construcción

```bash
# Construir imagen
docker build -f deployment/Dockerfile -t music-analyzer-ai:latest ..

# Construir sin cache
docker build --no-cache -f deployment/Dockerfile -t music-analyzer-ai:latest ..

# Ver tamaño de imagen
docker images music-analyzer-ai
```

### Limpieza

```bash
# Limpiar contenedores detenidos
docker container prune

# Limpiar imágenes no usadas
docker image prune

# Limpiar todo (cuidado!)
docker system prune -a
```

## 🔧 Configuración

### Variables de Entorno

Crea un archivo `.env` en la raíz del proyecto:

```env
ENVIRONMENT=production
SPOTIFY_CLIENT_ID=tu_client_id
SPOTIFY_CLIENT_SECRET=tu_client_secret
LOG_LEVEL=INFO
CACHE_ENABLED=true
POSTGRES_PASSWORD=password_seguro
REDIS_PASSWORD=password_seguro
```

### Puertos

- **8010**: API principal
- **80/443**: Nginx (producción)
- **6379**: Redis
- **5432**: PostgreSQL
- **3000**: Grafana
- **9090**: Prometheus

## 📊 Monitoreo

### Health Checks

```bash
# Health básico
curl http://localhost:8010/health

# Health detallado
curl http://localhost:8010/health/detailed

# Readiness
curl http://localhost:8010/health/ready

# Liveness
curl http://localhost:8010/health/live
```

### Dashboards

- **Grafana**: http://localhost:3000 (admin/admin)
- **Prometheus**: http://localhost:9090

## 🐛 Troubleshooting

### El contenedor no inicia

```bash
# Ver logs
docker logs music-analyzer-ai

# Ver logs de compose
docker-compose logs music-analyzer-ai
```

### Puerto en uso

```bash
# Cambiar puerto en docker-compose.yml
ports:
  - "8011:8010"  # Usar puerto diferente
```

### Problemas de memoria

```bash
# Aumentar límites en docker-compose.prod.yml
deploy:
  resources:
    limits:
      memory: 8G
```

## 📚 Documentación Completa

Para más detalles, consulta:
- `DOCKER.md` - Guía completa
- `README.md` - Documentación general
- `QUICK_START.md` - Inicio rápido cloud

## 🎯 Próximos Pasos

1. ✅ Configurar variables de entorno
2. ✅ Construir imagen Docker
3. ✅ Iniciar servicios
4. ✅ Verificar health checks
5. ✅ Configurar monitoreo
6. ✅ Revisar logs
7. ✅ Optimizar para producción




