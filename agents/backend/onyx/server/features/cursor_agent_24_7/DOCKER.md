# 🐳 Docker - Cursor Agent 24/7

Guía completa para usar Docker y Docker Compose.

## 🚀 Inicio Rápido

### Desarrollo

```bash
# Iniciar stack completo
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up

# O solo servicios básicos
docker-compose up api redis worker
```

### Producción

```bash
# Crear archivo .env con variables
cp .env.example .env

# Editar .env con valores de producción
# JWT_SECRET_KEY=your-secret-key
# REDIS_PASSWORD=your-redis-password
# AWS_REGION=us-east-1

# Iniciar
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

## 📦 Servicios Disponibles

### Servicios Principales

1. **api** - FastAPI Application
   - Puerto: 8024
   - Health: http://localhost:8024/api/health

2. **worker** - Celery Worker
   - Procesa tareas en background
   - Queues: default, tasks, heavy, notifications

3. **beat** - Celery Beat Scheduler
   - Tareas programadas

4. **redis** - Redis Cache & Broker
   - Puerto: 6379
   - Usado para cache y Celery broker

### Servicios Opcionales

5. **flower** - Celery Monitor
   - Puerto: 5555
   - UI: http://localhost:5555
   - Iniciar: `docker-compose --profile monitoring up flower`

6. **prometheus** - Métricas
   - Puerto: 9090
   - UI: http://localhost:9090
   - Iniciar: `docker-compose --profile monitoring up prometheus`

7. **grafana** - Dashboards
   - Puerto: 3000
   - UI: http://localhost:3000
   - Usuario: admin / Password: admin (cambiar en producción)
   - Iniciar: `docker-compose --profile monitoring up grafana`

8. **rabbitmq** - Message Broker
   - AMQP: 5672
   - Management UI: http://localhost:15672
   - Iniciar: `docker-compose --profile message-broker up rabbitmq`

9. **nginx** - Reverse Proxy
   - HTTP: 80
   - HTTPS: 443
   - Iniciar: `docker-compose --profile reverse-proxy up nginx`

## 🔧 Comandos Útiles

### Desarrollo

```bash
# Iniciar todo
docker-compose up

# Iniciar en background
docker-compose up -d

# Ver logs
docker-compose logs -f api
docker-compose logs -f worker

# Rebuild después de cambios
docker-compose build --no-cache api
docker-compose up -d api

# Ejecutar comando en contenedor
docker-compose exec api python cli.py version
docker-compose exec worker celery -A core.celery_worker.celery_app inspect active

# Detener todo
docker-compose down

# Detener y eliminar volúmenes
docker-compose down -v
```

### Producción

```bash
# Iniciar con replicas
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# Escalar workers
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d --scale worker=5

# Ver estado
docker-compose ps

# Logs
docker-compose logs -f --tail=100

# Actualizar servicio
docker-compose pull
docker-compose up -d --no-deps --build api
```

## 📊 Monitoreo

### Flower (Celery Monitor)

```bash
# Iniciar
docker-compose --profile monitoring up -d flower

# Acceder
open http://localhost:5555
```

### Prometheus + Grafana

```bash
# Iniciar stack de monitoreo
docker-compose --profile monitoring up -d prometheus grafana

# Prometheus
open http://localhost:9090

# Grafana
open http://localhost:3000
# Usuario: admin / Password: admin
```

## 🔐 Seguridad

### Variables de Entorno

Crear `.env`:

```bash
# Security
JWT_SECRET_KEY=your-very-secure-secret-key-change-in-production
REDIS_PASSWORD=your-redis-password

# AWS
AWS_REGION=us-east-1
DYNAMODB_TABLE_NAME=cursor-agent-state

# Observability
OTEL_EXPORTER_OTLP_ENDPOINT=http://otel-collector:4317

# RabbitMQ (opcional)
RABBITMQ_USER=admin
RABBITMQ_PASS=secure-password

# Grafana (opcional)
GRAFANA_PASSWORD=secure-password
```

### SSL/TLS con Nginx

```bash
# Generar certificados (desarrollo)
mkdir -p docker/nginx/ssl
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout docker/nginx/ssl/key.pem \
  -out docker/nginx/ssl/cert.pem

# Iniciar con nginx
docker-compose --profile reverse-proxy up -d nginx
```

## 🏗️ Arquitectura Docker

```
┌─────────────────────────────────────────┐
│           Nginx (Reverse Proxy)         │
│         (Rate Limiting, SSL)            │
└──────────────────┬──────────────────────┘
                   │
        ┌──────────┼──────────┐
        │          │          │
        ▼          ▼          ▼
    ┌───────┐  ┌───────┐  ┌───────┐
    │  API  │  │Worker │  │  Beat │
    │ (x2)  │  │ (x3)  │  │ (x1)  │
    └───┬───┘  └───┬───┘  └───┬───┘
        │          │          │
        └──────────┼──────────┘
                   │
                   ▼
            ┌──────────┐
            │  Redis   │
            └──────────┘
```

## 📈 Escalabilidad

### Escalar Workers

```bash
# Escalar a 5 workers
docker-compose up -d --scale worker=5

# Escalar API
docker-compose up -d --scale api=3
```

### Load Balancing

Nginx automáticamente balancea carga entre múltiples instancias de API.

## 🔄 Actualización

### Actualizar un Servicio

```bash
# Rebuild y restart
docker-compose build api
docker-compose up -d --no-deps api

# Rolling update (producción)
docker-compose up -d --no-deps --scale api=2 api
# Esperar que nuevas instancias estén listas
docker-compose up -d --scale api=1 api
```

### Actualizar Todo

```bash
# Pull latest images
docker-compose pull

# Rebuild
docker-compose build

# Restart
docker-compose up -d
```

## 🐛 Troubleshooting

### Ver Logs

```bash
# Todos los servicios
docker-compose logs -f

# Servicio específico
docker-compose logs -f api
docker-compose logs -f worker

# Últimas 100 líneas
docker-compose logs --tail=100 api
```

### Debugging

```bash
# Entrar al contenedor
docker-compose exec api bash
docker-compose exec worker bash

# Ejecutar comandos
docker-compose exec api python cli.py health
docker-compose exec worker celery -A core.celery_worker.celery_app inspect active

# Ver procesos
docker-compose exec api ps aux
```

### Limpiar

```bash
# Detener y eliminar contenedores
docker-compose down

# Eliminar también volúmenes
docker-compose down -v

# Eliminar imágenes
docker-compose down --rmi all

# Limpiar todo (cuidado!)
docker system prune -a
```

## 📚 Más Información

- [Dockerfile](Dockerfile) - Imagen Docker
- [ARCHITECTURE.md](ARCHITECTURE.md) - Arquitectura completa
- [AWS_DEPLOYMENT.md](aws/AWS_DEPLOYMENT.md) - Despliegue en AWS




