# Guía de Deployment - Robot Movement AI v2.0
## Deploy en Producción Paso a Paso

---

## 📋 Tabla de Contenidos

1. [Prerrequisitos](#prerrequisitos)
2. [Deployment con Docker](#deployment-con-docker)
3. [Deployment Manual](#deployment-manual)
4. [Configuración de Producción](#configuración-de-producción)
5. [Monitoreo y Observabilidad](#monitoreo-y-observabilidad)
6. [Troubleshooting](#troubleshooting)

---

## 🔧 Prerrequisitos

### Requisitos del Sistema

- **OS**: Linux (Ubuntu 20.04+ recomendado) o Windows Server
- **Python**: 3.8+ (si deployment manual)
- **Docker**: 20.10+ (si deployment con Docker)
- **Docker Compose**: 1.29+ (opcional)
- **Memoria**: Mínimo 2GB RAM
- **Disco**: Mínimo 10GB libres

### Dependencias

```bash
# Verificar Docker
docker --version
docker-compose --version

# Verificar Python (si deployment manual)
python3 --version
pip3 --version
```

---

## 🐳 Deployment con Docker

### Opción 1: Docker Compose (Recomendado)

#### Paso 1: Clonar y Configurar

```bash
# Clonar repositorio
git clone <repository-url>
cd robot_movement_ai

# Copiar y configurar variables de entorno
cp .env.example .env
nano .env  # Editar con tus configuraciones
```

#### Paso 2: Configurar Variables de Entorno

Editar `.env` con tus valores de producción:

```env
# Robot Configuration
ROBOT_IP=192.168.1.100
ROBOT_PORT=30001
ROBOT_BRAND=kuka
ROS_ENABLED=false
FEEDBACK_FREQUENCY=1000

# LLM Configuration
LLM_PROVIDER=openai
OPENAI_API_KEY=your_production_key

# Database Configuration
DATABASE_URL=postgresql://user:pass@postgres:5432/robots
POSTGRES_DB=robots
POSTGRES_USER=robotuser
POSTGRES_PASSWORD=strong_password_here
REPOSITORY_TYPE=sql_with_cache
CACHE_TTL=300

# API Configuration
API_PORT=8010
HOST=0.0.0.0

# Redis Configuration
REDIS_PORT=6379

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/robot_movement_ai.log

# Production
DEBUG=false
```

#### Paso 3: Ejecutar Deployment

```bash
# Construir e iniciar servicios
docker-compose up -d --build

# Ver logs
docker-compose logs -f robot-api

# Verificar estado
docker-compose ps
```

#### Paso 4: Verificar Deployment

```bash
# Health check
curl http://localhost:8010/health

# Readiness check
curl http://localhost:8010/health/ready

# Liveness check
curl http://localhost:8010/health/live
```

### Opción 2: Docker Standalone

```bash
# Construir imagen
docker build -t robot-movement-ai:latest .

# Ejecutar contenedor
docker run -d \
  --name robot-movement-ai \
  -p 8010:8010 \
  -e ROBOT_IP=192.168.1.100 \
  -e DATABASE_URL=postgresql://... \
  -v $(pwd)/db:/app/db \
  -v $(pwd)/logs:/app/logs \
  robot-movement-ai:latest

# Ver logs
docker logs -f robot-movement-ai
```

---

## 🐍 Deployment Manual

### Paso 1: Setup del Entorno

```bash
# Usar script de setup
bash scripts/dev_setup.sh

# O manualmente:
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Paso 2: Configurar Variables de Entorno

```bash
# Crear .env
cp .env.example .env
nano .env  # Editar con configuraciones de producción
```

### Paso 3: Configurar Base de Datos

```bash
# Para SQLite (desarrollo)
# No requiere configuración adicional

# Para PostgreSQL (producción)
createdb robots
# O usar docker-compose para postgres
```

### Paso 4: Ejecutar Aplicación

```bash
# Activar entorno virtual
source venv/bin/activate

# Ejecutar aplicación
python -m robot_movement_ai.main \
  --host 0.0.0.0 \
  --port 8010 \
  --robot-brand kuka
```

### Paso 5: Usar Systemd (Linux)

Crear servicio systemd `/etc/systemd/system/robot-movement-ai.service`:

```ini
[Unit]
Description=Robot Movement AI v2.0
After=network.target

[Service]
Type=simple
User=robotuser
WorkingDirectory=/opt/robot_movement_ai
Environment="PATH=/opt/robot_movement_ai/venv/bin"
ExecStart=/opt/robot_movement_ai/venv/bin/python -m robot_movement_ai.main
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Activar servicio:

```bash
sudo systemctl daemon-reload
sudo systemctl enable robot-movement-ai
sudo systemctl start robot-movement-ai
sudo systemctl status robot-movement-ai
```

---

## ⚙️ Configuración de Producción

### Variables de Entorno Críticas

```env
# Seguridad
DEBUG=false
SECRET_KEY=your_secret_key_here

# Base de Datos
DATABASE_URL=postgresql://user:pass@host:5432/robots
REPOSITORY_TYPE=sql_with_cache

# Performance
CACHE_TTL=300
MAX_CONNECTIONS=100
WORKER_THREADS=4

# Logging
LOG_LEVEL=INFO
LOG_FILE=/var/log/robot_movement_ai/app.log
LOG_MAX_SIZE=100MB
LOG_BACKUP_COUNT=10

# Monitoring
PROMETHEUS_ENABLED=true
METRICS_PORT=9090
```

### Optimizaciones de Producción

1. **Base de Datos**:
   - Usar PostgreSQL en lugar de SQLite
   - Configurar conexiones pool
   - Habilitar índices apropiados

2. **Cache**:
   - Usar Redis para cache distribuido
   - Configurar TTL apropiado
   - Monitorear hit rate

3. **Logging**:
   - Configurar rotación de logs
   - Enviar logs a sistema centralizado (ELK, CloudWatch, etc.)
   - Configurar niveles apropiados

4. **Seguridad**:
   - Usar HTTPS en producción
   - Configurar firewall
   - Rotar secretos regularmente
   - Implementar rate limiting

---

## 📊 Monitoreo y Observabilidad

### Health Checks

```bash
# Health check completo
curl http://localhost:8010/health

# Con métricas
curl http://localhost:8010/health?include_metrics=true

# Readiness
curl http://localhost:8010/health/ready

# Liveness
curl http://localhost:8010/health/live

# Métricas Prometheus
curl http://localhost:8010/health/metrics
```

### Prometheus y Grafana

Si usas Docker Compose con perfil de monitoring:

```bash
# Iniciar con monitoring
docker-compose --profile monitoring up -d

# Acceder a Prometheus
# http://localhost:9090

# Acceder a Grafana
# http://localhost:3000
# Usuario: admin
# Password: (configurado en .env)
```

### Métricas Clave a Monitorear

- **Latencia**: Tiempo de respuesta de API
- **Throughput**: Requests por segundo
- **Error Rate**: Porcentaje de errores
- **Circuit Breakers**: Estado de circuit breakers
- **Database**: Conexiones y queries
- **Cache**: Hit rate y tamaño

---

## 🔍 Troubleshooting

### Problemas Comunes

#### 1. Contenedor no inicia

```bash
# Ver logs
docker-compose logs robot-api

# Verificar configuración
docker-compose config

# Reiniciar servicios
docker-compose restart
```

#### 2. Base de datos no conecta

```bash
# Verificar conexión
docker-compose exec postgres psql -U robotuser -d robots

# Verificar variables de entorno
docker-compose exec robot-api env | grep DATABASE
```

#### 3. Health check falla

```bash
# Verificar logs
docker-compose logs robot-api | tail -50

# Verificar endpoints
curl -v http://localhost:8010/health

# Verificar dependencias
docker-compose ps
```

#### 4. Performance lento

```bash
# Verificar recursos
docker stats

# Verificar métricas
curl http://localhost:8010/health/metrics

# Verificar logs de errores
docker-compose logs robot-api | grep ERROR
```

### Comandos Útiles

```bash
# Ver logs en tiempo real
docker-compose logs -f robot-api

# Ejecutar comando en contenedor
docker-compose exec robot-api bash

# Reiniciar servicio específico
docker-compose restart robot-api

# Escalar servicios (si aplica)
docker-compose up -d --scale robot-api=3

# Backup de base de datos
docker-compose exec postgres pg_dump -U robotuser robots > backup.sql

# Restaurar backup
docker-compose exec -T postgres psql -U robotuser robots < backup.sql
```

---

## ✅ Checklist de Deployment

### Pre-Deployment

- [ ] Variables de entorno configuradas
- [ ] Base de datos configurada
- [ ] Secrets configurados
- [ ] Firewall configurado
- [ ] DNS configurado (si aplica)

### Deployment

- [ ] Código desplegado
- [ ] Servicios iniciados
- [ ] Health checks pasando
- [ ] Logs verificados
- [ ] Métricas funcionando

### Post-Deployment

- [ ] Tests de integración pasando
- [ ] Monitoreo configurado
- [ ] Alertas configuradas
- [ ] Documentación actualizada
- [ ] Equipo notificado

---

## 📚 Recursos Adicionales

- [Master Architecture Guide](./MASTER_ARCHITECTURE_GUIDE.md)
- [Implementation Roadmap](./IMPLEMENTATION_ROADMAP.md)
- [START_HERE.md](./START_HERE.md)

---

**Versión**: 1.0.0  
**Última actualización**: 2025-01-27




