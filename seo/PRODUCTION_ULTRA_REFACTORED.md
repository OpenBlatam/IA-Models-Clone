# PRODUCTION DEPLOYMENT GUIDE - Ultra-Optimized SEO Service

## 🚀 Guía Completa de Despliegue en Producción

Esta guía proporciona instrucciones detalladas para desplegar el servicio SEO ultra-refactorizado en un entorno de producción con alta disponibilidad, seguridad y rendimiento.

## 📋 Tabla de Contenidos

- [Requisitos del Sistema](#-requisitos-del-sistema)
- [Arquitectura de Producción](#-arquitectura-de-producción)
- [Preparación del Entorno](#-preparación-del-entorno)
- [Despliegue Automatizado](#-despliegue-automatizado)
- [Configuración de Seguridad](#-configuración-de-seguridad)
- [Monitoreo y Alertas](#-monitoreo-y-alertas)
- [Backup y Recuperación](#-backup-y-recuperación)
- [Escalabilidad](#-escalabilidad)
- [Mantenimiento](#-mantenimiento)
- [Troubleshooting](#-troubleshooting)

## 🖥️ Requisitos del Sistema

### Mínimos
- **CPU**: 4 cores (2.4 GHz+)
- **RAM**: 8GB
- **Almacenamiento**: 50GB SSD
- **Red**: 100 Mbps
- **Sistema Operativo**: Ubuntu 20.04+ / CentOS 8+ / RHEL 8+

### Recomendados
- **CPU**: 8+ cores (3.0 GHz+)
- **RAM**: 16GB+
- **Almacenamiento**: 100GB+ NVMe SSD
- **Red**: 1 Gbps+
- **Sistema Operativo**: Ubuntu 22.04 LTS

### Software Requerido
```bash
# Docker y Docker Compose
Docker Engine 20.10+
Docker Compose 2.0+

# Base de datos
PostgreSQL 13+
Redis 6.0+

# Monitoreo
Prometheus 2.30+
Grafana 8.0+
Node Exporter

# Proxy/Load Balancer
Nginx 1.20+
HAProxy 2.4+ (opcional)

# SSL/TLS
Certbot/Let's Encrypt
```

## 🏗️ Arquitectura de Producción

```
┌─────────────────────────────────────────────────────────────────┐
│                        Load Balancer (HAProxy/Nginx)           │
│                         SSL Termination                         │
└─────────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────────┐
│                    Application Layer                            │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ │
│  │ SEO API 1   │ │ SEO API 2   │ │ SEO API 3   │ │ SEO API N   │ │
│  │ (Container)  │ │ (Container)  │ │ (Container)  │ │ (Container)  │ │
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────────┐
│                    Data Layer                                   │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ │
│  │ PostgreSQL  │ │   Redis     │ │   Cache     │ │   Storage   │ │
│  │ (Primary)   │ │ (Cluster)   │ │ (Distributed)│ │ (Persistent) │ │
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────────┐
│                    Monitoring Layer                             │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ │
│  │ Prometheus  │ │   Grafana   │ │ AlertManager│ │   Logging   │ │
│  │ (Metrics)   │ │ (Dashboard) │ │ (Alerts)    │ │ (ELK Stack)  │ │
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### Componentes Principales

1. **Load Balancer**: Distribuye tráfico entre múltiples instancias
2. **Application Layer**: Múltiples contenedores del servicio SEO
3. **Data Layer**: Base de datos, cache y almacenamiento persistente
4. **Monitoring Layer**: Métricas, dashboards y alertas

## 🔧 Preparación del Entorno

### 1. Configuración del Sistema

```bash
# Actualizar sistema
sudo apt update && sudo apt upgrade -y

# Instalar dependencias
sudo apt install -y \
    curl \
    wget \
    git \
    unzip \
    htop \
    iotop \
    nethogs \
    net-tools \
    ufw \
    fail2ban \
    logrotate \
    cron \
    rsyslog

# Configurar firewall
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow ssh
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw allow 8000/tcp
sudo ufw allow 9090/tcp
sudo ufw allow 3000/tcp
sudo ufw enable
```

### 2. Instalación de Docker

```bash
# Instalar Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Agregar usuario al grupo docker
sudo usermod -aG docker $USER

# Configurar Docker daemon
sudo tee /etc/docker/daemon.json > /dev/null << EOF
{
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "100m",
    "max-file": "3"
  },
  "storage-driver": "overlay2",
  "storage-opts": [
    "overlay2.override_kernel_check=true"
  ],
  "default-ulimits": {
    "nofile": {
      "Hard": 64000,
      "Name": "nofile",
      "Soft": 64000
    }
  }
}
EOF

# Reiniciar Docker
sudo systemctl restart docker
sudo systemctl enable docker
```

### 3. Configuración de Redes

```bash
# Crear red Docker para el servicio
docker network create seo-network

# Configurar DNS
sudo tee /etc/systemd/resolved.conf > /dev/null << EOF
[Resolve]
DNS=8.8.8.8 8.8.4.4
FallbackDNS=1.1.1.1
EOF

sudo systemctl restart systemd-resolved
```

### 4. Configuración de Almacenamiento

```bash
# Crear directorios para datos persistentes
sudo mkdir -p /var/lib/seo-service/{data,cache,logs,backups}
sudo mkdir -p /etc/seo-service/{config,ssl}

# Configurar permisos
sudo chown -R $USER:$USER /var/lib/seo-service
sudo chown -R $USER:$USER /etc/seo-service

# Configurar límites del sistema
sudo tee /etc/security/limits.conf > /dev/null << EOF
* soft nofile 65536
* hard nofile 65536
* soft nproc 32768
* hard nproc 32768
EOF
```

## 🚀 Despliegue Automatizado

### 1. Script de Despliegue

```bash
# Clonar repositorio
git clone <repository-url>
cd seo-service

# Hacer ejecutable el script
chmod +x scripts/deploy_ultra_optimized.sh

# Ejecutar despliegue
./scripts/deploy_ultra_optimized.sh
```

### 2. Despliegue Manual

```bash
# 1. Configurar variables de entorno
cp .env.example .env.production
# Editar .env.production con valores reales

# 2. Construir imágenes
docker build -f Dockerfile.production -t seo-service:latest .

# 3. Desplegar servicios
docker-compose -f docker-compose.production.yml up -d

# 4. Verificar estado
docker-compose -f docker-compose.production.yml ps
```

### 3. Verificación del Despliegue

```bash
# Verificar servicios
docker-compose -f docker-compose.production.yml ps

# Verificar logs
docker-compose -f docker-compose.production.yml logs -f

# Health check
curl -f http://localhost:8000/health

# Métricas
curl http://localhost:8000/metrics
```

## 🔒 Configuración de Seguridad

### 1. SSL/TLS

```bash
# Instalar Certbot
sudo apt install certbot

# Obtener certificado SSL
sudo certbot certonly --standalone -d yourdomain.com

# Configurar renovación automática
sudo crontab -e
# Agregar: 0 12 * * * /usr/bin/certbot renew --quiet
```

### 2. Configuración de Nginx

```nginx
# /etc/nginx/sites-available/seo-service
server {
    listen 80;
    server_name yourdomain.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name yourdomain.com;

    ssl_certificate /etc/letsencrypt/live/yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/yourdomain.com/privkey.pem;

    # Security headers
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains";

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    limit_req zone=api burst=20 nodelay;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### 3. Configuración de Firewall

```bash
# Configurar iptables
sudo iptables -A INPUT -p tcp --dport 22 -j ACCEPT
sudo iptables -A INPUT -p tcp --dport 80 -j ACCEPT
sudo iptables -A INPUT -p tcp --dport 443 -j ACCEPT
sudo iptables -A INPUT -p tcp --dport 8000 -j ACCEPT
sudo iptables -A INPUT -j DROP

# Guardar configuración
sudo iptables-save > /etc/iptables/rules.v4
```

### 4. Fail2ban

```bash
# Configurar Fail2ban para API
sudo tee /etc/fail2ban/jail.local > /dev/null << EOF
[seo-api]
enabled = true
port = 8000
filter = seo-api
logpath = /var/log/seo-service/api.log
maxretry = 5
bantime = 3600
findtime = 600
EOF
```

## 📊 Monitoreo y Alertas

### 1. Configuración de Prometheus

```yaml
# prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "alert.rules"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093

scrape_configs:
  - job_name: 'seo-service'
    static_configs:
      - targets: ['seo-api:8000']
    metrics_path: '/metrics'
    scrape_interval: 5s

  - job_name: 'redis'
    static_configs:
      - targets: ['redis:6379']

  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres:5432']
```

### 2. Reglas de Alertas

```yaml
# alert.rules
groups:
  - name: seo-service
    rules:
      - alert: HighErrorRate
        expr: rate(seo_errors_total[5m]) > 0.1
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "High error rate detected"
          description: "Error rate is {{ $value }} errors per second"

      - alert: HighResponseTime
        expr: histogram_quantile(0.95, rate(seo_request_duration_seconds_bucket[5m])) > 2
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "High response time detected"
          description: "95th percentile response time is {{ $value }} seconds"

      - alert: ServiceDown
        expr: up{job="seo-service"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "SEO service is down"
          description: "SEO service has been down for more than 1 minute"
```

### 3. Dashboard de Grafana

```json
{
  "dashboard": {
    "title": "SEO Service Dashboard",
    "panels": [
      {
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(seo_requests_total[5m])",
            "legendFormat": "{{method}} {{endpoint}}"
          }
        ]
      },
      {
        "title": "Response Time",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(seo_request_duration_seconds_bucket[5m]))",
            "legendFormat": "95th percentile"
          }
        ]
      },
      {
        "title": "Error Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(seo_errors_total[5m])",
            "legendFormat": "{{type}}"
          }
        ]
      }
    ]
  }
}
```

## 💾 Backup y Recuperación

### 1. Script de Backup

```bash
#!/bin/bash
# backup.sh

BACKUP_DIR="/var/lib/seo-service/backups"
DATE=$(date +%Y%m%d-%H%M%S)
BACKUP_NAME="seo-backup-$DATE"

# Crear directorio de backup
mkdir -p "$BACKUP_DIR/$BACKUP_NAME"

# Backup de base de datos
docker-compose -f docker-compose.production.yml exec -T postgres \
    pg_dump -U postgres seo_db > "$BACKUP_DIR/$BACKUP_NAME/database.sql"

# Backup de datos de Redis
docker-compose -f docker-compose.production.yml exec -T redis \
    redis-cli -a "$REDIS_PASSWORD" BGSAVE

# Backup de archivos de configuración
cp -r /etc/seo-service "$BACKUP_DIR/$BACKUP_NAME/config"

# Backup de logs
tar -czf "$BACKUP_DIR/$BACKUP_NAME/logs.tar.gz" /var/log/seo-service

# Comprimir backup completo
tar -czf "$BACKUP_DIR/$BACKUP_NAME.tar.gz" -C "$BACKUP_DIR" "$BACKUP_NAME"

# Limpiar directorio temporal
rm -rf "$BACKUP_DIR/$BACKUP_NAME"

# Eliminar backups antiguos (más de 30 días)
find "$BACKUP_DIR" -name "*.tar.gz" -mtime +30 -delete

echo "Backup completed: $BACKUP_DIR/$BACKUP_NAME.tar.gz"
```

### 2. Script de Recuperación

```bash
#!/bin/bash
# restore.sh

BACKUP_FILE="$1"
BACKUP_DIR="/tmp/restore-$$"

if [ -z "$BACKUP_FILE" ]; then
    echo "Usage: $0 <backup-file>"
    exit 1
fi

# Extraer backup
tar -xzf "$BACKUP_FILE" -C /tmp
RESTORE_DIR=$(find /tmp -name "seo-backup-*" -type d | head -1)

# Restaurar base de datos
docker-compose -f docker-compose.production.yml exec -T postgres \
    psql -U postgres seo_db < "$RESTORE_DIR/database.sql"

# Restaurar configuración
sudo cp -r "$RESTORE_DIR/config"/* /etc/seo-service/

# Restaurar logs
tar -xzf "$RESTORE_DIR/logs.tar.gz" -C /

# Limpiar
rm -rf "$RESTORE_DIR"

echo "Restore completed successfully"
```

## 📈 Escalabilidad

### 1. Escalado Horizontal

```bash
# Escalar servicios
docker-compose -f docker-compose.production.yml up -d --scale seo-api=3

# Configurar load balancer
docker-compose -f docker-compose.production.yml up -d haproxy
```

### 2. Configuración de HAProxy

```haproxy
# haproxy.cfg
global
    daemon
    maxconn 4096

defaults
    mode http
    timeout connect 5000ms
    timeout client 50000ms
    timeout server 50000ms

frontend seo-frontend
    bind *:80
    bind *:443 ssl crt /etc/ssl/certs/seo-service.pem
    redirect scheme https if !{ ssl_fc }
    
    acl is_api path_beg /api
    use_backend seo-api if is_api
    default_backend seo-api

backend seo-api
    balance roundrobin
    option httpchk GET /health
    server seo-api1 seo-api1:8000 check
    server seo-api2 seo-api2:8000 check
    server seo-api3 seo-api3:8000 check
```

### 3. Auto-scaling

```yaml
# docker-compose.production.yml
services:
  seo-api:
    image: seo-service:latest
    deploy:
      replicas: 3
      update_config:
        parallelism: 1
        delay: 10s
      restart_policy:
        condition: on-failure
      resources:
        limits:
          cpus: '1.0'
          memory: 2G
        reservations:
          cpus: '0.5'
          memory: 1G
```

## 🔧 Mantenimiento

### 1. Actualizaciones

```bash
# Script de actualización
#!/bin/bash
# update.sh

# Crear backup antes de actualizar
./backup.sh

# Detener servicios
docker-compose -f docker-compose.production.yml down

# Actualizar código
git pull origin main

# Reconstruir imágenes
docker build -f Dockerfile.production -t seo-service:latest .

# Iniciar servicios
docker-compose -f docker-compose.production.yml up -d

# Verificar health
./health-check.sh
```

### 2. Limpieza de Sistema

```bash
# Limpiar contenedores no utilizados
docker container prune -f

# Limpiar imágenes no utilizadas
docker image prune -a -f

# Limpiar volúmenes no utilizados
docker volume prune -f

# Limpiar redes no utilizadas
docker network prune -f

# Limpiar logs antiguos
find /var/log/seo-service -name "*.log" -mtime +30 -delete
```

### 3. Monitoreo de Recursos

```bash
# Script de monitoreo
#!/bin/bash
# monitor.sh

echo "=== System Resources ==="
echo "CPU Usage: $(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)%"
echo "Memory Usage: $(free | grep Mem | awk '{printf "%.2f%%", $3/$2 * 100.0}')"
echo "Disk Usage: $(df / | awk 'NR==2 {print $5}')"

echo "=== Docker Resources ==="
docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}"

echo "=== Service Status ==="
docker-compose -f docker-compose.production.yml ps
```

## 🛠️ Troubleshooting

### 1. Problemas Comunes

#### Servicio no inicia
```bash
# Verificar logs
docker-compose -f docker-compose.production.yml logs seo-api

# Verificar configuración
docker-compose -f docker-compose.production.yml config

# Verificar recursos
docker stats
```

#### Alto uso de CPU/Memoria
```bash
# Identificar contenedor problemático
docker stats

# Verificar procesos dentro del contenedor
docker exec -it seo-api top

# Verificar logs de aplicación
docker logs seo-api
```

#### Problemas de red
```bash
# Verificar conectividad
docker exec seo-api ping redis
docker exec seo-api ping postgres

# Verificar puertos
netstat -tlnp | grep 8000
```

### 2. Recuperación de Emergencia

```bash
# Script de recuperación
#!/bin/bash
# emergency-recovery.sh

echo "Starting emergency recovery..."

# Detener todos los servicios
docker-compose -f docker-compose.production.yml down

# Limpiar recursos Docker
docker system prune -f

# Restaurar último backup
LATEST_BACKUP=$(ls -t /var/lib/seo-service/backups/*.tar.gz | head -1)
./restore.sh "$LATEST_BACKUP"

# Reiniciar servicios
docker-compose -f docker-compose.production.yml up -d

# Verificar estado
./health-check.sh
```

### 3. Logs y Debugging

```bash
# Ver logs en tiempo real
docker-compose -f docker-compose.production.yml logs -f

# Ver logs de un servicio específico
docker-compose -f docker-compose.production.yml logs -f seo-api

# Ver logs con timestamps
docker-compose -f docker-compose.production.yml logs -f --timestamps

# Exportar logs para análisis
docker-compose -f docker-compose.production.yml logs > logs.txt
```

## 📞 Soporte

### Contactos de Emergencia
- **DevOps Team**: devops@company.com
- **On-call Engineer**: +1-555-0123
- **Escalation Manager**: +1-555-0124

### Documentación Adicional
- [Runbook de Operaciones](RUNBOOK.md)
- [Guía de Troubleshooting](TROUBLESHOOTING.md)
- [Procedimientos de Escalación](ESCALATION.md)

---

**SEO Service Ultra-Optimized v2.0.0** - Listo para producción con arquitectura empresarial. 