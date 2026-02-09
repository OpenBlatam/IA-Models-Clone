#!/bin/bash

# Script de Despliegue en Producción para el Servicio SEO
# Autor: Sistema de Despliegue Automatizado
# Versión: 1.0.0

set -euo pipefail

# Colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuración
PROJECT_NAME="seo-service"
DOCKER_REGISTRY=""
IMAGE_TAG="latest"
ENVIRONMENT="production"

# Función para logging
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
    exit 1
}

info() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] INFO: $1${NC}"
}

# Función para verificar dependencias
check_dependencies() {
    log "Verificando dependencias..."
    
    local deps=("docker" "docker-compose" "curl" "openssl")
    
    for dep in "${deps[@]}"; do
        if ! command -v "$dep" &> /dev/null; then
            error "Dependencia requerida no encontrada: $dep"
        fi
    done
    
    log "Todas las dependencias están disponibles"
}

# Función para verificar variables de entorno
check_environment() {
    log "Verificando variables de entorno..."
    
    local required_vars=(
        "SECRET_KEY"
        "OPENAI_API_KEY"
        "DOMAIN"
        "SENTRY_DSN"
    )
    
    local missing_vars=()
    
    for var in "${required_vars[@]}"; do
        if [[ -z "${!var:-}" ]]; then
            missing_vars+=("$var")
        fi
    done
    
    if [[ ${#missing_vars[@]} -gt 0 ]]; then
        error "Variables de entorno requeridas faltantes: ${missing_vars[*]}"
    fi
    
    log "Todas las variables de entorno están configuradas"
}

# Función para generar certificados SSL
generate_ssl_certificates() {
    log "Generando certificados SSL..."
    
    if [[ ! -d "ssl" ]]; then
        mkdir -p ssl
    fi
    
    if [[ ! -f "ssl/cert.pem" ]] || [[ ! -f "ssl/key.pem" ]]; then
        warn "Generando certificados SSL autofirmados para desarrollo"
        openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
            -keyout ssl/key.pem \
            -out ssl/cert.pem \
            -subj "/C=US/ST=State/L=City/O=Organization/CN=$DOMAIN"
    else
        log "Certificados SSL ya existen"
    fi
}

# Función para configurar Redis
setup_redis() {
    log "Configurando Redis..."
    
    if [[ ! -f "redis.conf" ]]; then
        cat > redis.conf << EOF
# Configuración de Redis para Producción
bind 0.0.0.0
port 6379
timeout 300
tcp-keepalive 60
daemonize no
supervised no
pidfile /var/run/redis_6379.pid
loglevel notice
logfile ""
databases 16
save 900 1
save 300 10
save 60 10000
stop-writes-on-bgsave-error yes
rdbcompression yes
rdbchecksum yes
dbfilename dump.rdb
dir /data
maxmemory 256mb
maxmemory-policy allkeys-lru
appendonly yes
appendfilename "appendonly.aof"
appendfsync everysec
no-appendfsync-on-rewrite no
auto-aof-rewrite-percentage 100
auto-aof-rewrite-min-size 64mb
EOF
        log "Archivo de configuración de Redis creado"
    fi
}

# Función para configurar directorios
setup_directories() {
    log "Configurando directorios..."
    
    local dirs=("logs" "cache" "temp" "ssl" "grafana/dashboards" "grafana/datasources")
    
    for dir in "${dirs[@]}"; do
        if [[ ! -d "$dir" ]]; then
            mkdir -p "$dir"
            log "Directorio creado: $dir"
        fi
    done
    
    # Configurar permisos
    chmod 755 logs cache temp
    chmod 600 ssl/*
}

# Función para configurar Grafana
setup_grafana() {
    log "Configurando Grafana..."
    
    # Configurar datasource de Prometheus
    if [[ ! -f "grafana/datasources/prometheus.yml" ]]; then
        mkdir -p grafana/datasources
        cat > grafana/datasources/prometheus.yml << EOF
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    editable: true
EOF
        log "Datasource de Prometheus configurado"
    fi
    
    # Configurar dashboards
    if [[ ! -f "grafana/dashboards/dashboard.yml" ]]; then
        mkdir -p grafana/dashboards
        cat > grafana/dashboards/dashboard.yml << EOF
apiVersion: 1

providers:
  - name: 'SEO Service'
    orgId: 1
    folder: ''
    type: file
    disableDeletion: false
    updateIntervalSeconds: 10
    allowUiUpdates: true
    options:
      path: /etc/grafana/provisioning/dashboards
EOF
        log "Configuración de dashboards creada"
    fi
}

# Función para configurar Filebeat
setup_filebeat() {
    log "Configurando Filebeat..."
    
    if [[ ! -f "filebeat.yml" ]]; then
        cat > filebeat.yml << EOF
filebeat.inputs:
- type: log
  enabled: true
  paths:
    - /var/log/seo/*.log
  json.keys_under_root: true
  json.add_error_key: true
  json.message_key: message

processors:
- add_host_metadata: ~
- add_cloud_metadata: ~

output.elasticsearch:
  hosts: ["elasticsearch:9200"]
  indices:
    - index: "filebeat-%{[agent.version]}-%{+yyyy.MM.dd}"

setup.kibana:
  host: "kibana:5601"
EOF
        log "Configuración de Filebeat creada"
    fi
}

# Función para verificar salud del sistema
health_check() {
    log "Realizando health check..."
    
    local services=("seo-service" "redis" "prometheus" "grafana" "nginx")
    local max_attempts=30
    local attempt=1
    
    for service in "${services[@]}"; do
        log "Verificando servicio: $service"
        
        while [[ $attempt -le $max_attempts ]]; do
            if docker-compose ps "$service" | grep -q "Up"; then
                log "Servicio $service está funcionando"
                break
            else
                warn "Intento $attempt/$max_attempts: Servicio $service no está listo"
                sleep 2
                ((attempt++))
            fi
        done
        
        if [[ $attempt -gt $max_attempts ]]; then
            error "Servicio $service no pudo iniciarse correctamente"
        fi
        
        attempt=1
    done
    
    # Verificar endpoints
    log "Verificando endpoints..."
    
    local endpoints=(
        "http://localhost/health"
        "http://localhost/metrics"
        "http://localhost:3000"
        "http://localhost:9091"
    )
    
    for endpoint in "${endpoints[@]}"; do
        if curl -f -s "$endpoint" > /dev/null; then
            log "Endpoint $endpoint responde correctamente"
        else
            warn "Endpoint $endpoint no responde"
        fi
    done
}

# Función para backup
create_backup() {
    log "Creando backup..."
    
    local backup_dir="backups/$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$backup_dir"
    
    # Backup de datos
    docker-compose exec -T redis redis-cli BGSAVE
    sleep 5
    
    # Backup de volúmenes
    docker run --rm -v "${PROJECT_NAME}_redis-data:/data" -v "$(pwd)/$backup_dir:/backup" \
        alpine tar czf /backup/redis-data.tar.gz -C /data .
    
    # Backup de logs
    tar czf "$backup_dir/logs.tar.gz" logs/
    
    log "Backup creado en: $backup_dir"
}

# Función para limpiar recursos
cleanup() {
    log "Limpiando recursos..."
    
    # Limpiar contenedores detenidos
    docker container prune -f
    
    # Limpiar imágenes no utilizadas
    docker image prune -f
    
    # Limpiar volúmenes no utilizados
    docker volume prune -f
    
    # Limpiar redes no utilizadas
    docker network prune -f
    
    log "Limpieza completada"
}

# Función para mostrar información del sistema
show_system_info() {
    log "Información del sistema:"
    
    echo "=== Docker ==="
    docker version
    echo
    
    echo "=== Docker Compose ==="
    docker-compose version
    echo
    
    echo "=== Espacio en disco ==="
    df -h
    echo
    
    echo "=== Memoria ==="
    free -h
    echo
    
    echo "=== CPU ==="
    nproc
    echo
}

# Función principal de despliegue
deploy() {
    log "Iniciando despliegue en producción..."
    
    # Verificaciones previas
    check_dependencies
    check_environment
    show_system_info
    
    # Configuración
    generate_ssl_certificates
    setup_redis
    setup_directories
    setup_grafana
    setup_filebeat
    
    # Backup previo
    if [[ -d "backups" ]]; then
        create_backup
    fi
    
    # Construir y levantar servicios
    log "Construyendo y levantando servicios..."
    docker-compose build --no-cache
    docker-compose up -d
    
    # Esperar a que los servicios estén listos
    log "Esperando a que los servicios estén listos..."
    sleep 30
    
    # Health check
    health_check
    
    # Limpieza
    cleanup
    
    log "Despliegue completado exitosamente!"
    
    # Mostrar información de acceso
    echo
    echo "=== Información de Acceso ==="
    echo "API: https://$DOMAIN"
    echo "Documentación: https://$DOMAIN/docs"
    echo "Grafana: http://localhost:3000 (admin/admin)"
    echo "Prometheus: http://localhost:9091"
    echo "Kibana: http://localhost:5601"
    echo
}

# Función para rollback
rollback() {
    log "Iniciando rollback..."
    
    # Detener servicios
    docker-compose down
    
    # Restaurar backup más reciente
    local latest_backup=$(ls -t backups/ | head -1)
    if [[ -n "$latest_backup" ]]; then
        log "Restaurando backup: $latest_backup"
        tar xzf "backups/$latest_backup/redis-data.tar.gz" -C .
        tar xzf "backups/$latest_backup/logs.tar.gz"
    fi
    
    # Levantar servicios
    docker-compose up -d
    
    log "Rollback completado"
}

# Función para mostrar logs
show_logs() {
    local service=${1:-"seo-service"}
    docker-compose logs -f "$service"
}

# Función para mostrar estado
show_status() {
    log "Estado de los servicios:"
    docker-compose ps
    
    echo
    log "Uso de recursos:"
    docker stats --no-stream
}

# Función para mostrar ayuda
show_help() {
    echo "Script de Despliegue en Producción para el Servicio SEO"
    echo
    echo "Uso: $0 [COMANDO]"
    echo
    echo "Comandos:"
    echo "  deploy     - Desplegar en producción"
    echo "  rollback   - Hacer rollback a versión anterior"
    echo "  logs       - Mostrar logs de un servicio"
    echo "  status     - Mostrar estado de los servicios"
    echo "  backup     - Crear backup manual"
    echo "  cleanup    - Limpiar recursos Docker"
    echo "  health     - Realizar health check"
    echo "  help       - Mostrar esta ayuda"
    echo
    echo "Ejemplos:"
    echo "  $0 deploy"
    echo "  $0 logs seo-service"
    echo "  $0 status"
}

# Manejo de argumentos
case "${1:-deploy}" in
    "deploy")
        deploy
        ;;
    "rollback")
        rollback
        ;;
    "logs")
        show_logs "${2:-seo-service}"
        ;;
    "status")
        show_status
        ;;
    "backup")
        create_backup
        ;;
    "cleanup")
        cleanup
        ;;
    "health")
        health_check
        ;;
    "help"|"-h"|"--help")
        show_help
        ;;
    *)
        error "Comando desconocido: $1. Use '$0 help' para ver los comandos disponibles."
        ;;
esac 