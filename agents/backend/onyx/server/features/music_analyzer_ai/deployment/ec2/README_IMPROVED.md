# EC2 Deployment - Guía Mejorada

## 🚀 Inicio Rápido Mejorado

### Script de Despliegue Mejorado

El script `deploy.sh` ahora incluye:

- ✅ **Validación de recursos** (memoria, CPU, disco)
- ✅ **Detección mejorada de OS**
- ✅ **Manejo robusto de errores**
- ✅ **Retry logic** para instalaciones
- ✅ **Health checks** con reintentos
- ✅ **Logging detallado**
- ✅ **Verificación de permisos**

```bash
# Ejecutar despliegue mejorado
cd /opt
sudo git clone https://github.com/your-repo/music-analyzer-ai.git
cd music-analyzer-ai
sudo ./deployment/ec2/deploy.sh
```

## 📊 Nuevos Scripts de Utilidad

### Monitor Script

Monitorea el estado del sistema y servicios:

```bash
./deployment/ec2/monitor.sh
```

**Muestra:**
- Estado de Docker
- Estado de contenedores
- Health checks
- Uso de recursos (CPU, memoria, disco)
- Estado de puertos
- Logs recientes

### Update Script

Actualiza la aplicación con zero-downtime:

```bash
./deployment/ec2/update.sh
```

**Características:**
- Backup automático antes de actualizar
- Rolling updates (sin downtime)
- Rollback automático si falla
- Limpieza de imágenes antiguas

## 🔧 Mejoras Implementadas

### 1. Validaciones

- Verifica memoria mínima (2GB)
- Verifica espacio en disco (10GB)
- Verifica permisos de usuario
- Verifica conectividad de red

### 2. Manejo de Errores

- Retry logic para instalaciones
- Verificación de cada paso
- Mensajes de error claros
- Logging a archivo

### 3. Health Checks

- Reintentos automáticos (30 intentos)
- Verificación de endpoints
- Timeout configurable
- Reporte de estado

### 4. Optimizaciones

- Detección automática de arquitectura
- Instalación de versiones específicas
- Uso eficiente de recursos
- Limpieza automática

## 📝 Uso de Scripts

### Despliegue Inicial

```bash
sudo ./deployment/ec2/deploy.sh
```

### Monitoreo

```bash
# Monitoreo manual
./deployment/ec2/monitor.sh

# Monitoreo continuo (cada 60 segundos)
watch -n 60 ./deployment/ec2/monitor.sh
```

### Actualización

```bash
# Actualizar aplicación
./deployment/ec2/update.sh

# Ver logs de actualización
tail -f /var/log/music-analyzer-deploy.log
```

## 🔍 Troubleshooting Mejorado

### Ver Logs Detallados

```bash
# Logs de despliegue
tail -f /var/log/music-analyzer-deploy.log

# Logs de aplicación
cd /opt/music-analyzer-ai
docker-compose -f deployment/docker-compose.prod.yml logs -f
```

### Diagnóstico Rápido

```bash
# Ejecutar monitor
./deployment/ec2/monitor.sh

# Verificar recursos
free -h
df -h
docker stats --no-stream
```

### Recuperación

```bash
# Si el despliegue falla
cd /opt/music-analyzer-ai
docker-compose -f deployment/docker-compose.prod.yml down
docker-compose -f deployment/docker-compose.prod.yml up -d

# Restaurar desde backup
sudo cp -r /opt/music-analyzer-ai-backup /opt/music-analyzer-ai
cd /opt/music-analyzer-ai
docker-compose -f deployment/docker-compose.prod.yml up -d
```

## 🎯 Mejores Prácticas

1. **Siempre ejecutar con sudo** para permisos completos
2. **Revisar logs** después del despliegue
3. **Monitorear recursos** antes de desplegar
4. **Hacer backups** antes de actualizar
5. **Verificar health checks** después de cambios

## 📊 Métricas y Monitoreo

### CloudWatch Integration

```bash
# Instalar CloudWatch agent
sudo yum install -y amazon-cloudwatch-agent

# Configurar
sudo /opt/aws/amazon-cloudwatch-agent/bin/amazon-cloudwatch-agent-ctl \
  -a fetch-config \
  -m ec2 \
  -c ssm:AmazonCloudWatch-linux
```

### Alertas

Configurar alertas en CloudWatch para:
- CPU > 80%
- Memoria > 90%
- Health check failures
- Disk space < 20%

## 🔒 Seguridad Mejorada

### Firewall

```bash
# Ubuntu
sudo ufw allow 22/tcp
sudo ufw allow 8010/tcp
sudo ufw enable

# Amazon Linux
sudo firewall-cmd --permanent --add-port=8010/tcp
sudo firewall-cmd --reload
```

### Actualizaciones Automáticas

```bash
# Amazon Linux
sudo yum install -y yum-cron
sudo systemctl enable yum-cron

# Ubuntu
sudo apt-get install -y unattended-upgrades
sudo dpkg-reconfigure -plow unattended-upgrades
```

## 🚀 Performance Tips

1. **Usar instancias optimizadas** (t3.medium o superior)
2. **Habilitar swap** si la memoria es limitada
3. **Configurar límites de Docker** según recursos
4. **Usar EBS optimizado** para mejor I/O
5. **Habilitar CloudWatch** para monitoreo

## 📚 Recursos Adicionales

- Ver `README.md` para guía completa
- Ver logs en `/var/log/music-analyzer-deploy.log`
- Consultar documentación de Docker
- Revisar métricas en CloudWatch




