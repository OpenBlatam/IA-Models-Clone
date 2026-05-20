# EC2 Deployment Guide

Guía completa para desplegar Music Analyzer AI en cualquier instancia EC2.

## 🚀 Inicio Rápido

### Opción 1: Script Automático (Recomendado)

```bash
# Conectarse a la instancia EC2
ssh -i your-key.pem ec2-user@your-ec2-ip

# Ejecutar script de despliegue
curl -fsSL https://raw.githubusercontent.com/your-repo/music-analyzer-ai/main/deployment/ec2/deploy.sh | bash
```

O descargar y ejecutar:

```bash
wget https://raw.githubusercontent.com/your-repo/music-analyzer-ai/main/deployment/ec2/deploy.sh
chmod +x deploy.sh
sudo ./deploy.sh
```

### Opción 2: User Data Script

Al crear la instancia EC2, usar el User Data script:

**Amazon Linux 2:**
```bash
# Copiar contenido de user-data-amazon-linux.sh en User Data
```

**Ubuntu:**
```bash
# Copiar contenido de user-data-ubuntu.sh en User Data
```

### Opción 3: CloudFormation

```bash
aws cloudformation create-stack \
  --stack-name music-analyzer-ai \
  --template-body file://deployment/ec2/cloudformation-ec2.yaml \
  --parameters \
    ParameterKey=InstanceType,ParameterValue=t3.medium \
    ParameterKey=KeyPairName,ParameterValue=your-key-pair \
    ParameterKey=SpotifyClientId,ParameterValue=your_client_id \
    ParameterKey=SpotifyClientSecret,ParameterValue=your_client_secret
```

## 📋 Requisitos Previos

### Instancia EC2

- **Tipo mínimo**: t3.small (2 vCPU, 2GB RAM)
- **Recomendado**: t3.medium o superior (2+ vCPU, 4GB+ RAM)
- **Sistema Operativo**: Amazon Linux 2, Ubuntu 20.04+, o cualquier Linux con Docker

### Puertos a Abrir

Asegúrate de que el Security Group permita:

- **22**: SSH
- **8010**: API principal
- **80**: HTTP (si usas Nginx)
- **443**: HTTPS (si usas Nginx)
- **3000**: Grafana (opcional)
- **9090**: Prometheus (opcional)

## 🔧 Instalación Manual

### Paso 1: Conectarse a la Instancia

```bash
ssh -i your-key.pem ec2-user@your-ec2-ip
```

### Paso 2: Instalar Docker

**Amazon Linux 2:**
```bash
sudo yum update -y
sudo yum install -y docker
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -a -G docker ec2-user
```

**Ubuntu:**
```bash
sudo apt-get update
sudo apt-get install -y docker.io
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -aG docker ubuntu
```

### Paso 3: Instalar Docker Compose

```bash
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
sudo ln -sf /usr/local/bin/docker-compose /usr/bin/docker-compose
```

### Paso 4: Desplegar Aplicación

**Opción A: Clonar desde Git**
```bash
cd /opt
sudo git clone https://github.com/your-repo/music-analyzer-ai.git
sudo chown -R ec2-user:ec2-user music-analyzer-ai
cd music-analyzer-ai
```

**Opción B: Subir desde Local**
```bash
# Desde tu máquina local
scp -r -i your-key.pem /path/to/music-analyzer-ai ec2-user@your-ec2-ip:/opt/
```

**Opción C: Desde S3**
```bash
# Subir a S3 primero
aws s3 sync /path/to/music-analyzer-ai s3://your-bucket/music-analyzer-ai/

# Descargar en EC2
aws s3 sync s3://your-bucket/music-analyzer-ai /opt/music-analyzer-ai
```

### Paso 5: Configurar Variables de Entorno

```bash
cd /opt/music-analyzer-ai
nano .env
```

Editar con tus credenciales:
```env
ENVIRONMENT=production
SPOTIFY_CLIENT_ID=tu_client_id_real
SPOTIFY_CLIENT_SECRET=tu_client_secret_real
LOG_LEVEL=INFO
CACHE_ENABLED=true
POSTGRES_PASSWORD=password_seguro
REDIS_PASSWORD=password_seguro
```

### Paso 6: Iniciar Servicios

```bash
cd /opt/music-analyzer-ai
docker-compose -f deployment/docker-compose.prod.yml up -d
```

### Paso 7: Verificar

```bash
# Ver logs
docker-compose -f deployment/docker-compose.prod.yml logs -f

# Verificar health
curl http://localhost:8010/health
```

## 🔄 Actualización

### Actualizar Código

```bash
cd /opt/music-analyzer-ai

# Si usas Git
git pull origin main

# Reconstruir y reiniciar
docker-compose -f deployment/docker-compose.prod.yml down
docker-compose -f deployment/docker-compose.prod.yml build
docker-compose -f deployment/docker-compose.prod.yml up -d
```

### Actualizar Solo Imágenes

```bash
cd /opt/music-analyzer-ai
docker-compose -f deployment/docker-compose.prod.yml pull
docker-compose -f deployment/docker-compose.prod.yml up -d
```

## 🔧 Configuración Avanzada

### Auto-start con systemd

Crear servicio systemd:

```bash
sudo nano /etc/systemd/system/music-analyzer-ai.service
```

Contenido:
```ini
[Unit]
Description=Music Analyzer AI
Requires=docker.service
After=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/opt/music-analyzer-ai
ExecStart=/opt/music-analyzer-ai/deployment/start.sh
ExecStop=/usr/bin/docker-compose -f /opt/music-analyzer-ai/deployment/docker-compose.prod.yml down
User=ec2-user
Group=ec2-user

[Install]
WantedBy=multi-user.target
```

Habilitar:
```bash
sudo systemctl daemon-reload
sudo systemctl enable music-analyzer-ai.service
sudo systemctl start music-analyzer-ai.service
```

### Nginx Reverse Proxy

Si quieres usar Nginx como reverse proxy:

```bash
sudo yum install -y nginx  # Amazon Linux
# o
sudo apt-get install -y nginx  # Ubuntu

# Copiar configuración
sudo cp /opt/music-analyzer-ai/deployment/nginx/nginx.prod.conf /etc/nginx/nginx.conf

# Iniciar Nginx
sudo systemctl start nginx
sudo systemctl enable nginx
```

### SSL con Let's Encrypt

```bash
# Instalar Certbot
sudo yum install -y certbot python3-certbot-nginx  # Amazon Linux
# o
sudo apt-get install -y certbot python3-certbot-nginx  # Ubuntu

# Obtener certificado
sudo certbot --nginx -d your-domain.com
```

## 📊 Monitoreo

### CloudWatch Logs

Instalar CloudWatch agent:

```bash
# Amazon Linux 2
sudo yum install -y amazon-cloudwatch-agent

# Configurar
sudo /opt/aws/amazon-cloudwatch-agent/bin/amazon-cloudwatch-agent-ctl \
  -a fetch-config \
  -m ec2 \
  -c ssm:AmazonCloudWatch-linux \
  -s
```

### Logs Locales

```bash
# Ver logs de aplicación
docker-compose -f deployment/docker-compose.prod.yml logs -f music-analyzer-ai

# Ver logs de todos los servicios
docker-compose -f deployment/docker-compose.prod.yml logs -f
```

## 🔒 Seguridad

### Firewall (UFW - Ubuntu)

```bash
sudo ufw allow 22/tcp
sudo ufw allow 8010/tcp
sudo ufw enable
```

### Actualizaciones Automáticas

**Amazon Linux 2:**
```bash
sudo yum install -y yum-cron
sudo systemctl enable yum-cron
sudo systemctl start yum-cron
```

**Ubuntu:**
```bash
sudo apt-get install -y unattended-upgrades
sudo dpkg-reconfigure -plow unattended-upgrades
```

### Rotación de Logs

```bash
sudo nano /etc/logrotate.d/music-analyzer-ai
```

Contenido:
```
/var/log/music-analyzer-ai/*.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
}
```

## 🐛 Troubleshooting

### Docker no inicia

```bash
sudo systemctl status docker
sudo journalctl -u docker
```

### Contenedores no inician

```bash
# Ver logs
docker-compose logs

# Verificar recursos
docker stats

# Reiniciar Docker
sudo systemctl restart docker
```

### Puerto en uso

```bash
# Ver qué usa el puerto
sudo netstat -tulpn | grep 8010
# o
sudo lsof -i :8010

# Matar proceso
sudo kill -9 <PID>
```

### Problemas de memoria

```bash
# Ver uso de memoria
free -h

# Limpiar Docker
docker system prune -a
```

## 📝 Comandos Útiles

```bash
# Ver estado de contenedores
docker ps

# Ver logs en tiempo real
docker-compose -f deployment/docker-compose.prod.yml logs -f

# Reiniciar un servicio
docker-compose -f deployment/docker-compose.prod.yml restart music-analyzer-ai

# Detener todo
docker-compose -f deployment/docker-compose.prod.yml down

# Iniciar todo
docker-compose -f deployment/docker-compose.prod.yml up -d

# Ejecutar comando en contenedor
docker-compose -f deployment/docker-compose.prod.yml exec music-analyzer-ai bash
```

## 🌐 Acceso desde Internet

### Obtener IP Pública

```bash
curl http://169.254.169.254/latest/meta-data/public-ipv4
```

### Acceder a la Aplicación

```
http://your-ec2-public-ip:8010
```

### Configurar Dominio

1. Crear registro A en tu DNS apuntando a la IP de EC2
2. Configurar Nginx con el dominio
3. Obtener certificado SSL con Let's Encrypt

## 💰 Optimización de Costos

### Instancias Spot

Usar instancias Spot para desarrollo/testing:

```bash
# Crear instancia Spot
aws ec2 request-spot-instances \
  --instance-count 1 \
  --launch-specification file://spot-specification.json
```

### Auto Scaling

Configurar Auto Scaling Group para escalar según demanda.

### Reserved Instances

Para producción, considerar Reserved Instances para ahorro.

## 📚 Recursos Adicionales

- [AWS EC2 Documentation](https://docs.aws.amazon.com/ec2/)
- [Docker Documentation](https://docs.docker.com/)
- [Docker Compose Documentation](https://docs.docker.com/compose/)




