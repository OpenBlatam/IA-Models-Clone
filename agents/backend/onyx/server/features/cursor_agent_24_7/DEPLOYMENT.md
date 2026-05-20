# 🚀 Guía de Despliegue - Cursor Agent 24/7

## 📦 Preparación

### 1. Instalar Dependencias

```bash
# Instalación completa
pip install -r requirements.txt

# O instalación mínima
pip install -r requirements-minimal.txt
```

### 2. Configurar Variables de Entorno

```bash
# Copiar archivo de ejemplo
cp .env.example .env

# Editar .env con tus configuraciones
```

## 🖥️ Despliegue Local

### Desarrollo

```bash
python main.py
```

### Producción

```bash
python main.py --mode api --port 8024
```

## 🐳 Despliegue con Docker

### Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8024

CMD ["python", "main.py", "--mode", "api", "--port", "8024"]
```

### docker-compose.yml

```yaml
version: '3.8'

services:
  cursor-agent:
    build: .
    ports:
      - "8024:8024"
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    environment:
      - AGENT_PERSISTENT_STORAGE=true
      - API_PORT=8024
    restart: unless-stopped
```

### Ejecutar

```bash
docker-compose up -d
```

## ☁️ Despliegue en la Nube

### AWS Lambda

```python
# lambda_handler.py
from mangum import Mangum
from cursor_agent_24_7.api.agent_api import create_app

app = create_app()
handler = Mangum(app)
```

### Google Cloud Run

```bash
gcloud run deploy cursor-agent \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --port 8024
```

### Azure Container Instances

```bash
az container create \
  --resource-group myResourceGroup \
  --name cursor-agent \
  --image myregistry/cursor-agent:latest \
  --ports 8024 \
  --environment-variables API_PORT=8024
```

## 🔧 Configuración como Servicio

### Windows

```bash
python scripts/install_service.py
```

O manualmente con NSSM:

```bash
nssm install CursorAgent24_7 "C:\Python\python.exe" "C:\path\to\main.py" --mode service
nssm start CursorAgent24_7
```

### Linux

```bash
sudo python scripts/install_service.py
```

O manualmente:

```bash
sudo systemctl enable cursor-agent-24-7
sudo systemctl start cursor-agent-24-7
```

### macOS

```bash
python scripts/install_service.py
```

O manualmente:

```bash
launchctl load ~/Library/LaunchAgents/com.cursor.agent24-7.plist
```

## 📊 Monitoreo

### Usar el monitor incluido

```bash
python scripts/monitor.py
```

### Con Prometheus

El agente expone métricas en formato Prometheus (preparado).

### Health Checks

```bash
# Verificar salud
curl http://localhost:8024/api/health

# Verificar estado
curl http://localhost:8024/api/status
```

## 🔒 Seguridad

### 1. Cambiar credenciales por defecto

```python
from cursor_agent_24_7.core.auth import AuthManager

auth = AuthManager()
auth.create_user("admin", "nueva_password_segura", Role.ADMIN)
```

### 2. Configurar HTTPS

Usar un reverse proxy como nginx:

```nginx
server {
    listen 443 ssl;
    server_name agent.example.com;
    
    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;
    
    location / {
        proxy_pass http://localhost:8024;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### 3. Firewall

```bash
# Permitir solo puerto 8024
ufw allow 8024/tcp
```

## 📈 Escalabilidad

### Horizontal Scaling

Ejecutar múltiples instancias:

```bash
# Instancia 1
python main.py --port 8024

# Instancia 2
python main.py --port 8025

# Instancia 3
python main.py --port 8026
```

Usar load balancer (nginx, HAProxy, etc.) para distribuir carga.

### Vertical Scaling

Aumentar recursos:
- Más memoria
- Más CPU
- Más almacenamiento

## 🔄 Actualización

### Backup antes de actualizar

```bash
curl -X POST http://localhost:8024/api/backups/create?name=pre_update_backup
```

### Actualizar código

```bash
git pull
pip install -r requirements.txt
```

### Reiniciar servicio

```bash
# Linux
sudo systemctl restart cursor-agent-24-7

# Windows
nssm restart CursorAgent24_7
```

## 🛠️ Mantenimiento

### Limpieza automática

```bash
python scripts/maintenance.py cleanup --days 30
```

### Generar reporte

```bash
python scripts/maintenance.py report
```

### Verificar salud

```bash
python scripts/maintenance.py health
```

### Todo

```bash
python scripts/maintenance.py all
```

## 📝 Checklist de Despliegue

- [ ] Dependencias instaladas
- [ ] Variables de entorno configuradas
- [ ] Credenciales cambiadas
- [ ] Backups configurados
- [ ] Monitoreo configurado
- [ ] Logs configurados
- [ ] Firewall configurado
- [ ] HTTPS configurado (producción)
- [ ] Servicio instalado
- [ ] Health checks funcionando
- [ ] Tests pasando

## 🆘 Troubleshooting

### El agente no inicia

1. Verificar logs: `tail -f logs/agent.log`
2. Verificar puerto: `netstat -an | grep 8024`
3. Verificar permisos: `ls -la data/`

### Tareas no se ejecutan

1. Verificar estado: `curl http://localhost:8024/api/status`
2. Verificar health: `curl http://localhost:8024/api/health`
3. Verificar logs para errores

### Problemas de memoria

1. Reducir `max_concurrent_tasks`
2. Reducir `max_history` en métricas
3. Limpiar tareas antiguas

## 📚 Más Información

- Ver [README.md](README.md) para documentación general
- Ver [API_REFERENCE.md](API_REFERENCE.md) para referencia de API
- Ver [TROUBLESHOOTING.md](TROUBLESHOOTING.md) para troubleshooting detallado



