# 🚀 Guía de Deployment - Social Media Identity Clone AI

## Requisitos Previos

- Python 3.10+
- PostgreSQL (opcional, SQLite por defecto)
- Redis (opcional, para producción)
- OpenAI API Key

## Instalación

### 1. Clonar y Configurar

```bash
cd social_media_identity_clone_ai
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configurar Variables de Entorno

```bash
cp .env.example .env
# Editar .env con tus configuraciones
```

Variables importantes:
- `OPENAI_API_KEY` - Tu API key de OpenAI
- `DATABASE_URL` - URL de base de datos
- `API_KEY` - API key para autenticación

### 3. Inicializar Base de Datos

```bash
python scripts/init_db.py
```

### 4. Ejecutar Migraciones (si aplica)

```bash
python scripts/migrate.py
```

## Desarrollo

### Ejecutar Servidor de Desarrollo

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

O usando el script:

```bash
python run_api.py
```

## Producción

### Opción 1: Docker

```bash
# Construir imagen
docker build -t social-media-identity-clone .

# Ejecutar
docker run -d \
  -p 8000:8000 \
  -e OPENAI_API_KEY=your_key \
  -e DATABASE_URL=postgresql://... \
  social-media-identity-clone
```

### Opción 2: Docker Compose

```bash
docker-compose up -d
```

### Opción 3: Systemd Service

Crear `/etc/systemd/system/social-media-identity-clone.service`:

```ini
[Unit]
Description=Social Media Identity Clone AI
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/opt/social-media-identity-clone
Environment="PATH=/opt/social-media-identity-clone/venv/bin"
ExecStart=/opt/social-media-identity-clone/venv/bin/uvicorn api.main:app --host 0.0.0.0 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
```

Activar servicio:

```bash
sudo systemctl enable social-media-identity-clone
sudo systemctl start social-media-identity-clone
```

## Nginx Reverse Proxy

Configuración de ejemplo `/etc/nginx/sites-available/social-media-identity-clone`:

```nginx
server {
    listen 80;
    server_name api.example.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

## Monitoreo

### Health Checks

```bash
# Health check básico
curl http://localhost:8000/health

# Health check completo
curl http://localhost:8000/api/v1/health
```

### Métricas de Performance

```bash
curl http://localhost:8000/api/v1/performance
```

### Logs

```bash
# Ver logs en tiempo real
tail -f logs/app.log

# Con Docker
docker logs -f social-media-identity-clone
```

## Backup

### Backup Manual

```bash
python scripts/backup_db.py
```

### Backup Automático (Cron)

```bash
# Agregar a crontab
0 2 * * * cd /opt/social-media-identity-clone && python scripts/backup_db.py
```

## Mantenimiento

### Limpieza de Datos Antiguos

```bash
python scripts/cleanup.py
```

### Verificar Estado

```bash
# Health check
curl http://localhost:8000/health

# Performance
curl http://localhost:8000/api/v1/performance
```

## Escalabilidad

### Horizontal Scaling

1. Usar load balancer (Nginx, HAProxy)
2. Múltiples instancias de la API
3. Base de datos compartida (PostgreSQL)
4. Redis para cache compartido

### Vertical Scaling

1. Aumentar workers: `NUM_WORKERS=4`
2. Aumentar memoria
3. Optimizar queries de BD

## Seguridad

### Producción

1. Cambiar `require_api_key=True` en SecurityMiddleware
2. Usar HTTPS (Let's Encrypt)
3. Configurar CORS apropiadamente
4. Rate limiting activo
5. Logs de seguridad

### Variables de Entorno Sensibles

Nunca commitear:
- API keys
- Database passwords
- Secret keys

Usar secret management (AWS Secrets Manager, HashiCorp Vault, etc.)

## Troubleshooting

### Problemas Comunes

1. **Error de conexión a BD**
   - Verificar `DATABASE_URL`
   - Verificar permisos

2. **Error de OpenAI**
   - Verificar `OPENAI_API_KEY`
   - Verificar límites de API

3. **Alto uso de memoria**
   - Reducir `NUM_WORKERS`
   - Limpiar datos antiguos

4. **Lentitud**
   - Verificar performance metrics
   - Optimizar queries
   - Aumentar cache

## Actualización

```bash
# 1. Backup
python scripts/backup_db.py

# 2. Pull cambios
git pull

# 3. Actualizar dependencias
pip install -r requirements.txt

# 4. Migraciones
python scripts/migrate.py

# 5. Reiniciar
sudo systemctl restart social-media-identity-clone
```

## Checklist de Deployment

- [ ] Variables de entorno configuradas
- [ ] Base de datos inicializada
- [ ] Migraciones aplicadas
- [ ] API key configurada
- [ ] Health check funcionando
- [ ] Logs configurados
- [ ] Backups configurados
- [ ] Monitoreo activo
- [ ] Seguridad configurada
- [ ] Rate limiting activo
- [ ] CORS configurado
- [ ] HTTPS configurado (producción)




