# 🚀 Inicio con un Solo Comando

## Inicio Rápido

### Opción 1: Script Shell (Linux/Mac)

```bash
# Desarrollo
./deployment/start.sh

# Producción
./deployment/start.sh prod
```

### Opción 2: Script Windows

```cmd
REM Desarrollo
deployment\start.bat

REM Producción
deployment\start.bat prod
```

### Opción 3: Python (Multiplataforma)

```bash
# Desarrollo
python deployment/start.py

# Producción
python deployment/start.py prod
```

### Opción 4: Make (Linux/Mac)

```bash
# Desarrollo (por defecto)
make start

# Producción
make start ENV=prod
```

## ¿Qué hace el script?

1. ✅ Verifica que Docker esté corriendo
2. ✅ Verifica que docker-compose esté instalado
3. ✅ Crea archivo `.env` si no existe (producción)
4. ✅ Construye las imágenes Docker
5. ✅ Inicia todos los servicios
6. ✅ Espera a que los servicios estén listos
7. ✅ Verifica el health check
8. ✅ Muestra las URLs de los servicios

## Servicios Iniciados

Después de ejecutar el comando, tendrás acceso a:

- **🌐 API**: http://localhost:8010
- **❤️ Health**: http://localhost:8010/health
- **📖 Docs**: http://localhost:8010/docs
- **📈 Grafana**: http://localhost:3000 (dev/prod)
- **📊 Prometheus**: http://localhost:9090 (dev/prod)

## Detener Servicios

### Opción 1: Script Shell

```bash
./deployment/stop.sh
```

### Opción 2: Script Windows

```cmd
deployment\stop.bat
```

### Opción 3: Make

```bash
make stop
```

### Opción 4: Docker Compose

```bash
docker-compose down
```

## Requisitos Previos

1. **Docker** instalado y corriendo
2. **Docker Compose** instalado
3. (Opcional) **curl** para health checks

## Configuración

### Variables de Entorno

El script crea automáticamente un archivo `.env` si no existe. Para producción, edita el archivo `.env` en la raíz del proyecto:

```env
ENVIRONMENT=production
SPOTIFY_CLIENT_ID=tu_client_id_real
SPOTIFY_CLIENT_SECRET=tu_client_secret_real
POSTGRES_PASSWORD=password_seguro
REDIS_PASSWORD=password_seguro
```

## Troubleshooting

### Docker no está corriendo

```bash
# Linux/Mac
sudo systemctl start docker

# Windows
# Inicia Docker Desktop
```

### Puerto en uso

Si el puerto 8010 está en uso, cambia el puerto en `docker-compose.yml`:

```yaml
ports:
  - "8011:8010"  # Usa puerto diferente
```

### Ver logs

```bash
docker-compose logs -f
```

### Reiniciar servicios

```bash
docker-compose restart
```

## Ejemplos de Uso

### Desarrollo Local

```bash
# Iniciar todo
./deployment/start.sh dev

# Ver logs
docker-compose -f deployment/docker-compose.dev.yml logs -f

# Detener
./deployment/stop.sh dev
```

### Producción

```bash
# Configurar .env primero
nano .env

# Iniciar
./deployment/start.sh prod

# Verificar
curl http://localhost:8010/health
```

## Comandos Útiles

```bash
# Ver estado de contenedores
docker ps

# Ver logs de un servicio específico
docker-compose logs -f music-analyzer-ai

# Ejecutar comando en contenedor
docker-compose exec music-analyzer-ai bash

# Reconstruir imágenes
docker-compose build --no-cache

# Limpiar todo
docker-compose down -v
```

## Siguiente Paso

Una vez que todo esté corriendo:

1. ✅ Verifica el health check: http://localhost:8010/health
2. ✅ Revisa la documentación: http://localhost:8010/docs
3. ✅ Configura tus credenciales de Spotify en `.env`
4. ✅ Explora los dashboards de monitoreo

¡Listo! 🎉




