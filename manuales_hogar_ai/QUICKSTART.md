# Quick Start Guide - Manuales Hogar AI

## 🚀 Inicio en 30 Segundos

### Paso 1: Clonar/Acceder al proyecto
```bash
cd agents/backend/onyx/server/features/manuales_hogar_ai
```

### Paso 2: Ejecutar un comando

**Linux/Mac:**
```bash
chmod +x start.sh
./start.sh
```

**Windows:**
```powershell
.\start.ps1
```

**Python (cualquier plataforma):**
```bash
python run.py
```

**Opciones avanzadas:**
```bash
# Sin reconstruir imágenes
./start.sh dev --no-build

# Saltar verificación de salud
./start.sh dev --skip-health

# Ejecutar migraciones automáticamente
./start.sh dev --migrate

# Producción
./start.sh prod
```

### Paso 3: ¡Listo! 🎉

El servicio estará disponible en:
- **API**: http://localhost:8000
- **Documentación**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/api/v1/health

## 📝 Configuración Inicial

La primera vez que ejecutes el script, se creará un archivo `.env` desde `.env.example`. 

**Edita `.env` y configura:**
```bash
OPENROUTER_API_KEY=tu-api-key-aqui
```

## 🛑 Detener el Servicio

**Linux/Mac:**
```bash
./stop.sh
```

**Windows:**
```powershell
.\stop.ps1
```

**O manualmente:**
```bash
docker-compose down
```

## 📋 Comandos Útiles

### Ver logs
```bash
docker-compose logs -f
```

### Reiniciar
```bash
docker-compose restart
```

### Acceder al contenedor
```bash
docker-compose exec app bash
```

### Ejecutar migraciones
```bash
docker-compose exec app alembic upgrade head
```

### Ver estado de servicios
```bash
# Script de estado (recomendado)
./status.sh        # Linux/Mac
.\status.ps1       # Windows

# O manualmente
docker-compose ps
```

### Verificar salud del servicio
```bash
./scripts/check-health.sh
```

### Configuración inicial
```bash
./scripts/setup.sh
```

## 🔧 Solución de Problemas

### Docker no está corriendo
```bash
# Inicia Docker Desktop o Docker daemon
```

### Puerto 8000 ya está en uso
```bash
# Cambia el puerto en docker-compose.yml
# O detén el servicio que usa el puerto
```

### Error de conexión a base de datos
```bash
# Espera unos segundos más, la BD puede tardar en iniciar
# Verifica logs: docker-compose logs postgres
```

### El servicio no responde
```bash
# Verifica logs
docker-compose logs app

# Reinicia servicios
docker-compose restart
```

## 📚 Más Información

- **Documentación completa**: [README.md](README.md)
- **Guía de Docker**: [DOCKER.md](DOCKER.md)
- **Despliegue en AWS**: [AWS_DEPLOYMENT.md](AWS_DEPLOYMENT.md)

## 🎯 Próximos Pasos

1. Configura tu `OPENROUTER_API_KEY` en `.env`
2. Prueba el API en http://localhost:8000/docs
3. Genera tu primer manual con una imagen o texto
4. Explora todas las funcionalidades disponibles

¡Disfruta usando Manuales Hogar AI! 🏠🔧

