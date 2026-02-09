# Guía de Configuración - Music Analyzer AI

## 📋 Variables de Entorno

### Variables Requeridas

```env
# Spotify API (OBLIGATORIO)
SPOTIFY_CLIENT_ID=tu_client_id_aqui
SPOTIFY_CLIENT_SECRET=tu_client_secret_aqui
SPOTIFY_REDIRECT_URI=http://localhost:8010/callback
```

### Variables del Servidor

```env
# Host y Puerto
HOST=0.0.0.0              # Host del servidor (0.0.0.0 para todas las interfaces)
PORT=8010                 # Puerto del servidor

# Entorno
DEBUG=True                # Modo debug (True/False)
ENVIRONMENT=development   # development, staging, production
```

### Variables de Logging

```env
# Nivel de Log
LOG_LEVEL=INFO           # DEBUG, INFO, WARNING, ERROR, CRITICAL

# Formato de Log
LOG_FORMAT=json          # json o text

# Archivo de Log (opcional)
LOG_FILE=logs/app.log     # Ruta al archivo de log
```

### Variables de Análisis de Audio

```env
# Configuración de Audio
AUDIO_SAMPLE_RATE=44100   # Tasa de muestreo (Hz)
AUDIO_CHUNK_SIZE=1024     # Tamaño de chunk para procesamiento

# Límites de Análisis
MAX_AUDIO_DURATION=300    # Duración máxima en segundos (5 minutos)
ANALYSIS_DETAIL_LEVEL=detailed  # basic, detailed, expert
```

### Variables de Caché

```env
# Redis (opcional pero recomendado)
REDIS_URL=redis://localhost:6379/0

# Configuración de Caché
CACHE_ENABLED=True        # Habilitar/deshabilitar caché
CACHE_TTL=3600           # Tiempo de vida en segundos (1 hora)
```

### Variables de Base de Datos

```env
# PostgreSQL (opcional)
DATABASE_URL=postgresql://user:password@localhost:5432/music_analyzer

# SQLite (por defecto)
DATABASE_URL=sqlite:///./music_analyzer.db
```

### Variables de CORS

```env
# Orígenes permitidos (separados por coma)
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:8080
```

## 🔧 Configuración por Entorno

### Desarrollo (.env.development)

```env
# Desarrollo
HOST=0.0.0.0
PORT=8010
DEBUG=True
ENVIRONMENT=development
LOG_LEVEL=DEBUG
CACHE_ENABLED=False
```

### Staging (.env.staging)

```env
# Staging
HOST=0.0.0.0
PORT=8010
DEBUG=False
ENVIRONMENT=staging
LOG_LEVEL=INFO
CACHE_ENABLED=True
REDIS_URL=redis://staging-redis:6379/0
```

### Producción (.env.production)

```env
# Producción
HOST=0.0.0.0
PORT=8010
DEBUG=False
ENVIRONMENT=production
LOG_LEVEL=WARNING
CACHE_ENABLED=True
REDIS_URL=redis://production-redis:6379/0
DATABASE_URL=postgresql://user:pass@prod-db:5432/music_analyzer
```

## 🎛️ Configuración Avanzada

### Rate Limiting

```env
# Rate Limiting
RATE_LIMIT_ENABLED=True
RATE_LIMIT_MAX_REQUESTS=100
RATE_LIMIT_WINDOW_SECONDS=60
```

### Spotify API

```env
# Configuración avanzada de Spotify
SPOTIFY_API_TIMEOUT=30        # Timeout en segundos
SPOTIFY_RETRY_ATTEMPTS=3      # Intentos de reintento
SPOTIFY_RETRY_DELAY=1         # Delay entre reintentos (segundos)
```

### Análisis de Audio

```env
# Configuración avanzada de análisis
ANALYSIS_PARALLEL_WORKERS=4   # Workers paralelos
ANALYSIS_BATCH_SIZE=10        # Tamaño de batch
ANALYSIS_USE_GPU=False        # Usar GPU si está disponible
```

### Monitoring

```env
# Prometheus (opcional)
PROMETHEUS_ENABLED=True
PROMETHEUS_PORT=9090

# OpenTelemetry (opcional)
OTEL_ENABLED=False
OTEL_ENDPOINT=http://localhost:4317
```

## 📝 Archivo .env Completo de Ejemplo

```env
# ============================================
# Music Analyzer AI - Configuración Completa
# ============================================

# Spotify API (REQUERIDO)
SPOTIFY_CLIENT_ID=tu_client_id_aqui
SPOTIFY_CLIENT_SECRET=tu_client_secret_aqui
SPOTIFY_REDIRECT_URI=http://localhost:8010/callback
SPOTIFY_API_TIMEOUT=30
SPOTIFY_RETRY_ATTEMPTS=3
SPOTIFY_RETRY_DELAY=1

# Server Configuration
HOST=0.0.0.0
PORT=8010
DEBUG=True
ENVIRONMENT=development

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json
LOG_FILE=logs/app.log

# Audio Analysis
AUDIO_SAMPLE_RATE=44100
AUDIO_CHUNK_SIZE=1024
MAX_AUDIO_DURATION=300
ANALYSIS_DETAIL_LEVEL=detailed
ANALYSIS_PARALLEL_WORKERS=4
ANALYSIS_BATCH_SIZE=10
ANALYSIS_USE_GPU=False

# Cache
CACHE_ENABLED=True
CACHE_TTL=3600
REDIS_URL=redis://localhost:6379/0

# Database
DATABASE_URL=sqlite:///./music_analyzer.db
# Para PostgreSQL:
# DATABASE_URL=postgresql://user:password@localhost:5432/music_analyzer

# Rate Limiting
RATE_LIMIT_ENABLED=True
RATE_LIMIT_MAX_REQUESTS=100
RATE_LIMIT_WINDOW_SECONDS=60

# CORS
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:8080

# Monitoring (opcional)
PROMETHEUS_ENABLED=False
PROMETHEUS_PORT=9090
OTEL_ENABLED=False
OTEL_ENDPOINT=http://localhost:4317
```

## 🔐 Seguridad

### Variables Sensibles

**NUNCA** commits estas variables al repositorio:

- `SPOTIFY_CLIENT_SECRET`
- `DATABASE_URL` (si contiene password)
- Cualquier API key o token

### Mejores Prácticas

1. Usa `.env` para desarrollo local
2. Usa variables de entorno del sistema en producción
3. Usa secret managers (AWS Secrets Manager, HashiCorp Vault) en producción
4. Nunca hardcodees credenciales en el código

## 🧪 Validación de Configuración

### Verificar Configuración

```python
from config.settings import settings

# Verificar que las variables requeridas están configuradas
assert settings.SPOTIFY_CLIENT_ID, "SPOTIFY_CLIENT_ID no configurado"
assert settings.SPOTIFY_CLIENT_SECRET, "SPOTIFY_CLIENT_SECRET no configurado"
```

### Script de Validación

```bash
# Crear script de validación
python scripts/validate_config.py
```

## 📚 Referencias

- **Settings Module**: [../config/settings.py](../config/settings.py)
- **DI Setup**: [../config/di_setup.py](../config/di_setup.py)

---

**Última actualización**: 2025  
**Versión**: 2.21.0






