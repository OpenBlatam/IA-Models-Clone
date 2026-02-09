# Guía de Solución de Problemas - Music Analyzer AI

## 🚨 Problemas Comunes

### 1. El servidor no inicia

#### Error: `ModuleNotFoundError`

**Síntoma**:
```
ModuleNotFoundError: No module named 'fastapi'
```

**Solución**:
```bash
# Instalar dependencias
pip install -r requirements.txt

# O instalar específicamente
pip install fastapi uvicorn
```

#### Error: Puerto ya en uso

**Síntoma**:
```
ERROR:    [Errno 48] Address already in use
```

**Solución**:
```bash
# Opción 1: Cambiar puerto
uvicorn main:app --port 8011

# Opción 2: Encontrar y matar el proceso
# Linux/Mac
lsof -i :8010
kill -9 <PID>

# Windows
netstat -ano | findstr :8010
taskkill /PID <PID> /F
```

#### Error: Variables de entorno no cargadas

**Síntoma**:
```
KeyError: 'SPOTIFY_CLIENT_ID'
```

**Solución**:
```bash
# Verificar que .env existe
ls -la .env  # Linux/Mac
dir .env     # Windows

# Verificar contenido
cat .env     # Linux/Mac
type .env    # Windows

# Cargar manualmente (temporal)
export SPOTIFY_CLIENT_ID=tu_id  # Linux/Mac
set SPOTIFY_CLIENT_ID=tu_id     # Windows CMD
```

### 2. Errores de Spotify API

#### Error: `401 Unauthorized`

**Síntoma**:
```
HTTPError: 401 Client Error: Unauthorized
```

**Causas posibles**:
- Credenciales incorrectas
- Client Secret expirado
- Redirect URI no coincide

**Solución**:
1. Verificar credenciales en `.env`
2. Verificar que el Redirect URI en Spotify Dashboard coincide con `.env`
3. Regenerar Client Secret si es necesario

#### Error: `429 Too Many Requests`

**Síntoma**:
```
HTTPError: 429 Client Error: Too Many Requests
```

**Solución**:
```python
# Implementar rate limiting
# Ya está implementado en el middleware
# Ajustar en .env:
RATE_LIMIT_MAX_REQUESTS=50
RATE_LIMIT_WINDOW_SECONDS=60
```

#### Error: Track no encontrado

**Síntoma**:
```
404: Track not found
```

**Solución**:
- Verificar que el `track_id` es válido
- Verificar que el track existe en Spotify
- Usar búsqueda primero para obtener el ID correcto

### 3. Errores de Dependencias

#### Error: `librosa` no se instala

**Síntoma**:
```
ERROR: Failed building wheel for librosa
```

**Solución**:
```bash
# Linux
sudo apt-get install libsndfile1 ffmpeg

# Mac
brew install libsndfile ffmpeg

# Windows
# Instalar Visual Studio Build Tools
# O usar conda:
conda install -c conda-forge librosa
```

#### Error: `torch` no se instala

**Síntoma**:
```
ERROR: Could not find a version that satisfies the requirement torch
```

**Solución**:
```bash
# Instalar PyTorch desde el sitio oficial
# https://pytorch.org/get-started/locally/
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### 4. Errores de Base de Datos

#### Error: Conexión a PostgreSQL falla

**Síntoma**:
```
psycopg2.OperationalError: could not connect to server
```

**Solución**:
```bash
# Verificar que PostgreSQL está corriendo
# Linux
sudo systemctl status postgresql

# Verificar conexión
psql -U user -d music_analyzer -h localhost

# Verificar DATABASE_URL en .env
DATABASE_URL=postgresql://user:password@localhost:5432/music_analyzer
```

### 5. Errores de Caché

#### Error: Redis no conecta

**Síntoma**:
```
redis.exceptions.ConnectionError: Error connecting to Redis
```

**Solución**:
```bash
# Verificar que Redis está corriendo
redis-cli ping
# Debe responder: PONG

# Iniciar Redis si no está corriendo
redis-server

# Verificar REDIS_URL en .env
REDIS_URL=redis://localhost:6379/0
```

### 6. Errores de Importación

#### Error: Import circular

**Síntoma**:
```
ImportError: cannot import name 'X' from partially initialized module
```

**Solución**:
- Revisar imports circulares
- Usar imports locales cuando sea necesario
- Reorganizar estructura de módulos

#### Error: Módulo no encontrado

**Síntoma**:
```
ModuleNotFoundError: No module named 'api.music_api'
```

**Solución**:
```bash
# Verificar que estás en el directorio correcto
pwd  # Linux/Mac
cd  # Windows

# Debe estar en: agents/backend/onyx/server/features/music_analyzer_ai

# Agregar al PYTHONPATH si es necesario
export PYTHONPATH="${PYTHONPATH}:$(pwd)"  # Linux/Mac
set PYTHONPATH=%PYTHONPATH%;%CD%          # Windows CMD
```

## 🔍 Diagnóstico

### Verificar Estado del Sistema

```bash
# Health check
curl http://localhost:8010/health

# Verificar logs
tail -f logs/app.log  # Linux/Mac
type logs\app.log     # Windows
```

### Verificar Configuración

```python
# Script de diagnóstico
python scripts/diagnose.py
```

### Verificar Dependencias

```bash
# Listar dependencias instaladas
pip list | grep -E "fastapi|uvicorn|spotipy"

# Verificar versiones
python -c "import fastapi; print(fastapi.__version__)"
python -c "import uvicorn; print(uvicorn.__version__)"
python -c "import spotipy; print(spotipy.__version__)"
```

## 🛠️ Soluciones Avanzadas

### Limpiar Caché de Python

```bash
# Limpiar __pycache__
find . -type d -name __pycache__ -exec rm -r {} +  # Linux/Mac
Get-ChildItem -Path . -Recurse -Filter __pycache__ | Remove-Item -Recurse -Force  # Windows PowerShell

# Limpiar .pyc
find . -name "*.pyc" -delete  # Linux/Mac
```

### Reinstalar Dependencias

```bash
# Limpiar e reinstalar
pip uninstall -r requirements.txt -y
pip install -r requirements.txt
```

### Resetear Base de Datos

```bash
# SQLite
rm music_analyzer.db

# PostgreSQL
psql -U user -d music_analyzer -c "DROP SCHEMA public CASCADE; CREATE SCHEMA public;"
```

## 📞 Obtener Ayuda

### Logs

Los logs contienen información detallada:

```bash
# Ver logs en tiempo real
tail -f logs/app.log

# Buscar errores
grep -i error logs/app.log

# Buscar warnings
grep -i warning logs/app.log
```

### Debug Mode

Activar modo debug para más información:

```env
DEBUG=True
LOG_LEVEL=DEBUG
```

### Reportar Problemas

Al reportar un problema, incluye:

1. Versión de Python: `python --version`
2. Versión del sistema: `uname -a` (Linux/Mac) o `systeminfo` (Windows)
3. Logs relevantes
4. Pasos para reproducir
5. Configuración (sin credenciales)

## 📚 Recursos Adicionales

- **Documentación de FastAPI**: https://fastapi.tiangolo.com/troubleshooting/
- **Documentación de Spotify API**: https://developer.spotify.com/documentation/web-api/
- **Issues en GitHub**: (si hay repositorio)

---

**Última actualización**: 2025  
**Versión**: 2.21.0






