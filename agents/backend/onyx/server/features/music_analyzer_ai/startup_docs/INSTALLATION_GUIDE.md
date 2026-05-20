# Guía de Instalación - Music Analyzer AI

## 📋 Prerrequisitos

### Requisitos del Sistema

- **Python**: 3.8 o superior
- **pip**: Gestor de paquetes de Python
- **Git**: Para clonar el repositorio (opcional)

### Requisitos Adicionales

- **Cuenta de Spotify Developer**: Para obtener credenciales de API
- **Redis** (opcional): Para caché (recomendado para producción)
- **PostgreSQL** (opcional): Para base de datos (si se usa)

## 🔧 Instalación Paso a Paso

### Paso 1: Clonar o Navegar al Proyecto

```bash
# Si es un repositorio Git
git clone <repository-url>
cd music_analyzer_ai

# O navegar al directorio
cd agents/backend/onyx/server/features/music_analyzer_ai
```

### Paso 2: Crear Entorno Virtual (Recomendado)

#### Windows

```cmd
python -m venv venv
venv\Scripts\activate
```

#### Linux/Mac

```bash
python3 -m venv venv
source venv/bin/activate
```

### Paso 3: Instalar Dependencias

```bash
pip install -r requirements.txt
```

**Nota**: Algunas dependencias pueden requerir compilación. Si tienes problemas:

```bash
# Instalar herramientas de compilación
# Windows: Instalar Visual Studio Build Tools
# Linux: sudo apt-get install build-essential
# Mac: xcode-select --install
```

### Paso 4: Configurar Spotify API

#### 4.1. Crear Aplicación en Spotify

1. Ve a [Spotify Developer Dashboard](https://developer.spotify.com/dashboard)
2. Inicia sesión con tu cuenta de Spotify
3. Haz clic en "Create an App"
4. Completa el formulario:
   - **App name**: Music Analyzer AI (o el nombre que prefieras)
   - **App description**: Sistema de análisis musical
   - **Website**: http://localhost:8010 (o tu URL)
   - **Redirect URI**: `http://localhost:8010/callback`
5. Acepta los términos y condiciones
6. Haz clic en "Save"

#### 4.2. Obtener Credenciales

1. En el dashboard, haz clic en tu aplicación
2. Copia el **Client ID**
3. Haz clic en "Show Client Secret" y copia el **Client Secret**
4. Guarda estas credenciales de forma segura

### Paso 5: Configurar Variables de Entorno

Crea un archivo `.env` en la raíz del proyecto:

```env
# Spotify API (REQUERIDO)
SPOTIFY_CLIENT_ID=tu_client_id_aqui
SPOTIFY_CLIENT_SECRET=tu_client_secret_aqui
SPOTIFY_REDIRECT_URI=http://localhost:8010/callback

# Server Configuration
HOST=0.0.0.0
PORT=8010
DEBUG=True
ENVIRONMENT=development

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json

# Redis Configuration (opcional, para caché)
REDIS_URL=redis://localhost:6379/0

# Database Configuration (opcional)
DATABASE_URL=postgresql://user:password@localhost:5432/music_analyzer

# CORS (si tienes frontend)
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:8080
```

### Paso 6: Verificar Instalación

```bash
# Verificar que Python puede importar los módulos
python -c "from api.music_api import router; print('OK')"

# Verificar dependencias principales
python -c "import fastapi, uvicorn, spotipy; print('Dependencias OK')"
```

## 🚀 Iniciar el Servidor

### Desarrollo

```bash
python main.py
```

O con uvicorn directamente:

```bash
uvicorn main:app --host 0.0.0.0 --port 8010 --reload
```

### Producción

```bash
uvicorn main:app --host 0.0.0.0 --port 8010 --workers 4
```

## 🐳 Instalación con Docker

### Construir Imagen

```bash
docker build -t music-analyzer-ai .
```

### Ejecutar Contenedor

```bash
docker run -d \
  -p 8010:8010 \
  -e SPOTIFY_CLIENT_ID=tu_client_id \
  -e SPOTIFY_CLIENT_SECRET=tu_client_secret \
  -e HOST=0.0.0.0 \
  -e PORT=8010 \
  --name music-analyzer-ai \
  music-analyzer-ai
```

### Docker Compose (si está disponible)

```bash
# Windows
deployment\start.bat

# Linux/Mac
./deployment/start.sh
```

## 🔍 Verificación

### 1. Health Check

```bash
curl http://localhost:8010/health
```

Respuesta esperada:
```json
{
  "status": "healthy",
  "version": "2.21.0",
  "timestamp": "2025-01-XX..."
}
```

### 2. API Documentation

Abre en el navegador:
- http://localhost:8010/docs (Swagger UI)
- http://localhost:8010/redoc (ReDoc)

### 3. Test de Búsqueda

```bash
# Test de búsqueda de canciones
curl -X POST http://localhost:8010/music/search \
  -H "Content-Type: application/json" \
  -d '{"query": "Bohemian Rhapsody", "limit": 5}'
```

## 🛠️ Configuración Avanzada

### Redis para Caché

1. Instalar Redis:
```bash
# Ubuntu/Debian
sudo apt-get install redis-server

# Mac
brew install redis

# Windows
# Descargar de https://redis.io/download
```

2. Iniciar Redis:
```bash
redis-server
```

3. Configurar en `.env`:
```env
REDIS_URL=redis://localhost:6379/0
```

### PostgreSQL (si se usa)

1. Instalar PostgreSQL
2. Crear base de datos:
```sql
CREATE DATABASE music_analyzer;
CREATE USER music_user WITH PASSWORD 'tu_password';
GRANT ALL PRIVILEGES ON DATABASE music_analyzer TO music_user;
```

3. Configurar en `.env`:
```env
DATABASE_URL=postgresql://music_user:tu_password@localhost:5432/music_analyzer
```

### Configuración de Logging

```python
# En config/settings.py o .env
LOG_LEVEL=DEBUG  # DEBUG, INFO, WARNING, ERROR
LOG_FORMAT=json  # json o text
LOG_FILE=logs/app.log  # Opcional: archivo de log
```

## 🧪 Testing

### Instalar Dependencias de Desarrollo

```bash
pip install pytest pytest-asyncio pytest-cov
```

### Ejecutar Tests

```bash
# Todos los tests
pytest

# Tests específicos
pytest tests/test_spotify_service.py

# Con cobertura
pytest --cov=api --cov=services --cov-report=html
```

## 📦 Dependencias Principales

### Core Dependencies

- `fastapi` - Framework web
- `uvicorn` - Servidor ASGI
- `pydantic` - Validación de datos
- `spotipy` - Cliente de Spotify API

### Audio Processing

- `librosa` - Análisis de audio
- `numpy` - Operaciones numéricas
- `scipy` - Cálculos científicos
- `soundfile` - I/O de audio

### ML/AI (Opcional)

- `torch` - PyTorch para modelos
- `transformers` - Modelos de Hugging Face

### Optional Dependencies

- `redis` - Cliente Redis para caché
- `psycopg2` - PostgreSQL adapter
- `httpx` - Cliente HTTP async
- `structlog` - Logging estructurado

## 🚨 Troubleshooting

### Error: ModuleNotFoundError

```bash
# Verificar que el entorno virtual está activado
which python  # Linux/Mac
where python  # Windows

# Reinstalar dependencias
pip install -r requirements.txt --force-reinstall
```

### Error: Spotify API Authentication Failed

```bash
# Verificar credenciales en .env
# Verificar que Client ID y Secret son correctos
# Verificar que Redirect URI coincide con el configurado en Spotify
```

### Error: Puerto ya en uso

```bash
# Encontrar proceso usando el puerto
lsof -i :8010  # Linux/Mac
netstat -ano | findstr :8010  # Windows

# Cambiar puerto en .env o directamente
uvicorn main:app --port 8011
```

### Error: librosa no se instala

```bash
# Instalar dependencias del sistema primero
# Ubuntu/Debian
sudo apt-get install libsndfile1 ffmpeg

# Mac
brew install libsndfile ffmpeg

# Luego reinstalar librosa
pip install librosa --force-reinstall
```

### Error: Variables de entorno no cargadas

```bash
# Verificar que .env existe
ls -la .env  # Linux/Mac
dir .env     # Windows

# Cargar manualmente
export SPOTIFY_CLIENT_ID=tu_id  # Linux/Mac
set SPOTIFY_CLIENT_ID=tu_id     # Windows CMD
$env:SPOTIFY_CLIENT_ID="tu_id"  # Windows PowerShell
```

## 📚 Próximos Pasos

1. ✅ Verificar instalación con health check
2. 📖 Revisar documentación de API en `/docs`
3. 🎵 Probar búsqueda de canciones
4. 🎼 Probar análisis de canciones
5. 🧪 Ejecutar tests para verificar funcionalidad
6. 🔧 Configurar Redis y base de datos si es necesario
7. 🚀 Comenzar a usar la API

## 🔗 Recursos Adicionales

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Spotify Web API Documentation](https://developer.spotify.com/documentation/web-api/)
- [Spotipy Documentation](https://spotipy.readthedocs.io/)
- [Librosa Documentation](https://librosa.org/doc/latest/index.html)

---

**Última actualización**: 2025  
**Versión**: 2.21.0






