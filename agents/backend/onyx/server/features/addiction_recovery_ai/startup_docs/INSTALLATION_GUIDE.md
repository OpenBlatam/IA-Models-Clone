# Guía de Instalación - Addiction Recovery AI

## 📋 Prerrequisitos

### Requisitos del Sistema

- **Python**: 3.8 o superior
- **pip**: Gestor de paquetes de Python
- **Git**: Para clonar el repositorio (opcional)

### Recomendado

- **PostgreSQL**: 12+ (para base de datos)
- **Redis**: 6+ (para caché)
- **Docker**: 20+ (para despliegue con contenedores)

## 🔧 Instalación Paso a Paso

### Paso 1: Clonar o Navegar al Proyecto

```bash
# Si es un repositorio Git
git clone <repository-url>
cd addiction_recovery_ai

# O navegar al directorio
cd agents/backend/onyx/server/features/addiction_recovery_ai
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

#### Instalación Completa

```bash
pip install -r requirements.txt
```

#### Instalación Mínima

```bash
pip install -r requirements-minimal.txt
```

#### Instalación Optimizada para Velocidad

```bash
pip install -r requirements-speed.txt
```

### Paso 4: Configurar Variables de Entorno

Crea un archivo `.env` en la raíz del proyecto:

```env
# Server Configuration
HOST=0.0.0.0
PORT=8020
DEBUG=True
ENVIRONMENT=development

# API Keys (si usas servicios externos)
OPENAI_API_KEY=tu_openai_api_key
ANTHROPIC_API_KEY=tu_anthropic_api_key

# Database Configuration (opcional)
DATABASE_URL=postgresql://user:password@localhost:5432/addiction_recovery

# Redis Configuration (opcional, para caché)
REDIS_URL=redis://localhost:6379/0

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json

# Security
SECRET_KEY=tu_secret_key_muy_segura_aqui
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# CORS (si tienes frontend)
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:8080
```

### Paso 5: Verificar Instalación

```bash
# Verificar que Python puede importar los módulos
python -c "from core.app_factory import create_app; print('OK')"

# Verificar dependencias principales
python -c "import fastapi, uvicorn, pydantic; print('Dependencias OK')"
```

## 🚀 Iniciar el Servidor

### Desarrollo

```bash
python main.py
```

O con uvicorn directamente:

```bash
uvicorn main:app --host 0.0.0.0 --port 8020 --reload
```

### Producción

```bash
uvicorn main:app --host 0.0.0.0 --port 8020 --workers 4
```

## 🐳 Instalación con Docker

### Construir Imagen

```bash
docker build -t addiction-recovery-ai .
```

### Ejecutar Contenedor

```bash
docker run -d \
  -p 8020:8020 \
  -e HOST=0.0.0.0 \
  -e PORT=8020 \
  --name addiction-recovery-ai \
  addiction-recovery-ai
```

### Docker Compose (si está disponible)

```bash
docker-compose up -d
```

## 🔍 Verificación

### 1. Health Check

```bash
curl http://localhost:8020/health
```

Respuesta esperada:
```json
{
  "status": "healthy",
  "version": "3.4.0",
  "timestamp": "2025-01-XX..."
}
```

### 2. API Documentation

Abre en el navegador:
- http://localhost:8020/docs (Swagger UI)
- http://localhost:8020/redoc (ReDoc)

### 3. Test de Endpoint

```bash
# Test de evaluación
curl -X POST http://localhost:8020/api/assessment/create \
  -H "Content-Type: application/json" \
  -d '{"user_id": "test", "substance_type": "alcohol"}'
```

## 🛠️ Configuración Avanzada

### Base de Datos PostgreSQL

1. Instalar PostgreSQL
2. Crear base de datos:
```sql
CREATE DATABASE addiction_recovery;
CREATE USER recovery_user WITH PASSWORD 'tu_password';
GRANT ALL PRIVILEGES ON DATABASE addiction_recovery TO recovery_user;
```

3. Configurar en `.env`:
```env
DATABASE_URL=postgresql://recovery_user:tu_password@localhost:5432/addiction_recovery
```

### Redis para Caché

1. Instalar Redis
2. Iniciar Redis:
```bash
redis-server
```

3. Configurar en `.env`:
```env
REDIS_URL=redis://localhost:6379/0
```

### Configuración de Logging

```python
# En config/app_config.py o .env
LOG_LEVEL=DEBUG  # DEBUG, INFO, WARNING, ERROR
LOG_FORMAT=json  # json o text
LOG_FILE=logs/app.log  # Opcional: archivo de log
```

## 🧪 Testing

### Instalar Dependencias de Desarrollo

```bash
pip install -r requirements-dev.txt
```

### Ejecutar Tests

```bash
# Todos los tests
pytest

# Tests específicos
pytest tests/test_assessment.py

# Con cobertura
pytest --cov=api --cov=core --cov-report=html
```

## 📦 Estructura de Dependencias

### Core Dependencies

- `fastapi` - Framework web
- `uvicorn` - Servidor ASGI
- `pydantic` - Validación de datos
- `python-multipart` - Soporte para formularios

### ML/AI Dependencies

- `torch` - PyTorch para modelos
- `transformers` - Modelos de Hugging Face
- `diffusers` - Modelos de difusión

### Optional Dependencies

- `psycopg2` - PostgreSQL adapter
- `redis` - Cliente Redis
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

### Error: Puerto ya en uso

```bash
# Encontrar proceso usando el puerto
lsof -i :8020  # Linux/Mac
netstat -ano | findstr :8020  # Windows

# Cambiar puerto en .env o directamente
uvicorn main:app --port 8021
```

### Error: Importación circular

```bash
# Verificar estructura de imports
python -m py_compile main.py

# Revisar imports en archivos principales
```

### Error: Variables de entorno no cargadas

```bash
# Verificar que .env existe
ls -la .env  # Linux/Mac
dir .env     # Windows

# Cargar manualmente
export HOST=0.0.0.0  # Linux/Mac
set HOST=0.0.0.0     # Windows CMD
$env:HOST="0.0.0.0"  # Windows PowerShell
```

## 📚 Próximos Pasos

1. ✅ Verificar instalación con health check
2. 📖 Revisar documentación de API en `/docs`
3. 🧪 Ejecutar tests para verificar funcionalidad
4. 🔧 Configurar base de datos si es necesario
5. 🚀 Comenzar a usar la API

## 🔗 Recursos Adicionales

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Uvicorn Documentation](https://www.uvicorn.org/)
- [Pydantic Documentation](https://docs.pydantic.dev/)

---

**Última actualización**: 2025  
**Versión**: 3.4.0






