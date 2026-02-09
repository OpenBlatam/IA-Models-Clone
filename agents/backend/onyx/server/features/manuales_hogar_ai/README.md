# Manuales Hogar AI 🏠🔧

Sistema de IA para generar manuales paso a paso tipo LEGO para oficios populares. Permite procesar imágenes (fotos de problemas) o descripciones de texto y genera guías visuales y detalladas de cómo resolver problemas del hogar, trabajo y oficios.

**✨ Refactorizado para microservicios, serverless y cloud-native** - Ver [REFACTORING.md](REFACTORING.md) y [ARCHITECTURE.md](ARCHITECTURE.md)

**🛡️ Mejorado para máxima estabilidad** - Ver [STABILITY.md](STABILITY.md)

**☁️ Listo para cualquier EC2** - Ver [EC2_DEPLOYMENT.md](EC2_DEPLOYMENT.md)

## ⚡ Inicio Rápido

```bash
# Un solo comando para iniciar todo
./start.sh        # Linux/Mac
.\start.ps1       # Windows
python run.py     # Cualquier plataforma
```

Ver [QUICKSTART.md](QUICKSTART.md) para más detalles.

## 🛠️ Scripts de Utilidad

El proyecto incluye **25+ scripts** útiles para gestión y mantenimiento:

### Gestión Básica
```bash
./status.sh              # Estado de servicios
./scripts/setup.sh       # Configuración inicial
./scripts/clean.sh       # Limpieza
```

### Monitoreo y Diagnóstico
```bash
./scripts/monitor.sh      # Monitoreo en tiempo real
./scripts/health-monitor.sh # Monitoreo continuo con alertas
./scripts/diagnostics.sh  # Diagnósticos completos
./scripts/watch.sh        # Auto-reiniciar en cambios
```

### Testing y Performance
```bash
./scripts/test-api.sh     # Pruebas de API
./scripts/quick-test.sh   # Validación rápida
./scripts/performance-test.sh # Pruebas de rendimiento
```

### Backup y Seguridad
```bash
./scripts/backup.sh       # Backup de base de datos
./scripts/restore.sh      # Restaurar desde backup
./scripts/security-check.sh # Validación de seguridad
```

### Optimización y Mantenimiento
```bash
./scripts/optimize.sh     # Optimizar Docker
./scripts/update.sh       # Actualizar aplicación
./scripts/export-logs.sh  # Exportar logs
```

Ver [scripts/README.md](scripts/README.md) para lista completa y [FEATURES_COMPLETE.md](FEATURES_COMPLETE.md) para todas las funcionalidades.

## 🎯 Características

- ✅ **Soporte para múltiples modelos de IA** a través de OpenRouter
- ✅ **Procesamiento de imágenes** con modelos de visión
- ✅ **Soporte para múltiples imágenes** (hasta 5 imágenes por solicitud)
- ✅ **Procesamiento de texto** para descripciones de problemas
- ✅ **Detección automática de categorías** con algoritmo inteligente
- ✅ **Sistema de cache** (memoria + persistente en BD)
- ✅ **Validación y optimización de imágenes** automática
- ✅ **Persistencia en base de datos** con Alembic
- ✅ **Historial de manuales** con búsqueda avanzada
- ✅ **Estadísticas de uso** automáticas
- ✅ **Manuales tipo LEGO** paso a paso con formato visual
- ✅ **Múltiples categorías de oficios**:
  - Plomería
  - Techos y reparaciones
  - Carpintería
  - Electricidad
  - Albañilería
  - Pintura
  - Herrería
  - Jardinería
  - General

## 📋 Requisitos

- Python 3.8+ (o Docker)
- PostgreSQL 12+ (para base de datos)
- API Key de OpenRouter (configurar en variable de entorno `OPENROUTER_API_KEY`)

## 🚀 Instalación y Uso

### Inicio Rápido (Un Solo Comando)

**Linux/Mac:**
```bash
./start.sh
```

**Windows (PowerShell):**
```powershell
.\start.ps1
```

**Python (cualquier plataforma):**
```bash
python run.py
```

**Producción:**
```bash
./start.sh prod
# o
python run.py prod
```

El script automáticamente:
- ✅ Verifica que Docker esté corriendo
- ✅ Crea archivo `.env` si no existe
- ✅ Inicia todos los servicios (app, PostgreSQL, Redis)
- ✅ Espera a que el servicio esté listo
- ✅ Muestra la URL del API

### Opción 2: Docker Compose Manual

**Desarrollo:**
```bash
docker-compose up -d
```

**Producción:**
```bash
docker-compose -f docker-compose.prod.yml up -d
```

Ver [DOCKER.md](DOCKER.md) para más detalles.

### Opción 3: Instalación Local

```bash
pip install -r requirements.txt
```

## ⚙️ Configuración

### Variables de Entorno

Configurar las variables de entorno o crear un archivo `.env` (ver `.env.example`):

```bash
# OpenRouter API Key (requerido)
OPENROUTER_API_KEY=tu-api-key-aqui

# Base de Datos (opcional, para persistencia)
DATABASE_URL=postgresql+asyncpg://usuario:password@localhost:5432/manuales_hogar
# O individualmente:
DB_HOST=localhost
DB_PORT=5432
DB_USER=postgres
DB_PASSWORD=tu_password
DB_NAME=manuales_hogar
```

### Inicializar Base de Datos

Si quieres usar persistencia de datos, inicializa la base de datos:

```bash
# Crear migración inicial
alembic revision --autogenerate -m "Initial migration"

# Aplicar migraciones
alembic upgrade head
```

Ver [MIGRATIONS.md](MIGRATIONS.md) para más detalles sobre migraciones.

## 📖 Uso

### API Endpoints

#### 1. Health Check
```bash
GET /api/v1/health
```

#### 2. Listar Modelos Disponibles
```bash
GET /api/v1/models
```

#### 3. Generar Manual desde Texto
```bash
POST /api/v1/generate-from-text
Content-Type: application/json

{
  "problem_description": "Tengo una fuga de agua en el grifo de la cocina",
  "category": "plomeria",
  "include_safety": true,
  "include_tools": true,
  "include_materials": true
}
```

#### 4. Generar Manual desde Imagen
```bash
POST /api/v1/generate-from-image
Content-Type: multipart/form-data

file: [imagen.jpg]
problem_description: "Descripción adicional (opcional)"
category: "plomeria" (opcional, se detecta automáticamente)
```

#### 5. Generar Manual Combinado (Texto + Imagen)
```bash
POST /api/v1/generate-combined
Content-Type: multipart/form-data

problem_description: "Tengo un problema con..."
file: [imagen.jpg] (opcional)
category: "general"
```

#### 6. Generar Manual desde Múltiples Imágenes
```bash
POST /api/v1/generate-from-multiple-images
Content-Type: multipart/form-data

files: [imagen1.jpg, imagen2.jpg, imagen3.jpg] (máximo 5)
problem_description: "Descripción adicional" (opcional)
category: "plomeria" (opcional, se detecta automáticamente)
```

#### 7. Obtener Categorías Soportadas
```bash
GET /api/v1/categories
```

#### 8. Estadísticas del Cache
```bash
GET /api/v1/cache/stats
```

#### 9. Limpiar Cache (Memoria)
```bash
DELETE /api/v1/cache/clear
```

#### 10. Listar Manuales (Historial)
```bash
GET /api/v1/manuals?category=plomeria&limit=20&offset=0
```

#### 11. Obtener Manual por ID
```bash
GET /api/v1/manuals/{manual_id}
```

#### 12. Manuales Recientes
```bash
GET /api/v1/manuals/recent?limit=10
```

#### 13. Estadísticas de Uso
```bash
GET /api/v1/statistics?days=30
```

#### 14. Estadísticas de Cache Persistente
```bash
GET /api/v1/cache/stats-db
```

#### 15. Limpiar Cache Persistente
```bash
DELETE /api/v1/cache/clear-db
```

#### 16. Limpiar Cache Expirado
```bash
POST /api/v1/cache/cleanup-expired
```

### Ejemplo de Uso en Python

```python
from manuales_hogar_ai import ManualGenerator
from manuales_hogar_ai.infrastructure import OpenRouterClient

# Crear cliente y generador
client = OpenRouterClient()
generator = ManualGenerator(openrouter_client=client)

# Generar manual desde texto
result = await generator.generate_manual_from_text(
    problem_description="Fuga de agua en el grifo",
    category="plomeria"
)

print(result["manual"])

# Generar manual desde imagen
result = await generator.generate_manual_from_image(
    image_path="problema.jpg",
    problem_description="Fuga visible en la conexión",
    category="plomeria"
)

print(result["manual"])
```

## 🏗️ Estructura del Proyecto

```
manuales_hogar_ai/
├── __init__.py
├── alembic/                    # Migraciones de base de datos
│   ├── versions/               # Archivos de migración
│   ├── env.py                  # Configuración de Alembic
│   └── script.py.mako         # Template de migraciones
├── alembic.ini                 # Configuración de Alembic
├── api/
│   ├── __init__.py
│   └── routes/
│       ├── __init__.py
│       └── manuales.py
├── config/
│   ├── __init__.py
│   └── settings.py
├── core/
│   ├── __init__.py
│   └── manual_generator.py
├── database/                   # Modelos y sesiones de BD
│   ├── __init__.py
│   ├── models.py               # Modelos SQLAlchemy
│   └── session.py              # Gestión de sesiones
├── infrastructure/
│   ├── __init__.py
│   └── openrouter_client.py
├── scripts/                    # Scripts de utilidad
│   ├── __init__.py
│   └── init_db.py              # Inicializar BD
├── services/
│   └── __init__.py
├── utils/
│   ├── __init__.py
│   ├── cache_manager.py
│   ├── category_detector.py
│   └── image_validator.py
├── requirements.txt
├── README.md
└── MIGRATIONS.md               # Guía de migraciones
```

## 🎨 Formato de Manual

Los manuales generados siguen un formato tipo LEGO con:

1. **Título del Manual**
2. **Diagnóstico** del problema
3. **Advertencias de Seguridad** ⚠️
4. **Herramientas Necesarias** 🔧
5. **Materiales Necesarios** 📦
6. **Pasos de la Reparación** (formato LEGO):
   - Número de paso
   - Descripción clara
   - Ilustración verbal
   - Tiempo estimado
   - Dificultad (Fácil/Media/Difícil)
   - Precauciones
7. **Verificación**
8. **Mantenimiento Preventivo**
9. **Cuándo Llamar a un Profesional**

## 🤖 Modelos Soportados

El sistema puede usar cualquier modelo disponible en OpenRouter, incluyendo:

- `anthropic/claude-3.5-sonnet` (default)
- `openai/gpt-4o`
- `openai/gpt-4-turbo`
- `google/gemini-pro-1.5`
- `meta-llama/llama-3.1-70b-instruct`
- `anthropic/claude-3-opus`

Para modelos con visión, se recomienda usar:
- `anthropic/claude-3.5-sonnet`
- `openai/gpt-4o`
- `google/gemini-pro-1.5`

## 📝 Categorías Soportadas

- `plomeria` - Plomería
- `techos` - Reparación de Techos
- `carpinteria` - Carpintería
- `electricidad` - Electricidad
- `albanileria` - Albañilería
- `pintura` - Pintura
- `herreria` - Herrería
- `jardineria` - Jardinería
- `general` - Reparación General

## 🔒 Seguridad

- Las imágenes se procesan temporalmente y se eliminan después del procesamiento
- Tamaño máximo de imagen: 10MB
- Se validan tipos de archivo antes del procesamiento
- Se incluyen advertencias de seguridad en los manuales generados
- Validación de entrada en todos los endpoints
- Manejo seguro de errores sin exponer información sensible

## 📊 Base de Datos y Persistencia

El sistema incluye persistencia completa con:

- **Historial de Manuales**: Todos los manuales generados se guardan automáticamente
- **Cache Persistente**: Cache en base de datos con expiración automática
- **Estadísticas**: Métricas automáticas de uso, tokens, modelos, etc.
- **Búsqueda Avanzada**: Búsqueda por categoría, término, fecha, etc.

Ver [MIGRATIONS.md](MIGRATIONS.md) para configuración de base de datos y [FEATURES.md](FEATURES.md) para detalles de funcionalidades.

## 🐛 Troubleshooting

### Error: "OpenRouter API key not configured"
- Asegúrate de configurar la variable de entorno `OPENROUTER_API_KEY`

### Error: "La imagen es demasiado grande"
- El tamaño máximo es 10MB. Reduce el tamaño de la imagen antes de subirla.

### Error: "Categoría no soportada"
- Verifica que la categoría esté en la lista de categorías soportadas usando `GET /api/v1/categories`

## 📄 Licencia

Propietaria - Blatam Academy

## 👥 Autor

Blatam Academy

## 🔄 Versión

1.0.0

