# 🚀 OS Content System

**Sistema avanzado de generación y gestión de contenido con IA, optimizado para alto rendimiento y escalabilidad.**

## 📋 Tabla de Contenidos

- [Características](#-características)
- [Arquitectura](#-arquitectura)
- [Instalación](#-instalación)
- [Configuración](#-configuración)
- [Uso](#-uso)
- [API](#-api)
- [Testing](#-testing)
- [Despliegue](#-despliegue)
- [Contribución](#-contribución)
- [Licencia](#-licencia)

## ✨ Características

### 🎯 **Funcionalidades Principales**
- **Generación de Contenido con IA**: Integración con modelos avanzados de lenguaje
- **Procesamiento de Video**: Pipeline optimizado para generación y edición de video
- **Sistema de Matemáticas**: Plataforma unificada para cálculos y análisis matemático
- **Procesamiento de Audio**: Generación y manipulación de audio con IA
- **API RESTful**: Interfaz completa con documentación automática

### 🚀 **Características Técnicas**
- **Arquitectura Limpia**: Implementación de Clean Architecture y SOLID principles
- **Procesamiento Asíncrono**: Manejo eficiente de tareas concurrentes
- **Sistema de Caché Multi-nivel**: L1 (memoria), L2 (Redis), L3 (disco)
- **Monitoreo Avanzado**: Métricas en tiempo real con Prometheus y Grafana
- **Logging Estructurado**: Sistema de logs con contexto y análisis de rendimiento
- **Testing Automatizado**: Suite completo de pruebas unitarias e integración

### 🔧 **Optimizaciones de Rendimiento**
- **GPU Acceleration**: Soporte para CUDA y optimizaciones de hardware
- **Compresión Inteligente**: Múltiples algoritmos de compresión (ZSTD, LZ4, Brotli)
- **Load Balancing**: Distribución inteligente de carga
- **CDN Integration**: Gestión de contenido distribuido
- **Memory Management**: Gestión automática de memoria y limpieza

## 🏗️ Arquitectura

### **Diagrama de Arquitectura**
```
┌─────────────────────────────────────────────────────────────┐
│                    Frontend Layer                           │
├─────────────────────────────────────────────────────────────┤
│                    API Gateway                              │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │   Auth      │  │   Content   │  │   Media     │        │
│  │  Service    │  │  Generator  │  │  Processor  │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │   Cache     │  │   Queue     │  │   Storage   │        │
│  │  Manager    │  │  Manager    │  │   Service   │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
├─────────────────────────────────────────────────────────────┤
│                    Database Layer                           │
└─────────────────────────────────────────────────────────────┘
```

### **Componentes Principales**

#### **1. Core Services**
- **ContentGenerator**: Generación inteligente de contenido
- **MediaProcessor**: Procesamiento de audio y video
- **MathEngine**: Motor matemático unificado
- **NLPService**: Procesamiento de lenguaje natural

#### **2. Infrastructure**
- **CacheManager**: Gestión de caché multi-nivel
- **AsyncProcessor**: Procesamiento asíncrono de tareas
- **LoadBalancer**: Balanceo de carga inteligente
- **PerformanceMonitor**: Monitoreo de rendimiento

#### **3. Data Layer**
- **PostgreSQL**: Base de datos principal
- **Redis**: Caché y sesiones
- **Elasticsearch**: Búsqueda y análisis
- **MongoDB**: Almacenamiento de documentos

## 🛠️ Instalación

### **Requisitos del Sistema**
- **Python**: 3.8 o superior
- **RAM**: Mínimo 8GB, recomendado 16GB+
- **GPU**: Opcional, NVIDIA CUDA compatible
- **Sistema**: Linux, macOS, o Windows 10+

### **Instalación Rápida**

#### **1. Clonar el Repositorio**
```bash
git clone <repository-url>
cd os-content-system
```

#### **2. Ejecutar Setup Automático**
```bash
# Setup completo con tests
python setup.py

# Setup sin tests (más rápido)
python setup.py --skip-tests
```

#### **3. Configuración Manual (Alternativa)**
```bash
# Crear entorno virtual
python -m venv venv

# Activar entorno virtual
# Windows
venv\Scripts\activate
# Unix/Linux/macOS
source venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt

# Configurar variables de entorno
cp env.example .env
# Editar .env con tus configuraciones
```

### **Verificación de Instalación**
```bash
# Verificar instalación
python -c "import fastapi, torch, transformers; print('✅ Instalación exitosa')"

# Ejecutar tests básicos
python test_suite.py
```

## ⚙️ Configuración

### **Variables de Entorno Principales**

```bash
# Configuración de la Aplicación
ENVIRONMENT=development
APP_NAME="OS Content System"
APP_VERSION=1.0.0

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_DEBUG=true

# Database
DB_HOST=localhost
DB_PORT=5432
DB_USERNAME=postgres
DB_PASSWORD=your_password
DB_NAME=os_content

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379

# Security
SECRET_KEY=your-super-secret-key-here
```

### **Archivos de Configuración**

#### **config.py**
```python
from config import get_config

config = get_config()
print(f"Database: {config.database.connection_string}")
print(f"Redis: {config.redis.connection_string}")
```

#### **logger.py**
```python
from logger import get_logger, log_with_context

logger = get_logger("my_service")
log_with_context("INFO", "Operation completed", user_id=123, duration=0.5)
```

## 🚀 Uso

### **Iniciar el Sistema**

#### **Windows**
```bash
# Opción 1: Script automático
start.bat

# Opción 2: Manual
venv\Scripts\activate
python main.py
```

#### **Unix/Linux/macOS**
```bash
# Opción 1: Script automático
./start.sh

# Opción 2: Manual
source venv/bin/activate
python main.py
```

### **Acceso al Sistema**
- **API**: http://localhost:8000
- **Documentación**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health
- **Métricas**: http://localhost:9090

### **Ejemplos de Uso**

#### **Generar Contenido**
```python
import requests

# Generar texto
response = requests.post("http://localhost:8000/api/generate/text", json={
    "prompt": "Escribe un artículo sobre IA",
    "max_length": 500
})

# Generar video
response = requests.post("http://localhost:8000/api/generate/video", json={
    "script": "Script para el video",
    "duration": 60
})
```

#### **Procesar Media**
```python
# Procesar audio
response = requests.post("http://localhost:8000/api/process/audio", json={
    "operation": "enhance",
    "file_path": "/path/to/audio.mp3"
})

# Procesar video
response = requests.post("http://localhost:8000/api/process/video", json={
    "operation": "compress",
    "file_path": "/path/to/video.mp4"
})
```

## 🔌 API

### **Endpoints Principales**

#### **Generación de Contenido**
- `POST /api/generate/text` - Generar texto
- `POST /api/generate/video` - Generar video
- `POST /api/generate/audio` - Generar audio
- `POST /api/generate/image` - Generar imagen

#### **Procesamiento de Media**
- `POST /api/process/video` - Procesar video
- `POST /api/process/audio` - Procesar audio
- `POST /api/process/image` - Procesar imagen

#### **Sistema de Matemáticas**
- `POST /api/math/calculate` - Cálculos matemáticos
- `POST /api/math/optimize` - Optimización matemática
- `POST /api/math/analyze` - Análisis matemático

#### **Gestión del Sistema**
- `GET /health` - Estado del sistema
- `GET /metrics` - Métricas de rendimiento
- `GET /status` - Estado detallado

### **Ejemplos de Respuestas**

#### **Generación de Texto**
```json
{
  "success": true,
  "data": {
    "text": "Artículo generado sobre IA...",
    "tokens_used": 150,
    "processing_time": 2.5
  },
  "metadata": {
    "model": "gpt-4",
    "timestamp": "2024-01-15T10:30:00Z"
  }
}
```

#### **Estado del Sistema**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "uptime": "2h 15m 30s",
  "services": {
    "database": "connected",
    "redis": "connected",
    "gpu": "available"
  },
  "performance": {
    "cpu_usage": 45.2,
    "memory_usage": 67.8,
    "active_requests": 12
  }
}
```

## 🧪 Testing

### **Ejecutar Tests**

#### **Suite Completa**
```bash
python test_suite.py
```

#### **Tests Específicos**
```bash
# Solo tests unitarios
pytest test_suite.py::TestConfig -v

# Solo tests de rendimiento
pytest test_suite.py::TestPerformance -v

# Tests de integración
pytest test_suite.py::TestIntegration -v
```

#### **Tests con Cobertura**
```bash
pytest test_suite.py --cov=. --cov-report=html
```

### **Tipos de Tests**

- **Unit Tests**: Pruebas de componentes individuales
- **Integration Tests**: Pruebas de integración entre servicios
- **Performance Tests**: Pruebas de rendimiento y benchmarks
- **Error Handling Tests**: Pruebas de manejo de errores

## 🚀 Despliegue

### **Despliegue Local**

#### **Docker Compose**
```bash
# Construir y ejecutar
docker-compose up -d

# Ver logs
docker-compose logs -f

# Detener servicios
docker-compose down
```

#### **Kubernetes**
```bash
# Aplicar configuración
kubectl apply -f k8s/

# Verificar estado
kubectl get pods
kubectl get services
```

### **Despliegue en Producción**

#### **Variables de Entorno de Producción**
```bash
ENVIRONMENT=production
API_DEBUG=false
API_WORKERS=4
LOG_LEVEL=WARNING
PROMETHEUS_ENABLED=true
```

#### **Monitoreo de Producción**
- **Prometheus**: Métricas del sistema
- **Grafana**: Dashboards y visualización
- **Sentry**: Monitoreo de errores
- **ELK Stack**: Logs y análisis

## 📊 Monitoreo y Logs

### **Sistema de Logs**

#### **Tipos de Logs**
- **Application Logs**: `logs/os_content.log`
- **Error Logs**: `logs/os_content_error.log`
- **Structured Logs**: `logs/os_content_structured.log`

#### **Niveles de Log**
- **DEBUG**: Información detallada para desarrollo
- **INFO**: Información general del sistema
- **WARNING**: Advertencias que no impiden funcionamiento
- **ERROR**: Errores que afectan funcionalidad
- **CRITICAL**: Errores críticos del sistema

### **Métricas de Rendimiento**

#### **Métricas del Sistema**
- Uso de CPU y memoria
- I/O de disco y red
- Tiempo de respuesta de API
- Tasa de errores

#### **Métricas de Aplicación**
- Tiempo de procesamiento
- Uso de caché
- Cola de tareas
- Estado de servicios

## 🔧 Desarrollo

### **Estructura del Proyecto**
```
os-content-system/
├── main.py                 # Punto de entrada principal
├── config.py              # Sistema de configuración
├── logger.py              # Sistema de logging
├── requirements.txt       # Dependencias de Python
├── setup.py              # Script de instalación
├── test_suite.py         # Suite de pruebas
├── env.example           # Variables de entorno de ejemplo
├── README.md             # Este archivo
├── logs/                 # Directorio de logs
├── data/                 # Directorio de datos
├── models/               # Modelos de IA
├── cache/                # Directorio de caché
└── docs/                 # Documentación adicional
```

### **Flujo de Desarrollo**

#### **1. Configurar Entorno**
```bash
# Clonar y configurar
git clone <repo>
cd os-content-system
python setup.py --skip-tests
```

#### **2. Activar Entorno Virtual**
```bash
# Windows
venv\Scripts\activate

# Unix/Linux/macOS
source venv/bin/activate
```

#### **3. Instalar Dependencias de Desarrollo**
```bash
pip install -r requirements.txt
pip install pytest pytest-asyncio pytest-cov
```

#### **4. Ejecutar Tests**
```bash
python test_suite.py
```

#### **5. Desarrollo Local**
```bash
python main.py
```

### **Convenciones de Código**

#### **Estilo de Código**
- **PEP 8**: Estilo de código Python
- **Type Hints**: Anotaciones de tipo obligatorias
- **Docstrings**: Documentación de funciones y clases
- **Error Handling**: Manejo robusto de errores

#### **Estructura de Commits**
```
feat: nueva funcionalidad de generación de video
fix: corrección en el manejo de errores de caché
docs: actualización de la documentación de API
test: añadidos tests para el sistema de logging
refactor: refactorización del motor de caché
```

## 🤝 Contribución

### **Cómo Contribuir**

#### **1. Fork del Repositorio**
- Haz fork del proyecto en GitHub
- Clona tu fork localmente

#### **2. Crear Rama de Feature**
```bash
git checkout -b feature/nueva-funcionalidad
```

#### **3. Desarrollo**
- Implementa tu funcionalidad
- Añade tests apropiados
- Verifica que todos los tests pasen

#### **4. Commit y Push**
```bash
git add .
git commit -m "feat: descripción de la funcionalidad"
git push origin feature/nueva-funcionalidad
```

#### **5. Pull Request**
- Crea un Pull Request en GitHub
- Describe los cambios realizados
- Espera la revisión del código

### **Guidelines de Contribución**

#### **Código**
- Sigue las convenciones de estilo establecidas
- Incluye tests para nueva funcionalidad
- Mantén la cobertura de tests alta
- Documenta funciones y clases nuevas

#### **Documentación**
- Actualiza README.md si es necesario
- Documenta nuevas APIs
- Incluye ejemplos de uso
- Mantén la documentación actualizada

#### **Testing**
- Todos los tests deben pasar
- Nueva funcionalidad debe tener tests
- Mantén la cobertura de tests >90%
- Incluye tests de integración cuando sea apropiado

## 📚 Recursos Adicionales

### **Documentación**
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Pydantic Documentation](https://pydantic-docs.helpmanual.io/)
- [Python AsyncIO](https://docs.python.org/3/library/asyncio.html)

### **Herramientas Relacionadas**
- **MLflow**: Gestión de experimentos de ML
- **Ray**: Computación distribuida
- **Dask**: Procesamiento de datos paralelo
- **Optuna**: Optimización de hiperparámetros

### **Comunidad**
- **Discord**: [Link al servidor]
- **GitHub Issues**: [Link a issues]
- **Documentation**: [Link a docs]

## 📄 Licencia

Este proyecto está licenciado bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para detalles.

## 🙏 Agradecimientos

- **FastAPI**: Framework web moderno y rápido
- **Pydantic**: Validación de datos con Python
- **Transformers**: Modelos de IA de Hugging Face
- **PyTorch**: Framework de deep learning
- **Comunidad Open Source**: Por todas las contribuciones

---

**¿Necesitas ayuda?** 
- 📧 Email: support@oscontent.com
- 💬 Discord: [Link al servidor]
- 📖 Docs: [Link a documentación]
- 🐛 Issues: [Link a GitHub Issues]

---

**⭐ Si este proyecto te es útil, ¡déjanos una estrella en GitHub!**
