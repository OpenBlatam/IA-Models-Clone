# Export IA

Sistema de exportación de documentos con IA - Arquitectura modular y funcional.

## 🚀 **Características**

- ✅ **Exportación a múltiples formatos** (PDF, DOCX, HTML, Markdown, etc.)
- ✅ **API REST completa** con FastAPI
- ✅ **Gestión asíncrona de tareas**
- ✅ **Validación y mejora de calidad**
- ✅ **Arquitectura modular y escalable**
- ✅ **Configuración flexible**
- ✅ **Documentación completa**

## 📁 **Estructura del Proyecto**

```
export_ia/
├── 📁 app/                          # Aplicación principal
│   ├── 📁 core/                     # Lógica de negocio
│   ├── 📁 api/                      # API REST
│   ├── 📁 exporters/                # Exportadores
│   ├── 📁 services/                 # Servicios
│   └── 📁 utils/                    # Utilidades
├── 📁 config/                       # Configuración
├── 📁 database/                     # Base de datos
├── 📁 tests/                        # Pruebas
├── 📁 docs/                         # Documentación
├── 📁 scripts/                      # Scripts
├── 📁 docker/                       # Docker
├── 📁 examples/                     # Ejemplos
├── 📄 requirements.txt              # Dependencias
└── 📄 README.md                     # Este archivo
```

## 🛠️ **Instalación**

### **1. Clonar el repositorio**
```bash
git clone https://github.com/your-org/export-ia.git
cd export-ia
```

### **2. Crear entorno virtual**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# o
venv\Scripts\activate     # Windows
```

### **3. Instalar dependencias**
```bash
# Dependencias principales
pip install -r requirements.txt

# Dependencias de desarrollo (opcional)
pip install -r requirements-dev.txt
```

### **4. Configurar variables de entorno**
```bash
# Copiar archivo de ejemplo
cp .env.example .env

# Editar variables de entorno
nano .env
```

## 🚀 **Uso Rápido**

### **Ejecutar la API**
```bash
# Desarrollo
python -m app.api.main

# Producción
uvicorn app.api.main:app --host 0.0.0.0 --port 8000
```

### **Con Docker**
```bash
# Construir y ejecutar
docker-compose -f docker/docker-compose.yml up --build

# Acceder a la API
curl http://localhost:8000/health
```

## 📚 **API Endpoints**

### **Endpoints Principales**
```
GET  /api/v1/                    # Información del sistema
GET  /api/v1/health             # Health check
POST /api/v1/export             # Exportar documento
GET  /api/v1/export/{id}/status # Estado de tarea
GET  /api/v1/export/{id}/download # Descargar archivo
POST /api/v1/validate           # Validar contenido
GET  /api/v1/formats            # Formatos soportados
GET  /api/v1/templates/{type}   # Plantillas
```

### **Ejemplo de Uso**
```bash
# Exportar documento
curl -X POST "http://localhost:8000/api/v1/export" \
  -H "Content-Type: application/json" \
  -d '{
    "content": {
      "title": "Mi Documento",
      "sections": [
        {"heading": "Introducción", "content": "Contenido aquí..."}
      ]
    },
    "format": "pdf",
    "document_type": "report",
    "quality_level": "professional"
  }'

# Verificar estado
curl "http://localhost:8000/api/v1/export/{task_id}/status"

# Descargar archivo
curl "http://localhost:8000/api/v1/export/{task_id}/download" -o documento.pdf
```

## 🐍 **SDK Python**

### **Uso Básico**
```python
from app.core.engine import get_export_engine
from app.core.models import ExportConfig, ExportFormat, DocumentType

# Obtener motor
engine = get_export_engine()
await engine.initialize()

# Configurar exportación
config = ExportConfig(
    format=ExportConfig.PDF,
    document_type=DocumentType.REPORT
)

# Exportar documento
content = {
    "title": "Mi Documento",
    "sections": [
        {"heading": "Introducción", "content": "Contenido aquí..."}
    ]
}

task_id = await engine.export_document(content, config)

# Esperar completado
result = await engine.wait_for_completion(task_id)
print(f"Archivo exportado: {result['file_path']}")
```

## 🧪 **Pruebas**

### **Ejecutar Pruebas**
```bash
# Todas las pruebas
pytest

# Con cobertura
pytest --cov=app

# Pruebas específicas
pytest tests/unit/
pytest tests/integration/
```

## 📖 **Documentación**

### **Documentos Disponibles**
- **[API Documentation](docs/API.md)** - Documentación completa de la API
- **[Deployment Guide](docs/DEPLOYMENT.md)** - Guía de despliegue
- **[Development Guide](docs/DEVELOPMENT.md)** - Guía de desarrollo
- **[Examples](examples/)** - Ejemplos de uso

### **Documentación Interactiva**
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 🔧 **Configuración**

### **Variables de Entorno**
```bash
# API
API_TITLE="Export IA API"
API_VERSION="2.0.0"
DEBUG=false

# Base de datos
DATABASE_URL="sqlite:///./export_ia.db"

# Archivos
EXPORTS_DIR="./exports"
MAX_FILE_SIZE=52428800  # 50MB

# Logging
LOG_LEVEL="INFO"
LOG_FILE="./logs/export_ia.log"
```

## 🐳 **Docker**

### **Docker Compose**
```yaml
version: '3.8'
services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=sqlite:///./export_ia.db
    volumes:
      - ./exports:/app/exports
    restart: unless-stopped
```

### **Comandos Docker**
```bash
# Construir imagen
docker build -t export-ia .

# Ejecutar contenedor
docker run -p 8000:8000 export-ia

# Con Docker Compose
docker-compose up --build
```

## 🤝 **Contribuir**

### **1. Fork del repositorio**
```bash
git fork https://github.com/your-org/export-ia.git
```

### **2. Crear rama de feature**
```bash
git checkout -b feature/nueva-funcionalidad
```

### **3. Hacer cambios y commit**
```bash
git add .
git commit -m "feat: agregar nueva funcionalidad"
```

### **4. Push y crear Pull Request**
```bash
git push origin feature/nueva-funcionalidad
```

## 📄 **Licencia**

Este proyecto está bajo la Licencia MIT. Ver [LICENSE](LICENSE) para más detalles.

## 🆘 **Soporte**

- **Documentación**: [docs/](docs/)
- **Issues**: [GitHub Issues](https://github.com/your-org/export-ia/issues)
- **Discusiones**: [GitHub Discussions](https://github.com/your-org/export-ia/discussions)

## 🎯 **Roadmap**

- [ ] **v2.1.0**: Mejoras de rendimiento
- [ ] **v2.2.0**: Nuevos formatos de exportación
- [ ] **v2.3.0**: Integración con IA avanzada
- [ ] **v3.0.0**: Arquitectura de microservicios

---

**¡Export IA - Exportación de documentos simple, rápida y profesional!** 🚀