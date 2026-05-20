# AI History Comparison - Sistema Ultra Refactorizado Real

Sistema ultra refactorizado para comparación y análisis de historial de IA con tecnologías reales y funcionales.

## 🚀 Características

- **API RESTful** con FastAPI
- **Base de datos SQLite** con soporte asíncrono
- **Análisis de contenido** con scikit-learn y NLTK
- **Comparación semántica** usando TF-IDF y similitud coseno
- **Evaluación de calidad** automatizada
- **Sistema de trabajos** en segundo plano
- **Métricas del sistema** en tiempo real
- **Documentación automática** con Swagger/OpenAPI

## 📋 Requisitos

- Python 3.8+
- SQLite 3
- NLTK data (se descarga automáticamente)

## 🛠️ Instalación

1. **Clonar el repositorio:**
```bash
git clone <repository-url>
cd real_ultra_refactored
```

2. **Crear entorno virtual:**
```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

3. **Instalar dependencias:**
```bash
pip install -r requirements.txt
```

4. **Descargar datos de NLTK:**
```python
python -c "import nltk; nltk.download('punkt'); nltk.download('vader_lexicon')"
```

## 🚀 Uso

### Ejecutar el servidor:

```bash
python -m api.main
```

O usando uvicorn directamente:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

### Acceder a la documentación:

- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

## 📚 API Endpoints

### Historial
- `POST /history` - Crear entrada de historial
- `GET /history` - Obtener entradas con filtros
- `GET /history/{id}` - Obtener entrada específica
- `DELETE /history/{id}` - Eliminar entrada

### Comparaciones
- `POST /comparisons` - Crear comparación entre entradas
- `GET /comparisons` - Obtener comparaciones
- `GET /comparisons/{id}` - Obtener comparación específica

### Calidad
- `POST /quality` - Crear reporte de calidad
- `GET /quality` - Obtener reportes de calidad
- `GET /quality/{id}` - Obtener reporte específico

### Trabajos
- `POST /jobs` - Crear trabajo de análisis
- `GET /jobs` - Obtener trabajos
- `GET /jobs/{id}` - Obtener trabajo específico

### Métricas
- `GET /metrics` - Obtener métricas del sistema
- `GET /trends` - Obtener análisis de tendencias
- `GET /health` - Verificación de salud

## 🔧 Configuración

Crear archivo `.env`:

```env
DEBUG=false
HOST=0.0.0.0
PORT=8000
DATABASE_PATH=ai_history_comparison.db
LOG_LEVEL=INFO
MAX_CONTENT_LENGTH=10000
ANALYSIS_TIMEOUT=30
```

## 📊 Ejemplos de Uso

### Crear entrada de historial:

```python
import requests

entry_data = {
    "model_type": "gpt-4",
    "model_version": "gpt-4-1106-preview",
    "prompt": "Explain quantum computing",
    "response": "Quantum computing is a type of computation...",
    "response_time_ms": 1500,
    "token_count": 150,
    "cost_usd": 0.002
}

response = requests.post("http://localhost:8000/history", json=entry_data)
print(response.json())
```

### Crear comparación:

```python
comparison_data = {
    "entry_1_id": "entry-id-1",
    "entry_2_id": "entry-id-2"
}

response = requests.post("http://localhost:8000/comparisons", json=comparison_data)
print(response.json())
```

### Obtener métricas:

```python
response = requests.get("http://localhost:8000/metrics")
metrics = response.json()
print(f"Total entries: {metrics['total_entries']}")
print(f"Average quality: {metrics['average_quality_score']:.2f}")
```

## 🏗️ Arquitectura

```
real_ultra_refactored/
├── api/
│   └── main.py              # API principal con FastAPI
├── core/
│   └── models.py            # Modelos de datos con Pydantic
├── services/
│   └── analysis_service.py  # Servicios de análisis
├── database/
│   └── database.py          # Gestor de base de datos
├── config/
│   └── settings.py          # Configuración
├── requirements.txt         # Dependencias
└── README.md               # Documentación
```

## 🔍 Análisis de Contenido

El sistema incluye análisis automático de:

- **Métricas básicas:** conteo de palabras, oraciones, caracteres
- **Legibilidad:** puntuación Flesch-Kincaid
- **Sentimiento:** análisis de polaridad
- **Complejidad:** longitud promedio de palabras y oraciones
- **Vocabulario:** riqueza y diversidad

## 📈 Comparación de Entradas

- **Similitud semántica:** TF-IDF + similitud coseno
- **Similitud léxica:** Jaccard index
- **Similitud estructural:** comparación de métricas
- **Detección de diferencias:** análisis automático
- **Identificación de mejoras:** sugerencias automáticas

## 🎯 Evaluación de Calidad

- **Coherencia:** estructura y legibilidad
- **Relevancia:** relación con el prompt
- **Creatividad:** riqueza de vocabulario
- **Precisión:** uso apropiado del lenguaje
- **Claridad:** facilidad de comprensión

## 🧪 Testing

```bash
# Ejecutar tests
pytest

# Con cobertura
pytest --cov=.

# Tests específicos
pytest tests/test_analysis_service.py
```

## 📝 Logging

El sistema incluye logging estructurado:

```python
import logging
logger = logging.getLogger(__name__)
logger.info("Operation completed successfully")
```

## 🚀 Despliegue

### Docker:

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Docker Compose:

```yaml
version: '3.8'
services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_PATH=/app/data/ai_history.db
    volumes:
      - ./data:/app/data
```

## 🤝 Contribución

1. Fork el proyecto
2. Crear rama para feature (`git checkout -b feature/AmazingFeature`)
3. Commit cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abrir Pull Request

## 📄 Licencia

Este proyecto está bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para detalles.

## 🆘 Soporte

Para soporte, crear un issue en el repositorio o contactar al equipo de desarrollo.

---

**Sistema Ultra Refactorizado Real** - Tecnologías funcionales y prácticas para análisis de IA.




