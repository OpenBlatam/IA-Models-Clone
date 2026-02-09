# Dermatology AI - Sistema de Análisis de Piel y Skincare

## 🚀 Descripción

Sistema avanzado de IA para análisis de calidad de piel mediante fotos o videos con sensores. El sistema determina la calidad de la piel y proporciona recomendaciones personalizadas de skincare basadas en el análisis.

## ✨ Características Principales

### Análisis de Piel
- **Métricas de Calidad**: Análisis completo de textura, hidratación, elasticidad, pigmentación, poros, arrugas, enrojecimiento y manchas
- **Detección de Condiciones**: Identificación de acné, rosácea, eczema, hiperpigmentación, sequedad y sensibilidad
- **Análisis de Imágenes**: Procesamiento y análisis de fotografías de piel
- **Análisis de Videos**: Análisis agregado de secuencias de video para mayor precisión

### Recomendaciones de Skincare
- **Rutinas Personalizadas**: Rutinas de mañana, tarde y tratamientos semanales
- **Productos Recomendados**: Recomendaciones específicas según tipo de piel y condiciones
- **Tips Personalizados**: Consejos basados en el análisis individual
- **Priorización**: Identificación de áreas prioritarias para mejorar

## 📁 Estructura del Proyecto (Reorganizada)

```
dermatology_ai/
├── ml/                     # 🧠 Machine Learning Core
│   ├── models/            # Arquitecturas de modelos
│   ├── training/          # Componentes de entrenamiento
│   ├── data/              # Procesamiento de datos
│   ├── experiments/       # Gestión de experimentos
│   ├── inference/         # Motores de inferencia
│   └── visualization/     # Demos y visualización
│
├── core/                   # 💼 Business Logic
│   ├── application/       # Use cases (hexagonal)
│   ├── domain/            # Domain entities
│   ├── infrastructure/    # Infrastructure adapters
│   └── ...                # Core components
│
├── api/                    # 🌐 API Layer
│   ├── controllers/       # Request handlers
│   ├── routers/           # API routes
│   └── middleware/        # API middleware
│
├── services/               # 🔧 Business Services
│   └── ...                # Service implementations
│
├── utils/                  # 🛠️ Utilities & Optimizations
│   ├── optimization.py    # Performance optimizations
│   ├── profiling.py       # Profiling tools
│   └── ...                # Other utilities
│
├── config/                 # ⚙️ Configuration
│   ├── settings.py        # App settings
│   └── model_config.yaml  # Model config template
│
├── examples/               # 📚 Examples
│   ├── training_example.py
│   ├── inference_example.py
│   └── gradio_demo_example.py
│
├── docs/                   # 📖 Documentation
│   └── README.md          # Documentation index
│
├── tests/                  # 🧪 Tests
├── scripts/                # 🔨 Utility scripts
├── main.py                 # Servidor principal
└── requirements.txt        # Dependencies
```

**Ver [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) para estructura completa**
**Ver [ORGANIZATION_GUIDE.md](ORGANIZATION_GUIDE.md) para guía de organización**

## 🔧 Instalación

### Prerrequisitos

- Python 3.8+
- pip

### Instalación Rápida

```bash
cd dermatology_ai
pip install -r requirements.txt
```

## 🚀 Uso

### Iniciar Servidor

```bash
python main.py
```

El servidor estará disponible en `http://localhost:8006`

### Documentación API

Una vez iniciado el servidor, acceda a:
- **Swagger UI**: `http://localhost:8006/docs`
- **ReDoc**: `http://localhost:8006/redoc`

## 📖 Endpoints Principales

### 1. Analizar Imagen

```bash
POST /dermatology/analyze-image
Content-Type: multipart/form-data

file: [imagen]
enhance: true
```

**Respuesta:**
```json
{
  "success": true,
  "analysis": {
    "quality_scores": {
      "overall_score": 75.5,
      "texture_score": 80.0,
      "hydration_score": 70.0,
      ...
    },
    "conditions": [
      {
        "name": "acne",
        "confidence": 0.65,
        "severity": "moderate",
        ...
      }
    ],
    "skin_type": "combination",
    "recommendations_priority": ["hydration", "texture"]
  }
}
```

### 2. Analizar Video

```bash
POST /dermatology/analyze-video
Content-Type: multipart/form-data

file: [video]
max_frames: 30
```

### 3. Obtener Recomendaciones

```bash
POST /dermatology/get-recommendations
Content-Type: multipart/form-data

file: [imagen]
include_routine: true
```

**Respuesta:**
```json
{
  "success": true,
  "analysis": {...},
  "recommendations": {
    "routine": {
      "morning": [
        {
          "name": "Limpiador Suave",
          "category": "cleanser",
          "description": "...",
          "key_ingredients": ["Glicerina", "..."],
          "usage_frequency": "2 veces al día",
          "priority": 1
        },
        ...
      ],
      "evening": [...],
      "weekly": [...]
    },
    "specific_recommendations": [...],
    "tips": [...]
  }
}
```

## 💻 Uso Programático

### Análisis de Imagen (Legacy)

```python
from dermatology_ai import SkinAnalyzer, ImageProcessor
from PIL import Image
import numpy as np

# Inicializar
analyzer = SkinAnalyzer()
processor = ImageProcessor()

# Cargar imagen
image = Image.open("skin_photo.jpg")
img_array = np.array(image)

# Analizar
result = analyzer.analyze_image(img_array)
print(f"Score general: {result['quality_scores']['overall_score']}")
```

### Uso con ML Module (Recomendado)

```python
# Import organizado desde ml/
from ml import ViTSkinAnalyzer, Trainer, SkinDataset, get_train_transforms
from ml.inference import FastInferenceEngine
from ml.experiments import ExperimentTracker

# Crear modelo
model = ViTSkinAnalyzer(num_conditions=6, num_metrics=8)

# Inferencia optimizada
engine = FastInferenceEngine(model, use_compile=True)
output = engine.predict(input_tensor)
```

### Obtener Recomendaciones

```python
from dermatology_ai import SkincareRecommender

recommender = SkincareRecommender()
recommendations = recommender.generate_recommendations(analysis_result)

print("Rutina de la mañana:")
for product in recommendations["routine"]["morning"]:
    print(f"- {product['name']}")
```

**Ver [examples/](examples/) para ejemplos completos**

## 🔬 Métricas de Calidad

El sistema analiza las siguientes métricas (0-100, donde 100 es mejor):

- **Overall Score**: Puntuación general de calidad
- **Texture Score**: Suavidad y uniformidad de textura
- **Hydration Score**: Nivel de hidratación
- **Elasticity Score**: Elasticidad y firmeza
- **Pigmentation Score**: Uniformidad de pigmentación
- **Pore Size Score**: Tamaño de poros (100 = poros muy pequeños)
- **Wrinkles Score**: Presencia de arrugas (100 = sin arrugas)
- **Redness Score**: Enrojecimiento (100 = sin enrojecimiento)
- **Dark Spots Score**: Manchas oscuras (100 = sin manchas)

## 🎯 Condiciones Detectadas

El sistema puede detectar:

- **Acné**: Granos y protuberancias
- **Rosácea**: Enrojecimiento crónico
- **Eczema**: Dermatitis
- **Hiperpigmentación**: Manchas oscuras
- **Sequedad**: Piel deshidratada
- **Sensibilidad**: Piel sensible o irritada

## 🧴 Tipos de Productos Recomendados

- **Cleanser**: Limpiadores
- **Moisturizer**: Hidratantes
- **Serum**: Sérums de tratamiento
- **Sunscreen**: Protectores solares
- **Toner**: Tónicos
- **Exfoliant**: Exfoliantes
- **Mask**: Máscaras
- **Eye Cream**: Cremas para contorno de ojos

## 🔧 Configuración

### Variables de Entorno

Cree un archivo `.env`:

```env
# Configuración del servidor
HOST=0.0.0.0
PORT=8006

# Configuración de procesamiento
IMAGE_TARGET_SIZE=512,512
VIDEO_MAX_FRAMES=30
VIDEO_TARGET_FPS=1
```

## 🧪 Testing

```bash
# Ejecutar tests
pytest tests/

# Con cobertura
pytest --cov=dermatology_ai tests/
```

## 🆕 Mejoras en v1.1.0

### Nuevas Características
- ✅ **Análisis Avanzado**: Técnicas mejoradas de visión por computadora
- ✅ **Sistema de Logging**: Logging completo con archivos y métricas
- ✅ **Sistema de Cache**: Cache en memoria y disco para mejor rendimiento
- ✅ **Manejo de Errores**: Excepciones personalizadas y mejor manejo
- ✅ **Métricas Detalladas**: Análisis más profundo con métricas adicionales

Ver [IMPROVEMENTS.md](IMPROVEMENTS.md) para detalles completos.

## 📊 Mejoras Futuras

- [ ] Integración con modelos ML avanzados (CNN, Vision Transformers)
- [ ] Análisis de progreso temporal (comparación de antes/después)
- [ ] Base de datos de productos reales
- [ ] Integración con sensores especializados
- [ ] Análisis de diferentes áreas del cuerpo
- [ ] Soporte para múltiples idiomas
- [ ] Dashboard web interactivo
- [ ] Sistema de historial y tracking de progreso

## 🤝 Contribuir

Las contribuciones son bienvenidas. Por favor:

1. Fork el proyecto
2. Cree una rama para su feature (`git checkout -b feature/AmazingFeature`)
3. Commit sus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abra un Pull Request

## 📝 Licencia

Este proyecto es parte de Blatam Academy.

## 📞 Soporte

Para soporte, contacte al equipo de Blatam Academy.

---

**Desarrollado con ❤️ por Blatam Academy**

