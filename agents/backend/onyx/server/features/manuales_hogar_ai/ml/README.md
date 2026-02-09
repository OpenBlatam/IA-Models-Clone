# ML Module - Deep Learning Integration

## 🧠 Módulo de Machine Learning

Este módulo proporciona capacidades de deep learning para el sistema de manuales.

## 📦 Componentes

### 1. Modelos de Generación (`ml/models/`)

**ManualGeneratorModel**:
- Modelo basado en transformers para generación de texto
- Soporte para fine-tuning con LoRA
- Generación de manuales personalizados
- Múltiples modelos pre-entrenados soportados

### 2. Embeddings (`ml/embeddings/`)

**EmbeddingService**:
- Generación de embeddings semánticos
- Búsqueda por similitud
- Comparación de textos
- Modelos multilingües

### 3. Generación de Imágenes (`ml/image_generation/`)

**ImageGenerator**:
- Generación de imágenes con Stable Diffusion
- Ilustraciones para manuales
- Múltiples estilos (LEGO, técnico, realista)
- Optimización de memoria

### 4. Entrenamiento (`ml/training/`)

**ManualTrainer**:
- Fine-tuning de modelos
- Soporte para LoRA
- Integración con Weights & Biases
- Dataset personalizado
- Evaluación automática

### 5. Inferencia (`ml/inference/`)

**LocalLLMService**:
- Servicio para modelos locales
- Alternativa a OpenRouter
- Pipeline optimizado

## 🚀 Uso

### Generar Embeddings

```python
from ml.embeddings.embedding_service import EmbeddingService

service = EmbeddingService()
embeddings = service.encode(["texto 1", "texto 2"])
similarity = service.similarity("texto 1", "texto 2")
```

### Generar Imágenes

```python
from ml.image_generation.image_generator import ImageGenerator

generator = ImageGenerator()
image = generator.generate_manual_illustration(
    step_description="Cerrar llave de paso",
    category="plomeria",
    style="lego_instruction"
)
```

### Entrenar Modelo

```bash
python scripts/train_model.py
```

### Demo Gradio

```bash
python scripts/gradio_demo.py
```

## 📊 Endpoints de ML

### Embeddings
- `POST /api/v1/ml/embeddings/generate` - Generar embeddings
- `POST /api/v1/ml/embeddings/similarity` - Calcular similitud
- `POST /api/v1/ml/embeddings/find-similar` - Encontrar similares
- `GET /api/v1/ml/embeddings/info` - Info del servicio

### Imágenes
- `POST /api/v1/ml/images/generate` - Generar imagen
- `POST /api/v1/ml/images/generate-manual-illustration` - Ilustración de manual
- `GET /api/v1/ml/images/info` - Info del generador

### Búsqueda Semántica
- `POST /api/v1/search/semantic` - Búsqueda semántica

## ⚙️ Configuración

Configurar en `.env` o variables de entorno:

```bash
# Modelos
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
GENERATION_MODEL=microsoft/DialoGPT-medium
IMAGE_MODEL=runwayml/stable-diffusion-v1-5

# Dispositivos
DEVICE=cuda  # o cpu
USE_CUDA=true

# LoRA
USE_LORA=true
LORA_R=16
LORA_ALPHA=32

# Weights & Biases
USE_WANDB=false
WANDB_API_KEY=tu-key
```

## 🎯 Características

- ✅ Modelos locales opcionales
- ✅ Fine-tuning con LoRA
- ✅ Búsqueda semántica
- ✅ Generación de ilustraciones
- ✅ Optimización de GPU
- ✅ Mixed precision training
- ✅ Integración con WandB
- ✅ Demo interactivo con Gradio

## 📝 Requisitos

- GPU recomendada (CUDA)
- 8GB+ RAM
- Espacio en disco para modelos (varios GB)

## 🔧 Optimizaciones

- Lazy loading de modelos
- Mixed precision (FP16)
- Attention slicing
- VAE slicing
- Gradient accumulation




