# 🧠 Funcionalidades de Machine Learning

## Resumen

Se han agregado capacidades completas de deep learning y machine learning al sistema de manuales, incluyendo:

- ✅ Modelos de generación locales con Transformers
- ✅ Fine-tuning con LoRA
- ✅ Búsqueda semántica con embeddings
- ✅ Generación de imágenes con Stable Diffusion
- ✅ Sistema de entrenamiento completo
- ✅ Demo interactivo con Gradio
- ✅ Integración con Weights & Biases

## 📊 Nuevos Endpoints (8)

### Embeddings (4)
1. `POST /api/v1/ml/embeddings/generate` - Generar embeddings
2. `POST /api/v1/ml/embeddings/similarity` - Calcular similitud
3. `POST /api/v1/ml/embeddings/find-similar` - Encontrar textos similares
4. `GET /api/v1/ml/embeddings/info` - Información del servicio

### Imágenes (3)
1. `POST /api/v1/ml/images/generate` - Generar imagen desde prompt
2. `POST /api/v1/ml/images/generate-manual-illustration` - Ilustración para manual
3. `GET /api/v1/ml/images/info` - Información del generador

### Búsqueda Semántica (1)
1. `POST /api/v1/search/semantic` - Búsqueda semántica de manuales

## 🎯 Componentes Principales

### 1. ManualGeneratorModel
- Modelo basado en Transformers
- Soporte para múltiples modelos pre-entrenados
- Fine-tuning con LoRA (eficiente)
- Generación de manuales personalizados
- Mixed precision (FP16) para GPU

### 2. EmbeddingService
- Generación de embeddings semánticos
- Modelos multilingües
- Búsqueda por similitud coseno
- Comparación de textos
- Integración con búsqueda de manuales

### 3. ImageGenerator
- Stable Diffusion para generación de imágenes
- Soporte para SD XL
- Ilustraciones estilo LEGO
- Múltiples estilos (técnico, realista)
- Optimización de memoria (slicing)

### 4. ManualTrainer
- Sistema completo de entrenamiento
- Dataset personalizado
- LoRA para fine-tuning eficiente
- Integración con WandB
- Evaluación automática
- Early stopping
- Learning rate scheduling

### 5. SemanticSearchService
- Búsqueda semántica de manuales
- Manuales relacionados automáticos
- Scoring por similitud
- Integración con base de datos

## 🚀 Uso Rápido

### Generar Embeddings

```python
from ml.embeddings.embedding_service import EmbeddingService

service = EmbeddingService()
embeddings = service.encode(["texto 1", "texto 2"])
similarity = service.similarity("fuga de agua", "problema en grifo")
```

### Generar Ilustración

```python
from ml.image_generation.image_generator import ImageGenerator

generator = ImageGenerator()
image = generator.generate_manual_illustration(
    step_description="Cerrar llave de paso principal",
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
# Abre en http://localhost:7860
```

## 📦 Dependencias Agregadas

```txt
torch>=2.0.0
transformers>=4.35.0
diffusers>=0.21.0
accelerate>=0.24.0
peft>=0.6.0
sentence-transformers>=2.2.0
gradio>=4.0.0
numpy>=1.24.0
tqdm>=4.66.0
wandb>=0.15.0
tensorboard>=2.14.0
scikit-learn>=1.3.0
```

## ⚙️ Configuración

### Variables de Entorno

```bash
# Modelos
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
GENERATION_MODEL=microsoft/DialoGPT-medium
IMAGE_MODEL=runwayml/stable-diffusion-v1-5
USE_SD_XL=false

# Dispositivos
DEVICE=cuda  # o cpu
USE_CUDA=true
TORCH_DTYPE=float16

# LoRA
USE_LORA=true
LORA_R=16
LORA_ALPHA=32
LORA_DROPOUT=0.1

# Entrenamiento
BATCH_SIZE=4
LEARNING_RATE=2e-4
NUM_EPOCHS=3
GRADIENT_ACCUMULATION_STEPS=4
MAX_LENGTH=512

# Weights & Biases
USE_WANDB=false
WANDB_API_KEY=tu-api-key
WANDB_PROJECT=manuales-hogar-ai

# Generación de Imágenes
IMAGE_WIDTH=512
IMAGE_HEIGHT=512
NUM_INFERENCE_STEPS=50
GUIDANCE_SCALE=7.5
```

## 🎨 Características Avanzadas

### Fine-tuning con LoRA
- Entrenamiento eficiente
- Menor uso de memoria
- Rápida adaptación a datos específicos
- Preserva conocimiento del modelo base

### Búsqueda Semántica
- Encuentra manuales relacionados automáticamente
- Scoring por similitud semántica
- No requiere palabras exactas
- Multilingüe

### Generación de Ilustraciones
- Estilo LEGO para claridad
- Ilustraciones técnicas
- Fotos realistas
- Optimizado para manuales

### Optimizaciones
- Mixed precision (FP16)
- Attention slicing
- VAE slicing
- Gradient accumulation
- Lazy loading de modelos

## 📈 Estadísticas

- **Nuevos Endpoints**: 8
- **Nuevos Servicios**: 2 (SemanticSearchService, LocalLLMService)
- **Nuevos Modelos ML**: 3 (ManualGeneratorModel, EmbeddingService, ImageGenerator)
- **Scripts**: 2 (train_model.py, gradio_demo.py)
- **Dependencias ML**: 11

## 🔧 Integración

Las funcionalidades de ML se integran con:
- ✅ Sistema de búsqueda existente
- ✅ Generación de manuales
- ✅ Recomendaciones
- ✅ Base de datos
- ✅ API REST

## 🎯 Casos de Uso

1. **Búsqueda Semántica**: Encontrar manuales sin palabras exactas
2. **Ilustraciones Automáticas**: Generar imágenes para pasos
3. **Modelos Personalizados**: Fine-tuning con datos propios
4. **Comparación de Contenido**: Encontrar manuales similares
5. **Demo Interactivo**: Probar funcionalidades con Gradio

## 🚀 Próximos Pasos

- [ ] Caché de embeddings
- [ ] Indexación vectorial (FAISS)
- [ ] RAG (Retrieval Augmented Generation)
- [ ] Fine-tuning de modelos de visión
- [ ] Generación de videos
- [ ] Modelos especializados por categoría




