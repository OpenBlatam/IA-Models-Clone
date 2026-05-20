# 🧠 Deep Learning Features - 3D Prototype AI

## 🎯 Características de Deep Learning Agregadas

### ✨ Nuevos Sistemas Implementados

#### 1. Advanced LLM System (`utils/advanced_llm_system.py`)
Sistema avanzado de LLM con Transformers:
- ✅ Carga de modelos pre-entrenados (GPT, T5, etc.)
- ✅ Fine-tuning con LoRA (Low-Rank Adaptation)
- ✅ Cuantización 4-bit para eficiencia
- ✅ Generación de texto optimizada
- ✅ Soporte para modelos causal y seq2seq
- ✅ Optimizaciones de GPU y mixed precision

**Características:**
- Modelos soportados: Causal LM, Seq2Seq
- Fine-tuning eficiente con LoRA
- Cuantización para reducir memoria
- Generación configurable (temperature, top_p, top_k)

#### 2. Diffusion Models System (`utils/diffusion_models.py`)
Sistema de modelos de difusión para generación de imágenes:
- ✅ Stable Diffusion (v1.5 y XL)
- ✅ Múltiples schedulers (DDIM, DDPM, Euler, DPM-Solver)
- ✅ Generación de imágenes desde prompts
- ✅ Optimizaciones (attention slicing, VAE slicing)
- ✅ Generación de descripciones visuales 3D

**Características:**
- Pipelines: StableDiffusion, StableDiffusionXL
- Schedulers: DDIM, DDPM, PNDM, Euler, DPM-Solver
- Optimizaciones de memoria
- Generación batch

#### 3. Model Training System (`utils/model_training.py`)
Sistema de entrenamiento eficiente:
- ✅ Mixed precision training (FP16)
- ✅ Gradient accumulation
- ✅ Gradient clipping
- ✅ Learning rate scheduling
- ✅ Early stopping
- ✅ Checkpointing automático

**Características:**
- Entrenamiento optimizado para GPU
- Soporte para datasets grandes
- Validación durante entrenamiento
- Historial de métricas

#### 4. Gradio Interface (`utils/gradio_interface.py`)
Interfaz interactiva con Gradio:
- ✅ Generación de prototipos
- ✅ Generación con LLM
- ✅ Generación de imágenes
- ✅ Interfaz amigable
- ✅ Visualización en tiempo real

**Características:**
- Múltiples tabs para diferentes funcionalidades
- Visualización de resultados
- Inputs configurables
- Error handling robusto

## 🚀 Nuevos Endpoints API (9)

### LLM Endpoints (4)
1. `POST /api/v1/llm/models/load` - Carga modelo LLM
2. `POST /api/v1/llm/generate` - Genera texto con LLM
3. `POST /api/v1/llm/fine-tune` - Fine-tune de modelo
4. `GET /api/v1/llm/models/{id}` - Info de modelo

### Diffusion Endpoints (4)
5. `POST /api/v1/diffusion/pipelines/load` - Carga pipeline
6. `POST /api/v1/diffusion/generate-image` - Genera imagen
7. `POST /api/v1/diffusion/generate-3d-description` - Genera descripción 3D
8. `GET /api/v1/diffusion/pipelines/{id}` - Info de pipeline

### System Endpoints (1)
9. `GET /api/v1/system/health-check` - Health check mejorado

## 📦 Dependencias Agregadas

```txt
# Deep Learning
torch>=2.0.0
transformers>=4.35.0
diffusers>=0.21.0
accelerate>=0.24.0
peft>=0.6.0
bitsandbytes>=0.41.0

# Gradio
gradio>=4.0.0

# Utilities
numpy>=1.24.0
tqdm>=4.66.0
pillow>=10.0.0
tensorboard>=2.14.0
wandb>=0.15.0
```

## 💻 Ejemplos de Uso

### Cargar y Usar LLM

```python
from utils.advanced_llm_system import AdvancedLLMSystem, LLMConfig, ModelType

llm = AdvancedLLMSystem()

# Cargar modelo
config = LLMConfig(
    model_name="gpt2",
    model_type=ModelType.CAUSAL_LM,
    use_lora=True,
    use_quantization=True
)
llm.load_model("gpt2-model", config)

# Generar texto
result = llm.generate_text(
    "gpt2-model",
    "Describe a powerful blender:",
    max_new_tokens=100
)
print(result["generated_text"])
```

### Generar Imagen con Diffusion

```python
from utils.diffusion_models import DiffusionModelsSystem, DiffusionConfig, SchedulerType

diffusion = DiffusionModelsSystem()

# Cargar pipeline
config = DiffusionConfig(
    model_name="runwayml/stable-diffusion-v1-5",
    scheduler_type=SchedulerType.DPMSOLVER
)
diffusion.load_pipeline("sd-v1-5", config)

# Generar imagen
result = diffusion.generate_image(
    "sd-v1-5",
    "3D model of a blender, high quality, detailed",
    negative_prompt="blurry, low quality",
    num_images=1
)
```

### Fine-tuning con LoRA

```python
# Fine-tune modelo
training_texts = [
    "A powerful blender with 1000W motor...",
    "A modern stove with 4 burners...",
    # ... más textos
]

result = llm.fine_tune(
    "gpt2-model",
    training_texts,
    num_epochs=3,
    batch_size=4,
    learning_rate=2e-5
)
```

### Usar Gradio Interface

```python
from utils.gradio_interface import GradioInterface
from core.prototype_generator import PrototypeGenerator

generator = PrototypeGenerator()
llm = AdvancedLLMSystem()
diffusion = DiffusionModelsSystem()

interface = GradioInterface(
    prototype_generator=generator,
    llm_system=llm,
    diffusion_system=diffusion
)

interface.launch(server_port=7860)
```

## 🎯 Casos de Uso

### 1. Mejora de Descripciones
Usar LLM para mejorar descripciones de productos generadas automáticamente.

### 2. Generación Visual
Generar imágenes de referencia para prototipos usando diffusion models.

### 3. Fine-tuning Personalizado
Entrenar modelos específicos para el dominio de prototipos 3D.

### 4. Interfaz Interactiva
Proporcionar interfaz web amigable para usuarios no técnicos.

## ⚙️ Configuración

### GPU Requirements
- CUDA 11.8+ recomendado
- Mínimo 8GB VRAM para modelos base
- 16GB+ VRAM para Stable Diffusion XL

### Optimizaciones
- Mixed precision (FP16) habilitado por defecto
- Attention slicing para reducir memoria
- VAE slicing para eficiencia
- Cuantización 4-bit disponible

## 📊 Estadísticas

- **Nuevos módulos**: 4
- **Nuevos endpoints**: 9
- **Líneas de código**: ~1,500+
- **Dependencias**: 10 nuevas

## 🎉 Conclusión

El sistema ahora incluye capacidades completas de deep learning:
- ✅ LLM avanzado con fine-tuning
- ✅ Diffusion models para imágenes
- ✅ Entrenamiento optimizado
- ✅ Interfaz interactiva

**¡Sistema ahora con capacidades completas de IA generativa!** 🚀🧠




