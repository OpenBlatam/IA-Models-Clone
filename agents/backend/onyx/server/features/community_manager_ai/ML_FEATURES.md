# ML Features - Machine Learning Capabilities

## 🤖 Capacidades de Machine Learning

### 1. Content Analyzer
- ✅ Análisis de sentimiento usando transformers
- ✅ Análisis de calidad de contenido
- ✅ Validación por plataforma
- ✅ Detección de hashtags, menciones, enlaces
- ✅ Batch processing

**Modelos soportados:**
- `cardiffnlp/twitter-roberta-base-sentiment-latest`
- `distilbert-base-uncased`
- Modelos personalizados

### 2. Sentiment Analyzer
- ✅ Arquitectura personalizada con PyTorch
- ✅ Fine-tuning con LoRA
- ✅ Clasificación multi-clase
- ✅ Probabilidades de confianza
- ✅ Batch inference

**Características:**
- Dropout para regularización
- Capas de clasificación personalizadas
- Optimización para GPU

### 3. Advanced Text Generator
- ✅ Generación de texto con modelos grandes
- ✅ Soporte para GPT, Llama, etc.
- ✅ Fine-tuning con LoRA
- ✅ Cuantización 8-bit para ahorrar memoria
- ✅ Generación optimizada por plataforma

**Características:**
- Temperature control
- Nucleus sampling (top-p)
- Longitud configurable
- Múltiples secuencias

### 4. Image Generator
- ✅ Generación de imágenes con Stable Diffusion
- ✅ Soporte para SD v1.5 y SDXL
- ✅ Optimizaciones de memoria
- ✅ Generación de memes
- ✅ Batch generation

**Características:**
- Mixed precision (float16)
- Attention slicing
- VAE slicing
- Scheduler optimizado (DPMSolver)

### 5. Model Trainer
- ✅ Sistema de entrenamiento completo
- ✅ Mixed precision training
- ✅ Gradient accumulation
- ✅ Gradient clipping
- ✅ Learning rate scheduling
- ✅ Early stopping
- ✅ Experiment tracking (wandb)
- ✅ Checkpointing

**Características:**
- Multi-GPU support
- Progress bars
- Validation durante entrenamiento
- Best model saving

### 6. Gradio Demo
- ✅ Interfaz interactiva completa
- ✅ Análisis de contenido en tiempo real
- ✅ Generación de texto
- ✅ Generación de imágenes
- ✅ Múltiples tabs
- ✅ Input validation

## 🚀 Uso

### Análisis de Contenido
```python
from community_manager_ai.ml import ContentAnalyzer

analyzer = ContentAnalyzer()
analysis = analyzer.analyze_content_quality(
    content="Mi post aquí",
    platform="twitter"
)
```

### Generación de Texto
```python
from community_manager_ai.ml import AdvancedTextGenerator

generator = AdvancedTextGenerator(model_name="gpt2")
text = generator.generate_post(
    topic="Tecnología",
    platform="linkedin",
    tone="professional"
)
```

### Generación de Imágenes
```python
from community_manager_ai.ml import ImageGenerator

generator = ImageGenerator()
image = generator.generate(
    prompt="funny cat meme, colorful",
    num_inference_steps=30
)
```

### Entrenamiento
```python
from community_manager_ai.ml.training import ModelTrainer

trainer = ModelTrainer(model, train_loader, val_loader)
trainer.train(
    num_epochs=10,
    learning_rate=2e-5,
    use_wandb=True
)
```

### Demo Interactivo
```bash
python scripts/launch_gradio_demo.py --port 7860 --share
```

## 📊 Requisitos

### Hardware
- GPU recomendada (CUDA) para mejor performance
- Mínimo 8GB VRAM para generación de imágenes
- CPU funciona pero más lento

### Software
- PyTorch 2.0+
- CUDA 11.8+ (opcional pero recomendado)
- Python 3.9+

## 🔧 Configuración

### Variables de Entorno
```env
CUDA_VISIBLE_DEVICES=0  # Especificar GPU
WANDB_API_KEY=your_key  # Para experiment tracking
```

### Modelos Pre-entrenados
Los modelos se descargan automáticamente la primera vez.
Se guardan en `~/.cache/huggingface/`

## 📈 Performance

### Optimizaciones Implementadas
- ✅ Mixed precision training (float16)
- ✅ Gradient accumulation
- ✅ Attention slicing
- ✅ VAE slicing
- ✅ 8-bit quantization
- ✅ LoRA para fine-tuning eficiente

### Benchmarks
- **Análisis de sentimiento**: ~50ms por texto (GPU)
- **Generación de texto**: ~200-500ms (depende del modelo)
- **Generación de imagen**: ~5-10s (SD v1.5, 30 steps)

## 🎯 Casos de Uso

1. **Análisis automático de posts**: Validar contenido antes de publicar
2. **Generación de contenido**: Crear posts con IA
3. **Generación de memes**: Crear imágenes para memes
4. **Fine-tuning**: Entrenar modelos en datos específicos
5. **Análisis de engagement**: Predecir performance de posts

## 🔬 Experiment Tracking

Integración con Weights & Biases:
- Métricas de entrenamiento
- Visualización de pérdidas
- Comparación de experimentos
- Model versioning

## 📚 Documentación Adicional

Ver documentación de:
- [PyTorch](https://pytorch.org/docs/)
- [Transformers](https://huggingface.co/docs/transformers)
- [Diffusers](https://huggingface.co/docs/diffusers)
- [Gradio](https://gradio.app/docs/)




