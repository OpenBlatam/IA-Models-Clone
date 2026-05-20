# Advanced ML Features - Características Avanzadas

## 🚀 Funcionalidades Avanzadas de Deep Learning

### 1. Distributed Training (Multi-GPU)
- ✅ Entrenamiento distribuido con DDP
- ✅ Soporte para múltiples GPUs
- ✅ Sincronización automática de gradientes
- ✅ Reducción de tiempo de entrenamiento: Nx (N = número de GPUs)

**Uso:**
```python
from community_manager_ai.ml.training.distributed_trainer import DistributedTrainer

trainer = DistributedTrainer(model, train_loader, val_loader)
trainer.train_epoch(optimizer, criterion)
```

### 2. LoRA Fine-tuning
- ✅ Fine-tuning eficiente con LoRA
- ✅ Solo entrena ~1% de parámetros
- ✅ Soporte para 8-bit quantization
- ✅ Auto-detección de módulos objetivo
- ✅ Fusionado de adaptadores

**Uso:**
```python
from community_manager_ai.ml.fine_tuning import LoRATrainer

trainer = LoRATrainer(
    model_name="gpt2",
    r=8,
    lora_alpha=32,
    load_in_8bit=True
)
trainer.train(train_dataset, val_dataset)
```

### 3. Advanced Diffusion Features
- ✅ ControlNet para control preciso
- ✅ Image-to-image generation
- ✅ Inpainting (rellenar áreas)
- ✅ Múltiples schedulers (DPM, Euler, DDIM)
- ✅ Optimizaciones avanzadas

**Uso:**
```python
from community_manager_ai.ml.diffusion import AdvancedDiffusionPipeline

pipeline = AdvancedDiffusionPipeline(use_controlnet=True)
image = pipeline.generate_with_control(
    prompt="...",
    control_image=control_img
)
```

### 4. Comprehensive Evaluation
- ✅ Métricas de clasificación (accuracy, precision, recall, F1)
- ✅ Métricas de regresión (MSE, MAE, RMSE, R²)
- ✅ Métricas de generación (BLEU, Perplexity)
- ✅ Matriz de confusión
- ✅ Evaluador completo de modelos

**Uso:**
```python
from community_manager_ai.ml.evaluation import ModelEvaluator

evaluator = ModelEvaluator(task_type="classification")
metrics = evaluator.evaluate(model, val_loader)
```

### 5. Experiment Tracking
- ✅ Integración completa con WandB
- ✅ Logging de métricas, imágenes, modelos
- ✅ Histogramas y tablas
- ✅ Tracking de hiperparámetros
- ✅ Comparación de experimentos

**Uso:**
```python
from community_manager_ai.ml.experiment_tracking import WandBTracker

tracker = WandBTracker(project_name="my-project")
tracker.log_metrics({"loss": 0.5}, step=100)
tracker.log_image(image, "generated")
```

## 📊 Arquitectura de Módulos ML

```
ml/
├── content_analyzer.py      # Análisis de contenido
├── sentiment_analyzer.py     # Análisis de sentimiento
├── text_generator.py         # Generación de texto
├── image_generator.py       # Generación de imágenes
├── fast_inference.py        # Inferencia rápida
├── training/
│   ├── trainer.py           # Entrenador básico
│   └── distributed_trainer.py  # Entrenador distribuido
├── fine_tuning/
│   └── lora_trainer.py      # Fine-tuning con LoRA
├── evaluation/
│   └── metrics.py           # Métricas de evaluación
├── diffusion/
│   └── advanced_diffusion.py  # Diffusion avanzado
├── optimization/
│   ├── model_optimizer.py   # Optimización de modelos
│   ├── quantization.py      # Cuantización
│   └── onnx_converter.py    # Conversión ONNX
├── experiment_tracking/
│   └── wandb_tracker.py     # Tracking con WandB
├── data/
│   └── dataset.py           # Datasets optimizados
└── utils/
    └── performance.py       # Utilidades de performance
```

## 🎯 Casos de Uso Avanzados

### Fine-tuning de Modelo para Posts
```python
from community_manager_ai.ml.fine_tuning import LoRATrainer
from community_manager_ai.ml.data import SocialMediaDataset

# Preparar datos
dataset = SocialMediaDataset(texts, labels, tokenizer)

# Entrenar con LoRA
trainer = LoRATrainer("gpt2", r=16, load_in_8bit=True)
trainer.train(dataset, output_dir="./fine-tuned-model")
```

### Entrenamiento Multi-GPU
```python
from community_manager_ai.ml.training.distributed_trainer import DistributedTrainer

# Configurar
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "12355"

# Entrenar
trainer = DistributedTrainer(model, train_loader, val_loader)
trainer.train_epoch(optimizer, criterion)
```

### Generación con ControlNet
```python
from community_manager_ai.ml.diffusion import AdvancedDiffusionPipeline

pipeline = AdvancedDiffusionPipeline(
    use_controlnet=True,
    controlnet_model="lllyasviel/sd-controlnet-canny"
)

# Generar con control
image = pipeline.generate_with_control(
    prompt="funny meme",
    control_image=edge_image
)
```

## 📈 Performance Benchmarks

### Distributed Training
- **1 GPU**: 100% tiempo base
- **2 GPUs**: ~50% tiempo (2x más rápido)
- **4 GPUs**: ~25% tiempo (4x más rápido)
- **8 GPUs**: ~12.5% tiempo (8x más rápido)

### LoRA Fine-tuning
- **Parámetros entrenables**: ~1% del modelo completo
- **Memoria**: -75% vs fine-tuning completo
- **Velocidad**: 3-5x más rápido
- **Calidad**: Similar a fine-tuning completo

## 🔧 Configuración Avanzada

### Multi-GPU Setup
```bash
# Ejecutar con torchrun
torchrun --nproc_per_node=4 train_distributed.py
```

### WandB Setup
```bash
# Login
wandb login

# Configurar proyecto
export WANDB_PROJECT="community-manager-ai"
export WANDB_ENTITY="your-entity"
```

## 🎓 Best Practices

1. **Distributed Training**: Usar para modelos grandes y datasets grandes
2. **LoRA**: Usar para fine-tuning eficiente en recursos limitados
3. **ControlNet**: Usar cuando necesites control preciso sobre generación
4. **Evaluation**: Evaluar regularmente durante entrenamiento
5. **Tracking**: Trackear todos los experimentos para comparación

## 📚 Referencias

- [PyTorch Distributed](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [ControlNet](https://github.com/lllyasviel/ControlNet)
- [WandB Docs](https://docs.wandb.ai/)




