# ML Utils - Utilidades de Machine Learning y Deep Learning

Utilidades completas para entrenamiento, evaluación y fine-tuning de modelos siguiendo las mejores prácticas de PyTorch, Transformers y Deep Learning.

## Características

### 1. Training Utils (`training_utils.py`)

Utilidades para entrenamiento de modelos PyTorch:

- **Trainer**: Clase principal de entrenamiento con:
  - Mixed precision training (AMP)
  - Gradient accumulation
  - Gradient clipping
  - Early stopping
  - Learning rate scheduling
  - Checkpointing automático

- **TrainingConfig**: Configuración completa de entrenamiento
- **EarlyStopping**: Prevención de overfitting
- **LearningRateScheduler**: Múltiples estrategias (cosine, linear, step, plateau)

### 2. Data Utils (`data_utils.py`)

Utilidades para procesamiento de datos:

- **DataProcessor**: Pipeline de transformaciones
- **DatasetBuilder**: Construcción de datasets desde arrays o diccionarios
- **DataLoaderBuilder**: Creación de DataLoaders con configuración
- **Split utilities**: División train/val/test

### 3. Evaluation Utils (`evaluation_utils.py`)

Utilidades para evaluación de modelos:

- **ModelEvaluator**: Evaluación completa de modelos
- **MetricsCalculator**: Cálculo de métricas:
  - Clasificación: accuracy, precision, recall, F1
  - Regresión: MSE, MAE, RMSE, R²
  - Matriz de confusión
  - Reportes de clasificación

### 4. Fine-Tuning Utils (`fine_tuning_utils.py`)

Utilidades para fine-tuning de transformers:

- **LoRATrainer**: Fine-tuning eficiente con LoRA
- **LoRALayer**: Implementación de Low-Rank Adaptation
- **FineTuningConfig**: Configuración de fine-tuning
- **Tokenization**: Utilidades de tokenización

### 5. Model Utils (`model_utils.py`)

Utilidades para construcción y gestión de modelos:

- **ModelBuilder**: Construcción de modelos comunes (MLP, CNN)
- **ModelCheckpointer**: Gestión de checkpoints
- **load_pretrained_model**: Carga de modelos pre-entrenados

### 6. Diffusion Utils (`diffusion_utils.py`)

Utilidades para modelos de difusión:

- **DiffusionPipelineManager**: Gestor de pipelines de difusión
- **NoiseScheduler**: Gestor de noise schedulers

### 7. Experiment Tracking (`experiment_tracking.py`)

Utilidades para tracking de experimentos:

- **ExperimentTracker**: Tracking con TensorBoard y W&B
- **ExperimentManager**: Gestor de múltiples experimentos

### 8. Gradio Utils (`gradio_utils.py`)

Utilidades para interfaces Gradio:

- **GradioInterfaceBuilder**: Builder de interfaces
- **create_model_demo**: Crear demos de modelos
- **create_comparison_demo**: Comparar modelos

### 9. Optimization Utils (`optimization_utils.py`)

Utilidades de optimización:

- **MixedPrecisionManager**: Gestor de mixed precision
- **GradientAccumulator**: Acumulación de gradientes
- **ModelOptimizer**: Optimización de modelos
- **MultiGPUTrainer**: Entrenamiento multi-GPU
- **MemoryOptimizer**: Optimización de memoria

### 10. Augmentation Utils (`augmentation_utils.py`) - NUEVO

Utilidades de data augmentation:

- **TextAugmenter**: Augmentación de texto (sinónimos, inserción, eliminación, swap)
- **ImageAugmenter**: Augmentación de imágenes (rotación, color, blur, ruido)
- **TorchAugmenter**: Augmentación con torchvision transforms
- **MixUpAugmenter**: Técnica MixUp
- **CutMixAugmenter**: Técnica CutMix

### 11. Loss Utils (`loss_utils.py`) - NUEVO

Funciones de pérdida avanzadas:

- **FocalLoss**: Para desbalance de clases
- **DiceLoss**: Para segmentación
- **IoULoss**: Intersection over Union
- **LabelSmoothingLoss**: Label smoothing
- **TripletLoss**: Para embeddings
- **ContrastiveLoss**: Aprendizaje contrastivo
- **HuberLoss**: Regresión robusta
- **KLDivergenceLoss**: Knowledge distillation
- **CombinedLoss**: Combinación de losses

### 12. Cross-Validation Utils (`cv_utils.py`) - NUEVO

Utilidades de validación cruzada:

- **CrossValidator**: K-Fold y Stratified K-Fold
- **TimeSeriesCrossValidator**: Para series temporales
- **GroupKFold**: K-Fold con grupos
- Funciones helper para validación cruzada

### 13. Tokenization Utils (`tokenization_utils.py`) - NUEVO

Utilidades de tokenización:

- **TextPreprocessor**: Preprocesamiento avanzado de texto
- **AdvancedTokenizer**: Tokenizador con opciones avanzadas
- **DynamicPadding**: Padding dinámico
- **TokenizerWrapper**: Wrapper con preprocesamiento

### 14. Interpretability Utils (`interpretability_utils.py`) - NUEVO

Utilidades de interpretabilidad:

- **AttentionVisualizer**: Visualización de atención
- **GradientAnalyzer**: Análisis de gradientes
- **FeatureImportance**: Importancia de features
- **CaptumWrapper**: Integración con Captum

### 15. Ensemble Utils (`ensemble_utils.py`) - NUEVO

Utilidades de ensembles:

- **ModelEnsemble**: Ensemble de modelos (average, weighted, voting)
- **StackingEnsemble**: Stacking con meta-learner
- **BaggingEnsemble**: Bagging con bootstrap

### 16. Inference Utils (`inference_utils.py`) - NUEVO

Utilidades de inferencia:

- **BatchInferenceManager**: Gestor de inferencia por batches
- **ONNXExporter**: Exportación a ONNX
- **ONNXRuntimeInference**: Inferencia con ONNX Runtime
- **InferenceOptimizer**: Optimización para inferencia
- **TorchServeExporter**: Exportación para TorchServe
- **InferenceBenchmark**: Benchmark de inferencia

### 17. Optimizer Utils (`optimizer_utils.py`) - NUEVO

Optimizadores avanzados:

- **Lookahead**: Optimizador Lookahead
- **RAdam**: Rectified Adam
- **AdaBound**: AdaBound optimizer
- **create_optimizer**: Helper para crear optimizadores

## Uso

### Entrenamiento Básico

```python
from ml_utils import Trainer, TrainingConfig, DataLoaderBuilder

# Configuración
config = TrainingConfig(
    epochs=10,
    batch_size=32,
    learning_rate=1e-4,
    use_mixed_precision=True
)

# Crear trainer
trainer = Trainer(model, config)

# Entrenar
history = trainer.train(train_loader, val_loader)
```

### Fine-Tuning con LoRA

```python
from ml_utils import LoRATrainer, FineTuningConfig

# Configuración
config = FineTuningConfig(
    model_name="bert-base-uncased",
    use_lora=True,
    lora_r=8,
    lora_alpha=16
)

# Crear trainer
trainer = LoRATrainer("bert-base-uncased", config)

# Tokenizar datos
encodings = trainer.tokenize_data(texts, labels)
```

### Evaluación

```python
from ml_utils import ModelEvaluator

# Evaluar modelo
evaluator = ModelEvaluator(model)
metrics = evaluator.evaluate(val_loader, task_type="classification")

print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"F1 Score: {metrics['f1']:.4f}")
```

## Mejores Prácticas Implementadas

- ✅ Mixed precision training (AMP)
- ✅ Gradient accumulation
- ✅ Gradient clipping
- ✅ Early stopping
- ✅ Learning rate scheduling
- ✅ Checkpointing automático
- ✅ Evaluación durante entrenamiento
- ✅ LoRA para fine-tuning eficiente
- ✅ Procesamiento de datos optimizado
- ✅ Métricas completas de evaluación

#### Diffusion Utils (`diffusion_utils.py`)
- ✅ **DiffusionPipelineManager**: Gestor completo de pipelines de diffusion
  - Soporte para Stable Diffusion y Stable Diffusion XL
  - Múltiples schedulers (DDIM, DDPM, PNDM, Euler, DPM)
  - Generación de imágenes con configuración completa
  - Guardado automático de imágenes
- ✅ **NoiseScheduler**: Gestor de noise schedulers

#### Experiment Tracking (`experiment_tracking.py`)
- ✅ **ExperimentTracker**: Tracking completo de experimentos
  - Soporte para TensorBoard
  - Soporte para Weights & Biases
  - Logging de métricas, hiperparámetros e imágenes
  - Historial local de experimentos
- ✅ **ExperimentManager**: Gestor de múltiples experimentos

#### Gradio Utils (`gradio_utils.py`)
- ✅ **GradioInterfaceBuilder**: Builder para interfaces Gradio
  - Componentes predefinidos (text, image, slider, dropdown)
  - Interfaces simples y avanzadas
  - Soporte para múltiples inputs/outputs
- ✅ **create_model_demo**: Crear demos simples de modelos
- ✅ **create_comparison_demo**: Comparar múltiples modelos

#### Optimization Utils (`optimization_utils.py`)
- ✅ **MixedPrecisionManager**: Gestor de mixed precision training
- ✅ **GradientAccumulator**: Acumulación de gradientes
- ✅ **ModelOptimizer**: Optimización de modelos
  - Compilación de modelos (PyTorch 2.0+)
  - Optimización para inferencia
  - Cuantización (dynamic, static, QAT)
- ✅ **MultiGPUTrainer**: Entrenamiento multi-GPU
  - DataParallel
  - DistributedDataParallel
- ✅ **MemoryOptimizer**: Optimización de memoria
  - Gradient checkpointing
  - Limpieza de caché CUDA
  - Estadísticas de memoria

## Uso

### Entrenamiento Básico

```python
from ml_utils import Trainer, TrainingConfig, DataLoaderBuilder

# Configuración
config = TrainingConfig(
    epochs=10,
    batch_size=32,
    learning_rate=1e-4,
    use_mixed_precision=True
)

# Crear trainer
trainer = Trainer(model, config)

# Entrenar
history = trainer.train(train_loader, val_loader)
```

### Fine-Tuning con LoRA

```python
from ml_utils import LoRATrainer, FineTuningConfig

# Configuración
config = FineTuningConfig(
    model_name="bert-base-uncased",
    use_lora=True,
    lora_r=8,
    lora_alpha=16
)

# Crear trainer
trainer = LoRATrainer("bert-base-uncased", config)

# Tokenizar datos
encodings = trainer.tokenize_data(texts, labels)
```

### Diffusion Models

```python
from ml_utils import DiffusionPipelineManager

# Crear gestor
manager = DiffusionPipelineManager(
    model_id="runwayml/stable-diffusion-v1-5",
    pipeline_type="stable-diffusion"
)

# Cargar pipeline
manager.load_pipeline(scheduler_type="ddim")

# Generar imágenes
images = manager.generate(
    prompt="A beautiful landscape",
    num_inference_steps=50,
    guidance_scale=7.5
)

# Guardar
manager.save_images(images, "./output")
```

### Experiment Tracking

```python
from ml_utils import ExperimentTracker

# Crear tracker
tracker = ExperimentTracker(
    experiment_name="my_experiment",
    use_tensorboard=True,
    use_wandb=True
)

# Logging
tracker.log_metric("loss", 0.5, step=1)
tracker.log_hyperparameters({"lr": 1e-4, "batch_size": 32})
```

### Gradio Interface

```python
from ml_utils import GradioInterfaceBuilder

# Crear interfaz
builder = GradioInterfaceBuilder(
    title="Text Classifier",
    description="Classify text using our model"
)

builder.add_text_input("Input Text")
builder.set_function(predict_fn)

interface = builder.build()
interface.launch()
```

### Optimización

```python
from ml_utils import MixedPrecisionManager, ModelOptimizer

# Mixed precision
mp_manager = MixedPrecisionManager(enabled=True)

with mp_manager.autocast():
    output = model(input)

# Optimizar modelo
optimized_model = ModelOptimizer.compile_model(model)
optimized_model = ModelOptimizer.optimize_for_inference(optimized_model)
```

## Dependencias

- `torch` - PyTorch
- `transformers` - Hugging Face Transformers (opcional)
- `diffusers` - Diffusion models (opcional)
- `gradio` - Interfaces interactivas (opcional)
- `tensorboard` - Experiment tracking (opcional)
- `wandb` - Weights & Biases (opcional)
- `numpy` - Operaciones numéricas
- `sklearn` - Métricas de evaluación

