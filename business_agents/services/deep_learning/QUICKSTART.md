# Quick Start Guide

## 🚀 Inicio Rápido

### 1. Instalación

```bash
pip install torch transformers diffusers gradio numpy pyyaml
pip install structlog orjson  # Opcional pero recomendado
pip install peft  # Para LoRA
pip install tensorboard wandb  # Para experiment tracking
```

### 2. Uso Básico

```python
from deep_learning import DeepLearningService
import numpy as np

# Inicializar servicio
service = DeepLearningService()

# Crear modelo
model = service.create_model("transformer", model_id="my_model")

# Crear datos
data = np.random.randn(1000, 512)
labels = np.random.randint(0, 2, 1000)
dataset = service.create_dataset(data, labels)

# Entrenar
from deep_learning.data import create_dataloader, split_dataset
train_ds, val_ds, _ = split_dataset(dataset, 0.7, 0.15, 0.15)
train_loader = create_dataloader(train_ds, batch_size=32)
val_loader = create_dataloader(val_ds, batch_size=32)

history = service.train_model(model, train_loader, val_loader)
```

### 3. Con Configuración YAML

```python
from deep_learning import DeepLearningService

# Cargar configuración
service = DeepLearningService(config_path="config/default_config.yaml")

# El servicio usa automáticamente la configuración
model = service.create_model("transformer")
# ... resto del código
```

### 4. HuggingFace Models

```python
from deep_learning.models.transformers_models import HuggingFaceModel

model = HuggingFaceModel(
    model_name="bert-base-uncased",
    task_type="classification",
    num_labels=2
)

predictions = model.predict(["Positive text", "Negative text"])
```

### 5. Diffusion Models

```python
from deep_learning.models.diffusion_models import DiffusionModel

model = DiffusionModel("runwayml/stable-diffusion-v1-5")
result = model.generate("A beautiful landscape")
```

### 6. Gradio Demo

```python
from deep_learning.gradio_apps import create_transformers_demo

demo = create_transformers_demo(model)
demo.launch(share=True)
```

## 📖 Más Información

Ver `README.md` para documentación completa y `examples/example_usage.py` para más ejemplos.



