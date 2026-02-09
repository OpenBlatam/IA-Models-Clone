# 🎓 Guía Avanzada de Entrenamiento

## Sistema Completo de Entrenamiento

### 1. **Trainer Profesional** ✅

#### Características
- Mixed precision training (FP16)
- Gradient accumulation
- Gradient clipping
- Learning rate scheduling
- Early stopping
- Checkpointing automático
- Experiment tracking

**Uso:**
```python
from ml_advanced.training.trainer import Trainer
from torch.utils.data import DataLoader

trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    device="cuda",
    use_mixed_precision=True,
    gradient_accumulation_steps=4,
    max_grad_norm=1.0
)

# Entrenar
result = trainer.train(
    num_epochs=10,
    optimizer=optimizer,
    loss_fn=loss_fn,
    scheduler=scheduler,
    checkpoint_dir="./checkpoints",
    save_best=True,
    early_stopping_patience=5
)
```

### 2. **Experiment Tracking** ✅

#### WandB y TensorBoard
- Tracking automático de métricas
- Visualización de curvas de entrenamiento
- Logging de modelos
- Comparación de experimentos

**Uso:**
```python
from ml_advanced.training.experiment_tracker import ExperimentTracker

tracker = ExperimentTracker(
    tracker_type="wandb",
    project_name="identity-clone",
    experiment_name="experiment-1",
    config={"learning_rate": 5e-5, "batch_size": 8}
)

# Durante entrenamiento
tracker.log({"train_loss": 0.5, "val_loss": 0.4}, step=epoch)

# Finalizar
tracker.finish()
```

### 3. **Modelos Personalizados** ✅

#### Arquitecturas Custom
- `IdentityStyleEncoder` - Encoder de estilo
- `ContentGeneratorModel` - Generador de contenido
- `PositionalEncoding` - Encoding posicional

**Uso:**
```python
from ml_advanced.models.custom_models import IdentityStyleEncoder

encoder = IdentityStyleEncoder(
    vocab_size=50257,
    d_model=768,
    nhead=12,
    num_layers=6
)

# Forward pass
style_features = encoder(input_ids, attention_mask)
```

### 4. **Configuración YAML** ✅

#### Hyperparameters en YAML
- Configuración centralizada
- Fácil experimentación
- Versionado de configs

**Ejemplo:**
```yaml
training:
  num_epochs: 10
  batch_size: 8
  learning_rate: 5e-5
  mixed_precision: true

model:
  base_model: "gpt2"
  lora:
    r: 8
    lora_alpha: 16
```

**Cargar:**
```python
from ml_advanced.training.config_loader import ConfigLoader

config = ConfigLoader.load("config.yaml")
training_config = ConfigLoader.load_training_config("config.yaml")
```

### 5. **Sistema de Evaluación** ✅

#### Métricas Completas
- Clasificación: accuracy, precision, recall, F1
- Generación: BLEU, ROUGE (básico)
- Embeddings: similitud coseno

**Uso:**
```python
from ml_advanced.training.evaluator import Evaluator

evaluator = Evaluator(device="cuda")

# Evaluar clasificación
metrics = evaluator.evaluate_classification(
    model=model,
    data_loader=val_loader,
    num_classes=2
)
```

## Pipeline Completo de Entrenamiento

### 1. Preparar Datos

```python
from torch.utils.data import Dataset, DataLoader

class IdentityDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten()
        }

# Crear dataloaders
train_dataset = IdentityDataset(train_texts, tokenizer)
train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)
```

### 2. Configurar Modelo

```python
from ml_advanced.lora_finetuning import get_lora_finetuner, LoRAConfig

finetuner = get_lora_finetuner()
lora_config = LoRAConfig(r=8, lora_alpha=16)
model, tokenizer = finetuner.prepare_model_for_lora("gpt2", lora_config)
```

### 3. Configurar Optimizador y Scheduler

```python
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

optimizer = AdamW(
    model.parameters(),
    lr=5e-5,
    weight_decay=0.01,
    betas=(0.9, 0.999)
)

scheduler = CosineAnnealingLR(
    optimizer,
    T_max=1000,
    eta_min=1e-7
)
```

### 4. Definir Loss Function

```python
def loss_fn(outputs, batch):
    logits = outputs.logits
    labels = batch["input_ids"]
    
    # Shift para language modeling
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    loss_fct = nn.CrossEntropyLoss()
    loss = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1)
    )
    return loss
```

### 5. Entrenar

```python
from ml_advanced.training.trainer import Trainer
from ml_advanced.training.experiment_tracker import ExperimentTracker

# Tracker
tracker = ExperimentTracker("wandb", "identity-clone")

# Trainer
trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    device="cuda",
    use_mixed_precision=True
)

# Entrenar
result = trainer.train(
    num_epochs=10,
    optimizer=optimizer,
    loss_fn=loss_fn,
    scheduler=scheduler,
    checkpoint_dir="./checkpoints",
    experiment_tracker=tracker
)

tracker.finish()
```

## Mejores Prácticas

### 1. Data Loading
- Usar `num_workers > 0` para paralelismo
- `pin_memory=True` para GPU
- Shuffle en training
- Prefetch para eficiencia

### 2. Mixed Precision
- Usar FP16 en GPU
- Gradient scaling automático
- Detectar NaN/Inf

### 3. Gradient Accumulation
- Para batches efectivos grandes
- Útil con memoria limitada
- Ajustar learning rate proporcionalmente

### 4. Checkpointing
- Guardar regularmente
- Guardar mejor modelo
- Limitar número de checkpoints

### 5. Experiment Tracking
- Log todas las métricas
- Guardar configuraciones
- Comparar experimentos

## Troubleshooting

### Out of Memory
- Reducir batch_size
- Usar gradient_accumulation_steps
- Habilitar mixed precision
- Reducir max_length

### Training Instability
- Reducir learning_rate
- Aumentar warmup_steps
- Usar gradient clipping
- Verificar inicialización

### Slow Training
- Usar GPU
- Optimizar data loading
- Usar mixed precision
- Verificar bottlenecks

## Próximas Mejoras

- [ ] Distributed training (DDP)
- [ ] Model quantization
- [ ] Pruning
- [ ] Knowledge distillation
- [ ] Advanced schedulers
- [ ] Custom loss functions




