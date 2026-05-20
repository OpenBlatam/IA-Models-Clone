# Quick Start Guide - CustomLLMTrainer

## Instalación Rápida

```bash
pip install transformers torch datasets
```

## Uso Más Simple (3 líneas)

```python
from llm_trainer import CustomLLMTrainer

trainer = CustomLLMTrainer(
    model_name="gpt2",
    dataset_path="data/training.json",
    output_dir="./checkpoints"
)

trainer.train()
```

## Formato del Dataset

Tu dataset debe ser un archivo JSON con este formato:

```json
[
  {
    "prompt": "¿Qué es la inteligencia artificial?",
    "response": "La inteligencia artificial es la simulación de inteligencia humana en máquinas."
  },
  {
    "prompt": "Explica el machine learning.",
    "response": "El machine learning es un subconjunto de la IA que permite a las máquinas aprender de datos."
  }
]
```

## Configuración por Defecto

El trainer viene con configuraciones optimizadas por defecto:

- **Learning Rate**: 3e-5 (estándar para fine-tuning)
- **Epochs**: 3 (suficiente para la mayoría de casos)
- **Batch Size**: 8 (auto-ajustado según tu GPU)
- **Max Length**: 512 tokens

## Ejemplos Comunes

### Entrenamiento Básico

```python
from llm_trainer import CustomLLMTrainer

trainer = CustomLLMTrainer(
    model_name="gpt2",
    dataset_path="training.json"
)

trainer.train()
```

### Con Evaluación y Early Stopping

```python
trainer = CustomLLMTrainer(
    model_name="gpt2",
    dataset_path="training.json",
    evaluation_strategy="steps",
    eval_steps=100,
    early_stopping_patience=3,
    load_best_model_at_end=True
)

trainer.train()
```

### Optimizado para GPU Pequeña

```python
trainer = CustomLLMTrainer(
    model_name="gpt2",
    dataset_path="training.json",
    batch_size=2,
    gradient_accumulation_steps=4,  # Effective batch = 8
    gradient_checkpointing=True,
    fp16=True
)

trainer.train()
```

### Usando Factory (Presets)

```python
from llm_trainer import TrainerFactory

factory = TrainerFactory()

# Trainer básico
trainer = factory.create_basic_trainer(
    model_name="gpt2",
    dataset_path="training.json"
)

# Trainer avanzado
trainer = factory.create_advanced_trainer(
    model_name="gpt2",
    dataset_path="training.json",
    enable_early_stopping=True
)

# Trainer optimizado para memoria
trainer = factory.create_memory_efficient_trainer(
    model_name="gpt2",
    dataset_path="training.json"
)

trainer.train()
```

### Usando Builder (Configuración Compleja)

```python
from llm_trainer import ConfigBuilder, CustomLLMTrainer

config = (ConfigBuilder()
    .with_model("gpt2")
    .with_dataset("training.json")
    .with_learning_rate(3e-5)
    .with_epochs(3)
    .with_batch_size(8)
    .with_early_stopping(patience=3)
    .build())

trainer = CustomLLMTrainer(**config)
trainer.train()
```

## Generar Texto Después del Entrenamiento

```python
# Después de entrenar
trainer = CustomLLMTrainer(...)
trainer.train()

# Generar respuestas
responses = trainer.predict(
    prompts=["¿Qué es Python?", "Explica las redes neuronales."],
    max_new_tokens=100,
    temperature=0.7
)

for prompt, response in zip(prompts, responses):
    print(f"P: {prompt}")
    print(f"R: {response}\n")
```

## Solución de Problemas Comunes

### Error: "CUDA out of memory"

**Solución:**
```python
trainer = CustomLLMTrainer(
    model_name="gpt2",
    dataset_path="training.json",
    batch_size=2,  # Reducir
    gradient_accumulation_steps=4,  # Compensar
    gradient_checkpointing=True,  # Ahorrar memoria
    fp16=True  # Precisión mixta
)
```

### Error: "Dataset format invalid"

**Verifica que tu JSON tenga:**
- Formato correcto (lista de objetos)
- Campos "prompt" y "response"
- Ambos campos como strings

### Error: "Model not found"

**Soluciones:**
- Verifica que el nombre del modelo sea correcto
- Si es un modelo local, verifica la ruta
- Asegúrate de tener conexión a internet para descargar

### Entrenamiento muy lento

**Optimizaciones:**
```python
trainer = CustomLLMTrainer(
    model_name="gpt2",
    dataset_path="training.json",
    fp16=True,  # Activar precisión mixta
    dataloader_num_workers=4,  # Paralelizar carga de datos
    gradient_checkpointing=True  # Si tienes GPU pequeña
)
```

## Próximos Pasos

- Lee `README.md` para documentación completa
- Revisa `ARCHITECTURE.md` para entender la estructura
- Explora `examples/` para más ejemplos

