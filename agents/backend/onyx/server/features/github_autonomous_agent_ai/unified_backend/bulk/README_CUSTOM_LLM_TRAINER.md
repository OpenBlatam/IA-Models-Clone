# Custom LLM Trainer Module

## Descripción

El módulo `CustomLLMTrainer` proporciona una clase personalizada para entrenar modelos de lenguaje (LLMs) usando la biblioteca Hugging Face Transformers. Está diseñado para simplificar el proceso de fine-tuning en datasets JSON con formato prompt-response.

## Características Principales

- ✅ **Carga de datasets JSON**: Acepta datasets en formato JSON con campos "prompt" y "response"
- ✅ **Tokenización automática**: Usa tokenizers pre-entrenados de Hugging Face
- ✅ **Configuración flexible**: Parámetros de entrenamiento configurables
- ✅ **Soporte GPU/TPU**: Detección automática y uso de aceleradores hardware
- ✅ **Checkpoints automáticos**: Guarda checkpoints durante y al final del entrenamiento
- ✅ **Logging completo**: Sistema de logging detallado para monitoreo
- ✅ **Manejo de errores robusto**: Validación de datos y manejo de excepciones

## Instalación

Asegúrate de tener las dependencias necesarias instaladas:

```bash
pip install transformers torch datasets accelerate
```

Para soporte TPU (opcional):
```bash
pip install torch-xla
```

## Formato del Dataset

El dataset debe ser un archivo JSON con el siguiente formato:

```json
[
  {
    "prompt": "¿Qué es la inteligencia artificial?",
    "response": "La inteligencia artificial es la simulación de inteligencia humana en máquinas."
  },
  {
    "prompt": "Explica el aprendizaje profundo.",
    "response": "El aprendizaje profundo es un subconjunto del machine learning que usa redes neuronales multicapa."
  }
]
```

Alternativamente, puedes usar un objeto con claves "data" o "examples":

```json
{
  "data": [
    {
      "prompt": "...",
      "response": "..."
    }
  ]
}
```

## Uso Básico

```python
from custom_llm_trainer import CustomLLMTrainer

# Inicializar el trainer
trainer = CustomLLMTrainer(
    model_name="gpt2",
    dataset_path="data/training.json",
    output_dir="./checkpoints",
    learning_rate=3e-5,
    num_train_epochs=3,
    batch_size=8
)

# Entrenar el modelo
trainer.train()
```

## Parámetros Principales

### Parámetros Requeridos

- `model_name` (str): Nombre o ruta del modelo pre-entrenado (ej: "gpt2", "t5-small", "microsoft/DialoGPT-medium")
- `dataset_path` (str/Path): Ruta al archivo JSON con el dataset

### Parámetros de Entrenamiento

- `learning_rate` (float, default=3e-5): Tasa de aprendizaje
- `num_train_epochs` (int, default=3): Número de épocas
- `batch_size` (int, default=8): Tamaño del batch por dispositivo
- `max_length` (int, default=512): Longitud máxima de secuencia
- `gradient_accumulation_steps` (int, default=1): Pasos de acumulación de gradiente
- `warmup_steps` (int, optional): Pasos de warmup (auto-calculado si no se especifica)
- `weight_decay` (float, default=0.01): Decaimiento de pesos para regularización

### Parámetros de GPU/TPU

- `fp16` (bool, default=False): Usar precisión mixta FP16 (requiere GPU compatible)
- `bf16` (bool, default=False): Usar precisión mixta BF16 (requiere GPU Ampere+)

### Parámetros de Guardado

- `output_dir` (str/Path, default="./checkpoints"): Directorio para guardar checkpoints
- `save_steps` (int, default=500): Frecuencia de guardado de checkpoints
- `save_total_limit` (int, default=3): Número máximo de checkpoints a mantener
- `logging_steps` (int, default=10): Frecuencia de logging

## Tipos de Modelos Soportados

### Modelos Causales (Causal LM)
Modelos generativos como GPT-2, GPT-Neo, GPT-J, etc.:
```python
trainer = CustomLLMTrainer(
    model_name="gpt2",
    dataset_path="data.json",
    model_type="causal",
    ...
)
```

### Modelos Seq2Seq
Modelos de secuencia a secuencia como T5, BART, etc.:
```python
trainer = CustomLLMTrainer(
    model_name="t5-small",
    dataset_path="data.json",
    model_type="seq2seq",
    ...
)
```

## Ejemplos de Uso Avanzado

### Entrenamiento con GPU y FP16

```python
trainer = CustomLLMTrainer(
    model_name="gpt2",
    dataset_path="data/training.json",
    output_dir="./checkpoints",
    learning_rate=3e-5,
    num_train_epochs=3,
    batch_size=8,
    fp16=True,  # Activar precisión mixta
    gradient_accumulation_steps=4,  # Simular batch_size=32
)
trainer.train()
```

### Entrenamiento con Validación

```python
trainer = CustomLLMTrainer(
    model_name="gpt2",
    dataset_path="data/training.json",
    output_dir="./checkpoints",
    evaluation_strategy="steps",
    eval_steps=100,
    load_best_model_at_end=True,
)
trainer.train()
```

### Generación de Texto Después del Entrenamiento

```python
# Después del entrenamiento
trainer = CustomLLMTrainer(...)
trainer.train()

# Generar respuestas
prompts = [
    "¿Qué es machine learning?",
    "Explica las redes neuronales."
]
responses = trainer.predict(prompts, max_new_tokens=100)
```

### Evaluación del Modelo

```python
# Evaluar en el dataset de validación
eval_results = trainer.evaluate()

# O evaluar en un dataset personalizado
from datasets import Dataset
custom_eval_data = Dataset.from_list([...])
eval_results = trainer.evaluate(eval_dataset=custom_eval_data)
```

## Consideraciones sobre GPU/TPU

### GPU (CUDA)

El módulo detecta automáticamente GPUs CUDA disponibles:

```python
# El dispositivo se detecta automáticamente
trainer = CustomLLMTrainer(...)  # Usará GPU si está disponible
```

**Requisitos:**
- PyTorch con soporte CUDA
- GPU NVIDIA compatible
- Drivers CUDA instalados

### TPU (Tensor Processing Unit)

Para usar TPUs (en Google Colab o Cloud TPU):

```python
# Instalar torch-xla primero
# pip install torch-xla

# El módulo detectará automáticamente TPUs
trainer = CustomLLMTrainer(...)  # Usará TPU si está disponible
```

### Apple Silicon (MPS)

Soporte para chips Apple Silicon (M1/M2/M3):

```python
# Se detecta automáticamente
trainer = CustomLLMTrainer(...)  # Usará MPS si está disponible
```

### CPU (Fallback)

Si no hay GPU/TPU disponible, el entrenamiento se ejecutará en CPU (será más lento).

## Optimización de Memoria

### Para GPUs con poca memoria:

```python
trainer = CustomLLMTrainer(
    model_name="gpt2",
    dataset_path="data.json",
    batch_size=2,  # Reducir batch size
    gradient_accumulation_steps=8,  # Compensar con acumulación
    fp16=True,  # Usar precisión mixta
    max_length=256,  # Reducir longitud máxima
)
```

### Para modelos grandes:

```python
trainer = CustomLLMTrainer(
    model_name="gpt2-large",
    dataset_path="data.json",
    batch_size=1,
    gradient_accumulation_steps=16,
    fp16=True,
    dataloader_num_workers=0,  # Reducir workers
)
```

## Estructura de Checkpoints

Los checkpoints se guardan en la siguiente estructura:

```
checkpoints/
├── checkpoint-500/
│   ├── config.json
│   ├── pytorch_model.bin
│   ├── tokenizer_config.json
│   └── vocab.json
├── checkpoint-1000/
│   └── ...
├── final_checkpoint/
│   └── ...
└── logs/
    └── ...
```

## Métodos Principales

### `train(resume_from_checkpoint=None)`
Entrena el modelo y guarda checkpoints periódicamente y al final.

**Parámetros:**
- `resume_from_checkpoint` (str/bool): Ruta al checkpoint o `True` para reanudar desde el último

### `evaluate(eval_dataset=None)`
Evalúa el modelo en un dataset.

**Parámetros:**
- `eval_dataset` (Dataset, optional): Dataset de evaluación (usa validación si None)

**Retorna:**
- Dict con métricas de evaluación

### `predict(prompts, max_new_tokens=100)`
Genera predicciones desde prompts.

**Parámetros:**
- `prompts` (str/List[str]): Prompt(s) para generar
- `max_new_tokens` (int): Máximo de tokens a generar

**Retorna:**
- str o List[str] con respuestas generadas

### `save_model(output_dir)`
Guarda el modelo y tokenizer en un directorio.

**Parámetros:**
- `output_dir` (str/Path): Directorio donde guardar

## Troubleshooting

### Error: "CUDA out of memory"
- Reduce `batch_size`
- Aumenta `gradient_accumulation_steps`
- Reduce `max_length`
- Activa `fp16=True`

### Error: "Dataset format invalid"
- Verifica que el JSON tenga formato correcto
- Asegúrate de que cada ejemplo tenga "prompt" y "response"
- Verifica que los campos sean strings

### Error: "Model not found"
- Verifica que el `model_name` sea correcto
- Asegúrate de tener conexión a internet para descargar modelos
- O proporciona una ruta local al modelo

### Entrenamiento muy lento
- Usa GPU si está disponible
- Reduce `max_length`
- Aumenta `batch_size` si hay memoria disponible
- Reduce `num_train_epochs` para pruebas

## Mejores Prácticas

1. **Validación de datos**: Siempre valida tu dataset antes de entrenar
2. **Checkpoints frecuentes**: Guarda checkpoints frecuentemente para no perder progreso
3. **Monitoreo**: Revisa los logs regularmente durante el entrenamiento
4. **Experimentos**: Usa diferentes learning rates y documenta los resultados
5. **Evaluación**: Evalúa el modelo en un dataset de validación separado
6. **Backup**: Haz backup de tus checkpoints importantes

## Ejemplo Completo

Ver `example_llm_training.py` para un ejemplo completo de uso.

## Licencia

Este módulo es parte del sistema BUL (Business Universal Language).

## Soporte

Para problemas o preguntas, consulta la documentación de Hugging Face Transformers:
- https://huggingface.co/docs/transformers
- https://huggingface.co/docs/transformers/training

