# 🚀 Advanced AI Features - Cursor Agent 24/7

## Descripción General

El agente ahora incluye capacidades avanzadas de Deep Learning, Transformers y LLMs con integración profesional usando PyTorch, Transformers, y Gradio.

## 🧠 Componentes Avanzados

### 1. LLM Pipeline (`core/llm_pipeline.py`)

Pipeline profesional para procesamiento con modelos de transformers.

#### Características:
- **Generación de Texto**: Genera texto usando modelos causales (GPT-2, etc.)
- **Codificación a Embeddings**: Convierte texto a embeddings usando hidden states
- **Clasificación**: Clasifica texto usando el modelo
- **Completar Código**: Completa código automáticamente
- **Explicar Código**: Explica código en lenguaje natural
- **Corregir Código**: Corrige código con errores
- **Resumir**: Resume texto automáticamente
- **Soporte GPU/CPU**: Detección automática de dispositivo
- **Mixed Precision**: Soporte para float16 en GPU

#### Uso:

```python
from cursor_agent_24_7.core.llm_pipeline import LLMPipeline, LLMConfig

# Configurar pipeline
config = LLMConfig(
    model_name="gpt2",  # o cualquier modelo causal
    device="auto",  # auto-detecta GPU/CPU
    max_length=512,
    temperature=0.7
)

# Crear pipeline
pipeline = LLMPipeline(config)
await pipeline.initialize()

# Generar texto
text = pipeline.generate("Write a function to calculate", max_new_tokens=100)
print(text)

# Completar código
code = pipeline.complete_code("def fibonacci(n):\n    if n <= 1:\n        return n")
print(code)

# Explicar código
explanation = pipeline.explain_code("def fib(n): return n if n <= 1 else fib(n-1) + fib(n-2)")
print(explanation)

# Corregir código
fixed = pipeline.fix_code("def sum(a, b):\n    retun a + b", error="NameError: name 'retun' is not defined")
print(fixed)
```

### 2. Fine-Tuner (`core/llm_pipeline.py`)

Clase para fine-tuning de modelos usando técnicas modernas.

#### Características:
- **Fine-tuning Estándar**: Fine-tuning completo del modelo
- **LoRA Support**: Preparado para LoRA (Low-Rank Adaptation)
- **Gradient Clipping**: Previene gradientes explosivos
- **Learning Rate Scheduling**: Scheduler con warmup
- **Checkpointing**: Guarda checkpoints durante entrenamiento

#### Uso:

```python
from cursor_agent_24_7.core.llm_pipeline import LLMPipeline, FineTuner
import torch
from torch.utils.data import DataLoader

# Cargar modelo
pipeline = LLMPipeline(LLMConfig(model_name="gpt2"))
await pipeline.initialize()

# Crear fine-tuner
fine_tuner = FineTuner(pipeline.model, pipeline.tokenizer)

# Preparar entrenamiento
fine_tuner.prepare_training(
    learning_rate=5e-5,
    weight_decay=0.01,
    warmup_steps=100
)

# Entrenar
for batch in dataloader:
    loss_dict = fine_tuner.train_step(batch, num_training_steps=1000)
    print(f"Loss: {loss_dict['loss']:.4f}")

# Guardar checkpoint
fine_tuner.save_checkpoint("./checkpoints/finetuned_model")
```

### 3. Gradio Interface (`core/gradio_interface.py`)

Interfaz web interactiva usando Gradio.

#### Características:
- **Control del Agente**: Iniciar, detener, pausar, reanudar
- **Ejecución de Comandos**: Ejecutar comandos directamente
- **Procesamiento con IA**: Procesar texto con diferentes modos de IA
- **Búsqueda Semántica**: Buscar comandos similares
- **Estadísticas**: Ver estadísticas de patrones y métricas
- **Auto-refresh**: Actualización automática del estado

#### Uso:

```python
from cursor_agent_24_7.core.gradio_interface import GradioInterface
from cursor_agent_24_7.core.agent import CursorAgent, AgentConfig

# Crear agente
agent = CursorAgent(AgentConfig())
await agent.start()

# Crear interfaz
gradio_interface = GradioInterface(agent)
interface = gradio_interface.create_interface()

# Lanzar
interface.launch(share=False, server_port=7860)
```

O usar el script:

```bash
python scripts/launch_gradio.py
```

Con link público:

```bash
python scripts/launch_gradio.py --share
```

## 🔧 Integración con el Agente

### Procesamiento Automático

El agente usa automáticamente el LLM Pipeline cuando está disponible:

1. **Generación de Código**: Usa el pipeline para generar código
2. **Resumen**: Usa el pipeline para resumir resultados
3. **Procesamiento de Comandos**: Mejora el procesamiento de comandos

### Configuración

```python
from cursor_agent_24_7.core.agent import CursorAgent, AgentConfig
from cursor_agent_24_7.core.ai_processor import AIProcessor
from cursor_agent_24_7.core.llm_pipeline import LLMConfig

# Configurar LLM
llm_config = LLMConfig(
    model_name="gpt2",  # o "microsoft/DialoGPT-medium", etc.
    device="auto"
)

# El AI Processor cargará automáticamente el LLM Pipeline
agent = CursorAgent(AgentConfig())
await agent.start()
```

## 📊 Modelos Soportados

### Modelos Causales (Generación)
- `gpt2` - Modelo pequeño y rápido
- `gpt2-medium` - Modelo mediano
- `gpt2-large` - Modelo grande
- `microsoft/DialoGPT-medium` - Para diálogos
- Cualquier modelo causal de Hugging Face

### Modelos de Embeddings
- `sentence-transformers/all-MiniLM-L6-v2` - Embeddings semánticos
- `sentence-transformers/all-mpnet-base-v2` - Embeddings de alta calidad

## 🚀 Mejores Prácticas

### 1. Selección de Modelo

Para desarrollo:
```python
config = LLMConfig(model_name="gpt2")  # Pequeño y rápido
```

Para producción:
```python
config = LLMConfig(model_name="gpt2-medium")  # Mejor calidad
```

### 2. Optimización de GPU

```python
config = LLMConfig(
    model_name="gpt2",
    device="cuda"  # Forzar GPU
)
```

### 3. Fine-tuning con LoRA

```python
from peft import LoraConfig, get_peft_model

# Configurar LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1
)

# Aplicar LoRA
model = get_peft_model(pipeline.model, lora_config)
```

### 4. Mixed Precision Training

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    outputs = model(**batch)
    loss = outputs.loss

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

## 📡 API Endpoints

Los endpoints existentes ahora usan el LLM Pipeline cuando está disponible:

- `POST /api/ai/generate` - Usa LLM Pipeline si está disponible
- `POST /api/ai/summarize` - Usa LLM Pipeline para resumir
- `POST /api/ai/process` - Procesamiento mejorado con LLM

## 🔍 Ejemplos Avanzados

### Ejemplo 1: Pipeline Personalizado

```python
from cursor_agent_24_7.core.llm_pipeline import LLMPipeline, LLMConfig

config = LLMConfig(
    model_name="gpt2",
    temperature=0.3,  # Más determinístico
    top_p=0.9,
    top_k=50
)

pipeline = LLMPipeline(config)
await pipeline.initialize()

# Generar con parámetros personalizados
text = pipeline.generate(
    "Write Python code to",
    max_new_tokens=200,
    temperature=0.2,  # Override config
    do_sample=True
)
```

### Ejemplo 2: Fine-tuning Completo

```python
from cursor_agent_24_7.core.llm_pipeline import LLMPipeline, FineTuner
from transformers import DataCollatorForLanguageModeling
from torch.utils.data import Dataset

class CodeDataset(Dataset):
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
            padding="max_length"
        )
        return {k: torch.tensor(v) for k, v in encoding.items()}

# Preparar datos
dataset = CodeDataset(code_samples, pipeline.tokenizer)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Fine-tuning
fine_tuner = FineTuner(pipeline.model, pipeline.tokenizer)
fine_tuner.prepare_training(learning_rate=5e-5)

for epoch in range(3):
    for batch in dataloader:
        loss = fine_tuner.train_step(batch, num_training_steps=len(dataloader) * 3)
        print(f"Epoch {epoch}, Loss: {loss['loss']:.4f}")

fine_tuner.save_checkpoint("./checkpoints/code_model")
```

## 📝 Notas Importantes

1. **Memoria**: Modelos grandes requieren mucha memoria GPU
2. **Primera Carga**: Los modelos se descargan la primera vez (puede tardar)
3. **GPU Recomendado**: Para mejor rendimiento, usar GPU
4. **Quantización**: Usar `bitsandbytes` para reducir memoria
5. **LoRA**: Usar LoRA para fine-tuning eficiente

## 🆘 Troubleshooting

### Error: CUDA out of memory
- Usar modelo más pequeño
- Reducir `max_length`
- Usar `bitsandbytes` para quantización

### Error: Model not found
- Verificar conexión a internet
- Verificar nombre del modelo en Hugging Face

### Rendimiento lento
- Usar GPU si está disponible
- Reducir `max_new_tokens`
- Usar modelo más pequeño

## 🚀 Próximos Pasos

1. **Fine-tuning Específico**: Fine-tune con datos de código
2. **Modelos Especializados**: Usar modelos específicos para código
3. **Optimización**: Quantización y optimización de modelos
4. **Integración Avanzada**: Más integraciones con servicios de IA



