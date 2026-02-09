# 🤖 Funcionalidades de IA - Cursor Agent 24/7

## Descripción General

El agente ahora incluye capacidades avanzadas de Inteligencia Artificial para procesar comandos de manera más inteligente, aprender de patrones y mejorar continuamente.

## 🧠 Componentes de IA

### 1. AI Processor (`core/ai_processor.py`)

Procesa comandos usando LLMs para entender intención y generar código.

#### Características:
- **Detección de Intención**: Identifica la intención del comando (ejecutar, crear, actualizar, eliminar, analizar, buscar, consultar)
- **Extracción de Código**: Extrae código de comandos en lenguaje natural
- **Generación de Código**: Genera código a partir de descripciones
- **Resumen de Resultados**: Resume resultados largos automáticamente
- **Soporte para OpenAI y Modelos Locales**: Usa OpenAI API o modelos locales de Transformers

#### Uso:

```python
from cursor_agent_24_7.core.ai_processor import AIProcessor

# Inicializar procesador
processor = AIProcessor(use_local=True)  # o use_openai=True
await processor.initialize()

# Procesar comando
processed = await processor.process_command("create a function to calculate fibonacci")
print(f"Intent: {processed.intent}")
print(f"Confidence: {processed.confidence}")
print(f"Extracted Code: {processed.extracted_code}")

# Generar código
code = await processor.generate_code("function to sort a list", language="python")
print(code)
```

### 2. Embedding Store (`core/embeddings.py`)

Sistema de embeddings para búsqueda semántica de comandos y tareas.

#### Características:
- **Búsqueda Semántica**: Encuentra comandos similares usando embeddings
- **Almacenamiento Persistente**: Guarda embeddings en disco
- **Modelos Locales**: Usa Sentence Transformers o Transformers de Hugging Face
- **Búsqueda por Similitud**: Encuentra comandos relacionados por significado

#### Uso:

```python
from cursor_agent_24_7.core.embeddings import EmbeddingStore

# Inicializar store
store = EmbeddingStore()
await store.initialize()

# Agregar embedding
await store.add("task_1", "calculate fibonacci sequence", metadata={"type": "math"})

# Buscar similares
results = await store.search("compute fibonacci numbers", top_k=5)
for key, similarity, metadata in results:
    print(f"{key}: {similarity:.2f}")
```

### 3. Pattern Learner (`core/pattern_learner.py`)

Sistema de aprendizaje de patrones para mejorar la ejecución.

#### Características:
- **Aprendizaje de Patrones**: Aprende qué comandos tienen éxito
- **Predicción de Éxito**: Predice probabilidad de éxito de comandos
- **Sugerencias de Mejora**: Sugiere mejoras basadas en historial
- **Estadísticas**: Proporciona estadísticas de aprendizaje

#### Uso:

```python
from cursor_agent_24_7.core.pattern_learner import PatternLearner

# Inicializar learner
learner = PatternLearner()
await learner.load()

# Registrar comando
await learner.record_command("calculate sum", success=True, execution_time=0.5)

# Predecir éxito
prob, info = await learner.predict_success("calculate sum of numbers")
print(f"Success probability: {prob:.2f}")

# Obtener sugerencias
suggestions = await learner.suggest_improvements("calculate sum")
print(suggestions)

# Estadísticas
stats = await learner.get_statistics()
print(stats)
```

## 🔌 Integración en el Agente

Los componentes de IA se integran automáticamente en el agente:

1. **Procesamiento Automático**: Todos los comandos se procesan con IA si está disponible
2. **Aprendizaje Continuo**: El agente aprende de cada ejecución
3. **Búsqueda Semántica**: Los comandos se indexan para búsqueda semántica
4. **Resumen Automático**: Resultados largos se resumen automáticamente

## 📡 API Endpoints

### Procesar Comando con IA

```bash
POST /api/ai/process
Content-Type: application/json

{
  "command": "create a function to calculate fibonacci"
}
```

Respuesta:
```json
{
  "original": "create a function to calculate fibonacci",
  "intent": "create",
  "confidence": 0.85,
  "extracted_code": "def fibonacci(n):\n    ...",
  "parameters": {},
  "suggested_actions": []
}
```

### Generar Código

```bash
POST /api/ai/generate
Content-Type: application/json

{
  "description": "function to sort a list",
  "language": "python"
}
```

### Resumir Resultado

```bash
POST /api/ai/summarize
Content-Type: application/json

{
  "result": "very long result...",
  "max_length": 200
}
```

### Buscar en Embeddings

```bash
POST /api/embeddings/search
Content-Type: application/json

{
  "query": "calculate fibonacci",
  "top_k": 5,
  "threshold": 0.5
}
```

### Estadísticas de Patrones

```bash
GET /api/patterns/stats
```

### Predecir Éxito

```bash
POST /api/patterns/predict
Content-Type: application/json

{
  "command": "calculate sum"
}
```

## ⚙️ Configuración

### Modelos Locales

El agente usa modelos locales por defecto:
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2`
- **Transformers**: Modelos de Hugging Face

### OpenAI (Opcional)

Para usar OpenAI:

```python
from cursor_agent_24_7.core.ai_processor import AIProcessor

processor = AIProcessor(
    use_openai=True,
    api_key="your-api-key",
    model_name="gpt-3.5-turbo"
)
```

## 📊 Beneficios

1. **Comprensión Mejorada**: Entiende comandos en lenguaje natural
2. **Aprendizaje Continuo**: Mejora con cada ejecución
3. **Búsqueda Inteligente**: Encuentra comandos relacionados
4. **Generación de Código**: Genera código automáticamente
5. **Optimización**: Aprende qué comandos funcionan mejor

## 🔧 Requisitos

### Instalación Básica

```bash
pip install transformers sentence-transformers torch
```

### Instalación Completa

```bash
pip install -r requirements.txt
```

### Modelos

Los modelos se descargan automáticamente la primera vez que se usan.

## 📝 Notas

- Los modelos locales requieren más memoria pero no necesitan API keys
- OpenAI requiere API key pero es más potente
- El aprendizaje de patrones mejora con el tiempo
- Los embeddings se guardan en `./data/embeddings.json`
- Los patrones se guardan en `./data/patterns.json`

## 🚀 Próximos Pasos

1. **Fine-tuning**: Fine-tune modelos con datos específicos
2. **Modelos Personalizados**: Usar modelos entrenados específicamente
3. **Integración Avanzada**: Integrar con más servicios de IA
4. **Optimización**: Optimizar modelos para mejor rendimiento


