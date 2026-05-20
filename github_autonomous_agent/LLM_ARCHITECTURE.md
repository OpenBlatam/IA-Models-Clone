# 🏗️ Arquitectura del Servicio LLM

Este documento describe la arquitectura modular del servicio LLM, siguiendo principios de deep learning y desarrollo de LLMs.

## 📐 Principios Arquitectónicos

### 1. Modularidad y Separación de Responsabilidades
- Cada componente tiene una responsabilidad única y bien definida
- Componentes independientes y reutilizables
- Interfaces claras entre componentes

### 2. Configuración Centralizada
- Configuraciones en YAML/JSON
- Gestión centralizada de modelos y hyperparameters
- Versionado de configuraciones

### 3. Experiment Tracking
- Tracking completo de experimentos
- Métricas y comparaciones
- Persistencia de resultados

### 4. Evaluación y Validación
- Métricas de calidad automáticas
- Validación de respuestas
- Evaluación comparativa

## 🧩 Componentes Modulares

### 1. Prompt Templates (`prompt_templates.py`)

**Responsabilidad**: Gestión de templates de prompts reutilizables.

**Características**:
- Templates con variables usando `string.Template`
- Registry centralizado de templates
- Configuración por template (temperature, max_tokens)
- Templates predefinidos para casos comunes

**Uso**:
```python
from core.services.llm import get_template_registry

registry = get_template_registry()
template = registry.get("code_analysis")
prompts = template.render(
    code=code,
    language="python",
    analysis_type="security"
)
```

### 2. Token Manager (`token_manager.py`)

**Responsabilidad**: Gestión y optimización de tokens.

**Características**:
- Estimación precisa por tipo de contenido (text, code, mixed)
- Validación de límites por modelo
- Optimización automática de prompts
- Tracking de uso de tokens

**Uso**:
```python
from core.services.llm import get_token_manager

token_mgr = get_token_manager()
estimated = token_mgr.estimate_tokens(code, content_type="code")
is_valid, error, info = token_mgr.validate_request(model, messages, max_tokens)
```

### 3. Batch Processor (`batch_processor.py`)

**Responsabilidad**: Procesamiento eficiente de múltiples requests.

**Características**:
- Agrupación inteligente de requests
- Priorización de items
- Procesamiento paralelo con semáforos
- Rate limiting integrado

**Uso**:
```python
from core.services.llm import BatchProcessor

processor = BatchProcessor(batch_size=10, max_wait_time=0.5)
await processor.add_to_batch("analysis", item_id, code_data, priority=1)
results = await processor.process_batch("analysis", process_func)
```

### 4. Response Validator (`response_validator.py`)

**Responsabilidad**: Validación y evaluación de respuestas.

**Características**:
- Validación de formato (JSON, code, markdown)
- Detección de contenido problemático
- Niveles de validación (none, basic, strict)
- Evaluación de calidad (score 0.0-1.0)

**Uso**:
```python
from core.services.llm import get_validator, ValidationLevel

validator = get_validator(ValidationLevel.STRICT)
result = validator.validate(
    response,
    expected_format="markdown",
    min_length=100
)
```

### 5. Model Registry (`model_registry.py`)

**Responsabilidad**: Gestión centralizada de modelos y configuraciones.

**Características**:
- Configuraciones por modelo (límites, costos, capacidades)
- Recomendaciones por caso de uso
- Estimación de costos
- Filtrado y búsqueda de modelos

**Uso**:
```python
from core.services.llm import get_model_registry

registry = get_model_registry()
config = registry.get("openai/gpt-4o")
recommended = registry.get_recommended_model("code_analysis", budget_conscious=True)
cost = registry.estimate_cost("openai/gpt-4o", input_tokens=1000, output_tokens=500)
```

### 6. Experiment Tracker (`experiment_tracker.py`)

**Responsabilidad**: Tracking de experimentos y configuraciones.

**Características**:
- Registro de configuraciones de experimentos
- Tracking de métricas y resultados
- Comparación de experimentos
- Persistencia en JSON

**Uso**:
```python
from core.services.llm import get_experiment_tracker, ExperimentConfig

tracker = get_experiment_tracker()
config = ExperimentConfig(
    name="code_analysis_v1",
    model="openai/gpt-4o",
    temperature=0.3
)
exp_id = tracker.start_experiment(config)
tracker.log_result(exp_id, response, metrics, duration_ms, tokens_used)
comparison = tracker.compare_experiments([exp_id1, exp_id2], metric="duration_ms")
```

### 7. Config Manager (`config_manager.py`)

**Responsabilidad**: Gestión de configuraciones YAML/JSON.

**Características**:
- Carga/guardado de configuraciones
- Soporte para YAML y JSON
- Configuraciones por defecto
- Gestión centralizada

**Uso**:
```python
from core.services.llm import get_config_manager

config_mgr = get_config_manager()
models_config = config_mgr.load_config("models")
config_mgr.save_config("custom_config", my_config, format="yaml")
```

### 8. Evaluation Metrics (`evaluation_metrics.py`)

**Responsabilidad**: Cálculo de métricas de evaluación.

**Características**:
- Métricas de longitud, legibilidad, estructura
- Métricas de código (calidad, completitud)
- Métricas de similitud (Jaccard, ROUGE-L)
- Score general agregado

**Uso**:
```python
from core.services.llm import get_evaluation_metrics

metrics_calc = get_evaluation_metrics()
all_metrics = metrics_calc.calculate_all_metrics(
    response,
    reference=reference_text,
    expected_keywords=["error", "fix", "improvement"]
)
overall_score = metrics_calc.get_overall_score(all_metrics)
```

## 🔄 Flujo de Procesamiento

```
┌─────────────────┐
│  User Request   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Config Manager  │ ◄─── Carga configuración
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│Template Registry│ ◄─── Selecciona template
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Token Manager  │ ◄─── Valida y optimiza tokens
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Model Registry │ ◄─── Selecciona modelo
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   LLM Service   │ ◄─── Procesa request
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│Response Validator│ ◄─── Valida respuesta
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│Evaluation Metrics│ ◄─── Calcula métricas
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│Experiment Tracker│ ◄─── Registra experimento
└─────────────────┘
```

## 📊 Ejemplo Completo

```python
from core.services.llm import (
    get_template_registry,
    get_token_manager,
    get_model_registry,
    get_experiment_tracker,
    get_evaluation_metrics,
    ExperimentConfig
)

# 1. Configurar experimento
tracker = get_experiment_tracker()
config = ExperimentConfig(
    name="security_analysis",
    model="anthropic/claude-3.5-sonnet",
    temperature=0.3,
    tags=["security", "analysis"]
)
exp_id = tracker.start_experiment(config)

# 2. Preparar prompt con template
template_registry = get_template_registry()
template = template_registry.get("code_analysis")
prompts = template.render(
    code=code,
    language="python",
    analysis_type="security"
)

# 3. Validar tokens
token_mgr = get_token_manager()
is_valid, error, token_info = token_mgr.validate_request(
    config.model,
    [{"role": "system", "content": prompts["system_prompt"]},
     {"role": "user", "content": prompts["user_prompt"]}],
    max_tokens=2000
)

if not is_valid:
    raise ValueError(error)

# 4. Obtener modelo recomendado
model_registry = get_model_registry()
model_config = model_registry.get(config.model)

# 5. Generar respuesta
response = await llm_service.generate(
    prompt=prompts["user_prompt"],
    system_prompt=prompts["system_prompt"],
    model=config.model,
    temperature=config.temperature
)

# 6. Evaluar respuesta
metrics_calc = get_evaluation_metrics()
evaluation = metrics_calc.calculate_all_metrics(
    response.content,
    expected_keywords=["vulnerability", "security", "risk"]
)

# 7. Registrar resultado
tracker.log_result(
    exp_id,
    response.content,
    {
        "model": config.model,
        "temperature": config.temperature,
        **evaluation
    },
    response.latency_ms or 0,
    response.usage.get("total_tokens", 0) if response.usage else 0,
    cost=model_registry.estimate_cost(
        config.model,
        response.usage.get("prompt_tokens", 0) if response.usage else 0,
        response.usage.get("completion_tokens", 0) if response.usage else 0
    )
)
```

## 🎯 Beneficios de la Arquitectura

1. **Modularidad**: Componentes independientes y reutilizables
2. **Mantenibilidad**: Código organizado y fácil de extender
3. **Testabilidad**: Componentes aislados fáciles de testear
4. **Escalabilidad**: Fácil agregar nuevos componentes
5. **Observabilidad**: Tracking completo de experimentos y métricas
6. **Eficiencia**: Optimización automática de tokens y batching
7. **Calidad**: Validación y evaluación automática

## 📁 Estructura de Archivos

```
core/services/llm/
├── __init__.py              # Exports centralizados
├── prompt_templates.py      # Sistema de templates
├── token_manager.py         # Gestión de tokens
├── batch_processor.py       # Procesamiento por lotes
├── response_validator.py    # Validación de respuestas
├── model_registry.py        # Registry de modelos
├── experiment_tracker.py    # Tracking de experimentos
├── config_manager.py        # Gestión de configuraciones
└── evaluation_metrics.py    # Métricas de evaluación
```

## 🔧 Configuración

Las configuraciones se almacenan en `config/llm/`:

- `models.yaml` - Configuraciones de modelos
- `prompts.yaml` - Configuraciones de prompts
- `optimization.yaml` - Configuraciones de optimización

## 📈 Experiment Tracking

Los experimentos se almacenan en `storage/experiments/`:

- Cada experimento se guarda como JSON
- Incluye configuración, respuesta, métricas y costos
- Permite comparación y análisis posterior

---

Esta arquitectura sigue las mejores prácticas de frameworks de deep learning como PyTorch y Transformers, proporcionando una base sólida y extensible para el servicio LLM.



