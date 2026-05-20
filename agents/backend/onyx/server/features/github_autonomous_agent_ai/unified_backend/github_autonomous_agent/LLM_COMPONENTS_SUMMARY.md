# 📦 Resumen de Componentes LLM

Este documento resume todos los componentes modulares del servicio LLM.

## 🧩 Componentes Disponibles

### 1. **Prompt Templates** (`prompt_templates.py`)
**Propósito**: Gestión de templates de prompts reutilizables

**Características**:
- Templates con variables usando `string.Template`
- Registry centralizado
- Templates predefinidos (code_analysis, documentation, refactoring, etc.)
- Configuración por template

**Uso**:
```python
from core.services.llm import get_template_registry
registry = get_template_registry()
template = registry.get("code_analysis")
prompts = template.render(code=code, language="python", analysis_type="security")
```

### 2. **Token Manager** (`token_manager.py`)
**Propósito**: Gestión y optimización de tokens

**Características**:
- Estimación precisa por tipo de contenido
- Validación de límites por modelo
- Optimización automática de prompts
- Tracking de uso

**Uso**:
```python
from core.services.llm import get_token_manager
token_mgr = get_token_manager()
is_valid, error, info = token_mgr.validate_request(model, messages, max_tokens)
```

### 3. **Batch Processor** (`batch_processor.py`)
**Propósito**: Procesamiento eficiente por lotes

**Características**:
- Agrupación inteligente de requests
- Priorización
- Procesamiento paralelo
- Rate limiting integrado

**Uso**:
```python
from core.services.llm import BatchProcessor
processor = BatchProcessor(batch_size=10)
await processor.add_to_batch("analysis", item_id, data, priority=1)
```

### 4. **Response Validator** (`response_validator.py`)
**Propósito**: Validación y evaluación de respuestas

**Características**:
- Validación de formato (JSON, code, markdown)
- Detección de contenido problemático
- Niveles de validación (none, basic, strict)
- Score de calidad (0.0-1.0)

**Uso**:
```python
from core.services.llm import get_validator, ValidationLevel
validator = get_validator(ValidationLevel.STRICT)
result = validator.validate(response, expected_format="markdown")
```

### 5. **Model Registry** (`model_registry.py`)
**Propósito**: Gestión centralizada de modelos

**Características**:
- Configuraciones por modelo
- Recomendaciones por caso de uso
- Estimación de costos
- Filtrado y búsqueda

**Uso**:
```python
from core.services.llm import get_model_registry
registry = get_model_registry()
config = registry.get("openai/gpt-4o")
cost = registry.estimate_cost("openai/gpt-4o", 1000, 500)
```

### 6. **Experiment Tracker** (`experiment_tracker.py`)
**Propósito**: Tracking de experimentos

**Características**:
- Registro de configuraciones
- Tracking de métricas
- Comparación de experimentos
- Persistencia en JSON

**Uso**:
```python
from core.services.llm import get_experiment_tracker, ExperimentConfig
tracker = get_experiment_tracker()
exp_id = tracker.start_experiment(ExperimentConfig(name="test", model="gpt-4o"))
tracker.log_result(exp_id, response, metrics, duration_ms, tokens_used)
```

### 7. **Config Manager** (`config_manager.py`)
**Propósito**: Gestión de configuraciones YAML/JSON

**Características**:
- Carga/guardado de configuraciones
- Soporte YAML y JSON
- Configuraciones por defecto

**Uso**:
```python
from core.services.llm import get_config_manager
config_mgr = get_config_manager()
models_config = config_mgr.load_config("models")
```

### 8. **Evaluation Metrics** (`evaluation_metrics.py`)
**Propósito**: Cálculo de métricas de evaluación

**Características**:
- Métricas de longitud, legibilidad, estructura
- Métricas de código (calidad, completitud)
- Similitud (Jaccard, ROUGE-L)
- Score general

**Uso**:
```python
from core.services.llm import get_evaluation_metrics
metrics = get_evaluation_metrics()
results = metrics.calculate_all_metrics(response, reference=ref)
```

### 9. **Data Pipeline** (`data_pipeline.py`) ⭐ NUEVO
**Propósito**: Pipeline de procesamiento de datos

**Características**:
- Pipeline funcional de transformaciones
- Etapas de procesamiento (raw, cleaned, normalized, etc.)
- Pipelines predefinidos (code, text)
- Logging de transformaciones

**Uso**:
```python
from core.services.llm import DataPipeline
pipeline = DataPipeline.create_code_pipeline()
processed = pipeline.process(code)
```

### 10. **Checkpoint Manager** (`checkpoint_manager.py`) ⭐ NUEVO
**Propósito**: Gestión de checkpoints y estados

**Características**:
- Guardado/carga de estados
- Soporte JSON y Pickle
- Listado y búsqueda de checkpoints
- Verificación con hash

**Uso**:
```python
from core.services.llm import get_checkpoint_manager
checkpoint_mgr = get_checkpoint_manager()
path = checkpoint_mgr.save_checkpoint("state_v1", state_dict)
state = checkpoint_mgr.load_checkpoint(path)
```

### 11. **Performance Profiler** (`performance_profiler.py`) ⭐ NUEVO
**Propósito**: Profiling y análisis de performance

**Características**:
- Profiling de funciones
- Métricas de tiempo, tokens, costos
- Decoradores para profiling automático
- Reportes exportables

**Uso**:
```python
from core.services.llm import get_profiler
profiler = get_profiler()
profiler.start("my_function")
# ... código ...
profiler.stop("my_function", tokens=100, cost=0.01)
summary = profiler.get_summary()
```

### 12. **Model Selector** (`model_selector.py`) ⭐ NUEVO
**Propósito**: Selección inteligente de modelos

**Características**:
- Estrategias de selección (best_quality, cost_efficient, fastest, balanced)
- Filtrado por criterios
- Selección para ejecución paralela
- Recomendaciones automáticas

**Uso**:
```python
from core.services.llm import get_model_selector, SelectionStrategy, SelectionCriteria
selector = get_model_selector()
criteria = SelectionCriteria(
    use_case="code_analysis",
    strategy=SelectionStrategy.BALANCED,
    max_cost=0.10
)
model = selector.select_model(criteria)
```

### 13. **Cost Optimizer** (`cost_optimizer.py`) ⭐ NUEVO
**Propósito**: Optimización de costos

**Características**:
- Tracking de costos (diario, mensual)
- Presupuestos y límites
- Alertas de presupuesto
- Recomendaciones de optimización

**Uso**:
```python
from core.services.llm import get_cost_optimizer, CostBudget
optimizer = get_cost_optimizer()
cost, within_budget = optimizer.record_cost("gpt-4o", 1000, 500, "analysis")
stats = optimizer.get_stats()
recommendations = optimizer.get_recommendations()
```

## 🔄 Flujo Completo con Todos los Componentes

```python
from core.services.llm import (
    get_template_registry,
    get_token_manager,
    get_model_selector,
    get_cost_optimizer,
    get_profiler,
    get_data_pipeline,
    get_experiment_tracker,
    get_evaluation_metrics,
    SelectionStrategy,
    SelectionCriteria,
    ExperimentConfig
)

# 1. Procesar datos
pipeline = get_data_pipeline()
code_pipeline = DataPipeline.create_code_pipeline()
processed_code = code_pipeline.process(code).processed

# 2. Seleccionar modelo óptimo
selector = get_model_selector()
criteria = SelectionCriteria(
    use_case="code_analysis",
    strategy=SelectionStrategy.BALANCED
)
model = selector.select_model(criteria)

# 3. Preparar prompt con template
template = get_template_registry().get("code_analysis")
prompts = template.render(code=processed_code, language="python", analysis_type="security")

# 4. Validar tokens
token_mgr = get_token_manager()
is_valid, error, token_info = token_mgr.validate_request(model, [
    {"role": "system", "content": prompts["system_prompt"]},
    {"role": "user", "content": prompts["user_prompt"]}
], max_tokens=2000)

# 5. Iniciar experimento
tracker = get_experiment_tracker()
exp_id = tracker.start_experiment(ExperimentConfig(
    name="security_analysis",
    model=model,
    temperature=0.3
))

# 6. Profiling
profiler = get_profiler()
profiler.start("llm_generation")

# 7. Generar respuesta
response = await llm_service.generate(
    prompt=prompts["user_prompt"],
    system_prompt=prompts["system_prompt"],
    model=model
)

# 8. Detener profiling
duration = profiler.stop("llm_generation", 
    tokens=response.usage.get("total_tokens", 0) if response.usage else 0,
    cost=0.01
)

# 9. Evaluar respuesta
metrics_calc = get_evaluation_metrics()
evaluation = metrics_calc.calculate_all_metrics(
    response.content,
    expected_keywords=["vulnerability", "security"]
)

# 10. Registrar experimento
tracker.log_result(
    exp_id,
    response.content,
    {**evaluation, "duration_ms": duration * 1000},
    duration * 1000,
    response.usage.get("total_tokens", 0) if response.usage else 0,
    cost=0.01
)

# 11. Registrar costo
cost_optimizer = get_cost_optimizer()
cost, within_budget = cost_optimizer.record_cost(
    model,
    response.usage.get("prompt_tokens", 0) if response.usage else 0,
    response.usage.get("completion_tokens", 0) if response.usage else 0,
    use_case="code_analysis"
)
```

## 📊 Métricas y Estadísticas Disponibles

### Performance
```python
profiler = get_profiler()
summary = profiler.get_summary()
# Incluye: total_time, total_calls, tokens_per_second, slowest_functions
```

### Costos
```python
optimizer = get_cost_optimizer()
stats = optimizer.get_stats()
# Incluye: total_cost, daily_cost, cost_by_model, recommendations
```

### Tokens
```python
token_mgr = get_token_manager()
stats = token_mgr.get_usage_stats("openai/gpt-4o")
# Incluye: total_requests, average_tokens, total_tokens
```

### Experimentos
```python
tracker = get_experiment_tracker()
comparison = tracker.compare_experiments([exp_id1, exp_id2], metric="duration_ms")
# Compara experimentos y encuentra el mejor
```

## 🎯 Casos de Uso Comunes

### Análisis de Código con Optimización
```python
# Seleccionar modelo económico
model = llm_service.select_optimal_model(
    "code_analysis",
    strategy=SelectionStrategy.COST_EFFICIENT
)

# Procesar código
processed = llm_service.process_code_with_pipeline(code)

# Analizar
response = await llm_service.analyze_code(
    processed,
    language="python",
    analysis_type="security"
)
```

### Generación con Tracking
```python
# Iniciar experimento
exp_id = tracker.start_experiment(ExperimentConfig(
    name="doc_generation",
    model="claude-3.5-sonnet"
))

# Generar
response = await llm_service.generate_documentation(code, language="python")

# Registrar
tracker.log_result(exp_id, response.content, {}, response.latency_ms, tokens)
```

### Optimización de Costos
```python
# Obtener recomendaciones
optimizer = get_cost_optimizer()
recommendations = optimizer.get_recommendations()

# Aplicar recomendaciones
for rec in recommendations:
    if rec["type"] == "model_replacement":
        logger.info(f"Recomendación: {rec['message']}")
```

## 🔧 Configuración

Todos los componentes pueden configurarse vía YAML:

- `config/llm/models.yaml` - Modelos y configuraciones
- `config/llm/prompts.yaml` - Templates y prompts
- `config/llm/optimization.yaml` - Optimizaciones

## 📈 Beneficios

1. **Modularidad**: Componentes independientes y reutilizables
2. **Observabilidad**: Tracking completo de performance y costos
3. **Optimización**: Selección automática de modelos y optimización de costos
4. **Calidad**: Validación y evaluación automática
5. **Escalabilidad**: Batching y procesamiento paralelo
6. **Mantenibilidad**: Código organizado y bien estructurado

---

Esta arquitectura proporciona una base sólida y profesional para el servicio LLM, siguiendo las mejores prácticas de frameworks de deep learning.



