# Manufacturing AI

Sistema inteligente de manufactura con planificación de producción, control de calidad, optimización de procesos y monitoreo en tiempo real.

## Características

### 1. Planificación de Producción
- Creación y gestión de órdenes de producción
- Programación automática optimizada
- Asignación de recursos
- Priorización inteligente

### 2. Control de Calidad
- Inspección visual con visión por computadora
- Checks dimensionales
- Detección de defectos
- Análisis estadístico de calidad

### 3. Optimización de Procesos
- Optimización de parámetros usando deep learning
- Predicción de mejoras
- Recomendaciones automáticas
- Múltiples objetivos (eficiencia, calidad, costo)

### 4. Monitoreo en Tiempo Real
- Estado de equipos
- Métricas de producción
- Alertas automáticas
- Historial de métricas

## API Endpoints

### Producción
- `POST /api/v1/manufacturing/orders`: Crear orden
- `POST /api/v1/manufacturing/orders/{id}/schedule`: Programar orden
- `POST /api/v1/manufacturing/orders/{id}/start`: Iniciar orden
- `POST /api/v1/manufacturing/orders/{id}/complete`: Completar orden
- `POST /api/v1/manufacturing/optimize-schedule`: Optimizar schedule

### Calidad
- `POST /api/v1/manufacturing/quality/checks`: Crear check
- `POST /api/v1/manufacturing/quality/checks/{id}/dimensional`: Check dimensional

### Optimización
- `POST /api/v1/manufacturing/processes`: Registrar proceso
- `POST /api/v1/manufacturing/processes/optimize`: Optimizar proceso

### Monitoreo
- `POST /api/v1/manufacturing/equipment`: Registrar equipo
- `GET /api/v1/manufacturing/statistics`: Estadísticas

### PlanOps
- `POST /api/v1/manufacturing/demand/forecast`: Predecir demanda
- `POST /api/v1/manufacturing/demand/add-data`: Agregar datos históricos
- `POST /api/v1/manufacturing/maintenance/predict`: Predecir mantenimiento
- `POST /api/v1/manufacturing/maintenance/sensor-data`: Agregar datos de sensores
- `POST /api/v1/manufacturing/capacity/optimize`: Optimizar capacidad
- `POST /api/v1/manufacturing/capacity/allocate`: Asignar recurso

## Ejemplo de Uso

```python
# Crear orden de producción
POST /api/v1/manufacturing/orders
{
    "product_id": "PROD001",
    "quantity": 100,
    "due_date": "2025-11-15T10:00:00",
    "priority": "high"
}

# Optimizar schedule
POST /api/v1/manufacturing/optimize-schedule

# Realizar check de calidad
POST /api/v1/manufacturing/quality/checks
{
    "product_id": "PROD001",
    "check_type": "visual"
}
```

## Integración con Robots

El sistema se integra con el sistema de robots universal para automatización completa de manufactura.

## Modelos de Deep Learning

### Quality Predictor
Modelo CNN+MLP para predecir calidad de productos usando imágenes y características numéricas.

**Arquitecturas disponibles:**
- **Standard**: CNN + MLP básico
- **Advanced**: Con atención multi-head y bloques residuales

### Process Optimizer Transformer
Modelo transformer para optimizar parámetros de procesos de manufactura.

**Arquitecturas disponibles:**
- **Standard**: Transformer básico con encoder
- **Advanced**: Transformer completo con atención avanzada y feed-forward mejorado

## Arquitectura Avanzada

### Componentes Arquitectónicos

El sistema incluye componentes reutilizables para construir modelos avanzados:

#### Attention Layers
- `MultiHeadAttention`: Atención multi-head completa
- `SelfAttention`: Self-attention wrapper
- `CrossAttention`: Cross-attention entre secuencias

#### Residual Blocks
- `ResidualBlock`: Bloque residual con normalización
- `ResidualConnection`: Wrapper para conexiones residuales

#### Normalization
- `LayerNorm`: Layer normalization
- `BatchNorm1d`: Batch normalization
- `GroupNorm`: Group normalization

#### Activations
- `GELU`: Gaussian Error Linear Unit
- `Swish`: Swish activation
- `Mish`: Mish activation

#### Model Builder
- `ModelBuilder`: Builder pattern para construir modelos complejos
- `ArchitectureConfig`: Configuración flexible de arquitecturas

#### Component Factory
- Factory methods para crear componentes consistentes
- Soporte para bloques transformer completos

#### Distributed Training
- Soporte para `DataParallel` (multi-GPU)
- Soporte para `DistributedDataParallel` (entrenamiento distribuido)
- Gestión automática de batch size por GPU

### Uso de Arquitecturas Avanzadas

```python
# Crear modelo de calidad avanzado
POST /api/v1/manufacturing/models/quality/create
{
    "model_id": "quality_advanced_001",
    "image_input_size": 224,
    "num_features": 10,
    "use_advanced": true
}

# Crear modelo de optimización avanzado
POST /api/v1/manufacturing/models/optimizer/create
{
    "model_id": "optimizer_advanced_001",
    "input_dim": 10,
    "output_dim": 10,
    "use_advanced": true
}
```

## Mejoras Avanzadas de Arquitectura

### Advanced Training
- **Mixed Precision Training**: Entrenamiento con FP16 para aceleración
- **Gradient Accumulation**: Permite batch sizes grandes
- **Advanced Trainer**: Entrenador completo con todas las optimizaciones

### Profiling y Optimización
- **Model Profiler**: Análisis de tiempo y memoria
- **Model Optimizer**: Optimización para inferencia (JIT, fusion)

### Checkpointing Avanzado
- **CheckpointManager**: Gestión completa de checkpoints
- Guardado/carga con metadata
- Gestión de mejor modelo
- Limpieza automática

### Embeddings
- **Positional Encoding**: Encoding posicional para transformers
- **Token Embedding**: Embedding de tokens con normalización
- **Feature Embedding**: Embedding de características numéricas

### Ensembling
- **Model Ensemble**: Combinación de múltiples modelos
- **Stacking Ensemble**: Meta-modelo para combinar predicciones
- Múltiples métodos (average, weighted, voting)

### Data Augmentation
- **Advanced Image Augmentation**: Transformaciones avanzadas
- **Feature Augmentation**: Augmentación de características
- **MixUp/CutMix**: Técnicas modernas de augmentación

## Dashboards Gradio

Dashboards interactivos para:
- Monitoreo de producción en tiempo real
- Análisis de calidad con imágenes
- Optimización de procesos

## Experiment Tracking

Soporte para TensorBoard y Weights & Biases para tracking de experimentos de modelos.

## PlanOps (Planning Operations)

### 1. Demand Forecasting
- Predicción de demanda usando LSTM
- Intervalos de confianza
- Integración con planificación

### 2. Predictive Maintenance
- Predicción de fallas usando CNN
- Análisis de sensores en tiempo real
- Recomendaciones de mantenimiento

### 3. Capacity Optimization
- Optimización de capacidad con MLP
- Asignación inteligente de recursos
- Recomendaciones de ajuste

Ver `IMPROVEMENTS.md` y `PLANOPS_IMPROVEMENTS.md` para más detalles.

