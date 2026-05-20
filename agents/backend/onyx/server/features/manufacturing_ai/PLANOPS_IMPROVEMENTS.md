# PlanOps Improvements: Advanced Planning Operations

## Resumen

Se han agregado sistemas avanzados de planificación de operaciones (PlanOps) usando deep learning para optimizar manufactura.

## Nuevos Sistemas Implementados

### 1. Demand Forecasting (`core/demand_forecasting.py`)

**Modelo:**
- LSTM para modelar patrones temporales
- Predicción de demanda futura
- Intervalos de confianza

**Características:**
- Predicción a múltiples horizontes temporales
- Análisis de factores que afectan demanda
- Integración con planificación de producción

**Ejemplo:**
```python
from manufacturing_ai.core.demand_forecasting import get_demand_forecasting_system

system = get_demand_forecasting_system()
system.create_model("PROD001")
system.add_historical_data("PROD001", [100, 120, 110, 130, 125])
forecast = system.forecast("PROD001", forecast_days=30)
```

### 2. Predictive Maintenance (`core/predictive_maintenance.py`)

**Modelo:**
- CNN 1D para análisis de señales de sensores
- Clasificación de estado (healthy, warning, critical, failure)
- Regresión para tiempo restante de vida

**Características:**
- Análisis de datos de sensores en tiempo real
- Predicción de fallas antes de que ocurran
- Recomendaciones de mantenimiento

**Ejemplo:**
```python
from manufacturing_ai.core.predictive_maintenance import (
    get_predictive_maintenance_system,
    SensorData
)

system = get_predictive_maintenance_system()
system.create_model("EQUIP001")

# Agregar datos de sensores
sensor_data = SensorData(
    equipment_id="EQUIP001",
    temperature=75.5,
    vibration=2.3,
    pressure=101.2,
    current=15.8
)
system.add_sensor_data(sensor_data)

# Predecir falla
prediction = system.predict_failure("EQUIP001")
```

### 3. Capacity Optimization (`core/capacity_optimizer.py`)

**Modelo:**
- MLP para predecir capacidad óptima
- Optimización de utilización de recursos
- Asignación inteligente de recursos

**Características:**
- Optimización de capacidad basada en demanda
- Recomendaciones de ajuste de capacidad
- Asignación eficiente de recursos a tareas

**Ejemplo:**
```python
from manufacturing_ai.core.capacity_optimizer import get_capacity_optimizer

optimizer = get_capacity_optimizer()
optimizer.create_model("RESOURCE001")

plan = optimizer.optimize_capacity(
    resource_id="RESOURCE001",
    current_load=80.0,
    demand_forecast=100.0,
    efficiency=0.85
)

allocation = optimizer.allocate_resource(
    resource_id="RESOURCE001",
    task_id="TASK001",
    required_capacity=20.0,
    duration=8.0
)
```

## API Endpoints

### Demand Forecasting
- `POST /api/v1/manufacturing/demand/forecast`: Predecir demanda
- `POST /api/v1/manufacturing/demand/add-data`: Agregar datos históricos

### Predictive Maintenance
- `POST /api/v1/manufacturing/maintenance/predict`: Predecir mantenimiento
- `POST /api/v1/manufacturing/maintenance/sensor-data`: Agregar datos de sensores

### Capacity Optimization
- `POST /api/v1/manufacturing/capacity/optimize`: Optimizar capacidad
- `POST /api/v1/manufacturing/capacity/allocate`: Asignar recurso

## Flujo de Trabajo Completo

### 1. Predicción de Demanda
```python
# Agregar datos históricos
POST /api/v1/manufacturing/demand/add-data
{
    "product_id": "PROD001",
    "data": [100, 120, 110, 130, 125, 140]
}

# Predecir demanda
POST /api/v1/manufacturing/demand/forecast
{
    "product_id": "PROD001",
    "forecast_days": 30
}
```

### 2. Mantenimiento Predictivo
```python
# Agregar datos de sensores
POST /api/v1/manufacturing/maintenance/sensor-data
{
    "equipment_id": "EQUIP001",
    "temperature": 75.5,
    "vibration": 2.3,
    "pressure": 101.2,
    "current": 15.8
}

# Predecir falla
POST /api/v1/manufacturing/maintenance/predict
{
    "equipment_id": "EQUIP001"
}
```

### 3. Optimización de Capacidad
```python
# Optimizar capacidad
POST /api/v1/manufacturing/capacity/optimize
{
    "resource_id": "RESOURCE001",
    "current_load": 80.0,
    "demand_forecast": 100.0,
    "efficiency": 0.85
}

# Asignar recurso
POST /api/v1/manufacturing/capacity/allocate
{
    "resource_id": "RESOURCE001",
    "task_id": "TASK001",
    "required_capacity": 20.0,
    "duration": 8.0
}
```

## Arquitectura de Modelos

### Demand Forecasting Model
- **Input**: Secuencia histórica de demanda
- **Architecture**: LSTM con múltiples capas
- **Output**: Predicción + intervalo de confianza

### Predictive Maintenance Model
- **Input**: Señales de sensores (temperatura, vibración, presión, corriente)
- **Architecture**: CNN 1D + Clasificador + Regresor
- **Output**: Estado + tiempo restante de vida

### Capacity Optimizer Model
- **Input**: Carga actual, demanda pronosticada, eficiencia
- **Architecture**: MLP con dropout
- **Output**: Capacidad óptima

## Integración con Sistemas Existentes

- **Production Planner**: Usa pronósticos de demanda para planificar
- **Monitoring**: Integra predicciones de mantenimiento
- **Process Optimizer**: Usa optimización de capacidad

## Ventajas

1. **Predicción Precisa**: Modelos de deep learning para predicciones más precisas
2. **Mantenimiento Proactivo**: Evita fallas antes de que ocurran
3. **Optimización Continua**: Ajusta capacidad según demanda
4. **Reducción de Costos**: Optimiza recursos y reduce downtime
5. **Toma de Decisiones**: Datos para decisiones informadas

## Estado

✅ **Completado y listo para producción**

Todos los sistemas de PlanOps están implementados y listos para uso.

