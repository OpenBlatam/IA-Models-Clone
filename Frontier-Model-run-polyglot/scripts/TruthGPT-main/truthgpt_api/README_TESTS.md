# 🧪 Tests de la API TruthGPT

## Ejecutar Tests

### Opción 1: Script Automático (Recomendado)

```bash
python run_tests.py
```

Este script te pedirá que confirmes que el servidor está corriendo antes de ejecutar los tests.

### Opción 2: Pytest Directamente

1. **Asegúrate de que el servidor esté corriendo:**
```bash
python start_server.py
```

2. **En otra terminal, ejecuta los tests:**
```bash
pytest tests/ -v
```

O con más opciones:
```bash
pytest tests/ -v --tb=short --cov=.
```

### Opción 3: Tests Específicos

```bash
# Solo health check
pytest tests/test_api_endpoints.py::TestHealthCheck -v

# Solo creación de modelos
pytest tests/test_api_endpoints.py::TestModelCreation -v

# Test específico
pytest tests/test_api_endpoints.py::TestModelCreation::test_create_model_simple -v
```

## Tests Incluidos

### ✅ test_api_endpoints.py
- **TestHealthCheck**: Health check y root endpoint
- **TestModelCreation**: Creación de modelos (válidos e inválidos)
- **TestModelCompilation**: Compilación de modelos
- **TestModelTraining**: Entrenamiento de modelos
- **TestModelEvaluation**: Evaluación de modelos
- **TestModelPrediction**: Predicciones
- **TestModelManagement**: Gestión de modelos (listar, obtener, eliminar)

### ✅ test_layers.py
- **TestDenseLayers**: Layers Dense con diferentes configuraciones
- **TestConv2DLayers**: Layers Conv2D y pooling
- **TestLSTMLayers**: Layers LSTM
- **TestGRULayers**: Layers GRU
- **TestDropoutLayers**: Layers Dropout
- **TestPoolingLayers**: MaxPooling y AveragePooling
- **TestUtilityLayers**: Flatten, Reshape
- **TestComplexArchitectures**: Arquitecturas CNN, RNN, híbridas

### ✅ test_optimizers.py
- **TestAdamOptimizer**: Optimizer Adam con diferentes parámetros
- **TestSGDOptimizer**: Optimizer SGD con momentum
- **TestRMSpropOptimizer**: Optimizer RMSprop
- **TestAdagradOptimizer**: Optimizer Adagrad
- **TestAdamWOptimizer**: Optimizer AdamW
- **TestInvalidOptimizers**: Validación de optimizers inválidos

### ✅ test_losses.py
- **TestSparseCategoricalCrossentropy**: Loss para clasificación sparse
- **TestCategoricalCrossentropy**: Loss para clasificación categórica
- **TestBinaryCrossentropy**: Loss para clasificación binaria
- **TestMeanSquaredError**: Loss MSE
- **TestMeanAbsoluteError**: Loss MAE
- **TestInvalidLosses**: Validación de losses inválidos

### ✅ test_edge_cases.py
- **TestInvalidRequests**: Requests inválidos y validaciones
- **TestDataValidation**: Validación de datos de entrenamiento
- **TestModelOperations**: Operaciones repetidas (eliminar, compilar)
- **TestLargeModels**: Modelos grandes con muchas capas
- **TestConcurrentOperations**: Operaciones concurrentes

### ✅ test_unit_models.py
- **TestSequentialModel**: Tests unitarios para Sequential model
- **TestDenseLayer**: Tests unitarios para Dense layer
- **TestDropoutLayer**: Tests unitarios para Dropout layer
- **TestFlattenLayer**: Tests unitarios para Flatten layer
- **TestAdamOptimizer**: Tests unitarios para Adam optimizer
- **TestSGDOptimizer**: Tests unitarios para SGD optimizer
- **TestLossFunctions**: Tests unitarios para loss functions
- **TestMetrics**: Tests unitarios para metrics
- **TestModelCompilation**: Tests para compilación
- **TestModelTraining**: Tests para entrenamiento
- **TestModelEvaluation**: Tests para evaluación
- **TestModelPrediction**: Tests para predicción

### ✅ test_unit_utils.py
- **TestToCategorical**: Tests para to_categorical
- **TestNormalize**: Tests para normalize
- **TestDataUtils**: Tests para utilidades de datos
- **TestTypeConversions**: Tests para conversiones de tipos

### ✅ test_unit_api_helpers.py
- **TestLayerCreation**: Tests para creación de layers desde config
- **TestOptimizerCreation**: Tests para creación de optimizers
- **TestLossCreation**: Tests para creación de loss functions
- **TestJSONSerialization**: Tests para serialización JSON

### ✅ test_integration.py
- **TestLayerIntegration**: Integración entre diferentes layers
- **TestOptimizerLossIntegration**: Integración optimizers y losses
- **TestDataIntegration**: Integración con diferentes tipos de datos
- **TestMetricsIntegration**: Integración con múltiples metrics
- **TestConcurrentOperations**: Operaciones concurrentes

### ✅ test_performance.py
- **TestAPILatency**: Latencia de endpoints de la API
- **TestTrainingPerformance**: Rendimiento de entrenamiento
- **TestPredictionPerformance**: Rendimiento de predicciones
- **TestMemoryUsage**: Uso de memoria con múltiples modelos
- **TestScalability**: Escalabilidad con modelos grandes

### ✅ test_validation.py
- **TestInputValidation**: Validación de inputs y parámetros
- **TestDataValidation**: Validación de datos de entrenamiento
- **TestParameterValidation**: Validación de parámetros (learning_rate, epochs, etc.)
- **TestTypeValidation**: Validación de tipos de datos

### ✅ test_advanced_layers.py
- **TestAdvancedConv2D**: Conv2D con padding, diferentes kernel sizes, redes profundas
- **TestAdvancedRNN**: LSTM bidireccional, GRU apiladas
- **TestResidualConnections**: Patrones de skip connections
- **TestAttentionLayers**: Layers de atención y multi-head attention
- **TestBatchNormalization**: BatchNorm después de Conv y Dense
- **TestEmbeddingLayers**: Embedding layers
- **TestComplexArchitectures**: Transformer-like, CNN+RNN híbridas, redes anchas y profundas
- **TestLayerCombinations**: Patrones Conv-Pool, Dense-Dropout repetidos

### ✅ test_data_types.py
- **TestNumericDataTypes**: float32, float64, int32, int64
- **TestDataShapes**: Input 1D, 2D
- **TestDataRanges**: Datos normalizados, valores grandes/pequeños
- **TestCategoricalData**: Clasificación binaria y multiclase

### ✅ test_error_handling.py
- **TestHTTPErrors**: 404, 405, 422
- **TestModelErrors**: Errores de modelos (JSON inválido, modelo inexistente)
- **TestDataErrors**: Errores de datos (arrays vacíos, shapes incorrectos)
- **TestParameterErrors**: Errores de parámetros (valores inválidos)
- **TestErrorMessages**: Formato de mensajes de error

### ✅ test_concurrency.py
- **TestConcurrentHealthChecks**: Health checks simultáneos y rápidos
- **TestConcurrentModelCreation**: Creación concurrente de modelos
- **TestConcurrentTraining**: Entrenamiento concurrente
- **TestConcurrentPredictions**: Predicciones concurrentes
- **TestConcurrentListOperations**: Listado mientras se crean modelos
- **TestStressConcurrency**: Tests de stress con concurrencia

### ✅ test_model_serialization.py
- **TestModelInfo**: Obtener información de modelos
- **TestModelList**: Listar modelos (con paginación)
- **TestModelDeletion**: Eliminar modelos
- **TestModelState**: Estado antes/después de compilar/entrenar
- **TestModelHistory**: Historial de entrenamiento

### ✅ test_security.py
- **TestInputValidation**: Validación contra SQL injection, XSS, path traversal
- **TestDataValidation**: Validación de payloads grandes, JSON anidado
- **TestResourceLimits**: Límites de memoria y recursos
- **TestRateLimiting**: Rate limiting (si está implementado)
- **TestErrorHandling**: Manejo seguro de errores
- **TestCORS**: Headers CORS

### ✅ test_compatibility.py
- **TestJSONCompatibility**: Compatibilidad con diferentes formatos JSON
- **TestHTTPMethods**: GET, POST, DELETE
- **TestContentTypes**: Diferentes content types
- **TestDataFormats**: Numpy arrays, listas Python
- **TestClientCompatibility**: Curl, navegador, cliente API
- **TestVersionCompatibility**: Compatibilidad de versiones
- **TestErrorResponseFormat**: Formato de respuestas de error

## Requisitos

```bash
pip install -r requirements.txt
```

Especialmente necesitas:
- `pytest>=6.0.0`
- `pytest-timeout>=2.1.0`
- `requests>=2.28.0`

## Solución de Problemas

### Error: "Servidor no está corriendo"
- Inicia el servidor: `python start_server.py`
- Verifica que esté en `http://localhost:8000`

### Error: "Module not found"
- Instala dependencias: `pip install -r requirements.txt`

### Timeout en tests
- Los tests tienen timeout de 300 segundos
- Si un test falla por timeout, puede ser que el modelo esté tomando mucho tiempo
- Ajusta el timeout en `pytest.ini` si es necesario

### Tests fallan intermitentemente
- Asegúrate de que el servidor tenga suficientes recursos
- Verifica que no haya otros procesos usando el puerto 8000

## Cobertura de Tests

Para ver la cobertura:
```bash
pytest tests/ --cov=. --cov-report=html
```

Luego abre `htmlcov/index.html` en tu navegador.

## Estructura de Tests

```
tests/
├── __init__.py
├── test_api_endpoints.py    # Tests principales de endpoints
├── test_layers.py           # Tests para diferentes tipos de layers
├── test_optimizers.py       # Tests para diferentes optimizers
├── test_losses.py           # Tests para diferentes loss functions
├── test_edge_cases.py       # Tests para casos edge y validaciones
├── test_e2e.py              # Tests End-to-End completos
├── test_unit_models.py      # Tests unitarios para models y layers
├── test_unit_utils.py       # Tests unitarios para utilidades
├── test_unit_api_helpers.py # Tests unitarios para helpers de API
├── test_integration.py       # Tests de integración entre componentes
├── test_performance.py       # Tests de rendimiento y latencia
├── test_validation.py       # Tests de validación de datos
├── test_advanced_layers.py   # Tests avanzados de layers
├── test_data_types.py       # Tests para diferentes tipos de datos
├── test_error_handling.py   # Tests de manejo de errores
├── test_concurrency.py      # Tests de concurrencia
├── test_model_serialization.py  # Tests de serialización de modelos
├── test_security.py         # Tests de seguridad
└── test_compatibility.py    # Tests de compatibilidad
```

## Ejecutar Tests Específicos

```bash
# Solo tests de layers
pytest tests/test_layers.py -v

# Solo tests de optimizers
pytest tests/test_optimizers.py -v

# Solo tests de losses
pytest tests/test_losses.py -v

# Solo tests de casos edge
pytest tests/test_edge_cases.py -v

# Tests E2E (End-to-End)
python run_e2e_tests.py
# O directamente:
pytest tests/test_e2e.py -v -s

# Tests unitarios (no requieren servidor)
pytest tests/test_unit_models.py -v
pytest tests/test_unit_utils.py -v
pytest tests/test_unit_api_helpers.py -v

# Todos los tests unitarios
pytest tests/test_unit_*.py -v

# Tests de integración
pytest tests/test_integration.py -v

# Tests de rendimiento
pytest tests/test_performance.py -v -s

# Tests de validación
pytest tests/test_validation.py -v

# Tests avanzados
pytest tests/test_advanced_layers.py -v
pytest tests/test_data_types.py -v
pytest tests/test_error_handling.py -v

# Tests de concurrencia
pytest tests/test_concurrency.py -v

# Tests de serialización
pytest tests/test_model_serialization.py -v

# Tests de seguridad
pytest tests/test_security.py -v

# Tests de compatibilidad
pytest tests/test_compatibility.py -v

# Todos los tests (unitarios + integración + E2E + performance + validation + avanzados + nuevos)
pytest tests/ -v
```

## Tests E2E (End-to-End)

Los tests E2E verifican flujos completos de trabajo:

### ✅ TestCompleteMLWorkflow
- Flujo completo: Crear → Compilar → Entrenar → Evaluar → Predecir
- Verifica que todo el pipeline funcione correctamente
- Valida formatos de datos y resultados

### ✅ TestCNNWorkflow
- Flujo completo para modelos CNN
- Incluye Conv2D, Pooling, Flatten, Dense
- Verifica entrenamiento con datos de imagen simulados

### ✅ TestRNNWorkflow
- Flujo completo para modelos LSTM/RNN
- Incluye múltiples capas LSTM con dropout
- Verifica entrenamiento con datos secuenciales

### ✅ TestModelPersistence
- Guardar y cargar modelos
- Verificar que los modelos persistan correctamente

### ✅ TestMultipleModels
- Crear y gestionar múltiples modelos simultáneamente
- Listar, entrenar y eliminar modelos
- Verificar que no haya conflictos entre modelos

### ✅ TestErrorRecovery
- Verificar recuperación de errores
- Operaciones inválidas → Operaciones válidas
- Asegurar que el sistema se recupere correctamente

## Agregar Nuevos Tests

1. Agrega tu clase de test en `test_api_endpoints.py`
2. Usa el fixture `server_running` para asegurar que el servidor esté disponible
3. Sigue el patrón de los tests existentes
4. Ejecuta los tests para verificar que funcionan

Ejemplo:
```python
class TestMiNuevoFeature:
    def test_mi_funcionalidad(self, server_running):
        response = requests.get(f"{BASE_URL}/mi-endpoint")
        assert response.status_code == 200
```

