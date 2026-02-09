# 🧪 Tests Unitarios - TruthGPT API

## ✅ Tests Unitarios Disponibles

Los tests unitarios **NO requieren que el servidor esté corriendo**. Solo prueban componentes individuales.

### 1. `test_unit_models.py` - Models y Layers

**TestSequentialModel**
- ✅ Crear Sequential vacío
- ✅ Crear Sequential con layers
- ✅ Agregar layer a Sequential
- ✅ Forward pass de Sequential

**TestDenseLayer**
- ✅ Crear Dense layer
- ✅ Dense con activación
- ✅ Forward pass de Dense
- ✅ Dense sin bias

**TestDropoutLayer**
- ✅ Crear Dropout layer
- ✅ Dropout en modo training
- ✅ Dropout en modo evaluación

**TestFlattenLayer**
- ✅ Crear Flatten layer
- ✅ Forward pass de Flatten

**TestAdamOptimizer**
- ✅ Crear Adam optimizer
- ✅ Adam con parámetros

**TestSGDOptimizer**
- ✅ Crear SGD optimizer
- ✅ SGD con momentum

**TestLossFunctions**
- ✅ SparseCategoricalCrossentropy
- ✅ MeanSquaredError

**TestMetrics**
- ✅ Accuracy metric

**TestModelCompilation**
- ✅ Compilar modelo
- ✅ Compilar con metrics

**TestModelTraining**
- ✅ Fit básico
- ✅ Fit con validación

**TestModelEvaluation**
- ✅ Evaluate modelo

**TestModelPrediction**
- ✅ Predict con validación de probabilidades

### 2. `test_unit_utils.py` - Utilidades

**TestToCategorical**
- ✅ Conversión básica a one-hot
- ✅ Con una sola clase
- ✅ Inferir número de clases

**TestNormalize**
- ✅ Normalización básica
- ✅ Con una sola muestra
- ✅ Vector cero

**TestDataUtils**
- ✅ Casos edge de utilidades

**TestTypeConversions**
- ✅ Numpy a Torch
- ✅ Torch a Numpy

### 3. `test_unit_api_helpers.py` - Helpers de API

**TestLayerCreation**
- ✅ Crear Dense desde config
- ✅ Crear Conv2D desde config
- ✅ Crear Dropout desde config
- ✅ Validar layer inválido

**TestOptimizerCreation**
- ✅ Crear Adam desde config
- ✅ Crear SGD desde config
- ✅ Crear RMSprop desde config
- ✅ Validar optimizer inválido

**TestLossCreation**
- ✅ Crear SparseCategoricalCrossentropy
- ✅ Crear MSE
- ✅ Crear MAE
- ✅ Validar loss inválido

**TestJSONSerialization**
- ✅ Serializar numpy array
- ✅ Deserializar JSON
- ✅ Serializar history del modelo

## 🚀 Ejecutar Tests Unitarios

### Opción 1: Script Automático (Recomendado)

```bash
cd truthgpt_api
python run_unit_tests.py
```

### Opción 2: Pytest Directamente

```bash
cd truthgpt_api

# Todos los unitarios
pytest tests/test_unit_*.py -v

# Específicos
pytest tests/test_unit_models.py -v
pytest tests/test_unit_utils.py -v
pytest tests/test_unit_api_helpers.py -v
```

### Opción 3: Test Específico

```bash
# Solo Sequential model
pytest tests/test_unit_models.py::TestSequentialModel -v

# Solo Dense layer
pytest tests/test_unit_models.py::TestDenseLayer -v

# Solo un test específico
pytest tests/test_unit_models.py::TestSequentialModel::test_sequential_creation_empty -v
```

## 📊 Ventajas de Tests Unitarios

✅ **No requieren servidor** - Puedes ejecutarlos en cualquier momento  
✅ **Rápidos** - Prueban componentes individuales  
✅ **Aislados** - Cada test es independiente  
✅ **Fáciles de depurar** - Falla en el componente específico  

## 🔍 Ver Cobertura

```bash
pytest tests/test_unit_*.py --cov=. --cov-report=html
# Luego abre: htmlcov/index.html
```

## 📝 Ejemplo de Output

```
🧪 Ejecutando Tests Unitarios de TruthGPT API
======================================================================

🧪 Ejecutando: test_unit_models.py
======================================================================

tests/test_unit_models.py::TestSequentialModel::test_sequential_creation_empty PASSED
tests/test_unit_models.py::TestSequentialModel::test_sequential_creation_with_layers PASSED
tests/test_unit_models.py::TestDenseLayer::test_dense_creation PASSED
...

✅ test_unit_models.py: PASÓ
✅ test_unit_utils.py: PASÓ
✅ test_unit_api_helpers.py: PASÓ

🎉 ¡Todos los tests unitarios pasaron!
```

## ⚠️ Nota

Si los tests fallan porque no se pueden importar los módulos, asegúrate de que:
1. Estás en el directorio correcto: `truthgpt_api`
2. Los módulos están disponibles en el path
3. Las dependencias están instaladas: `pip install -r requirements.txt`











