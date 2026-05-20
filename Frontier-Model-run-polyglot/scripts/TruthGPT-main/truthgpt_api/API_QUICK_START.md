# TruthGPT API - Guía Rápida

## 🚀 Inicio Rápido

### 1. Instalar Dependencias

```bash
pip install -r requirements.txt
```

### 2. Iniciar el Servidor

```bash
python start_server.py
```

El servidor se iniciará en `http://localhost:8000` por defecto.

### 3. Verificar que Funciona

Abre tu navegador y visita:
- **Documentación de la API**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

### 4. Usar la API

#### Desde Python:

```python
import requests

# Crear un modelo
response = requests.post('http://localhost:8000/models/create', json={
    "layers": [
        {"type": "dense", "params": {"units": 128, "activation": "relu"}},
        {"type": "dense", "params": {"units": 10, "activation": "softmax"}}
    ]
})
model_id = response.json()["model_id"]

# Compilar el modelo
requests.post(f'http://localhost:8000/models/{model_id}/compile', json={
    "optimizer": "adam",
    "optimizer_params": {"learning_rate": 0.001},
    "loss": "sparsecategoricalcrossentropy",
    "metrics": ["accuracy"]
})

# Entrenar el modelo
requests.post(f'http://localhost:8000/models/{model_id}/train', json={
    "x_train": [[...]],  # Tus datos de entrenamiento
    "y_train": [...],     # Tus etiquetas
    "epochs": 10,
    "batch_size": 32
})

# Hacer predicciones
response = requests.post(f'http://localhost:8000/models/{model_id}/predict', json={
    "x": [[...]]  # Tus datos de entrada
})
predictions = response.json()["predictions"]
```

#### Desde JavaScript/TypeScript:

```javascript
// Crear un modelo
const response = await fetch('http://localhost:8000/models/create', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
        layers: [
            { type: "dense", params: { units: 128, activation: "relu" } },
            { type: "dense", params: { units: 10, activation: "softmax" } }
        ]
    })
});
const { model_id } = await response.json();

// Compilar el modelo
await fetch(`http://localhost:8000/models/${model_id}/compile`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
        optimizer: "adam",
        optimizer_params: { learning_rate: 0.001 },
        loss: "sparsecategoricalcrossentropy",
        metrics: ["accuracy"]
    })
});

// Entrenar el modelo
await fetch(`http://localhost:8000/models/${model_id}/train`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
        x_train: [[...]],  // Tus datos de entrenamiento
        y_train: [...],    // Tus etiquetas
        epochs: 10,
        batch_size: 32
    })
});

// Hacer predicciones
const predResponse = await fetch(`http://localhost:8000/models/${model_id}/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
        x: [[...]]  // Tus datos de entrada
    })
});
const { predictions } = await predResponse.json();
```

## 📋 Endpoints Disponibles

### Modelos
- `POST /models/create` - Crear un nuevo modelo
- `GET /models` - Listar todos los modelos
- `GET /models/{model_id}` - Obtener información de un modelo
- `DELETE /models/{model_id}` - Eliminar un modelo
- `POST /models/{model_id}/compile` - Compilar un modelo
- `POST /models/{model_id}/train` - Entrenar un modelo
- `POST /models/{model_id}/evaluate` - Evaluar un modelo
- `POST /models/{model_id}/predict` - Hacer predicciones
- `POST /models/{model_id}/save` - Guardar un modelo
- `POST /models/load` - Cargar un modelo

### Utilidades
- `GET /` - Información del servidor
- `GET /health` - Health check

## 🔧 Configuración Avanzada

### Iniciar con opciones personalizadas:

```bash
# Puerto personalizado
python start_server.py --port 3000

# Host personalizado
python start_server.py --host 0.0.0.0 --port 8000

# Modo desarrollo (auto-reload)
python start_server.py --reload

# Múltiples workers
python start_server.py --workers 4
```

## 📚 Ejemplos

Ejecuta el ejemplo completo:

```bash
# Terminal 1: Inicia el servidor
python start_server.py

# Terminal 2: Ejecuta el ejemplo
python example_api_usage.py
```

## 🐛 Solución de Problemas

### Error: "Cannot connect to server"
- Verifica que el servidor esté ejecutándose
- Verifica que el puerto no esté en uso
- Verifica que no haya firewall bloqueando la conexión

### Error: "Module not found"
- Asegúrate de haber instalado todas las dependencias: `pip install -r requirements.txt`
- Verifica que estés en el directorio correcto

### Error: "Model not found"
- Verifica que el `model_id` sea correcto
- Lista los modelos con `GET /models` para ver los IDs disponibles

## 📞 Soporte

Para más información, consulta:
- README.md - Documentación completa
- /docs - Documentación interactiva de la API (cuando el servidor está corriendo)











