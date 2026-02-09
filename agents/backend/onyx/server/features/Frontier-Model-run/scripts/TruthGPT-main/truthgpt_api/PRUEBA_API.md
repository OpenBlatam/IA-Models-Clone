# 🧪 Cómo Probar la API TruthGPT

## Opción 1: Prueba Automática (Recomendado)

### Paso 1: Iniciar el servidor
```bash
cd agents/backend/onyx/server/features/Frontier-Model-run/scripts/TruthGPT-main/truthgpt_api
python start_server.py
```

El servidor debería iniciar en `http://localhost:8000`

### Paso 2: Ejecutar prueba
En otra terminal:
```bash
cd agents/backend/onyx/server/features/Frontier-Model-run/scripts/TruthGPT-main/truthgpt_api
python test_api_quick.py
```

O usar el ejemplo completo:
```bash
python example_api_usage.py
```

## Opción 2: Prueba Manual con cURL

### Health Check
```bash
curl http://localhost:8000/health
```

### Crear Modelo
```bash
curl -X POST http://localhost:8000/models/create \
  -H "Content-Type: application/json" \
  -d '{
    "layers": [
      {"type": "dense", "params": {"units": 64, "activation": "relu"}},
      {"type": "dense", "params": {"units": 10, "activation": "softmax"}}
    ],
    "name": "test-model"
  }'
```

### Listar Modelos
```bash
curl http://localhost:8000/models
```

## Opción 3: Prueba desde el Navegador

1. Inicia el servidor: `python start_server.py`
2. Abre en tu navegador:
   - **Documentación interactiva**: http://localhost:8000/docs
   - **Health check**: http://localhost:8000/health
   - **Lista de modelos**: http://localhost:8000/models

3. En la documentación interactiva (`/docs`), puedes:
   - Probar cada endpoint directamente
   - Ver los esquemas de request/response
   - Ejecutar llamadas desde el navegador

## Opción 4: Prueba desde Python

```python
import requests

BASE_URL = "http://localhost:8000"

# 1. Health check
response = requests.get(f"{BASE_URL}/health")
print("Health:", response.json())

# 2. Crear modelo
response = requests.post(f"{BASE_URL}/models/create", json={
    "layers": [
        {"type": "dense", "params": {"units": 64, "activation": "relu"}},
        {"type": "dense", "params": {"units": 10, "activation": "softmax"}}
    ]
})
model_id = response.json()["model_id"]
print("Modelo creado:", model_id)

# 3. Compilar modelo
response = requests.post(f"{BASE_URL}/models/{model_id}/compile", json={
    "optimizer": "adam",
    "optimizer_params": {"learning_rate": 0.001},
    "loss": "sparsecategoricalcrossentropy",
    "metrics": ["accuracy"]
})
print("Modelo compilado:", response.json())

# 4. Listar modelos
response = requests.get(f"{BASE_URL}/models")
print("Modelos:", response.json())
```

## Verificación Rápida

Si el servidor está funcionando, deberías ver:
- ✅ Health check responde con `{"status": "healthy", ...}`
- ✅ Puedes crear modelos
- ✅ Puedes listar modelos
- ✅ La documentación en `/docs` está accesible

## Solución de Problemas

### Error: "Cannot connect to server"
- Verifica que el servidor esté corriendo
- Verifica que el puerto 8000 no esté en uso
- Verifica que no haya firewall bloqueando

### Error: "Module not found"
- Instala dependencias: `pip install -r requirements.txt`

### Error: "Address already in use"
- Cambia el puerto: `python start_server.py --port 8001`











