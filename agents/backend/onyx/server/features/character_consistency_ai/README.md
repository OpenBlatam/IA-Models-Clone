# Character Consistency AI - Flux2-based Model

Modelo basado en Flux2 para generar safe tensors de consistencia de personaje. Acepta una o varias imágenes y genera embeddings que mantienen la consistencia del personaje en diferentes generaciones.

## 🎯 Características

- **Modelo Flux2**: Basado en la arquitectura Flux2 de Black Forest Labs
- **Múltiples Imágenes**: Procesa una o varias imágenes para extraer características consistentes
- **Safe Tensors**: Genera archivos safe tensors listos para usar en workflows
- **API REST**: Endpoints FastAPI para integración fácil
- **Optimizaciones**: Soporte para optimizaciones de memoria y velocidad

## 📋 Requisitos

```bash
pip install -r requirements.txt
```

## 🚀 Uso Rápido

### 1. Uso Básico con Python

```python
from character_consistency_ai.core.character_consistency_service import CharacterConsistencyService
from character_consistency_ai.config.character_consistency_config import CharacterConsistencyConfig

# Crear configuración
config = CharacterConsistencyConfig.from_env()

# Crear servicio
service = CharacterConsistencyService(config=config)

# Inicializar modelo
service.initialize_model()

# Generar embedding desde imágenes
result = service.generate_character_embedding(
    images=["path/to/image1.jpg", "path/to/image2.jpg"],
    character_name="MyCharacter",
    save_tensor=True,
)

print(f"Embedding guardado en: {result['saved_path']}")
```

### 2. Crear Workflow Tensor

```python
# Crear tensor optimizado para workflow
workflow_result = service.create_workflow_tensor(
    embedding_path="character_embeddings/my_character_20240101_120000.safetensors",
    prompt_template="A photo of {character} in a {setting}",
    negative_prompt="blurry, low quality",
    num_inference_steps=50,
    guidance_scale=7.5,
)
```

### 3. Usar la API

```bash
# Iniciar servidor
python -m character_consistency_ai.main

# O usar uvicorn directamente
uvicorn character_consistency_ai.api.character_consistency_api:app --host 0.0.0.0 --port 8001
```

#### Endpoints Disponibles

- `POST /api/v1/generate` - Generar embedding desde imágenes
- `POST /api/v1/workflow` - Crear workflow tensor
- `GET /api/v1/embeddings` - Listar todos los embeddings
- `GET /api/v1/embedding/{id}` - Descargar embedding específico
- `GET /api/v1/model/info` - Información del modelo
- `POST /api/v1/initialize` - Inicializar modelo
- `GET /api/v1/health` - Health check

#### Ejemplo de Uso con cURL

```bash
# Generar embedding
curl -X POST "http://localhost:8001/api/v1/generate" \
  -F "images=@image1.jpg" \
  -F "images=@image2.jpg" \
  -F "character_name=MyCharacter" \
  -F "save_tensor=true"

# Crear workflow tensor
curl -X POST "http://localhost:8001/api/v1/workflow" \
  -F "embedding_path=character_embeddings/my_character.safetensors" \
  -F "prompt_template=A photo of {character}" \
  -F "negative_prompt=blurry" \
  -F "num_inference_steps=50" \
  -F "guidance_scale=7.5"
```

## 📁 Estructura del Proyecto

```
character_consistency_ai/
├── models/
│   ├── flux2_character_model.py      # Modelo Flux2 principal
│   └── safe_tensor_generator.py     # Generador de safe tensors
├── core/
│   └── character_consistency_service.py  # Servicio principal
├── api/
│   └── character_consistency_api.py      # API FastAPI
├── config/
│   └── character_consistency_config.py   # Configuración
├── main.py                               # Punto de entrada
└── requirements.txt                      # Dependencias
```

## ⚙️ Configuración

### Variables de Entorno

```bash
# Modelo
CHARACTER_CONSISTENCY_MODEL_ID=black-forest-labs/flux2-dev
CHARACTER_CONSISTENCY_DEVICE=cuda  # o cpu, auto
CHARACTER_CONSISTENCY_DTYPE=float16  # o float32

# Optimizaciones
CHARACTER_CONSISTENCY_OPTIMIZATIONS=true
CHARACTER_CONSISTENCY_EMBEDDING_DIM=768

# Output
CHARACTER_CONSISTENCY_OUTPUT_DIR=./character_embeddings

# API
CHARACTER_CONSISTENCY_API_HOST=0.0.0.0
CHARACTER_CONSISTENCY_API_PORT=8001

# Performance
CHARACTER_CONSISTENCY_MAX_BATCH_SIZE=4
CHARACTER_CONSISTENCY_MAX_IMAGE_SIZE=1024

# Workflow defaults
CHARACTER_CONSISTENCY_INFERENCE_STEPS=50
CHARACTER_CONSISTENCY_GUIDANCE_SCALE=7.5
```

### Configuración Programática

```python
from character_consistency_ai.config.character_consistency_config import CharacterConsistencyConfig

config = CharacterConsistencyConfig(
    model_id="black-forest-labs/flux2-dev",
    device="cuda",
    enable_optimizations=True,
    embedding_dim=768,
    output_dir="./my_embeddings",
)
```

## 🔧 Uso en Workflows

Los safe tensors generados pueden usarse en workflows de generación de imágenes:

```python
from safetensors.torch import load_file
import torch

# Cargar embedding
data = load_file("character_embeddings/my_character.safetensors")
character_embedding = data["character_embedding"]

# Usar en tu pipeline de generación
# (ejemplo con diffusers)
from diffusers import StableDiffusionPipeline

pipeline = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")

# Inyectar embedding en el pipeline
# (implementación específica depende de tu workflow)
```

## 📊 Formato de Safe Tensors

Los safe tensors generados contienen:

- `character_embedding`: Tensor con el embedding del personaje [embedding_dim]
- Metadata JSON asociada con información sobre:
  - Nombre del personaje
  - Número de imágenes usadas
  - Fecha de creación
  - Configuración del modelo

## 🎨 Ejemplos de Uso

### Procesar Múltiples Imágenes

```python
# Procesar varias imágenes del mismo personaje
images = [
    "character_front.jpg",
    "character_side.jpg",
    "character_back.jpg",
]

result = service.generate_character_embedding(
    images=images,
    character_name="Hero",
    metadata={"source": "game_assets", "version": "1.0"},
)
```

### Batch Processing

```python
# Procesar múltiples personajes
image_groups = [
    ["hero1_front.jpg", "hero1_side.jpg"],
    ["hero2_front.jpg", "hero2_side.jpg"],
    ["villain_front.jpg"],
]

character_names = ["Hero1", "Hero2", "Villain"]

results = service.batch_generate(
    image_groups=image_groups,
    character_names=character_names,
)
```

## 🔍 Información del Modelo

```python
# Obtener información del modelo
info = service.get_model_info()
print(info)
```

## 📝 Notas

- El modelo requiere GPU para mejor rendimiento (CUDA)
- Los embeddings tienen dimensión configurable (default: 768)
- Los safe tensors son compatibles con la librería `safetensors`
- El modelo se inicializa bajo demanda (lazy loading)

## 🐛 Troubleshooting

### Error: "Diffusers not available"
```bash
pip install diffusers transformers
```

### Error: "CUDA out of memory"
- Reduce `max_batch_size` en la configuración
- Usa `enable_optimizations=True`
- Considera usar CPU si la GPU no tiene suficiente memoria

### Error: "Model not found"
- Verifica que el modelo ID sea correcto
- Asegúrate de tener acceso a HuggingFace (token si es necesario)

## 📚 Referencias

- [Flux2 Model](https://github.com/black-forest-labs/flux2)
- [Safe Tensors](https://huggingface.co/docs/safetensors/)
- [Diffusers Library](https://huggingface.co/docs/diffusers/)

## 📄 Licencia

Este proyecto sigue las licencias de los modelos base utilizados.


