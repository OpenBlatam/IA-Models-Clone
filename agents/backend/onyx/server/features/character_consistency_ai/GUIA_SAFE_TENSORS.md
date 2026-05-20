# Guía Rápida: Safe Tensors de Consistencia de Personaje

## 🎯 ¿Qué son los Safe Tensors?

Los safe tensors son archivos `.safetensors` que contienen embeddings de personajes generados desde imágenes. Estos archivos permiten mantener la consistencia de un personaje en diferentes generaciones de imágenes.

## 📦 Generar Safe Tensors

### Método 1: Script de Línea de Comandos

```bash
# Generar desde una o múltiples imágenes
python generate_safe_tensors.py generate image1.jpg image2.jpg --name MyCharacter

# Listar safe tensors existentes
python generate_safe_tensors.py list

# Crear workflow tensor desde un embedding
python generate_safe_tensors.py workflow character_embeddings/my_character.safetensors
```

### Método 2: Desde Python

```python
from character_consistency_ai.core.character_consistency_service import CharacterConsistencyService
from character_consistency_ai.config.character_consistency_config import CharacterConsistencyConfig

# Crear servicio
config = CharacterConsistencyConfig(output_dir="./character_embeddings")
service = CharacterConsistencyService(config=config)
service.initialize_model()

# Generar safe tensor
result = service.generate_character_embedding(
    images=["image1.jpg", "image2.jpg"],
    character_name="MyCharacter",
    save_tensor=True,
)

print(f"Safe tensor guardado en: {result['saved_path']}")
```

### Método 3: Usando la API

```bash
# Generar embedding desde imágenes
curl -X POST "http://localhost:8001/api/v1/generate" \
  -F "images=@image1.jpg" \
  -F "images=@image2.jpg" \
  -F "character_name=MyCharacter" \
  -F "save_tensor=true"
```

## 📂 Ubicación de los Safe Tensors

Por defecto, los safe tensors se guardan en:
```
./character_embeddings/
```

Cada safe tensor incluye:
- `nombre_character_timestamp.safetensors` - El archivo del embedding
- `nombre_character_timestamp.json` - Metadata con información del personaje

## 🔍 Ver Safe Tensors Existentes

### Desde Python

```python
from character_consistency_ai.core.character_consistency_service import CharacterConsistencyService

service = CharacterConsistencyService()
embeddings = service.list_embeddings()

for emb in embeddings:
    print(f"Archivo: {emb['filename']}")
    print(f"Personaje: {emb['metadata'].get('character_name', 'N/A')}")
    print(f"Imágenes: {emb['metadata'].get('num_images', 'N/A')}")
    print()
```

### Desde Línea de Comandos

```bash
python generate_safe_tensors.py list
```

## 💾 Cargar Safe Tensors

### Método 1: Usando el modelo

```python
from character_consistency_ai.models.flux2_character_model import Flux2CharacterConsistencyModel

embedding, metadata = Flux2CharacterConsistencyModel.load_embedding(
    "character_embeddings/my_character.safetensors",
    device="cuda"
)

print(f"Embedding shape: {embedding.shape}")
print(f"Character: {metadata['character_name']}")
```

### Método 2: Directamente con safetensors

```python
from safetensors.torch import load_file
import json
from pathlib import Path

# Cargar safe tensor
data = load_file("character_embeddings/my_character.safetensors")
embedding = data["character_embedding"]

# Cargar metadata
metadata_path = Path("character_embeddings/my_character.safetensors").with_suffix(".json")
with open(metadata_path, "r") as f:
    metadata = json.load(f)

print(f"Embedding: {embedding.shape}")
print(f"Metadata: {metadata}")
```

### Método 3: Script de ejemplo

```bash
python load_safe_tensors_example.py
```

## 🎨 Usar Safe Tensors en Workflows

### Ejemplo Básico

```python
from safetensors.torch import load_file
import torch

# Cargar embedding
data = load_file("character_embeddings/my_character.safetensors")
character_embedding = data["character_embedding"].to("cuda")

# Usar en tu pipeline de generación
# (ejemplo con diffusers)
from diffusers import StableDiffusionPipeline

pipeline = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5"
).to("cuda")

# Inyectar embedding en la generación
prompt = "A photo of my character in a forest"
# image = pipeline(prompt, character_embedding=character_embedding).images[0]
```

### Workflow Tensor Completo

```python
from character_consistency_ai.models.safe_tensor_generator import SafeTensorGenerator

generator = SafeTensorGenerator()
embedding, workflow_config, metadata = generator.load_workflow_tensor(
    "character_embeddings/workflow_20240101_120000.safetensors",
    device="cuda"
)

# Usar configuración del workflow
prompt_template = workflow_config["prompt_template"]
negative_prompt = workflow_config["negative_prompt"]
num_inference_steps = workflow_config["num_inference_steps"]
guidance_scale = workflow_config["guidance_scale"]

# Generar prompt
character_name = metadata.get("character_name", "character")
prompt = prompt_template.format(character=character_name)
```

## 📊 Estructura de los Safe Tensors

### Embedding Simple

```python
{
    "character_embedding": torch.Tensor  # Shape: [embedding_dim]
}
```

### Workflow Tensor

```python
{
    "character_embedding": torch.Tensor,  # Shape: [embedding_dim]
    "workflow_config": {
        "prompt_template": str,
        "negative_prompt": str,
        "num_inference_steps": int,
        "guidance_scale": float,
        "character_metadata": dict
    }
}
```

### Metadata JSON

```json
{
    "character_name": "MyCharacter",
    "num_images": 2,
    "embedding_dim": 768,
    "created_at": "2024-01-01T12:00:00",
    "model_id": "black-forest-labs/flux2-dev",
    "device": "cuda"
}
```

## 🔧 Comandos Útiles

### Listar todos los safe tensors

```bash
python generate_safe_tensors.py list
```

### Generar desde múltiples imágenes

```bash
python generate_safe_tensors.py generate \
    front.jpg side.jpg back.jpg \
    --name HeroCharacter \
    --output-dir ./my_embeddings
```

### Crear workflow tensor

```bash
python generate_safe_tensors.py workflow \
    character_embeddings/my_character.safetensors \
    --prompt "A photo of {character} in {setting}" \
    --negative "blurry, low quality" \
    --steps 50 \
    --guidance 7.5
```

## 📝 Ejemplos Completos

Ver los archivos:
- `generate_safe_tensors.py` - Script completo para generar
- `load_safe_tensors_example.py` - Ejemplos de carga y uso
- `example_usage.py` - Más ejemplos de uso
- `WORKFLOW_USAGE.md` - Guía detallada de workflows

## ❓ Preguntas Frecuentes

### ¿Dónde se guardan los safe tensors?

Por defecto en `./character_embeddings/`, pero puedes configurarlo con `CHARACTER_CONSISTENCY_OUTPUT_DIR` o en la configuración.

### ¿Puedo usar safe tensors de otros modelos?

Sí, siempre que tengan la misma estructura. Los safe tensors son formato estándar.

### ¿Cómo comparto un safe tensor?

Simplemente comparte el archivo `.safetensors` y su `.json` asociado (si existe).

### ¿Puedo editar un safe tensor?

Sí, puedes cargarlo, modificarlo y guardarlo de nuevo usando `save_file()` de safetensors.

## 🚀 Próximos Pasos

1. Genera tu primer safe tensor con imágenes de un personaje
2. Experimenta con diferentes números de imágenes
3. Crea workflow tensors para automatizar generaciones
4. Integra los safe tensors en tu pipeline de generación


