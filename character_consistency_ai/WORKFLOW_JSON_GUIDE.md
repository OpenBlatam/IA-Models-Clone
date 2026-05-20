# 📋 Guía: Safe Tensor en Formato JSON para Workflows

## 🎯 Formato JSON del Workflow

El workflow JSON tiene esta estructura:

```json
{
  "workflow_type": "character_consistency",
  "version": "1.0.0",
  "character": {
    "name": "NombreDelPersonaje",
    "embedding_dim": 768
  },
  "embedding": {
    "source": "external",
    "path": "character_embeddings/personaje.safetensors",
    "format": "safetensors"
  },
  "workflow_config": {
    "prompt_template": "A photo of {character} in a {setting}",
    "negative_prompt": "blurry, low quality",
    "num_inference_steps": 50,
    "guidance_scale": 7.5
  }
}
```

## 📦 Generar Workflow JSON desde Safe Tensor

### Opción 1: Con Embedding Incluido (Base64)

```bash
python export_to_workflow_json.py export character.safetensors -o workflow.json
```

Esto crea un JSON con el embedding serializado en base64.

### Opción 2: Referencia Externa (Recomendado)

```bash
python export_to_workflow_json.py export character.safetensors --no-embedding -o workflow.json
```

Esto crea un JSON que referencia el archivo .safetensors externo (más eficiente).

### Opción 3: Template Vacío

```bash
python export_to_workflow_json.py template --name MyCharacter -o workflow.json
```

## 📄 Ejemplo Completo de Workflow JSON

Ver el archivo `workflow_example.json` para un ejemplo completo.

## 💻 Usar el Workflow JSON en Python

```python
import json
from safetensors.torch import load_file
import torch

# 1. Cargar workflow JSON
with open('workflow.json', 'r') as f:
    workflow = json.load(f)

# 2. Cargar embedding desde referencia externa
embedding_path = workflow['embedding']['path']
data = load_file(embedding_path)
character_embedding = data['character_embedding'].to('cuda')

# 3. Formatear prompt
prompt_template = workflow['workflow_config']['prompt_template']
prompt = prompt_template.format(
    character=workflow['character']['name'],
    setting="a beautiful forest"
)

# 4. Usar en generación
# (ejemplo con diffusers)
from diffusers import FluxPipeline

pipeline = FluxPipeline.from_pretrained(
    workflow['model']['model_id']
).to('cuda')

image = pipeline(
    prompt=prompt,
    negative_prompt=workflow['workflow_config']['negative_prompt'],
    character_embedding=character_embedding,
    num_inference_steps=workflow['workflow_config']['num_inference_steps'],
    guidance_scale=workflow['workflow_config']['guidance_scale'],
).images[0]
```

## 🔧 Si el Embedding está en Base64

```python
import json
import base64
import numpy as np
import torch

# Cargar workflow
with open('workflow.json', 'r') as f:
    workflow = json.load(f)

# Decodificar embedding
embedding_data = workflow['embedding']
array_bytes = base64.b64decode(embedding_data['data'])
dtype = np.dtype(embedding_data['dtype'])
shape = tuple(embedding_data['shape'])

embedding = torch.from_numpy(
    np.frombuffer(array_bytes, dtype=dtype).reshape(shape)
).to('cuda')
```

## 📝 Estructura Completa del Workflow JSON

```json
{
  "workflow_type": "character_consistency",
  "version": "1.0.0",
  "character": {
    "name": "MyCharacter",
    "num_images": 2,
    "embedding_dim": 768,
    "created_at": "2024-01-01T12:00:00"
  },
  "model": {
    "model_id": "black-forest-labs/flux2-dev",
    "device": "cuda"
  },
  "embedding": {
    "source": "external",
    "path": "character_embeddings/my_character.safetensors",
    "format": "safetensors",
    "key": "character_embedding"
  },
  "workflow_config": {
    "prompt_template": "A photo of {character} in a {setting}",
    "negative_prompt": "blurry, low quality",
    "num_inference_steps": 50,
    "guidance_scale": 7.5,
    "height": 1024,
    "width": 1024
  },
  "steps": [
    {
      "step": 1,
      "name": "load_embedding",
      "action": "load_safetensor",
      "params": {
        "path": "character_embeddings/my_character.safetensors",
        "key": "character_embedding"
      }
    },
    {
      "step": 2,
      "name": "generate",
      "action": "diffusion_generate",
      "params": {
        "model": "black-forest-labs/flux2-dev",
        "prompt": "{formatted_prompt}",
        "character_embedding": "{loaded_embedding}"
      }
    }
  ]
}
```

## 🚀 Integración con ComfyUI, Automatic1111, etc.

### Para ComfyUI:

```json
{
  "workflow": {
    "character_embedding_path": "character_embeddings/my_character.safetensors",
    "prompt": "A photo of {character}",
    "negative_prompt": "blurry",
    "steps": 50,
    "cfg_scale": 7.5
  }
}
```

### Para Automatic1111:

Usa el JSON para configurar los parámetros en el script de generación.

## 📂 Archivos Relacionados

- `workflow_example.json` - Ejemplo completo de workflow
- `export_to_workflow_json.py` - Script para exportar
- `workflow_usage_example.py` - Ejemplos de uso en Python

## ✅ Checklist para Usar en Workflow

1. ✅ Generar safe tensor desde imágenes
2. ✅ Exportar a JSON con `export_to_workflow_json.py`
3. ✅ Cargar JSON en tu workflow
4. ✅ Cargar embedding desde path o base64
5. ✅ Usar en generación de imágenes


