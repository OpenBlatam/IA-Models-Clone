# Uso de Safe Tensors en Workflows

Esta guía explica cómo usar los safe tensors generados en workflows de generación de imágenes.

## 📦 Formato de Safe Tensors

Los safe tensors generados contienen:

1. **character_embedding**: Tensor con el embedding del personaje
2. **metadata.json**: Archivo JSON con información adicional

## 🔧 Cargar Safe Tensors

### Método 1: Usando el modelo directamente

```python
from character_consistency_ai.models.flux2_character_model import Flux2CharacterConsistencyModel

# Cargar embedding
embedding, metadata = Flux2CharacterConsistencyModel.load_embedding(
    "character_embeddings/my_character.safetensors",
    device="cuda"
)

print(f"Embedding shape: {embedding.shape}")
print(f"Character: {metadata.get('character_name')}")
```

### Método 2: Usando safetensors directamente

```python
from safetensors.torch import load_file
import json
from pathlib import Path

# Cargar safe tensor
data = load_file("character_embeddings/my_character.safetensors")
embedding = data["character_embedding"]

# Cargar metadata
metadata_path = Path("character_embeddings/my_character.json")
if metadata_path.exists():
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
```

## 🎨 Integración con Workflows

### Ejemplo 1: Stable Diffusion con ControlNet

```python
from diffusers import StableDiffusionPipeline, ControlNetModel
from safetensors.torch import load_file
import torch

# Cargar embedding del personaje
character_data = load_file("character_embeddings/my_character.safetensors")
character_embedding = character_data["character_embedding"]

# Cargar pipeline
pipeline = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
).to("cuda")

# Inyectar embedding en el prompt encoding
# (implementación específica depende de tu workflow)
prompt = "A photo of my character in a forest"
# ... usar character_embedding para condicionar la generación ...
```

### Ejemplo 2: Flux2 Pipeline

```python
from diffusers import FluxPipeline
from safetensors.torch import load_file
import torch

# Cargar embedding
character_data = load_file("character_embeddings/my_character.safetensors")
character_embedding = character_data["character_embedding"]

# Cargar Flux2 pipeline
pipeline = FluxPipeline.from_pretrained(
    "black-forest-labs/flux2-dev",
    torch_dtype=torch.float16
).to("cuda")

# Usar embedding para condicionar generación
# (adaptar según la API de Flux2)
```

### Ejemplo 3: Workflow Tensor Completo

```python
from character_consistency_ai.models.safe_tensor_generator import SafeTensorGenerator
from safetensors.torch import load_file

# Cargar workflow tensor
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

# Usar en pipeline
# ... implementar generación con estos parámetros ...
```

## 🔄 Flujo de Trabajo Completo

### Paso 1: Generar Embedding

```python
from character_consistency_ai.core.character_consistency_service import CharacterConsistencyService

service = CharacterConsistencyService()
service.initialize_model()

result = service.generate_character_embedding(
    images=["character1.jpg", "character2.jpg"],
    character_name="MyCharacter",
    save_tensor=True,
)
```

### Paso 2: Crear Workflow Tensor

```python
workflow_result = service.create_workflow_tensor(
    embedding_path=result["saved_path"],
    prompt_template="A photo of {character} in {setting}",
    negative_prompt="blurry, low quality",
    num_inference_steps=50,
    guidance_scale=7.5,
)
```

### Paso 3: Usar en Generación

```python
# Cargar workflow tensor
embedding, config, metadata = generator.load_workflow_tensor(
    workflow_result["workflow_tensor_path"]
)

# Generar imágenes con el embedding
# ... implementar generación ...
```

## 📝 Notas Importantes

1. **Dimensiones**: Los embeddings tienen dimensión configurable (default: 768)
2. **Dispositivo**: Asegúrate de cargar los tensors en el dispositivo correcto (CUDA/CPU)
3. **Metadata**: Siempre revisa el metadata para información adicional sobre el personaje
4. **Compatibilidad**: Los safe tensors son compatibles con cualquier librería que soporte safetensors

## 🎯 Casos de Uso

### Mantener Consistencia en Series de Imágenes

```python
# Generar múltiples imágenes del mismo personaje
character_embedding = load_file("character.safetensors")["character_embedding"]

for setting in ["forest", "city", "beach"]:
    prompt = f"A photo of my character in a {setting}"
    # Usar character_embedding para mantener consistencia
    # ... generar imagen ...
```

### Transferencia de Estilo de Personaje

```python
# Usar embedding de un personaje para generar variaciones
source_embedding = load_file("source_character.safetensors")["character_embedding"]

# Aplicar a diferentes estilos
styles = ["realistic", "anime", "cartoon"]
for style in styles:
    prompt = f"A {style} style photo of my character"
    # Combinar source_embedding con style conditioning
    # ... generar imagen ...
```

## 🔍 Debugging

### Verificar Embedding

```python
import torch

embedding = load_file("character.safetensors")["character_embedding"]

print(f"Shape: {embedding.shape}")
print(f"Device: {embedding.device}")
print(f"Dtype: {embedding.dtype}")
print(f"Min: {embedding.min()}, Max: {embedding.max()}")
print(f"Mean: {embedding.mean()}, Std: {embedding.std()}")
```

### Comparar Embeddings

```python
emb1 = load_file("character1.safetensors")["character_embedding"]
emb2 = load_file("character2.safetensors")["character_embedding"]

# Calcular similitud (cosine similarity)
similarity = torch.nn.functional.cosine_similarity(
    emb1.unsqueeze(0),
    emb2.unsqueeze(0)
)
print(f"Similarity: {similarity.item():.4f}")
```


