# 🚀 Acceso Rápido a Safe Tensors

## 📋 Ver Safe Tensors Disponibles

### Opción 1: Vista HTML Interactiva (Recomendado)

**Windows:**
```bash
open_safe_tensors.bat
```

O directamente:
```bash
python list_safe_tensors.py --html --open
```

Esto abrirá una página web interactiva donde puedes:
- ✅ Ver todos los safe tensors disponibles
- ✅ Copiar rutas con un clic
- ✅ Ver código de ejemplo para cargar cada tensor
- ✅ Ver metadata de cada personaje

### Opción 2: Lista en Consola

```bash
python list_safe_tensors.py
```

### Opción 3: Desde Python

```python
from character_consistency_ai.core.character_consistency_service import CharacterConsistencyService

service = CharacterConsistencyService()
embeddings = service.list_embeddings()

for emb in embeddings:
    print(f"Archivo: {emb['filename']}")
    print(f"Ruta: {emb['path']}")
    print(f"Personaje: {emb['metadata'].get('character_name', 'N/A')}")
    print()
```

## 📂 Ubicación de los Safe Tensors

Los safe tensors se guardan en:
```
./character_embeddings/
```

Y el visor HTML está en:
```
./character_embeddings/safe_tensors_viewer.html
```

## 🔗 Acceso Directo

Puedes abrir directamente el archivo HTML:
```
character_embeddings/safe_tensors_viewer.html
```

O usar el script:
```bash
python list_safe_tensors.py --html --open
```

## 💻 Cargar un Safe Tensor

Una vez que veas los safe tensors disponibles, puedes cargarlos así:

```python
from safetensors.torch import load_file

# Reemplaza con la ruta real de tu safe tensor
data = load_file("character_embeddings/my_character.safetensors")
embedding = data["character_embedding"]

print(f"Shape: {embedding.shape}")
```

## 🎯 Generar Nuevos Safe Tensors

Si no tienes safe tensors aún:

```bash
python generate_safe_tensors.py generate image1.jpg --name MyCharacter
```

O desde la API:
```bash
curl -X POST "http://localhost:8001/api/v1/generate" \
  -F "images=@image1.jpg" \
  -F "character_name=MyCharacter"
```


