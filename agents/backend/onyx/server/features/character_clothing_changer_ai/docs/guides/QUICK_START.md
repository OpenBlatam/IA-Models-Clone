# 🚀 Inicio Rápido - Character Clothing Changer AI

## Ejecutar el Servidor

### Opción 1: Script de inicio (Recomendado)

**Windows:**
```bash
scripts/start.bat
```

**Linux/Mac:**
```bash
chmod +x scripts/start.sh
./scripts/start.sh
```

### Opción 2: Python directo

```bash
cd character_clothing_changer_ai
python run_server.py
```

### Opción 3: Como módulo

```bash
cd features
python -m character_clothing_changer_ai.main
```

## Verificar que está funcionando

Una vez iniciado, el servidor estará disponible en:

- **API Base:** http://localhost:8002
- **Documentación:** http://localhost:8002/docs
- **Health Check:** http://localhost:8002/api/v1/health

### Probar con curl:

```bash
# Health check
curl http://localhost:8002/api/v1/health

# Listar tensors
curl http://localhost:8002/api/v1/tensors

# Info del modelo
curl http://localhost:8002/api/v1/model/info
```

## Cambiar la ropa de un personaje

### Desde la API:

```bash
curl -X POST "http://localhost:8002/api/v1/change-clothing" \
  -F "image=@character.jpg" \
  -F "clothing_description=a red elegant dress" \
  -F "character_name=MyCharacter" \
  -F "save_tensor=true"
```

### Desde Python:

```python
from character_clothing_changer_ai.core.clothing_changer_service import ClothingChangerService

service = ClothingChangerService()
service.initialize_model()

result = service.change_clothing(
    image="character.jpg",
    clothing_description="a red elegant dress",
    character_name="MyCharacter",
    save_tensor=True,
)

print(f"Safe tensor: {result['saved_path']}")
```

## Generar Safe Tensors para ComfyUI

Los safe tensors generados están listos para usar en ComfyUI:

1. Los tensors se guardan en `./comfyui_tensors/`
2. Cada tensor incluye:
   - `character_embedding`: Embedding del personaje
   - `clothing_embedding`: Embedding de la ropa
   - `combined_embedding`: Embedding combinado para ComfyUI

3. También se genera un archivo JSON con metadata

## Crear Workflow de ComfyUI

```bash
curl -X POST "http://localhost:8002/api/v1/create-workflow" \
  -F "tensor_path=./comfyui_tensors/comfyui_character_red_dress_20240101_120000.safetensors" \
  -F "prompt=a character wearing a red elegant dress, high quality" \
  -F "negative_prompt=blurry, low quality"
```

Esto generará un archivo JSON de workflow que puedes importar directamente en ComfyUI.

## Configuración

El servidor usa estas configuraciones por defecto:

- **Host:** 0.0.0.0
- **Port:** 8002
- **Output Dir:** ./comfyui_tensors
- **Model:** black-forest-labs/flux2-dev

Puedes cambiarlas con variables de entorno:

```bash
set CLOTHING_CHANGER_API_PORT=8080
set CLOTHING_CHANGER_OUTPUT_DIR=./my_tensors
python run_server.py
```

## Próximos Pasos

1. ✅ Servidor iniciado
2. 📸 Sube una imagen de personaje
3. 👔 Describe la nueva ropa
4. 💾 Obtén el safe tensor para ComfyUI
5. 🎨 Usa el tensor en tus workflows de ComfyUI

## Troubleshooting

### Error: "Module not found"
```bash
pip install -r requirements.txt
```

### Error: "Port already in use"
Cambia el puerto:
```bash
set CLOTHING_CHANGER_API_PORT=8003
python run_server.py
```

### Error: "CUDA out of memory"
Usa CPU o reduce el batch size:
```bash
set CLOTHING_CHANGER_DEVICE=cpu
python run_server.py
```

