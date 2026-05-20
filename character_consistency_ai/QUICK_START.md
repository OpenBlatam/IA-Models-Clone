# 🚀 Inicio Rápido - Character Consistency AI

## Ejecutar el Servidor

### Opción 1: Script de inicio (Recomendado)

**Windows:**
```bash
start.bat
```

**Linux/Mac:**
```bash
chmod +x start.sh
./start.sh
```

### Opción 2: Python directo

```bash
cd character_consistency_ai
python run_server.py
```

### Opción 3: Como módulo

```bash
cd features
python -m character_consistency_ai.main
```

## Verificar que está funcionando

Una vez iniciado, el servidor estará disponible en:

- **API Base:** http://localhost:8001
- **Documentación:** http://localhost:8001/docs
- **Health Check:** http://localhost:8001/api/v1/health

### Probar con curl:

```bash
# Health check
curl http://localhost:8001/api/v1/health

# Listar embeddings
curl http://localhost:8001/api/v1/embeddings

# Info del modelo
curl http://localhost:8001/api/v1/model/info
```

## Generar tu primer Safe Tensor

### Desde la API:

```bash
curl -X POST "http://localhost:8001/api/v1/generate" \
  -F "images=@image1.jpg" \
  -F "character_name=MyCharacter" \
  -F "save_tensor=true"
```

### Desde Python:

```python
from character_consistency_ai.core.character_consistency_service import CharacterConsistencyService

service = CharacterConsistencyService()
service.initialize_model()

result = service.generate_character_embedding(
    images=["image1.jpg"],
    character_name="MyCharacter",
    save_tensor=True,
)

print(f"Safe tensor: {result['saved_path']}")
```

### Desde línea de comandos:

```bash
python generate_safe_tensors.py generate image1.jpg --name MyCharacter
```

## Configuración

El servidor usa estas configuraciones por defecto:

- **Host:** 0.0.0.0
- **Port:** 8001
- **Output Dir:** ./character_embeddings
- **Model:** black-forest-labs/flux2-dev

Puedes cambiarlas con variables de entorno:

```bash
set CHARACTER_CONSISTENCY_API_PORT=8080
set CHARACTER_CONSISTENCY_OUTPUT_DIR=./my_embeddings
python run_server.py
```

## Próximos Pasos

1. ✅ Servidor iniciado
2. 📸 Sube imágenes para generar safe tensors
3. 💾 Usa los safe tensors en tus workflows
4. 📚 Consulta `GUIA_SAFE_TENSORS.md` para más detalles

## Troubleshooting

### Error: "Module not found"
```bash
pip install -r requirements.txt
```

### Error: "Port already in use"
Cambia el puerto:
```bash
set CHARACTER_CONSISTENCY_API_PORT=8002
python run_server.py
```

### Error: "CUDA out of memory"
Usa CPU o reduce el batch size:
```bash
set CHARACTER_CONSISTENCY_DEVICE=cpu
python run_server.py
```


