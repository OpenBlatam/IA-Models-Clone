# Guía de Inicio Rápido

## Instalación

```bash
cd analizador_de_documentos
pip install -r requirements.txt
```

## Iniciar Servidor

```bash
python main.py
# o
python start.py
```

## Probar la API

1. Abre tu navegador en: `http://localhost:8000/docs`
2. Prueba el endpoint `/api/analizador-documentos/health`

## Ejemplo Básico

```python
import requests

# Analizar texto
response = requests.post(
    "http://localhost:8000/api/analizador-documentos/analyze",
    json={
        "document_content": "Este es un documento sobre inteligencia artificial...",
        "tasks": ["classification", "summarization"]
    }
)

print(response.json())
```

## Entrenar Modelo Personalizado

```bash
# Crear datos de ejemplo
python training/train_model.py --create-sample --data datos.json

# Entrenar modelo
python training/train_model.py --data datos.json --num-labels 3 --epochs 5
```

¡Listo para usar! 🚀
















