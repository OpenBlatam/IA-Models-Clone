# Guía de Inicio Rápido - Dermatology AI

## 🚀 Inicio Rápido en 5 Minutos

### 1. Instalación

```bash
cd dermatology_ai
pip install -r requirements.txt
```

### 2. Configuración (Opcional)

```bash
cp .env.example .env
# Editar .env si es necesario
```

### 3. Iniciar Servidor

```bash
python main.py
```

El servidor estará disponible en `http://localhost:8006`

### 4. Probar la API

#### Opción A: Usando curl

```bash
# Analizar imagen
curl -X POST "http://localhost:8006/dermatology/analyze-image" \
  -F "file=@ruta/a/tu/imagen.jpg" \
  -F "enhance=true"

# Obtener recomendaciones
curl -X POST "http://localhost:8006/dermatology/get-recommendations" \
  -F "file=@ruta/a/tu/imagen.jpg" \
  -F "include_routine=true"
```

#### Opción B: Usando Python

```python
import requests

# Analizar imagen
with open("skin_photo.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8006/dermatology/analyze-image",
        files={"file": f},
        data={"enhance": True}
    )
    print(response.json())
```

#### Opción C: Usando la interfaz web

1. Abra `http://localhost:8006/docs` en su navegador
2. Expanda el endpoint `/dermatology/analyze-image`
3. Haga clic en "Try it out"
4. Suba una imagen
5. Haga clic en "Execute"

## 📊 Ejemplo de Respuesta

```json
{
  "success": true,
  "analysis": {
    "quality_scores": {
      "overall_score": 75.5,
      "texture_score": 80.0,
      "hydration_score": 70.0,
      "elasticity_score": 75.0,
      "pigmentation_score": 72.0,
      "pore_size_score": 68.0,
      "wrinkles_score": 80.0,
      "redness_score": 85.0,
      "dark_spots_score": 78.0
    },
    "conditions": [],
    "skin_type": "normal",
    "recommendations_priority": ["hydration", "pore_care"]
  }
}
```

## 🎯 Próximos Pasos

1. **Explorar la documentación**: Visite `/docs` para ver todos los endpoints
2. **Integrar en tu aplicación**: Usa los endpoints en tu frontend o aplicación móvil
3. **Personalizar recomendaciones**: Modifica `services/skincare_recommender.py` para agregar tus productos

## ❓ Problemas Comunes

### Error: "No module named 'cv2'"
```bash
pip install opencv-python
```

### Error: "No module named 'scipy'"
```bash
pip install scipy
```

### El servidor no inicia
- Verifique que el puerto 8006 no esté en uso
- Cambie el puerto en `.env` o `main.py`

## 📚 Más Información

- **Documentación completa**: Ver `README.md`
- **API Reference**: `http://localhost:8006/docs`
- **Código fuente**: Explora los módulos en `core/` y `services/`






