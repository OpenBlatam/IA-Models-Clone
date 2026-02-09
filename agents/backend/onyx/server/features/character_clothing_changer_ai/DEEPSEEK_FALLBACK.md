# 🤖 DeepSeek Fallback Mode

## 📋 Descripción

El sistema ahora incluye un **modo de fallback automático** que usa DeepSeek cuando el modelo Flux2 no está disponible. Esto permite que el sistema funcione incluso si no puedes configurar el token de HuggingFace o si Flux2 falla por cualquier razón.

## 🔄 Cómo Funciona

### Modo Normal (Flux2)
1. El sistema intenta cargar el modelo Flux2
2. Si tiene éxito, usa Flux2 para todas las operaciones
3. Genera tensors para ComfyUI

### Modo Fallback (DeepSeek)
1. Si Flux2 falla al cargar, automáticamente cambia a DeepSeek
2. Usa DeepSeek API para mejorar prompts
3. Aplica transformaciones a la imagen original basadas en el prompt mejorado
4. Retorna la imagen modificada

## ✨ Características del Modo DeepSeek

### Mejora de Prompts
- **Automático**: DeepSeek mejora automáticamente tus descripciones de ropa
- **Inteligente**: Crea prompts más detallados y optimizados
- **Contextual**: Considera el nombre del personaje y contexto

### Transformaciones de Imagen
- **Ajustes de color**: Basados en la descripción (rojo, azul, negro, blanco, etc.)
- **Ajustes de estilo**: Elegante (sharpening) o casual (blur suave)
- **Blending**: Mezcla con la imagen original según el parámetro `strength`

### API Key Configurada
- **Key por defecto**: `sk-753365753f074509bb52496e038691f6`
- **Configurable**: Usa variable de entorno `DEEPSEEK_API_KEY` para cambiar

## 🚀 Uso

### Automático
El sistema detecta automáticamente si Flux2 falla y cambia a DeepSeek:

```python
# Si Flux2 falla, automáticamente usa DeepSeek
service = ClothingChangerService()
service.initialize_model()  # Intenta Flux2, si falla usa DeepSeek

result = service.change_clothing(
    image="character.jpg",
    clothing_description="a red elegant dress",
)
```

### Verificar Modo Actual
```python
info = service.get_model_info()
if info.get("fallback_mode"):
    print("Usando DeepSeek como fallback")
else:
    print("Usando Flux2")
```

### Desde la API
```bash
# Health check muestra el modo actual
curl http://localhost:8002/api/v1/health

# Respuesta:
# {
#   "status": "healthy",
#   "model_initialized": true,
#   "using_deepseek_fallback": true,
#   "model_type": "DeepSeek"
# }
```

## 📊 Diferencias entre Modos

| Característica | Flux2 | DeepSeek Fallback |
|---------------|-------|-------------------|
| **Generación de Imágenes** | ✅ Completa | ⚠️ Transformaciones básicas |
| **Tensors ComfyUI** | ✅ Sí | ❌ No |
| **Mejora de Prompts** | ⚠️ Básica | ✅ Avanzada (DeepSeek API) |
| **Velocidad** | 🐌 Lenta (modelo grande) | ⚡ Rápida (API) |
| **Requisitos** | Token HuggingFace | API Key DeepSeek |
| **Calidad** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ (mejorable) |

## 🔧 Configuración

### API Key de DeepSeek

**Opción 1: Usar la key por defecto**
Ya está configurada en el código, no necesitas hacer nada.

**Opción 2: Configurar tu propia key**
```bash
# Windows
set DEEPSEEK_API_KEY=tu_key_aqui

# Linux/Mac
export DEEPSEEK_API_KEY=tu_key_aqui
```

### Mejorar Calidad (Opcional)

Para mejor calidad en modo DeepSeek, puedes configurar APIs de generación de imágenes:

**Stability AI:**
```bash
set STABILITY_AI_API_KEY=tu_key
```

**Replicate:**
```bash
set REPLICATE_API_TOKEN=tu_token
```

## 💡 Ventajas del Modo DeepSeek

1. **✅ Funciona inmediatamente**: No necesitas configurar HuggingFace
2. **⚡ Más rápido**: No necesita descargar modelos grandes
3. **🤖 Prompts mejorados**: DeepSeek optimiza tus descripciones
4. **💾 Menos memoria**: No carga modelos pesados en memoria
5. **🌐 Basado en API**: No requiere GPU potente

## ⚠️ Limitaciones

1. **Calidad de imagen**: Las transformaciones son básicas comparadas con Flux2
2. **No genera tensors**: No puede crear archivos .safetensors para ComfyUI
3. **Dependencia de internet**: Requiere conexión para usar DeepSeek API
4. **Transformaciones simples**: No hace inpainting real, solo ajustes de color/estilo

## 🎯 Cuándo Usar Cada Modo

### Usa Flux2 cuando:
- ✅ Necesitas la mejor calidad de imagen
- ✅ Quieres generar tensors para ComfyUI
- ✅ Tienes GPU potente y tiempo
- ✅ Puedes configurar el token de HuggingFace

### Usa DeepSeek cuando:
- ✅ Flux2 no está disponible
- ✅ Necesitas resultados rápidos
- ✅ No tienes GPU potente
- ✅ Solo necesitas mejoras básicas de ropa
- ✅ Quieres prompts mejorados automáticamente

## 🔍 Detección Automática

El sistema detecta automáticamente qué modo usar:

```python
# El servicio intenta Flux2 primero
service.initialize_model()

# Si Flux2 falla con este error:
# "Cannot load model black-forest-labs/flux2-dev: model is not cached locally"
# 
# Automáticamente cambia a DeepSeek y continúa funcionando
```

## 📝 Logs

El sistema registra qué modo está usando:

```
INFO: Initializing Flux2 Clothing Changer Model...
ERROR: Error initializing Flux2 model: Failed to load pipeline...
INFO: Falling back to DeepSeek model...
INFO: DeepSeek model initialized as fallback
```

## 🎨 Ejemplo de Uso

```python
from character_clothing_changer_ai.core.clothing_changer_service import ClothingChangerService

service = ClothingChangerService()

# Inicializa (intenta Flux2, si falla usa DeepSeek)
service.initialize_model()

# Verifica qué modo está usando
info = service.get_model_info()
print(f"Modo: {info.get('primary_model')}")
print(f"Fallback: {info.get('fallback_mode')}")

# Usa normalmente - funciona con ambos modos
result = service.change_clothing(
    image="character.jpg",
    clothing_description="a red elegant dress",
    character_name="MyCharacter",
)

print(f"Imagen procesada: {result.get('changed')}")
```

## ✅ Estado Actual

- ✅ **Fallback automático** implementado
- ✅ **Mejora de prompts** con DeepSeek
- ✅ **Transformaciones básicas** de imagen
- ✅ **Detección automática** del modo
- ✅ **API Key configurada** por defecto
- ⚠️ **Generación de imágenes real** requiere APIs adicionales (opcional)

## 🚀 Próximos Pasos

Para mejorar la calidad en modo DeepSeek:

1. **Configurar Stability AI** para generación real de imágenes
2. **Configurar Replicate** como alternativa
3. **Usar DALL-E API** si está disponible
4. **Implementar inpainting** usando APIs externas

## 📚 Referencias

- **DeepSeek API**: https://api.deepseek.com
- **API Key**: Ya configurada en el código
- **Documentación**: Ver código en `models/deepseek_clothing_model.py`


