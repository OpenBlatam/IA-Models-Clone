# Diffusion Pipeline System

Sistema de pipelines de difusión usando la librería Diffusers, implementando StableDiffusionPipeline y StableDiffusionXLPipeline.

## Características

- **Stable Diffusion 1.5**: Pipeline optimizado para generación de imágenes 512x512
- **Stable Diffusion XL**: Pipeline para imágenes de alta resolución 1024x1024
- **Optimizaciones de memoria**: Attention slicing, VAE slicing, CPU offload
- **Múltiples schedulers**: DPM, Euler, DDIM, PNDM
- **Generación por lotes**: Procesamiento eficiente de múltiples prompts
- **Soporte LoRA**: Carga y descarga dinámica de pesos LoRA
- **Control de semillas**: Reproducibilidad en generación
- **Manejo de errores**: Logging estructurado y recuperación de errores

## Instalación

```bash
pip install -r requirements_diffusion_pipeline.txt
```

## Uso Básico

### Inicialización

```python
from diffusion_pipeline_system import DiffusionPipelineSystem

# Inicializar con optimizaciones por defecto
pipeline_system = DiffusionPipelineSystem(
    enable_attention_slicing=True,
    enable_vae_slicing=True
)
```

### Generación de Imágenes

```python
# Generar con SD 1.5
sd_images = pipeline_system.generate_image_sd(
    prompt="Un gato sentado en el alféizar",
    num_inference_steps=30,
    guidance_scale=7.5,
    seed=42
)

# Generar con SDXL
sdxl_images = pipeline_system.generate_image_sdxl(
    prompt="Un paisaje futurista de noche",
    num_inference_steps=40,
    guidance_scale=8.0,
    width=1024,
    height=1024
)
```

### Generación por Lotes

```python
prompts = [
    "Un gato en el jardín",
    "Un robot en el espacio",
    "Una nave espacial en órbita"
]

batch_results = pipeline_system.batch_generate(
    prompts=prompts,
    pipeline_type="sd",
    num_inference_steps=25
)
```

### Uso con LoRA

```python
# Generar con modelo LoRA fine-tuned
lora_images = pipeline_system.generate_with_lora(
    prompt="Un retrato en estilo anime",
    lora_path="./models/anime_style_lora.safetensors",
    num_inference_steps=35
)
```

## Configuración

### Parámetros de Inicialización

- `model_id`: ID del modelo SD 1.5 (default: "runwayml/stable-diffusion-v1-5")
- `xl_model_id`: ID del modelo SDXL (default: "stabilityai/stable-diffusion-xl-base-1.0")
- `device`: Dispositivo de cómputo ("cuda" o "cpu")
- `torch_dtype`: Tipo de datos PyTorch (default: torch.float16)
- `enable_attention_slicing`: Habilitar división de atención para ahorrar memoria
- `enable_vae_slicing`: Habilitar división VAE para ahorrar memoria
- `enable_model_cpu_offload`: Descargar modelos a CPU cuando no se usen
- `enable_sequential_cpu_offload`: Descarga secuencial a CPU

### Parámetros de Generación

- `prompt`: Descripción de la imagen a generar
- `negative_prompt`: Descripción de lo que NO debe aparecer
- `num_inference_steps`: Número de pasos de denoising (20-100)
- `guidance_scale`: Fuerza de guía del prompt (1.0-20.0)
- `width/height`: Dimensiones de la imagen
- `num_images`: Número de imágenes a generar
- `seed`: Semilla para reproducibilidad
- `scheduler`: Tipo de scheduler ("dpm", "euler", "ddim", "pndm")

## Schedulers Disponibles

- **DPM**: DPMSolverMultistepScheduler - Rápido y eficiente
- **Euler**: EulerDiscreteScheduler - Balance entre velocidad y calidad
- **DDIM**: DDIMScheduler - Determinístico, bueno para control
- **PNDM**: PNDMScheduler - Estable pero más lento

## Optimizaciones de Memoria

### Attention Slicing
```python
pipeline_system = DiffusionPipelineSystem(
    enable_attention_slicing=True  # Reduce uso de memoria GPU
)
```

### VAE Slicing
```python
pipeline_system = DiffusionPipelineSystem(
    enable_vae_slicing=True  # Procesa VAE en chunks
)
```

### CPU Offload
```python
pipeline_system = DiffusionPipelineSystem(
    enable_model_cpu_offload=True,  # Descarga modelos no usados a CPU
    enable_sequential_cpu_offload=True  # Descarga secuencial
)
```

## Manejo de Errores

El sistema incluye manejo robusto de errores:

```python
try:
    images = pipeline_system.generate_image_sd(prompt="Mi prompt")
except RuntimeError as e:
    print(f"Error de GPU: {e}")
except Exception as e:
    print(f"Error general: {e}")
```

## Logging y Monitoreo

```python
# Obtener información del sistema
system_info = pipeline_system.get_system_info()
print(f"Memoria GPU: {system_info['cuda_memory_allocated']} bytes")

# Guardar configuración
pipeline_system.save_pipeline_config("pipeline_config.json")
```

## Limpieza de Recursos

```python
# Limpiar memoria y recursos
pipeline_system.cleanup()
```

## Ejemplos Avanzados

### Generación con Scheduler Personalizado

```python
# Usar scheduler DPM para generación rápida
images = pipeline_system.generate_image_sd(
    prompt="Un paisaje de montañas",
    scheduler="dpm",
    num_inference_steps=20
)
```

### Generación con Semilla Específica

```python
# Generar imagen reproducible
images = pipeline_system.generate_image_sd(
    prompt="Un retrato de mujer",
    seed=12345,
    num_inference_steps=50
)
```

### Generación de Múltiples Imágenes

```python
# Generar 4 variaciones
images = pipeline_system.generate_image_sd(
    prompt="Un castillo en la colina",
    num_images=4,
    guidance_scale=8.5
)
```

## Estructura del Proyecto

```
diffusion_pipeline_system.py     # Sistema principal
requirements_diffusion_pipeline.txt  # Dependencias
README_DIFFUSION_PIPELINE.md     # Documentación
```

## Dependencias Principales

- **torch**: Framework de deep learning
- **diffusers**: Librería de modelos de difusión
- **transformers**: Modelos de lenguaje y tokenizers
- **accelerate**: Aceleración de modelos
- **Pillow**: Procesamiento de imágenes
- **numpy**: Computación numérica

## Requisitos del Sistema

- **GPU**: NVIDIA con CUDA 11.8+ (recomendado)
- **RAM**: Mínimo 8GB, recomendado 16GB+
- **VRAM**: Mínimo 4GB, recomendado 8GB+
- **Python**: 3.8+

## Solución de Problemas

### Error de Memoria GPU
```python
# Reducir batch size y habilitar optimizaciones
pipeline_system = DiffusionPipelineSystem(
    enable_attention_slicing=True,
    enable_vae_slicing=True,
    enable_model_cpu_offload=True
)
```

### Error de Modelo no Encontrado
```python
# Verificar IDs de modelo
pipeline_system = DiffusionPipelineSystem(
    model_id="runwayml/stable-diffusion-v1-5",
    xl_model_id="stabilityai/stable-diffusion-xl-base-1.0"
)
```

### Lento Rendimiento
```python
# Usar scheduler más rápido
images = pipeline_system.generate_image_sd(
    prompt="Mi prompt",
    scheduler="dpm",
    num_inference_steps=20
)
```

## Contribución

Para contribuir al sistema:

1. Fork del repositorio
2. Crear rama feature
3. Implementar cambios
4. Ejecutar tests
5. Crear Pull Request

## Licencia

Este proyecto está bajo la licencia MIT.

## Contacto

Para soporte técnico o preguntas, abrir un issue en el repositorio.


