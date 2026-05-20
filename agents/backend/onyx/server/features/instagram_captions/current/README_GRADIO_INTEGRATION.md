# Gradio Integration System

Sistema completo de integración con Gradio implementando demos interactivos, interfaces amigables, manejo de errores y validación de entrada.

## Características

- **Demos Interactivos**: Interfaces completas para inferencia de modelos
- **Interfaces Amigables**: Diseño UX optimizado con temas y CSS personalizado
- **Manejo de Errores**: Sistema robusto de manejo y recuperación de errores
- **Validación de Entrada**: Validación automática de todos los inputs
- **Múltiples Tipos de Modelos**: Texto, imagen, comparación de modelos
- **Builder Pattern**: Construcción modular de interfaces
- **Temas Personalizables**: Soporte para temas Gradio y CSS custom
- **Logging Completo**: Sistema de logging para debugging y monitoreo

## Instalación

```bash
pip install -r requirements_gradio_integration.txt
```

## Uso Básico

### Interfaz de Generación de Texto

```python
from gradio_integration_system import TextGenerationInterface

# Crear interfaz para modelo de texto
text_interface = TextGenerationInterface(
    model=your_text_model,
    tokenizer=your_tokenizer,
    device="cuda"
)

# Lanzar interfaz
text_interface.create_interface().launch(
    server_name="0.0.0.0",
    server_port=7860
)
```

### Interfaz de Generación de Imágenes

```python
from gradio_integration_system import ImageGenerationInterface

# Crear interfaz para modelo de imagen
image_interface = ImageGenerationInterface(
    model=your_image_model,
    device="cuda"
)

# Lanzar interfaz
image_interface.create_interface().launch(
    server_name="0.0.0.0",
    server_port=7861
)
```

### Comparación de Modelos

```python
from gradio_integration_system import ModelComparisonInterface

# Crear interfaz de comparación
models = {
    "Model A": model_a,
    "Model B": model_b,
    "Model C": model_c
}

comparison_interface = ModelComparisonInterface(
    models=models,
    device="cuda"
)

# Lanzar interfaz
comparison_interface.create_interface().launch(
    server_name="0.0.0.0",
    server_port=7862
)
```

## Componentes Principales

### GradioInterfaceBuilder

```python
from gradio_integration_system import GradioInterfaceBuilder

# Construir interfaz personalizada
builder = GradioInterfaceBuilder(
    title="Mi Demo AI",
    description="Descripción de la interfaz"
)

# Agregar componentes
builder.add_component("textbox", label="Input", lines=3)
builder.add_component("slider", minimum=0, maximum=100, value=50)
builder.add_component("dropdown", choices=["A", "B", "C"])

# Configurar CSS personalizado
builder.set_custom_css("""
    .gradio-container {
        max-width: 1200px !important;
        margin: auto !important;
    }
""")

# Construir interfaz
interface = builder.build()
```

### TextGenerationInterface

```python
# Interfaz completa para generación de texto
text_interface = TextGenerationInterface(
    model=your_model,
    tokenizer=your_tokenizer
)

# Configurar parámetros de generación
interface = text_interface.create_interface()

# Lanzar con configuración personalizada
interface.launch(
    server_name="0.0.0.0",
    server_port=7860,
    share=True,  # Crear enlace público
    debug=True,   # Modo debug
)
```

### ImageGenerationInterface

```python
# Interfaz para generación de imágenes
image_interface = ImageGenerationInterface(
    model=your_diffusion_model
)

# Lanzar interfaz
interface = image_interface.create_interface()
interface.launch(
    server_name="0.0.0.0",
    server_port=7861
)
```

## Validación de Entrada

### Validación Automática

```python
from gradio_integration_system import InputValidator

# Validar texto
is_valid, message = InputValidator.validate_text_input(
    text="Mi prompt",
    min_length=5,
    max_length=500
)

# Validar número
is_valid, message = InputValidator.validate_numeric_input(
    value=0.7,
    min_val=0.1,
    max_val=2.0
)

# Validar imagen
is_valid, message = InputValidator.validate_image_input(
    image=pil_image,
    max_size=(1024, 1024)
)
```

### Reglas de Validación Personalizadas

```python
from gradio_integration_system import ErrorHandler

# Definir reglas de validación
validation_rules = {
    "required": True,
    "min_length": 10,
    "max_length": 1000,
    "pattern": r"^[a-zA-Z0-9\s]+$"
}

# Validar input
is_valid, message = ErrorHandler.handle_input_validation(
    value="Mi texto de entrada",
    validation_rules=validation_rules
)
```

## Manejo de Errores

### Manejo Automático de Errores

```python
# El sistema maneja errores automáticamente
try:
    result = interface.process_input(user_input)
except Exception as e:
    # Error manejado automáticamente
    error_output = ErrorHandler.create_error_output(e, "text generation")
```

### Logging de Errores

```python
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Los errores se logean automáticamente
logger.info("Interfaz iniciada correctamente")
logger.error("Error en procesamiento de input")
```

## Personalización de UI

### Temas Personalizados

```python
import gradio as gr

# Usar tema personalizado
custom_theme = gr.themes.Soft().set(
    primary_hue="blue",
    secondary_hue="gray",
    neutral_hue="slate"
)

# Aplicar tema
interface = gr.Interface(
    fn=your_function,
    inputs=inputs,
    outputs=outputs,
    theme=custom_theme
)
```

### CSS Personalizado

```python
# CSS personalizado para mejor UX
custom_css = """
    .gradio-container {
        max-width: 1400px !important;
        margin: auto !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    .input-container {
        border-radius: 12px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        padding: 20px;
        margin: 15px 0;
    }
    
    .output-container {
        border-radius: 12px;
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
        padding: 20px;
        margin: 15px 0;
    }
    
    .status-success {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 8px;
        padding: 10px;
        color: #155724;
    }
    
    .status-error {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 8px;
        padding: 10px;
        color: #721c24;
    }
"""

# Aplicar CSS
interface = gr.Interface(
    fn=your_function,
    inputs=inputs,
    outputs=outputs,
    css=custom_css
)
```

## Configuraciones Avanzadas

### Múltiples Puertos

```python
# Lanzar múltiples interfaces en puertos diferentes
text_interface.create_interface().launch(server_port=7860)
image_interface.create_interface().launch(server_port=7861)
comparison_interface.create_interface().launch(server_port=7862)
```

### Configuración de Servidor

```python
# Configuración completa del servidor
interface.launch(
    server_name="0.0.0.0",      # Escuchar en todas las interfaces
    server_port=7860,            # Puerto específico
    share=True,                  # Crear enlace público
    debug=True,                  # Modo debug
    show_error=True,             # Mostrar errores
    quiet=False,                 # Logging verbose
    inbrowser=True,              # Abrir en navegador
    auth=("user", "pass"),      # Autenticación básica
    auth_message="Login required" # Mensaje de login
)
```

### Manejo de Eventos

```python
# Agregar manejadores de eventos personalizados
def on_input_change(value):
    print(f"Input changed: {value}")

def on_output_generated(value):
    print(f"Output generated: {value}")

# Aplicar manejadores
interface.load(on_input_change, inputs="textbox")
interface.load(on_output_generated, outputs="textbox")
```

## Ejemplos de Uso

### Demo Completo de Texto

```python
from gradio_integration_system import TextGenerationInterface
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Cargar modelo y tokenizer
model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Crear interfaz
text_interface = TextGenerationInterface(
    model=model,
    tokenizer=tokenizer,
    device="cuda" if torch.cuda.is_available() else "cpu"
)

# Lanzar interfaz
interface = text_interface.create_interface()
interface.launch(
    server_name="0.0.0.0",
    server_port=7860,
    share=True
)
```

### Demo de Comparación de Modelos

```python
from gradio_integration_system import ModelComparisonInterface

# Modelos para comparar
models = {
    "GPT-2 Small": gpt2_small,
    "GPT-2 Medium": gpt2_medium,
    "GPT-2 Large": gpt2_large
}

# Crear interfaz de comparación
comparison_interface = ModelComparisonInterface(
    models=models,
    device="cuda"
)

# Lanzar interfaz
interface = comparison_interface.create_interface()
interface.launch(
    server_name="0.0.0.0",
    server_port=7861
)
```

## Estructura del Proyecto

```
gradio_integration_system.py     # Sistema principal
requirements_gradio_integration.txt  # Dependencias
README_GRADIO_INTEGRATION.md     # Documentación
```

## Dependencias Principales

- **gradio**: Framework de interfaces web
- **torch**: Framework de deep learning
- **Pillow**: Procesamiento de imágenes
- **matplotlib/seaborn**: Visualización
- **numpy**: Computación numérica

## Requisitos del Sistema

- **Python**: 3.8+
- **RAM**: Mínimo 4GB, recomendado 8GB+
- **GPU**: Opcional, para modelos grandes

## Solución de Problemas

### Error de Puerto en Uso

```python
# Cambiar puerto
interface.launch(server_port=7861)

# O liberar puerto
import socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
```

### Error de Memoria

```python
# Reducir batch size
# Usar CPU en lugar de GPU
device = "cpu"

# Limpiar memoria
import gc
gc.collect()
torch.cuda.empty_cache()
```

### Error de Gradio

```python
# Actualizar Gradio
pip install --upgrade gradio

# Verificar versión
import gradio as gr
print(gr.__version__)
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


