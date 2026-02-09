# Input Prompt Management

## 📋 Descripción

Sistema para gestión de prompts de entrada con modelos, esquemas, servicios y API RESTful.

## 🚀 Características Principales

- **Gestión de Prompts**: Creación y gestión de prompts de entrada
- **Modelos de Datos**: Modelos bien definidos
- **Esquemas Pydantic**: Validación de datos
- **Servicios**: Servicios de negocio
- **API RESTful**: Interfaz API para integración

## 📁 Estructura

```
input_prompt/
├── models.py              # Modelos de datos
├── schemas.py            # Esquemas Pydantic
├── service.py            # Servicios de negocio
└── api.py                # Endpoints de API
```

## 🔧 Instalación

Este módulo requiere las dependencias del sistema principal. No requiere instalación separada.

## 💻 Uso Básico

```python
from input_prompt.service import InputPromptService
from input_prompt.schemas import InputPromptCreate

# Inicializar servicio
service = InputPromptService()

# Crear prompt
prompt = service.create(InputPromptCreate(
    name="Prompt de ejemplo",
    content="Genera contenido sobre...",
    category="marketing"
))
```

## 🔗 Integración

Este módulo se integra con:
- **Integration System**: Para orquestación
- **AI Document Processor**: Para procesamiento con IA
- **Business Agents**: Para automatización
- Todos los módulos que requieren gestión de prompts



