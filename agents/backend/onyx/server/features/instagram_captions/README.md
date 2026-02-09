# Instagram Captions Generation System

## 📋 Descripción

Sistema para generación de captions para Instagram con arquitectura modular y optimizaciones avanzadas.

## 🚀 Características Principales

- **Generación de Captions**: Sistema completo para crear captions atractivos para Instagram
- **Arquitectura Modular**: Diseño modular y escalable
- **Seguridad**: Principios de seguridad integrados
- **Configuración Flexible**: Sistema de configuración adaptable

## 📁 Estructura

```
instagram_captions/
├── config/                 # Configuraciones
├── current/                # Versión actual
├── demos/                  # Demostraciones
├── docs/                   # Documentación
├── legacy/                 # Código legado
└── utils/                  # Utilidades
```

## 🔧 Instalación

Las dependencias se instalan con el sistema principal.

## 💻 Uso

```python
from instagram_captions.current import InstagramCaptionGenerator

# Inicializar generador
generator = InstagramCaptionGenerator()

# Generar caption
caption = generator.generate(
    image_description="Foto de producto",
    style="moderno",
    hashtags=True
)
```

## 🔗 Integración

Este módulo se integra con:
- **Blatam AI**: Motor de IA
- **Integration System**: Para orquestación
- **Export IA**: Para exportación



