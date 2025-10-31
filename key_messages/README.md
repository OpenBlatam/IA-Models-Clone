# Key Messages Management

## 📋 Descripción

Sistema completo para gestión de mensajes clave con capacidades de ML, seguridad cibernética, y arquitectura funcional.

## 🚀 Características Principales

- **Gestión de Mensajes Clave**: Sistema completo para mensajes clave
- **Machine Learning**: Integración con ML para análisis
- **Seguridad Cibernética**: Sistema de seguridad avanzado
- **Arquitectura Funcional**: Diseño funcional y modular
- **Reporting**: Sistema de reportes integrado
- **Routers**: Sistema de enrutamiento
- **Tipos**: Tipos bien definidos

## 📁 Estructura

```
key_messages/
├── ml/                    # Machine Learning
├── routers/               # Routers de API
├── types/                 # Tipos y esquemas
├── attackers/             # Sistema de seguridad
├── reporting/             # Sistema de reportes
├── utils/                 # Utilidades
└── docs/                  # Documentación
```

## 🔧 Instalación

```bash
# Instalación mínima
pip install -r requirements-minimal.txt

# Para desarrollo
pip install -r requirements-dev.txt

# Para producción
pip install -r requirements-prod.txt

# Con seguridad cibernética
pip install -r requirements-cyber.txt
```

## 💻 Uso Básico

```python
from key_messages.service import KeyMessagesService
from key_messages.config import Config

# Inicializar servicio
service = KeyMessagesService(Config())

# Crear mensaje clave
message = service.create_key_message(
    content="Mensaje importante",
    category="marketing"
)
```

## 📚 Documentación

- [Project Definition](PROJECT_DEFINITION.md)
- [Migration Summary](MIGRATION_SUMMARY.md)
- [Dependencies](dependencies.md)

## 🔗 Integración

Este módulo se integra con:
- **Integration System**: Para orquestación
- **Business Agents**: Para automatización
- **Export IA**: Para exportación de mensajes
- **Security Systems**: Para seguridad

