# Notifications System

## 📋 Descripción

Sistema de notificaciones para el ecosistema Blatam Academy.

## 🚀 Características Principales

- **Sistema de Notificaciones**: API para gestión de notificaciones
- **Integración**: Integración con otros módulos del sistema

## 📁 Estructura

```
notifications/
└── api.py                 # API de notificaciones
```

## 🔧 Instalación

Este módulo requiere las dependencias del sistema principal.

## 💻 Uso

```python
from notifications.api import NotificationService

# Inicializar servicio
service = NotificationService()

# Enviar notificación
service.send(
    user_id=123,
    message="Nueva actualización disponible",
    type="info"
)
```

## 🔗 Integración

Este módulo se integra con:
- Todos los módulos que requieren notificaciones
- **Integration System**: Para orquestación

