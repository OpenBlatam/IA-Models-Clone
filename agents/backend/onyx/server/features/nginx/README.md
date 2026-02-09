# Nginx Configuration

## 📋 Descripción

Configuración de Nginx para el sistema Blatam Academy, incluyendo proxy reverso, balanceador de carga, y configuraciones de producción.

## 🚀 Características Principales

- **Proxy Reverso**: Configuración de proxy reverso
- **Balanceador de Carga**: Balanceo de carga entre servicios
- **SSL/TLS**: Configuración de seguridad
- **Cache**: Configuración de caché

## 📁 Estructura

```
nginx/
└── nginx.conf              # Configuración principal de Nginx
```

## 🔧 Configuración

La configuración se encuentra en `nginx.conf` y se carga automáticamente por Docker Compose.

## 🔗 Integración

Este módulo se usa con:
- **Integration System**: Para enrutamiento
- **Docker Compose**: Para orquestación
- Todos los servicios del sistema



