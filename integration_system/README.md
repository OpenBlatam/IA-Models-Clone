# Integration System

## 📋 Descripción

Sistema principal de integración y API Gateway para Blatam Academy. Orquesta todos los servicios y actúa como punto de entrada único.

## 🚀 Características Principales

- **API Gateway**: Puerta de enlace para todos los servicios
- **Orquestación**: Orquesta todos los servicios del sistema
- **Enrutamiento**: Enrutamiento inteligente de peticiones
- **Autenticación**: Autenticación centralizada
- **Monitoreo**: Monitoreo del sistema completo

## 📁 Estructura

```
integration_system/
├── api/                    # Endpoints de API
├── config/                 # Configuraciones
├── core/                   # Lógica central
└── main.py                 # Punto de entrada
```

## 🔧 Instalación

Este módulo se instala con el sistema principal usando `start_system.py`.

## 💻 Uso

El sistema se inicia automáticamente con:

```bash
python start_system.py start
```

## 📊 Puerto

- Puerto por defecto: **8000**
- Health endpoint: `http://localhost:8000/health`
- API Docs: `http://localhost:8000/docs`

## 🔗 Integración

Este sistema orquesta:
- Todos los servicios del ecosistema Blatam Academy
- **Content Redundancy Detector** (puerto 8001)
- **BUL** (puerto 8002)
- **Gamma App** (puerto 8003)
- **Business Agents** (puerto 8004)
- **Export IA** (puerto 8005)

