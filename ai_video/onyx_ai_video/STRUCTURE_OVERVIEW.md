# Onyx AI Video System - Structure Overview

## 🏗️ Modular Architecture

El sistema Onyx AI Video ha sido reorganizado en una estructura modular altamente organizada y escalable. Esta nueva estructura mejora significativamente la mantenibilidad, extensibilidad y organización del código.

## 📁 Estructura de Directorios

```
onyx_ai_video/
├── 📁 core/                    # Componentes principales del sistema
│   ├── 🔧 integration.py      # Gestor de integración con Onyx
│   ├── ⚠️ exceptions.py       # Excepciones personalizadas
│   └── 📊 models.py          # Modelos de datos Pydantic
│
├── 📁 workflows/              # Flujos de trabajo de generación de video
│   ├── 🎬 video_workflow.py   # Flujo principal de video
│   └── 🔄 workflow_manager.py # Gestión de flujos de trabajo
│
├── 📁 plugins/                # Sistema de plugins
│   ├── 🔌 plugin_manager.py   # Gestión de plugins
│   └── 🏗️ plugin_base.py      # Clases base para plugins
│
├── 📁 config/                 # Gestión de configuración
│   ├── ⚙️ config_manager.py   # Gestor de configuración
│   └── 🔧 settings.py         # Configuración específica de Onyx
│
├── 📁 utils/                  # Utilidades del sistema
│   ├── 📝 logger.py          # Utilidades de logging
│   ├── 📈 performance.py     # Monitoreo de rendimiento
│   └── 🔒 security.py        # Utilidades de seguridad
│
├── 📁 api/                   # Endpoints de API
│   ├── 🚀 main.py           # Sistema principal de API
│   └── 🌐 endpoints.py      # Endpoints REST
│
├── 📁 cli/                   # Interfaz de línea de comandos
│   └── 💻 main.py           # Implementación del CLI
│
├── 📁 examples/              # Ejemplos de uso
│   └── 🎯 basic_usage.py    # Ejemplos básicos de uso
│
├── 📁 docs/                  # Documentación
├── 📁 tests/                 # Suite de pruebas
│
├── 📄 __init__.py           # Paquete principal
├── 📄 README.md             # Documentación principal
├── 📄 requirements.txt      # Dependencias
├── 📄 install.py            # Script de instalación
└── 📄 STRUCTURE_OVERVIEW.md # Este archivo
```

## 🔧 Módulos Principales

### 📁 Core - Componentes Principales

**Propósito**: Contiene los componentes fundamentales del sistema.

- **`integration.py`**: Gestor de integración con Onyx que adapta el sistema AI Video para usar las funciones, utilidades e infraestructura existente de Onyx.
- **`exceptions.py`**: Excepciones personalizadas con manejo estructurado de errores e integración con los patrones de Onyx.
- **`models.py`**: Modelos de datos Pydantic para requests, responses, configuración y estado del sistema.

### 📁 Workflows - Flujos de Trabajo

**Propósito**: Maneja los flujos de trabajo de generación de video.

- **`video_workflow.py`**: Flujo principal de generación de video con integración completa de Onyx.
- **`workflow_manager.py`**: Gestión y orquestación de flujos de trabajo complejos.

### 📁 Plugins - Sistema de Plugins

**Propósito**: Sistema extensible de plugins para capacidades personalizadas.

- **`plugin_manager.py`**: Gestión completa del ciclo de vida de plugins con integración de Onyx.
- **`plugin_base.py`**: Clases base y contextos para desarrollo de plugins.

### 📁 Config - Gestión de Configuración

**Propósito**: Gestión flexible de configuración del sistema.

- **`config_manager.py`**: Gestor de configuración con soporte para variables de entorno y archivos de configuración.
- **`settings.py`**: Configuración específica para integración con Onyx.

### 📁 Utils - Utilidades

**Propósito**: Utilidades del sistema para logging, rendimiento y seguridad.

- **`logger.py`**: Sistema de logging compatible con Onyx con logging estructurado.
- **`performance.py`**: Monitoreo de rendimiento en tiempo real con métricas y optimización.
- **`security.py`**: Framework de seguridad con encriptación, control de acceso y validación.

### 📁 API - Endpoints de API

**Propósito**: Interfaz de programación para el sistema.

- **`main.py`**: Sistema principal de API con inicialización unificada y gestión de requests.
- **`endpoints.py`**: Endpoints REST para todas las funcionalidades del sistema.

### 📁 CLI - Interfaz de Línea de Comandos

**Propósito**: Interfaz de línea de comandos completa.

- **`main.py`**: CLI completo con gestión del sistema, generación de video y administración de plugins.

### 📁 Examples - Ejemplos de Uso

**Propósito**: Ejemplos prácticos de uso del sistema.

- **`basic_usage.py`**: Ejemplos básicos que demuestran todas las funcionalidades principales.

## 🚀 Beneficios de la Estructura Modular

### 1. **Organización Clara**
- Separación lógica de responsabilidades
- Fácil navegación y comprensión del código
- Estructura predecible y consistente

### 2. **Mantenibilidad Mejorada**
- Módulos independientes y cohesivos
- Cambios localizados sin afectar otros módulos
- Testing más fácil y específico

### 3. **Extensibilidad**
- Sistema de plugins modular
- Fácil adición de nuevas funcionalidades
- Arquitectura abierta para extensiones

### 4. **Reutilización**
- Componentes reutilizables entre módulos
- Utilidades compartidas
- Patrones consistentes

### 5. **Integración con Onyx**
- Integración profunda con la infraestructura de Onyx
- Aprovechamiento de utilidades existentes
- Consistencia con patrones de Onyx

## 🔄 Flujo de Trabajo

### Inicialización del Sistema
```
1. Configuración (config/) → Carga de configuración
2. Logging (utils/) → Configuración de logging
3. Seguridad (utils/) → Inicialización de seguridad
4. Integración Onyx (core/) → Conexión con Onyx
5. Workflows (workflows/) → Inicialización de flujos
6. Plugins (plugins/) → Carga de plugins
7. API (api/) → Inicialización de endpoints
```

### Generación de Video
```
1. API/CLI → Recibe request
2. Seguridad → Valida acceso y entrada
3. Workflow → Procesa video
4. Plugins → Ejecuta plugins necesarios
5. Onyx Integration → Usa servicios de Onyx
6. Performance → Monitorea rendimiento
7. Logging → Registra eventos
8. Response → Retorna resultado
```

## 🛠️ Desarrollo y Extensión

### Agregar Nuevo Plugin
1. Crear archivo en `plugins/`
2. Heredar de `OnyxPluginBase`
3. Implementar método `execute()`
4. Registrar en `plugin_manager.py`

### Agregar Nueva Utilidad
1. Crear archivo en `utils/`
2. Implementar funcionalidad
3. Exportar en `__init__.py`
4. Documentar uso

### Agregar Nuevo Endpoint
1. Crear en `api/endpoints.py`
2. Implementar lógica
3. Registrar en router
4. Documentar API

## 📊 Métricas y Monitoreo

### Métricas del Sistema
- Rendimiento de generación de video
- Uso de recursos (CPU, memoria, GPU)
- Estadísticas de plugins
- Métricas de caché
- Tasas de error

### Monitoreo en Tiempo Real
- Estado del sistema
- Requests activos
- Rendimiento de componentes
- Alertas de umbral

## 🔒 Seguridad

### Características de Seguridad
- Encriptación de datos sensibles
- Validación de entrada
- Control de acceso por usuario
- Rate limiting
- Limpieza de tokens expirados

### Integración con Onyx
- Uso de utilidades de seguridad de Onyx
- Validación de acceso con Onyx
- Encriptación compatible con Onyx

## 🎯 Próximos Pasos

1. **Migración**: Mover archivos existentes a la nueva estructura
2. **Testing**: Crear tests para cada módulo
3. **Documentación**: Completar documentación de cada módulo
4. **Optimización**: Optimizar rendimiento de cada componente
5. **Integración**: Probar integración completa con Onyx

## 📝 Notas de Implementación

- Todos los módulos siguen patrones consistentes
- Integración completa con Onyx en todos los niveles
- Manejo robusto de errores en cada módulo
- Logging estructurado en todo el sistema
- Configuración flexible y extensible
- Documentación completa de cada componente

Esta estructura modular proporciona una base sólida para el crecimiento y mantenimiento del sistema Onyx AI Video, asegurando que sea escalable, mantenible y fácil de extender. 