# Arquitectura de Contabilidad Mexicana AI SAM3

## Visión General

Este módulo combina la funcionalidad de contabilidad mexicana con la arquitectura SAM3, proporcionando:

- Procesamiento paralelo y continuo
- Integración con OpenRouter para LLM
- Integración con TruthGPT para optimización
- Gestión automática de tareas

## Componentes Principales

### 1. ContadorSAM3Agent

El agente principal que orquesta todos los servicios.

**Responsabilidades:**
- Gestión del ciclo de vida del agente
- Enrutamiento de tareas a servicios específicos
- Integración con OpenRouter y TruthGPT
- Ejecución paralela de tareas

### 2. TaskManager

Gestiona las tareas con cola de prioridades.

**Características:**
- Cola de prioridades
- Persistencia de tareas
- Seguimiento de estado
- Almacenamiento de resultados

### 3. ParallelExecutor

Ejecuta tareas en paralelo con pool de workers.

**Características:**
- Pool de workers configurable
- Cola de tareas asíncrona
- Manejo de errores robusto
- Estadísticas de ejecución

### 4. OpenRouterClient

Cliente para la API de OpenRouter.

**Características:**
- Connection pooling
- Reintentos con backoff exponencial
- Manejo de timeouts
- Soporte para múltiples modelos

### 5. TruthGPTClient

Cliente para integración con TruthGPT.

**Características:**
- Optimización de consultas
- Analytics y monitoreo
- Integración opcional

## Flujo de Procesamiento

```
1. Usuario envía solicitud
   ↓
2. TaskManager crea tarea con prioridad
   ↓
3. ParallelExecutor asigna worker
   ↓
4. ContadorSAM3Agent procesa tarea
   ↓
5. TruthGPT optimiza query (opcional)
   ↓
6. OpenRouter genera respuesta
   ↓
7. Resultado se guarda y retorna
```

## Patrones de Diseño

### 1. Arquitectura SAM3

- Procesamiento continuo 24/7
- Ejecución paralela
- Gestión automática de tareas
- Resiliencia y recuperación de errores

### 2. Single Responsibility

Cada clase tiene una responsabilidad única:
- `PromptBuilder`: Construcción de prompts
- `SystemPromptsBuilder`: Prompts del sistema
- `TaskManager`: Gestión de tareas
- `ParallelExecutor`: Ejecución paralela

### 3. Dependency Injection

La configuración se inyecta en lugar de ser hardcodeada:
- `ContadorSAM3Config` para configuración
- Clientes inyectados en el agente

## Extensibilidad

### Agregar Nuevo Servicio

1. Agregar método en `PromptBuilder`
2. Agregar prompt en `SystemPromptsBuilder`
3. Agregar handler en `ContadorSAM3Agent`
4. Agregar método público en `ContadorSAM3Agent`

### Personalizar Configuración

Modificar `ContadorSAM3Config` o usar variables de entorno.

## Seguridad

- API keys en variables de entorno
- Validación de entrada
- Manejo seguro de errores
- Timeouts y límites de recursos

## Performance

- Connection pooling para HTTP
- Ejecución paralela de tareas
- Caching de resultados (futuro)
- Optimización con TruthGPT
