# Changelog

## [1.1.0] - Mejoras Avanzadas

### ✨ Nuevas Características

- **Detección Inteligente Mejorada**
  - Soporte para 11 tipos diferentes de IA
  - Detección automática de características avanzadas
  - Detección de proveedores de modelos
  - Detección de complejidad del proyecto

- **API Mejorada**
  - Validación de entrada mejorada
  - Nuevos endpoints: `/api/v1/stats`, `/api/v1/projects`
  - Endpoint para eliminar proyectos: `DELETE /api/v1/project/{id}`
  - Filtros y paginación en listado de proyectos

- **Generación de Código Inteligente**
  - Genera WebSocket automáticamente si es necesario
  - Genera endpoints de file upload si se detecta
  - Dependencias automáticas según características
  - Configuración adaptativa según tipo de IA

- **Mejoras en Backend**
  - Soporte condicional para WebSocket
  - Soporte condicional para file upload
  - Dependencias dinámicas en requirements.txt
  - Configuración mejorada según keywords

### 🔧 Mejoras

- Mejor manejo de errores
- Validación de entrada más robusta
- Estadísticas y métricas del generador
- Mejor documentación generada

### 📊 Estadísticas

- Tasa de éxito de proyectos
- Tiempo promedio de procesamiento
- Contadores de proyectos procesados/completados/fallidos

## [1.0.0] - Versión Inicial

- Generación básica de proyectos
- Backend FastAPI
- Frontend React
- Sistema de cola básico
- API REST básica


