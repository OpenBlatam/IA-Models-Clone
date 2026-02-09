# 🚀 Servidor MCP para Mejora de Código - Blatam Academy

## 📋 Descripción

Este servidor MCP (Model Context Protocol) proporciona herramientas avanzadas para analizar, mejorar y refactorizar el código del proyecto Blatam Academy. Incluye 16 herramientas especializadas para diferentes aspectos de la mejora de código.

## 🛠️ Herramientas Disponibles

### 1. **analyze_code_quality**
Analiza la calidad del código detectando problemas de estilo, complejidad ciclomática y violaciones de mejores prácticas.

**Uso:**
```json
{
  "path": "features/suno_clone_ai/core/",
  "language": "python",
  "checks": ["complexity", "security", "performance"]
}
```

### 2. **detect_code_duplication**
Detecta código duplicado en el proyecto, identificando patrones repetitivos.

**Uso:**
```json
{
  "minLines": 5,
  "threshold": 0.8,
  "excludePatterns": ["__pycache__", "node_modules"]
}
```

### 3. **suggest_refactoring**
Sugiere refactorizaciones específicas basadas en patrones identificados.

**Uso:**
```json
{
  "filePath": "features/suno_clone_ai/core/optimizer.py",
  "refactoringType": "extract_method",
  "apply": false
}
```

### 4. **analyze_architecture**
Analiza la arquitectura del proyecto, detectando violaciones de principios SOLID y Clean Architecture.

**Uso:**
```json
{
  "featurePath": "features/facebook_posts",
  "checkCleanArchitecture": true,
  "checkSOLID": true,
  "checkDependencies": true
}
```

### 5. **optimize_performance**
Identifica oportunidades de optimización de performance.

**Uso:**
```json
{
  "filePath": "features/music_analyzer_ai/api.py",
  "optimizationTypes": ["cache", "async", "database"],
  "profile": true
}
```

### 6. **check_security**
Analiza el código en busca de vulnerabilidades de seguridad.

**Uso:**
```json
{
  "path": "features/",
  "severity": "medium",
  "checkSecrets": true,
  "checkInjection": true
}
```

### 7. **improve_documentation**
Analiza y mejora la documentación del código.

**Uso:**
```json
{
  "filePath": "features/dermatology_ai/services.py",
  "addDocstrings": true,
  "improveComments": true,
  "generateREADME": false
}
```

### 8. **analyze_dependencies**
Analiza las dependencias del proyecto: versiones, conflictos, vulnerabilidades.

**Uso:**
```json
{
  "checkVulnerabilities": true,
  "checkUnused": true,
  "checkConflicts": true,
  "suggestUpdates": true
}
```

### 9. **standardize_code_style**
Estandariza el estilo de código según las mejores prácticas.

**Uso:**
```json
{
  "filePath": "features/blog_posts/generator.py",
  "language": "python",
  "apply": false
}
```

### 10. **detect_anti_patterns**
Detecta anti-patrones comunes en el código.

**Uso:**
```json
{
  "path": "features/copywriting/",
  "antiPatterns": ["dead_code", "god_object", "long_method"]
}
```

### 11. **optimize_imports**
Optimiza y organiza los imports.

**Uso:**
```json
{
  "filePath": "features/seo/optimizer.py",
  "removeUnused": true,
  "organize": true,
  "apply": false
}
```

### 12. **generate_tests**
Genera tests unitarios y de integración.

**Uso:**
```json
{
  "filePath": "features/ai_video/processor.py",
  "testType": "both",
  "framework": "pytest",
  "coverage": 80
}
```

### 13. **migrate_to_clean_architecture**
Ayuda a migrar código existente a Clean Architecture.

**Uso:**
```json
{
  "featurePath": "features/ads",
  "createStructure": true,
  "migrateCode": true,
  "generateInterfaces": true
}
```

### 14. **analyze_feature_consistency**
Analiza la consistencia entre features.

**Uso:**
```json
{
  "compareWith": "instagram_captions",
  "features": ["facebook_posts", "blog_posts"]
}
```

### 15. **optimize_config_files**
Optimiza y unifica archivos de configuración.

**Uso:**
```json
{
  "configType": "all",
  "consolidate": false
}
```

### 16. **detect_unified_patterns**
Detecta patrones que pueden ser unificados (como los optimizadores).

**Uso:**
```json
{
  "patternType": "optimizers",
  "minOccurrences": 3
}
```

## 📚 Recursos Disponibles

El servidor proporciona acceso a los siguientes recursos:

- **codebase://best-practices** - Mejores prácticas del proyecto
- **codebase://architecture-patterns** - Patrones arquitectónicos
- **codebase://refactoring-opportunities** - Oportunidades de refactorización
- **codebase://anti-patterns** - Anti-patrones a evitar
- **codebase://code-quality-standards** - Estándares de calidad

## 🎯 Prompts Disponibles

1. **improve_code_quality** - Mejora la calidad general del código
2. **refactor_code** - Refactoriza código siguiendo mejores prácticas
3. **optimize_performance** - Optimiza performance del código

## 🚀 Configuración

### Instalación

1. Asegúrate de tener Python 3.8+ instalado
2. Instala las dependencias necesarias:

```bash
pip install mcp pylint black isort mypy bandit safety
```

### Configuración en Cursor

1. Abre la configuración de MCP en Cursor
2. Agrega el servidor usando el archivo `mcp_code_improvement_server.json`
3. El servidor se conectará automáticamente

### Variables de Entorno

```bash
export PYTHONPATH="${workspace}/agents/backend/onyx/server/features"
```

## 💡 Ejemplos de Uso

### Análisis Completo de una Feature

```python
# Usar analyze_code_quality + analyze_architecture + check_security
{
  "tool": "analyze_code_quality",
  "arguments": {
    "path": "features/suno_clone_ai",
    "language": "python",
    "checks": ["all"]
  }
}
```

### Refactorización de Optimizadores

```python
# Detectar patrones unificables
{
  "tool": "detect_unified_patterns",
  "arguments": {
    "patternType": "optimizers",
    "minOccurrences": 3
  }
}

# Luego aplicar refactorización
{
  "tool": "suggest_refactoring",
  "arguments": {
    "filePath": "features/suno_clone_ai/core/optimizers/",
    "refactoringType": "extract_class",
    "apply": true
  }
}
```

### Migración a Clean Architecture

```python
{
  "tool": "migrate_to_clean_architecture",
  "arguments": {
    "featurePath": "features/facebook_posts",
    "createStructure": true,
    "migrateCode": true,
    "generateInterfaces": true
  }
}
```

## 📊 Métricas y Reportes

El servidor genera reportes detallados que incluyen:

- **Calidad de Código**: Puntuación general y desglose por categoría
- **Duplicación**: Porcentaje de código duplicado y ubicaciones
- **Seguridad**: Vulnerabilidades encontradas y severidad
- **Performance**: Oportunidades de optimización identificadas
- **Arquitectura**: Conformidad con Clean Architecture y SOLID
- **Documentación**: Cobertura de documentación y mejoras sugeridas

## 🔧 Personalización

Puedes personalizar el servidor editando `mcp_code_improvement_server.json`:

- Agregar nuevas herramientas
- Modificar parámetros por defecto
- Agregar recursos adicionales
- Configurar reglas específicas del proyecto

## 🎓 Mejores Prácticas

1. **Usa análisis incremental**: Empieza con análisis de calidad antes de refactorizar
2. **Revisa sugerencias**: Siempre revisa las sugerencias antes de aplicar cambios automáticos
3. **Pruebas primero**: Genera tests antes de refactorizar código crítico
4. **Documenta cambios**: Usa `improve_documentation` después de refactorizaciones importantes
5. **Mantén consistencia**: Usa `analyze_feature_consistency` para mantener uniformidad

## 🐛 Troubleshooting

### El servidor no se conecta
- Verifica que Python esté en el PATH
- Revisa las variables de entorno
- Asegúrate de que las dependencias estén instaladas

### Herramientas no funcionan
- Verifica los permisos de lectura/escritura
- Revisa los logs del servidor
- Asegúrate de que las rutas sean correctas

### Performance lenta
- Limita el alcance del análisis
- Usa exclusiones apropiadas
- Considera análisis incremental

## 📝 Notas

- El servidor está diseñado específicamente para el proyecto Blatam Academy
- Las herramientas están optimizadas para Python, pero soportan múltiples lenguajes
- Los cambios automáticos siempre se pueden revertir (usa control de versiones)
- Se recomienda ejecutar análisis en branches separados antes de mergear

## 🤝 Contribución

Para mejorar el servidor MCP:

1. Identifica áreas de mejora
2. Agrega nuevas herramientas según necesidad
3. Actualiza la documentación
4. Prueba con diferentes features del proyecto

## 📄 Licencia

MIT License - Blatam Academy











