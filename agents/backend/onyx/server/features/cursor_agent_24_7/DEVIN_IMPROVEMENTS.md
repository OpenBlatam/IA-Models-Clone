# 🚀 Mejoras Devin - Cursor Agent 24/7

Este documento describe las mejoras implementadas basadas en el prompt de Devin para hacer el agente más inteligente, seguro y profesional.

## 📋 Resumen de Mejoras

### 1. Sistema de Personalidad Devin ✅

**Archivo:** `core/devin_persona.py`

Sistema completo que implementa el comportamiento de Devin:

- **Modos de Operación:**
  - `PLANNING`: Modo de planificación donde el agente recopila información
  - `STANDARD`: Modo estándar de ejecución

- **Comunicación con Usuario:**
  - Sistema de mensajes con niveles (INFO, WARNING, ERROR, SUCCESS, DEBUG)
  - Soporte para adjuntos
  - Solicitud de autenticación cuando es necesario
  - Uso del mismo idioma que el usuario

- **Reporte de Problemas de Entorno:**
  - Detección automática de problemas
  - Sugerencias de solución
  - Niveles de severidad (low, medium, high, critical)

- **Razonamiento Interno:**
  - Contextos de razonamiento con observaciones, consideraciones y conclusiones
  - Historial de sesiones de razonamiento

**Uso:**
```python
from cursor_agent_24_7.core.devin_persona import DevinPersona, AgentMode

devin = DevinPersona()
devin.set_mode(AgentMode.PLANNING)
await devin.message_user("Analizando código base...")
await devin.report_environment_issue(
    issue_type="missing_dependency",
    description="Module 'xyz' not found",
    suggestion="Install with: pip install xyz",
    severity="medium"
)
```

### 2. Sistema de Comprensión de Código (LSP-like) ✅

**Archivo:** `core/code_understanding.py`

Sistema similar a LSP para entender el código base:

- **Encontrar Definiciones:**
  - Busca definiciones de funciones, clases, variables
  - Analiza archivos Python usando AST

- **Encontrar Referencias:**
  - Busca todas las referencias a un símbolo
  - Escanea todo el workspace

- **Información de Hover:**
  - Obtiene docstrings, firmas de funciones
  - Información de tipos

- **Análisis de Estructura:**
  - Analiza estructura completa del código base
  - Cuenta funciones, clases, archivos

**Uso:**
```python
from cursor_agent_24_7.core.code_understanding import CodeUnderstanding

code_understanding = CodeUnderstanding(workspace_root="/path/to/project")
symbol = code_understanding.find_definition("my_function", "file.py", 10)
references = code_understanding.find_references("my_function", "file.py", 10)
hover_info = code_understanding.get_hover_info("my_function", "file.py", 10)
```

### 3. Mejoras de Seguridad ✅

**Archivo:** `core/security.py`

Sistema robusto de seguridad:

- **Detección de Secretos:**
  - Detecta API keys, passwords, tokens, credentials
  - Patrones comunes (AWS keys, OpenAI keys, GitHub tokens, etc.)
  - Escaneo de código y comandos

- **Sanitización de Salidas:**
  - Remueve secretos de las salidas antes de mostrarlas
  - Previene exposición accidental de información sensible

- **Validación de Comandos Peligrosos:**
  - Bloquea comandos destructivos (`rm -rf`, `format`, etc.)
  - Sistema de bloqueo/desbloqueo de comandos

**Uso:**
```python
from cursor_agent_24_7.core.security import SecurityManager

security = SecurityManager()
is_valid, error = security.validate_command("rm -rf /")
secrets = security.scan_command_for_secrets(command)
sanitized = security.sanitize_output(output)
```

### 4. Integración en el Agente ✅

**Archivo:** `core/agent.py`

El agente ahora incluye:

- **Personalidad Devin integrada:**
  - Configurable en `AgentConfig`
  - Modo y idioma configurables
  - Reporte automático de problemas de entorno

- **Gestor de seguridad:**
  - Validación automática de comandos
  - Detección de secretos
  - Sanitización de resultados

- **Comprensión de código:**
  - Análisis automático del workspace
  - Disponible para consultas

**Configuración:**
```python
from cursor_agent_24_7.core.agent import AgentConfig, CursorAgent

config = AgentConfig(
    enable_devin_persona=True,
    devin_mode="standard",  # o "planning"
    devin_language="es"
)

agent = CursorAgent(config)
```

### 5. Mejoras en Manejo de Errores ✅

- **Reporte automático de problemas de entorno:**
  - Cuando el agente falla al iniciar
  - Cuando hay errores en la ejecución de tareas
  - Con sugerencias de solución

- **Comunicación proactiva:**
  - El agente informa al usuario cuando encuentra problemas
  - Usa el mismo idioma que el usuario
  - Proporciona contexto útil

## 🎯 Características Principales

### Comunicación Estratégica

El agente ahora comunica con el usuario cuando:
- Encuentra problemas de entorno
- Necesita compartir entregables
- Información crítica no está disponible
- Necesita permisos o claves
- Detecta problemas de seguridad

### Seguridad Mejorada

- **Nunca expone secretos:**
  - Detecta y bloquea secretos en comandos
  - Sanitiza todas las salidas
  - Previene commits de secretos

- **Validación de comandos:**
  - Bloquea comandos peligrosos
  - Sistema de whitelist/blacklist

### Comprensión de Código

- **Análisis inteligente:**
  - Entiende la estructura del código
  - Encuentra definiciones y referencias
  - Proporciona información contextual

### Razonamiento

- **Sesiones de razonamiento:**
  - El agente puede razonar sobre problemas
  - Mantiene contexto de observaciones y conclusiones
  - Planifica pasos siguientes

## 📝 Ejemplos de Uso

### Ejemplo 1: Modo Planificación

```python
agent.devin.set_mode(AgentMode.PLANNING)
context = agent.devin.start_reasoning()
context.add_observation("El código base tiene 50 archivos Python")
context.add_consideration("Necesito entender la estructura antes de hacer cambios")
context.add_conclusion("Debo analizar los archivos principales primero")
context.add_next_step("Analizar core/agent.py")
```

### Ejemplo 2: Reporte de Problema de Entorno

```python
await agent.devin.report_environment_issue(
    issue_type="missing_dependency",
    description="Module 'httpx' not found",
    suggestion="Install with: pip install httpx",
    severity="high"
)
```

### Ejemplo 3: Uso de Comprensión de Código

```python
# Encontrar definición
symbol = agent.code_understanding.find_definition(
    "CursorAgent",
    "core/agent.py",
    124
)

# Encontrar referencias
references = agent.code_understanding.find_references(
    "CursorAgent",
    "core/agent.py",
    124
)

# Obtener información
info = agent.code_understanding.get_hover_info(
    "CursorAgent",
    "core/agent.py",
    124
)
```

## 🔧 Configuración

### Habilitar/Deshabilitar Devin

```python
config = AgentConfig(
    enable_devin_persona=True,  # Habilitar personalidad Devin
    devin_mode="standard",      # Modo: "planning" o "standard"
    devin_language="es"        # Idioma: "es", "en", etc.
)
```

### Configurar Seguridad

```python
# Bloquear comandos específicos
agent.security_manager.block_command("rm -rf")

# Desbloquear comandos
agent.security_manager.unblock_command("rm -rf")
```

## 6. Sistema de Comandos Devin ✅

**Archivo:** `core/devin_commands.py`

Sistema estructurado de comandos que simula los comandos disponibles en Devin:

- **Comandos Estructurados:**
  - `reasoning`: Ejecutar razonamiento interno
  - `shell`: Ejecutar comandos shell
  - `search`: Realizar búsquedas en el código
  - `plan`: Gestionar planes de trabajo

- **Sistema de Planificación:**
  - Crear planes con múltiples pasos
  - Gestionar dependencias entre pasos
  - Sugerir planes cuando están listos
  - Seguimiento del estado de ejecución

**Uso:**
```python
# Crear un plan
result = await agent.devin_commands.execute_command(
    "plan",
    action="create",
    title="Mejorar sistema de autenticación",
    description="Implementar mejoras de seguridad"
)

# Agregar pasos
await agent.devin_commands.execute_command(
    "plan",
    action="add_step",
    plan_id="plan_123",
    step_description="Revisar código actual de autenticación"
)

# Sugerir plan cuando esté listo
agent.devin_commands.suggest_plan("plan_123")
```

## 7. Gestor de Herramientas ✅

**Archivo:** `core/tool_manager.py`

Sistema que detecta y gestiona herramientas disponibles en el sistema:

- **Detección Automática:**
  - Git, Python, pip, Node.js, npm
  - Docker, curl, wget
  - Verificación de versiones

- **Verificación de Librerías:**
  - Verificar si librerías Python están disponibles
  - Útil para seguir la regla de Devin de no asumir librerías

**Uso:**
```python
# Verificar si una herramienta está disponible
if agent.tool_manager.is_available("git"):
    print("Git está disponible")

# Verificar librería Python
if agent.tool_manager.check_library_available("requests"):
    print("requests está instalado")

# Obtener información de herramienta
info = agent.tool_manager.get_tool_info("python")
print(f"Python version: {info.version}")
```

## 8. Analizador de Convenciones de Código ✅

**Archivo:** `core/code_conventions.py`

Sistema que analiza y detecta convenciones de código del proyecto:

- **Detección Automática:**
  - Estilo de indentación (espacios/tabs, cantidad)
  - Longitud máxima de líneas
  - Estilo de comillas (simples/dobles)
  - Estilo de imports (absolutos/relativos)
  - Convenciones de nombres (snake_case, PascalCase, etc.)
  - Uso de type hints
  - Librerías utilizadas en el proyecto

- **Verificación de Archivos:**
  - Verificar si un archivo sigue las convenciones
  - Detectar violaciones

**Uso:**
```python
# Analizar workspace
conventions = agent.code_conventions.analyze_workspace()

# Verificar si una librería se usa
if agent.code_conventions.check_library_usage("requests"):
    print("requests se usa en el proyecto")

# Verificar archivo
result = agent.code_conventions.check_file_conventions("file.py")
```

## 9. Verificador de Cambios ✅

**Archivo:** `core/change_verifier.py`

Sistema que verifica que todos los cambios están completos antes de reportar:

- **Gestión de Cambios:**
  - Crear conjuntos de cambios
  - Agregar cambios individuales
  - Gestionar referencias a actualizar
  - Especificar tests a ejecutar

- **Verificación Completa:**
  - Verificar que todos los archivos existen
  - Verificar que todas las referencias están actualizadas
  - Detectar problemas antes de reportar

**Uso:**
```python
# Crear conjunto de cambios
change_set = agent.change_verifier.create_change_set(
    "Mejorar sistema de autenticación"
)

# Agregar cambios
change_set.add_change(
    "core/auth.py",
    "modify",
    "Agregar validación de tokens"
)

# Agregar referencias a actualizar
change_set.add_reference("api/routes/auth.py")

# Verificar antes de reportar
result = agent.change_verifier.verify_change_set(change_set.id)
if result["success"]:
    print("Todos los cambios están completos")
```

## 10. Verificación Mejorada de Librerías ✅

**Archivo:** `core/tool_manager.py`

Mejora en la verificación de librerías:

- **Verificación en Proyecto:**
  - Verificar si una librería está en requirements.txt
  - Verificar en pyproject.toml
  - Verificar en setup.py
  - Siguiendo la regla de Devin: nunca asumir librerías

**Uso:**
```python
# Verificar si está en el proyecto (no solo instalada)
if agent.tool_manager.check_library_in_project("requests"):
    print("requests está en requirements.txt")
```

## 🚀 Próximos Pasos

1. **API Endpoints:**
   - Endpoints para acceder a funcionalidades de Devin
   - Endpoints para gestión de seguridad
   - Endpoints para comprensión de código
   - Endpoints para comandos Devin y planes
   - Endpoints para convenciones y verificación

2. **Integración con LSP Real:**
   - Integración con pyright/pylance
   - Soporte para múltiples lenguajes

3. **Mejoras en Razonamiento:**
   - Razonamiento más sofisticado
   - Integración con LLMs para razonamiento avanzado

4. **Sistema de Ejecución Inteligente:**
   - Decisión automática de qué herramientas usar
   - Ejecución paralela de comandos independientes

5. **Sistema de Tests y Linting:**
   - Ejecutar tests automáticamente antes de reportar
   - Ejecutar linting y verificar que pasa

## 11. Ejecutor de Tests y Linting ✅

**Archivo:** `core/test_runner.py`

Sistema para ejecutar tests y verificaciones antes de reportar cambios:

- **Ejecución de Tests:**
  - Detección automática de framework de tests (pytest, unittest)
  - Ejecución de tests específicos o todos
  - Captura de resultados y errores

- **Ejecución de Linting:**
  - Detección automática de herramienta de linting (flake8, ruff, pylint)
  - Verificación de archivos específicos
  - Reporte de errores y warnings

**Uso:**
```python
# Ejecutar tests
test_results = await agent.test_runner.run_tests()

# Ejecutar linting
lint_result = await agent.test_runner.run_lint()

# Verificar cambios con tests y linting
result = await agent.change_verifier.verify_change_set(
    change_set_id,
    run_tests=True,
    run_lint=True
)
```

## 12. Rastreador de Referencias ✅

**Archivo:** `core/reference_tracker.py`

Sistema que rastrea y verifica referencias a código modificado:

- **Búsqueda de Referencias:**
  - Encuentra todas las referencias a un símbolo
  - Escanea todo el workspace
  - Identifica tipo de referencia

- **Rastreo de Cambios:**
  - Rastrea cambios en símbolos
  - Verifica que todas las referencias se actualicen
  - Detecta referencias faltantes

**Uso:**
```python
# Rastrear cambio en símbolo
reference_changes = agent.reference_tracker.track_symbol_change(
    "my_function",
    "core/module.py",
    10,
    "rename"
)

# Verificar que todas las referencias fueron actualizadas
result = agent.reference_tracker.verify_references_updated(change_id)
```

## 13. Ejecutor Paralelo ✅

**Archivo:** `core/parallel_executor.py`

Sistema para ejecutar comandos en paralelo cuando no tienen dependencias:

- **Ejecución Paralela:**
  - Ejecuta tareas independientes en paralelo
  - Respeta dependencias entre tareas
  - Control de concurrencia máxima

- **Gestión de Dependencias:**
  - Define dependencias entre tareas
  - Ejecuta en orden correcto
  - Detecta dependencias circulares

**Uso:**
```python
# Agregar tareas
agent.parallel_executor.add_task(
    "task1", "Procesar datos",
    process_data,
    dependencies=[]
)

agent.parallel_executor.add_task(
    "task2", "Generar reporte",
    generate_report,
    dependencies=["task1"]
)

# Ejecutar todas respetando dependencias
results = await agent.parallel_executor.execute_all()
```

## 14. Analizador de Contexto ✅

**Archivo:** `core/context_analyzer.py`

Sistema que analiza el contexto del código antes de hacer cambios:

- **Análisis de Archivo:**
  - Detecta imports y librerías usadas
  - Identifica frameworks (FastAPI, Flask, Django, etc.)
  - Detecta patrones de código
  - Analiza convenciones

- **Análisis de Componentes:**
  - Encuentra componentes similares existentes
  - Analiza cómo están escritos
  - Identifica convenciones de nombres y typing

- **Contexto Alrededor:**
  - Obtiene contexto alrededor de una línea
  - Analiza imports y frameworks usados
  - Útil para entender código antes de modificarlo

**Uso:**
```python
# Analizar contexto de archivo antes de modificar
context = agent.context_analyzer.analyze_file_context("core/module.py")
print(f"Frameworks: {context.frameworks}")
print(f"Libraries: {context.libraries}")

# Analizar componentes similares antes de crear uno nuevo
analyses = agent.context_analyzer.analyze_component(
    "AuthHandler",
    "class"
)

# Obtener contexto alrededor de una línea
surrounding = agent.context_analyzer.get_surrounding_context(
    "core/module.py",
    50,
    context_lines=10
)
```

## 15. Verificador de Completitud ✅

**Archivo:** `core/completion_verifier.py`

Sistema que verifica que se cumplieron todos los requisitos:

- **Gestión de Requisitos:**
  - Crear tareas con requisitos específicos
  - Verificar cada requisito
  - Métodos de verificación (test, lint, reference)

- **Verificación Completa:**
  - Verifica que todos los requisitos se cumplieron
  - Ejecuta tests y linting si es necesario
  - Detecta requisitos faltantes

**Uso:**
```python
# Crear tarea con requisitos
task = agent.completion_verifier.create_task(
    "task_123",
    "Implementar nueva funcionalidad"
)

# Agregar requisitos
task.add_requirement("Código debe pasar tests", "test")
task.add_requirement("Código debe pasar linting", "lint")
task.add_requirement("Todas las referencias actualizadas", "reference")

# Verificar completitud
result = await agent.completion_verifier.verify_task_completion(
    "task_123",
    run_tests=True,
    run_lint=True
)
```

## 16. Gestor de Iteraciones ✅

**Archivo:** `core/iteration_manager.py`

Sistema que gestiona iteraciones sobre cambios hasta que sean correctos:

- **Gestión de Iteraciones:**
  - Crea tareas con múltiples iteraciones
  - Registra cambios en cada iteración
  - Limita número máximo de iteraciones

- **Verificación por Iteración:**
  - Ejecuta tests y linting en cada iteración
  - Detecta problemas
  - Marca como completado cuando todo pasa

**Uso:**
```python
# Crear tarea con iteraciones
task = agent.iteration_manager.create_task(
    "task_123",
    "Corregir bug en autenticación",
    max_iterations=5
)

# Iniciar iteración
iteration = agent.iteration_manager.start_iteration(
    "task_123",
    "Primera corrección"
)

# Registrar cambios
agent.iteration_manager.record_change(
    "task_123",
    "Agregar validación de token"
)

# Verificar iteración
result = await agent.iteration_manager.verify_iteration(
    "task_123",
    run_tests=True,
    run_lint=True
)

# Si falla, iterar de nuevo hasta que funcione
```

## 17. Verificador Crítico ✅

**Archivo:** `core/critical_verifier.py`

Sistema que realiza verificación crítica antes de reportar al usuario:

- **Verificaciones Críticas:**
  - Tests pasan
  - Linting pasa
  - Referencias actualizadas
  - No hay secretos expuestos
  - Cambios verificados
  - Requisitos cumplidos

- **Verificación Automática:**
  - Se ejecuta automáticamente antes de reportar
  - Verifica todos los aspectos críticos
  - Bloquea reporte si hay problemas críticos

**Uso:**
```python
# Verificar críticamente antes de reportar
result = await agent.critical_verifier.verify_before_reporting(
    "task_123",
    "Implementar nueva funcionalidad",
    agent=agent
)

if result["can_report"]:
    await agent.devin.message_user(
        "✅ Tarea completada exitosamente",
        level=agent.devin.CommunicationLevel.SUCCESS
    )
else:
    await agent.devin.message_user(
        f"⚠️ Verificación falló: {result['issues']}",
        level=agent.devin.CommunicationLevel.WARNING
    )
```

## 18. Sistema de Triggers de Razonamiento ✅

**Archivo:** `core/reasoning_trigger.py`

Sistema que activa automáticamente el razonamiento antes de decisiones críticas:

- **Acciones que Requieren Razonamiento:**
  - Decisiones de git (branch, checkout, PR)
  - Inicio de cambios de código
  - Reporte de completitud
  - Fallos de tests/linting/CI
  - Problemas de entorno
  - Sin siguiente paso claro

- **Razonamiento Automático:**
  - Se activa automáticamente en situaciones críticas
  - Agrega observaciones y consideraciones relevantes
  - Ayuda a tomar mejores decisiones

**Uso:**
```python
from cursor_agent_24_7.core.reasoning_trigger import CriticalActionType

# Antes de hacer cambios de código
await agent.reasoning_trigger.trigger_reasoning(
    CriticalActionType.CODE_CHANGE_START,
    context={
        "files_to_edit": ["core/auth.py", "api/routes.py"]
    }
)

# Antes de reportar completitud
await agent.reasoning_trigger.trigger_reasoning(
    CriticalActionType.COMPLETION_REPORT,
    context={
        "verification": {"success": True}
    }
)

# Cuando tests fallan
await agent.reasoning_trigger.trigger_reasoning(
    CriticalActionType.TEST_FAILURE,
    context={
        "test_results": [...]
    }
)
```

## 19. Verificador de Intención del Usuario ✅

**Archivo:** `core/intent_verifier.py`

Sistema que verifica que se cumplió la intención del usuario, no solo los requisitos técnicos:

- **Verificaciones de Intención:**
  - La descripción de la tarea coincide con la solicitud
  - Se realizaron cambios relevantes
  - Todos los archivos mencionados fueron modificados
  - La funcionalidad solicitada está implementada

- **Verificación Automática:**
  - Se ejecuta automáticamente antes de reportar
  - Calcula un score de confianza
  - Bloquea reporte si la intención no se cumplió

**Uso:**
```python
# Verificar intención del usuario
result = await agent.intent_verifier.verify_user_intent(
    "task_123",
    "Agregar autenticación JWT al sistema",
    "Implementar autenticación JWT",
    changes_made=[
        {"file_path": "core/auth.py", "description": "Agregar JWT"},
        {"file_path": "api/routes.py", "description": "Proteger rutas"}
    ]
)

if result["can_report"]:
    await agent.devin.message_user(
        "✅ Intención del usuario cumplida",
        level=agent.devin.CommunicationLevel.SUCCESS
    )
```

## 20. Integración Automática en Flujo de Tareas ✅

**Archivo:** `core/task/task_processor.py`

Integración automática de todos los sistemas de verificación en el flujo de procesamiento de tareas:

- **Verificación Automática Antes de Reportar:**
  1. Activa razonamiento automático (COMPLETION_REPORT)
  2. Ejecuta verificación crítica
  3. Verifica intención del usuario
  4. Solo reporta si todas las verificaciones pasan

- **Flujo Integrado:**
  - Se ejecuta automáticamente en cada tarea completada
  - Comunica problemas al usuario si los hay
  - Previene reportes incompletos o incorrectos

**Uso:**
```python
# El flujo se ejecuta automáticamente cuando una tarea se completa
# No requiere configuración adicional

# El agente automáticamente:
# 1. Activa razonamiento antes de reportar
# 2. Verifica críticamente todos los aspectos
# 3. Verifica que se cumplió la intención del usuario
# 4. Solo reporta si todo está correcto
```

## 📚 Referencias

- **Acciones que Requieren Razonamiento:**
  - Decisiones de git (branch, checkout, PR)
  - Inicio de cambios de código
  - Reporte de completitud
  - Fallos de tests/linting/CI
  - Problemas de entorno
  - Sin siguiente paso claro

- **Razonamiento Automático:**
  - Se activa automáticamente en situaciones críticas
  - Agrega observaciones y consideraciones relevantes
  - Ayuda a tomar mejores decisiones

**Uso:**
```python
from cursor_agent_24_7.core.reasoning_trigger import CriticalActionType

# Antes de hacer cambios de código
await agent.reasoning_trigger.trigger_reasoning(
    CriticalActionType.CODE_CHANGE_START,
    context={
        "files_to_edit": ["core/auth.py", "api/routes.py"]
    }
)

# Antes de reportar completitud
await agent.reasoning_trigger.trigger_reasoning(
    CriticalActionType.COMPLETION_REPORT,
    context={
        "verification": {"success": True}
    }
)

# Cuando tests fallan
await agent.reasoning_trigger.trigger_reasoning(
    CriticalActionType.TEST_FAILURE,
    context={
        "test_results": [...]
    }
)
```

## 19. Verificador de Intención del Usuario ✅

**Archivo:** `core/intent_verifier.py`

Sistema que verifica que se cumplió la intención del usuario, no solo los requisitos técnicos:

- **Verificaciones de Intención:**
  - La descripción de la tarea coincide con la solicitud
  - Se realizaron cambios relevantes
  - Todos los archivos mencionados fueron modificados
  - La funcionalidad solicitada está implementada

- **Verificación Automática:**
  - Se ejecuta automáticamente antes de reportar
  - Calcula un score de confianza
  - Bloquea reporte si la intención no se cumplió

**Uso:**
```python
# Verificar intención del usuario
result = await agent.intent_verifier.verify_user_intent(
    "task_123",
    "Agregar autenticación JWT al sistema",
    "Implementar autenticación JWT",
    changes_made=[
        {"file_path": "core/auth.py", "description": "Agregar JWT"},
        {"file_path": "api/routes.py", "description": "Proteger rutas"}
    ]
)

if result["can_report"]:
    await agent.devin.message_user(
        "✅ Intención del usuario cumplida",
        level=agent.devin.CommunicationLevel.SUCCESS
    )
```

## 20. Integración Automática en Flujo de Tareas ✅

**Archivo:** `core/task/task_processor.py`

Integración automática de todos los sistemas de verificación en el flujo de procesamiento de tareas:

- **Verificación Automática Antes de Reportar:**
  1. Activa razonamiento automático (COMPLETION_REPORT)
  2. Ejecuta verificación crítica
  3. Verifica intención del usuario
  4. Solo reporta si todas las verificaciones pasan

- **Flujo Integrado:**
  - Se ejecuta automáticamente en cada tarea completada
  - Comunica problemas al usuario si los hay
  - Previene reportes incompletos o incorrectos

**Uso:**
```python
# El flujo se ejecuta automáticamente cuando una tarea se completa
# No requiere configuración adicional

# El agente automáticamente:
# 1. Activa razonamiento antes de reportar
# 2. Verifica críticamente todos los aspectos
# 3. Verifica que se cumplió la intención del usuario
# 4. Solo reporta si todo está correcto
```

## 21. Protector de Tests ✅

**Archivo:** `core/test_protector.py`

Sistema que previene la modificación de tests a menos que se solicite explícitamente:

- **Detección de Tests:**
  - Detecta archivos de test automáticamente
  - Identifica frameworks de test (pytest, unittest, nose)
  - Escanea directorios de tests

- **Protección:**
  - Bloquea modificación de tests a menos que la tarea lo requiera explícitamente
  - Detecta palabras clave como "modify test", "update test", etc.
  - Registra todos los intentos de modificación

**Uso:**
```python
# Verificar si se puede modificar un test
can_modify, reason = agent.test_protector.can_modify_test(
    "tests/test_auth.py",
    "task_123",
    "Fix authentication bug"
)

if not can_modify:
    await agent.devin.message_user(
        f"⚠️ No se puede modificar el test: {reason}",
        level=agent.devin.CommunicationLevel.WARNING
    )

# Permitir explícitamente modificación de test
agent.test_protector.allow_test_modification("tests/test_auth.py")
```

## 22. Integración con CI ✅

**Archivo:** `core/ci_integration.py`

Sistema de integración con CI para testing cuando hay problemas de entorno:

- **Detección de CI:**
  - Detecta sistemas de CI automáticamente (GitHub Actions, GitLab CI, etc.)
  - Escanea archivos de configuración de CI

- **Testing vía CI:**
  - Ejecuta tests vía CI cuando hay problemas de entorno
  - No intenta arreglar problemas de entorno
  - Usa CI en lugar del entorno local

**Uso:**
```python
# Cuando hay problema de entorno, usar CI para testing
if agent.ci_integration.should_use_ci(has_environment_issue=True):
    result = await agent.ci_integration.run_tests_via_ci(
        "task_123",
        reason="environment_issue",
        environment_issue="Missing dependency: xyz"
    )
    
    if result["success"]:
        await agent.devin.message_user(
            f"✅ Tests ejecutados vía CI: {result['ci_system']}",
            level=agent.devin.CommunicationLevel.SUCCESS
        )
```

## 23. Gestor de Git ✅

**Archivo:** `core/git_manager.py`

Sistema de gestión de Git siguiendo las reglas específicas de Devin:

- **Reglas Implementadas:**
  - Nunca hacer force push
  - Nunca usar `git add .`
  - Usar gh cli para operaciones de GitHub
  - No cambiar git config a menos que se solicite
  - Formato de branch: `devin/{timestamp}-{feature-name}`
  - Actualizar el mismo PR en iteraciones
  - Pedir ayuda si CI no pasa después del tercer intento

- **Operaciones:**
  - Crear branches con formato Devin
  - Agregar archivos específicos (nunca `git add .`)
  - Hacer commits
  - Hacer push (sin force)
  - Crear y actualizar PRs
  - Verificar estado de CI

**Uso:**
```python
# Crear branch con formato Devin
success, branch_name, error = await agent.git_manager.create_branch(
    "add-authentication",
    base_branch="main"
)

# Agregar archivos específicos (nunca git add .)
success, error = await agent.git_manager.add_files([
    "core/auth.py",
    "api/routes.py"
])

# Hacer commit
success, error = await agent.git_manager.commit(
    "Add JWT authentication",
    files=["core/auth.py", "api/routes.py"]
)

# Hacer push (sin force)
success, error = await agent.git_manager.push(branch_name)

# Crear PR
success, pr_info, error = await agent.git_manager.create_pr(
    "Add JWT Authentication",
    "Implements JWT-based authentication system",
    base="main",
    head=branch_name
)

# Verificar estado de CI
ci_status = await agent.git_manager.check_ci_status(pr_info["number"])
if ci_status.get("should_ask_help"):
    await agent.devin.message_user(
        "⚠️ CI no ha pasado después de 3 intentos. Necesito ayuda.",
        level=agent.devin.CommunicationLevel.WARNING
    )
```

## 24. Verificador de Múltiples Ubicaciones ✅

**Archivo:** `core/multi_location_verifier.py`

Sistema que verifica que todas las ubicaciones relevantes fueron editadas en tareas que requieren modificar muchas ubicaciones:

- **Gestión de Ubicaciones:**
  - Registrar todas las ubicaciones a editar antes de empezar
  - Marcar ubicaciones como editadas
  - Verificar que todas fueron editadas antes de reportar

- **Verificación Completa:**
  - Verifica que todas las ubicaciones fueron editadas
  - Verifica que todas fueron verificadas
  - Detecta ubicaciones faltantes

**Uso:**
```python
# Crear tarea con múltiples ubicaciones
task = agent.multi_location_verifier.create_task(
    "task_123",
    "Actualizar todas las referencias a User"
)

# Agregar ubicaciones a editar
task.add_location("models/user.py", line=10, symbol="User", reason="Update User model")
task.add_location("api/routes.py", line=25, symbol="User", reason="Update route handler")
task.add_location("services/auth.py", line=5, symbol="User", reason="Update auth service")

# Marcar ubicaciones como editadas
agent.multi_location_verifier.mark_location_edited("task_123", "models/user.py", verified=True)
agent.multi_location_verifier.mark_location_edited("task_123", "api/routes.py", verified=True)
agent.multi_location_verifier.mark_location_edited("task_123", "services/auth.py", verified=True)

# Verificar antes de reportar
result = await agent.multi_location_verifier.verify_all_locations("task_123", agent=agent)
if result["success"]:
    await agent.devin.message_user(
        f"✅ Todas las ubicaciones fueron editadas: {result['locations_edited']}/{result['locations_total']}",
        level=agent.devin.CommunicationLevel.SUCCESS
    )
else:
    await agent.devin.message_user(
        f"⚠️ Faltan ubicaciones por editar: {result['missing_locations']}",
        level=agent.devin.CommunicationLevel.WARNING
    )
```

## 25. Integración con Navegador ✅

**Archivo:** `core/browser_integration.py`

Sistema de integración con navegador para inspeccionar páginas web:

- **Visitar Links:**
  - No asume contenido de links sin visitarlos
  - Usa capacidades de navegación cuando sea necesario
  - Soporta Playwright y requests como fallback

- **Gestión de Sesiones:**
  - Mantiene sesiones de navegación
  - Almacena contenido recuperado
  - Rastrea links visitados

**Uso:**
```python
# Verificar si se debe visitar un link
if agent.browser_integration.should_visit_link("https://example.com", reason="Get API documentation"):
    result = await agent.browser_integration.visit_link(
        "https://example.com",
        reason="Get API documentation",
        task_id="task_123"
    )
    
    if result["success"]:
        session = agent.browser_integration.get_session(result["session_id"])
        if session and session.content:
            await agent.devin.message_user(
                f"✅ Contenido recuperado de {result['url']}: {result['content_length']} caracteres",
                level=agent.devin.CommunicationLevel.SUCCESS
            )
```

## 26. Verificador de Planificación ✅

**Archivo:** `core/planning_verifier.py`

Sistema que verifica que se tiene toda la información necesaria antes de sugerir un plan:

- **Verificaciones:**
  - Verifica que se conocen todas las ubicaciones a editar
  - Verifica que se conocen todas las referencias a actualizar
  - Verifica que se recopiló toda la información necesaria
  - Verifica que se entiende la estructura del código base
  - Verifica que se conocen las convenciones de código

- **Integración:**
  - Se integra automáticamente con `suggest_plan()`
  - Bloquea sugerencia de plan si faltan verificaciones
  - Proporciona detalles de lo que falta

**Uso:**
```python
# Verificar antes de sugerir plan
result = await agent.planning_verifier.verify_before_suggesting_plan(
    plan_id="plan_123",
    locations_to_edit=[
        {"file_path": "core/auth.py", "symbol": "User", "line": 10},
        {"file_path": "api/routes.py", "symbol": "User", "line": 25}
    ],
    references_to_update=[
        {"file_path": "services/auth.py", "symbol": "User"}
    ],
    information_gathered={"complete": True},
    agent=agent
)

if result["can_suggest_plan"]:
    await agent.devin.suggest_plan({
        "id": "plan_123",
        "title": "Update User model",
        "description": "Update all references to User model",
        "locations_to_edit": [...],
        "references_to_update": [...]
    })
else:
    await agent.devin.message_user(
        f"⚠️ No se puede sugerir el plan aún. Faltan: {result['issues']}",
        level=agent.devin.CommunicationLevel.WARNING
    )
```

## 📚 Referencias

- [README.md](README.md) - Documentación principal
- [FEATURES.md](FEATURES.md) - Lista completa de características
- [API_REFERENCE.md](API_REFERENCE.md) - Referencia de API

