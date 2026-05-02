# Architecture Documentation

## 📐 Arquitectura del TruthGPT Continuous Agent

### Visión General

El agente está diseñado con una arquitectura modular y desacoplada, integrando conceptos de 10 papers de investigación sobre agentes autónomos.

### 🏗️ Estructura de Capas

```
┌─────────────────────────────────────────────────────────┐
│              Application Layer                          │
│  (turtlegpt_continuous_agent.py)                      │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│              Integration Layer                           │
│  (agent_integrator.py, component_factory.py)            │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│              Strategy Layer                              │
│  (paper_strategies.py, strategy_selector.py)            │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│              Management Layer                            │
│  (task_manager, reflection_planner, metrics_manager,    │
│   learning_manager, agi_capabilities_manager)           │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│              Infrastructure Layer                       │
│  (event_system, health_monitor, config_validator)       │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│              Data Layer                                  │
│  (models.py, memory systems)                            │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│              Client Layer                                │
│  (openrouter_client.py)                                │
└─────────────────────────────────────────────────────────┘
```

### 🔄 Flujo de Datos

```
Task Submission
    ↓
TaskManager (queue)
    ↓
StrategySelector (select strategy)
    ↓
Paper Strategy (execute)
    ├─ ReAct
    ├─ LATS
    ├─ Tree of Thoughts
    └─ Standard (Generative Agents)
    ↓
EventBus (publish events)
    ↓
Memory Systems (store experience)
    ↓
ReflectionPlanner (periodic reflection)
    ↓
LearningManager (identify opportunities)
    ↓
MetricsManager (track metrics)
    ↓
HealthMonitor (check health)
    ↓
CallbackManager (notify callbacks)
```

### 📦 Componentes Principales

#### 1. Core Modules
- **models.py**: Modelos de datos (TaskStatus, AgentTask, etc.)
- **task_manager.py**: Gestión de cola de tareas
- **reflection_planner.py**: Reflexión y planificación
- **metrics_manager.py**: Tracking de métricas
- **maintenance_manager.py**: Mantenimiento periódico
- **callback_manager.py**: Gestión de callbacks

#### 2. Paper Strategies
- **paper_strategies.py**: Implementaciones de estrategias de papers
  - ReactStrategy
  - LATSStrategy
  - TreeOfThoughtsStrategy
  - TheoryOfMindStrategy
  - PersonalityStrategy
  - ToolformerStrategy

#### 3. Managers
- **strategy_selector.py**: Selección automática de estrategias
- **learning_manager.py**: Gestión de aprendizaje autónomo
- **prompt_builder.py**: Construcción de prompts
- **agi_capabilities_manager.py**: Evaluación de capacidades AGI
- **task_processor.py**: Procesamiento de tareas ✨

#### 4. Infrastructure
- **component_factory.py**: Factory para crear componentes
- **config_validator.py**: Validación de configuración
- **config_helper.py**: Helpers de configuración
- **event_system.py**: Sistema de eventos Pub-Sub
- **health_monitor.py**: Monitoreo de salud
- **utils.py**: Funciones de utilidad
- **agent_integrator.py**: Integración de componentes
- **error_handler.py**: Manejo centralizado de errores ✨
- **decorators.py**: Decoradores reutilizables ✨
- **middleware.py**: Sistema de middleware ✨
- **async_utils.py**: Utilidades asíncronas ✨
- **memory_context_builder.py**: Construcción de contexto de memoria ✨
- **agent_operations.py**: Operaciones básicas del agente ✨
- **llm_service.py**: Servicio encapsulado para LLM ✨
- **status_builder.py**: Constructor de estado del agente ✨
- **event_publisher.py**: Helper para publicación de eventos ✨
- **signal_handler.py**: Gestión centralizada de señales ✨
- **agent_lifecycle.py**: Gestión del ciclo de vida ✨
- **startup_logger.py**: Logging de inicio ✨
- **loop_configurator.py**: Configuración del loop ✨
- **task_validator.py**: Validación de tareas ✨
- **task_executor.py**: Ejecutor de tareas con concurrencia ✨
- **hook_manager.py**: Gestión centralizada de hooks ✨
- **memory_operations.py**: Operaciones centralizadas de memoria ✨
- **resilient_operations.py**: Operaciones resilientes con circuit breakers ✨
- **tool_executor.py**: Ejecutor centralizado de herramientas ✨
- **state_manager.py**: Gestión centralizada del estado ✨
- **metrics_tracker.py**: Tracking centralizado de métricas ✨
- **task_submitter.py**: Gestión centralizada del envío de tareas ✨
- **component_registry.py**: Registro centralizado de componentes ✨
- **periodic_scheduler.py**: Programación de tareas periódicas ✨
- **test_generator.py**: Generador comprehensivo de tests unitarios con AAA pattern, edge cases y mocks automáticos ✨
- **test_generator_example.py**: Ejemplos prácticos de uso del generador de tests ✨
- **TEST_GENERATOR_STRATEGY.md**: Documento estratégico completo con objetivos, métricas, timeline y mejores prácticas ✨
- **TEST_GENERATOR_USAGE_GUIDE.md**: Guía completa de uso con ejemplos, troubleshooting y mejores prácticas ✨
- **config_manager.py**: Gestión centralizada de configuración con historial y persistencia ✨
- **logging_manager.py**: Gestión centralizada de logging con formateo estructurado, rotación y historial ✨
- **performance_profiler.py**: Perfilado de rendimiento con medición de tiempo, memoria y análisis de cuellos de botella ✨
- **state_persistence.py**: Persistencia y serialización del estado del agente con múltiples formatos y snapshots ✨
- **report_generator.py**: Generación de reportes comprehensivos en múltiples formatos (JSON, Markdown, HTML, CSV) ✨
- **strategy_manager.py**: Gestión centralizada de estrategias ✨
- **event_publisher.py**: Helper para publicación de eventos ✨
- **signal_handler.py**: Gestión centralizada de señales ✨

#### 5. Client
- **openrouter_client.py**: Cliente OpenRouter/TruthGPT

### 🔌 Patrones de Diseño

#### Factory Pattern
- `ComponentFactory`: Crea todos los componentes del agente

#### Strategy Pattern
- `StrategySelector`: Selecciona estrategia según prioridad
- `paper_strategies.py`: Diferentes estrategias intercambiables

#### Observer Pattern
- `EventBus`: Sistema Pub-Sub para comunicación desacoplada

#### Manager Pattern
- Múltiples managers para diferentes responsabilidades

### 🔄 Ciclo de Vida del Agente

1. **Inicialización**
   - Validar configuración
   - Crear componentes (Factory)
   - Configurar eventos y health checks
   - Inicializar estrategias

2. **Operación**
   - Loop principal 24/7
   - Procesar tareas
   - Reflexión periódica
   - Aprendizaje continuo
   - Monitoreo de salud

3. **Finalización**
   - Detención graceful
   - Cerrar conexiones
   - Guardar estado
   - Reportes finales

### 📊 Integración de Papers

Cada paper aporta conceptos específicos:

1. **Generative Agents**: Memoria, reflexión, planificación
2. **ReAct**: Ciclo reasoning-acting-observation
3. **LATS**: Tree search unificado
4. **LLM to Autonomous**: Niveles de autonomía
5. **Self-Initiated Learning**: Aprendizaje autónomo
6. **Tree of Thoughts**: Razonamiento deliberado
7. **Theory of Mind**: Modelado de otros agentes
8. **Personality-Driven**: Decisiones basadas en personalidad
9. **Toolformer**: Auto-aprendizaje de herramientas
10. **Sparks of AGI**: Capacidades tipo AGI

### 🔐 Principios de Diseño

- **Separación de Responsabilidades**: Cada módulo tiene una responsabilidad clara
- **Desacoplamiento**: Componentes se comunican vía eventos
- **Extensibilidad**: Fácil agregar nuevos papers/estrategias
- **Testabilidad**: Componentes independientes y mockeables
- **Mantenibilidad**: Código organizado y documentado

### 🚀 Extensión

Para agregar un nuevo paper:

1. Crear estrategia en `paper_strategies.py`
2. Agregar al `ComponentFactory`
3. Configurar en `ConfigValidator`
4. Integrar en `StrategySelector` si aplica
5. Actualizar documentación
