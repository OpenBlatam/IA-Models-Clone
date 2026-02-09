# Cursor Agent 24/7 🤖

Agente persistente que escucha comandos desde la ventana de Cursor y los ejecuta continuamente, incluso cuando la computadora está apagada (como servicio).

## 🚀 Características

### Core Features
- ✅ **Escucha comandos desde Cursor**: Se activa automáticamente cuando escribes algo en Cursor
- ✅ **Ejecución continua**: Ejecuta tareas sin parar, 24/7
- ✅ **Control simple**: Botón para iniciar/detener fácilmente
- ✅ **Servicio persistente**: Puede correr en background como servicio del sistema
- ✅ **API REST**: Control completo vía API

### Advanced Features
- ✅ **Serverless Ready**: Soporte para AWS Lambda, Azure Functions, GCP Functions
- ✅ **API Gateway**: Integración con Kong y AWS API Gateway
- ✅ **High Performance**: JSON parsing 3-10x más rápido, compresión avanzada
- ✅ **Circuit Breakers**: Resiliencia automática con circuit breakers
- ✅ **Rate Limiting**: Rate limiting distribuido con Redis
- ✅ **Service Discovery**: Consul, etcd, Kubernetes DNS
- ✅ **Elasticsearch**: Búsqueda avanzada integrada
- ✅ **Webhooks**: Sistema completo de webhooks
- ✅ **Bulk Operations**: Operaciones en lote optimizadas
- ✅ **Interfaz web**: Panel de control simple y moderno
- ✅ **Estado persistente**: Guarda el estado incluso después de reiniciar
- ✅ **WebSocket**: Comunicación en tiempo real
- ✅ **Notificaciones**: Sistema de notificaciones avanzado
- ✅ **Métricas**: Recopilación y análisis de métricas
- ✅ **Health Checks**: Monitoreo de salud del sistema
- ✅ **Rate Limiting**: Control de tasa de requests
- ✅ **Exportación de datos**: Exportar tareas y métricas
- ✅ **Programación de tareas**: Ejecutar tareas en horarios específicos
- ✅ **Backups**: Sistema de respaldos automáticos
- ✅ **Plugins**: Sistema extensible de plugins
- ✅ **Autenticación**: Sistema de autenticación y autorización
- ✅ **Caché**: Caché de resultados de comandos
- ✅ **Templates**: Plantillas de comandos reutilizables
- ✅ **Validación**: Validación y sanitización de comandos
- ✅ **Event Bus**: Sistema de eventos pub/sub
- ✅ **Clustering**: Soporte básico para múltiples instancias
- ✅ **Logging avanzado**: Sistema de logging profesional
- ✅ **Middleware**: Middleware para seguridad y logging
- ✅ **Scripts de mantenimiento**: Herramientas de mantenimiento automatizado
- ✅ **Monitor en tiempo real**: Monitor visual del estado del agente
- ✅ **Procesamiento con IA**: Procesamiento inteligente de comandos con LLMs
- ✅ **Búsqueda semántica**: Búsqueda de comandos usando embeddings
- ✅ **Aprendizaje de patrones**: Aprende de comandos exitosos para mejorar
- ✅ **Generación de código**: Genera código automáticamente desde descripciones
- ✅ **Resumen automático**: Resume resultados largos automáticamente
- ✅ **LLM Pipeline**: Pipeline profesional con PyTorch y Transformers
- ✅ **Fine-tuning**: Soporte para fine-tuning de modelos
- ✅ **Interfaz Gradio**: Interfaz web interactiva con Gradio
- ✅ **Completar Código**: Completa código automáticamente
- ✅ **Explicar Código**: Explica código en lenguaje natural
- ✅ **Corregir Código**: Corrige código con errores
- ✅ **Soporte GPU**: Detección automática y uso de GPU
- ✅ **Mixed Precision**: Soporte para entrenamiento con float16
- ✅ **Arquitectura Modular**: Separación clara de modelos, datos, entrenamiento y evaluación
- ✅ **Configuraciones YAML**: Gestión de hiperparámetros en archivos YAML
- ✅ **Experiment Tracking**: Preparado para WandB/TensorBoard
- ✅ **Callbacks System**: Sistema de callbacks para entrenamiento
- ✅ **Checkpointing**: Guardado automático de modelos
- ✅ **🤖 Personalidad Devin**: Sistema de personalidad inteligente con razonamiento
- ✅ **🔒 Seguridad Mejorada**: Detección y protección de secretos
- ✅ **📚 Comprensión de Código**: Sistema LSP-like para entender el código base
- ✅ **💬 Comunicación Inteligente**: Comunicación estratégica con el usuario
- ✅ **⚠️ Reporte de Problemas**: Detección y reporte automático de problemas de entorno
- ✅ **🔧 Sistema de Comandos Devin**: Comandos estructurados estilo Devin
- ✅ **📋 Sistema de Planificación**: Gestión de planes de trabajo con pasos y dependencias
- ✅ **🛠️ Gestor de Herramientas**: Detección automática de herramientas del sistema
- ✅ **📐 Analizador de Convenciones**: Detecta y sigue convenciones de código del proyecto
- ✅ **✅ Verificador de Cambios**: Verifica que todos los cambios están completos antes de reportar
- ✅ **📚 Verificación de Librerías**: Verifica librerías en requirements.txt antes de usar
- ✅ **🧪 Ejecutor de Tests**: Ejecuta tests automáticamente antes de reportar cambios
- ✅ **🔗 Rastreador de Referencias**: Rastrea y verifica referencias a código modificado
- ✅ **⚡ Ejecutor Paralelo**: Ejecuta comandos en paralelo cuando no tienen dependencias
- ✅ **🔍 Analizador de Contexto**: Analiza contexto del código antes de hacer cambios
- ✅ **✅ Verificador de Completitud**: Verifica que se cumplieron todos los requisitos
- ✅ **🔄 Gestor de Iteraciones**: Gestiona iteraciones hasta que los cambios sean correctos
- ✅ **🔍 Verificador Crítico**: Verificación crítica antes de reportar al usuario
- ✅ **🧠 Sistema de Triggers de Razonamiento**: Activa razonamiento automático en decisiones críticas
- ✅ **🎯 Verificador de Intención**: Verifica que se cumplió la intención del usuario
- ✅ **🔄 Integración Automática**: Integración automática en flujo de tareas
- ✅ **🛡️ Protector de Tests**: Previene modificación de tests a menos que se solicite explícitamente
- ✅ **🔗 Integración con CI**: Usa CI para testing cuando hay problemas de entorno
- ✅ **🔀 Gestor de Git**: Gestión de Git siguiendo las reglas específicas de Devin
- ✅ **📍 Verificador de Múltiples Ubicaciones**: Verifica que todas las ubicaciones fueron editadas
- ✅ **🌐 Integración con Navegador**: Inspecciona páginas web sin asumir contenido
- ✅ **📋 Verificador de Planificación**: Verifica que se tiene toda la información antes de sugerir un plan

## 📦 Instalación

### 🐳 Docker (Recomendado - Más Fácil)

```bash
# Clonar o navegar al directorio
cd agents/backend/onyx/server/features/cursor_agent_24_7

# Iniciar con Docker Compose
docker-compose up

# O usar Makefile
make quick-start
```

Ver [DOCKER.md](DOCKER.md) para documentación completa de Docker.

### Instalación Local (Sin Docker)

```bash
cd agents/backend/onyx/server/features/cursor_agent_24_7
pip install -r requirements.txt
```

### Instalación Mínima (Solo lo esencial)

```bash
pip install -r requirements-minimal.txt
```

### Instalación para Desarrollo

```bash
pip install -r requirements-dev.txt
```

### Instalación Rápida con UV (Más rápido que pip)

```bash
# Instalar UV
pip install uv

# Instalar dependencias
uv pip install -r requirements.txt
```

### Notas de Instalación

- **Python 3.10+** requerido (recomendado 3.12+)
- Algunas librerías requieren compilación (orjson, etc.)
- Para mejor rendimiento: `pip install --no-cache-dir -r requirements.txt`
- En Windows, algunas librerías pueden requerir Visual C++ Build Tools

## 🎯 Uso Rápido

### 🐳 Docker (Recomendado)

```bash
# Inicio rápido con Docker Compose
docker-compose up

# O con Makefile
make quick-start

# Stack completo (API + Workers + Redis)
make dev

# Producción
make prod
```

Ver [DOCKER.md](DOCKER.md) para más opciones de Docker.

### ⚡ Comando Simple (Sin Docker)

```bash
# Iniciar API en puerto 8024 (default)
python run.py

# O con opciones
python run.py --port 8080
python run.py --aws
python run.py --mode service
```

Luego abre tu navegador en: `http://localhost:8024`

### Modo API (Alternativa)

```bash
python main.py --mode api --port 8024
```

### Modo Servicio

```bash
python run.py --mode service
# O
python main.py --mode service
```

> 💡 **Tip**: Usa Docker para la mejor experiencia. Ver [DOCKER.md](DOCKER.md) para más información.

## 🖥️ Interfaz Web

Al iniciar en modo API, tendrás acceso a una interfaz web simple con:

- **Botón Iniciar**: Inicia el agente
- **Botón Pausar**: Pausa temporalmente el agente
- **Botón Detener**: Detiene completamente el agente
- **Campo de comando**: Escribe comandos y presiona Enter para agregarlos
- **Lista de tareas**: Ver todas las tareas ejecutadas

## 📡 API Endpoints

### Control del Agente

- `POST /api/start` - Iniciar el agente
- `POST /api/stop` - Detener el agente
- `POST /api/pause` - Pausar el agente
- `POST /api/resume` - Reanudar el agente
- `GET /api/status` - Obtener estado del agente

### Tareas

- `POST /api/tasks` - Agregar una nueva tarea
  ```json
  {
    "command": "tu comando aquí"
  }
  ```
- `GET /api/tasks?limit=50` - Obtener lista de tareas

## 🔧 Configuración

Puedes configurar el agente editando `AgentConfig`:

```python
from cursor_agent_24_7.core.agent import AgentConfig, CursorAgent

config = AgentConfig(
    check_interval=1.0,  # Segundos entre checks
    max_concurrent_tasks=5,  # Máximo de tareas simultáneas
    task_timeout=300.0,  # Timeout por tarea (segundos)
    auto_restart=True,  # Reiniciar automáticamente en caso de error
    persistent_storage=True,  # Guardar estado en disco
    storage_path="./data/agent_state.json"  # Ruta del archivo de estado
)

agent = CursorAgent(config)
```

## 🛠️ Ejecutar como Servicio del Sistema

### Windows

Usar **NSSM** (Non-Sucking Service Manager):

```bash
# Instalar NSSM
# Descargar de: https://nssm.cc/download

# Crear servicio
nssm install CursorAgent24_7 "C:\Python\python.exe" "C:\ruta\al\main.py" --mode service

# Iniciar servicio
nssm start CursorAgent24_7
```

O usar **Task Scheduler** de Windows.

### Linux (systemd)

Crear archivo `/etc/systemd/system/cursor-agent-24-7.service`:

```ini
[Unit]
Description=Cursor Agent 24/7
After=network.target

[Service]
Type=simple
User=tu_usuario
WorkingDirectory=/ruta/al/cursor_agent_24_7
ExecStart=/usr/bin/python3 /ruta/al/cursor_agent_24_7/main.py --mode service
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Luego:

```bash
sudo systemctl daemon-reload
sudo systemctl enable cursor-agent-24-7
sudo systemctl start cursor-agent-24-7
```

### macOS (launchd)

Crear archivo `~/Library/LaunchAgents/com.cursor.agent24-7.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.cursor.agent24-7</string>
    <key>ProgramArguments</key>
    <array>
        <string>/usr/bin/python3</string>
        <string>/ruta/al/cursor_agent_24_7/main.py</string>
        <string>--mode</string>
        <string>service</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
</dict>
</plist>
```

Luego:

```bash
launchctl load ~/Library/LaunchAgents/com.cursor.agent24-7.plist
```

## 🔌 Integración con Cursor API

**TODO**: Actualmente el agente está preparado para integrarse con la API de Cursor, pero la integración real necesita ser implementada.

Para integrar:

1. Obtener acceso a la API de Cursor
2. Implementar `CommandListener._listen_loop()` para recibir comandos reales
3. Implementar `TaskExecutor._run_command()` para ejecutar comandos reales

## 📊 Estado Persistente

El agente guarda su estado en `./data/agent_state.json` por defecto. Esto incluye:

- Estado actual del agente
- Todas las tareas (pendientes, en ejecución, completadas, fallidas)
- Resultados y errores

Al reiniciar, el agente carga automáticamente el estado guardado.

## 🛠️ Scripts de Utilidad

### Monitor en Tiempo Real

Monitorea el estado del agente en tiempo real:

```bash
python scripts/monitor.py
```

O con intervalo personalizado:

```bash
python scripts/monitor.py --interval 1.0
```

### Mantenimiento

Ejecutar tareas de mantenimiento:

```bash
# Limpiar tareas antiguas
python scripts/maintenance.py cleanup --days 30

# Verificar salud
python scripts/maintenance.py health

# Generar reporte
python scripts/maintenance.py report

# Ejecutar todo
python scripts/maintenance.py all
```

### Instalación como Servicio

Instalar el agente como servicio del sistema:

```bash
python scripts/install_service.py
```

### Interfaz Gradio

Lanzar la interfaz web interactiva con Gradio:

```bash
python scripts/launch_gradio.py
```

Con link público:

```bash
python scripts/launch_gradio.py --share
```

La interfaz estará disponible en `http://localhost:7860`

## 🤖 Mejoras Devin

El agente ahora incluye un sistema completo de personalidad Devin que lo hace más inteligente, seguro y profesional:

- **Sistema de Personalidad Devin**: Comportamiento similar a Devin con razonamiento interno
- **Comunicación Estratégica**: Comunica con el usuario cuando es necesario
- **Reporte de Problemas de Entorno**: Detecta y reporta problemas automáticamente
- **Modos de Operación**: Modo planificación y modo estándar
- **Comprensión de Código**: Sistema LSP-like para entender el código base
- **Seguridad Mejorada**: Detección y protección de secretos

Ver **[DEVIN_IMPROVEMENTS.md](DEVIN_IMPROVEMENTS.md)** para más detalles.

## 📚 Documentación Adicional

### 🚀 Despliegue y Orquestación
- **[ORCHESTRATION.md](ORCHESTRATION.md)**: Guía completa de orquestación (Docker, K8s, ECS)
- **[ORCHESTRATION_QUICK_START.md](ORCHESTRATION_QUICK_START.md)**: Inicio rápido de orquestación
- **[DOCKER.md](DOCKER.md)**: Docker Compose completo
- **[AWS_DEPLOYMENT.md](aws/AWS_DEPLOYMENT.md)**: Despliegue en AWS
- **[DEPLOYMENT.md](DEPLOYMENT.md)**: Guía de despliegue general

### 🏗️ Arquitectura y Características
- **[ARCHITECTURE.md](ARCHITECTURE.md)**: Arquitectura completa del sistema
- **[ADVANCED_FEATURES.md](ADVANCED_FEATURES.md)**: Características avanzadas
- **[MORE_IMPROVEMENTS.md](MORE_IMPROVEMENTS.md)**: Mejoras adicionales
- **[FINAL_IMPROVEMENTS.md](FINAL_IMPROVEMENTS.md)**: Mejoras finales (Service Discovery, Elasticsearch)
- **[API_IMPROVEMENTS.md](API_IMPROVEMENTS.md)**: Mejoras de API (Webhooks, Bulk, Cache)
- **[LIBRARIES_IMPROVEMENTS.md](LIBRARIES_IMPROVEMENTS.md)**: Mejoras de librerías (Serverless, Performance)
- **[MODULAR_ARCHITECTURE.md](MODULAR_ARCHITECTURE.md)**: Arquitectura modular

### 🤖 Funcionalidades
- **[DEVIN_IMPROVEMENTS.md](DEVIN_IMPROVEMENTS.md)**: Mejoras basadas en Devin
- **[FEATURES.md](FEATURES.md)**: Lista completa de características
- **[AI_FEATURES.md](AI_FEATURES.md)**: Funcionalidades de IA y Machine Learning
- **[ADVANCED_AI.md](ADVANCED_AI.md)**: Funcionalidades avanzadas de Deep Learning

### 📖 Referencias
- **[EXAMPLES.md](EXAMPLES.md)**: Ejemplos de uso
- **[API_REFERENCE.md](API_REFERENCE.md)**: Referencia completa de la API
- **[COMMANDS.md](COMMANDS.md)**: Comandos disponibles
- **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)**: Solución de problemas
- **[LIBRARIES.md](LIBRARIES.md)**: Documentación de librerías usadas
- **[QUICK_START.md](QUICK_START.md)**: Guía de inicio rápido

## 🐛 Troubleshooting

Para troubleshooting detallado, consulta [TROUBLESHOOTING.md](TROUBLESHOOTING.md).

### Problemas Comunes

**El agente no inicia:**
- Verifica que el puerto 8024 esté disponible
- Revisa los logs: `tail -f logs/agent.log`
- Asegúrate de tener todas las dependencias instaladas

**Las tareas no se ejecutan:**
- Verifica que el agente esté en estado "running": `curl http://localhost:8024/api/status`
- Revisa los logs para ver errores de ejecución
- Verifica health: `curl http://localhost:8024/api/health`

**El servicio no persiste:**
- Verifica los permisos de escritura en el directorio de datos
- Revisa la configuración del servicio del sistema
- Asegúrate de que `persistent_storage=True` en la configuración

## 📝 Licencia

Parte del proyecto Blatam Academy.

## 🤝 Contribuir

Las contribuciones son bienvenidas. Por favor:

1. Fork el proyecto
2. Crea una rama para tu feature
3. Commit tus cambios
4. Push a la rama
5. Abre un Pull Request

## 📞 Soporte

Para soporte, abre un issue en el repositorio o contacta al equipo de Blatam Academy.

