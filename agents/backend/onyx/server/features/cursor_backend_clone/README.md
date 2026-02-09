# Cursor Agent 24/7 🤖

Agente persistente que escucha comandos desde la ventana de Cursor y los ejecuta continuamente, incluso cuando la computadora está apagada (como servicio).

## 🚀 Características

- ✅ **Escucha comandos desde Cursor**: Se activa automáticamente cuando escribes algo en Cursor
- ✅ **Integración MCP**: Soporte completo para Model Context Protocol para integración nativa con Cursor IDE
  - Prometheus Metrics Export
  - Adaptive Rate Limiting
  - Request Deduplication
  - Advanced Metrics con percentiles
  - WebSocket Connection Limits
- ✅ **Ejecución continua**: Ejecuta tareas sin parar, 24/7
- ✅ **Control simple**: Botón para iniciar/detener fácilmente
- ✅ **Servicio persistente**: Puede correr en background como servicio del sistema
- ✅ **API REST**: Control completo vía API
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

## 📦 Instalación

### Instalación Completa (Recomendado)

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

### Modo API (Recomendado)

```bash
python main.py --mode api --port 8024
```

Luego abre tu navegador en: `http://localhost:8024`

### Modo Servicio

```bash
python main.py --mode service
```

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

## 🔌 Integración con Cursor IDE (MCP)

El agente ahora incluye soporte completo para **Model Context Protocol (MCP)**, permitiendo integración nativa con Cursor IDE.

### Iniciar con soporte MCP

```bash
python main.py --mode api --enable-mcp --mcp-port 8025
```

Esto iniciará:
- API REST en el puerto 8024
- Servidor MCP en el puerto 8025

### Configurar Cursor IDE

1. Abre Cursor IDE Settings
2. Ve a Extensions → MCP Servers
3. Agrega configuración:

```json
{
  "name": "cursor-agent-24-7",
  "url": "http://localhost:8025",
  "transport": "http"
}
```

### Uso desde Cursor

Una vez configurado, puedes ejecutar comandos directamente desde Cursor IDE usando la paleta de comandos (Ctrl+Shift+P) y buscando "MCP: Execute Command".

Para más detalles, consulta [MCP_INTEGRATION.md](MCP_INTEGRATION.md).

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

## 📚 Documentación Adicional

- **[FEATURES.md](FEATURES.md)**: Lista completa de características
- **[AI_FEATURES.md](AI_FEATURES.md)**: Funcionalidades de IA y Machine Learning
- **[ADVANCED_AI.md](ADVANCED_AI.md)**: Funcionalidades avanzadas de Deep Learning y Transformers
- **[MODULAR_ARCHITECTURE.md](MODULAR_ARCHITECTURE.md)**: Arquitectura modular del proyecto
- **[EXAMPLES.md](EXAMPLES.md)**: Ejemplos de uso
- **[API_REFERENCE.md](API_REFERENCE.md)**: Referencia completa de la API
- **[DEPLOYMENT.md](DEPLOYMENT.md)**: Guía de despliegue
- **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)**: Solución de problemas
- **[LIBRARIES.md](LIBRARIES.md)**: Documentación de librerías usadas
- **[QUICK_START.md](QUICK_START.md)**: Guía de inicio rápido
- **[MCP_INTEGRATION.md](MCP_INTEGRATION.md)**: Guía de integración MCP con Cursor IDE

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

