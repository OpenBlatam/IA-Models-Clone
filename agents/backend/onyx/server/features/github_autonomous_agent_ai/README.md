# GitHub Autonomous Agent AI

Agente autónomo que se conecta a cualquier repositorio de GitHub y ejecuta instrucciones de forma continua, incluso con la computadora apagada, hasta que el usuario le da la orden de parar.

## 🚀 Características

- ✅ **Conexión a cualquier repositorio de GitHub** - Soporta cualquier repositorio público o privado
- ✅ **Recepción de instrucciones desde el frontend** - Interfaz web para enviar comandos
- ✅ **Ejecución continua de tareas** - El agente trabaja sin parar hasta que se le ordene detenerse
- ✅ **Funcionamiento como servicio/daemon** - Puede ejecutarse en segundo plano
- ✅ **Control de inicio/parada** - Control total desde el frontend o API
- ✅ **Sistema de cola de tareas** - Gestión eficiente de múltiples tareas
- ✅ **Persistencia de datos** - Las tareas se guardan en base de datos SQLite

## 📋 Requisitos

- Python 3.8+
- Token de GitHub (opcional, para repositorios privados)
- FastAPI y Uvicorn

## 🔧 Instalación

```bash
# Instalar dependencias
pip install -r requirements.txt

# Configurar variables de entorno (automático)
python setup_env.py

# O manualmente, copia env.example a .env
# cp env.example .env
```

### Configuración de GitHub OAuth

Las credenciales de GitHub OAuth ya están configuradas:
- **Client ID**: `Ov23liSy9XyIj3dD0dQc`
- **Client Secret**: `6ed948f00e7662bbba0eacd356e60747dd12f08f`

**⚠️ IMPORTANTE**: Asegúrate de que en tu GitHub OAuth App la **Authorization callback URL** sea exactamente:
```
http://localhost:8025/api/github/auth/callback
```

Ver `SETUP_GITHUB_OAUTH.md` para más detalles.

## 🎯 Uso

### Modo Servicio (Daemon)

Ejecuta el agente como servicio persistente:

```bash
python main.py --mode service
```

### Modo API

Ejecuta el servidor API:

```bash
python main.py --mode api --port 8025
```

## 📡 API Endpoints

### Agente

- `GET /api/agent/status` - Obtener estado del agente
- `POST /api/agent/start` - Iniciar el agente
- `POST /api/agent/stop` - Detener el agente
- `POST /api/agent/pause` - Pausar el agente
- `POST /api/agent/resume` - Reanudar el agente

### Tareas

- `POST /api/tasks/` - Crear una nueva tarea
- `GET /api/tasks/` - Listar tareas
- `GET /api/tasks/{task_id}` - Obtener una tarea específica

## 📝 Ejemplo de Uso

### Crear una tarea

```bash
curl -X POST "http://localhost:8025/api/tasks/" \
  -H "Content-Type: application/json" \
  -d '{
    "repository": "owner/repo",
    "instruction": "Crear un archivo README.md con información del proyecto",
    "metadata": {
      "file_path": "README.md",
      "file_content": "# Mi Proyecto\n\nDescripción del proyecto"
    }
  }'
```

### Iniciar el agente

```bash
curl -X POST "http://localhost:8025/api/agent/start"
```

### Ver estado

```bash
curl "http://localhost:8025/api/agent/status"
```

## 🏗️ Arquitectura

```
github_autonomous_agent_ai/
├── api/                    # API FastAPI
│   ├── routes/            # Rutas de la API
│   └── models/            # Modelos Pydantic
├── core/                   # Lógica principal
│   ├── agent.py           # Agente principal
│   ├── service.py         # Servicio persistente
│   ├── github_client.py   # Cliente de GitHub
│   ├── task_queue.py      # Cola de tareas
│   └── task_executor.py   # Ejecutor de tareas
├── config/                 # Configuración
│   └── settings.py        # Settings de la aplicación
├── frontend/               # Frontend (próximamente)
├── main.py                 # Punto de entrada
└── requirements.txt        # Dependencias
```

## 🔐 Configuración

Variables de entorno:

- `GITHUB_TOKEN` - Token de GitHub (opcional)
- `API_PORT` - Puerto del servidor API (default: 8025)
- `AGENT_POLL_INTERVAL` - Intervalo de polling en segundos (default: 5)
- `AGENT_MAX_CONCURRENT_TASKS` - Máximo de tareas concurrentes (default: 3)
- `STORAGE_PATH` - Ruta de almacenamiento (default: ./data)

## 🚧 Próximas Mejoras

- [ ] Frontend completo con React/Next.js
- [ ] Soporte para más tipos de instrucciones
- [ ] Integración con modelos de IA para procesar instrucciones naturales
- [ ] Sistema de notificaciones
- [ ] Dashboard de monitoreo
- [ ] Soporte para múltiples repositorios simultáneos

## 📄 Licencia

MIT


