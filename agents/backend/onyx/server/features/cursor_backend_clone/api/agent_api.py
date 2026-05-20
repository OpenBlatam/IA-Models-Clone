"""
Agent API - REST API para control del agente
============================================

API REST para controlar el agente 24/7 con endpoints simples.
"""

import logging
from typing import Optional, Dict, Any, List
from datetime import datetime
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field, field_validator
import uvicorn

# Usar structlog para logging estructurado
try:
    import structlog
    logger = structlog.get_logger(__name__)
except ImportError:
    logger = logging.getLogger(__name__)

from fastapi import WebSocket, WebSocketDisconnect

from ..core.agent import CursorAgent, AgentConfig
from ..core.services.persistent_service import PersistentService
from ..core.infrastructure.messaging.websocket import WebSocketManager


# Modelos Pydantic
class TaskRequest(BaseModel):
    """Request para agregar tarea con validación mejorada"""
    command: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="Comando a ejecutar",
        examples=["python:print('Hello World')", "shell:ls -la"]
    )
    priority: Optional[int] = Field(
        default=0,
        ge=0,
        le=10,
        description="Prioridad de la tarea (0-10, mayor es más prioritario)"
    )
    timeout: Optional[float] = Field(
        default=None,
        gt=0,
        le=3600,
        description="Timeout personalizado en segundos (máximo 1 hora)"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Metadatos adicionales para la tarea"
    )
    
    @field_validator('command')
    @classmethod
    def validate_command(cls, v: str) -> str:
        """Validar que el comando no esté vacío después de trim"""
        v = v.strip()
        if not v:
            raise ValueError("Command cannot be empty or whitespace only")
        return v


class TaskResponse(BaseModel):
    """Response de tarea con información completa"""
    task_id: str = Field(..., description="Identificador único de la tarea")
    status: str = Field(..., description="Estado actual de la tarea")
    message: str = Field(..., description="Mensaje descriptivo")
    timestamp: Optional[str] = Field(
        default=None,
        description="Timestamp ISO de cuando se creó la tarea"
    )


class StatusResponse(BaseModel):
    """Response de estado con métricas opcionales"""
    status: str = Field(..., description="Estado actual del agente")
    running: bool = Field(..., description="Si el agente está corriendo")
    tasks_total: int = Field(..., ge=0, description="Total de tareas")
    tasks_pending: int = Field(..., ge=0, description="Tareas pendientes")
    tasks_running: int = Field(..., ge=0, description="Tareas en ejecución")
    tasks_completed: int = Field(..., ge=0, description="Tareas completadas")
    tasks_failed: int = Field(..., ge=0, description="Tareas fallidas")
    metrics: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Métricas adicionales si están disponibles"
    )


def create_app(agent: Optional[CursorAgent] = None) -> FastAPI:
    """Crear aplicación FastAPI"""
    
    app = FastAPI(
        title="Cursor Agent 24/7 API",
        description="""
        API REST para controlar el agente persistente de Cursor.
        
        ## Características
        
        - ✅ Ejecución de comandos Python, Shell y HTTP
        - ✅ Gestión de tareas con prioridades y timeouts
        - ✅ Health checks y métricas del sistema
        - ✅ WebSocket para comunicación en tiempo real
        - ✅ Circuit breaker para resiliencia
        - ✅ Rate limiting y validación de seguridad
        
        ## Autenticación
        
        Actualmente la API no requiere autenticación. En producción, se recomienda
        agregar autenticación mediante tokens o API keys.
        
        ## Documentación
        
        - Swagger UI: `/docs`
        - ReDoc: `/redoc`
        - OpenAPI Schema: `/openapi.json`
        """,
        version="1.0.0",
        contact={
            "name": "Cursor Agent Support",
            "url": "https://cursor.sh",
        },
        license_info={
            "name": "MIT",
        },
        servers=[
            {
                "url": "http://localhost:8024",
                "description": "Servidor de desarrollo local"
            },
            {
                "url": "https://api.example.com",
                "description": "Servidor de producción"
            }
        ],
        tags_metadata=[
            {
                "name": "tasks",
                "description": "Operaciones relacionadas con tareas (agregar, listar, obtener estado)"
            },
            {
                "name": "status",
                "description": "Estado del agente y estadísticas del sistema"
            },
            {
                "name": "health",
                "description": "Health checks y verificación de salud del sistema"
            },
            {
                "name": "metrics",
                "description": "Métricas y estadísticas del agente"
            },
            {
                "name": "resources",
                "description": "Gestión de recursos del sistema (memoria, CPU, limpieza)"
            },
            {
                "name": "cache",
                "description": "Operaciones de caché (estadísticas, limpieza)"
            },
            {
                "name": "websocket",
                "description": "Estadísticas de conexiones WebSocket"
            },
            {
                "name": "ai",
                "description": "Endpoints de procesamiento con IA"
            },
            {
                "name": "patterns",
                "description": "Patrones aprendidos y predicciones"
            }
        ]
    )
    
    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Middleware adicional
    try:
        from ..core.utils.middleware.middleware import (
            LoggingMiddleware,
            SecurityHeadersMiddleware,
            ErrorHandlingMiddleware
        )
        app.add_middleware(LoggingMiddleware)
        app.add_middleware(SecurityHeadersMiddleware)
        app.add_middleware(ErrorHandlingMiddleware)
    except ImportError:
        pass  # Middleware opcional
    
    # Agente global
    if agent is None:
        agent = CursorAgent(AgentConfig())
        app.state.agent = agent
    else:
        app.state.agent = agent
    
    # WebSocket Manager
    ws_manager = WebSocketManager()
    app.state.ws_manager = ws_manager
    
    # Registrar handlers de WebSocket
    async def handle_command_message(websocket: WebSocket, message: dict):
        """Manejar mensaje de comando desde WebSocket"""
        command = message.get("command")
        if command:
            task_id = await agent.add_task(command)
            await ws_manager.send_personal_message({
                "type": "task_added",
                "task_id": task_id,
                "command": command[:50]
            }, websocket)
    
    ws_manager.register_handler("command", handle_command_message)
    
    # Endpoints
    
    @app.get("/", response_class=HTMLResponse)
    async def root():
        """Página principal con control simple"""
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Cursor Agent 24/7</title>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <style>
                * { margin: 0; padding: 0; box-sizing: border-box; }
                body {
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    min-height: 100vh;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    padding: 20px;
                }
                .container {
                    background: white;
                    border-radius: 20px;
                    padding: 40px;
                    box-shadow: 0 20px 60px rgba(0,0,0,0.3);
                    max-width: 500px;
                    width: 100%;
                }
                h1 {
                    color: #333;
                    margin-bottom: 10px;
                    font-size: 28px;
                }
                .subtitle {
                    color: #666;
                    margin-bottom: 30px;
                    font-size: 14px;
                }
                .status {
                    background: #f5f5f5;
                    padding: 15px;
                    border-radius: 10px;
                    margin-bottom: 20px;
                    font-size: 14px;
                }
                .status-item {
                    display: flex;
                    justify-content: space-between;
                    margin-bottom: 8px;
                }
                .status-label {
                    color: #666;
                }
                .status-value {
                    font-weight: 600;
                    color: #333;
                }
                .status-value.running { color: #10b981; }
                .status-value.stopped { color: #ef4444; }
                .status-value.paused { color: #f59e0b; }
                .controls {
                    display: flex;
                    gap: 10px;
                    margin-bottom: 20px;
                }
                button {
                    flex: 1;
                    padding: 15px;
                    border: none;
                    border-radius: 10px;
                    font-size: 16px;
                    font-weight: 600;
                    cursor: pointer;
                    transition: all 0.3s;
                }
                .btn-start {
                    background: #10b981;
                    color: white;
                }
                .btn-start:hover { background: #059669; }
                .btn-stop {
                    background: #ef4444;
                    color: white;
                }
                .btn-stop:hover { background: #dc2626; }
                .btn-pause {
                    background: #f59e0b;
                    color: white;
                }
                .btn-pause:hover { background: #d97706; }
                button:disabled {
                    opacity: 0.5;
                    cursor: not-allowed;
                }
                .command-input {
                    width: 100%;
                    padding: 12px;
                    border: 2px solid #e5e7eb;
                    border-radius: 10px;
                    font-size: 14px;
                    margin-bottom: 10px;
                }
                .command-input:focus {
                    outline: none;
                    border-color: #667eea;
                }
                .tasks {
                    margin-top: 20px;
                    max-height: 300px;
                    overflow-y: auto;
                }
                .task-item {
                    background: #f9fafb;
                    padding: 10px;
                    border-radius: 8px;
                    margin-bottom: 8px;
                    font-size: 12px;
                }
                .task-command {
                    color: #333;
                    margin-bottom: 4px;
                }
                .task-status {
                    color: #666;
                    font-size: 11px;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🤖 Cursor Agent 24/7</h1>
                <p class="subtitle">Agente persistente que ejecuta comandos sin parar</p>
                
                <div class="status" id="status">
                    <div class="status-item">
                        <span class="status-label">Estado:</span>
                        <span class="status-value" id="status-value">Cargando...</span>
                    </div>
                    <div class="status-item">
                        <span class="status-label">Tareas:</span>
                        <span class="status-value" id="tasks-value">-</span>
                    </div>
                </div>
                
                <div class="controls">
                    <button class="btn-start" id="btn-start" onclick="startAgent()">▶️ Iniciar</button>
                    <button class="btn-pause" id="btn-pause" onclick="pauseAgent()" disabled>⏸️ Pausar</button>
                    <button class="btn-stop" id="btn-stop" onclick="stopAgent()" disabled>⏹️ Detener</button>
                </div>
                
                <input type="text" class="command-input" id="command-input" 
                       placeholder="Escribe un comando y presiona Enter..." 
                       onkeypress="if(event.key==='Enter') sendCommand()">
                
                <div class="tasks" id="tasks"></div>
            </div>
            
            <script>
                let statusInterval;
                
                async function updateStatus() {
                    try {
                        const response = await fetch('/api/status');
                        const data = await response.json();
                        
                        document.getElementById('status-value').textContent = data.status.toUpperCase();
                        document.getElementById('status-value').className = 'status-value ' + data.status;
                        document.getElementById('tasks-value').textContent = 
                            `${data.tasks_completed}/${data.tasks_total} completadas`;
                        
                        // Actualizar botones
                        const isRunning = data.status === 'running';
                        const isPaused = data.status === 'paused';
                        
                        document.getElementById('btn-start').disabled = isRunning || isPaused;
                        document.getElementById('btn-pause').disabled = !isRunning;
                        document.getElementById('btn-stop').disabled = !isRunning && !isPaused;
                        
                        // Cargar tareas
                        loadTasks();
                    } catch (error) {
                        console.error('Error updating status:', error);
                    }
                }
                
                async function loadTasks() {
                    try {
                        const response = await fetch('/api/tasks?limit=10');
                        const data = await response.json();
                        
                        const tasksDiv = document.getElementById('tasks');
                        tasksDiv.innerHTML = data.tasks.map(task => `
                            <div class="task-item">
                                <div class="task-command">${task.command.substring(0, 50)}${task.command.length > 50 ? '...' : ''}</div>
                                <div class="task-status">${task.status} - ${new Date(task.timestamp).toLocaleString()}</div>
                            </div>
                        `).join('');
                    } catch (error) {
                        console.error('Error loading tasks:', error);
                    }
                }
                
                async function startAgent() {
                    try {
                        await fetch('/api/start', { method: 'POST' });
                        updateStatus();
                    } catch (error) {
                        alert('Error al iniciar: ' + error.message);
                    }
                }
                
                async function stopAgent() {
                    if (!confirm('¿Detener el agente?')) return;
                    try {
                        await fetch('/api/stop', { method: 'POST' });
                        updateStatus();
                    } catch (error) {
                        alert('Error al detener: ' + error.message);
                    }
                }
                
                async function pauseAgent() {
                    try {
                        await fetch('/api/pause', { method: 'POST' });
                        updateStatus();
                    } catch (error) {
                        alert('Error al pausar: ' + error.message);
                    }
                }
                
                async function sendCommand() {
                    const input = document.getElementById('command-input');
                    const command = input.value.trim();
                    if (!command) return;
                    
                    try {
                        await fetch('/api/tasks', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ command })
                        });
                        input.value = '';
                        updateStatus();
                    } catch (error) {
                        alert('Error al enviar comando: ' + error.message);
                    }
                }
                
                // Actualizar estado cada 2 segundos
                updateStatus();
                statusInterval = setInterval(updateStatus, 2000);
            </script>
        </body>
        </html>
        """
    
    @app.post("/api/start")
    async def start_agent():
        """Iniciar el agente"""
        try:
            agent = app.state.agent
            await agent.start()
            return {"status": "started", "message": "Agent started successfully"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/api/stop")
    async def stop_agent():
        """Detener el agente"""
        try:
            agent = app.state.agent
            await agent.stop()
            return {"status": "stopped", "message": "Agent stopped successfully"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/api/pause")
    async def pause_agent():
        """Pausar el agente"""
        try:
            agent = app.state.agent
            await agent.pause()
            return {"status": "paused", "message": "Agent paused successfully"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/api/resume")
    async def resume_agent():
        """Reanudar el agente"""
        try:
            agent = app.state.agent
            await agent.resume()
            return {"status": "resumed", "message": "Agent resumed successfully"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get(
        "/api/status",
        response_model=StatusResponse,
        tags=["status"],
        summary="Obtener estado del agente",
        description="Retorna el estado actual del agente incluyendo estadísticas de tareas"
    )
    async def get_status():
        """
        Obtener estado del agente.
        
        Incluye información sobre tareas, métricas y notificaciones.
        """
        try:
            agent = app.state.agent
            status = await agent.get_status()
            return status
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post(
        "/api/tasks",
        response_model=TaskResponse,
        tags=["tasks"],
        summary="Agregar nueva tarea",
        description="""
        Agrega una nueva tarea al agente para ejecución.
        
        El comando puede ser:
        - Código Python (prefijo `python:` o `py:`)
        - Comando shell (prefijo `shell:` o `sh:`)
        - Llamada HTTP (URL completa con `http://` o `https://`)
        
        La tarea se ejecutará de forma asíncrona y se puede consultar su estado
        usando el `task_id` retornado.
        """,
        response_description="Información de la tarea creada incluyendo su ID único"
    )
    async def add_task(request: TaskRequest):
        """
        Agregar una tarea con validación mejorada.
        
        Valida el comando, prioridad y timeout antes de agregar la tarea.
        """
        try:
            agent = app.state.agent
            
            # Aplicar timeout personalizado si se proporciona
            original_timeout = agent.config.task_timeout
            if request.timeout:
                agent.config.task_timeout = request.timeout
            
            try:
                task_id = await agent.add_task(
                    command=request.command,
                    priority=request.priority,
                    timeout=request.timeout,
                    metadata=request.metadata
                )
                return TaskResponse(
                    task_id=task_id,
                    status="pending",
                    message="Task added successfully",
                    timestamp=datetime.now().isoformat()
                )
            finally:
                # Restaurar timeout original
                if request.timeout:
                    agent.config.task_timeout = original_timeout
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/api/metrics")
    async def get_metrics():
        """
        Obtener métricas detalladas del agente.
        
        Returns:
            Métricas completas del sistema
        """
        try:
            agent = app.state.agent
            if not agent.metrics:
                return {"message": "Metrics collector not available"}
            
            return agent.metrics.get_summary()
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get(
        "/api/websocket/stats",
        tags=["websocket"],
        summary="Estadísticas de WebSocket",
        description="Retorna estadísticas de conexiones WebSocket activas"
    )
    async def get_websocket_stats():
        """
        Obtener estadísticas de conexiones WebSocket.
        
        Returns:
            Estadísticas de WebSocket
        """
        try:
            ws_manager = app.state.ws_manager
            return ws_manager.get_stats()
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get(
        "/api/tasks",
        tags=["tasks"],
        summary="Listar tareas",
        description="Obtiene una lista de tareas con opción de filtrar por estado y limitar resultados"
    )
    async def get_tasks(limit: int = 50):
        """
        Obtener lista de tareas.
        
        Args:
            limit: Número máximo de tareas a retornar (default: 50, max: 1000)
        
        Returns:
            Lista de tareas ordenadas por timestamp (más recientes primero)
        """
        try:
            # Validar límite
            limit = min(max(1, limit), 1000)
            
            agent = app.state.agent
            tasks = await agent.get_tasks(limit=limit)
            return {"tasks": tasks, "count": len(tasks), "limit": limit}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        """Endpoint WebSocket para comunicación en tiempo real"""
        await ws_manager.listen(websocket)
    
    @app.get("/api/notifications")
    async def get_notifications(limit: int = 50, unread_only: bool = False):
        """Obtener notificaciones"""
        try:
            agent = app.state.agent
            if agent.notifications:
                notifications = agent.notifications.get_notifications(
                    unread_only=unread_only,
                    limit=limit
                )
                return {
                    "notifications": [
                        {
                            "id": n.id,
                            "title": n.title,
                            "message": n.message,
                            "level": n.level.value,
                            "timestamp": n.timestamp.isoformat(),
                            "read": n.read,
                            "metadata": n.metadata
                        }
                        for n in notifications
                    ]
                }
            return {"notifications": []}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/api/notifications/{notification_id}/read")
    async def mark_notification_read(notification_id: str):
        """Marcar notificación como leída"""
        try:
            agent = app.state.agent
            if agent.notifications:
                success = agent.notifications.mark_as_read(notification_id)
                return {"success": success}
            return {"success": False}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/api/metrics")
    async def get_metrics():
        """Obtener métricas del agente"""
        try:
            agent = app.state.agent
            if agent.metrics:
                return agent.metrics.get_summary()
            return {"message": "Metrics not available"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/api/ws/connections")
    async def get_ws_connections():
        """Obtener información de conexiones WebSocket"""
        try:
            return {
                "count": ws_manager.get_connection_count(),
                "connections": ws_manager.get_connections_info()
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get(
        "/api/health",
        tags=["health"],
        summary="Health check básico",
        description="Verificación rápida del estado de salud del agente usando HealthChecker"
    )
    async def health_check():
        """Health check del agente"""
        try:
            from ..core.infrastructure.monitoring.health import HealthChecker
            checker = HealthChecker(agent)
            health = await checker.check_all()
            return health
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/api/export/tasks")
    async def export_tasks(format: str = "json", limit: int = 1000):
        """Exportar tareas"""
        try:
            from ..core.services.exporters import DataExporter
            from pathlib import Path
            import tempfile
            
            exporter = DataExporter(agent)
            output_dir = Path("./data/exports")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = output_dir / f"tasks_export_{timestamp}.{format}"
            
            file_path = await exporter.export_tasks(
                str(output_path),
                format=format,
                limit=limit
            )
            
            return {
                "success": True,
                "file_path": file_path,
                "format": format,
                "message": f"Exported {limit} tasks to {file_path}"
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/api/export/status")
    async def export_status():
        """Exportar estado del agente"""
        try:
            from ..core.services.exporters import DataExporter
            from pathlib import Path
            
            exporter = DataExporter(agent)
            output_dir = Path("./data/exports")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = output_dir / f"status_export_{timestamp}.json"
            
            file_path = await exporter.export_status(str(output_path))
            
            return {
                "success": True,
                "file_path": file_path,
                "message": f"Exported status to {file_path}"
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/api/rate-limit/stats")
    async def get_rate_limit_stats():
        """Obtener estadísticas de rate limiting"""
        try:
            if agent.rate_limiter:
                return agent.rate_limiter.get_stats()
            return {"message": "Rate limiter not available"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/api/scheduler/tasks")
    async def get_scheduled_tasks():
        """Obtener tareas programadas"""
        try:
            if agent.scheduler:
                return {"tasks": agent.scheduler.get_scheduled_tasks()}
            return {"message": "Scheduler not available"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/api/scheduler/tasks")
    async def schedule_task(
        name: str,
        command: str,
        schedule_type: str,
        schedule_value: str,
        max_runs: Optional[int] = None
    ):
        """Programar una tarea"""
        try:
            from ..core.infrastructure.scheduling.scheduler import TaskScheduler, ScheduleType
            
            if not agent.scheduler:
                raise HTTPException(status_code=400, detail="Scheduler not available")
            
            schedule_enum = ScheduleType(schedule_type)
            task_id = agent.scheduler.schedule_task(
                name=name,
                command=command,
                schedule_type=schedule_enum,
                schedule_value=schedule_value,
                max_runs=max_runs
            )
            
            return {"task_id": task_id, "message": "Task scheduled successfully"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/api/backups")
    async def list_backups():
        """Listar backups"""
        try:
            if agent.backup_manager:
                return {"backups": agent.backup_manager.list_backups()}
            return {"message": "Backup manager not available"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/api/backups/create")
    async def create_backup(name: Optional[str] = None):
        """Crear backup"""
        try:
            if agent.backup_manager:
                backup_path = await agent.backup_manager.create_backup(name)
                return {"success": True, "backup_path": backup_path}
            return {"success": False, "message": "Backup manager not available"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/api/backups/{backup_name}/restore")
    async def restore_backup(backup_name: str):
        """Restaurar backup"""
        try:
            if agent.backup_manager:
                success = await agent.backup_manager.restore_backup(backup_name)
                return {"success": success}
            return {"success": False, "message": "Backup manager not available"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/api/plugins")
    async def get_plugins():
        """Obtener lista de plugins"""
        try:
            if agent.plugin_manager:
                return {"plugins": agent.plugin_manager.get_plugins()}
            return {"message": "Plugin manager not available"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/api/templates")
    async def get_templates(category: Optional[str] = None):
        """Obtener plantillas de comandos"""
        try:
            if agent.template_manager:
                return {"templates": agent.template_manager.list_templates(category)}
            return {"message": "Template manager not available"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/api/templates")
    async def create_template(
        name: str,
        description: str,
        template: str,
        variables: Optional[List[str]] = None,
        category: str = "general"
    ):
        """Crear plantilla de comando"""
        try:
            if agent.template_manager:
                template_id = agent.template_manager.create_template(
                    name=name,
                    description=description,
                    template=template,
                    variables=variables,
                    category=category
                )
                return {"template_id": template_id, "message": "Template created"}
            return {"message": "Template manager not available"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/api/templates/{template_id}/render")
    async def render_template(template_id: str, variables: Dict[str, Any]):
        """Renderizar plantilla"""
        try:
            if agent.template_manager:
                rendered = agent.template_manager.render_template(template_id, variables)
                if rendered:
                    return {"command": rendered}
                raise HTTPException(status_code=400, detail="Template not found or invalid")
            return {"message": "Template manager not available"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/api/cache/stats")
    async def get_cache_stats():
        """Obtener estadísticas del caché"""
        try:
            if agent.command_cache:
                return await agent.command_cache.get_stats()
            return {"message": "Cache not available"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/api/cache/clear")
    async def clear_cache():
        """Limpiar caché"""
        try:
            if agent.command_cache:
                await agent.command_cache.cache.clear()
                return {"success": True, "message": "Cache cleared"}
            return {"message": "Cache not available"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/api/events")
    async def get_events(event_type: Optional[str] = None, limit: int = 100):
        """Obtener eventos del event bus"""
        try:
            if agent.event_bus:
                from ..core.infrastructure.messaging.event_bus import EventType
                event_type_enum = EventType(event_type) if event_type else None
                events = agent.event_bus.get_events(event_type_enum, limit)
                return {
                    "events": [
                        {
                            "type": e.type.value,
                            "data": e.data,
                            "timestamp": e.timestamp.isoformat(),
                            "source": e.source
                        }
                        for e in events
                    ]
                }
            return {"message": "Event bus not available"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/api/cluster/info")
    async def get_cluster_info():
        """Obtener información del cluster"""
        try:
            if agent.cluster_manager:
                return agent.cluster_manager.get_cluster_info()
            return {"message": "Cluster manager not available"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/api/events/stats")
    async def get_event_stats():
        """Obtener estadísticas del event bus"""
        try:
            if agent.event_bus:
                return agent.event_bus.get_stats()
            return {"message": "Event bus not available"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/api/alerts")
    async def get_alerts(severity: Optional[str] = None, active_only: bool = True):
        """Obtener alertas"""
        try:
            if agent.alert_manager:
                from ..core.utils.alerts.alerting import AlertSeverity
                severity_enum = AlertSeverity(severity) if severity else None
                
                if active_only:
                    alerts = agent.alert_manager.get_active_alerts(severity_enum)
                else:
                    alerts = agent.alert_manager.get_alert_history(limit=100)
                
                return {
                    "alerts": [
                        {
                            "id": a.id,
                            "rule_name": a.rule_name,
                            "severity": a.severity.value,
                            "message": a.message,
                            "value": a.value,
                            "threshold": a.threshold,
                            "timestamp": a.timestamp.isoformat(),
                            "acknowledged": a.acknowledged,
                            "resolved": a.resolved
                        }
                        for a in alerts
                    ]
                }
            return {"message": "Alert manager not available"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/api/alerts/{alert_id}/acknowledge")
    async def acknowledge_alert(alert_id: str):
        """Reconocer alerta"""
        try:
            if agent.alert_manager:
                success = agent.alert_manager.acknowledge_alert(alert_id)
                return {"success": success}
            return {"success": False}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/api/alerts/{alert_id}/resolve")
    async def resolve_alert(alert_id: str):
        """Resolver alerta"""
        try:
            if agent.alert_manager:
                success = agent.alert_manager.resolve_alert(alert_id)
                return {"success": success}
            return {"success": False}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/api/config")
    async def get_config(key: Optional[str] = None):
        """Obtener configuración"""
        try:
            agent = app.state.agent
            if agent.config_manager:
                if key:
                    value = agent.config_manager.get(key)
                    return {"key": key, "value": value}
                else:
                    return {"config": agent.config_manager.config}
            return {"message": "Config manager not available"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/api/resources/memory")
    async def get_memory_info():
        """Obtener información de memoria del sistema"""
        try:
            agent = app.state.agent
            if not agent.resource_manager:
                return {"message": "Resource manager not available"}
            return agent.resource_manager.get_memory_info()
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/api/resources/cpu")
    async def get_cpu_info():
        """Obtener información de CPU"""
        try:
            agent = app.state.agent
            if not agent.resource_manager:
                return {"message": "Resource manager not available"}
            return agent.resource_manager.get_cpu_info()
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/api/resources/system")
    async def get_system_info():
        """Obtener información completa del sistema"""
        try:
            agent = app.state.agent
            if not agent.resource_manager:
                return {"message": "Resource manager not available"}
            return agent.resource_manager.get_system_info()
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/api/resources/cleanup")
    async def trigger_cleanup():
        """Forzar limpieza de recursos del sistema"""
        try:
            agent = app.state.agent
            if not agent.resource_manager:
                return {"message": "Resource manager not available", "success": False}
            await agent.resource_manager.cleanup()
            return {"success": True, "message": "Cleanup completed"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get(
        "/api/health/detailed",
        tags=["health"],
        summary="Health check detallado",
        description="""
        Verificación detallada del estado de salud del sistema.
        
        Incluye información sobre:
        - Estado del agente y sus tareas
        - Recursos del sistema (CPU, memoria, disco)
        - Estado de componentes individuales
        """
    )
    async def detailed_health_check():
        """
        Health check detallado del sistema.
        
        Incluye información sobre:
        - Estado del agente
        - Recursos del sistema
        - Componentes individuales
        """
        try:
            from ..core.infrastructure.monitoring.health import HealthChecker
            
            agent = app.state.agent
            health_checker = HealthChecker(agent=agent)
            
            health_status = await health_checker.check_all()
            
            status_code = 200
            if health_status["status"] == "unhealthy":
                status_code = 503
            elif health_status["status"] == "degraded":
                status_code = 200
            
            return JSONResponse(
                content=health_status,
                status_code=status_code
            )
        except Exception as e:
            return JSONResponse(
                content={
                    "status": "unhealthy",
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                },
                status_code=503
            )
    
    @app.post("/api/config")
    async def set_config(key: str, value: Any):
        """Establecer configuración"""
        try:
            if agent.config_manager:
                agent.config_manager.set(key, value)
                return {"success": True, "message": f"Config {key} updated"}
            return {"success": False, "message": "Config manager not available"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    # ============================================================================
    # AI ENDPOINTS - Endpoints de IA
    # ============================================================================
    
    @app.post("/api/ai/process")
    async def process_command_ai(request: dict):
        """Procesar comando con IA"""
        try:
            agent = app.state.agent
            if not agent.ai_processor:
                return {"message": "AI processor not available"}
            
            command = request.get("command", "")
            if not command:
                raise HTTPException(status_code=400, detail="Command is required")
            
            processed = await agent.ai_processor.process_command(command)
            return {
                "original": processed.original,
                "intent": processed.intent.value,
                "confidence": processed.confidence,
                "extracted_code": processed.extracted_code,
                "parameters": processed.parameters or {},
                "suggested_actions": processed.suggested_actions or []
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/api/ai/generate")
    async def generate_code_ai(request: dict):
        """Generar código con IA"""
        try:
            agent = app.state.agent
            if not agent.ai_processor:
                return {"message": "AI processor not available"}
            
            description = request.get("description", "")
            language = request.get("language", "python")
            
            if not description:
                raise HTTPException(status_code=400, detail="Description is required")
            
            code = await agent.ai_processor.generate_code(description, language)
            return {"code": code, "language": language}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/api/ai/summarize")
    async def summarize_result_ai(request: dict):
        """Resumir resultado con IA"""
        try:
            agent = app.state.agent
            if not agent.ai_processor:
                return {"message": "AI processor not available"}
            
            result = request.get("result", "")
            max_length = request.get("max_length", 200)
            
            if not result:
                raise HTTPException(status_code=400, detail="Result is required")
            
            summary = await agent.ai_processor.summarize_result(result, max_length)
            return {"summary": summary, "original_length": len(result)}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/api/embeddings/search")
    async def search_embeddings(request: dict):
        """Buscar en embeddings"""
        try:
            agent = app.state.agent
            if not agent.embedding_store:
                return {"message": "Embedding store not available"}
            
            query = request.get("query", "")
            top_k = request.get("top_k", 5)
            threshold = request.get("threshold", 0.5)
            
            if not query:
                raise HTTPException(status_code=400, detail="Query is required")
            
            results = await agent.embedding_store.search(query, top_k=top_k, threshold=threshold)
            return {
                "query": query,
                "results": [
                    {
                        "key": key,
                        "similarity": similarity,
                        "metadata": metadata
                    }
                    for key, similarity, metadata in results
                ]
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/api/patterns/stats")
    async def get_pattern_stats():
        """Obtener estadísticas de patrones"""
        try:
            agent = app.state.agent
            if not agent.pattern_learner:
                return {"message": "Pattern learner not available"}
            
            stats = await agent.pattern_learner.get_statistics()
            return stats
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/api/patterns/predict")
    async def predict_success(request: dict):
        """Predecir éxito de comando"""
        try:
            agent = app.state.agent
            if not agent.pattern_learner:
                return {"message": "Pattern learner not available"}
            
            command = request.get("command", "")
            if not command:
                raise HTTPException(status_code=400, detail="Command is required")
            
            success_prob, pattern_info = await agent.pattern_learner.predict_success(command)
            suggestions = await agent.pattern_learner.suggest_improvements(command)
            
            return {
                "command": command,
                "success_probability": success_prob,
                "pattern_info": pattern_info,
                "suggestions": suggestions
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    return app


class AgentAPI:
    """Clase wrapper para la API"""
    
    def __init__(self, agent: Optional[CursorAgent] = None, host: str = "0.0.0.0", port: int = 8024):
        self.agent = agent or CursorAgent(AgentConfig())
        self.host = host
        self.port = port
        self.app = create_app(self.agent)
        self.server: Optional[uvicorn.Server] = None
        
    async def run(self):
        """Ejecutar servidor API"""
        config = uvicorn.Config(
            self.app,
            host=self.host,
            port=self.port,
            log_level="info"
        )
        self.server = uvicorn.Server(config)
        await self.server.serve()
    
    async def shutdown(self) -> None:
        """Detener servidor API de forma graceful"""
        if self.server:
            logger.info("🛑 Shutting down API server...")
            self.server.should_exit = True
            if hasattr(self.server, 'force_exit'):
                self.server.force_exit = True

