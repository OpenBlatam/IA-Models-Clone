"""
Cursor Agent - Agente principal 24/7
=====================================

Agente persistente que escucha y ejecuta comandos desde Cursor.
"""

import asyncio
import logging
from typing import Optional, Dict, Any, List, Callable
from pathlib import Path

# Import domain models
from .domain.agent import AgentStatus, AgentConfig, Task
from .domain.exceptions import (
    AgentNotRunningException,
    TaskValidationException,
    RateLimitExceededException,
    TaskExecutionException,
    TaskTimeoutException,
    StorageException
)

# Usar orjson para JSON más rápido
try:
    import orjson as json
except ImportError:
    import json  # Fallback a json estándar

# Usar structlog para logging estructurado
try:
    import structlog
    logger = structlog.get_logger(__name__)
except ImportError:
    logger = logging.getLogger(__name__)

# Importar observabilidad
try:
    from .infrastructure.monitoring.observability import observe_async
except ImportError:
    observe_async = lambda **kwargs: lambda f: f  # No-op decorator si no está disponible


class CursorAgent:
    """Agente principal que escucha y ejecuta comandos"""
    
    def __init__(self, config: Optional[AgentConfig] = None):
        self.config = config or AgentConfig()
        self.status = AgentStatus.IDLE
        self.tasks: Dict[str, Task] = {}
        self.task_queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self.running = False
        self._listener_task: Optional[asyncio.Task] = None
        self._executor_task: Optional[asyncio.Task] = None
        self._callbacks: List[Callable] = []
        
        # Estado persistente
        self._state_file = self.config.storage_path
        
        # Sistema de notificaciones y métricas
        try:
            from .infrastructure.messaging.notifications import NotificationManager, NotificationLevel
            from .infrastructure.monitoring.metrics import MetricsCollector
            self.notifications = NotificationManager()
            self.metrics = MetricsCollector()
        except ImportError:
            self.notifications = None
            self.metrics = None
        
        # Rate limiter
        try:
            from .utils.rate_limiting.rate_limiter import TaskRateLimiter
            self.rate_limiter = TaskRateLimiter(
                max_tasks_per_minute=60,
                max_concurrent_tasks=self.config.max_concurrent_tasks
            )
        except ImportError:
            self.rate_limiter = None
        
        # Plugin manager
        try:
            from .infrastructure.plugins.plugins import PluginManager
            self.plugin_manager = PluginManager(self)
        except ImportError:
            self.plugin_manager = None
        
        # Scheduler
        try:
            from .infrastructure.scheduling.scheduler import TaskScheduler
            self.scheduler = TaskScheduler(self)
        except ImportError:
            self.scheduler = None
        
        # Backup manager
        try:
            from .infrastructure.persistence.backup import BackupManager
            self.backup_manager = BackupManager(self)
        except ImportError:
            self.backup_manager = None
        
        # Cache
        try:
            from .infrastructure.caching.cache import CommandCache
            self.command_cache = CommandCache()
        except ImportError:
            self.command_cache = None
        
        # Template manager
        try:
            from .utils.templates.templates import TemplateManager
            self.template_manager = TemplateManager()
        except ImportError:
            self.template_manager = None
        
        # Validator
        try:
            from .utils.validation.validators import CommandValidator
            self.validator = CommandValidator()
        except ImportError:
            self.validator = None
        
        # Event bus
        try:
            from .infrastructure.messaging.event_bus import EventBus, EventType
            self.event_bus = EventBus()
        except ImportError:
            self.event_bus = None
        
        # Cluster manager
        try:
            from .infrastructure.clustering.cluster import ClusterManager
            self.cluster_manager = ClusterManager(self)
        except ImportError:
            self.cluster_manager = None
        
        # Config manager
        try:
            from .utils.config.config_manager import ConfigManager
            config_path = Path(self.config.storage_path).parent / "config.json"
            self.config_manager = ConfigManager(str(config_path))
        except ImportError:
            self.config_manager = None
        
        # Alert manager
        try:
            from .utils.alerts.alerting import AlertManager
            self.alert_manager = AlertManager(self)
        except ImportError:
            self.alert_manager = None
        
        # AI Processor (lazy load - solo se inicializa cuando se usa)
        self._ai_processor = None
        self._ai_processor_initialized = False
        
        # Embeddings (lazy load)
        self._embedding_store = None
        self._embedding_store_initialized = False
        
        # Pattern Learner (lazy load)
        self._pattern_learner = None
        self._pattern_learner_initialized = False
        
        # Resource Manager (lazy load)
        self._resource_manager = None
        self._resource_manager_initialized = False
    
    @property
    def resource_manager(self):
        """Lazy load resource manager"""
        if not self._resource_manager_initialized:
            try:
                from .resource_manager import ResourceManager
                self._resource_manager = ResourceManager(cleanup_interval=300.0)
            except ImportError:
                self._resource_manager = None
            finally:
                self._resource_manager_initialized = True
        return self._resource_manager
    
    @property
    def ai_processor(self):
        """Lazy load AI processor"""
        if not self._ai_processor_initialized:
            try:
                from .ai.ai_processor import AIProcessor
                self._ai_processor = AIProcessor(use_local=True)
            except ImportError:
                self._ai_processor = None
            finally:
                self._ai_processor_initialized = True
        return self._ai_processor
    
    @property
    def embedding_store(self):
        """Lazy load embedding store"""
        if not self._embedding_store_initialized:
            try:
                from .ai.embeddings import EmbeddingStore
                self._embedding_store = EmbeddingStore()
            except ImportError:
                self._embedding_store = None
            finally:
                self._embedding_store_initialized = True
        return self._embedding_store
    
    @property
    def pattern_learner(self):
        """Lazy load pattern learner"""
        if not self._pattern_learner_initialized:
            try:
                from .ai.pattern_learner import PatternLearner
                self._pattern_learner = PatternLearner()
            except ImportError:
                self._pattern_learner = None
            finally:
                self._pattern_learner_initialized = True
        return self._pattern_learner
        
    async def start(self) -> None:
        """Iniciar el agente"""
        if self.status == AgentStatus.RUNNING:
            logger.warning("Agent is already running")
            return
            
        logger.info("🚀 Starting Cursor Agent 24/7...")
        self.running = True
        self.status = AgentStatus.RUNNING
        
        # Inicializar componentes de IA (lazy load - solo si se usan)
        # La inicialización real se hace cuando se accede a las propiedades
        
        # Cargar estado persistente si existe
        if self.config.persistent_storage:
            await self._load_state()
        
        # Iniciar listener y executor
        self._listener_task = asyncio.create_task(self._listen_loop())
        self._executor_task = asyncio.create_task(self._executor_loop())
        
        # Iniciar plugins
        if self.plugin_manager:
            await self.plugin_manager.start_all()
        
        # Iniciar scheduler
        if self.scheduler:
            await self.scheduler.start()
        
        # Iniciar backup automático
        if self.backup_manager:
            await self.backup_manager.start_auto_backup()
        
        # Iniciar cluster manager
        if self.cluster_manager:
            await self.cluster_manager.start()
        
        # Iniciar alert manager
        if self.alert_manager:
            await self.alert_manager.start()
        
        # Iniciar resource manager
        if self.resource_manager:
            await self.resource_manager.start()
        
        logger.info("✅ Agent started successfully")
        await self._notify_callbacks("started")
        
        # Publicar evento
        if self.event_bus:
            await self.event_bus.publish(
                self.event_bus.EventType.AGENT_STARTED,
                {"agent_id": id(self)},
                source="agent"
            )
        
        # Notificación y métrica
        if self.notifications:
            await self.notifications.notify(
                "Agent Started",
                "Cursor Agent 24/7 is now running",
                level=self.notifications.NotificationLevel.SUCCESS
            )
        if self.metrics:
            self.metrics.increment("agent_starts")
            self.metrics.set_gauge("agent_running", 1.0)
        
    async def stop(self) -> None:
        """Detener el agente"""
        if self.status == AgentStatus.STOPPED:
            return
            
        logger.info("🛑 Stopping Cursor Agent...")
        self.running = False
        self.status = AgentStatus.STOPPED
        
        # Cancelar tareas
        if self._listener_task:
            self._listener_task.cancel()
        if self._executor_task:
            self._executor_task.cancel()
            
        # Esperar a que terminen
        await asyncio.gather(
            self._listener_task,
            self._executor_task,
            return_exceptions=True
        )
        
        # Detener componentes
        if self.resource_manager:
            await self.resource_manager.stop()
        
        if self.alert_manager:
            await self.alert_manager.stop()
        
        if self.cluster_manager:
            await self.cluster_manager.stop()
        
        if self.scheduler:
            await self.scheduler.stop()
        
        if self.backup_manager:
            await self.backup_manager.stop_auto_backup()
        
        if self.plugin_manager:
            await self.plugin_manager.stop_all()
        
        # Publicar evento
        if self.event_bus:
            await self.event_bus.publish(
                self.event_bus.EventType.AGENT_STOPPED,
                {"agent_id": id(self)},
                source="agent"
            )
        
        # Guardar estado
        if self.config.persistent_storage:
            await self._save_state()
            
        logger.info("✅ Agent stopped")
        await self._notify_callbacks("stopped")
        
        # Notificación y métrica
        if self.notifications:
            await self.notifications.notify(
                "Agent Stopped",
                "Cursor Agent 24/7 has been stopped",
                level=self.notifications.NotificationLevel.INFO
            )
        if self.metrics:
            self.metrics.increment("agent_stops")
            self.metrics.set_gauge("agent_running", 0.0)
        
    async def pause(self) -> None:
        """Pausar el agente"""
        if self.status == AgentStatus.RUNNING:
            self.status = AgentStatus.PAUSED
            logger.info("⏸️ Agent paused")
            await self._notify_callbacks("paused")
            
    async def resume(self) -> None:
        """Reanudar el agente"""
        if self.status == AgentStatus.PAUSED:
            self.status = AgentStatus.RUNNING
            logger.info("▶️ Agent resumed")
            await self._notify_callbacks("resumed")
            
    @observe_async(operation_name="add_task", log_args=False, track_metrics=True)
    async def add_task(
        self,
        command: str,
        priority: int = 0,
        timeout: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Agregar una tarea al queue con prioridad.
        
        Args:
            command: Comando a ejecutar
            priority: Prioridad de la tarea (0-10, mayor es más prioritario). Default: 0
            timeout: Timeout personalizado en segundos (opcional)
            metadata: Metadatos adicionales para la tarea (opcional)
            
        Returns:
            task_id: Identificador único de la tarea creada
            
        Raises:
            AgentNotRunningException: Si el agente no está corriendo
            TaskValidationException: Si el comando no pasa la validación
            RateLimitExceededException: Si se excede el límite de rate
        """
        # Verificar que el agente esté corriendo
        if self.status != AgentStatus.RUNNING:
            raise AgentNotRunningException("add_task")
        
        # Validar prioridad
        priority = max(0, min(10, priority))
        
        # Validar comando
        if self.validator:
            validation = self.validator.validate(command)
            if not validation.is_valid:
                raise TaskValidationException(
                    message=f"Command validation failed: {', '.join(validation.errors)}",
                    validation_errors=validation.errors
                )
            if validation.warnings:
                logger.warning(f"Command warnings: {', '.join(validation.warnings)}")
            # Sanitizar comando
            command = self.validator.sanitize(command)
        
        # Verificar rate limit
        if self.rate_limiter:
            if not await self.rate_limiter.can_add_task():
                raise RateLimitExceededException(
                    message="Rate limit exceeded. Too many tasks.",
                    limit=self.config.max_concurrent_tasks
                )
            await self.rate_limiter.add_task()
        
        task_id = f"task_{datetime.now().timestamp()}_{len(self.tasks)}"
        task = Task(
            id=task_id,
            command=command,
            status="pending",
            priority=priority,
            metadata=metadata or {}
        )
        
        self.tasks[task_id] = task
        # PriorityQueue ordena por el primer elemento de la tupla (menor primero)
        # Usamos -priority para que mayor prioridad = menor número = se procesa primero
        await self.task_queue.put((-priority, task.timestamp.timestamp(), task))
        
        logger.info(f"📝 Task added: {task_id} - {command[:50]}...")
        
        # Métricas
        if self.metrics:
            self.metrics.increment("tasks_added")
            self.metrics.set_gauge("tasks_pending", sum(1 for t in self.tasks.values() if t.status == "pending"))
        
        # Notificar plugins
        if self.plugin_manager:
            await self.plugin_manager.notify_task_added(task_id, command)
        
        # Publicar evento
        if self.event_bus:
            await self.event_bus.publish(
                self.event_bus.EventType.TASK_ADDED,
                {"task_id": task_id, "command": command[:100]},
                source="agent"
            )
        
        return task_id
        
    async def get_status(self) -> Dict[str, Any]:
        """
        Obtener estado del agente.
        
        Returns:
            Diccionario con el estado actual del agente y estadísticas
        """
        # Optimización: usar comprensión de listas más eficiente
        tasks_list = list(self.tasks.values())
        
        status = {
            "status": self.status.value,
            "running": self.running,
            "tasks_total": len(tasks_list),
            "tasks_pending": sum(1 for t in tasks_list if t.status == "pending"),
            "tasks_running": sum(1 for t in tasks_list if t.status == "running"),
            "tasks_completed": sum(1 for t in tasks_list if t.status == "completed"),
            "tasks_failed": sum(1 for t in tasks_list if t.status == "failed"),
        }
        
        # Agregar métricas si están disponibles
        if self.metrics:
            status["metrics"] = self.metrics.get_summary()
        
        # Agregar notificaciones si están disponibles
        if self.notifications:
            status["notifications"] = self.notifications.get_stats()
        
        return status
        
    async def get_tasks(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Obtener lista de tareas ordenadas por timestamp.
        
        Args:
            limit: Número máximo de tareas a retornar
            
        Returns:
            Lista de diccionarios con información de tareas
        """
        # Optimización: usar sorted con reverse=True es más eficiente que sort
        tasks_list = sorted(
            self.tasks.values(),
            key=lambda x: x.timestamp,
            reverse=True
        )[:limit]
        
        # Optimización: construir diccionarios de forma más eficiente
        return [
            {
                "id": task.id,
                "command": task.command,
                "status": task.status,
                "priority": task.priority,
                "timestamp": task.timestamp.isoformat(),
                "result": task.result,
                "error": task.error,
                "metadata": task.metadata or {}
            }
            for task in tasks_list
        ]
        
    def on_event(self, callback: Callable) -> None:
        """Registrar callback para eventos"""
        self._callbacks.append(callback)
        
    async def _notify_callbacks(self, event: str) -> None:
        """Notificar callbacks de eventos"""
        for callback in self._callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event)
                else:
                    callback(event)
            except Exception as e:
                logger.error(f"Error in callback: {e}")
                
    async def _listen_loop(self) -> None:
        """Loop principal que escucha comandos"""
        logger.info("👂 Starting command listener...")
        
        # Inicializar file watcher
        from .services.file_watcher import FileWatcher
        
        # Determinar archivo de comandos
        if self.config.command_file:
            command_file = Path(self.config.command_file)
        else:
            command_file = Path(self.config.storage_path).parent / "commands.txt"
        
        self.file_watcher = FileWatcher(
            command_file=str(command_file) if command_file else None,
            watch_dir=self.config.watch_directory
        )
        self.file_watcher.set_callback(self._on_command_received)
        await self.file_watcher.start()
        
        if command_file:
            logger.info(f"📝 Listening for commands in: {command_file}")
            logger.info(f"💡 Write commands to: {command_file}")
        elif self.config.watch_directory:
            logger.info(f"📁 Monitoring directory: {self.config.watch_directory}")
        
        while self.running:
            try:
                await asyncio.sleep(self.config.check_interval)
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in listener loop: {e}")
                if not self.config.auto_restart:
                    break
                await asyncio.sleep(5)
        
        # Detener file watcher
        if hasattr(self, 'file_watcher'):
            await self.file_watcher.stop()
    
    async def _on_command_received(self, command: str) -> None:
        """Callback cuando se recibe un comando"""
        if command and command.strip():
            logger.info(f"📨 Command received: {command[:50]}...")
            await self.add_task(command.strip())
                
    async def _executor_loop(self) -> None:
        """Loop que ejecuta tareas"""
        logger.info("⚙️ Starting task executor...")
        
        active_tasks: Dict[str, asyncio.Task] = {}
        
        while self.running:
            try:
                # Procesar tareas pendientes
                if len(active_tasks) < self.config.max_concurrent_tasks:
                    try:
                        # PriorityQueue retorna tupla (priority, timestamp, task)
                        _, _, task = await asyncio.wait_for(
                            self.task_queue.get(),
                            timeout=0.5
                        )
                        
                        if task.status == "pending":
                            task.status = "running"
                            task_executor = asyncio.create_task(
                                self._execute_task(task)
                            )
                            active_tasks[task.id] = task_executor
                            
                    except asyncio.TimeoutError:
                        pass
                
                # Limpiar tareas completadas
                completed = []
                for task_id, exec_task in active_tasks.items():
                    if exec_task.done():
                        completed.append(task_id)
                        
                for task_id in completed:
                    del active_tasks[task_id]
                    
                await asyncio.sleep(0.1)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in executor loop: {e}")
                await asyncio.sleep(1)
                
        # Esperar a que terminen todas las tareas activas
        if active_tasks:
            await asyncio.gather(*active_tasks.values(), return_exceptions=True)
            
    @observe_async(operation_name="execute_task", log_args=False, track_metrics=True)
    async def _execute_task(self, task: Task) -> None:
        """
        Ejecutar una tarea con optimizaciones de rendimiento.
        
        Incluye:
        - Procesamiento con IA (opcional)
        - Predicción de éxito (opcional)
        - Almacenamiento de embeddings (opcional)
        - Caché de resultados
        - Manejo robusto de errores
        """
        start_time = datetime.now()
        success = False
        
        try:
            logger.info(f"🔨 Executing task: {task.id}")
            
            # Procesar comando con IA si está disponible (en paralelo con otras operaciones)
            ai_processing_task = None
            if self.ai_processor:
                try:
                    ai_processing_task = asyncio.create_task(
                        self.ai_processor.process_command(task.command)
                    )
                except Exception as e:
                    logger.debug(f"AI processing task creation failed: {e}")
            
            # Predecir éxito con pattern learner (en paralelo)
            pattern_prediction_task = None
            if self.pattern_learner:
                try:
                    pattern_prediction_task = asyncio.create_task(
                        self.pattern_learner.predict_success(task.command)
                    )
                except Exception as e:
                    logger.debug(f"Pattern prediction task creation failed: {e}")
            
            # Verificar caché primero (más rápido)
            cached_result = None
            if self.command_cache:
                cached_result = await self.command_cache.get_result(task.command)
                if cached_result:
                    logger.debug(f"💾 Using cached result for task {task.id}")
                    result = cached_result
                else:
                    # Ejecutar comando real
                    result = await self._execute_command_with_retry(task)
                    # Guardar en caché
                    if result:
                        await self.command_cache.set_result(task.command, result)
            else:
                # Ejecutar comando real sin caché
                result = await self._execute_command_with_retry(task)
            
            # Procesar resultados de IA y pattern learner si están disponibles
            if ai_processing_task:
                try:
                    processed_command = await ai_processing_task
                    logger.debug(f"🤖 AI processed command: intent={processed_command.intent}, confidence={processed_command.confidence:.2f}")
                    if processed_command.extracted_code and not cached_result:
                        task.command = processed_command.extracted_code
                except Exception as e:
                    logger.debug(f"AI processing failed: {e}")
            
            if pattern_prediction_task:
                try:
                    success_prob, pattern_info = await pattern_prediction_task
                    logger.debug(f"📊 Success probability: {success_prob:.2f}")
                except Exception as e:
                    logger.debug(f"Pattern prediction failed: {e}")
            
            # Agregar a embeddings si está disponible (no bloquea la ejecución)
            if self.embedding_store and not cached_result:
                try:
                    asyncio.create_task(
                        self.embedding_store.add(
                            f"task_{task.id}",
                            task.command,
                            metadata={"task_id": task.id, "timestamp": task.timestamp.isoformat()}
                        )
                    )
                except Exception as e:
                    logger.debug(f"Embedding storage task creation failed: {e}")
            
            task.status = "completed"
            task.result = result
            success = True
            
            # Resumir resultado con IA si es muy largo
            if self.ai_processor and len(result) > 1000:
                try:
                    summarized = await self.ai_processor.summarize_result(result, max_length=500)
                    task.result = summarized
                except Exception as e:
                    logger.debug(f"Result summarization failed: {e}")
            
            # Registrar en pattern learner
            execution_time = (datetime.now() - start_time).total_seconds()
            if self.pattern_learner:
                try:
                    await self.pattern_learner.record_command(
                        task.command,
                        success=True,
                        execution_time=execution_time,
                        result=result
                    )
                except Exception as e:
                    logger.debug(f"Pattern recording failed: {e}")
            
            logger.info(f"✅ Task completed: {task.id}")
            await self._notify_callbacks(f"task_completed:{task.id}")
            
            # Notificación y métricas
            if self.notifications:
                await self.notifications.notify(
                    "Task Completed",
                    f"Task {task.id[:8]}... completed successfully",
                    level=self.notifications.NotificationLevel.SUCCESS,
                    metadata={"task_id": task.id, "command": task.command[:50]}
                )
            if self.metrics:
                self.metrics.increment("tasks_completed")
                self.metrics.set_gauge("tasks_completed_total", sum(1 for t in self.tasks.values() if t.status == "completed"))
            
            # Rate limiter
            if self.rate_limiter:
                await self.rate_limiter.complete_task()
            
            # Notificar plugins
            if self.plugin_manager:
                await self.plugin_manager.notify_task_completed(task.id, result)
            
            # Publicar evento
            if self.event_bus:
                await self.event_bus.publish(
                    self.event_bus.EventType.TASK_COMPLETED,
                    {"task_id": task.id, "result": result[:200]},
                    source="agent"
                )
            
        except Exception as e:
            task.status = "failed"
            task.error = str(e)
            logger.error(f"❌ Task failed: {task.id} - {e}")
            
            # Registrar fallo en pattern learner
            execution_time = (datetime.now() - start_time).total_seconds()
            if self.pattern_learner:
                try:
                    await self.pattern_learner.record_command(
                        task.command,
                        success=False,
                        execution_time=execution_time,
                        result=None
                    )
                except Exception as e2:
                    logger.debug(f"Pattern recording failed: {e2}")
            
            await self._notify_callbacks(f"task_failed:{task.id}")
            
            # Notificación y métricas
            if self.notifications:
                await self.notifications.notify(
                    "Task Failed",
                    f"Task {task.id[:8]}... failed: {str(e)[:100]}",
                    level=self.notifications.NotificationLevel.ERROR,
                    metadata={"task_id": task.id, "error": str(e), "command": task.command[:50]}
                )
            if self.metrics:
                self.metrics.increment("tasks_failed")
                self.metrics.set_gauge("tasks_failed_total", sum(1 for t in self.tasks.values() if t.status == "failed"))
            
            # Rate limiter
            if self.rate_limiter:
                await self.rate_limiter.complete_task()
            
            # Notificar plugins
            if self.plugin_manager:
                await self.plugin_manager.notify_task_failed(task.id, str(e))
            
            # Publicar evento
            if self.event_bus:
                await self.event_bus.publish(
                    self.event_bus.EventType.TASK_FAILED,
                    {"task_id": task.id, "error": str(e)[:200]},
                    source="agent"
                )
            
    async def _save_state(self) -> None:
        """
        Guardar estado persistente del agente.
        
        Raises:
            StorageException: Si hay un error al guardar el estado
        """
        if not self.config.persistent_storage:
            return
        
        try:
            from pathlib import Path
            from .constants import DEFAULT_STORAGE_PATH
            from .domain.exceptions import StorageException
            
            state_file = Path(self._state_file or DEFAULT_STORAGE_PATH)
            state_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Preparar estado para guardar
            state = {
                "version": "1.0",
                "saved_at": datetime.now().isoformat(),
                "status": self.status.value,
                "tasks": {}
            }
            
            # Guardar solo tareas completadas o fallidas (no pendientes/running)
            for task_id, task in self.tasks.items():
                if task.status in ("completed", "failed"):
                    state["tasks"][task_id] = {
                        "id": task.id,
                        "command": task.command,
                        "timestamp": task.timestamp.isoformat(),
                        "status": task.status,
                        "priority": task.priority,
                        "result": task.result,
                        "error": task.error,
                        "metadata": task.metadata or {}
                    }
            
            # Guardar con escritura atómica (escribir a archivo temporal primero)
            temp_file = state_file.with_suffix('.tmp')
            
            try:
                if hasattr(json, 'dumps'):
                    # orjson
                    with open(temp_file, 'wb') as f:
                        f.write(json.dumps(state))
                else:
                    # json estándar
                    with open(temp_file, 'w', encoding='utf-8') as f:
                        json.dump(state, f, indent=2, ensure_ascii=False)
                
                # Mover archivo temporal al destino (operación atómica)
                temp_file.replace(state_file)
                
                logger.debug(f"💾 State saved to {state_file}: {len(state['tasks'])} tasks")
                
            except Exception as e:
                # Limpiar archivo temporal en caso de error
                try:
                    if temp_file.exists():
                        temp_file.unlink()
                except Exception:
                    pass
                raise
            
        except StorageException:
            raise
        except Exception as e:
            logger.error(f"Error saving state: {e}")
            raise StorageException(
                message=f"Failed to save state: {str(e)}",
                storage_path=str(self._state_file),
                operation="save"
            ) from e
            
    async def _load_state(self) -> None:
        """
        Cargar estado persistente del agente.
        
        Raises:
            StorageException: Si hay un error al cargar el estado
        """
        try:
            from pathlib import Path
            
            if not Path(self._state_file).exists():
                return
                
            # Usar orjson si está disponible (más rápido)
            if hasattr(json, 'loads'):
                # orjson
                with open(self._state_file, 'rb') as f:
                    state = json.loads(f.read())
            else:
                # json estándar
                with open(self._state_file, 'r') as f:
                    state = json.load(f)
                
            # Restaurar tareas con validación
            tasks_loaded = 0
            for task_id, task_data in state.get("tasks", {}).items():
                try:
                    if not isinstance(task_data, dict):
                        continue
                    
                    # Validar campos requeridos
                    if "id" not in task_data or "command" not in task_data:
                        continue
                    
                    task = Task(
                        id=str(task_data["id"]),
                        command=str(task_data["command"]),
                        timestamp=datetime.fromisoformat(task_data.get("timestamp", datetime.now().isoformat())),
                        status=str(task_data.get("status", "pending")),
                        priority=int(task_data.get("priority", 0)),
                        result=task_data.get("result"),
                        error=task_data.get("error"),
                        metadata=task_data.get("metadata", {})
                    )
                    self.tasks[task_id] = task
                    tasks_loaded += 1
                except (ValueError, KeyError, TypeError) as e:
                    logger.warning(f"Error loading task {task_id}: {e}")
                    continue
                
            logger.info(f"📂 State loaded from {self._state_file}: {tasks_loaded} tasks restored")
            
        except Exception as e:
            logger.error(f"Error loading state: {e}")
            raise StorageException(
                message=f"Failed to load state: {str(e)}",
                storage_path=self._state_file,
                operation="load"
            ) from e
    
    async def _execute_command_with_retry(self, task: Task) -> str:
        """
        Ejecutar comando con reintentos automáticos.
        
        Args:
            task: Tarea a ejecutar
            
        Returns:
            Resultado de la ejecución del comando
            
        Raises:
            TaskExecutionException: Si la ejecución falla después de todos los reintentos
            TaskTimeoutException: Si la ejecución excede el timeout
        """
        from .task_executor import TaskExecutor
        from .domain.exceptions import TaskExecutionException, TaskTimeoutException
        
        executor = TaskExecutor(timeout=self.config.task_timeout)
        max_retries = 2
        last_exception = None
        
        for attempt in range(max_retries + 1):
            try:
                exec_result = await executor.execute(task.command, task.id)
                return exec_result.output
            except TaskTimeoutException as e:
                if attempt < max_retries:
                    logger.warning(f"Task timeout (attempt {attempt + 1}/{max_retries + 1}), retrying...")
                    await asyncio.sleep(1.0 * (attempt + 1))
                    last_exception = e
                else:
                    raise Exception(f"Task timeout after {max_retries + 1} attempts: {e.message}") from e
            except TaskExecutionException as e:
                if attempt < max_retries and "temporary" in str(e).lower():
                    logger.warning(f"Temporary error (attempt {attempt + 1}/{max_retries + 1}), retrying...")
                    await asyncio.sleep(1.0 * (attempt + 1))
                    last_exception = e
                else:
                    raise Exception(f"Task execution failed: {e.message}") from e
        
        if last_exception:
            raise last_exception
        raise Exception("Task execution failed after all retries")

