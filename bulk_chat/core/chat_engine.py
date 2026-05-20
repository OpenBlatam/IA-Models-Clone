"""
Continuous Chat Engine
=======================

Motor de chat que genera respuestas de forma continua y proactiva
hasta que el usuario pause la sesión.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Callable, Any, AsyncGenerator
from datetime import datetime

from .chat_session import ChatSession, ChatState, ChatMessage
from .session_storage import SessionStorage, JSONSessionStorage
from .metrics import MetricsCollector
from .response_cache import ResponseCache
from .plugins import PluginManager, PluginType, PluginContext
from .webhooks import WebhookManager, WebhookEvent

logger = logging.getLogger(__name__)


class ContinuousChatEngine:
    """
    Motor de chat continuo que genera respuestas automáticamente.
    
    Características:
    - Genera respuestas continuamente hasta que el usuario pause
    - Proactivo: genera respuestas automáticamente
    - Control de pausa/continuación
    - Streaming de respuestas
    - Soporte para múltiples modelos de IA
    """
    
    def __init__(
        self,
        llm_provider: Optional[Callable] = None,
        auto_continue: bool = True,
        response_interval: float = 2.0,
        max_consecutive_responses: int = 100,
        storage: Optional[SessionStorage] = None,
        enable_metrics: bool = True,
        auto_save: bool = True,
        save_interval: float = 30.0,
        enable_cache: bool = True,
        cache_size: int = 1000,
        cache_ttl: int = 3600,
        plugin_manager: Optional[PluginManager] = None,
        webhook_manager: Optional[WebhookManager] = None,
    ):
        """
        Inicializar el motor de chat.
        
        Args:
            llm_provider: Función que genera respuestas (async function)
            auto_continue: Si True, continúa generando respuestas automáticamente
            response_interval: Intervalo entre respuestas automáticas (segundos)
            max_consecutive_responses: Máximo de respuestas consecutivas antes de pausar
            storage: Sistema de almacenamiento persistente (opcional)
            enable_metrics: Habilitar recolección de métricas
            auto_save: Guardar sesiones automáticamente
            save_interval: Intervalo entre guardados automáticos (segundos)
        """
        self.llm_provider = llm_provider or self._default_llm_provider
        self.auto_continue = auto_continue
        self.response_interval = response_interval
        self.max_consecutive_responses = max_consecutive_responses
        self.storage = storage
        self.enable_metrics = enable_metrics
        self.auto_save = auto_save
        self.save_interval = save_interval
        
        # Sesiones activas
        self.sessions: Dict[str, ChatSession] = {}
        
        # Tareas de generación continua
        self._generation_tasks: Dict[str, asyncio.Task] = {}
        
        # Métricas
        self.metrics = MetricsCollector() if enable_metrics else None
        
        # Cache de respuestas
        self.cache = ResponseCache(
            max_size=cache_size,
            ttl_seconds=cache_ttl,
        ) if enable_cache else None
        
        # Plugin manager
        self.plugin_manager = plugin_manager
        
        # Webhook manager
        self.webhook_manager = webhook_manager
        
        # Tarea de guardado automático
        self._auto_save_task: Optional[asyncio.Task] = None
        if auto_save and storage:
            self._auto_save_task = asyncio.create_task(self._auto_save_loop())
        
        logger.info("Continuous Chat Engine initialized")
    
    async def _default_llm_provider(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> str:
        """
        Proveedor de LLM por defecto (mock).
        
        En producción, esto debería conectarse a OpenAI, Anthropic, etc.
        """
        # Simulación de respuesta
        await asyncio.sleep(0.5)
        last_message = messages[-1]["content"] if messages else ""
        return f"Respuesta automática a: {last_message[:50]}..."
    
    async def create_session(
        self,
        user_id: Optional[str] = None,
        initial_message: Optional[str] = None,
        auto_continue: Optional[bool] = None,
        session_id: Optional[str] = None,
    ) -> ChatSession:
        """Crear una nueva sesión de chat o cargar una existente."""
        # Si se proporciona session_id, intentar cargar desde storage
        if session_id and self.storage:
            existing_session = await self.storage.load_session(session_id)
            if existing_session:
                self.sessions[session_id] = existing_session
                if self.metrics:
                    await self.metrics.record_session_created(session_id)
                logger.info(f"Loaded existing session: {session_id}")
                return existing_session
        
        # Crear nueva sesión
        session = ChatSession(
            session_id=session_id or None,
            user_id=user_id,
            auto_continue=auto_continue if auto_continue is not None else self.auto_continue,
        )
        
        self.sessions[session.session_id] = session
        
        if initial_message:
            await session.add_message("user", initial_message)
        
        # Guardar en storage
        if self.storage:
            await self.storage.save_session(session)
        
        # Registrar métricas
        if self.metrics:
            await self.metrics.record_session_created(session.session_id)
            await self.metrics.record_message_sent(session.session_id)
        
        # Disparar webhook
        if self.webhook_manager:
            await self.webhook_manager.trigger(
                WebhookEvent.SESSION_CREATED,
                {"session_id": session.session_id, "user_id": session.user_id},
            )
        
        logger.info(f"Created chat session: {session.session_id}")
        return session
    
    async def start_continuous_chat(
        self,
        session_id: str,
        initial_prompt: Optional[str] = None,
    ):
        """
        Iniciar chat continuo para una sesión.
        
        El chat generará respuestas automáticamente hasta que se pause.
        """
        if session_id not in self.sessions:
            raise ValueError(f"Session {session_id} not found")
        
        session = self.sessions[session_id]
        
        # Activar sesión
        await session.activate()
        
        # Agregar mensaje inicial si existe
        if initial_prompt:
            await session.add_message("user", initial_prompt)
        
        # Iniciar generación continua si no está ya corriendo
        if session_id not in self._generation_tasks:
            task = asyncio.create_task(
                self._continuous_generation_loop(session_id)
            )
            self._generation_tasks[session_id] = task
            logger.info(f"Started continuous chat for session: {session_id}")
        else:
            logger.warning(f"Continuous chat already running for session: {session_id}")
    
    async def _continuous_generation_loop(self, session_id: str):
        """
        Loop principal de generación continua.
        
        Genera respuestas automáticamente hasta que la sesión se pause o detenga.
        """
        session = self.sessions[session_id]
        consecutive_responses = 0
        
        try:
            while not session.is_stopped():
                # Esperar si está pausado
                await session.wait_if_paused()
                
                if session.is_stopped():
                    break
                
                # Generar respuesta
                try:
                    # Obtener historial de conversación
                    messages = session.get_conversation_history()
                    
                    # Si no hay mensajes, esperar
                    if not messages:
                        await asyncio.sleep(self.response_interval)
                        continue
                    
                    # Verificar cache primero
                    cached_response = None
                    if self.cache:
                        cached_response = await self.cache.get(messages)
                    
                    # Generar respuesta usando el proveedor de LLM con retry
                    logger.info(f"Generating response for session {session_id}")
                    start_time = time.time()
                    
                    if cached_response:
                        response = cached_response
                        logger.debug(f"Using cached response for session {session_id}")
                    else:
                        # Ejecutar plugins pre-processor
                        if self.plugin_manager:
                            context = PluginContext(
                                session_id=session_id,
                                user_id=session.user_id,
                                message=messages[-1]["content"] if messages else None,
                            )
                            context = await self.plugin_manager.execute_plugins(
                                PluginType.PRE_PROCESSOR,
                                context,
                            )
                            if context.message:
                                messages[-1]["content"] = context.message
                        
                        response = await self._generate_with_retry(messages, max_retries=3)
                        
                        # Ejecutar plugins post-processor
                        if self.plugin_manager:
                            context = PluginContext(
                                session_id=session_id,
                                user_id=session.user_id,
                                response=response,
                            )
                            context = await self.plugin_manager.execute_plugins(
                                PluginType.POST_PROCESSOR,
                                context,
                            )
                            if context.response:
                                response = context.response
                        
                        # Guardar en cache
                        if self.cache and not cached_response:
                            await self.cache.set(messages, response)
                    
                    response_time = time.time() - start_time
                    
                    # Agregar respuesta a la sesión
                    await session.add_message("assistant", response)
                    consecutive_responses += 1
                    
                    # Registrar métricas
                    if self.metrics:
                        await self.metrics.record_response(
                            session_id,
                            response_time,
                            tokens_generated=len(response.split()),  # Estimación
                            tokens_prompt=sum(len(m["content"].split()) for m in messages),
                        )
                        await self.metrics.record_message_received(session_id)
                    
                    logger.info(f"Generated response #{consecutive_responses} for session {session_id} (took {response_time:.2f}s)")
                    
                    # Si se alcanza el máximo, pausar automáticamente
                    if consecutive_responses >= self.max_consecutive_responses:
                        logger.info(f"Reached max consecutive responses, pausing session {session_id}")
                        await session.pause("Maximum consecutive responses reached")
                        consecutive_responses = 0
                        continue
                    
                    # Si auto_continue está deshabilitado, pausar después de responder
                    if not session.auto_continue:
                        await session.pause("Auto-continue disabled")
                        break
                    
                    # Esperar antes de la siguiente respuesta
                    await asyncio.sleep(self.response_interval)
                    
                except Exception as e:
                    logger.error(f"Error generating response for session {session_id}: {e}")
                    await session.add_message(
                        "system",
                        f"Error: {str(e)}",
                        metadata={"error": True}
                    )
                    
                    # Registrar error en métricas
                    if self.metrics:
                        await self.metrics.record_error(session_id, type(e).__name__)
                    
                    # Esperar antes de reintentar
                    await asyncio.sleep(self.response_interval * 2)
        
        except asyncio.CancelledError:
            logger.info(f"Generation loop cancelled for session {session_id}")
        except Exception as e:
            logger.error(f"Fatal error in generation loop for session {session_id}: {e}")
            session.state = ChatState.ERROR
        finally:
            # Limpiar tarea
            if session_id in self._generation_tasks:
                del self._generation_tasks[session_id]
            logger.info(f"Stopped continuous chat for session: {session_id}")
    
    async def pause_session(self, session_id: str, reason: Optional[str] = None):
        """Pausar una sesión de chat."""
        if session_id not in self.sessions:
            raise ValueError(f"Session {session_id} not found")
        
        session = self.sessions[session_id]
        await session.pause(reason)
        
        # Disparar webhook
        if self.webhook_manager:
            await self.webhook_manager.trigger(
                WebhookEvent.SESSION_PAUSED,
                {"session_id": session_id, "reason": reason},
            )
        
        logger.info(f"Paused session {session_id}: {reason or 'User requested'}")
    
    async def resume_session(self, session_id: str):
        """Reanudar una sesión de chat."""
        if session_id not in self.sessions:
            raise ValueError(f"Session {session_id} not found")
        
        session = self.sessions[session_id]
        
        if session.state == ChatState.PAUSED:
            await session.resume()
            
            # Reiniciar generación si no está corriendo
            if session_id not in self._generation_tasks and session.auto_continue:
                await self.start_continuous_chat(session_id)
            
            # Disparar webhook
            if self.webhook_manager:
                await self.webhook_manager.trigger(
                    WebhookEvent.SESSION_RESUMED,
                    {"session_id": session_id},
                )
            
            logger.info(f"Resumed session {session_id}")
        else:
            logger.warning(f"Cannot resume session {session_id}: not paused")
    
    async def stop_session(self, session_id: str):
        """Detener completamente una sesión de chat."""
        if session_id not in self.sessions:
            raise ValueError(f"Session {session_id} not found")
        
        session = self.sessions[session_id]
        await session.stop()
        
        # Cancelar tarea de generación
        if session_id in self._generation_tasks:
            self._generation_tasks[session_id].cancel()
            try:
                await self._generation_tasks[session_id]
            except asyncio.CancelledError:
                pass
            del self._generation_tasks[session_id]
        
        logger.info(f"Stopped session {session_id}")
    
    async def add_user_message(self, session_id: str, message: str):
        """Agregar mensaje del usuario a la sesión."""
        if session_id not in self.sessions:
            raise ValueError(f"Session {session_id} not found")
        
        session = self.sessions[session_id]
        await session.add_message("user", message)
        
        # Si la sesión está pausada y tiene auto_continue, reanudar
        if session.state == ChatState.PAUSED and session.auto_continue:
            await self.resume_session(session_id)
    
    async def stream_response(
        self,
        session_id: str,
        message: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Stream de respuesta en tiempo real.
        
        Genera tokens de forma incremental mientras se genera la respuesta.
        """
        if session_id not in self.sessions:
            raise ValueError(f"Session {session_id} not found")
        
        session = self.sessions[session_id]
        
        # Agregar mensaje del usuario si se proporciona
        if message:
            await session.add_message("user", message)
        
        # Obtener historial
        messages = session.get_conversation_history()
        
        # Generar respuesta con streaming
        # Esto requiere que el llm_provider soporte streaming
        # Por ahora, simulamos el streaming
        response = await self.llm_provider(messages)
        
        # Simular streaming token por token
        words = response.split()
        for i, word in enumerate(words):
            if session.is_stopped():
                break
            yield word + (" " if i < len(words) - 1 else "")
            await asyncio.sleep(0.05)  # Simular latencia de generación
        
        # Guardar respuesta completa
        await session.add_message("assistant", response)
    
    def get_session(self, session_id: str) -> Optional[ChatSession]:
        """Obtener sesión por ID."""
        return self.sessions.get(session_id)
    
    async def _generate_with_retry(
        self,
        messages: List[Dict[str, str]],
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> str:
        """Generar respuesta con lógica de reintento."""
        last_error = None
        
        for attempt in range(max_retries):
            try:
                response = await self.llm_provider(messages)
                return response
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                    logger.warning(f"LLM provider error (attempt {attempt + 1}/{max_retries}), retrying in {wait_time}s: {e}")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"LLM provider failed after {max_retries} attempts: {e}")
        
        # Si todos los reintentos fallan, retornar mensaje de error
        raise Exception(f"Failed to generate response after {max_retries} attempts: {last_error}")
    
    async def _auto_save_loop(self):
        """Loop de guardado automático de sesiones."""
        while True:
            try:
                await asyncio.sleep(self.save_interval)
                
                if self.storage:
                    for session_id, session in list(self.sessions.items()):
                        try:
                            await self.storage.save_session(session)
                        except Exception as e:
                            logger.error(f"Error auto-saving session {session_id}: {e}")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in auto-save loop: {e}")
                await asyncio.sleep(self.save_interval)
    
    async def save_session(self, session_id: str) -> bool:
        """Guardar sesión manualmente."""
        if session_id not in self.sessions:
            return False
        
        if not self.storage:
            return False
        
        session = self.sessions[session_id]
        return await self.storage.save_session(session)
    
    async def load_session(self, session_id: str) -> Optional[ChatSession]:
        """Cargar sesión desde storage."""
        if not self.storage:
            return None
        
        session = await self.storage.load_session(session_id)
        if session:
            self.sessions[session_id] = session
            if self.metrics:
                await self.metrics.record_session_created(session_id)
        
        return session
    
    async def cleanup_session(self, session_id: str):
        """Limpiar y eliminar una sesión."""
        await self.stop_session(session_id)
        
        # Guardar antes de eliminar
        if self.storage and session_id in self.sessions:
            await self.storage.save_session(self.sessions[session_id])
        
        # Registrar fin de sesión en métricas
        if self.metrics:
            await self.metrics.record_session_ended(session_id)
        
        if session_id in self.sessions:
            del self.sessions[session_id]
        
        logger.info(f"Cleaned up session {session_id}")

