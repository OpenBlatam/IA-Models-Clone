"""
Task Processor - Procesador de tareas del agente
================================================

Gestiona la ejecución completa de tareas incluyendo:
- Procesamiento con IA
- Predicción de éxito
- Ejecución de comandos
- Caché
- Notificaciones y métricas
"""

import logging
from typing import Optional, TYPE_CHECKING
from datetime import datetime

from ..error_handling import safe_async_call

if TYPE_CHECKING:
    from ..agent import CursorAgent, Task

logger = logging.getLogger(__name__)


class TaskProcessor:
    """
    Procesador de tareas del agente.
    
    Gestiona la ejecución completa de tareas incluyendo
    procesamiento con IA, caché, notificaciones y métricas.
    """
    
    def __init__(self, agent: "CursorAgent") -> None:
        """
        Inicializar procesador de tareas.
        
        Args:
            agent: Instancia del agente.
        """
        self.agent = agent
    
    async def process(self, task: "Task") -> None:
        """
        Procesar una tarea completa.
        
        Ejecuta todos los pasos necesarios para procesar una tarea:
        1. Procesamiento con IA (si está disponible)
        2. Predicción de éxito (si está disponible)
        3. Almacenamiento en embeddings (si está disponible)
        4. Ejecución del comando (con caché si está disponible)
        5. Resumen de resultados (si es necesario)
        6. Registro en pattern learner
        7. Notificaciones y métricas
        
        Args:
            task: Tarea a procesar.
        """
        start_time = datetime.now()
        
        try:
            logger.info(f"🔨 Executing task: {task.id}")
            
            # Paso 1: Procesar comando con IA
            await self._process_with_ai(task)
            
            # Paso 2: Predecir éxito
            await self._predict_success(task)
            
            # Paso 3: Agregar a embeddings
            await self._store_embeddings(task)
            
            # Paso 4: Ejecutar comando
            result = await self._execute_command(task)
            
            # Paso 5: Resumir resultado si es necesario
            if result and len(result) > 1000:
                result = await self._summarize_result(result)
            
            # Actualizar tarea
            task.status = "completed"
            task.result = result
            
            # Paso 6: Registrar éxito
            execution_time = (datetime.now() - start_time).total_seconds()
            await self._record_success(task, execution_time, result)
            
            # Paso 7: Verificación crítica antes de reportar
            await self._verify_before_reporting(task)
            
            # Paso 8: Notificaciones y métricas
            await self._notify_completion(task, result)
            
            logger.info(f"✅ Task completed: {task.id}")
            await self.agent._notify_callbacks(f"task_completed:{task.id}")
        
        except Exception as e:
            task.status = "failed"
            task.error = str(e)
            logger.error(f"❌ Task failed: {task.id} - {e}", exc_info=True)
            
            # Registrar fallo
            execution_time = (datetime.now() - start_time).total_seconds()
            await self._record_failure(task, execution_time)
            
            # Notificaciones y métricas
            await self._notify_failure(task, str(e))
            
            await self.agent._notify_callbacks(f"task_failed:{task.id}")
    
    async def _process_with_ai(self, task: "Task") -> None:
        """Procesar comando con IA si está disponible."""
        if not self.agent.ai_processor:
            return
        
        async def process():
            processed_command = await self.agent.ai_processor.process_command(task.command)
            logger.debug(
                f"🤖 AI processed command: intent={processed_command.intent}, "
                f"confidence={processed_command.confidence:.2f}"
            )
            
            if processed_command.extracted_code:
                task.command = processed_command.extracted_code
        
        await safe_async_call(
            process,
            operation="AI processing",
            logger_instance=logger,
            reraise=False
        )
    
    async def _predict_success(self, task: "Task") -> None:
        """Predecir éxito con pattern learner si está disponible."""
        if not self.agent.pattern_learner:
            return
        
        async def predict():
            success_prob, pattern_info = await self.agent.pattern_learner.predict_success(
                task.command
            )
            logger.debug(f"📊 Success probability: {success_prob:.2f}")
        
        await safe_async_call(
            predict,
            operation="pattern prediction",
            logger_instance=logger,
            reraise=False
        )
    
    async def _store_embeddings(self, task: "Task") -> None:
        """Almacenar comando en embeddings si está disponible."""
        if not self.agent.embedding_store:
            return
        
        async def store():
            await self.agent.embedding_store.add(
                f"task_{task.id}",
                task.command,
                metadata={
                    "task_id": task.id,
                    "timestamp": task.timestamp.isoformat()
                }
            )
        
        await safe_async_call(
            store,
            operation="embedding storage",
            logger_instance=logger,
            reraise=False
        )
    
    async def _execute_command(self, task: "Task") -> Optional[str]:
        """
        Ejecutar comando con soporte de caché y comandos Devin.
        
        Detecta si el comando es un comando Devin (XML) y lo ejecuta usando
        DevinCommandExecutor. Si no, usa el ejecutor estándar.
        
        Returns:
            Resultado de la ejecución o None si falló.
        
        Raises:
            RuntimeError: Si la ejecución falla.
        """
        import re
        
        # Detectar si es un comando Devin (formato XML)
        is_devin_command = bool(re.search(r'<\w+[^>]*>', task.command))
        
        if is_devin_command and hasattr(self.agent, 'devin_commands'):
            # Usar DevinCommandExecutor para comandos XML
            try:
                logger.debug(f"🔧 Detected Devin command format, parsing...")
                results = await self.agent.devin_commands.parse_and_execute_commands(
                    task.command
                )
                
                # Combinar resultados
                outputs = []
                errors = []
                for result in results:
                    if result.success:
                        if result.output:
                            outputs.append(result.output)
                    else:
                        errors.append(result.error or "Unknown error")
                
                if errors:
                    error_msg = "; ".join(errors)
                    logger.error(f"❌ Devin command execution failed: {error_msg}")
                    raise RuntimeError(error_msg)
                
                result = "\n".join(outputs) if outputs else "Command executed successfully"
                
                # Sanitizar resultado para remover secretos
                if self.agent.security_manager:
                    result = self.agent.security_manager.sanitize_output(result)
                
                # Guardar en caché si está disponible
                if self.agent.command_cache and result:
                    await self.agent.command_cache.set_result(task.command, result)
                
                return result
                
            except Exception as e:
                logger.error(f"Error executing Devin command: {e}", exc_info=True)
                raise RuntimeError(f"Devin command execution failed: {str(e)}") from e
        
        # Ejecución estándar para comandos no-XML
        from ..task_executor import TaskExecutor
        
        # Verificar caché si está disponible
        if self.agent.command_cache:
            cached_result = await self.agent.command_cache.get_result(task.command)
            if cached_result:
                logger.debug(f"💾 Using cached result for task {task.id}")
                return cached_result
        
        # Ejecutar comando real
        executor = TaskExecutor(timeout=self.agent.config.task_timeout)
        exec_result = await executor.execute(task.command, task.id)
        
        if not exec_result.success:
            raise RuntimeError(exec_result.error or "Execution failed")
        
        result = exec_result.output or ""
        
        # Sanitizar resultado para remover secretos
        if self.agent.security_manager:
            result = self.agent.security_manager.sanitize_output(result)
        
        # Guardar en caché si está disponible
        if self.agent.command_cache and result:
            await self.agent.command_cache.set_result(task.command, result)
        
        return result
    
    async def _summarize_result(self, result: str) -> str:
        """Resumir resultado con IA si está disponible."""
        if not self.agent.ai_processor:
            return result
        
        async def summarize():
            return await self.agent.ai_processor.summarize_result(
                result,
                max_length=500
            )
        
        summarized = await safe_async_call(
            summarize,
            operation="result summarization",
            default_return=result,
            logger_instance=logger,
            reraise=False
        )
        
        return summarized if summarized is not None else result
    
    async def _record_success(
        self,
        task: "Task",
        execution_time: float,
        result: Optional[str]
    ) -> None:
        """Registrar éxito en pattern learner."""
        if not self.agent.pattern_learner:
            return
        
        async def record():
            await self.agent.pattern_learner.record_command(
                task.command,
                success=True,
                execution_time=execution_time,
                result=result
            )
        
        await safe_async_call(
            record,
            operation="pattern recording (success)",
            logger_instance=logger,
            reraise=False
        )
    
    async def _record_failure(
        self,
        task: "Task",
        execution_time: float
    ) -> None:
        """Registrar fallo en pattern learner."""
        if not self.agent.pattern_learner:
            return
        
        async def record():
            await self.agent.pattern_learner.record_command(
                task.command,
                success=False,
                execution_time=execution_time,
                result=None
            )
        
        await safe_async_call(
            record,
            operation="pattern recording (failure)",
            logger_instance=logger,
            reraise=False
        )
    
    async def _notify_completion(self, task: "Task", result: Optional[str]) -> None:
        """Enviar notificaciones y actualizar métricas de éxito."""
        # Notificaciones
        if self.agent.notifications:
            try:
                await self.agent.notifications.notify(
                    "Task Completed",
                    f"Task {task.id[:8]}... completed successfully",
                    level=self.agent.notifications.NotificationLevel.SUCCESS,
                    metadata={"task_id": task.id, "command": task.command[:50]}
                )
            except Exception as e:
                logger.error(f"Error sending completion notification: {e}")
        
        # Métricas
        if self.agent.metrics:
            try:
                self.agent.metrics.increment("tasks_completed")
                self.agent.metrics.set_gauge(
                    "tasks_completed_total",
                    sum(1 for t in self.agent.tasks.values() if t.status == "completed")
                )
            except Exception as e:
                logger.error(f"Error updating completion metrics: {e}")
        
        # Rate limiter
        if self.agent.rate_limiter:
            try:
                await self.agent.rate_limiter.complete_task()
            except Exception as e:
                logger.error(f"Error updating rate limiter: {e}")
        
        # Plugins
        if self.agent.plugin_manager:
            try:
                await self.agent.plugin_manager.notify_task_completed(task.id, result or "")
            except Exception as e:
                logger.error(f"Error notifying plugin manager: {e}")
        
        # Event bus
        if self.agent.event_bus:
            try:
                await self.agent.event_bus.publish(
                    self.agent.event_bus.EventType.TASK_COMPLETED,
                    {"task_id": task.id, "result": (result or "")[:200]},
                    source="agent"
                )
            except Exception as e:
                logger.error(f"Error publishing completion event: {e}")
    
    async def _verify_before_reporting(self, task: "Task") -> None:
        """
        Verificar críticamente antes de reportar al usuario.
        
        Integra:
        - Razonamiento automático (COMPLETION_REPORT)
        - Verificación crítica
        - Verificación de intención del usuario
        """
        if not self.agent:
            return
        
        try:
            # 1. Activar razonamiento automático
            if hasattr(self.agent, 'reasoning_trigger') and self.agent.reasoning_trigger:
                from ..reasoning_trigger import CriticalActionType
                await self.agent.reasoning_trigger.trigger_reasoning(
                    CriticalActionType.COMPLETION_REPORT,
                    context={
                        "task_id": task.id,
                        "task_description": task.command,
                        "verification": {"status": "pending"}
                    }
                )
            
            # 2. Verificación crítica
            if hasattr(self.agent, 'critical_verifier') and self.agent.critical_verifier:
                critical_result = await self.agent.critical_verifier.verify_before_reporting(
                    task.id,
                    task.command,
                    agent=self.agent
                )
                
                if not critical_result.get("can_report", False):
                    logger.warning(
                        f"⚠️ Critical verification failed for task {task.id}: "
                        f"{critical_result.get('issues', [])}"
                    )
                    
                    if hasattr(self.agent, 'devin') and self.agent.devin:
                        await self.agent.devin.message_user(
                            f"⚠️ Verificación crítica falló para la tarea {task.id[:8]}...\n"
                            f"Problemas encontrados: {', '.join(critical_result.get('issues', []))}",
                            level=self.agent.devin.CommunicationLevel.WARNING
                        )
            
            # 3. Verificación de intención del usuario
            if hasattr(self.agent, 'intent_verifier') and self.agent.intent_verifier:
                # Obtener cambios realizados si están disponibles
                changes_made = []
                if hasattr(self.agent, 'change_verifier') and self.agent.change_verifier:
                    change_sets = self.agent.change_verifier.get_all_change_sets()
                    for cs in change_sets:
                        if cs.get('id') == task.id:
                            changes_made = [
                                {
                                    "file_path": c.get("file_path"),
                                    "description": c.get("description", "")
                                }
                                for c in cs.get("changes", [])
                            ]
                            break
                
                intent_result = await self.agent.intent_verifier.verify_user_intent(
                    task.id,
                    task.command,
                    task.command,
                    changes_made=changes_made if changes_made else None,
                    agent=self.agent
                )
                
                if not intent_result.get("can_report", False):
                    logger.warning(
                        f"⚠️ Intent verification failed for task {task.id}: "
                        f"confidence={intent_result.get('confidence', 0.0):.2f}"
                    )
                    
                    if hasattr(self.agent, 'devin') and self.agent.devin:
                        await self.agent.devin.message_user(
                            f"⚠️ Verificación de intención falló para la tarea {task.id[:8]}...\n"
                            f"Confianza: {intent_result.get('confidence', 0.0):.2%}",
                            level=self.agent.devin.CommunicationLevel.WARNING
                        )
        
        except Exception as e:
            logger.error(f"Error in critical verification: {e}", exc_info=True)
    
    async def _notify_failure(self, task: "Task", error: str) -> None:
        """Enviar notificaciones y actualizar métricas de fallo."""
        # Notificaciones
        if self.agent.notifications:
            try:
                await self.agent.notifications.notify(
                    "Task Failed",
                    f"Task {task.id[:8]}... failed: {error[:100]}",
                    level=self.agent.notifications.NotificationLevel.ERROR,
                    metadata={
                        "task_id": task.id,
                        "error": error,
                        "command": task.command[:50]
                    }
                )
            except Exception as e:
                logger.error(f"Error sending failure notification: {e}")
        
        # Métricas
        if self.agent.metrics:
            try:
                self.agent.metrics.increment("tasks_failed")
                self.agent.metrics.set_gauge(
                    "tasks_failed_total",
                    sum(1 for t in self.agent.tasks.values() if t.status == "failed")
                )
            except Exception as e:
                logger.error(f"Error updating failure metrics: {e}")
        
        # Rate limiter
        if self.agent.rate_limiter:
            try:
                await self.agent.rate_limiter.complete_task()
            except Exception as e:
                logger.error(f"Error updating rate limiter: {e}")
        
        # Plugins
        if self.agent.plugin_manager:
            try:
                await self.agent.plugin_manager.notify_task_failed(task.id, error)
            except Exception as e:
                logger.error(f"Error notifying plugin manager: {e}")
        
        # Event bus
        if self.agent.event_bus:
            try:
                await self.agent.event_bus.publish(
                    self.agent.event_bus.EventType.TASK_FAILED,
                    {"task_id": task.id, "error": error[:200]},
                    source="agent"
                )
            except Exception as e:
                logger.error(f"Error publishing failure event: {e}")

