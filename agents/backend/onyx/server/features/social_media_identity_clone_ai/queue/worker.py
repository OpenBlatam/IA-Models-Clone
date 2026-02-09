"""
Workers para procesar tareas de la cola
"""

import logging
import asyncio
from typing import Dict, Any, Callable

from .task_queue import TaskQueue, Task, TaskStatus, get_task_queue
from ..services.profile_extractor import ProfileExtractor
from ..services.identity_analyzer import IdentityAnalyzer
from ..services.content_generator import ContentGenerator
from ..services.storage_service import StorageService
from ..notifications.notification_service import get_notification_service, NotificationType

logger = logging.getLogger(__name__)


class Worker:
    """Worker para procesar tareas"""
    
    def __init__(self, task_queue: TaskQueue, worker_id: str):
        self.task_queue = task_queue
        self.worker_id = worker_id
        self.running = False
        self.processors: Dict[str, Callable] = {
            "extract_profile": self._process_extract_profile,
            "build_identity": self._process_build_identity,
            "generate_content": self._process_generate_content,
        }
    
    async def start(self):
        """Inicia el worker"""
        self.running = True
        logger.info(f"Worker {self.worker_id} iniciado")
        
        while self.running:
            try:
                task = await self.task_queue.get_task()
                if task:
                    await self._process_task(task)
                else:
                    await asyncio.sleep(1)  # Esperar si no hay tareas
            except Exception as e:
                logger.error(f"Error en worker {self.worker_id}: {e}", exc_info=True)
                await asyncio.sleep(1)
    
    async def stop(self):
        """Detiene el worker"""
        self.running = False
        logger.info(f"Worker {self.worker_id} detenido")
    
    async def _process_task(self, task: Task):
        """Procesa una tarea"""
        logger.info(f"Procesando tarea {task.task_id} ({task.task_type})")
        
        await self.task_queue.update_task_status(task.task_id, TaskStatus.PROCESSING)
        
        try:
            processor = self.processors.get(task.task_type)
            if not processor:
                raise ValueError(f"Tipo de tarea desconocido: {task.task_type}")
            
            result = await processor(task.payload)
            
            await self.task_queue.update_task_status(
                task.task_id,
                TaskStatus.COMPLETED,
                result=result
            )
            
            # Enviar notificación
            notification_service = get_notification_service()
            notification_service.create_notification(
                notification_type=NotificationType.TASK_COMPLETED,
                title="Tarea completada",
                message=f"La tarea {task.task_type} se completó exitosamente",
                data={"task_id": task.task_id, "task_type": task.task_type, "result": result}
            )
            
            logger.info(f"Tarea {task.task_id} completada exitosamente")
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error procesando tarea {task.task_id}: {error_msg}", exc_info=True)
            
            # Reintentar si es posible
            if task.retry_count < task.max_retries:
                task.retry_count += 1
                await self.task_queue.update_task_status(
                    task.task_id,
                    TaskStatus.PENDING,
                    error=f"Retry {task.retry_count}/{task.max_retries}: {error_msg}"
                )
                # Re-enqueue
                await self.task_queue._enqueue_task(task.task_id)
            else:
                await self.task_queue.update_task_status(
                    task.task_id,
                    TaskStatus.FAILED,
                    error=error_msg
                )
                
                # Enviar notificación de error
                notification_service = get_notification_service()
                notification_service.create_notification(
                    notification_type=NotificationType.TASK_FAILED,
                    title="Tarea falló",
                    message=f"La tarea {task.task_type} falló después de {task.max_retries} intentos",
                    data={"task_id": task.task_id, "task_type": task.task_type, "error": error_msg}
                )
    
    async def _process_extract_profile(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Procesa extracción de perfil"""
        platform = payload["platform"]
        username = payload["username"]
        use_cache = payload.get("use_cache", True)
        
        extractor = ProfileExtractor()
        
        if platform == "tiktok":
            profile = await extractor.extract_tiktok_profile(username, use_cache=use_cache)
        elif platform == "instagram":
            profile = await extractor.extract_instagram_profile(username, use_cache=use_cache)
        elif platform == "youtube":
            profile = await extractor.extract_youtube_profile(username, use_cache=use_cache)
        else:
            raise ValueError(f"Plataforma no soportada: {platform}")
        
        return {
            "success": True,
            "profile": profile.model_dump()
        }
    
    async def _process_build_identity(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Procesa construcción de identidad"""
        extractor = ProfileExtractor()
        analyzer = IdentityAnalyzer()
        storage = StorageService()
        
        tiktok_profile = None
        instagram_profile = None
        youtube_profile = None
        
        if payload.get("tiktok_username"):
            tiktok_profile = await extractor.extract_tiktok_profile(
                payload["tiktok_username"]
            )
        
        if payload.get("instagram_username"):
            instagram_profile = await extractor.extract_instagram_profile(
                payload["instagram_username"]
            )
        
        if payload.get("youtube_channel_id"):
            youtube_profile = await extractor.extract_youtube_profile(
                payload["youtube_channel_id"]
            )
        
        identity = await analyzer.build_identity(
            tiktok_profile=tiktok_profile,
            instagram_profile=instagram_profile,
            youtube_profile=youtube_profile
        )
        
        storage.save_identity(identity)
        
        return {
            "success": True,
            "identity_id": identity.profile_id,
            "identity": identity.model_dump()
        }
    
    async def _process_generate_content(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Procesa generación de contenido"""
        storage = StorageService()
        identity = storage.get_identity(payload["identity_profile_id"])
        
        if not identity:
            raise ValueError(f"Identidad no encontrada: {payload['identity_profile_id']}")
        
        generator = ContentGenerator(identity_profile=identity)
        
        platform = payload["platform"]
        content_type = payload["content_type"]
        
        if platform == "instagram" and content_type == "post":
            generated = await generator.generate_instagram_post(
                topic=payload.get("topic"),
                style=payload.get("style")
            )
        elif platform == "tiktok" and content_type == "video":
            generated = await generator.generate_tiktok_script(
                topic=payload.get("topic"),
                duration=payload.get("duration", 60)
            )
        elif platform == "youtube" and content_type == "video":
            generated = await generator.generate_youtube_description(
                video_title=payload.get("video_title", "Video"),
                tags=payload.get("tags")
            )
        else:
            raise ValueError(f"Combinación no soportada: {platform}/{content_type}")
        
        storage.save_generated_content(generated)
        
        return {
            "success": True,
            "content_id": generated.content_id,
            "content": generated.model_dump()
        }


async def start_workers(num_workers: int = 2):
    """Inicia workers para procesar tareas"""
    task_queue = get_task_queue()
    workers = []
    
    for i in range(num_workers):
        worker = Worker(task_queue, f"worker-{i+1}")
        workers.append(worker)
        asyncio.create_task(worker.start())
    
    logger.info(f"Iniciados {num_workers} workers")
    return workers

