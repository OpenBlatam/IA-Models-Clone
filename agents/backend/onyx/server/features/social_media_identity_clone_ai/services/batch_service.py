"""
Servicio para procesamiento por lotes
"""

import logging
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime

from ..services.profile_extractor import ProfileExtractor
from ..services.identity_analyzer import IdentityAnalyzer
from ..services.storage_service import StorageService
from ..queue.task_queue import get_task_queue
from ..analytics.metrics import get_metrics_collector

logger = logging.getLogger(__name__)


class BatchService:
    """Servicio para operaciones por lotes"""
    
    def __init__(self):
        self.extractor = ProfileExtractor()
        self.analyzer = IdentityAnalyzer()
        self.storage = StorageService()
        self.task_queue = get_task_queue()
        self.metrics = get_metrics_collector()
    
    async def batch_extract_profiles(
        self,
        profiles: List[Dict[str, str]],
        use_async_tasks: bool = True
    ) -> Dict[str, Any]:
        """
        Extrae múltiples perfiles en lote
        
        Args:
            profiles: Lista de dicts con 'platform' y 'username'
            use_async_tasks: Si usar tareas asíncronas
            
        Returns:
            Resultados del batch
        """
        logger.info(f"Procesando batch de {len(profiles)} perfiles")
        self.metrics.increment("batch_extract_profiles", tags={"count": str(len(profiles))})
        
        if use_async_tasks:
            # Crear tareas asíncronas
            task_ids = []
            for profile in profiles:
                task_id = await self.task_queue.add_task(
                    task_type="extract_profile",
                    payload={
                        "platform": profile["platform"],
                        "username": profile["username"],
                        "use_cache": profile.get("use_cache", True)
                    }
                )
                task_ids.append({
                    "task_id": task_id,
                    "platform": profile["platform"],
                    "username": profile["username"]
                })
            
            return {
                "success": True,
                "method": "async",
                "total": len(profiles),
                "tasks": task_ids
            }
        else:
            # Procesar síncronamente
            results = []
            errors = []
            
            for i, profile in enumerate(profiles):
                try:
                    logger.info(f"Extrayendo perfil {i+1}/{len(profiles)}: {profile['platform']}/{profile['username']}")
                    
                    if profile["platform"] == "tiktok":
                        result = await self.extractor.extract_tiktok_profile(
                            profile["username"],
                            use_cache=profile.get("use_cache", True)
                        )
                    elif profile["platform"] == "instagram":
                        result = await self.extractor.extract_instagram_profile(
                            profile["username"],
                            use_cache=profile.get("use_cache", True)
                        )
                    elif profile["platform"] == "youtube":
                        result = await self.extractor.extract_youtube_profile(
                            profile["username"],
                            use_cache=profile.get("use_cache", True)
                        )
                    else:
                        raise ValueError(f"Plataforma no soportada: {profile['platform']}")
                    
                    results.append({
                        "platform": profile["platform"],
                        "username": profile["username"],
                        "success": True,
                        "profile": result.model_dump()
                    })
                    
                except Exception as e:
                    logger.error(f"Error extrayendo perfil {profile}: {e}")
                    errors.append({
                        "platform": profile["platform"],
                        "username": profile["username"],
                        "error": str(e)
                    })
            
            return {
                "success": True,
                "method": "sync",
                "total": len(profiles),
                "successful": len(results),
                "failed": len(errors),
                "results": results,
                "errors": errors
            }
    
    async def batch_generate_content(
        self,
        identity_id: str,
        content_requests: List[Dict[str, Any]],
        use_async_tasks: bool = True
    ) -> Dict[str, Any]:
        """
        Genera múltiples contenidos en lote
        
        Args:
            identity_id: ID de la identidad
            content_requests: Lista de requests de contenido
            use_async_tasks: Si usar tareas asíncronas
            
        Returns:
            Resultados del batch
        """
        logger.info(f"Generando batch de {len(content_requests)} contenidos")
        self.metrics.increment("batch_generate_content", tags={"count": str(len(content_requests))})
        
        if use_async_tasks:
            task_ids = []
            for request in content_requests:
                task_id = await self.task_queue.add_task(
                    task_type="generate_content",
                    payload={
                        "identity_profile_id": identity_id,
                        "platform": request["platform"],
                        "content_type": request["content_type"],
                        "topic": request.get("topic"),
                        "style": request.get("style"),
                        "duration": request.get("duration"),
                        "video_title": request.get("video_title"),
                        "tags": request.get("tags")
                    }
                )
                task_ids.append({
                    "task_id": task_id,
                    "platform": request["platform"],
                    "content_type": request["content_type"]
                })
            
            return {
                "success": True,
                "method": "async",
                "identity_id": identity_id,
                "total": len(content_requests),
                "tasks": task_ids
            }
        else:
            from ..services.content_generator import ContentGenerator
            from ..core.models import Platform, ContentType
            
            identity = self.storage.get_identity(identity_id)
            if not identity:
                raise ValueError(f"Identidad no encontrada: {identity_id}")
            
            generator = ContentGenerator(identity_profile=identity)
            results = []
            errors = []
            
            for request in content_requests:
                try:
                    platform = Platform(request["platform"])
                    content_type = ContentType(request["content_type"])
                    
                    if platform == Platform.INSTAGRAM and content_type == ContentType.POST:
                        generated = await generator.generate_instagram_post(
                            topic=request.get("topic"),
                            style=request.get("style")
                        )
                    elif platform == Platform.TIKTOK and content_type == ContentType.VIDEO:
                        generated = await generator.generate_tiktok_script(
                            topic=request.get("topic"),
                            duration=request.get("duration", 60)
                        )
                    elif platform == Platform.YOUTUBE and content_type == ContentType.VIDEO:
                        generated = await generator.generate_youtube_description(
                            video_title=request.get("video_title", "Video"),
                            tags=request.get("tags")
                        )
                    else:
                        raise ValueError(f"Combinación no soportada: {platform}/{content_type}")
                    
                    self.storage.save_generated_content(generated)
                    
                    results.append({
                        "platform": request["platform"],
                        "content_type": request["content_type"],
                        "success": True,
                        "content_id": generated.content_id
                    })
                    
                except Exception as e:
                    logger.error(f"Error generando contenido {request}: {e}")
                    errors.append({
                        "platform": request.get("platform"),
                        "content_type": request.get("content_type"),
                        "error": str(e)
                    })
            
            return {
                "success": True,
                "method": "sync",
                "identity_id": identity_id,
                "total": len(content_requests),
                "successful": len(results),
                "failed": len(errors),
                "results": results,
                "errors": errors
            }




