"""
Batch Service - Servicio de Procesamiento por Lotes
====================================================

Servicio para procesar operaciones en lotes.
"""

import logging
from typing import List, Dict, Any, Callable, Optional
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)


class BatchService:
    """Servicio para procesamiento por lotes"""
    
    def __init__(self, max_workers: int = 5):
        """
        Inicializar servicio de lotes
        
        Args:
            max_workers: Número máximo de workers paralelos
        """
        self.max_workers = max_workers
        logger.info(f"Batch Service inicializado con {max_workers} workers")
    
    def publish_batch(
        self,
        posts: List[Dict[str, Any]],
        connector: Any,
        on_success: Optional[Callable] = None,
        on_error: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Publicar múltiples posts en paralelo
        
        Args:
            posts: Lista de posts a publicar
            connector: Instancia de SocialMediaConnector
            on_success: Callback para éxito
            on_error: Callback para error
            
        Returns:
            Dict con resultados
        """
        results = {
            "total": len(posts),
            "success": 0,
            "failed": 0,
            "results": []
        }
        
        def publish_single(post: Dict[str, Any]):
            """Publicar un solo post"""
            try:
                post_id = post.get("id")
                content = post.get("content")
                platforms = post.get("platforms", [])
                media_paths = post.get("media_paths", [])
                
                result = connector.publish_multiple(
                    platforms=platforms,
                    content=content,
                    media_paths=media_paths
                )
                
                if on_success:
                    on_success(post_id, result)
                
                return {
                    "post_id": post_id,
                    "status": "success",
                    "result": result
                }
            except Exception as e:
                logger.error(f"Error publicando post {post.get('id')}: {e}")
                
                if on_error:
                    on_error(post.get("id"), str(e))
                
                return {
                    "post_id": post.get("id"),
                    "status": "error",
                    "error": str(e)
                }
        
        # Procesar en paralelo
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(publish_single, post): post for post in posts}
            
            for future in as_completed(futures):
                result = future.result()
                results["results"].append(result)
                
                if result["status"] == "success":
                    results["success"] += 1
                else:
                    results["failed"] += 1
        
        logger.info(f"Batch publicado: {results['success']}/{results['total']} exitosos")
        return results
    
    def schedule_batch(
        self,
        posts_data: List[Dict[str, Any]],
        scheduler: Any
    ) -> List[str]:
        """
        Programar múltiples posts
        
        Args:
            posts_data: Lista de datos de posts
            scheduler: Instancia de PostScheduler
            
        Returns:
            Lista de IDs de posts creados
        """
        post_ids = []
        
        for post_data in posts_data:
            try:
                post_id = scheduler.add_post(post_data)
                post_ids.append(post_id)
            except Exception as e:
                logger.error(f"Error programando post: {e}")
        
        logger.info(f"Batch programado: {len(post_ids)} posts")
        return post_ids
    
    def analyze_batch(
        self,
        contents: List[str],
        platform: str,
        analyzer: Any
    ) -> List[Dict[str, Any]]:
        """
        Analizar múltiples contenidos
        
        Args:
            contents: Lista de contenidos
            platform: Plataforma objetivo
            analyzer: Instancia de analizador
            
        Returns:
            Lista de análisis
        """
        analyses = []
        
        for content in contents:
            try:
                analysis = analyzer.analyze_content(content, platform)
                analyses.append(analysis)
            except Exception as e:
                logger.error(f"Error analizando contenido: {e}")
                analyses.append({
                    "content": content,
                    "error": str(e)
                })
        
        return analyses
    
    def generate_batch(
        self,
        topics: List[str],
        platform: str,
        generator: Any
    ) -> List[Dict[str, Any]]:
        """
        Generar contenido para múltiples temas
        
        Args:
            topics: Lista de temas
            platform: Plataforma objetivo
            generator: Instancia de ContentGenerator
            
        Returns:
            Lista de contenidos generados
        """
        generated = []
        
        for topic in topics:
            try:
                content = generator.generate_post(
                    topic=topic,
                    platform=platform
                )
                generated.append({
                    "topic": topic,
                    "content": content,
                    "platform": platform,
                    "status": "success"
                })
            except Exception as e:
                logger.error(f"Error generando contenido para {topic}: {e}")
                generated.append({
                    "topic": topic,
                    "content": "",
                    "platform": platform,
                    "status": "error",
                    "error": str(e)
                })
        
        return generated
    
    def validate_batch(
        self,
        posts: List[Dict[str, Any]],
        validator: Any
    ) -> Dict[str, Any]:
        """
        Validar múltiples posts
        
        Args:
            posts: Lista de posts a validar
            validator: Función o instancia de validador
            
        Returns:
            Dict con resultados de validación
        """
        results = {
            "total": len(posts),
            "valid": 0,
            "invalid": 0,
            "errors": []
        }
        
        for post in posts:
            try:
                if callable(validator):
                    is_valid = validator(post)
                else:
                    is_valid = validator.validate(post)
                
                if is_valid:
                    results["valid"] += 1
                else:
                    results["invalid"] += 1
                    results["errors"].append({
                        "post_id": post.get("id"),
                        "error": "Validación fallida"
                    })
            except Exception as e:
                results["invalid"] += 1
                results["errors"].append({
                    "post_id": post.get("id"),
                    "error": str(e)
                })
        
        return results
    
    def get_batch_progress(
        self,
        batch_id: str,
        total: int,
        completed: int
    ) -> Dict[str, Any]:
        """
        Obtener progreso de un batch
        
        Args:
            batch_id: ID del batch
            total: Total de items
            completed: Items completados
            
        Returns:
            Dict con información de progreso
        """
        percentage = (completed / total * 100) if total > 0 else 0
        
        return {
            "batch_id": batch_id,
            "total": total,
            "completed": completed,
            "remaining": total - completed,
            "percentage": round(percentage, 2),
            "status": "completed" if completed >= total else "in_progress"
        }



