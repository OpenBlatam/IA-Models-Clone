"""
Post Scheduler - Programador de Publicaciones
==============================================

Sistema de programación y gestión de cola de publicaciones.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import uuid
from threading import Lock

logger = logging.getLogger(__name__)


class PostScheduler:
    """Programador de publicaciones en redes sociales"""
    
    def __init__(self):
        """Inicializar el programador"""
        self.posts: Dict[str, Dict[str, Any]] = {}
        self.queue: List[str] = []  # IDs ordenados por fecha
        self.lock = Lock()
        logger.info("Post Scheduler inicializado")
    
    def add_post(self, post_data: Dict[str, Any]) -> str:
        """
        Agregar un post a la cola de programación
        
        Args:
            post_data: Datos del post a programar
            
        Returns:
            ID único del post
        """
        post_id = str(uuid.uuid4())
        
        post_entry = {
            "id": post_id,
            "status": "scheduled",
            **post_data
        }
        
        with self.lock:
            self.posts[post_id] = post_entry
            
            # Insertar en la cola ordenada por fecha
            scheduled_time = post_data.get("scheduled_time", datetime.now())
            self._insert_sorted(post_id, scheduled_time)
        
        logger.info(f"Post agregado a la cola: {post_id}")
        return post_id
    
    def _insert_sorted(self, post_id: str, scheduled_time: datetime):
        """Insertar post en la cola ordenada por fecha"""
        if not self.queue:
            self.queue.append(post_id)
            return
        
        # Insertar en orden cronológico
        for i, existing_id in enumerate(self.queue):
            existing_time = self.posts[existing_id].get("scheduled_time", datetime.max)
            if scheduled_time < existing_time:
                self.queue.insert(i, post_id)
                return
        
        # Si no se insertó, agregar al final
        self.queue.append(post_id)
    
    def get_pending_posts(
        self,
        limit: Optional[int] = None,
        before_time: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """
        Obtener posts pendientes de publicación
        
        Args:
            limit: Límite de posts a retornar
            before_time: Solo posts programados antes de esta fecha
            
        Returns:
            Lista de posts pendientes
        """
        now = datetime.now()
        pending = []
        
        with self.lock:
            for post_id in self.queue:
                post = self.posts.get(post_id)
                if not post:
                    continue
                
                if post.get("status") != "scheduled":
                    continue
                
                scheduled_time = post.get("scheduled_time")
                if scheduled_time and scheduled_time > now:
                    continue
                
                if before_time and scheduled_time and scheduled_time > before_time:
                    continue
                
                pending.append(post)
                
                if limit and len(pending) >= limit:
                    break
        
        return pending
    
    def mark_as_published(self, post_id: str, results: Dict[str, Any]):
        """
        Marcar un post como publicado
        
        Args:
            post_id: ID del post
            results: Resultados de la publicación
        """
        with self.lock:
            if post_id in self.posts:
                self.posts[post_id]["status"] = "published"
                self.posts[post_id]["published_at"] = datetime.now()
                self.posts[post_id]["results"] = results
                
                # Remover de la cola
                if post_id in self.queue:
                    self.queue.remove(post_id)
                
                logger.info(f"Post marcado como publicado: {post_id}")
    
    def cancel_post(self, post_id: str) -> bool:
        """
        Cancelar un post programado
        
        Args:
            post_id: ID del post
            
        Returns:
            True si se canceló exitosamente
        """
        with self.lock:
            if post_id in self.posts:
                self.posts[post_id]["status"] = "cancelled"
                if post_id in self.queue:
                    self.queue.remove(post_id)
                logger.info(f"Post cancelado: {post_id}")
                return True
            return False
    
    def get_post(self, post_id: str) -> Optional[Dict[str, Any]]:
        """
        Obtener información de un post
        
        Args:
            post_id: ID del post
            
        Returns:
            Datos del post o None
        """
        return self.posts.get(post_id)
    
    def get_all_posts(
        self,
        status: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Obtener todos los posts
        
        Args:
            status: Filtrar por status (scheduled, published, cancelled)
            
        Returns:
            Lista de posts
        """
        with self.lock:
            posts = list(self.posts.values())
            
            if status:
                posts = [p for p in posts if p.get("status") == status]
            
            return posts
    
    def update_post(
        self,
        post_id: str,
        updates: Dict[str, Any]
    ) -> bool:
        """
        Actualizar un post programado
        
        Args:
            post_id: ID del post
            updates: Diccionario con campos a actualizar
            
        Returns:
            True si se actualizó exitosamente
        """
        with self.lock:
            if post_id not in self.posts:
                logger.warning(f"Post {post_id} no encontrado para actualizar")
                return False
            
            post = self.posts[post_id]
            
            if post.get("status") != "scheduled":
                logger.warning(f"No se puede actualizar post {post_id} con status {post.get('status')}")
                return False
            
            old_scheduled_time = post.get("scheduled_time")
            
            for key, value in updates.items():
                if key in ["id", "status"]:
                    continue
                post[key] = value
            
            post["updated_at"] = datetime.now()
            
            new_scheduled_time = post.get("scheduled_time")
            if new_scheduled_time != old_scheduled_time:
                if post_id in self.queue:
                    self.queue.remove(post_id)
                self._insert_sorted(post_id, new_scheduled_time)
            
            logger.info(f"Post {post_id} actualizado")
            return True



