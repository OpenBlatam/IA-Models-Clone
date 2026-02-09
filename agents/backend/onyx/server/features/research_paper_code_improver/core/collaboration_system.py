"""
Collaboration System - Sistema de colaboración y comentarios
=============================================================
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import json
from pathlib import Path
import uuid

logger = logging.getLogger(__name__)


class CollaborationSystem:
    """
    Sistema de colaboración con comentarios y revisiones.
    """
    
    def __init__(self, collaboration_dir: str = "data/collaboration"):
        """
        Inicializar sistema de colaboración.
        
        Args:
            collaboration_dir: Directorio para almacenar datos
        """
        self.collaboration_dir = Path(collaboration_dir)
        self.collaboration_dir.mkdir(parents=True, exist_ok=True)
    
    def add_comment(
        self,
        improvement_id: str,
        user_id: str,
        comment: str,
        line_number: Optional[int] = None,
        parent_comment_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Agrega un comentario a una mejora.
        
        Args:
            improvement_id: ID de la mejora
            user_id: ID del usuario
            comment: Texto del comentario
            line_number: Número de línea (opcional)
            parent_comment_id: ID del comentario padre (opcional, para respuestas)
            
        Returns:
            Información del comentario creado
        """
        comment_id = str(uuid.uuid4())
        
        comment_data = {
            "comment_id": comment_id,
            "improvement_id": improvement_id,
            "user_id": user_id,
            "comment": comment,
            "line_number": line_number,
            "parent_comment_id": parent_comment_id,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "reactions": {}
        }
        
        self._save_comment(comment_data)
        
        logger.info(f"Comentario agregado: {comment_id}")
        
        return comment_data
    
    def get_comments(
        self,
        improvement_id: str,
        include_replies: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Obtiene comentarios de una mejora.
        
        Args:
            improvement_id: ID de la mejora
            include_replies: Incluir respuestas
            
        Returns:
            Lista de comentarios
        """
        comments = []
        
        for comment_file in self.collaboration_dir.glob("comments/*.json"):
            try:
                with open(comment_file, "r", encoding="utf-8") as f:
                    comment = json.load(f)
                    
                    if comment.get("improvement_id") == improvement_id:
                        if not include_replies and comment.get("parent_comment_id"):
                            continue
                        comments.append(comment)
            except Exception as e:
                logger.warning(f"Error cargando comentario {comment_file}: {e}")
                continue
        
        # Ordenar por fecha
        comments.sort(key=lambda x: x["created_at"])
        
        return comments
    
    def add_reaction(
        self,
        comment_id: str,
        user_id: str,
        reaction: str
    ) -> bool:
        """
        Agrega una reacción a un comentario.
        
        Args:
            comment_id: ID del comentario
            user_id: ID del usuario
            reaction: Tipo de reacción (👍, ❤️, etc.)
            
        Returns:
            True si se agregó exitosamente
        """
        comment = self._load_comment(comment_id)
        if not comment:
            return False
        
        if "reactions" not in comment:
            comment["reactions"] = {}
        
        if reaction not in comment["reactions"]:
            comment["reactions"][reaction] = []
        
        if user_id not in comment["reactions"][reaction]:
            comment["reactions"][reaction].append(user_id)
            comment["updated_at"] = datetime.now().isoformat()
            self._save_comment(comment)
            return True
        
        return False
    
    def create_review(
        self,
        improvement_id: str,
        reviewer_id: str,
        status: str,
        comments: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Crea una revisión de una mejora.
        
        Args:
            improvement_id: ID de la mejora
            reviewer_id: ID del revisor
            status: Estado (approved, rejected, needs_changes)
            comments: Comentarios de revisión (opcional)
            
        Returns:
            Información de la revisión
        """
        review_id = str(uuid.uuid4())
        
        review = {
            "review_id": review_id,
            "improvement_id": improvement_id,
            "reviewer_id": reviewer_id,
            "status": status,
            "comments": comments or [],
            "created_at": datetime.now().isoformat()
        }
        
        self._save_review(review)
        
        logger.info(f"Revisión creada: {review_id} ({status})")
        
        return review
    
    def _save_comment(self, comment: Dict[str, Any]):
        """Guarda comentario en disco"""
        try:
            comments_dir = self.collaboration_dir / "comments"
            comments_dir.mkdir(parents=True, exist_ok=True)
            
            comment_file = comments_dir / f"{comment['comment_id']}.json"
            with open(comment_file, "w", encoding="utf-8") as f:
                json.dump(comment, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error guardando comentario: {e}")
    
    def _load_comment(self, comment_id: str) -> Optional[Dict[str, Any]]:
        """Carga un comentario"""
        comment_file = self.collaboration_dir / "comments" / f"{comment_id}.json"
        
        if not comment_file.exists():
            return None
        
        try:
            with open(comment_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error cargando comentario: {e}")
            return None
    
    def _save_review(self, review: Dict[str, Any]):
        """Guarda revisión en disco"""
        try:
            reviews_dir = self.collaboration_dir / "reviews"
            reviews_dir.mkdir(parents=True, exist_ok=True)
            
            review_file = reviews_dir / f"{review['review_id']}.json"
            with open(review_file, "w", encoding="utf-8") as f:
                json.dump(review, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error guardando revisión: {e}")




