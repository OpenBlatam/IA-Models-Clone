"""
Servicio de Comunidad - Grupos de apoyo y conexión social
"""

from typing import Dict, List, Optional
from datetime import datetime
from enum import Enum


class PostType(str, Enum):
    """Tipos de publicaciones"""
    SUCCESS_STORY = "success_story"
    SUPPORT_REQUEST = "support_request"
    TIP = "tip"
    QUESTION = "question"
    MILESTONE = "milestone"


class CommunityService:
    """Servicio de comunidad y grupos de apoyo"""
    
    def __init__(self):
        """Inicializa el servicio de comunidad"""
        pass
    
    def create_post(
        self,
        user_id: str,
        post_type: str,
        title: str,
        content: str,
        is_anonymous: bool = False
    ) -> Dict:
        """
        Crea una publicación en la comunidad
        
        Args:
            user_id: ID del usuario
            post_type: Tipo de publicación
            title: Título
            content: Contenido
            is_anonymous: Si es anónima
        
        Returns:
            Publicación creada
        """
        post = {
            "id": f"post_{datetime.now().timestamp()}",
            "user_id": user_id if not is_anonymous else None,
            "post_type": post_type,
            "title": title,
            "content": content,
            "is_anonymous": is_anonymous,
            "likes": 0,
            "comments": [],
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }
        
        return post
    
    def get_community_posts(
        self,
        post_type: Optional[str] = None,
        limit: int = 20,
        offset: int = 0
    ) -> List[Dict]:
        """
        Obtiene publicaciones de la comunidad
        
        Args:
            post_type: Filtrar por tipo (opcional)
            limit: Límite de resultados
            offset: Offset para paginación
        
        Returns:
            Lista de publicaciones
        """
        # En implementación real, esto vendría de la base de datos
        posts = []
        
        # Ejemplos de publicaciones
        example_posts = [
            {
                "id": "post_1",
                "user_id": "user_1",
                "post_type": PostType.SUCCESS_STORY,
                "title": "30 días completados!",
                "content": "Hoy cumplo 30 días de sobriedad. Ha sido difícil pero valió la pena.",
                "likes": 15,
                "comments": 3,
                "created_at": datetime.now().isoformat()
            },
            {
                "id": "post_2",
                "user_id": "user_2",
                "post_type": PostType.TIP,
                "title": "Consejo que me ayudó",
                "content": "Cuando tengo cravings, bebo un vaso de agua fría y respiro profundamente.",
                "likes": 8,
                "comments": 2,
                "created_at": datetime.now().isoformat()
            }
        ]
        
        if post_type:
            posts = [p for p in example_posts if p.get("post_type") == post_type]
        else:
            posts = example_posts
        
        return posts[offset:offset+limit]
    
    def like_post(self, post_id: str, user_id: str) -> Dict:
        """
        Da like a una publicación
        
        Args:
            post_id: ID de la publicación
            user_id: ID del usuario
        
        Returns:
            Resultado del like
        """
        return {
            "post_id": post_id,
            "user_id": user_id,
            "liked": True,
            "timestamp": datetime.now().isoformat()
        }
    
    def add_comment(
        self,
        post_id: str,
        user_id: str,
        comment: str
    ) -> Dict:
        """
        Agrega un comentario a una publicación
        
        Args:
            post_id: ID de la publicación
            user_id: ID del usuario
            comment: Comentario
        
        Returns:
            Comentario creado
        """
        comment_obj = {
            "id": f"comment_{datetime.now().timestamp()}",
            "post_id": post_id,
            "user_id": user_id,
            "comment": comment,
            "created_at": datetime.now().isoformat()
        }
        
        return comment_obj
    
    def create_support_group(
        self,
        name: str,
        description: str,
        addiction_type: str,
        is_private: bool = False
    ) -> Dict:
        """
        Crea un grupo de apoyo
        
        Args:
            name: Nombre del grupo
            description: Descripción
            addiction_type: Tipo de adicción
            is_private: Si es privado
        
        Returns:
            Grupo creado
        """
        group = {
            "id": f"group_{datetime.now().timestamp()}",
            "name": name,
            "description": description,
            "addiction_type": addiction_type,
            "is_private": is_private,
            "members": [],
            "created_at": datetime.now().isoformat()
        }
        
        return group
    
    def join_group(self, group_id: str, user_id: str) -> Dict:
        """
        Une un usuario a un grupo
        
        Args:
            group_id: ID del grupo
            user_id: ID del usuario
        
        Returns:
            Resultado de la unión
        """
        return {
            "group_id": group_id,
            "user_id": user_id,
            "joined_at": datetime.now().isoformat(),
            "status": "joined"
        }
    
    def get_user_groups(self, user_id: str) -> List[Dict]:
        """
        Obtiene grupos del usuario
        
        Args:
            user_id: ID del usuario
        
        Returns:
            Lista de grupos
        """
        # En implementación real, esto vendría de la base de datos
        return []
    
    def get_support_resources(self, location: Optional[str] = None) -> List[Dict]:
        """
        Obtiene recursos de apoyo locales
        
        Args:
            location: Ubicación (opcional)
        
        Returns:
            Lista de recursos
        """
        resources = [
            {
                "type": "support_group",
                "name": "Grupos de 12 Pasos",
                "description": "Grupos locales de AA, NA, etc.",
                "website": "https://www.aa.org"
            },
            {
                "type": "therapy",
                "name": "Terapia y Consejería",
                "description": "Profesionales especializados en adicciones",
                "website": "https://www.psychologytoday.com"
            },
            {
                "type": "online",
                "name": "Comunidades en Línea",
                "description": "Foros y grupos de apoyo en línea",
                "website": "https://www.reddit.com/r/stopdrinking"
            }
        ]
        
        return resources

