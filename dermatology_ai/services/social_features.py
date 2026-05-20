"""
Sistema de características sociales
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
from collections import defaultdict
import uuid


@dataclass
class UserConnection:
    """Conexión entre usuarios"""
    user_id: str
    connected_user_id: str
    connection_type: str  # "follow", "friend", etc.
    created_at: str = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario"""
        return {
            "user_id": self.user_id,
            "connected_user_id": self.connected_user_id,
            "connection_type": self.connection_type,
            "created_at": self.created_at
        }


@dataclass
class SocialPost:
    """Post social"""
    id: str
    user_id: str
    content: str
    analysis_id: Optional[str] = None
    likes: int = 0
    comments: int = 0
    shares: int = 0
    created_at: str = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario"""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "content": self.content,
            "analysis_id": self.analysis_id,
            "likes": self.likes,
            "comments": self.comments,
            "shares": self.shares,
            "created_at": self.created_at
        }


class SocialFeatures:
    """Sistema de características sociales"""
    
    def __init__(self):
        """Inicializa el sistema"""
        self.connections: List[UserConnection] = []
        self.posts: Dict[str, SocialPost] = {}
        self.post_likes: Dict[str, set] = {}  # post_id -> {user_ids}
        self.post_comments: Dict[str, List[Dict]] = {}  # post_id -> [comments]
    
    def follow_user(self, user_id: str, follow_user_id: str) -> bool:
        """
        Sigue a un usuario
        
        Args:
            user_id: ID del usuario
            follow_user_id: ID del usuario a seguir
            
        Returns:
            True si se siguió correctamente
        """
        # Verificar si ya sigue
        if any(c.user_id == user_id and c.connected_user_id == follow_user_id
               for c in self.connections):
            return False
        
        connection = UserConnection(
            user_id=user_id,
            connected_user_id=follow_user_id,
            connection_type="follow"
        )
        
        self.connections.append(connection)
        return True
    
    def unfollow_user(self, user_id: str, unfollow_user_id: str) -> bool:
        """Deja de seguir a un usuario"""
        self.connections = [
            c for c in self.connections
            if not (c.user_id == user_id and c.connected_user_id == unfollow_user_id)
        ]
        return True
    
    def get_followers(self, user_id: str) -> List[str]:
        """Obtiene seguidores de un usuario"""
        return [
            c.user_id for c in self.connections
            if c.connected_user_id == user_id and c.connection_type == "follow"
        ]
    
    def get_following(self, user_id: str) -> List[str]:
        """Obtiene usuarios que sigue"""
        return [
            c.connected_user_id for c in self.connections
            if c.user_id == user_id and c.connection_type == "follow"
        ]
    
    def create_post(self, user_id: str, content: str,
                   analysis_id: Optional[str] = None) -> str:
        """
        Crea un post
        
        Args:
            user_id: ID del usuario
            content: Contenido del post
            analysis_id: ID del análisis (opcional)
            
        Returns:
            ID del post
        """
        post_id = str(uuid.uuid4())
        
        post = SocialPost(
            id=post_id,
            user_id=user_id,
            content=content,
            analysis_id=analysis_id
        )
        
        self.posts[post_id] = post
        return post_id
    
    def like_post(self, post_id: str, user_id: str) -> bool:
        """Da like a un post"""
        if post_id not in self.posts:
            return False
        
        if post_id not in self.post_likes:
            self.post_likes[post_id] = set()
        
        if user_id in self.post_likes[post_id]:
            # Unlike
            self.post_likes[post_id].discard(user_id)
            self.posts[post_id].likes -= 1
        else:
            # Like
            self.post_likes[post_id].add(user_id)
            self.posts[post_id].likes += 1
        
        return True
    
    def comment_post(self, post_id: str, user_id: str, comment: str) -> str:
        """Comenta un post"""
        if post_id not in self.posts:
            return ""
        
        comment_id = str(uuid.uuid4())
        
        if post_id not in self.post_comments:
            self.post_comments[post_id] = []
        
        self.post_comments[post_id].append({
            "id": comment_id,
            "user_id": user_id,
            "comment": comment,
            "created_at": datetime.now().isoformat()
        })
        
        self.posts[post_id].comments += 1
        return comment_id
    
    def get_feed(self, user_id: str, limit: int = 20) -> List[SocialPost]:
        """Obtiene feed del usuario"""
        following = self.get_following(user_id)
        following.append(user_id)  # Incluir posts propios
        
        feed_posts = [
            post for post in self.posts.values()
            if post.user_id in following
        ]
        
        # Ordenar por fecha (más reciente primero)
        feed_posts.sort(key=lambda p: p.created_at, reverse=True)
        
        return feed_posts[:limit]
    
    def get_user_stats(self, user_id: str) -> Dict:
        """Obtiene estadísticas sociales de un usuario"""
        user_posts = [p for p in self.posts.values() if p.user_id == user_id]
        
        total_likes = sum(p.likes for p in user_posts)
        total_comments = sum(p.comments for p in user_posts)
        
        return {
            "user_id": user_id,
            "followers": len(self.get_followers(user_id)),
            "following": len(self.get_following(user_id)),
            "posts_count": len(user_posts),
            "total_likes": total_likes,
            "total_comments": total_comments
        }






