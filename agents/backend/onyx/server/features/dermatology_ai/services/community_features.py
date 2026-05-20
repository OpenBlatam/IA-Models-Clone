"""
Sistema de características de comunidad
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
import uuid


@dataclass
class CommunityPost:
    """Post de comunidad"""
    id: str
    user_id: str
    username: str
    title: str
    content: str
    post_type: str  # "question", "tip", "review", "progress"
    tags: List[str] = None
    likes: int = 0
    comments_count: int = 0
    views: int = 0
    created_at: str = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
        if self.tags is None:
            self.tags = []
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario"""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "username": self.username,
            "title": self.title,
            "content": self.content,
            "post_type": self.post_type,
            "tags": self.tags,
            "likes": self.likes,
            "comments_count": self.comments_count,
            "views": self.views,
            "created_at": self.created_at
        }


@dataclass
class PostComment:
    """Comentario en post"""
    id: str
    post_id: str
    user_id: str
    username: str
    content: str
    likes: int = 0
    created_at: str = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario"""
        return {
            "id": self.id,
            "post_id": self.post_id,
            "user_id": self.user_id,
            "username": self.username,
            "content": self.content,
            "likes": self.likes,
            "created_at": self.created_at
        }


class CommunityFeatures:
    """Sistema de características de comunidad"""
    
    def __init__(self):
        """Inicializa el sistema"""
        self.posts: Dict[str, CommunityPost] = {}  # post_id -> post
        self.comments: Dict[str, List[PostComment]] = {}  # post_id -> [comments]
        self.likes: Dict[str, set] = {}  # post_id -> {user_ids}
        self.user_posts: Dict[str, List[str]] = {}  # user_id -> [post_ids]
    
    def create_post(self, user_id: str, username: str, title: str,
                   content: str, post_type: str, tags: Optional[List[str]] = None) -> CommunityPost:
        """Crea un post"""
        post = CommunityPost(
            id=str(uuid.uuid4()),
            user_id=user_id,
            username=username,
            title=title,
            content=content,
            post_type=post_type,
            tags=tags or []
        )
        
        self.posts[post.id] = post
        
        if user_id not in self.user_posts:
            self.user_posts[user_id] = []
        self.user_posts[user_id].append(post.id)
        
        return post
    
    def add_comment(self, post_id: str, user_id: str, username: str,
                   content: str) -> PostComment:
        """Agrega comentario a un post"""
        if post_id not in self.posts:
            raise ValueError("Post not found")
        
        comment = PostComment(
            id=str(uuid.uuid4()),
            post_id=post_id,
            user_id=user_id,
            username=username,
            content=content
        )
        
        if post_id not in self.comments:
            self.comments[post_id] = []
        
        self.comments[post_id].append(comment)
        
        # Actualizar contador
        self.posts[post_id].comments_count = len(self.comments[post_id])
        
        return comment
    
    def like_post(self, post_id: str, user_id: str) -> bool:
        """Da like a un post"""
        if post_id not in self.posts:
            return False
        
        if post_id not in self.likes:
            self.likes[post_id] = set()
        
        if user_id not in self.likes[post_id]:
            self.likes[post_id].add(user_id)
            self.posts[post_id].likes = len(self.likes[post_id])
            return True
        
        return False
    
    def get_posts(self, limit: int = 50, post_type: Optional[str] = None,
                 tags: Optional[List[str]] = None, sort_by: str = "newest") -> List[CommunityPost]:
        """Obtiene posts"""
        posts = list(self.posts.values())
        
        # Filtrar por tipo
        if post_type:
            posts = [p for p in posts if p.post_type == post_type]
        
        # Filtrar por tags
        if tags:
            posts = [p for p in posts if any(tag in p.tags for tag in tags)]
        
        # Ordenar
        if sort_by == "newest":
            posts.sort(key=lambda x: x.created_at, reverse=True)
        elif sort_by == "most_liked":
            posts.sort(key=lambda x: x.likes, reverse=True)
        elif sort_by == "most_commented":
            posts.sort(key=lambda x: x.comments_count, reverse=True)
        elif sort_by == "most_viewed":
            posts.sort(key=lambda x: x.views, reverse=True)
        
        return posts[:limit]
    
    def get_post_comments(self, post_id: str) -> List[PostComment]:
        """Obtiene comentarios de un post"""
        return self.comments.get(post_id, [])
    
    def get_user_posts(self, user_id: str) -> List[CommunityPost]:
        """Obtiene posts del usuario"""
        post_ids = self.user_posts.get(user_id, [])
        return [self.posts[pid] for pid in post_ids if pid in self.posts]






