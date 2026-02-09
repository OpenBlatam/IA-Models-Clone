"""
Community Service - Sistema de comunidad y foros
=================================================

Sistema de comunidad donde los usuarios pueden compartir experiencias,
hacer preguntas y ayudarse mutuamente.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class PostType(str, Enum):
    """Tipos de posts"""
    QUESTION = "question"
    EXPERIENCE = "experience"
    TIP = "tip"
    SUCCESS_STORY = "success_story"
    RESOURCE = "resource"
    DISCUSSION = "discussion"


class PostStatus(str, Enum):
    """Estado del post"""
    DRAFT = "draft"
    PUBLISHED = "published"
    ARCHIVED = "archived"
    DELETED = "deleted"


@dataclass
class Post:
    """Post en la comunidad"""
    id: str
    user_id: str
    title: str
    content: str
    post_type: PostType
    tags: List[str] = field(default_factory=list)
    status: PostStatus = PostStatus.PUBLISHED
    likes: int = 0
    views: int = 0
    comments_count: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: Optional[datetime] = None
    pinned: bool = False


@dataclass
class Comment:
    """Comentario en un post"""
    id: str
    post_id: str
    user_id: str
    content: str
    likes: int = 0
    parent_comment_id: Optional[str] = None  # Para respuestas anidadas
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: Optional[datetime] = None


@dataclass
class UserProfile:
    """Perfil de usuario en la comunidad"""
    user_id: str
    username: str
    bio: Optional[str] = None
    reputation: int = 0
    posts_count: int = 0
    comments_count: int = 0
    helpful_votes: int = 0
    badges: List[str] = field(default_factory=list)
    joined_at: datetime = field(default_factory=datetime.now)


class CommunityService:
    """Servicio de comunidad"""
    
    def __init__(self):
        """Inicializar servicio"""
        self.posts: Dict[str, Post] = {}
        self.comments: Dict[str, List[Comment]] = {}  # post_id -> [comments]
        self.user_profiles: Dict[str, UserProfile] = {}
        self.post_likes: Dict[str, set] = {}  # post_id -> {user_ids}
        self.comment_likes: Dict[str, set] = {}  # comment_id -> {user_ids}
        logger.info("CommunityService initialized")
    
    def create_post(
        self,
        user_id: str,
        title: str,
        content: str,
        post_type: PostType,
        tags: Optional[List[str]] = None
    ) -> Post:
        """Crear un nuevo post"""
        post = Post(
            id=f"post_{user_id}_{int(datetime.now().timestamp())}",
            user_id=user_id,
            title=title,
            content=content,
            post_type=post_type,
            tags=tags or [],
        )
        
        self.posts[post.id] = post
        self.comments[post.id] = []
        
        # Actualizar perfil del usuario
        profile = self._get_or_create_profile(user_id)
        profile.posts_count += 1
        
        logger.info(f"Post created: {post.id} by user {user_id}")
        return post
    
    def get_posts(
        self,
        post_type: Optional[PostType] = None,
        tags: Optional[List[str]] = None,
        limit: int = 20,
        offset: int = 0
    ) -> List[Post]:
        """Obtener posts"""
        posts = list(self.posts.values())
        
        # Filtrar por tipo
        if post_type:
            posts = [p for p in posts if p.post_type == post_type]
        
        # Filtrar por tags
        if tags:
            posts = [p for p in posts if any(tag in p.tags for tag in tags)]
        
        # Solo publicados
        posts = [p for p in posts if p.status == PostStatus.PUBLISHED]
        
        # Ordenar por fecha (más recientes primero)
        posts.sort(key=lambda x: x.created_at, reverse=True)
        
        # Paginación
        return posts[offset:offset + limit]
    
    def get_post(self, post_id: str) -> Optional[Post]:
        """Obtener un post específico"""
        post = self.posts.get(post_id)
        if post:
            post.views += 1
        return post
    
    def create_comment(
        self,
        post_id: str,
        user_id: str,
        content: str,
        parent_comment_id: Optional[str] = None
    ) -> Comment:
        """Crear comentario"""
        if post_id not in self.posts:
            raise ValueError(f"Post {post_id} not found")
        
        comment = Comment(
            id=f"comment_{post_id}_{int(datetime.now().timestamp())}",
            post_id=post_id,
            user_id=user_id,
            content=content,
            parent_comment_id=parent_comment_id
        )
        
        if post_id not in self.comments:
            self.comments[post_id] = []
        
        self.comments[post_id].append(comment)
        
        # Actualizar contador en post
        post = self.posts[post_id]
        post.comments_count += 1
        post.updated_at = datetime.now()
        
        # Actualizar perfil
        profile = self._get_or_create_profile(user_id)
        profile.comments_count += 1
        
        logger.info(f"Comment created: {comment.id} on post {post_id}")
        return comment
    
    def get_comments(self, post_id: str) -> List[Comment]:
        """Obtener comentarios de un post"""
        return self.comments.get(post_id, [])
    
    def like_post(self, post_id: str, user_id: str) -> bool:
        """Dar like a un post"""
        if post_id not in self.posts:
            return False
        
        if post_id not in self.post_likes:
            self.post_likes[post_id] = set()
        
        if user_id in self.post_likes[post_id]:
            # Unlike
            self.post_likes[post_id].remove(user_id)
            self.posts[post_id].likes -= 1
            return False
        else:
            # Like
            self.post_likes[post_id].add(user_id)
            self.posts[post_id].likes += 1
            return True
    
    def like_comment(self, comment_id: str, user_id: str) -> bool:
        """Dar like a un comentario"""
        # Buscar comentario
        comment = None
        for comments_list in self.comments.values():
            for c in comments_list:
                if c.id == comment_id:
                    comment = c
                    break
            if comment:
                break
        
        if not comment:
            return False
        
        if comment_id not in self.comment_likes:
            self.comment_likes[comment_id] = set()
        
        if user_id in self.comment_likes[comment_id]:
            # Unlike
            self.comment_likes[comment_id].remove(user_id)
            comment.likes -= 1
            return False
        else:
            # Like
            self.comment_likes[comment_id].add(user_id)
            comment.likes += 1
            return True
    
    def mark_as_helpful(self, comment_id: str, user_id: str) -> bool:
        """Marcar comentario como útil"""
        # Buscar comentario
        comment = None
        for comments_list in self.comments.values():
            for c in comments_list:
                if c.id == comment_id:
                    comment = c
                    break
            if comment:
                break
        
        if not comment:
            return False
        
        # Actualizar reputación del autor del comentario
        profile = self._get_or_create_profile(comment.user_id)
        profile.helpful_votes += 1
        profile.reputation += 5  # 5 puntos de reputación por voto útil
        
        return True
    
    def get_user_profile(self, user_id: str) -> UserProfile:
        """Obtener perfil de usuario"""
        return self._get_or_create_profile(user_id)
    
    def get_trending_posts(self, limit: int = 10) -> List[Post]:
        """Obtener posts trending"""
        posts = [
            p for p in self.posts.values()
            if p.status == PostStatus.PUBLISHED
        ]
        
        # Calcular score trending (likes + comments + views recientes)
        def trending_score(post: Post) -> float:
            return (
                post.likes * 2 +
                post.comments_count * 3 +
                post.views * 0.1
            )
        
        posts.sort(key=trending_score, reverse=True)
        return posts[:limit]
    
    def search_posts(self, query: str, limit: int = 20) -> List[Post]:
        """Buscar posts"""
        query_lower = query.lower()
        posts = [
            p for p in self.posts.values()
            if p.status == PostStatus.PUBLISHED and (
                query_lower in p.title.lower() or
                query_lower in p.content.lower() or
                any(query_lower in tag.lower() for tag in p.tags)
            )
        ]
        
        posts.sort(key=lambda x: x.created_at, reverse=True)
        return posts[:limit]
    
    def _get_or_create_profile(self, user_id: str) -> UserProfile:
        """Obtener o crear perfil de usuario"""
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = UserProfile(
                user_id=user_id,
                username=f"user_{user_id[:8]}",
            )
        return self.user_profiles[user_id]




