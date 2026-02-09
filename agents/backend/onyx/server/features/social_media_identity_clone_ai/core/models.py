"""
Modelos de datos para Social Media Identity Clone AI
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from enum import Enum
from pydantic import BaseModel, Field


class Platform(str, Enum):
    """Plataformas de redes sociales soportadas"""
    TIKTOK = "tiktok"
    INSTAGRAM = "instagram"
    YOUTUBE = "youtube"


class ContentType(str, Enum):
    """Tipos de contenido"""
    VIDEO = "video"
    POST = "post"
    STORY = "story"
    REEL = "reel"
    COMMENT = "comment"


class VideoContent(BaseModel):
    """Contenido de video extraído"""
    video_id: str
    url: str
    title: Optional[str] = None
    description: Optional[str] = None
    transcript: Optional[str] = None
    duration: Optional[float] = None  # en segundos
    views: Optional[int] = None
    likes: Optional[int] = None
    comments: Optional[int] = None
    created_at: Optional[datetime] = None
    hashtags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class PostContent(BaseModel):
    """Contenido de post extraído"""
    post_id: str
    url: str
    caption: Optional[str] = None
    image_urls: List[str] = Field(default_factory=list)
    likes: Optional[int] = None
    comments: Optional[int] = None
    created_at: Optional[datetime] = None
    hashtags: List[str] = Field(default_factory=list)
    mentions: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CommentContent(BaseModel):
    """Contenido de comentario extraído"""
    comment_id: str
    text: str
    author: Optional[str] = None
    likes: Optional[int] = None
    created_at: Optional[datetime] = None
    replies: List["CommentContent"] = Field(default_factory=list)


class SocialProfile(BaseModel):
    """Perfil completo de una red social"""
    platform: Platform
    username: str
    display_name: Optional[str] = None
    bio: Optional[str] = None
    profile_image_url: Optional[str] = None
    followers_count: Optional[int] = None
    following_count: Optional[int] = None
    posts_count: Optional[int] = None
    videos: List[VideoContent] = Field(default_factory=list)
    posts: List[PostContent] = Field(default_factory=list)
    comments: List[CommentContent] = Field(default_factory=list)
    extracted_at: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ContentAnalysis(BaseModel):
    """Análisis de contenido extraído"""
    topics: List[str] = Field(default_factory=list)
    themes: List[str] = Field(default_factory=list)
    tone: Optional[str] = None  # formal, casual, humorous, etc.
    personality_traits: List[str] = Field(default_factory=list)
    communication_style: Optional[str] = None
    common_phrases: List[str] = Field(default_factory=list)
    values: List[str] = Field(default_factory=list)
    interests: List[str] = Field(default_factory=list)
    language_patterns: Dict[str, Any] = Field(default_factory=dict)
    sentiment_analysis: Dict[str, float] = Field(default_factory=dict)


class IdentityProfile(BaseModel):
    """Perfil de identidad clonada completo"""
    profile_id: str
    username: str
    display_name: Optional[str] = None
    bio: Optional[str] = None
    
    # Perfiles de cada plataforma
    tiktok_profile: Optional[SocialProfile] = None
    instagram_profile: Optional[SocialProfile] = None
    youtube_profile: Optional[SocialProfile] = None
    
    # Análisis consolidado
    content_analysis: ContentAnalysis
    
    # Base de conocimiento
    knowledge_base: Dict[str, Any] = Field(default_factory=dict)
    
    # Estadísticas
    total_videos: int = 0
    total_posts: int = 0
    total_comments: int = 0
    
    # Metadatos
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class GeneratedContent(BaseModel):
    """Contenido generado basado en identidad"""
    content_id: str
    identity_profile_id: str
    platform: Platform
    content_type: ContentType
    content: str
    title: Optional[str] = None
    hashtags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    generated_at: datetime = Field(default_factory=datetime.now)
    confidence_score: Optional[float] = None  # 0.0 - 1.0




