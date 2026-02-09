"""
Social Sharing System
=====================
Sistema de compartición en redes sociales
"""

import json
import time
import hashlib
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum


class SocialPlatform(Enum):
    """Plataformas sociales"""
    TWITTER = "twitter"
    FACEBOOK = "facebook"
    INSTAGRAM = "instagram"
    LINKEDIN = "linkedin"
    PINTEREST = "pinterest"
    REDDIT = "reddit"
    DISCORD = "discord"
    TELEGRAM = "telegram"


@dataclass
class SharePost:
    """Post compartido"""
    id: str
    result_id: str
    platform: SocialPlatform
    content: Dict[str, Any]
    shared_at: float
    share_count: int = 0
    likes: int = 0
    comments: int = 0
    views: int = 0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class SocialSharing:
    """
    Sistema de compartición social
    """
    
    def __init__(self):
        self.shares: Dict[str, List[SharePost]] = {}  # result_id -> posts
        self.platform_configs: Dict[SocialPlatform, Dict[str, Any]] = {}
        self.share_templates: Dict[SocialPlatform, Dict[str, str]] = {}
        self._init_templates()
    
    def _init_templates(self):
        """Inicializar plantillas de compartición"""
        self.share_templates = {
            SocialPlatform.TWITTER: {
                'text': 'Check out this amazing character clothing change! 🎨✨',
                'hashtags': ['#AI', '#CharacterDesign', '#FashionAI']
            },
            SocialPlatform.FACEBOOK: {
                'text': 'Look at this incredible AI-powered character clothing transformation!',
                'hashtags': ['#AI', '#CharacterDesign']
            },
            SocialPlatform.INSTAGRAM: {
                'text': 'Amazing AI character clothing change! ✨',
                'hashtags': ['#AI', '#CharacterDesign', '#FashionAI', '#AIArt']
            },
            SocialPlatform.LINKEDIN: {
                'text': 'Exploring AI-powered character design with advanced clothing transformation technology.',
                'hashtags': ['#AI', '#MachineLearning', '#CharacterDesign']
            },
            SocialPlatform.PINTEREST: {
                'text': 'AI Character Clothing Transformation',
                'hashtags': ['#AI', '#CharacterDesign', '#Fashion']
            },
            SocialPlatform.REDDIT: {
                'text': 'AI-powered character clothing changer - before and after',
                'hashtags': []
            },
            SocialPlatform.DISCORD: {
                'text': 'Check out this character clothing change!',
                'hashtags': []
            },
            SocialPlatform.TELEGRAM: {
                'text': 'Amazing AI character clothing transformation!',
                'hashtags': []
            }
        }
    
    def share(
        self,
        result_id: str,
        platform: SocialPlatform,
        custom_text: Optional[str] = None,
        image_url: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> SharePost:
        """
        Compartir resultado en plataforma social
        
        Args:
            result_id: ID del resultado
            platform: Plataforma social
            custom_text: Texto personalizado
            image_url: URL de la imagen
            metadata: Metadata adicional
        """
        post_id = hashlib.sha256(
            f"{result_id}{platform.value}{time.time()}".encode()
        ).hexdigest()[:16]
        
        template = self.share_templates.get(platform, {})
        text = custom_text or template.get('text', '')
        hashtags = template.get('hashtags', [])
        
        if hashtags:
            text += ' ' + ' '.join(hashtags)
        
        content = {
            'text': text,
            'image_url': image_url,
            'hashtags': hashtags,
            'result_id': result_id
        }
        
        post = SharePost(
            id=post_id,
            result_id=result_id,
            platform=platform,
            content=content,
            shared_at=time.time(),
            metadata=metadata or {}
        )
        
        if result_id not in self.shares:
            self.shares[result_id] = []
        
        self.shares[result_id].append(post)
        
        return post
    
    def share_to_multiple(
        self,
        result_id: str,
        platforms: List[SocialPlatform],
        custom_text: Optional[str] = None,
        image_url: Optional[str] = None
    ) -> List[SharePost]:
        """Compartir a múltiples plataformas"""
        posts = []
        for platform in platforms:
            post = self.share(
                result_id=result_id,
                platform=platform,
                custom_text=custom_text,
                image_url=image_url
            )
            posts.append(post)
        
        return posts
    
    def get_shares(self, result_id: str) -> List[SharePost]:
        """Obtener todos los shares de un resultado"""
        return self.shares.get(result_id, [])
    
    def get_share_statistics(self, result_id: str) -> Dict[str, Any]:
        """Obtener estadísticas de compartición"""
        posts = self.get_shares(result_id)
        
        if not posts:
            return {
                'total_shares': 0,
                'platforms': {},
                'total_engagement': 0
            }
        
        platform_stats = {}
        total_engagement = 0
        
        for post in posts:
            platform = post.platform.value
            if platform not in platform_stats:
                platform_stats[platform] = {
                    'count': 0,
                    'likes': 0,
                    'comments': 0,
                    'views': 0
                }
            
            platform_stats[platform]['count'] += 1
            platform_stats[platform]['likes'] += post.likes
            platform_stats[platform]['comments'] += post.comments
            platform_stats[platform]['views'] += post.views
            
            total_engagement += post.likes + post.comments + post.views
        
        return {
            'total_shares': len(posts),
            'platforms': platform_stats,
            'total_engagement': total_engagement,
            'average_engagement': total_engagement / len(posts) if posts else 0
        }
    
    def update_engagement(
        self,
        post_id: str,
        likes: Optional[int] = None,
        comments: Optional[int] = None,
        views: Optional[int] = None
    ) -> bool:
        """Actualizar engagement de un post"""
        for posts in self.shares.values():
            for post in posts:
                if post.id == post_id:
                    if likes is not None:
                        post.likes = likes
                    if comments is not None:
                        post.comments = comments
                    if views is not None:
                        post.views = views
                    return True
        
        return False
    
    def get_popular_results(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Obtener resultados más populares"""
        result_stats = []
        
        for result_id, posts in self.shares.items():
            stats = self.get_share_statistics(result_id)
            result_stats.append({
                'result_id': result_id,
                'total_shares': stats['total_shares'],
                'total_engagement': stats['total_engagement'],
                'platforms': list(stats['platforms'].keys())
            })
        
        result_stats.sort(key=lambda x: x['total_engagement'], reverse=True)
        return result_stats[:limit]
    
    def configure_platform(
        self,
        platform: SocialPlatform,
        config: Dict[str, Any]
    ):
        """Configurar plataforma"""
        self.platform_configs[platform] = config
    
    def set_template(
        self,
        platform: SocialPlatform,
        template: Dict[str, str]
    ):
        """Establecer plantilla personalizada"""
        self.share_templates[platform] = template
    
    def generate_share_url(
        self,
        result_id: str,
        platform: SocialPlatform
    ) -> str:
        """Generar URL de compartición"""
        base_urls = {
            SocialPlatform.TWITTER: 'https://twitter.com/intent/tweet',
            SocialPlatform.FACEBOOK: 'https://www.facebook.com/sharer/sharer.php',
            SocialPlatform.LINKEDIN: 'https://www.linkedin.com/sharing/share-offsite',
            SocialPlatform.PINTEREST: 'https://pinterest.com/pin/create/button',
            SocialPlatform.REDDIT: 'https://reddit.com/submit'
        }
        
        base_url = base_urls.get(platform)
        if not base_url:
            return ''
        
        # En implementación real, construir URL con parámetros
        return f"{base_url}?url=...&text=..."
    
    def get_platform_insights(self) -> Dict[str, Any]:
        """Obtener insights por plataforma"""
        platform_data = {}
        
        for posts in self.shares.values():
            for post in posts:
                platform = post.platform.value
                if platform not in platform_data:
                    platform_data[platform] = {
                        'total_posts': 0,
                        'total_likes': 0,
                        'total_comments': 0,
                        'total_views': 0
                    }
                
                platform_data[platform]['total_posts'] += 1
                platform_data[platform]['total_likes'] += post.likes
                platform_data[platform]['total_comments'] += post.comments
                platform_data[platform]['total_views'] += post.views
        
        return platform_data


# Instancia global
social_sharing = SocialSharing()

