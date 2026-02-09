"""
Servicio de almacenamiento para persistir identidades y contenido
"""

import logging
import uuid
from typing import Optional, List
from datetime import datetime

from sqlalchemy.orm import Session

from ..core.models import IdentityProfile, SocialProfile, GeneratedContent
from ..db.models import (
    IdentityProfileModel,
    SocialProfileModel,
    GeneratedContentModel,
    ContentAnalysisModel,
)
from ..db.base import get_db_session

logger = logging.getLogger(__name__)


class StorageService:
    """Servicio para almacenar y recuperar identidades"""
    
    def save_identity(self, identity: IdentityProfile) -> str:
        """
        Guarda perfil de identidad en base de datos
        
        Args:
            identity: Perfil de identidad a guardar
            
        Returns:
            ID del perfil guardado
        """
        with db_transaction(log_operation="save_identity") as db:
            # Upsert identity using helper
            upsert_model(
                db,
                IdentityProfileModel,
                identifier={"id": identity.profile_id},
                update_data={
                    "username": identity.username,
                    "display_name": identity.display_name,
                    "bio": identity.bio,
                    "total_videos": identity.total_videos,
                    "total_posts": identity.total_posts,
                    "total_comments": identity.total_comments,
                    "knowledge_base": identity.knowledge_base,
                    "metadata": identity.metadata
                }
            )
            
            # Upsert content analysis using helper
            if identity.content_analysis:
                upsert_model(
                    db,
                    ContentAnalysisModel,
                    identifier={"identity_profile_id": identity.profile_id},
                    update_data={
                        "topics": identity.content_analysis.topics,
                        "themes": identity.content_analysis.themes,
                        "tone": identity.content_analysis.tone,
                        "personality_traits": identity.content_analysis.personality_traits,
                        "communication_style": identity.content_analysis.communication_style,
                        "common_phrases": identity.content_analysis.common_phrases,
                        "values": identity.content_analysis.values,
                        "interests": identity.content_analysis.interests,
                        "language_patterns": identity.content_analysis.language_patterns,
                        "sentiment_analysis": identity.content_analysis.sentiment_analysis
                    },
                    create_data={"id": generate_id("content_analysis")}
                )
            
            # Guardar perfiles sociales
            for social_profile in [identity.tiktok_profile, identity.instagram_profile, identity.youtube_profile]:
                if social_profile:
                    self._save_social_profile(db, identity.profile_id, social_profile)
            
            return identity.profile_id
    
    def _save_social_profile(self, db: Session, identity_id: str, profile: SocialProfile):
        """Guarda perfil social en base de datos"""
        # Serialize content using helper
        videos_data = serialize_models(profile.videos) if profile.videos else None
        posts_data = serialize_models(profile.posts) if profile.posts else None
        comments_data = serialize_models(profile.comments) if profile.comments else None
        
        # Upsert using helper
        upsert_model(
            db,
            SocialProfileModel,
            identifier={
                "identity_profile_id": identity_id,
                "platform": profile.platform.value
            },
            update_data={
                "username": profile.username,
                "display_name": profile.display_name,
                "bio": profile.bio,
                "profile_image_url": profile.profile_image_url,
                "followers_count": profile.followers_count,
                "following_count": profile.following_count,
                "posts_count": profile.posts_count,
                "videos_data": videos_data,
                "posts_data": posts_data,
                "comments_data": comments_data,
                "metadata": profile.metadata
            },
            create_data={
                "id": generate_id("social_profile"),
                "extracted_at": profile.extracted_at
            }
        )
    
    def get_identity(self, identity_id: str) -> Optional[IdentityProfile]:
        """
        Obtiene perfil de identidad de base de datos
        
        Args:
            identity_id: ID del perfil
            
        Returns:
            IdentityProfile o None si no existe
        """
        with db_transaction(auto_commit=False, log_operation="get_identity") as db:
            db_model = query_one(db, IdentityProfileModel, {"id": identity_id})
            
            if not db_model:
                return None
            
            # Reconstruir IdentityProfile desde modelo de DB
            # (simplificado, en producción necesitarías reconstruir todos los objetos)
            from ..core.models import ContentAnalysis
            
            analysis = query_one(db, ContentAnalysisModel, {"identity_profile_id": identity_id})
            
            content_analysis = ContentAnalysis(
                topics=analysis.topics if analysis else [],
                themes=analysis.themes if analysis else [],
                tone=analysis.tone if analysis else None,
                personality_traits=analysis.personality_traits if analysis else [],
                communication_style=analysis.communication_style if analysis else None,
                common_phrases=analysis.common_phrases if analysis else [],
                values=analysis.values if analysis else [],
                interests=analysis.interests if analysis else [],
                language_patterns=analysis.language_patterns if analysis else {},
                sentiment_analysis=analysis.sentiment_analysis if analysis else {}
            )
            
            identity = IdentityProfile(
                profile_id=db_model.id,
                username=db_model.username,
                display_name=db_model.display_name,
                bio=db_model.bio,
                content_analysis=content_analysis,
                knowledge_base=db_model.knowledge_base or {},
                total_videos=db_model.total_videos,
                total_posts=db_model.total_posts,
                total_comments=db_model.total_comments,
                created_at=db_model.created_at,
                updated_at=db_model.updated_at,
                metadata=db_model.metadata or {}
            )
            
            return identity
    
    def save_generated_content(self, content: GeneratedContent) -> str:
        """
        Guarda contenido generado en base de datos
        
        Args:
            content: Contenido generado
            
        Returns:
            ID del contenido guardado
        """
        with db_transaction(log_operation="save_generated_content") as db:
            upsert_model(
                db,
                GeneratedContentModel,
                identifier={"id": content.content_id},
                update_data={
                    "identity_profile_id": content.identity_profile_id,
                    "platform": content.platform.value,
                    "content_type": content.content_type.value,
                    "content": content.content,
                    "title": content.title,
                    "hashtags": content.hashtags,
                    "confidence_score": content.confidence_score,
                    "metadata": content.metadata,
                    "generated_at": content.generated_at
                }
            )
            return content.content_id
    
    def get_generated_content(self, identity_id: str, limit: int = 10) -> List[GeneratedContent]:
        """
        Obtiene contenido generado para una identidad
        
        Args:
            identity_id: ID de la identidad
            limit: Límite de resultados
            
        Returns:
            Lista de contenido generado
        """
        from ..db.query_helpers import query_many
        
        with db_transaction(auto_commit=False, log_operation="get_generated_content") as db:
            db_models = query_many(
                db,
                GeneratedContentModel,
                filters={"identity_profile_id": identity_id},
                order_by="generated_at",
                limit=limit
            )
            
            from ..core.models import Platform, ContentType
            
            results = []
            for db_model in db_models:
                content = GeneratedContent(
                    content_id=db_model.id,
                    identity_profile_id=db_model.identity_profile_id,
                    platform=Platform(db_model.platform),
                    content_type=ContentType(db_model.content_type),
                    content=db_model.content,
                    title=db_model.title,
                    hashtags=db_model.hashtags or [],
                    confidence_score=db_model.confidence_score,
                    metadata=db_model.metadata or {},
                    generated_at=db_model.generated_at
                )
                results.append(content)
            
            return results




