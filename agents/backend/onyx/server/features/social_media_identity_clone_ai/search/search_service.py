"""
Servicio de búsqueda y filtrado avanzado
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
from sqlalchemy.orm import Session
from sqlalchemy import or_, and_, func

from ..db.base import get_db_session
from ..db.models import (
    IdentityProfileModel,
    GeneratedContentModel,
    SocialProfileModel,
    ContentAnalysisModel
)

logger = logging.getLogger(__name__)


@dataclass
class SearchFilter:
    """Filtros de búsqueda"""
    query: Optional[str] = None
    platform: Optional[str] = None
    min_videos: Optional[int] = None
    max_videos: Optional[int] = None
    min_posts: Optional[int] = None
    max_posts: Optional[int] = None
    created_after: Optional[datetime] = None
    created_before: Optional[datetime] = None
    topics: Optional[List[str]] = None
    tone: Optional[str] = None
    tags: Optional[List[str]] = None
    limit: int = 50
    offset: int = 0


@dataclass
class SearchResult:
    """Resultado de búsqueda"""
    identity_id: str
    username: str
    display_name: Optional[str]
    score: float
    match_reason: str
    metadata: Dict[str, Any]


class SearchService:
    """Servicio de búsqueda avanzada"""
    
    def search_identities(self, filters: SearchFilter) -> Dict[str, Any]:
        """
        Busca identidades con filtros avanzados
        
        Args:
            filters: Filtros de búsqueda
            
        Returns:
            Resultados de búsqueda
        """
        with get_db_session() as db:
            query = db.query(IdentityProfileModel)
            
            # Filtro por texto (username, display_name, bio)
            if filters.query:
                search_term = f"%{filters.query.lower()}%"
                query = query.filter(
                    or_(
                        func.lower(IdentityProfileModel.username).like(search_term),
                        func.lower(IdentityProfileModel.display_name).like(search_term),
                        func.lower(IdentityProfileModel.bio).like(search_term)
                    )
                )
            
            # Filtro por plataforma (buscar en social_profiles)
            if filters.platform:
                query = query.join(SocialProfileModel).filter(
                    SocialProfileModel.platform == filters.platform
                )
            
            # Filtro por número de videos
            if filters.min_videos is not None:
                query = query.filter(IdentityProfileModel.total_videos >= filters.min_videos)
            if filters.max_videos is not None:
                query = query.filter(IdentityProfileModel.total_videos <= filters.max_videos)
            
            # Filtro por número de posts
            if filters.min_posts is not None:
                query = query.filter(IdentityProfileModel.total_posts >= filters.min_posts)
            if filters.max_posts is not None:
                query = query.filter(IdentityProfileModel.total_posts <= filters.max_posts)
            
            # Filtro por fecha
            if filters.created_after:
                query = query.filter(IdentityProfileModel.created_at >= filters.created_after)
            if filters.created_before:
                query = query.filter(IdentityProfileModel.created_at <= filters.created_before)
            
            # Filtro por topics (en content_analysis)
            if filters.topics:
                query = query.join(ContentAnalysisModel).filter(
                    func.json_extract(ContentAnalysisModel.topics, '$[*]').in_(filters.topics)
                )
            
            # Filtro por tone
            if filters.tone:
                query = query.join(ContentAnalysisModel).filter(
                    ContentAnalysisModel.tone == filters.tone
                )
            
            # Contar total
            total = query.count()
            
            # Aplicar paginación
            results = query.order_by(
                IdentityProfileModel.created_at.desc()
            ).offset(filters.offset).limit(filters.limit).all()
            
            # Construir resultados
            search_results = []
            for identity in results:
                score = self._calculate_score(identity, filters)
                match_reason = self._get_match_reason(identity, filters)
                
                search_results.append(SearchResult(
                    identity_id=identity.id,
                    username=identity.username,
                    display_name=identity.display_name,
                    score=score,
                    match_reason=match_reason,
                    metadata={
                        "total_videos": identity.total_videos,
                        "total_posts": identity.total_posts,
                        "created_at": identity.created_at.isoformat()
                    }
                ))
            
            return {
                "results": [
                    {
                        "identity_id": r.identity_id,
                        "username": r.username,
                        "display_name": r.display_name,
                        "score": r.score,
                        "match_reason": r.match_reason,
                        "metadata": r.metadata
                    }
                    for r in search_results
                ],
                "total": total,
                "offset": filters.offset,
                "limit": filters.limit,
                "has_more": (filters.offset + filters.limit) < total
            }
    
    def search_content(
        self,
        query: Optional[str] = None,
        platform: Optional[str] = None,
        identity_id: Optional[str] = None,
        content_type: Optional[str] = None,
        limit: int = 50,
        offset: int = 0
    ) -> Dict[str, Any]:
        """Busca contenido generado"""
        with get_db_session() as db:
            search_query = db.query(GeneratedContentModel)
            
            if query:
                search_term = f"%{query.lower()}%"
                search_query = search_query.filter(
                    or_(
                        func.lower(GeneratedContentModel.content).like(search_term),
                        func.lower(GeneratedContentModel.title).like(search_term)
                    )
                )
            
            if platform:
                search_query = search_query.filter(GeneratedContentModel.platform == platform)
            
            if identity_id:
                search_query = search_query.filter(
                    GeneratedContentModel.identity_profile_id == identity_id
                )
            
            if content_type:
                search_query = search_query.filter(
                    GeneratedContentModel.content_type == content_type
                )
            
            total = search_query.count()
            
            results = search_query.order_by(
                GeneratedContentModel.generated_at.desc()
            ).offset(offset).limit(limit).all()
            
            return {
                "results": [
                    {
                        "content_id": r.id,
                        "identity_id": r.identity_profile_id,
                        "platform": r.platform,
                        "content_type": r.content_type,
                        "title": r.title,
                        "content": r.content[:200] + "..." if len(r.content) > 200 else r.content,
                        "hashtags": r.hashtags,
                        "generated_at": r.generated_at.isoformat()
                    }
                    for r in results
                ],
                "total": total,
                "offset": offset,
                "limit": limit,
                "has_more": (offset + limit) < total
            }
    
    def _calculate_score(self, identity: IdentityProfileModel, filters: SearchFilter) -> float:
        """Calcula score de relevancia"""
        score = 1.0
        
        # Boost por match de query
        if filters.query:
            query_lower = filters.query.lower()
            if query_lower in identity.username.lower():
                score += 2.0
            if identity.display_name and query_lower in identity.display_name.lower():
                score += 1.5
            if identity.bio and query_lower in identity.bio.lower():
                score += 1.0
        
        # Boost por cantidad de contenido
        score += (identity.total_videos + identity.total_posts) * 0.01
        
        return score
    
    def _get_match_reason(self, identity: IdentityProfileModel, filters: SearchFilter) -> str:
        """Obtiene razón del match"""
        reasons = []
        
        if filters.query and filters.query.lower() in identity.username.lower():
            reasons.append("username match")
        
        if filters.platform:
            reasons.append(f"platform: {filters.platform}")
        
        if filters.topics:
            reasons.append(f"topics: {', '.join(filters.topics)}")
        
        return ", ".join(reasons) if reasons else "general match"




