"""
PDF Variantes Database Repository
Repositorio de base de datos para el sistema PDF Variantes
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from uuid import UUID
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete, func, and_, or_
from sqlalchemy.orm import selectinload

from .models import (
    User, Document, Variant, Topic, BrainstormIdea, 
    Collaboration, Annotation, Export, Analytics
)

logger = logging.getLogger(__name__)

class BaseRepository:
    """Repositorio base"""
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def commit(self):
        """Confirmar transacción"""
        try:
            await self.session.commit()
        except Exception as e:
            await self.session.rollback()
            raise e
    
    async def rollback(self):
        """Revertir transacción"""
        await self.session.rollback()

class UserRepository(BaseRepository):
    """Repositorio de usuarios"""
    
    async def create(self, user_data: Dict[str, Any]) -> User:
        """Crear usuario"""
        try:
            user = User(**user_data)
            self.session.add(user)
            await self.commit()
            return user
        except Exception as e:
            await self.rollback()
            raise e
    
    async def get_by_id(self, user_id: UUID) -> Optional[User]:
        """Obtener usuario por ID"""
        try:
            result = await self.session.execute(
                select(User).where(User.id == user_id)
            )
            return result.scalar_one_or_none()
        except Exception as e:
            logger.error(f"Error getting user by ID: {e}")
            return None
    
    async def get_by_username(self, username: str) -> Optional[User]:
        """Obtener usuario por nombre de usuario"""
        try:
            result = await self.session.execute(
                select(User).where(User.username == username)
            )
            return result.scalar_one_or_none()
        except Exception as e:
            logger.error(f"Error getting user by username: {e}")
            return None
    
    async def get_by_email(self, email: str) -> Optional[User]:
        """Obtener usuario por email"""
        try:
            result = await self.session.execute(
                select(User).where(User.email == email)
            )
            return result.scalar_one_or_none()
        except Exception as e:
            logger.error(f"Error getting user by email: {e}")
            return None
    
    async def update(self, user_id: UUID, update_data: Dict[str, Any]) -> Optional[User]:
        """Actualizar usuario"""
        try:
            await self.session.execute(
                update(User)
                .where(User.id == user_id)
                .values(**update_data)
            )
            await self.commit()
            return await self.get_by_id(user_id)
        except Exception as e:
            await self.rollback()
            raise e
    
    async def delete(self, user_id: UUID) -> bool:
        """Eliminar usuario"""
        try:
            await self.session.execute(
                delete(User).where(User.id == user_id)
            )
            await self.commit()
            return True
        except Exception as e:
            await self.rollback()
            raise e
    
    async def list_users(self, skip: int = 0, limit: int = 100) -> List[User]:
        """Listar usuarios"""
        try:
            result = await self.session.execute(
                select(User)
                .offset(skip)
                .limit(limit)
                .order_by(User.created_at.desc())
            )
            return result.scalars().all()
        except Exception as e:
            logger.error(f"Error listing users: {e}")
            return []

class DocumentRepository(BaseRepository):
    """Repositorio de documentos"""
    
    async def create(self, document_data: Dict[str, Any]) -> Document:
        """Crear documento"""
        try:
            document = Document(**document_data)
            self.session.add(document)
            await self.commit()
            return document
        except Exception as e:
            await self.rollback()
            raise e
    
    async def get_by_id(self, document_id: UUID) -> Optional[Document]:
        """Obtener documento por ID"""
        try:
            result = await self.session.execute(
                select(Document)
                .where(Document.id == document_id)
                .options(selectinload(Document.owner))
            )
            return result.scalar_one_or_none()
        except Exception as e:
            logger.error(f"Error getting document by ID: {e}")
            return None
    
    async def get_by_hash(self, content_hash: str) -> Optional[Document]:
        """Obtener documento por hash"""
        try:
            result = await self.session.execute(
                select(Document).where(Document.content_hash == content_hash)
            )
            return result.scalar_one_or_none()
        except Exception as e:
            logger.error(f"Error getting document by hash: {e}")
            return None
    
    async def update(self, document_id: UUID, update_data: Dict[str, Any]) -> Optional[Document]:
        """Actualizar documento"""
        try:
            await self.session.execute(
                update(Document)
                .where(Document.id == document_id)
                .values(**update_data)
            )
            await self.commit()
            return await self.get_by_id(document_id)
        except Exception as e:
            await self.rollback()
            raise e
    
    async def delete(self, document_id: UUID) -> bool:
        """Eliminar documento"""
        try:
            await self.session.execute(
                delete(Document).where(Document.id == document_id)
            )
            await self.commit()
            return True
        except Exception as e:
            await self.rollback()
            raise e
    
    async def list_by_user(self, user_id: UUID, skip: int = 0, limit: int = 100) -> List[Document]:
        """Listar documentos por usuario"""
        try:
            result = await self.session.execute(
                select(Document)
                .where(Document.owner_id == user_id)
                .offset(skip)
                .limit(limit)
                .order_by(Document.created_at.desc())
            )
            return result.scalars().all()
        except Exception as e:
            logger.error(f"Error listing documents by user: {e}")
            return []
    
    async def search(self, query: str, skip: int = 0, limit: int = 100) -> List[Document]:
        """Buscar documentos"""
        try:
            result = await self.session.execute(
                select(Document)
                .where(
                    or_(
                        Document.title.ilike(f"%{query}%"),
                        Document.filename.ilike(f"%{query}%")
                    )
                )
                .offset(skip)
                .limit(limit)
                .order_by(Document.created_at.desc())
            )
            return result.scalars().all()
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            return []

class VariantRepository(BaseRepository):
    """Repositorio de variantes"""
    
    async def create(self, variant_data: Dict[str, Any]) -> Variant:
        """Crear variante"""
        try:
            variant = Variant(**variant_data)
            self.session.add(variant)
            await self.commit()
            return variant
        except Exception as e:
            await self.rollback()
            raise e
    
    async def get_by_id(self, variant_id: UUID) -> Optional[Variant]:
        """Obtener variante por ID"""
        try:
            result = await self.session.execute(
                select(Variant)
                .where(Variant.id == variant_id)
                .options(selectinload(Variant.document))
            )
            return result.scalar_one_or_none()
        except Exception as e:
            logger.error(f"Error getting variant by ID: {e}")
            return None
    
    async def list_by_document(self, document_id: UUID, skip: int = 0, limit: int = 100) -> List[Variant]:
        """Listar variantes por documento"""
        try:
            result = await self.session.execute(
                select(Variant)
                .where(Variant.document_id == document_id)
                .offset(skip)
                .limit(limit)
                .order_by(Variant.created_at.desc())
            )
            return result.scalars().all()
        except Exception as e:
            logger.error(f"Error listing variants by document: {e}")
            return []
    
    async def update(self, variant_id: UUID, update_data: Dict[str, Any]) -> Optional[Variant]:
        """Actualizar variante"""
        try:
            await self.session.execute(
                update(Variant)
                .where(Variant.id == variant_id)
                .values(**update_data)
            )
            await self.commit()
            return await self.get_by_id(variant_id)
        except Exception as e:
            await self.rollback()
            raise e
    
    async def delete(self, variant_id: UUID) -> bool:
        """Eliminar variante"""
        try:
            await self.session.execute(
                delete(Variant).where(Variant.id == variant_id)
            )
            await self.commit()
            return True
        except Exception as e:
            await self.rollback()
            raise e

class TopicRepository(BaseRepository):
    """Repositorio de temas"""
    
    async def create(self, topic_data: Dict[str, Any]) -> Topic:
        """Crear tema"""
        try:
            topic = Topic(**topic_data)
            self.session.add(topic)
            await self.commit()
            return topic
        except Exception as e:
            await self.rollback()
            raise e
    
    async def list_by_document(self, document_id: UUID, skip: int = 0, limit: int = 100) -> List[Topic]:
        """Listar temas por documento"""
        try:
            result = await self.session.execute(
                select(Topic)
                .where(Topic.document_id == document_id)
                .offset(skip)
                .limit(limit)
                .order_by(Topic.relevance_score.desc())
            )
            return result.scalars().all()
        except Exception as e:
            logger.error(f"Error listing topics by document: {e}")
            return []
    
    async def get_top_topics(self, document_id: UUID, limit: int = 10) -> List[Topic]:
        """Obtener temas principales"""
        try:
            result = await self.session.execute(
                select(Topic)
                .where(Topic.document_id == document_id)
                .order_by(Topic.relevance_score.desc())
                .limit(limit)
            )
            return result.scalars().all()
        except Exception as e:
            logger.error(f"Error getting top topics: {e}")
            return []

class BrainstormRepository(BaseRepository):
    """Repositorio de ideas de brainstorming"""
    
    async def create(self, idea_data: Dict[str, Any]) -> BrainstormIdea:
        """Crear idea"""
        try:
            idea = BrainstormIdea(**idea_data)
            self.session.add(idea)
            await self.commit()
            return idea
        except Exception as e:
            await self.rollback()
            raise e
    
    async def list_by_document(self, document_id: UUID, skip: int = 0, limit: int = 100) -> List[BrainstormIdea]:
        """Listar ideas por documento"""
        try:
            result = await self.session.execute(
                select(BrainstormIdea)
                .where(BrainstormIdea.document_id == document_id)
                .offset(skip)
                .limit(limit)
                .order_by(BrainstormIdea.priority_score.desc())
            )
            return result.scalars().all()
        except Exception as e:
            logger.error(f"Error listing brainstorm ideas by document: {e}")
            return []

class CollaborationRepository(BaseRepository):
    """Repositorio de colaboraciones"""
    
    async def create(self, collaboration_data: Dict[str, Any]) -> Collaboration:
        """Crear colaboración"""
        try:
            collaboration = Collaboration(**collaboration_data)
            self.session.add(collaboration)
            await self.commit()
            return collaboration
        except Exception as e:
            await self.rollback()
            raise e
    
    async def list_by_document(self, document_id: UUID) -> List[Collaboration]:
        """Listar colaboraciones por documento"""
        try:
            result = await self.session.execute(
                select(Collaboration)
                .where(Collaboration.document_id == document_id)
                .options(selectinload(Collaboration.user))
            )
            return result.scalars().all()
        except Exception as e:
            logger.error(f"Error listing collaborations by document: {e}")
            return []
    
    async def get_user_permissions(self, document_id: UUID, user_id: UUID) -> Optional[Collaboration]:
        """Obtener permisos de usuario"""
        try:
            result = await self.session.execute(
                select(Collaboration)
                .where(
                    and_(
                        Collaboration.document_id == document_id,
                        Collaboration.user_id == user_id,
                        Collaboration.is_active == True
                    )
                )
            )
            return result.scalar_one_or_none()
        except Exception as e:
            logger.error(f"Error getting user permissions: {e}")
            return None

class AnalyticsRepository(BaseRepository):
    """Repositorio de analytics"""
    
    async def create(self, analytics_data: Dict[str, Any]) -> Analytics:
        """Crear entrada de analytics"""
        try:
            analytics = Analytics(**analytics_data)
            self.session.add(analytics)
            await self.commit()
            return analytics
        except Exception as e:
            await self.rollback()
            raise e
    
    async def get_document_stats(self, document_id: UUID) -> Dict[str, Any]:
        """Obtener estadísticas de documento"""
        try:
            # Contar eventos por tipo
            result = await self.session.execute(
                select(
                    Analytics.event_type,
                    func.count(Analytics.id).label('count')
                )
                .where(Analytics.document_id == document_id)
                .group_by(Analytics.event_type)
            )
            
            stats = {}
            for row in result:
                stats[row.event_type] = row.count
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting document stats: {e}")
            return {}
    
    async def get_user_activity(self, user_id: UUID, days: int = 30) -> List[Analytics]:
        """Obtener actividad de usuario"""
        try:
            from datetime import timedelta
            start_date = datetime.utcnow() - timedelta(days=days)
            
            result = await self.session.execute(
                select(Analytics)
                .where(
                    and_(
                        Analytics.user_id == user_id,
                        Analytics.timestamp >= start_date
                    )
                )
                .order_by(Analytics.timestamp.desc())
            )
            return result.scalars().all()
            
        except Exception as e:
            logger.error(f"Error getting user activity: {e}")
            return []

# Factory functions
def create_user_repository(session: AsyncSession) -> UserRepository:
    """Crear repositorio de usuarios"""
    return UserRepository(session)

def create_document_repository(session: AsyncSession) -> DocumentRepository:
    """Crear repositorio de documentos"""
    return DocumentRepository(session)

def create_variant_repository(session: AsyncSession) -> VariantRepository:
    """Crear repositorio de variantes"""
    return VariantRepository(session)

def create_topic_repository(session: AsyncSession) -> TopicRepository:
    """Crear repositorio de temas"""
    return TopicRepository(session)

def create_brainstorm_repository(session: AsyncSession) -> BrainstormRepository:
    """Crear repositorio de brainstorming"""
    return BrainstormRepository(session)

def create_collaboration_repository(session: AsyncSession) -> CollaborationRepository:
    """Crear repositorio de colaboraciones"""
    return CollaborationRepository(session)

def create_analytics_repository(session: AsyncSession) -> AnalyticsRepository:
    """Crear repositorio de analytics"""
    return AnalyticsRepository(session)
