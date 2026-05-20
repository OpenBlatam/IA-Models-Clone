"""
Rating Service
=============

Servicio principal para gestión de ratings.
"""

from typing import Optional, List
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession

from ...core.base.service_base import BaseService
from ...database.models import Manual, ManualRating
from .rating_repository import RatingRepository
from ..notification.notification_service import NotificationService


class RatingService(BaseService):
    """Servicio principal para gestión de ratings."""
    
    def __init__(self, db: AsyncSession):
        """
        Inicializar servicio.
        
        Args:
            db: Sesión de base de datos
        """
        super().__init__(logger_name=__name__)
        self.db = db
        self.repository = RatingRepository(db)
        self.notification_service = NotificationService(db)
    
    async def add_rating(
        self,
        manual_id: int,
        rating: int,
        user_id: Optional[str] = None,
        comment: Optional[str] = None
    ) -> ManualRating:
        """
        Agregar o actualizar rating de un manual.
        
        Args:
            manual_id: ID del manual
            rating: Rating (1-5)
            user_id: ID del usuario (opcional)
            comment: Comentario (opcional)
        
        Returns:
            Rating creado/actualizado
        """
        if rating < 1 or rating > 5:
            raise ValueError("Rating debe estar entre 1 y 5")
        
        try:
            manual = await self._get_manual(manual_id)
            if not manual:
                raise ValueError(f"Manual {manual_id} no encontrado")
            
            existing_rating = None
            if user_id:
                existing_rating = await self.repository.get_by_manual_and_user(
                    manual_id, user_id
                )
            
            if existing_rating:
                existing_rating.rating = rating
                existing_rating.comment = comment
                existing_rating.updated_at = datetime.now()
                rating_obj = existing_rating
            else:
                rating_obj = ManualRating(
                    manual_id=manual_id,
                    user_id=user_id,
                    rating=rating,
                    comment=comment
                )
                rating_obj = await self.repository.save(rating_obj)
            
            await self._update_manual_rating(manual_id)
            
            if manual.user_id and manual.user_id != user_id:
                await self._notify_manual_owner(manual, rating, manual_id)
            
            self.log_info(f"Rating agregado/actualizado: Manual {manual_id}, Rating: {rating}")
            return rating_obj
        
        except Exception as e:
            await self.db.rollback()
            self.log_error(f"Error agregando rating: {str(e)}")
            raise
    
    async def get_ratings(
        self,
        manual_id: int,
        limit: int = 20,
        offset: int = 0
    ) -> List[ManualRating]:
        """
        Obtener ratings de un manual.
        
        Args:
            manual_id: ID del manual
            limit: Límite de resultados
            offset: Offset para paginación
        
        Returns:
            Lista de ratings
        """
        return await self.repository.get_by_manual(manual_id, limit, offset)
    
    async def get_user_rating(
        self,
        manual_id: int,
        user_id: str
    ) -> Optional[ManualRating]:
        """
        Obtener rating de un usuario específico.
        
        Args:
            manual_id: ID del manual
            user_id: ID del usuario
        
        Returns:
            Rating o None
        """
        return await self.repository.get_by_manual_and_user(manual_id, user_id)
    
    async def _get_manual(self, manual_id: int) -> Optional[Manual]:
        """Obtener manual por ID."""
        from sqlalchemy import select
        result = await self.db.execute(
            select(Manual).where(Manual.id == manual_id)
        )
        return result.scalar_one_or_none()
    
    async def _update_manual_rating(self, manual_id: int):
        """Actualizar promedio de rating del manual."""
        try:
            avg_rating = await self.repository.get_average_rating(manual_id)
            rating_count = await self.repository.get_rating_count(manual_id)
            
            manual = await self._get_manual(manual_id)
            if manual:
                manual.average_rating = round(avg_rating, 2)
                manual.rating_count = rating_count
                await self.db.commit()
        except Exception as e:
            self.log_warning(f"Error actualizando rating del manual: {str(e)}")
            await self.db.rollback()
    
    async def _notify_manual_owner(
        self,
        manual: Manual,
        rating: int,
        manual_id: int
    ):
        """Notificar al dueño del manual sobre nuevo rating."""
        try:
            await self.notification_service.create_notification(
                user_id=manual.user_id,
                notification_type="rating",
                title=f"Nuevo rating en tu manual",
                message=f"Tu manual '{manual.title or 'Sin título'}' recibió un rating de {rating} estrellas",
                manual_id=manual_id
            )
        except Exception as e:
            self.log_warning(f"Error creando notificación: {str(e)}")

