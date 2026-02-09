"""
Rating Repository
================

Repository para acceso a datos de ratings.
"""

from typing import Optional, List
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_

from ...core.base.service_base import BaseService
from ...database.models import Manual, ManualRating


class RatingRepository(BaseService):
    """Repository para acceso a datos de ratings."""
    
    def __init__(self, db: AsyncSession):
        """
        Inicializar repository.
        
        Args:
            db: Sesión de base de datos
        """
        super().__init__(logger_name=__name__)
        self.db = db
    
    async def save(self, rating: ManualRating) -> ManualRating:
        """
        Guardar rating en base de datos.
        
        Args:
            rating: Instancia de ManualRating
        
        Returns:
            Rating guardado
        """
        try:
            self.db.add(rating)
            await self.db.commit()
            await self.db.refresh(rating)
            self.log_info(f"Rating guardado: ID {rating.id}")
            return rating
        except Exception as e:
            await self.db.rollback()
            self.log_error(f"Error guardando rating: {str(e)}")
            raise
    
    async def get_by_manual_and_user(
        self,
        manual_id: int,
        user_id: str
    ) -> Optional[ManualRating]:
        """
        Obtener rating por manual y usuario.
        
        Args:
            manual_id: ID del manual
            user_id: ID del usuario
        
        Returns:
            Rating o None
        """
        try:
            result = await self.db.execute(
                select(ManualRating).where(
                    and_(
                        ManualRating.manual_id == manual_id,
                        ManualRating.user_id == user_id
                    )
                )
            )
            return result.scalar_one_or_none()
        except Exception as e:
            self.log_error(f"Error obteniendo rating: {str(e)}")
            return None
    
    async def get_by_manual(
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
        try:
            result = await self.db.execute(
                select(ManualRating)
                .where(ManualRating.manual_id == manual_id)
                .order_by(ManualRating.created_at.desc())
                .limit(limit)
                .offset(offset)
            )
            return list(result.scalars().all())
        except Exception as e:
            self.log_error(f"Error obteniendo ratings: {str(e)}")
            return []
    
    async def get_average_rating(self, manual_id: int) -> float:
        """
        Obtener promedio de rating de un manual.
        
        Args:
            manual_id: ID del manual
        
        Returns:
            Promedio de rating
        """
        try:
            result = await self.db.execute(
                select(func.avg(ManualRating.rating)).where(
                    ManualRating.manual_id == manual_id
                )
            )
            return result.scalar() or 0.0
        except Exception as e:
            self.log_error(f"Error obteniendo promedio: {str(e)}")
            return 0.0
    
    async def get_rating_count(self, manual_id: int) -> int:
        """
        Obtener cantidad de ratings de un manual.
        
        Args:
            manual_id: ID del manual
        
        Returns:
            Cantidad de ratings
        """
        try:
            result = await self.db.execute(
                select(func.count(ManualRating.id)).where(
                    ManualRating.manual_id == manual_id
                )
            )
            return result.scalar() or 0
        except Exception as e:
            self.log_error(f"Error obteniendo cantidad: {str(e)}")
            return 0

