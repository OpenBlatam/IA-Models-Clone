"""
Favorite Repository
==================

Repository para acceso a datos de favoritos.
"""

from typing import Optional, List
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_

from ...core.base.service_base import BaseService
from ...database.models import Manual, ManualFavorite


class FavoriteRepository(BaseService):
    """Repository para acceso a datos de favoritos."""
    
    def __init__(self, db: AsyncSession):
        """
        Inicializar repository.
        
        Args:
            db: Sesión de base de datos
        """
        super().__init__(logger_name=__name__)
        self.db = db
    
    async def save(self, favorite: ManualFavorite) -> ManualFavorite:
        """
        Guardar favorito en base de datos.
        
        Args:
            favorite: Instancia de ManualFavorite
        
        Returns:
            Favorito guardado
        """
        try:
            self.db.add(favorite)
            await self.db.commit()
            await self.db.refresh(favorite)
            self.log_info(f"Favorito guardado: ID {favorite.id}")
            return favorite
        except Exception as e:
            await self.db.rollback()
            self.log_error(f"Error guardando favorito: {str(e)}")
            raise
    
    async def get_by_manual_and_user(
        self,
        manual_id: int,
        user_id: str
    ) -> Optional[ManualFavorite]:
        """
        Obtener favorito por manual y usuario.
        
        Args:
            manual_id: ID del manual
            user_id: ID del usuario
        
        Returns:
            Favorito o None
        """
        try:
            result = await self.db.execute(
                select(ManualFavorite).where(
                    and_(
                        ManualFavorite.manual_id == manual_id,
                        ManualFavorite.user_id == user_id
                    )
                )
            )
            return result.scalar_one_or_none()
        except Exception as e:
            self.log_error(f"Error obteniendo favorito: {str(e)}")
            return None
    
    async def get_user_favorites(
        self,
        user_id: str,
        limit: int = 20,
        offset: int = 0
    ) -> List[Manual]:
        """
        Obtener favoritos de un usuario.
        
        Args:
            user_id: ID del usuario
            limit: Límite de resultados
            offset: Offset para paginación
        
        Returns:
            Lista de manuales favoritos
        """
        try:
            result = await self.db.execute(
                select(Manual)
                .join(ManualFavorite, Manual.id == ManualFavorite.manual_id)
                .where(ManualFavorite.user_id == user_id)
                .order_by(ManualFavorite.created_at.desc())
                .limit(limit)
                .offset(offset)
            )
            return list(result.scalars().all())
        except Exception as e:
            self.log_error(f"Error obteniendo favoritos: {str(e)}")
            return []
    
    async def delete(self, favorite: ManualFavorite):
        """
        Eliminar favorito.
        
        Args:
            favorite: Instancia de ManualFavorite
        """
        try:
            await self.db.delete(favorite)
            await self.db.commit()
            self.log_info(f"Favorito eliminado: ID {favorite.id}")
        except Exception as e:
            await self.db.rollback()
            self.log_error(f"Error eliminando favorito: {str(e)}")
            raise

