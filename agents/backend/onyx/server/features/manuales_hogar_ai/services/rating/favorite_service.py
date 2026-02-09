"""
Favorite Service
===============

Servicio especializado para gestión de favoritos.
"""

from typing import List
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_

from ...core.base.service_base import BaseService
from ...database.models import Manual, ManualFavorite
from .favorite_repository import FavoriteRepository


class FavoriteService(BaseService):
    """Servicio para gestión de favoritos."""
    
    def __init__(self, db: AsyncSession):
        """
        Inicializar servicio de favoritos.
        
        Args:
            db: Sesión de base de datos
        """
        super().__init__(logger_name=__name__)
        self.db = db
        self.repository = FavoriteRepository(db)
    
    async def add_favorite(
        self,
        manual_id: int,
        user_id: str
    ) -> ManualFavorite:
        """
        Agregar manual a favoritos.
        
        Args:
            manual_id: ID del manual
            user_id: ID del usuario
        
        Returns:
            Favorito creado
        """
        try:
            existing = await self.repository.get_by_manual_and_user(manual_id, user_id)
            if existing:
                return existing
            
            favorite = await self.repository.save(
                ManualFavorite(
                    manual_id=manual_id,
                    user_id=user_id
                )
            )
            
            await self._update_manual_favorite_count(manual_id, increment=1)
            
            self.log_info(f"Favorito agregado: Manual {manual_id}, Usuario {user_id}")
            return favorite
        
        except Exception as e:
            await self.db.rollback()
            self.log_error(f"Error agregando favorito: {str(e)}")
            raise
    
    async def remove_favorite(
        self,
        manual_id: int,
        user_id: str
    ) -> bool:
        """
        Remover manual de favoritos.
        
        Args:
            manual_id: ID del manual
            user_id: ID del usuario
        
        Returns:
            True si se removió exitosamente
        """
        try:
            favorite = await self.repository.get_by_manual_and_user(manual_id, user_id)
            if not favorite:
                return False
            
            await self.repository.delete(favorite)
            await self._update_manual_favorite_count(manual_id, increment=-1)
            
            self.log_info(f"Favorito removido: Manual {manual_id}, Usuario {user_id}")
            return True
        
        except Exception as e:
            await self.db.rollback()
            self.log_error(f"Error removiendo favorito: {str(e)}")
            return False
    
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
        return await self.repository.get_user_favorites(user_id, limit, offset)
    
    async def is_favorite(
        self,
        manual_id: int,
        user_id: str
    ) -> bool:
        """
        Verificar si un manual está en favoritos.
        
        Args:
            manual_id: ID del manual
            user_id: ID del usuario
        
        Returns:
            True si está en favoritos
        """
        favorite = await self.repository.get_by_manual_and_user(manual_id, user_id)
        return favorite is not None
    
    async def _update_manual_favorite_count(
        self,
        manual_id: int,
        increment: int
    ):
        """Actualizar contador de favoritos del manual."""
        try:
            result = await self.db.execute(
                select(Manual).where(Manual.id == manual_id)
            )
            manual = result.scalar_one_or_none()
            
            if manual:
                manual.favorite_count = max(0, manual.favorite_count + increment)
                await self.db.commit()
        except Exception as e:
            self.log_warning(f"Error actualizando contador: {str(e)}")
            await self.db.rollback()

