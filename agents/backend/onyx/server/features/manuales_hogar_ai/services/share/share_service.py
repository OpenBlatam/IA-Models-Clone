"""
Share Service
============

Servicio principal para compartir manuales.
"""

from typing import Optional
from datetime import datetime, timedelta
from sqlalchemy.ext.asyncio import AsyncSession

from ...core.base.service_base import BaseService
from ...database.models import Manual, ManualShare
from .token_generator import TokenGenerator
from .share_repository import ShareRepository


class ShareService(BaseService):
    """Servicio para gestionar compartir manuales."""
    
    def __init__(self, db: AsyncSession):
        """
        Inicializar servicio.
        
        Args:
            db: Sesión de base de datos
        """
        super().__init__(logger_name=__name__)
        self.db = db
        self.token_generator = TokenGenerator()
        self.repository = ShareRepository(db)
    
    async def create_share(
        self,
        manual_id: int,
        shared_by: Optional[str] = None,
        expires_in_days: Optional[int] = None
    ) -> ManualShare:
        """
        Crear enlace para compartir manual.
        
        Args:
            manual_id: ID del manual
            shared_by: Usuario que comparte (opcional)
            expires_in_days: Días hasta expiración (opcional)
        
        Returns:
            Share creado
        """
        try:
            manual = await self._get_manual(manual_id)
            if not manual:
                raise ValueError(f"Manual {manual_id} no encontrado")
            
            share_token = self.token_generator.generate()
            expires_at = None
            if expires_in_days:
                expires_at = datetime.now() + timedelta(days=expires_in_days)
            
            share = ManualShare(
                manual_id=manual_id,
                share_token=share_token,
                shared_by=shared_by,
                expires_at=expires_at
            )
            
            share = await self.repository.save(share)
            
            await self._update_manual_share(manual, share_token)
            
            self.log_info(f"Share creado: Manual {manual_id}, Token: {share_token[:8]}...")
            return share
        
        except Exception as e:
            await self.db.rollback()
            self.log_error(f"Error creando share: {str(e)}")
            raise
    
    async def get_manual_by_token(
        self,
        share_token: str
    ) -> Optional[Manual]:
        """
        Obtener manual por token de share.
        
        Args:
            share_token: Token del share
        
        Returns:
            Manual o None
        """
        return await self.repository.get_manual_by_token(share_token)
    
    async def _get_manual(self, manual_id: int) -> Optional[Manual]:
        """Obtener manual por ID."""
        from sqlalchemy import select
        result = await self.db.execute(
            select(Manual).where(Manual.id == manual_id)
        )
        return result.scalar_one_or_none()
    
    async def _update_manual_share(self, manual: Manual, share_token: str):
        """Actualizar información de share en manual."""
        try:
            if not manual.share_token:
                manual.share_token = share_token
            manual.share_count += 1
            await self.db.commit()
        except Exception as e:
            self.log_warning(f"Error actualizando share del manual: {str(e)}")
            await self.db.rollback()

