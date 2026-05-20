"""
Share Repository
===============

Repository para acceso a datos de shares.
"""

from typing import Optional
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_

from ...core.base.service_base import BaseService
from ...database.models import Manual, ManualShare


class ShareRepository(BaseService):
    """Repository para acceso a datos de shares."""
    
    def __init__(self, db: AsyncSession):
        """
        Inicializar repository.
        
        Args:
            db: Sesión de base de datos
        """
        super().__init__(logger_name=__name__)
        self.db = db
    
    async def save(self, share: ManualShare) -> ManualShare:
        """
        Guardar share en base de datos.
        
        Args:
            share: Instancia de ManualShare
        
        Returns:
            Share guardado
        """
        try:
            self.db.add(share)
            await self.db.commit()
            await self.db.refresh(share)
            self.log_info(f"Share guardado: ID {share.id}")
            return share
        except Exception as e:
            await self.db.rollback()
            self.log_error(f"Error guardando share: {str(e)}")
            raise
    
    async def get_by_token(self, share_token: str) -> Optional[ManualShare]:
        """
        Obtener share por token.
        
        Args:
            share_token: Token del share
        
        Returns:
            Share o None
        """
        try:
            result = await self.db.execute(
                select(ManualShare).where(
                    and_(
                        ManualShare.share_token == share_token,
                        (ManualShare.expires_at.is_(None)) |
                        (ManualShare.expires_at > datetime.now())
                    )
                )
            )
            return result.scalar_one_or_none()
        except Exception as e:
            self.log_error(f"Error obteniendo share: {str(e)}")
            return None
    
    async def get_manual_by_token(self, share_token: str) -> Optional[Manual]:
        """
        Obtener manual por token de share.
        
        Args:
            share_token: Token del share
        
        Returns:
            Manual o None
        """
        try:
            result = await self.db.execute(
                select(Manual).join(
                    ManualShare,
                    Manual.id == ManualShare.manual_id
                ).where(
                    and_(
                        ManualShare.share_token == share_token,
                        (ManualShare.expires_at.is_(None)) |
                        (ManualShare.expires_at > datetime.now())
                    )
                )
            )
            return result.scalar_one_or_none()
        except Exception as e:
            self.log_error(f"Error obteniendo manual por token: {str(e)}")
            return None

