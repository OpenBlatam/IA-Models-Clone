"""
Variant Module - Presentation Layer
"""

from typing import Dict, Any, List, Optional
from fastapi import Request

from .domain import VariantEntity
from .application import (
    GenerateVariantsUseCase,
    GetVariantUseCase,
    ListVariantsUseCase
)


class VariantPresenter:
    """Presenter for variant entities"""
    
    @staticmethod
    def to_dict(variant: VariantEntity) -> Dict[str, Any]:
        """Convert variant to dictionary"""
        return {
            "id": variant.id,
            "document_id": variant.document_id,
            "variant_type": variant.variant_type,
            "similarity_score": variant.similarity_score,
            "status": variant.status,
            "created_at": variant.created_at.isoformat(),
            "metadata": variant.metadata or {}
        }
    
    @staticmethod
    def to_list(variants: List[VariantEntity]) -> Dict[str, Any]:
        """Convert variant list to response format"""
        return {
            "items": [VariantPresenter.to_dict(v) for v in variants],
            "count": len(variants),
            "high_quality_count": sum(1 for v in variants if v.has_high_quality())
        }


class VariantController:
    """Controller for variant operations"""
    
    def __init__(
        self,
        generate_use_case: GenerateVariantsUseCase,
        get_use_case: GetVariantUseCase,
        list_use_case: ListVariantsUseCase
    ):
        self.generate_use_case = generate_use_case
        self.get_use_case = get_use_case
        self.list_use_case = list_use_case
        self.presenter = VariantPresenter()
    
    async def generate(
        self,
        request: Request,
        command
    ) -> Dict[str, Any]:
        """Handle generate request"""
        try:
            variants = await self.generate_use_case.execute(command)
            return {
                "success": True,
                "data": self.presenter.to_list(variants),
                "request_id": getattr(request.state, 'request_id', None)
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "request_id": getattr(request.state, 'request_id', None)
            }
    
    async def get(
        self,
        request: Request,
        variant_id: str,
        user_id: str
    ) -> Dict[str, Any]:
        """Handle get request"""
        try:
            from .application import GetVariantQuery
            query = GetVariantQuery(variant_id=variant_id, user_id=user_id)
            variant = await self.get_use_case.execute(query)
            
            if not variant:
                return {
                    "success": False,
                    "error": "Variant not found",
                    "request_id": getattr(request.state, 'request_id', None)
                }
            
            return {
                "success": True,
                "data": self.presenter.to_dict(variant),
                "request_id": getattr(request.state, 'request_id', None)
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "request_id": getattr(request.state, 'request_id', None)
            }
    
    async def list(
        self,
        request: Request,
        document_id: str,
        user_id: str,
        limit: int = 20,
        offset: int = 0
    ) -> Dict[str, Any]:
        """Handle list request"""
        try:
            from .application import ListVariantsQuery
            query = ListVariantsQuery(
                document_id=document_id,
                user_id=user_id,
                limit=limit,
                offset=offset
            )
            variants = await self.list_use_case.execute(query)
            
            return {
                "success": True,
                "data": self.presenter.to_list(variants),
                "request_id": getattr(request.state, 'request_id', None)
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "request_id": getattr(request.state, 'request_id', None)
            }
