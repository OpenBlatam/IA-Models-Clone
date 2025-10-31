"""
Module Router
Auto-generate FastAPI routers from modules
"""

from typing import Dict, Any, List
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.routing import APIRoute

from .module_registry import get_registry
from ..utils.auth import get_current_user


class ModuleRouter:
    """Generate FastAPI routers from modules"""
    
    def __init__(self, registry=None):
        self.registry = registry or get_registry()
        self.routers: Dict[str, APIRouter] = {}
    
    def create_router(
        self,
        module_name: str,
        prefix: str = None,
        tags: List[str] = None
    ) -> APIRouter:
        """Create FastAPI router for a module"""
        if module_name in self.routers:
            return self.routers[module_name]
        
        # Load module if not loaded
        if not self.registry.get_module_info(module_name) or \
           not self.registry.get_module_info(module_name).get("loaded"):
            self.registry.load_module(module_name)
        
        # Create router
        router = APIRouter(
            prefix=prefix or f"/api/v1/{module_name}",
            tags=tags or [module_name.title()]
        )
        
        # Get controller from module
        controller = self.registry.get_controller(module_name, "controller")
        
        if not controller:
            return router
        
        # Auto-generate routes based on controller methods
        self._add_controller_routes(router, controller, module_name)
        
        self.routers[module_name] = router
        return router
    
    def _add_controller_routes(
        self,
        router: APIRouter,
        controller_class: Type,
        module_name: str
    ):
        """Add routes from controller methods"""
        controller_instance = None
        
        # Document module routes
        if module_name == "document":
            from .document.application import (
                UploadDocumentCommand,
                GetDocumentQuery,
                ListDocumentsQuery,
                DeleteDocumentCommand
            )
            
            @router.post("/upload", summary="Upload document")
            async def upload(
                request: Request,
                user_id: str = Depends(get_current_user),
                filename: str = None,
                file_content: bytes = None
            ):
                if not controller_instance:
                    controller_instance = self._get_controller_instance(controller_class, module_name)
                
                command = UploadDocumentCommand(
                    user_id=user_id,
                    filename=filename,
                    file_content=file_content
                )
                return await controller_instance.upload(request, command)
            
            @router.get("/{document_id}", summary="Get document")
            async def get(
                request: Request,
                document_id: str,
                user_id: str = Depends(get_current_user)
            ):
                if not controller_instance:
                    controller_instance = self._get_controller_instance(controller_class, module_name)
                return await controller_instance.get(request, document_id, user_id)
            
            @router.get("", summary="List documents")
            async def list(
                request: Request,
                user_id: str = Depends(get_current_user),
                limit: int = 20,
                offset: int = 0,
                search: str = None
            ):
                if not controller_instance:
                    controller_instance = self._get_controller_instance(controller_class, module_name)
                return await controller_instance.list(request, user_id, limit, offset, search)
            
            @router.delete("/{document_id}", summary="Delete document")
            async def delete(
                request: Request,
                document_id: str,
                user_id: str = Depends(get_current_user)
            ):
                if not controller_instance:
                    controller_instance = self._get_controller_instance(controller_class, module_name)
                return await controller_instance.delete(request, document_id, user_id)
        
        # Variant module routes
        elif module_name == "variant":
            from .variant.application import GenerateVariantsCommand
            
            @router.post("/generate", summary="Generate variants")
            async def generate(
                request: Request,
                document_id: str,
                user_id: str = Depends(get_current_user),
                variant_count: int = 10
            ):
                if not controller_instance:
                    controller_instance = self._get_controller_instance(controller_class, module_name)
                
                command = GenerateVariantsCommand(
                    document_id=document_id,
                    user_id=user_id,
                    variant_count=variant_count
                )
                return await controller_instance.generate(request, command)
            
            @router.get("/documents/{document_id}/variants", summary="List variants")
            async def list(
                request: Request,
                document_id: str,
                user_id: str = Depends(get_current_user),
                limit: int = 20,
                offset: int = 0
            ):
                if not controller_instance:
                    controller_instance = self._get_controller_instance(controller_class, module_name)
                return await controller_instance.list(request, document_id, user_id, limit, offset)
        
        # Topic module routes
        elif module_name == "topic":
            from .topic.application import ExtractTopicsCommand
            
            @router.post("/extract", summary="Extract topics")
            async def extract(
                request: Request,
                document_id: str,
                user_id: str = Depends(get_current_user),
                min_relevance: float = 0.5
            ):
                if not controller_instance:
                    controller_instance = self._get_controller_instance(controller_class, module_name)
                
                command = ExtractTopicsCommand(
                    document_id=document_id,
                    user_id=user_id,
                    min_relevance=min_relevance
                )
                return await controller_instance.extract(request, command)
            
            @router.get("/documents/{document_id}/topics", summary="List topics")
            async def list(
                request: Request,
                document_id: str,
                user_id: str = Depends(get_current_user),
                min_relevance: float = 0.5
            ):
                if not controller_instance:
                    controller_instance = self._get_controller_instance(controller_class, module_name)
                return await controller_instance.list(request, document_id, user_id, min_relevance)
    
    def _get_controller_instance(self, controller_class: Type, module_name: str):
        """Get or create controller instance"""
        # This would properly instantiate with dependencies
        # For now, return a placeholder
        return controller_class
    
    def register_all_modules(self, app) -> None:
        """Register all modules as routers"""
        modules = self.registry.list_modules()
        
        for module_name in modules:
            router = self.create_router(module_name)
            app.include_router(router)






