"""
Module Factory
Factory for creating and configuring modules
"""

from typing import Dict, Any, Optional, Type
from .module_registry import ModuleRegistry, get_registry


class ModuleFactory:
    """Factory for creating module instances"""
    
    def __init__(self, registry: Optional[ModuleRegistry] = None):
        self.registry = registry or get_registry()
    
    def create_document_module(
        self,
        repository: Any,
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create and configure document module"""
        from .document import (
            DocumentController,
            DocumentRepository
        )
        
        # Register module
        self.registry.register_module("document", {
            "path": "api.modules.document",
            "config": config or {}
        })
        
        # Return module components
        return {
            "repository": repository,
            "controller": DocumentController
        }
    
    def create_variant_module(
        self,
        repository: Any,
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create and configure variant module"""
        from .variant import (
            VariantController,
            VariantRepository
        )
        
        self.registry.register_module("variant", {
            "path": "api.modules.variant",
            "config": config or {}
        })
        
        return {
            "repository": repository,
            "controller": VariantController
        }
    
    def create_topic_module(
        self,
        repository: Any,
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create and configure topic module"""
        from .topic import (
            TopicController,
            TopicRepository
        )
        
        self.registry.register_module("topic", {
            "path": "api.modules.topic",
            "config": config or {}
        })
        
        return {
            "repository": repository,
            "controller": TopicController
        }
    
    def create_all_modules(
        self,
        repositories: Dict[str, Any],
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """Create all modules at once"""
        modules = {}
        
        if "document" in repositories:
            modules["document"] = self.create_document_module(
                repositories["document"],
                config
            )
        
        if "variant" in repositories:
            modules["variant"] = self.create_variant_module(
                repositories["variant"],
                config
            )
        
        if "topic" in repositories:
            modules["topic"] = self.create_topic_module(
                repositories["topic"],
                config
            )
        
        return modules






