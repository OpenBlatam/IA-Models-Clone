"""
Schema Factory - Centralized schema creation and management
"""

from typing import Dict, Type, Any, Optional
from schemas.domains import get_schema, get_all_schemas, auto_discover_schemas


class SchemaFactory:
    """Factory for accessing schema classes"""
    
    def __init__(self):
        """Initialize the schema factory"""
        auto_discover_schemas()
        self._schemas: Dict[str, Type[Any]] = {}
    
    def get_schema(self, domain: str, schema_name: str) -> Type[Any]:
        """
        Get a schema class
        
        Args:
            domain: Schema domain (e.g., 'assessment', 'recovery')
            schema_name: Name of the schema
        
        Returns:
            Schema class
        """
        key = f"{domain}.{schema_name}"
        
        if key in self._schemas:
            return self._schemas[key]
        
        try:
            schema = get_schema(domain, schema_name)
            self._schemas[key] = schema
            return schema
        except ValueError as e:
            raise ValueError(f"Schema {key} not found: {e}")
    
    def list_available_schemas(self) -> Dict[str, Type[Any]]:
        """List all available schemas"""
        return get_all_schemas()
    
    def clear_cache(self) -> None:
        """Clear all cached schemas"""
        self._schemas.clear()


_global_factory: Optional[SchemaFactory] = None


def get_schema_factory() -> SchemaFactory:
    """Get the global schema factory instance"""
    global _global_factory
    if _global_factory is None:
        _global_factory = SchemaFactory()
    return _global_factory


def get_schema_class(domain: str, schema_name: str) -> Type[Any]:
    """Convenience function to get a schema class"""
    return get_schema_factory().get_schema(domain, schema_name)



