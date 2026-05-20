"""
Schema management for KV cache.

This module provides schema validation, evolution, and versioning
for cache data structures.
"""

import time
import threading
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import jsonschema
from jsonschema import validate, ValidationError


class SchemaVersion(Enum):
    """Schema versioning strategies."""
    STRICT = "strict"  # No changes allowed
    BACKWARD_COMPATIBLE = "backward_compatible"  # New fields only
    FORWARD_COMPATIBLE = "forward_compatible"  # Remove fields only
    FULLY_COMPATIBLE = "fully_compatible"  # Any compatible changes


@dataclass
class SchemaDefinition:
    """Schema definition."""
    schema_id: str
    version: str
    schema: Dict[str, Any]  # JSON Schema
    created_at: float
    description: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def validate(self, data: Any) -> bool:
        """Validate data against schema."""
        try:
            validate(instance=data, schema=self.schema)
            return True
        except ValidationError:
            return False


@dataclass
class SchemaEvolution:
    """Schema evolution information."""
    from_version: str
    to_version: str
    migration_function: Optional[Callable[[Any], Any]] = None
    compatibility_level: SchemaVersion = SchemaVersion.BACKWARD_COMPATIBLE


@dataclass
class ValidationResult:
    """Result of schema validation."""
    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class CacheSchemaManager:
    """Manages schemas for cache data."""
    
    def __init__(self):
        self._schemas: Dict[str, Dict[str, SchemaDefinition]] = {}  # {schema_id: {version: SchemaDefinition}}
        self._evolutions: Dict[str, List[SchemaEvolution]] = {}  # {schema_id: [evolutions]}
        self._default_schemas: Dict[str, str] = {}  # {schema_id: default_version}
        self._lock = threading.Lock()
        
    def register_schema(
        self,
        schema_id: str,
        version: str,
        schema: Dict[str, Any],
        description: Optional[str] = None,
        set_as_default: bool = False
    ) -> SchemaDefinition:
        """Register a schema."""
        with self._lock:
            if schema_id not in self._schemas:
                self._schemas[schema_id] = {}
                
            schema_def = SchemaDefinition(
                schema_id=schema_id,
                version=version,
                schema=schema,
                created_at=time.time(),
                description=description
            )
            
            self._schemas[schema_id][version] = schema_def
            
            if set_as_default or schema_id not in self._default_schemas:
                self._default_schemas[schema_id] = version
                
            return schema_def
            
    def get_schema(
        self,
        schema_id: str,
        version: Optional[str] = None
    ) -> Optional[SchemaDefinition]:
        """Get a schema."""
        with self._lock:
            if schema_id not in self._schemas:
                return None
                
            if version:
                return self._schemas[schema_id].get(version)
            else:
                default_version = self._default_schemas.get(schema_id)
                if default_version:
                    return self._schemas[schema_id].get(default_version)
                # Return latest version
                versions = sorted(self._schemas[schema_id].keys())
                if versions:
                    return self._schemas[schema_id][versions[-1]]
                return None
                
    def validate_data(
        self,
        schema_id: str,
        data: Any,
        version: Optional[str] = None
    ) -> ValidationResult:
        """Validate data against a schema."""
        schema_def = self.get_schema(schema_id, version)
        if not schema_def:
            return ValidationResult(
                valid=False,
                errors=[f"Schema '{schema_id}' not found"]
            )
            
        errors = []
        warnings = []
        
        try:
            validate(instance=data, schema=schema_def.schema)
            return ValidationResult(valid=True, errors=errors, warnings=warnings)
        except ValidationError as e:
            errors.append(str(e))
            return ValidationResult(valid=False, errors=errors, warnings=warnings)
            
    def register_evolution(
        self,
        schema_id: str,
        from_version: str,
        to_version: str,
        migration_function: Optional[Callable[[Any], Any]] = None,
        compatibility_level: SchemaVersion = SchemaVersion.BACKWARD_COMPATIBLE
    ) -> None:
        """Register a schema evolution."""
        with self._lock:
            if schema_id not in self._evolutions:
                self._evolutions[schema_id] = []
                
            evolution = SchemaEvolution(
                from_version=from_version,
                to_version=to_version,
                migration_function=migration_function,
                compatibility_level=compatibility_level
            )
            
            self._evolutions[schema_id].append(evolution)
            
    def migrate_data(
        self,
        schema_id: str,
        data: Any,
        from_version: str,
        to_version: str
    ) -> Any:
        """Migrate data from one schema version to another."""
        if from_version == to_version:
            return data
            
        evolutions = self._evolutions.get(schema_id, [])
        
        # Find evolution path
        evolution_path = self._find_evolution_path(
            evolutions,
            from_version,
            to_version
        )
        
        if not evolution_path:
            raise ValueError(
                f"No migration path from {from_version} to {to_version}"
            )
            
        # Apply migrations in sequence
        migrated_data = data
        for evolution in evolution_path:
            if evolution.migration_function:
                migrated_data = evolution.migration_function(migrated_data)
                
        return migrated_data
        
    def _find_evolution_path(
        self,
        evolutions: List[SchemaEvolution],
        from_version: str,
        to_version: str
    ) -> List[SchemaEvolution]:
        """Find path through evolution chain."""
        # Simple implementation - find direct path
        # Real implementation would use graph traversal
        
        path = []
        current_version = from_version
        
        while current_version != to_version:
            # Find evolution from current version
            found = False
            for evolution in evolutions:
                if evolution.from_version == current_version:
                    path.append(evolution)
                    current_version = evolution.to_version
                    found = True
                    break
                    
            if not found:
                return []  # No path found
                
            if len(path) > 100:  # Prevent infinite loops
                return []
                
        return path
        
    def list_schemas(self) -> List[str]:
        """List all registered schema IDs."""
        return list(self._schemas.keys())
        
    def list_versions(self, schema_id: str) -> List[str]:
        """List versions for a schema."""
        if schema_id not in self._schemas:
            return []
        return list(self._schemas[schema_id].keys())
        
    def set_default_version(self, schema_id: str, version: str) -> None:
        """Set default version for a schema."""
        with self._lock:
            if schema_id in self._schemas and version in self._schemas[schema_id]:
                self._default_schemas[schema_id] = version
                
    def delete_schema(self, schema_id: str, version: Optional[str] = None) -> None:
        """Delete a schema."""
        with self._lock:
            if schema_id not in self._schemas:
                return
                
            if version:
                if version in self._schemas[schema_id]:
                    del self._schemas[schema_id][version]
                    if self._default_schemas.get(schema_id) == version:
                        # Set new default
                        versions = list(self._schemas[schema_id].keys())
                        if versions:
                            self._default_schemas[schema_id] = versions[0]
                        else:
                            del self._default_schemas[schema_id]
            else:
                # Delete all versions
                del self._schemas[schema_id]
                if schema_id in self._default_schemas:
                    del self._default_schemas[schema_id]
                if schema_id in self._evolutions:
                    del self._evolutions[schema_id]


class SchemaAwareCache:
    """Cache wrapper with schema validation."""
    
    def __init__(
        self,
        cache: Any,
        schema_manager: CacheSchemaManager,
        schema_id: str,
        schema_version: Optional[str] = None,
        strict_validation: bool = True
    ):
        self.cache = cache
        self.schema_manager = schema_manager
        self.schema_id = schema_id
        self.schema_version = schema_version
        self.strict_validation = strict_validation
        
    def get(self, key: str) -> Any:
        """Get value with schema validation."""
        value = self.cache.get(key)
        if value is None:
            return None
            
        # Validate on read
        result = self.schema_manager.validate_data(
            self.schema_id,
            value,
            self.schema_version
        )
        
        if not result.valid and self.strict_validation:
            raise ValueError(f"Data validation failed: {result.errors}")
            
        return value
        
    def put(self, key: str, value: Any) -> bool:
        """Put value with schema validation."""
        # Validate before write
        result = self.schema_manager.validate_data(
            self.schema_id,
            value,
            self.schema_version
        )
        
        if not result.valid:
            if self.strict_validation:
                raise ValueError(f"Data validation failed: {result.errors}")
            else:
                # Log warning but allow
                print(f"Warning: Validation errors: {result.errors}")
                
        return self.cache.put(key, value)
        
    def migrate_key(
        self,
        key: str,
        from_version: str,
        to_version: str
    ) -> bool:
        """Migrate data for a key to new schema version."""
        value = self.cache.get(key)
        if value is None:
            return False
            
        try:
            migrated_value = self.schema_manager.migrate_data(
                self.schema_id,
                value,
                from_version,
                to_version
            )
            return self.cache.put(key, migrated_value)
        except Exception as e:
            print(f"Migration failed: {e}")
            return False
















