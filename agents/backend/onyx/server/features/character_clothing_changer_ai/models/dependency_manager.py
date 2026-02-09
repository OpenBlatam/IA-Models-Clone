"""
Dependency Manager for Flux2 Clothing Changer
==============================================

Advanced dependency management and resolution system.
"""

import time
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class Dependency:
    """Dependency information."""
    dependency_id: str
    name: str
    version: str
    dependencies: List[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []
        if self.metadata is None:
            self.metadata = {}


class DependencyManager:
    """Advanced dependency management system."""
    
    def __init__(self):
        """Initialize dependency manager."""
        self.dependencies: Dict[str, Dependency] = {}
        self.dependency_graph: Dict[str, Set[str]] = defaultdict(set)
        self.resolved_cache: Dict[str, List[str]] = {}
    
    def register_dependency(
        self,
        dependency_id: str,
        name: str,
        version: str,
        dependencies: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dependency:
        """
        Register dependency.
        
        Args:
            dependency_id: Dependency identifier
            name: Dependency name
            version: Dependency version
            dependencies: List of dependency IDs
            metadata: Optional metadata
            
        Returns:
            Created dependency
        """
        dependency = Dependency(
            dependency_id=dependency_id,
            name=name,
            version=version,
            dependencies=dependencies or [],
            metadata=metadata or {},
        )
        
        self.dependencies[dependency_id] = dependency
        
        # Update dependency graph
        for dep_id in dependency.dependencies:
            self.dependency_graph[dependency_id].add(dep_id)
        
        logger.info(f"Registered dependency: {dependency_id}")
        return dependency
    
    def resolve_dependencies(
        self,
        dependency_id: str,
        include_transitive: bool = True,
    ) -> List[str]:
        """
        Resolve dependencies.
        
        Args:
            dependency_id: Dependency identifier
            include_transitive: Include transitive dependencies
            
        Returns:
            List of resolved dependency IDs
        """
        if dependency_id not in self.dependencies:
            return []
        
        cache_key = f"{dependency_id}_{include_transitive}"
        if cache_key in self.resolved_cache:
            return self.resolved_cache[cache_key]
        
        resolved = []
        visited = set()
        
        def resolve_recursive(dep_id: str):
            if dep_id in visited:
                return
            
            visited.add(dep_id)
            
            if dep_id in self.dependencies:
                dependency = self.dependencies[dep_id]
                
                if include_transitive:
                    for sub_dep in dependency.dependencies:
                        resolve_recursive(sub_dep)
                
                if dep_id not in resolved:
                    resolved.append(dep_id)
        
        resolve_recursive(dependency_id)
        
        # Cache result
        self.resolved_cache[cache_key] = resolved
        
        return resolved
    
    def check_conflicts(
        self,
        dependency_ids: List[str],
    ) -> List[Dict[str, Any]]:
        """
        Check for dependency conflicts.
        
        Args:
            dependency_ids: List of dependency IDs
            
        Returns:
            List of conflicts
        """
        conflicts = []
        dependency_versions: Dict[str, Dict[str, str]] = defaultdict(dict)
        
        for dep_id in dependency_ids:
            resolved = self.resolve_dependencies(dep_id)
            for resolved_id in resolved:
                if resolved_id in self.dependencies:
                    dep = self.dependencies[resolved_id]
                    if dep.name in dependency_versions:
                        existing_version = dependency_versions[dep.name].get("version")
                        if existing_version and existing_version != dep.version:
                            conflicts.append({
                                "dependency_name": dep.name,
                                "version1": existing_version,
                                "version2": dep.version,
                                "source1": dependency_versions[dep.name].get("source"),
                                "source2": dep_id,
                            })
                    else:
                        dependency_versions[dep.name] = {
                            "version": dep.version,
                            "source": dep_id,
                        }
        
        return conflicts
    
    def get_dependency_tree(
        self,
        dependency_id: str,
    ) -> Dict[str, Any]:
        """
        Get dependency tree.
        
        Args:
            dependency_id: Dependency identifier
            
        Returns:
            Dependency tree
        """
        if dependency_id not in self.dependencies:
            return {}
        
        def build_tree(dep_id: str, visited: Set[str]) -> Dict[str, Any]:
            if dep_id in visited:
                return {"circular": True}
            
            visited.add(dep_id)
            
            if dep_id not in self.dependencies:
                return {}
            
            dependency = self.dependencies[dep_id]
            tree = {
                "id": dep_id,
                "name": dependency.name,
                "version": dependency.version,
                "dependencies": [],
            }
            
            for sub_dep_id in dependency.dependencies:
                tree["dependencies"].append(build_tree(sub_dep_id, visited.copy()))
            
            return tree
        
        return build_tree(dependency_id, set())
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dependency manager statistics."""
        return {
            "total_dependencies": len(self.dependencies),
            "dependency_graph_size": len(self.dependency_graph),
            "cached_resolutions": len(self.resolved_cache),
        }


