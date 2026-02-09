"""
Pipeline Dependencies
=====================

Sistema de dependencias entre etapas de pipeline.
Optimizado siguiendo mejores prácticas de Python y FastAPI.
"""

import logging
from typing import Dict, Any, List, Optional, Set, Tuple
from dataclasses import dataclass
from enum import Enum
from pydantic import BaseModel, Field

from .stages import PipelineStage

logger = logging.getLogger(__name__)


class DependencyType(str, Enum):
    """Tipo de dependencia."""
    REQUIRES = "requires"
    PROVIDES = "provides"
    CONFLICTS = "conflicts"


@dataclass(frozen=True)
class StageDependency:
    """
    Dependencia entre etapas.
    Inmutable para mejor seguridad.
    """
    stage_name: str
    dependency_type: DependencyType
    depends_on: Optional[str] = None
    provides: Optional[str] = None
    conflicts_with: Optional[str] = None


class DependencyGraph:
    """Grafo de dependencias optimizado."""
    
    def __init__(self) -> None:
        """Inicializar grafo."""
        self._graph: Dict[str, Set[str]] = {}
        self._conflicts: Dict[str, Set[str]] = {}
    
    def add_dependency(self, from_stage: str, to_stage: str) -> None:
        """
        Agregar dependencia.
        
        Args:
            from_stage: Etapa que depende
            to_stage: Etapa requerida
        """
        if from_stage not in self._graph:
            self._graph[from_stage] = set()
        self._graph[from_stage].add(to_stage)
    
    def add_conflict(self, stage1: str, stage2: str) -> None:
        """
        Agregar conflicto entre etapas.
        
        Args:
            stage1: Primera etapa
            stage2: Segunda etapa
        """
        if stage1 not in self._conflicts:
            self._conflicts[stage1] = set()
        if stage2 not in self._conflicts:
            self._conflicts[stage2] = set()
        self._conflicts[stage1].add(stage2)
        self._conflicts[stage2].add(stage1)
    
    def get_dependencies(self, stage_name: str) -> Set[str]:
        """Obtener dependencias de una etapa."""
        return self._graph.get(stage_name, set())
    
    def has_conflict(self, stage1: str, stage2: str) -> bool:
        """Verificar si hay conflicto entre dos etapas."""
        return (
            stage1 in self._conflicts and stage2 in self._conflicts[stage1]
        ) or (
            stage2 in self._conflicts and stage1 in self._conflicts[stage2]
        )


def _detect_circular_dependency(
    graph: Dict[str, Set[str]],
    stage_name: str,
    visited: Set[str],
    rec_stack: Set[str]
) -> Optional[List[str]]:
    """
    Detectar dependencias circulares usando DFS.
    
    Args:
        graph: Grafo de dependencias
        stage_name: Nombre de la etapa actual
        visited: Etapas visitadas
        rec_stack: Stack de recursión
        
    Returns:
        Lista de etapas en ciclo o None
    """
    if stage_name in rec_stack:
        return list(rec_stack) + [stage_name]
    
    if stage_name in visited:
        return None
    
    visited.add(stage_name)
    rec_stack.add(stage_name)
    
    for dep in graph.get(stage_name, set()):
        cycle = _detect_circular_dependency(graph, dep, visited, rec_stack)
        if cycle:
            return cycle
    
    rec_stack.remove(stage_name)
    return None


def _topological_sort(
    graph: Dict[str, Set[str]],
    stage_map: Dict[str, PipelineStage]
) -> List[PipelineStage]:
    """
    Ordenamiento topológico de etapas.
    
    Args:
        graph: Grafo de dependencias
        stage_map: Mapa de nombres a etapas
        
    Returns:
        Etapas ordenadas
        
    Raises:
        ValueError: Si hay dependencia circular
    """
    visited: Set[str] = set()
    temp_visited: Set[str] = set()
    ordered: List[PipelineStage] = []
    
    def visit(stage_name: str) -> None:
        if stage_name in temp_visited:
            cycle = _detect_circular_dependency(
                graph, stage_name, set(), set()
            )
            cycle_str = " -> ".join(cycle) if cycle else stage_name
            raise ValueError(f"Circular dependency detected: {cycle_str}")
        
        if stage_name in visited:
            return
        
        temp_visited.add(stage_name)
        
        for dep_name in graph.get(stage_name, set()):
            if dep_name in stage_map:
                visit(dep_name)
        
        temp_visited.remove(stage_name)
        visited.add(stage_name)
        ordered.append(stage_map[stage_name])
    
    for stage_name in stage_map.keys():
        if stage_name not in visited:
            visit(stage_name)
    
    return ordered


class DependencyResolver:
    """
    Resolvedor de dependencias para pipelines.
    Optimizado con funciones puras y mejor manejo de errores.
    """
    
    def __init__(self) -> None:
        """Inicializar resolvedor."""
        self._dependencies: Dict[str, List[StageDependency]] = {}
        self._graph = DependencyGraph()
    
    def add_dependency(self, dependency: StageDependency) -> None:
        """
        Agregar dependencia.
        
        Args:
            dependency: Dependencia a agregar
        """
        if not dependency.stage_name:
            raise ValueError("Stage name cannot be empty")
        
        if dependency.stage_name not in self._dependencies:
            self._dependencies[dependency.stage_name] = []
        self._dependencies[dependency.stage_name].append(dependency)
        
        if dependency.dependency_type == DependencyType.REQUIRES:
            if dependency.depends_on:
                self._graph.add_dependency(
                    dependency.stage_name,
                    dependency.depends_on
                )
        elif dependency.dependency_type == DependencyType.CONFLICTS:
            if dependency.conflicts_with:
                self._graph.add_conflict(
                    dependency.stage_name,
                    dependency.conflicts_with
                )
                logger.warning(
                    f"Conflict detected: {dependency.stage_name} <-> "
                    f"{dependency.conflicts_with}"
                )
    
    def resolve_order(
        self,
        stages: List[PipelineStage]
    ) -> List[PipelineStage]:
        """
        Resolver orden de ejecución basado en dependencias.
        
        Args:
            stages: Lista de etapas
            
        Returns:
            Etapas ordenadas
            
        Raises:
            ValueError: Si hay dependencia circular o conflicto
        """
        if not stages:
            return []
        
        stage_map = {stage.get_name(): stage for stage in stages}
        
        graph: Dict[str, Set[str]] = {
            name: self._graph.get_dependencies(name)
            for name in stage_map.keys()
        }
        
        for stage_name, deps in self._dependencies.items():
            if stage_name not in stage_map:
                continue
            
            for dep in deps:
                if dep.dependency_type == DependencyType.REQUIRES:
                    if dep.depends_on and dep.depends_on in stage_map:
                        if dep.depends_on not in graph[stage_name]:
                            graph[stage_name].add(dep.depends_on)
        
        return _topological_sort(graph, stage_map)
    
    def validate_dependencies(
        self,
        stages: List[PipelineStage]
    ) -> Tuple[bool, List[str]]:
        """
        Validar dependencias.
        
        Args:
            stages: Lista de etapas
            
        Returns:
            Tupla (es_válido, lista_de_errores)
        """
        if not stages:
            return True, []
        
        errors: List[str] = []
        stage_names = {stage.get_name() for stage in stages}
        
        for stage_name, deps in self._dependencies.items():
            if stage_name not in stage_names:
                continue
            
            for dep in deps:
                if dep.dependency_type == DependencyType.REQUIRES:
                    if dep.depends_on and dep.depends_on not in stage_names:
                        errors.append(
                            f"Stage '{stage_name}' requires '{dep.depends_on}' "
                            f"which is not in the pipeline"
                        )
                elif dep.dependency_type == DependencyType.CONFLICTS:
                    if dep.conflicts_with and dep.conflicts_with in stage_names:
                        errors.append(
                            f"Stage '{stage_name}' conflicts with "
                            f"'{dep.conflicts_with}'"
                        )
        
        return len(errors) == 0, errors


class DependencyAwarePipeline:
    """
    Pipeline con soporte para dependencias.
    Optimizado con mejor validación y manejo de errores.
    """
    
    def __init__(
        self,
        name: str = "pipeline",
        dependency_resolver: Optional[DependencyResolver] = None,
        **kwargs: Any
    ) -> None:
        """
        Inicializar pipeline con dependencias.
        
        Args:
            name: Nombre del pipeline
            dependency_resolver: Resolvedor de dependencias
            **kwargs: Argumentos adicionales
        """
        from .pipeline import Pipeline
        
        self._pipeline = Pipeline(name, **kwargs)
        self.dependency_resolver = dependency_resolver or DependencyResolver()
        self._original_stage_order: List[PipelineStage] = []
    
    def add_stage(
        self,
        stage: PipelineStage,
        requires: Optional[List[str]] = None,
        provides: Optional[str] = None,
        conflicts_with: Optional[List[str]] = None
    ) -> 'DependencyAwarePipeline':
        """
        Agregar etapa con dependencias.
        
        Args:
            stage: Etapa
            requires: Lista de etapas requeridas
            provides: Nombre del recurso que proporciona
            conflicts_with: Lista de etapas con las que conflictúa
            
        Returns:
            Self para chaining
        """
        if not stage:
            raise ValueError("Stage cannot be None")
        
        self._original_stage_order.append(stage)
        
        if requires:
            for req in requires:
                if not req:
                    continue
                dep = StageDependency(
                    stage_name=stage.get_name(),
                    dependency_type=DependencyType.REQUIRES,
                    depends_on=req
                )
                self.dependency_resolver.add_dependency(dep)
        
        if provides:
            dep = StageDependency(
                stage_name=stage.get_name(),
                dependency_type=DependencyType.PROVIDES,
                provides=provides
            )
            self.dependency_resolver.add_dependency(dep)
        
        if conflicts_with:
            for conflict in conflicts_with:
                if not conflict:
                    continue
                dep = StageDependency(
                    stage_name=stage.get_name(),
                    dependency_type=DependencyType.CONFLICTS,
                    conflicts_with=conflict
                )
                self.dependency_resolver.add_dependency(dep)
        
        self._pipeline.stages = self.dependency_resolver.resolve_order(
            self._original_stage_order
        )
        
        return self
    
    def validate(self) -> Tuple[bool, List[str]]:
        """
        Validar dependencias.
        
        Returns:
            Tupla (es_válido, lista_de_errores)
        """
        return self.dependency_resolver.validate_dependencies(
            self._pipeline.stages
        )
    
    @property
    def stages(self) -> List[PipelineStage]:
        """Obtener etapas del pipeline."""
        return self._pipeline.stages
