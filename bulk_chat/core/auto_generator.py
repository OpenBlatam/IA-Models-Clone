"""
Auto Generator - Generador Automático
======================================

Sistema de generación automática de código, configuración y documentación basado en plantillas y patrones.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from collections import defaultdict
import json

logger = logging.getLogger(__name__)


class GenerationType(Enum):
    """Tipo de generación."""
    CODE = "code"
    CONFIG = "config"
    DOCUMENTATION = "documentation"
    TEST = "test"
    API = "api"
    SCHEMA = "schema"


@dataclass
class GenerationTemplate:
    """Plantilla de generación."""
    template_id: str
    generation_type: GenerationType
    name: str
    template: str
    variables: List[str] = field(default_factory=list)
    examples: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GeneratedArtifact:
    """Artefacto generado."""
    artifact_id: str
    generation_type: GenerationType
    template_id: str
    content: str
    variables_used: Dict[str, Any] = field(default_factory=dict)
    generated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class AutoGenerator:
    """Generador automático."""
    
    def __init__(self):
        self.templates: Dict[str, GenerationTemplate] = {}
        self.generated_artifacts: Dict[str, GeneratedArtifact] = {}
        self.generation_history: List[Dict[str, Any]] = []
        self._lock = asyncio.Lock()
    
    def register_template(
        self,
        template_id: str,
        generation_type: GenerationType,
        name: str,
        template: str,
        variables: Optional[List[str]] = None,
        examples: Optional[List[Dict[str, Any]]] = None,
    ):
        """Registrar plantilla de generación."""
        tmpl = GenerationTemplate(
            template_id=template_id,
            generation_type=generation_type,
            name=name,
            template=template,
            variables=variables or [],
            examples=examples or [],
        )
        
        self.templates[template_id] = tmpl
        logger.info(f"Registered template: {template_id} - {name}")
    
    async def generate(
        self,
        template_id: str,
        variables: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Generar artefacto desde plantilla.
        
        Args:
            template_id: ID de plantilla
            variables: Variables para la plantilla
            metadata: Metadatos adicionales
        
        Returns:
            Contenido generado
        """
        template = self.templates.get(template_id)
        if not template:
            raise ValueError(f"Template not found: {template_id}")
        
        # Renderizar plantilla (simplificado - usar Jinja2 o similar en producción)
        content = template.template
        
        for var_name, var_value in variables.items():
            placeholder = f"{{{{{var_name}}}}}"
            content = content.replace(placeholder, str(var_value))
        
        artifact_id = f"gen_{template_id}_{datetime.now().timestamp()}"
        
        artifact = GeneratedArtifact(
            artifact_id=artifact_id,
            generation_type=template.generation_type,
            template_id=template_id,
            content=content,
            variables_used=variables,
            metadata=metadata or {},
        )
        
        async with self._lock:
            self.generated_artifacts[artifact_id] = artifact
            self.generation_history.append({
                "artifact_id": artifact_id,
                "template_id": template_id,
                "generation_type": template.generation_type.value,
                "timestamp": datetime.now(),
            })
        
        logger.info(f"Generated artifact: {artifact_id} from template {template_id}")
        return content
    
    async def generate_batch(
        self,
        template_id: str,
        variables_list: List[Dict[str, Any]],
    ) -> List[str]:
        """Generar múltiples artefactos."""
        results = []
        
        for variables in variables_list:
            content = await self.generate(template_id, variables)
            results.append(content)
        
        return results
    
    def get_template(self, template_id: str) -> Optional[Dict[str, Any]]:
        """Obtener plantilla."""
        template = self.templates.get(template_id)
        if not template:
            return None
        
        return {
            "template_id": template.template_id,
            "generation_type": template.generation_type.value,
            "name": template.name,
            "variables": template.variables,
            "examples": template.examples,
            "metadata": template.metadata,
        }
    
    def list_templates(
        self,
        generation_type: Optional[GenerationType] = None,
    ) -> List[Dict[str, Any]]:
        """Listar plantillas."""
        templates = list(self.templates.values())
        
        if generation_type:
            templates = [t for t in templates if t.generation_type == generation_type]
        
        return [
            {
                "template_id": t.template_id,
                "generation_type": t.generation_type.value,
                "name": t.name,
                "variables": t.variables,
            }
            for t in templates
        ]
    
    def get_generation_history(
        self,
        template_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Obtener historial de generaciones."""
        history = self.generation_history
        
        if template_id:
            history = [h for h in history if h["template_id"] == template_id]
        
        return history[-limit:]
    
    def get_generation_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas de generación."""
        by_type: Dict[str, int] = defaultdict(int)
        
        for artifact in self.generated_artifacts.values():
            by_type[artifact.generation_type.value] += 1
        
        return {
            "total_templates": len(self.templates),
            "total_generated": len(self.generated_artifacts),
            "generated_by_type": dict(by_type),
            "total_generations": len(self.generation_history),
        }
















