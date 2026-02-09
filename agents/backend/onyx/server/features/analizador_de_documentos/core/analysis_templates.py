"""
Sistema de Plantillas para Análisis
====================================

Sistema para crear y usar plantillas personalizadas de análisis.
"""

import logging
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path

from .document_analyzer import DocumentAnalyzer, AnalysisTask

logger = logging.getLogger(__name__)


@dataclass
class AnalysisTemplate:
    """Plantilla de análisis"""
    name: str
    description: str
    tasks: List[str]
    parameters: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    created_at: str = None
    updated_at: str = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
        if self.updated_at is None:
            self.updated_at = datetime.now().isoformat()


class TemplateManager:
    """
    Gestor de plantillas de análisis
    
    Permite crear, guardar y usar plantillas personalizadas
    para análisis de documentos.
    """
    
    def __init__(self, templates_dir: Optional[str] = None):
        """
        Inicializar gestor de plantillas
        
        Args:
            templates_dir: Directorio para guardar plantillas
        """
        if templates_dir is None:
            templates_dir = os.path.join(
                Path(__file__).parent.parent.parent,
                "templates"
            )
        
        self.templates_dir = Path(templates_dir)
        self.templates_dir.mkdir(parents=True, exist_ok=True)
        
        self.templates: Dict[str, AnalysisTemplate] = {}
        self._load_templates()
        self._register_default_templates()
        
        logger.info(f"TemplateManager inicializado con {len(self.templates)} plantillas")
    
    def _load_templates(self):
        """Cargar plantillas desde disco"""
        for template_file in self.templates_dir.glob("*.json"):
            try:
                with open(template_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    template = AnalysisTemplate(**data)
                    self.templates[template.name] = template
            except Exception as e:
                logger.warning(f"Error cargando plantilla {template_file}: {e}")
    
    def _register_default_templates(self):
        """Registrar plantillas por defecto"""
        default_templates = [
            AnalysisTemplate(
                name="basic_analysis",
                description="Análisis básico: clasificación y resumen",
                tasks=["classification", "summarization"],
                parameters={"summary_max_length": 150}
            ),
            AnalysisTemplate(
                name="comprehensive_analysis",
                description="Análisis completo: todas las tareas",
                tasks=["classification", "summarization", "keyword_extraction", "sentiment", "entity_recognition", "topic_modeling"],
                parameters={}
            ),
            AnalysisTemplate(
                name="sentiment_focus",
                description="Enfoque en análisis de sentimiento",
                tasks=["sentiment", "classification"],
                parameters={}
            ),
            AnalysisTemplate(
                name="content_extraction",
                description="Extracción de contenido: keywords y entidades",
                tasks=["keyword_extraction", "entity_recognition"],
                parameters={"keywords_top_k": 20}
            ),
            AnalysisTemplate(
                name="topic_analysis",
                description="Análisis de temas y modelado",
                tasks=["topic_modeling", "keyword_extraction"],
                parameters={"topics_num_topics": 5}
            )
        ]
        
        for template in default_templates:
            if template.name not in self.templates:
                self.templates[template.name] = template
                self._save_template(template)
    
    def create_template(
        self,
        name: str,
        description: str,
        tasks: List[str],
        parameters: Optional[Dict[str, Any]] = None
    ) -> AnalysisTemplate:
        """
        Crear nueva plantilla
        
        Args:
            name: Nombre de la plantilla
            description: Descripción
            tasks: Lista de tareas
            parameters: Parámetros personalizados
        
        Returns:
            AnalysisTemplate creada
        """
        template = AnalysisTemplate(
            name=name,
            description=description,
            tasks=tasks,
            parameters=parameters or {}
        )
        
        self.templates[name] = template
        self._save_template(template)
        logger.info(f"Plantilla creada: {name}")
        
        return template
    
    def get_template(self, name: str) -> Optional[AnalysisTemplate]:
        """Obtener plantilla por nombre"""
        return self.templates.get(name)
    
    def list_templates(self) -> List[Dict[str, Any]]:
        """Listar todas las plantillas"""
        return [
            {
                "name": t.name,
                "description": t.description,
                "tasks": t.tasks,
                "enabled": t.enabled,
                "created_at": t.created_at
            }
            for t in self.templates.values()
        ]
    
    def delete_template(self, name: str):
        """Eliminar plantilla"""
        if name in self.templates:
            del self.templates[name]
            template_file = self.templates_dir / f"{name}.json"
            if template_file.exists():
                template_file.unlink()
            logger.info(f"Plantilla eliminada: {name}")
    
    def _save_template(self, template: AnalysisTemplate):
        """Guardar plantilla en disco"""
        template_file = self.templates_dir / f"{template.name}.json"
        with open(template_file, "w", encoding="utf-8") as f:
            json.dump(asdict(template), f, indent=2, ensure_ascii=False)
    
    async def apply_template(
        self,
        template_name: str,
        content: str,
        analyzer: DocumentAnalyzer
    ) -> Dict[str, Any]:
        """
        Aplicar plantilla a un documento
        
        Args:
            template_name: Nombre de la plantilla
            content: Contenido del documento
            analyzer: Instancia de DocumentAnalyzer
        
        Returns:
            Resultado del análisis
        """
        template = self.get_template(template_name)
        if not template:
            raise ValueError(f"Plantilla no encontrada: {template_name}")
        
        if not template.enabled:
            raise ValueError(f"Plantilla deshabilitada: {template_name}")
        
        # Convertir tasks a AnalysisTask
        tasks = [AnalysisTask(task) for task in template.tasks]
        
        # Aplicar análisis con parámetros de la plantilla
        result = await analyzer.analyze_document(
            document_content=content,
            tasks=tasks,
            **template.parameters
        )
        
        return result.__dict__ if hasattr(result, "__dict__") else result


# Instancia global
_template_manager: Optional[TemplateManager] = None


def get_template_manager() -> TemplateManager:
    """Obtener instancia global del gestor de plantillas"""
    global _template_manager
    if _template_manager is None:
        _template_manager = TemplateManager()
    return _template_manager


# Importar os para path handling
import os
















