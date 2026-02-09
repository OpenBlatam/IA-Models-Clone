"""
Template System - Sistema de templates para mejoras comunes
============================================================
"""

import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class TemplateSystem:
    """
    Sistema de templates para mejoras de código comunes.
    """
    
    def __init__(self, templates_dir: str = "data/templates"):
        """
        Inicializar sistema de templates.
        
        Args:
            templates_dir: Directorio de templates
        """
        self.templates_dir = Path(templates_dir)
        self.templates_dir.mkdir(parents=True, exist_ok=True)
        
        self.templates: Dict[str, Dict[str, Any]] = {}
        self._load_templates()
    
    def create_template(
        self,
        name: str,
        description: str,
        code_pattern: str,
        improvement_pattern: str,
        language: str = "python",
        tags: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Crea un nuevo template.
        
        Args:
            name: Nombre del template
            description: Descripción
            code_pattern: Patrón de código a mejorar
            improvement_pattern: Patrón de mejora
            language: Lenguaje de programación
            tags: Tags (opcional)
            
        Returns:
            Información del template creado
        """
        template = {
            "name": name,
            "description": description,
            "code_pattern": code_pattern,
            "improvement_pattern": improvement_pattern,
            "language": language,
            "tags": tags or [],
            "usage_count": 0,
            "created_at": datetime.now().isoformat()
        }
        
        self.templates[name] = template
        self._save_template(template)
        
        logger.info(f"Template creado: {name}")
        
        return template
    
    def find_matching_template(
        self,
        code: str,
        language: str = "python"
    ) -> Optional[Dict[str, Any]]:
        """
        Encuentra un template que coincida con el código.
        
        Args:
            code: Código a mejorar
            language: Lenguaje de programación
            
        Returns:
            Template coincidente o None
        """
        for template in self.templates.values():
            if template["language"] != language:
                continue
            
            # Verificar si el patrón coincide
            pattern = template["code_pattern"]
            if self._pattern_matches(code, pattern):
                template["usage_count"] += 1
                self._save_template(template)
                return template
        
        return None
    
    def apply_template(
        self,
        template_name: str,
        code: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Aplica un template a código.
        
        Args:
            template_name: Nombre del template
            code: Código a mejorar
            context: Contexto adicional (opcional)
            
        Returns:
            Código mejorado o None
        """
        if template_name not in self.templates:
            logger.warning(f"Template no encontrado: {template_name}")
            return None
        
        template = self.templates[template_name]
        improvement_pattern = template["improvement_pattern"]
        
        # Aplicar template (simplificado)
        # En producción, esto sería más sofisticado con regex o AST
        improved_code = code
        
        # Aquí se aplicaría la transformación del template
        # Por ahora, retornamos el código original
        
        return improved_code
    
    def list_templates(
        self,
        language: Optional[str] = None,
        tag: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Lista templates disponibles.
        
        Args:
            language: Filtrar por lenguaje (opcional)
            tag: Filtrar por tag (opcional)
            
        Returns:
            Lista de templates
        """
        templates = list(self.templates.values())
        
        if language:
            templates = [t for t in templates if t["language"] == language]
        
        if tag:
            templates = [t for t in templates if tag in t.get("tags", [])]
        
        # Ordenar por uso (más usados primero)
        templates.sort(key=lambda x: x.get("usage_count", 0), reverse=True)
        
        return templates
    
    def _pattern_matches(self, code: str, pattern: str) -> bool:
        """Verifica si un patrón coincide con código"""
        # Implementación básica - en producción sería más sofisticada
        import re
        
        # Convertir patrón a regex básico
        regex_pattern = pattern.replace("*", ".*")
        
        try:
            return bool(re.search(regex_pattern, code, re.MULTILINE | re.DOTALL))
        except Exception:
            return False
    
    def _load_templates(self):
        """Carga templates desde disco"""
        for template_file in self.templates_dir.glob("*.json"):
            try:
                with open(template_file, "r", encoding="utf-8") as f:
                    template = json.load(f)
                    self.templates[template["name"]] = template
            except Exception as e:
                logger.warning(f"Error cargando template {template_file}: {e}")
    
    def _save_template(self, template: Dict[str, Any]):
        """Guarda template en disco"""
        try:
            template_file = self.templates_dir / f"{template['name']}.json"
            with open(template_file, "w", encoding="utf-8") as f:
                json.dump(template, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error guardando template: {e}")

