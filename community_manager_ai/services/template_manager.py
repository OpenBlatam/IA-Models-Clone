"""
Template Manager - Gestor de Plantillas
========================================

Sistema para gestionar plantillas de contenido reutilizables.
"""

import logging
import json
import os
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from pathlib import Path
from collections import defaultdict
import uuid
import re

logger = logging.getLogger(__name__)


class TemplateManager:
    """Gestor de plantillas de contenido"""
    
    def __init__(self, storage_path: str = "data/templates"):
        """
        Inicializar gestor de plantillas
        
        Args:
            storage_path: Ruta para almacenar plantillas persistentes
        """
        self.templates: Dict[str, Dict[str, Any]] = {}
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.templates_file = self.storage_path / "templates.json"
        self._load_templates()
        logger.info(f"Template Manager inicializado (storage: {storage_path})")
    
    def _load_templates(self):
        """Cargar plantillas desde archivo"""
        if self.templates_file.exists():
            try:
                with open(self.templates_file, 'r', encoding='utf-8') as f:
                    self.templates = json.load(f)
                logger.debug(f"Plantillas cargadas desde {self.templates_file}")
            except Exception as e:
                logger.error(f"Error cargando plantillas: {e}")
                self.templates = {}
        else:
            logger.info("No se encontró archivo de plantillas, iniciando vacío")
    
    def _save_templates(self):
        """Guardar plantillas en archivo"""
        try:
            with open(self.templates_file, 'w', encoding='utf-8') as f:
                json.dump(self.templates, f, indent=2, default=str)
            logger.debug(f"Plantillas guardadas en {self.templates_file}")
        except Exception as e:
            logger.error(f"Error guardando plantillas: {e}")
    
    def create_template(
        self,
        name: str,
        content: str,
        platform: Optional[str] = None,
        variables: Optional[List[str]] = None,
        category: Optional[str] = None
    ) -> str:
        """
        Crear una nueva plantilla
        
        Args:
            name: Nombre de la plantilla
            content: Contenido de la plantilla (puede incluir {variables})
            platform: Plataforma objetivo (opcional)
            variables: Lista de variables disponibles
            category: Categoría de la plantilla
            
        Returns:
            ID de la plantilla creada
        """
        template_id = str(uuid.uuid4())
        
        # Extraer variables del contenido si no se proporcionan
        if variables is None:
            variables = re.findall(r'\{(\w+)\}', content)
        
        template_data = {
            "id": template_id,
            "name": name,
            "content": content,
            "platform": platform,
            "variables": list(set(variables)),  # Remover duplicados
            "category": category or "general",
            "created_at": datetime.now().isoformat(),
            "usage_count": 0
        }
        
        self.templates[template_id] = template_data
        self._save_templates()
        logger.info(f"Plantilla creada: {template_id} - {name}")
        
        return template_id
    
    def get_template(self, template_id: str) -> Optional[Dict[str, Any]]:
        """
        Obtener una plantilla
        
        Args:
            template_id: ID de la plantilla
            
        Returns:
            Datos de la plantilla o None
        """
        return self.templates.get(template_id)
    
    def render_template(
        self,
        template_id: str,
        variables: Dict[str, str]
    ) -> str:
        """
        Renderizar una plantilla con variables
        
        Args:
            template_id: ID de la plantilla
            variables: Dict con valores para las variables
            
        Returns:
            Contenido renderizado
        """
        template = self.get_template(template_id)
        if not template:
            raise ValueError(f"Plantilla {template_id} no encontrada")
        
        content = template["content"]
        
        # Reemplazar variables
        for var_name, var_value in variables.items():
            content = content.replace(f"{{{var_name}}}", str(var_value))
        
        # Incrementar contador de uso
        template["usage_count"] = template.get("usage_count", 0) + 1
        template["last_used_at"] = datetime.now().isoformat()
        self._save_templates()
        
        logger.info(f"Plantilla {template_id} renderizada")
        return content
    
    def search_templates(
        self,
        query: Optional[str] = None,
        platform: Optional[str] = None,
        category: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Buscar plantillas
        
        Args:
            query: Búsqueda por texto
            platform: Filtrar por plataforma
            category: Filtrar por categoría
            
        Returns:
            Lista de plantillas que coinciden
        """
        results = list(self.templates.values())
        
        if query:
            query_lower = query.lower()
            results = [
                t for t in results
                if query_lower in t.get("name", "").lower()
                or query_lower in t.get("content", "").lower()
            ]
        
        if platform:
            results = [t for t in results if t.get("platform") == platform]
        
        if category:
            results = [t for t in results if t.get("category") == category]
        
        return results
    
    def get_templates_by_platform(self, platform: str) -> List[Dict[str, Any]]:
        """
        Obtener plantillas para una plataforma
        
        Args:
            platform: Nombre de la plataforma
            
        Returns:
            Lista de plantillas
        """
        return [
            t for t in self.templates.values()
            if t.get("platform") == platform or t.get("platform") is None
        ]
    
    def delete_template(self, template_id: str) -> bool:
        """
        Eliminar una plantilla
        
        Args:
            template_id: ID de la plantilla
            
        Returns:
            True si se eliminó exitosamente
        """
        if template_id in self.templates:
            del self.templates[template_id]
            self._save_templates()
            logger.info(f"Plantilla eliminada: {template_id}")
            return True
        return False
    
    def get_categories(self) -> List[str]:
        """
        Obtener lista de categorías
        
        Returns:
            Lista de categorías
        """
        categories = set()
        for template in self.templates.values():
            category = template.get("category")
            if category:
                categories.add(category)
        return sorted(list(categories))
    
    def clone_template(self, template_id: str, new_name: str) -> Optional[str]:
        """
        Clonar una plantilla
        
        Args:
            template_id: ID de la plantilla a clonar
            new_name: Nombre para la nueva plantilla
            
        Returns:
            ID de la nueva plantilla o None si no se encontró
        """
        template = self.get_template(template_id)
        if not template:
            return None
        
        return self.create_template(
            name=new_name,
            content=template["content"],
            platform=template.get("platform"),
            variables=template.get("variables", []),
            category=template.get("category")
        )
    
    def update_template(
        self,
        template_id: str,
        updates: Dict[str, Any]
    ) -> bool:
        """
        Actualizar una plantilla
        
        Args:
            template_id: ID de la plantilla
            updates: Dict con campos a actualizar
            
        Returns:
            True si se actualizó exitosamente
        """
        if template_id not in self.templates:
            return False
        
        template = self.templates[template_id]
        
        for key, value in updates.items():
            if key in ["id", "created_at"]:
                continue
            template[key] = value
        
        template["updated_at"] = datetime.now().isoformat()
        
        if "content" in updates:
            variables = re.findall(r'\{(\w+)\}', updates["content"])
            template["variables"] = list(set(variables))
        
        logger.info(f"Plantilla actualizada: {template_id}")
        return True
    
    def get_most_used(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Obtener plantillas más usadas
        
        Args:
            limit: Número de plantillas a retornar
            
        Returns:
            Lista de plantillas ordenadas por uso
        """
        templates = list(self.templates.values())
        templates.sort(key=lambda x: x.get("usage_count", 0), reverse=True)
        return templates[:limit]
    
    def validate_template(self, template_id: str, variables: Dict[str, str]) -> Tuple[bool, Optional[str]]:
        """
        Validar que todas las variables estén proporcionadas
        
        Args:
            template_id: ID de la plantilla
            variables: Variables proporcionadas
            
        Returns:
            Tuple (is_valid, error_message)
        """
        template = self.get_template(template_id)
        if not template:
            return False, f"Plantilla {template_id} no encontrada"
        
        required_vars = set(template.get("variables", []))
        provided_vars = set(variables.keys())
        
        missing = required_vars - provided_vars
        if missing:
            return False, f"Variables faltantes: {', '.join(missing)}"
        
        return True, None
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Obtener estadísticas de plantillas
        
        Returns:
            Dict con estadísticas
        """
        total = len(self.templates)
        
        by_platform = defaultdict(int)
        by_category = defaultdict(int)
        total_usage = 0
        
        for template in self.templates.values():
            platform = template.get("platform") or "general"
            by_platform[platform] += 1
            
            category = template.get("category", "general")
            by_category[category] += 1
            
            total_usage += template.get("usage_count", 0)
        
        return {
            "total_templates": total,
            "by_platform": dict(by_platform),
            "by_category": dict(by_category),
            "total_usage": total_usage,
            "average_usage": total_usage / total if total > 0 else 0
        }



