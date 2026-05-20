"""
Frontend Generator - Generador de Frontend
==========================================

Genera automáticamente la estructura completa de frontend para proyectos de IA.
Optimizado siguiendo mejores prácticas de Python y FastAPI.
"""

from pathlib import Path
from typing import Dict, Any, Optional

from .frontend_file_generator import FrontendFileGenerator
from .constants import FrameworkType
from .shared_utils import get_logger

logger = get_logger(__name__)


def _validate_project_info(project_info: Dict[str, Any]) -> None:
    """
    Validar información del proyecto (función pura).
    
    Args:
        project_info: Información del proyecto
        
    Raises:
        ValueError: Si la información es inválida
    """
    if not project_info:
        raise ValueError("project_info cannot be empty")
    
    if 'name' not in project_info:
        raise ValueError("project_info must contain 'name'")
    
    if not project_info['name']:
        raise ValueError("project name cannot be empty")


class FrontendGenerator:
    """
    Generador de estructura de frontend.
    Optimizado con funciones puras y mejor manejo de errores.
    """
    
    def __init__(self, framework: str = FrameworkType.REACT) -> None:
        """
        Inicializa el generador de frontend.

        Args:
            framework: Framework a usar (react, vue, nextjs)
            
        Raises:
            ValueError: Si el framework no es soportado
        """
        if framework not in [FrameworkType.REACT, FrameworkType.VUE, FrameworkType.NEXTJS]:
            raise ValueError(f"Unsupported framework: {framework}")
        
        self.framework = framework
        self.file_generator = FrontendFileGenerator()
    
    async def generate(
        self,
        project_dir: Path,
        description: str,
        keywords: Dict[str, Any],
        project_info: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Genera la estructura completa del frontend.

        Args:
            project_dir: Directorio donde generar el frontend
            description: Descripción del proyecto
            keywords: Keywords extraídos
            project_info: Información del proyecto

        Returns:
            Información del frontend generado
            
        Raises:
            ValueError: Si los parámetros son inválidos
            IOError: Si no se puede crear la estructura
        """
        if not project_dir:
            raise ValueError("project_dir cannot be None")
        
        if not description:
            raise ValueError("description cannot be empty")
        
        _validate_project_info(project_info)
        
        project_dir.mkdir(parents=True, exist_ok=True)
        
        if self.framework == FrameworkType.REACT:
            return await self._generate_react(project_dir, description, keywords, project_info)
        
        raise ValueError(f"Framework {self.framework} not yet supported")
    
    async def _generate_react(
        self,
        project_dir: Path,
        description: str,
        keywords: Dict[str, Any],
        project_info: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Genera estructura React.
        
        Args:
            project_dir: Directorio del proyecto
            description: Descripción del proyecto
            keywords: Keywords extraídos
            project_info: Información del proyecto
            
        Returns:
            Información del frontend generado
        """
        logger.info("Generando estructura React...")
        
        project_name = project_info['name']
        version = project_info.get('version', '1.0.0')
        
        self.file_generator.create_directory_structure(project_dir)
        self.file_generator.generate_config_files(project_dir, project_name, version, description)
        self.file_generator.generate_html_and_css(project_dir, project_name)
        self.file_generator.generate_react_files(project_dir, project_name, description)
        self.file_generator.generate_utility_files(project_dir)
        self.file_generator.generate_readme(project_dir, project_name, description)
        
        return {
            "framework": "React",
            "port": 3000,
            "status": "generated",
            "structure": {
                "src": "src/",
                "components": "src/components/",
                "pages": "src/pages/",
                "services": "src/services/",
            }
        }
