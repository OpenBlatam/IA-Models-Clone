"""
Frontend File Generator - Generador de archivos frontend
========================================================

Módulo especializado para generar archivos específicos del frontend,
separando la lógica de generación de archivos del flujo principal.
Optimizado siguiendo mejores prácticas de Python y FastAPI.
"""

import json
from pathlib import Path
from typing import Dict, Any

from .frontend_templates import FrontendTemplates
from .shared_utils import get_logger, ensure_directory, safe_write_file
from .project_validators import validate_project_dir, validate_project_name

logger = get_logger(__name__)

_FRONTEND_DIRECTORIES: tuple[str, ...] = (
    "src",
    "src/components",
    "src/pages",
    "src/services",
    "src/utils",
    "src/hooks",
    "public"
)


class FrontendFileGenerator:
    """Generador de archivos para frontend."""
    
    def __init__(self) -> None:
        """Inicializa el generador de archivos frontend."""
        self.templates = FrontendTemplates()
    
    def create_directory_structure(self, project_dir: Path) -> None:
        """
        Crea la estructura de directorios del proyecto.
        
        Args:
            project_dir: Directorio del proyecto
            
        Raises:
            ValueError: Si el directorio es inválido
        """
        validate_project_dir(project_dir)
        
        for directory in _FRONTEND_DIRECTORIES:
            ensure_directory(project_dir / directory)
        
        logger.debug("Created frontend directory structure")
    
    def generate_config_files(
        self,
        project_dir: Path,
        project_name: str,
        version: str,
        description: str,
    ) -> None:
        """
        Genera archivos de configuración.
        
        Args:
            project_dir: Directorio del proyecto
            project_name: Nombre del proyecto
            version: Versión del proyecto
            description: Descripción del proyecto
            
        Raises:
            ValueError: Si los parámetros son inválidos
        """
        validate_project_dir(project_dir)
        validate_project_name(project_name)
        
        if not version:
            raise ValueError("version cannot be empty")
        
        if not description:
            raise ValueError("description cannot be empty")
        
        package_json = self.templates.package_json(project_name, version, description)
        safe_write_file(
            project_dir / "package.json",
            json.dumps(package_json, indent=2),
            logger=logger
        )
        
        safe_write_file(
            project_dir / "vite.config.ts",
            self.templates.vite_config(),
            logger=logger
        )
        
        safe_write_file(
            project_dir / "tsconfig.json",
            json.dumps(self.templates.tsconfig(), indent=2),
            logger=logger
        )
        
        safe_write_file(
            project_dir / "tsconfig.node.json",
            json.dumps(self.templates.tsconfig_node(), indent=2),
            logger=logger
        )
        
        safe_write_file(
            project_dir / "tailwind.config.js",
            self.templates.tailwind_config(),
            logger=logger
        )
        
        safe_write_file(
            project_dir / "postcss.config.js",
            self.templates.postcss_config(),
            logger=logger
        )
        
        logger.debug("Generated frontend config files")
    
    def generate_html_and_css(
        self,
        project_dir: Path,
        project_name: str,
    ) -> None:
        """
        Genera archivos HTML y CSS.
        
        Args:
            project_dir: Directorio del proyecto
            project_name: Nombre del proyecto
            
        Raises:
            ValueError: Si los parámetros son inválidos
        """
        validate_project_dir(project_dir)
        validate_project_name(project_name)
        
        safe_write_file(
            project_dir / "index.html",
            self.templates.index_html(project_name),
            logger=logger
        )
        
        safe_write_file(
            project_dir / "src" / "index.css",
            self.templates.index_css(),
            logger=logger
        )
        
        safe_write_file(
            project_dir / "src" / "App.css",
            self.templates.app_css(),
            logger=logger
        )
        
        logger.debug("Generated HTML and CSS files")
    
    def generate_react_files(
        self,
        project_dir: Path,
        project_name: str,
        description: str,
    ) -> None:
        """
        Genera archivos React/TypeScript.
        
        Args:
            project_dir: Directorio del proyecto
            project_name: Nombre del proyecto
            description: Descripción del proyecto
            
        Raises:
            ValueError: Si los parámetros son inválidos
        """
        validate_project_dir(project_dir)
        validate_project_name(project_name)
        
        if not description:
            raise ValueError("description cannot be empty")
        
        safe_write_file(
            project_dir / "src" / "main.tsx",
            self.templates.main_tsx(),
            logger=logger
        )
        
        safe_write_file(
            project_dir / "src" / "App.tsx",
            self.templates.app_tsx(project_name),
            logger=logger
        )
        
        safe_write_file(
            project_dir / "src" / "pages" / "Home.tsx",
            self.templates.home_tsx(project_name, description),
            logger=logger
        )
        
        safe_write_file(
            project_dir / "src" / "pages" / "AIProcessor.tsx",
            self.templates.ai_processor_tsx(),
            logger=logger
        )
        
        safe_write_file(
            project_dir / "src" / "services" / "aiService.ts",
            self.templates.ai_service_ts(),
            logger=logger
        )
        
        logger.debug("Generated React/TypeScript files")
    
    def generate_utility_files(self, project_dir: Path) -> None:
        """
        Genera archivos de utilidades.
        
        Args:
            project_dir: Directorio del proyecto
            
        Raises:
            ValueError: Si el directorio es inválido
        """
        validate_project_dir(project_dir)
        
        safe_write_file(
            project_dir / ".gitignore",
            self.templates.gitignore(),
            logger=logger
        )
        
        safe_write_file(
            project_dir / ".env.example",
            self.templates.env_example(),
            logger=logger
        )
        
        safe_write_file(
            project_dir / "Dockerfile",
            self.templates.dockerfile(),
            logger=logger
        )
        
        logger.debug("Generated utility files")
    
    def generate_readme(
        self,
        project_dir: Path,
        project_name: str,
        description: str,
    ) -> None:
        """
        Genera README.md.
        
        Args:
            project_dir: Directorio del proyecto
            project_name: Nombre del proyecto
            description: Descripción del proyecto
            
        Raises:
            ValueError: Si los parámetros son inválidos
        """
        validate_project_dir(project_dir)
        validate_project_name(project_name)
        
        if not description:
            raise ValueError("description cannot be empty")
        
        content = self.templates.readme(project_name, description)
        safe_write_file(project_dir / "README.md", content, logger=logger)
        logger.debug("Generated README.md")
