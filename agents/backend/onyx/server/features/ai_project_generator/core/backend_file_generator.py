"""
Backend File Generator - Generador de archivos backend
======================================================

Módulo especializado para generar archivos específicos del backend,
separando la lógica de generación de archivos del flujo principal.
Optimizado siguiendo mejores prácticas de Python y FastAPI.
"""

from pathlib import Path
from typing import Dict, Any

from .backend_templates import BackendTemplates
from .library_manager import LibraryManager
from .shared_utils import get_logger, ensure_directory, safe_write_file
from .project_validators import validate_project_dir, validate_keywords, validate_project_info

logger = get_logger(__name__)

_BACKEND_DIRECTORIES: tuple[str, ...] = (
    "app",
    "app/api",
    "app/api/endpoints",
    "app/core",
    "app/models",
    "app/services",
    "app/utils",
    "app/data",
    "app/training",
    "tests",
)

_INIT_FILES: tuple[tuple[str, str], ...] = (
    ("app", '"""App package"""'),
    ("app/models", '"""Data models"""'),
    ("app/utils", '"""Utility functions"""'),
)


class BackendFileGenerator:
    """Generador de archivos para backend."""
    
    def __init__(self) -> None:
        """Inicializa el generador de archivos backend."""
        self.templates = BackendTemplates()
        self.lib_manager = LibraryManager()
    
    def create_directory_structure(self, project_dir: Path) -> None:
        """
        Crea la estructura de directorios del proyecto.
        
        Args:
            project_dir: Directorio del proyecto
            
        Raises:
            ValueError: Si el directorio es inválido
        """
        validate_project_dir(project_dir)
        
        for directory in _BACKEND_DIRECTORIES:
            ensure_directory(project_dir / directory)
    
    def generate_main_py(
        self,
        project_dir: Path,
        project_info: Dict[str, Any],
        description: str,
        keywords: Dict[str, Any],
    ) -> None:
        """
        Genera main.py.
        
        Args:
            project_dir: Directorio del proyecto
            project_info: Información del proyecto
            description: Descripción del proyecto
            keywords: Keywords del proyecto
            
        Raises:
            ValueError: Si los parámetros son inválidos
        """
        validate_project_dir(project_dir)
        validate_project_info(project_info)
        validate_keywords(keywords)
        
        if not description:
            raise ValueError("description cannot be empty")
        
        content = self.templates.main_py(project_info, description, keywords)
        safe_write_file(project_dir / "main.py", content, logger=logger)
        logger.debug("Generated main.py")
    
    def generate_config_files(
        self,
        project_dir: Path,
        keywords: Dict[str, Any],
    ) -> None:
        """
        Genera archivos de configuración.
        
        Args:
            project_dir: Directorio del proyecto
            keywords: Keywords del proyecto
            
        Raises:
            ValueError: Si los parámetros son inválidos
        """
        validate_project_dir(project_dir)
        validate_keywords(keywords)
        
        config_content = self.templates.config_py(keywords)
        safe_write_file(
            project_dir / "app" / "core" / "config.py",
            config_content,
            logger=logger
        )
        safe_write_file(
            project_dir / "app" / "core" / "__init__.py",
            '"""Core modules"""',
            logger=logger
        )
        logger.debug("Generated config files")
    
    def generate_api_files(
        self,
        project_dir: Path,
        project_info: Dict[str, Any],
        description: str,
        keywords: Dict[str, Any],
    ) -> None:
        """
        Genera archivos de API.
        
        Args:
            project_dir: Directorio del proyecto
            project_info: Información del proyecto
            description: Descripción del proyecto
            keywords: Keywords del proyecto
            
        Raises:
            ValueError: Si los parámetros son inválidos
        """
        validate_project_dir(project_dir)
        validate_project_info(project_info)
        validate_keywords(keywords)
        
        if not description:
            raise ValueError("description cannot be empty")
        
        api_init_content = self.templates.api_init_py(keywords)
        safe_write_file(
            project_dir / "app" / "api" / "__init__.py",
            api_init_content,
            logger=logger
        )
        
        ai_endpoint_content = self.templates.ai_endpoint_py(
            project_info, description, keywords
        )
        safe_write_file(
            project_dir / "app" / "api" / "endpoints" / "ai.py",
            ai_endpoint_content,
            logger=logger
        )
        safe_write_file(
            project_dir / "app" / "api" / "endpoints" / "__init__.py",
            '"""API endpoints"""',
            logger=logger
        )
        logger.debug("Generated API files")
    
    def generate_service_files(
        self,
        project_dir: Path,
        keywords: Dict[str, Any],
        project_info: Dict[str, Any],
    ) -> None:
        """
        Genera archivos de servicios.
        
        Args:
            project_dir: Directorio del proyecto
            keywords: Keywords del proyecto
            project_info: Información del proyecto
            
        Raises:
            ValueError: Si los parámetros son inválidos
        """
        validate_project_dir(project_dir)
        validate_keywords(keywords)
        _validate_project_info(project_info)
        
        ai_service_content = self.templates.ai_service_py(keywords, project_info)
        safe_write_file(
            project_dir / "app" / "services" / "ai_service.py",
            ai_service_content,
            logger=logger
        )
        safe_write_file(
            project_dir / "app" / "services" / "__init__.py",
            '"""Business logic services"""',
            logger=logger
        )
        logger.debug("Generated service files")
    
    def generate_init_files(self, project_dir: Path) -> None:
        """
        Genera archivos __init__.py básicos.
        
        Args:
            project_dir: Directorio del proyecto
            
        Raises:
            ValueError: Si el directorio es inválido
        """
        validate_project_dir(project_dir)
        
        for path, content in _INIT_FILES:
            safe_write_file(
                project_dir / path / "__init__.py",
                content,
                logger=logger
            )
    
    def generate_requirements_txt(
        self,
        project_dir: Path,
        keywords: Dict[str, Any],
    ) -> None:
        """
        Genera requirements.txt.
        
        Args:
            project_dir: Directorio del proyecto
            keywords: Keywords del proyecto
            
        Raises:
            ValueError: Si los parámetros son inválidos
        """
        validate_project_dir(project_dir)
        validate_keywords(keywords)
        
        libraries = self.lib_manager.get_libraries_by_keywords(keywords)
        requirements_content = self.lib_manager.format_requirements(libraries)
        
        if keywords.get("is_deep_learning"):
            requirements_content += "\n\n# Optional but Recommended\n"
            requirements_content += "# lightning>=2.0.0  # PyTorch Lightning\n"
            requirements_content += "# flash-attn>=2.3.0  # Flash Attention (requires CUDA)\n"
            requirements_content += "# vllm>=0.2.0  # Fast LLM inference\n"
            requirements_content += "# onnx>=1.15.0  # ONNX export\n"
            requirements_content += "# onnxruntime>=1.16.0  # ONNX inference\n"
            requirements_content += "# tensorrt>=8.6.0  # NVIDIA TensorRT\n"
        
        safe_write_file(
            project_dir / "requirements.txt",
            requirements_content,
            logger=logger
        )
        logger.debug("Generated requirements.txt")
    
    def generate_env_example(
        self,
        project_dir: Path,
        keywords: Dict[str, Any],
    ) -> None:
        """
        Genera .env.example.
        
        Args:
            project_dir: Directorio del proyecto
            keywords: Keywords del proyecto
            
        Raises:
            ValueError: Si los parámetros son inválidos
        """
        validate_project_dir(project_dir)
        validate_keywords(keywords)
        
        content = self.templates.env_example(keywords)
        safe_write_file(project_dir / ".env.example", content, logger=logger)
        logger.debug("Generated .env.example")
    
    def generate_dockerfile(self, project_dir: Path) -> None:
        """
        Genera Dockerfile.
        
        Args:
            project_dir: Directorio del proyecto
            
        Raises:
            ValueError: Si el directorio es inválido
        """
        validate_project_dir(project_dir)
        
        content = self.templates.dockerfile()
        safe_write_file(project_dir / "Dockerfile", content, logger=logger)
        logger.debug("Generated Dockerfile")
    
    def generate_readme(
        self,
        project_dir: Path,
        project_info: Dict[str, Any],
        description: str,
        keywords: Dict[str, Any],
    ) -> None:
        """
        Genera README.md.
        
        Args:
            project_dir: Directorio del proyecto
            project_info: Información del proyecto
            description: Descripción del proyecto
            keywords: Keywords del proyecto
            
        Raises:
            ValueError: Si los parámetros son inválidos
        """
        validate_project_dir(project_dir)
        validate_project_info(project_info)
        validate_keywords(keywords)
        
        if not description:
            raise ValueError("description cannot be empty")
        
        readme_lines = [
            f"# {project_info['name'].replace('_', ' ').title()} - Backend",
            "",
            f"Backend API para {description}",
            "",
            "## 🚀 Instalación",
            "",
            "```bash",
            "python -m venv venv",
            "source venv/bin/activate",
            "pip install -r requirements.txt",
            "cp .env.example .env",
            "```",
            "",
            "## 🏃 Ejecutar",
            "",
            "```bash",
            "uvicorn main:app --reload",
            "```",
            "",
            "## 📚 Documentación",
            "",
            "- Swagger UI: http://localhost:8000/docs",
            "- ReDoc: http://localhost:8000/redoc",
            "",
            "## 🧪 Testing",
            "",
            "```bash",
            "pytest",
            "```",
        ]
        
        if keywords.get("is_deep_learning") or keywords.get("requires_pytorch"):
            readme_lines.extend([
                "",
                "## 🤖 Deep Learning",
                "",
                "### Entrenar Modelo",
                "",
                "```bash",
                "python app/training/train.py",
                "```",
            ])
        
        if keywords.get("requires_gradio"):
            readme_lines.extend([
                "",
                "## 🎨 Interfaz Gradio",
                "",
                "```bash",
                "python -m app.services.gradio_interface",
                "```",
            ])
        
        readme_content = "\n".join(readme_lines)
        safe_write_file(project_dir / "README.md", readme_content, logger=logger)
        logger.debug("Generated README.md")
