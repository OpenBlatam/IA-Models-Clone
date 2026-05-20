"""
Dependency Analyzer
===================

Analizador de dependencias para proyectos generados.
"""

import logging
import ast
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class DependencyInfo:
    """Información de dependencia."""
    module: str
    import_type: str  # 'import' o 'from'
    alias: Optional[str] = None
    submodules: List[str] = field(default_factory=list)


@dataclass
class ProjectDependencies:
    """Dependencias del proyecto."""
    stdlib: Set[str] = field(default_factory=set)
    third_party: Set[str] = field(default_factory=set)
    local: Set[str] = field(default_factory=set)
    missing: Set[str] = field(default_factory=set)
    by_file: Dict[str, List[DependencyInfo]] = field(default_factory=lambda: defaultdict(list))


class DependencyAnalyzer:
    """
    Analizador de dependencias.
    """
    
    def __init__(self):
        """Inicializar analizador."""
        # Módulos estándar de Python
        self.stdlib_modules = {
            'os', 'sys', 'pathlib', 'typing', 'logging', 'json', 'datetime',
            'collections', 'itertools', 'functools', 'operator', 'math',
            'random', 'hashlib', 'base64', 'urllib', 'http', 'socket',
            'threading', 'multiprocessing', 'asyncio', 'concurrent',
            'dataclasses', 'enum', 'abc', 'contextlib', 'copy', 'pickle'
        }
    
    def analyze_file(self, file_path: Path) -> List[DependencyInfo]:
        """
        Analizar dependencias de un archivo.
        
        Args:
            file_path: Ruta del archivo
            
        Returns:
            Lista de dependencias
        """
        if not file_path.exists() or file_path.suffix != '.py':
            return []
        
        try:
            content = file_path.read_text(encoding='utf-8')
            tree = ast.parse(content)
            
            dependencies = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        dep = DependencyInfo(
                            module=alias.name.split('.')[0],
                            import_type='import',
                            alias=alias.asname
                        )
                        dependencies.append(dep)
                
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        dep = DependencyInfo(
                            module=node.module.split('.')[0],
                            import_type='from',
                            submodules=[name.name for name in node.names]
                        )
                        dependencies.append(dep)
            
            return dependencies
        except Exception as e:
            logger.warning(f"Error analizando {file_path}: {e}")
            return []
    
    def analyze_project(self, project_dir: Path) -> ProjectDependencies:
        """
        Analizar dependencias de todo el proyecto.
        
        Args:
            project_dir: Directorio del proyecto
            
        Returns:
            Dependencias del proyecto
        """
        deps = ProjectDependencies()
        
        # Analizar todos los archivos Python
        for py_file in project_dir.rglob("*.py"):
            file_deps = self.analyze_file(py_file)
            deps.by_file[str(py_file.relative_to(project_dir))] = file_deps
            
            for dep in file_deps:
                module = dep.module
                
                if module in self.stdlib_modules:
                    deps.stdlib.add(module)
                elif self._is_local_module(module, project_dir):
                    deps.local.add(module)
                else:
                    deps.third_party.add(module)
        
        # Verificar dependencias faltantes
        deps.missing = self._check_missing_dependencies(deps.third_party, project_dir)
        
        return deps
    
    def _is_local_module(self, module: str, project_dir: Path) -> bool:
        """Verificar si es módulo local."""
        # Buscar en el proyecto
        possible_paths = [
            project_dir / module,
            project_dir / f"{module}.py",
            project_dir / "app" / module,
            project_dir / "app" / f"{module}.py"
        ]
        return any(p.exists() for p in possible_paths)
    
    def _check_missing_dependencies(
        self,
        third_party: Set[str],
        project_dir: Path
    ) -> Set[str]:
        """Verificar dependencias faltantes."""
        missing = set()
        
        # Leer requirements.txt si existe
        requirements_file = project_dir / "requirements.txt"
        installed_packages = set()
        
        if requirements_file.exists():
            try:
                content = requirements_file.read_text(encoding='utf-8')
                for line in content.split('\n'):
                    line = line.strip()
                    if line and not line.startswith('#'):
                        # Extraer nombre del paquete
                        package = line.split('==')[0].split('>=')[0].split('<=')[0].split('>')[0].split('<')[0].strip()
                        installed_packages.add(package.lower())
            except Exception as e:
                logger.warning(f"Error leyendo requirements.txt: {e}")
        
        # Verificar cada dependencia de terceros
        for module in third_party:
            module_lower = module.lower()
            # Mapeo común de módulos a paquetes
            package_map = {
                'torch': 'torch',
                'tensorflow': 'tensorflow',
                'jax': 'jax',
                'transformers': 'transformers',
                'diffusers': 'diffusers',
                'numpy': 'numpy',
                'pandas': 'pandas',
                'matplotlib': 'matplotlib',
                'seaborn': 'seaborn',
                'sklearn': 'scikit-learn',
                'gradio': 'gradio',
                'wandb': 'wandb',
                'tensorboard': 'tensorboard'
            }
            
            package = package_map.get(module_lower, module_lower)
            if package not in installed_packages:
                missing.add(module)
        
        return missing
    
    def generate_requirements(
        self,
        project_dir: Path,
        output_file: Optional[Path] = None
    ) -> str:
        """
        Generar requirements.txt basado en dependencias.
        
        Args:
            project_dir: Directorio del proyecto
            output_file: Archivo de salida (opcional)
            
        Returns:
            Contenido de requirements.txt
        """
        deps = self.analyze_project(project_dir)
        
        # Mapeo de módulos a paquetes con versiones recomendadas
        package_versions = {
            'torch': 'torch>=2.0.0',
            'tensorflow': 'tensorflow>=2.13.0',
            'jax': 'jax>=0.4.0',
            'transformers': 'transformers>=4.30.0',
            'diffusers': 'diffusers>=0.20.0',
            'numpy': 'numpy>=1.24.0',
            'pandas': 'pandas>=2.0.0',
            'matplotlib': 'matplotlib>=3.7.0',
            'seaborn': 'seaborn>=0.12.0',
            'sklearn': 'scikit-learn>=1.3.0',
            'gradio': 'gradio>=3.40.0',
            'wandb': 'wandb>=0.15.0',
            'tensorboard': 'tensorboard>=2.13.0'
        }
        
        requirements = []
        for module in sorted(deps.third_party):
            module_lower = module.lower()
            if module_lower in package_versions:
                requirements.append(package_versions[module_lower])
            else:
                requirements.append(module_lower)
        
        content = '\n'.join(requirements)
        
        if output_file:
            output_file.write_text(content, encoding='utf-8')
            logger.info(f"Requirements.txt generado: {output_file}")
        
        return content


# Instancia global
_global_analyzer: Optional[DependencyAnalyzer] = None


def get_analyzer() -> DependencyAnalyzer:
    """
    Obtener instancia global del analizador.
    
    Returns:
        Instancia del analizador
    """
    global _global_analyzer
    
    if _global_analyzer is None:
        _global_analyzer = DependencyAnalyzer()
    
    return _global_analyzer

