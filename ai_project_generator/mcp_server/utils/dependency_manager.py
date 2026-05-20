"""
Dependency Manager - Sistema mejorado de gestión de dependencias
=================================================================

Sistema completo para gestionar, validar y optimizar dependencias
del proyecto con detección de conflictos, actualizaciones y análisis.
"""

import logging
import subprocess
import sys
from typing import Dict, List, Optional, Set, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import re
import json
from packaging import version, requirements

logger = logging.getLogger(__name__)


class DependencyType(Enum):
    """Tipos de dependencias"""
    CORE = "core"
    WEB_FRAMEWORK = "web_framework"
    DATABASE = "database"
    AI_ML = "ai_ml"
    SECURITY = "security"
    MONITORING = "monitoring"
    TESTING = "testing"
    DEVELOPMENT = "development"
    OPTIONAL = "optional"
    PERFORMANCE = "performance"


class DependencyStatus(Enum):
    """Estado de una dependencia"""
    INSTALLED = "installed"
    MISSING = "missing"
    OUTDATED = "outdated"
    CONFLICT = "conflict"
    INCOMPATIBLE = "incompatible"


@dataclass
class Dependency:
    """Representa una dependencia con toda su información"""
    name: str
    version_spec: str
    category: DependencyType
    required: bool = True
    description: str = ""
    alternatives: List[str] = field(default_factory=list)
    security_issues: List[str] = field(default_factory=list)
    size_mb: Optional[float] = None
    install_time_sec: Optional[float] = None
    
    def __post_init__(self):
        """Validar y normalizar la especificación de versión"""
        if not self.version_spec:
            self.version_spec = ">=0.0.0"
        elif not any(op in self.version_spec for op in [">=", "<=", "==", ">", "<", "~=", "!="]):
            self.version_spec = f">={self.version_spec}"


@dataclass
class DependencyConflict:
    """Representa un conflicto entre dependencias"""
    package: str
    required_versions: List[str]
    conflict_type: str
    severity: str
    resolution: Optional[str] = None


class DependencyManager:
    """Gestor mejorado de dependencias"""
    
    def __init__(self, requirements_file: Optional[Path] = None):
        """
        Inicializar gestor de dependencias.
        
        Args:
            requirements_file: Ruta al archivo requirements.txt
        """
        self.requirements_file = requirements_file or Path("requirements.txt")
        self.dependencies: Dict[str, Dependency] = {}
        self.installed_packages: Dict[str, str] = {}
        self.conflicts: List[DependencyConflict] = []
        self._load_installed_packages()
        self._load_dependencies()
    
    def _load_installed_packages(self) -> None:
        """Cargar paquetes instalados en el entorno actual"""
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "list", "--format=json"],
                capture_output=True,
                text=True,
                check=True
            )
            packages = json.loads(result.stdout)
            self.installed_packages = {
                pkg["name"].lower(): pkg["version"]
                for pkg in packages
            }
        except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
            logger.warning(f"Error loading installed packages: {e}")
            self.installed_packages = {}
    
    def _load_dependencies(self) -> None:
        """Cargar dependencias desde requirements.txt"""
        if not self.requirements_file.exists():
            logger.warning(f"Requirements file not found: {self.requirements_file}")
            return
        
        with open(self.requirements_file, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                
                try:
                    dep = self._parse_dependency_line(line, line_num)
                    if dep:
                        self.dependencies[dep.name.lower()] = dep
                except Exception as e:
                    logger.error(f"Error parsing line {line_num}: {line} - {e}")
    
    def _parse_dependency_line(self, line: str, line_num: int) -> Optional[Dependency]:
        """
        Parsear una línea de requirements.txt.
        
        Soporta formatos:
        - package==1.0.0
        - package>=1.0.0
        - package>=1.0.0,<2.0.0
        - package; python_version < "3.8"
        - -r other_file.txt
        """
        line = line.strip()
        
        if line.startswith("-r") or line.startswith("--requirement"):
            return None
        
        if ";" in line:
            line = line.split(";")[0].strip()
        
        if "[" in line:
            base_name = line.split("[")[0].strip()
        else:
            base_name = line.split(">=")[0].split("==")[0].split("<=")[0].split(">")[0].split("<")[0].split("~=")[0].split("!=")[0].strip()
        
        version_spec = line[len(base_name):].strip()
        if not version_spec:
            version_spec = ">=0.0.0"
        
        category = self._infer_category(base_name)
        
        return Dependency(
            name=base_name,
            version_spec=version_spec,
            category=category,
            description=self._get_package_description(base_name)
        )
    
    def _infer_category(self, package_name: str) -> DependencyType:
        """Inferir categoría de un paquete por su nombre"""
        name_lower = package_name.lower()
        
        if any(x in name_lower for x in ["fastapi", "uvicorn", "starlette", "flask", "django"]):
            return DependencyType.WEB_FRAMEWORK
        elif any(x in name_lower for x in ["torch", "tensorflow", "transformers", "sklearn", "numpy", "pandas"]):
            return DependencyType.AI_ML
        elif any(x in name_lower for x in ["sqlalchemy", "psycopg", "asyncpg", "redis", "mongodb"]):
            return DependencyType.DATABASE
        elif any(x in name_lower for x in ["pytest", "unittest", "coverage", "mock"]):
            return DependencyType.TESTING
        elif any(x in name_lower for x in ["black", "ruff", "mypy", "pylint", "flake8"]):
            return DependencyType.DEVELOPMENT
        elif any(x in name_lower for x in ["cryptography", "jwt", "passlib", "bcrypt"]):
            return DependencyType.SECURITY
        elif any(x in name_lower for x in ["prometheus", "sentry", "opentelemetry", "structlog"]):
            return DependencyType.MONITORING
        else:
            return DependencyType.CORE
    
    def _get_package_description(self, package_name: str) -> str:
        """Obtener descripción de un paquete"""
        descriptions = {
            "fastapi": "Modern web framework for building APIs",
            "uvicorn": "ASGI server for FastAPI",
            "pydantic": "Data validation using Python type annotations",
            "torch": "PyTorch deep learning framework",
            "transformers": "Hugging Face Transformers library",
            "pytest": "Testing framework",
            "black": "Code formatter",
            "ruff": "Fast Python linter",
        }
        return descriptions.get(package_name.lower(), "")
    
    def check_dependencies(self) -> Dict[str, DependencyStatus]:
        """
        Verificar estado de todas las dependencias.
        
        Returns:
            Diccionario con estado de cada dependencia
        """
        statuses: Dict[str, DependencyStatus] = {}
        
        for name, dep in self.dependencies.items():
            status = self._check_dependency_status(dep)
            statuses[name] = status
        
        return statuses
    
    def _check_dependency_status(self, dep: Dependency) -> DependencyStatus:
        """Verificar estado de una dependencia específica"""
        installed_version = self.installed_packages.get(dep.name.lower())
        
        if not installed_version:
            return DependencyStatus.MISSING
        
        try:
            req = requirements.Requirement(f"{dep.name}{dep.version_spec}")
            if not req.specifier.contains(installed_version):
                return DependencyStatus.OUTDATED
            return DependencyStatus.INSTALLED
        except Exception:
            return DependencyStatus.INCOMPATIBLE
    
    def detect_conflicts(self) -> List[DependencyConflict]:
        """
        Detectar conflictos entre dependencias.
        
        Returns:
            Lista de conflictos detectados
        """
        self.conflicts = []
        
        for name, dep in self.dependencies.items():
            installed_version = self.installed_packages.get(dep.name.lower())
            if installed_version:
                try:
                    req = requirements.Requirement(f"{dep.name}{dep.version_spec}")
                    if not req.specifier.contains(installed_version):
                        self.conflicts.append(DependencyConflict(
                            package=dep.name,
                            required_versions=[dep.version_spec],
                            conflict_type="version_mismatch",
                            severity="high",
                            resolution=f"Update {dep.name} to match {dep.version_spec}"
                        ))
                except Exception as e:
                    self.conflicts.append(DependencyConflict(
                        package=dep.name,
                        required_versions=[dep.version_spec],
                        conflict_type="parse_error",
                        severity="medium",
                        resolution=f"Fix version specification: {e}"
                    ))
        
        return self.conflicts
    
    def get_outdated_packages(self) -> List[Dict[str, Any]]:
        """
        Obtener paquetes desactualizados.
        
        Returns:
            Lista de paquetes con información de actualización
        """
        outdated = []
        
        for name, dep in self.dependencies.items():
            installed_version = self.installed_packages.get(dep.name.lower())
            if installed_version:
                try:
                    req = requirements.Requirement(f"{dep.name}{dep.version_spec}")
                    if not req.specifier.contains(installed_version):
                        outdated.append({
                            "name": dep.name,
                            "installed": installed_version,
                            "required": dep.version_spec,
                            "category": dep.category.value,
                            "description": dep.description
                        })
                except Exception:
                    pass
        
        return outdated
    
    def generate_report(self) -> Dict[str, Any]:
        """
        Generar reporte completo de dependencias.
        
        Returns:
            Diccionario con reporte completo
        """
        statuses = self.check_dependencies()
        conflicts = self.detect_conflicts()
        outdated = self.get_outdated_packages()
        
        total = len(self.dependencies)
        installed = sum(1 for s in statuses.values() if s == DependencyStatus.INSTALLED)
        missing = sum(1 for s in statuses.values() if s == DependencyStatus.MISSING)
        outdated_count = len(outdated)
        
        by_category: Dict[str, int] = {}
        for dep in self.dependencies.values():
            cat = dep.category.value
            by_category[cat] = by_category.get(cat, 0) + 1
        
        return {
            "summary": {
                "total_dependencies": total,
                "installed": installed,
                "missing": missing,
                "outdated": outdated_count,
                "conflicts": len(conflicts),
                "installation_rate": (installed / total * 100) if total > 0 else 0
            },
            "by_category": by_category,
            "statuses": {name: status.value for name, status in statuses.items()},
            "conflicts": [
                {
                    "package": c.package,
                    "type": c.conflict_type,
                    "severity": c.severity,
                    "resolution": c.resolution
                }
                for c in conflicts
            ],
            "outdated": outdated,
            "recommendations": self._generate_recommendations(statuses, conflicts, outdated)
        }
    
    def _generate_recommendations(
        self,
        statuses: Dict[str, DependencyStatus],
        conflicts: List[DependencyConflict],
        outdated: List[Dict[str, Any]]
    ) -> List[str]:
        """Generar recomendaciones basadas en el análisis"""
        recommendations = []
        
        if conflicts:
            recommendations.append(
                f"Resolve {len(conflicts)} dependency conflict(s) to ensure compatibility"
            )
        
        if outdated:
            recommendations.append(
                f"Update {len(outdated)} outdated package(s) for security and features"
            )
        
        missing = [name for name, status in statuses.items() if status == DependencyStatus.MISSING]
        if missing:
            recommendations.append(
                f"Install {len(missing)} missing package(s): {', '.join(missing[:5])}"
            )
        
        return recommendations
    
    def optimize_requirements(self, output_file: Optional[Path] = None) -> Path:
        """
        Optimizar archivo de requirements eliminando duplicados y conflictos.
        
        Args:
            output_file: Archivo de salida (default: requirements_optimized.txt)
        
        Returns:
            Ruta al archivo generado
        """
        output_file = output_file or self.requirements_file.parent / "requirements_optimized.txt"
        
        optimized_lines = []
        seen_packages: Set[str] = set()
        
        for dep in sorted(self.dependencies.values(), key=lambda x: (x.category.value, x.name)):
            if dep.name.lower() not in seen_packages:
                optimized_lines.append(f"{dep.name}{dep.version_spec}")
                if dep.description:
                    optimized_lines[-1] += f"  # {dep.description}"
                seen_packages.add(dep.name.lower())
        
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("# Optimized Requirements File\n")
            f.write("# Generated by DependencyManager\n\n")
            
            current_category = None
            for dep in sorted(self.dependencies.values(), key=lambda x: (x.category.value, x.name)):
                if dep.category.value != current_category:
                    current_category = dep.category.value
                    f.write(f"\n# {current_category.upper().replace('_', ' ')}\n")
                
                line = f"{dep.name}{dep.version_spec}"
                if dep.description:
                    line += f"  # {dep.description}"
                f.write(f"{line}\n")
        
        logger.info(f"Optimized requirements written to {output_file}")
        return output_file


def analyze_dependencies(requirements_file: Optional[Path] = None) -> Dict[str, Any]:
    """
    Función de conveniencia para analizar dependencias.
    
    Args:
        requirements_file: Ruta al archivo requirements.txt
    
    Returns:
        Reporte completo de dependencias
    """
    manager = DependencyManager(requirements_file)
    return manager.generate_report()


def check_dependency_health(requirements_file: Optional[Path] = None) -> Tuple[bool, List[str]]:
    """
    Verificar salud de las dependencias.
    
    Args:
        requirements_file: Ruta al archivo requirements.txt
    
    Returns:
        Tupla (is_healthy, issues)
    """
    manager = DependencyManager(requirements_file)
    report = manager.generate_report()
    
    issues = []
    is_healthy = True
    
    if report["summary"]["conflicts"] > 0:
        is_healthy = False
        issues.append(f"{report['summary']['conflicts']} dependency conflict(s) detected")
    
    if report["summary"]["missing"] > 0:
        is_healthy = False
        issues.append(f"{report['summary']['missing']} missing dependency(ies)")
    
    return is_healthy, issues

