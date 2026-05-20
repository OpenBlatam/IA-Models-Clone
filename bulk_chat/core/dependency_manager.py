"""
Dependency Manager - Gestor de Dependencias
===========================================

Sistema avanzado de gestión de dependencias con análisis de vulnerabilidades, versionado y resolución automática.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict

logger = logging.getLogger(__name__)


class DependencyStatus(Enum):
    """Estado de dependencia."""
    UP_TO_DATE = "up_to_date"
    OUTDATED = "outdated"
    VULNERABLE = "vulnerable"
    DEPRECATED = "deprecated"
    MISSING = "missing"


class VulnerabilitySeverity(Enum):
    """Severidad de vulnerabilidad."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class Dependency:
    """Dependencia."""
    name: str
    version: str
    latest_version: Optional[str] = None
    status: DependencyStatus = DependencyStatus.UP_TO_DATE
    vulnerabilities: List[Dict[str, Any]] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Vulnerability:
    """Vulnerabilidad."""
    vuln_id: str
    dependency_name: str
    version_range: str
    severity: VulnerabilitySeverity
    description: str
    cve_id: Optional[str] = None
    fixed_version: Optional[str] = None
    discovered_at: Optional[datetime] = None


class DependencyManager:
    """Gestor de dependencias."""
    
    def __init__(self):
        self.dependencies: Dict[str, Dependency] = {}
        self.vulnerabilities: Dict[str, Vulnerability] = {}
        self.dependency_graph: Dict[str, List[str]] = defaultdict(list)
        self.update_history: List[Dict[str, Any]] = []
        self._lock = asyncio.Lock()
    
    def register_dependency(
        self,
        name: str,
        version: str,
        dependencies: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Registrar dependencia."""
        dep = Dependency(
            name=name,
            version=version,
            dependencies=dependencies or [],
            metadata=metadata or {},
        )
        
        self.dependencies[name] = dep
        self.dependency_graph[name] = dep.dependencies
        
        logger.info(f"Registered dependency: {name} v{version}")
    
    async def check_updates(self, dependency_name: Optional[str] = None) -> Dict[str, Any]:
        """Verificar actualizaciones disponibles."""
        deps_to_check = [dependency_name] if dependency_name else list(self.dependencies.keys())
        
        updates = {}
        
        for dep_name in deps_to_check:
            dep = self.dependencies.get(dep_name)
            if not dep:
                continue
            
            # Simular verificación de última versión
            # En producción, consultar PyPI, npm, etc.
            latest_version = f"{dep.version.split('.')[0]}.{int(dep.version.split('.')[1]) + 1}.0"
            dep.latest_version = latest_version
            
            if dep.latest_version != dep.version:
                dep.status = DependencyStatus.OUTDATED
                updates[dep_name] = {
                    "current": dep.version,
                    "latest": dep.latest_version,
                    "status": dep.status.value,
                }
        
        return updates
    
    async def scan_vulnerabilities(self, dependency_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """Escanear vulnerabilidades."""
        deps_to_scan = [dependency_name] if dependency_name else list(self.dependencies.keys())
        
        vulnerabilities_found = []
        
        for dep_name in deps_to_scan:
            dep = self.dependencies.get(dep_name)
            if not dep:
                continue
            
            # Simular escaneo de vulnerabilidades
            # En producción, consultar bases de datos de CVEs
            if dep.version.startswith("0.") or "beta" in dep.version.lower():
                vuln = Vulnerability(
                    vuln_id=f"vuln_{dep_name}_{datetime.now().timestamp()}",
                    dependency_name=dep_name,
                    version_range=f"<1.0.0",
                    severity=VulnerabilitySeverity.MEDIUM,
                    description=f"Potential security issues in {dep_name} < 1.0.0",
                    fixed_version="1.0.0",
                )
                
                self.vulnerabilities[vuln.vuln_id] = vuln
                dep.vulnerabilities.append({
                    "vuln_id": vuln.vuln_id,
                    "severity": vuln.severity.value,
                    "description": vuln.description,
                })
                
                dep.status = DependencyStatus.VULNERABLE
                vulnerabilities_found.append({
                    "dependency": dep_name,
                    "vulnerability": vuln.vuln_id,
                    "severity": vuln.severity.value,
                    "description": vuln.description,
                })
        
        return vulnerabilities_found
    
    async def update_dependency(
        self,
        dependency_name: str,
        target_version: Optional[str] = None,
    ) -> bool:
        """Actualizar dependencia."""
        dep = self.dependencies.get(dependency_name)
        if not dep:
            return False
        
        new_version = target_version or dep.latest_version
        if not new_version:
            return False
        
        old_version = dep.version
        
        async with self._lock:
            dep.version = new_version
            dep.status = DependencyStatus.UP_TO_DATE
            
            self.update_history.append({
                "dependency": dependency_name,
                "old_version": old_version,
                "new_version": new_version,
                "timestamp": datetime.now(),
            })
        
        logger.info(f"Updated dependency: {dependency_name} {old_version} -> {new_version}")
        return True
    
    def get_dependency_tree(self, root_name: str, max_depth: int = 5) -> Dict[str, Any]:
        """Obtener árbol de dependencias."""
        def build_tree(name: str, depth: int = 0) -> Dict[str, Any]:
            if depth > max_depth:
                return {"name": name, "depth": depth, "truncated": True}
            
            dep = self.dependencies.get(name)
            if not dep:
                return {"name": name, "error": "Not found"}
            
            return {
                "name": name,
                "version": dep.version,
                "status": dep.status.value,
                "dependencies": [
                    build_tree(child, depth + 1)
                    for child in dep.dependencies
                ],
            }
        
        return build_tree(root_name)
    
    def get_vulnerabilities(
        self,
        dependency_name: Optional[str] = None,
        severity: Optional[VulnerabilitySeverity] = None,
    ) -> List[Dict[str, Any]]:
        """Obtener vulnerabilidades."""
        vulns = list(self.vulnerabilities.values())
        
        if dependency_name:
            vulns = [v for v in vulns if v.dependency_name == dependency_name]
        
        if severity:
            vulns = [v for v in vulns if v.severity == severity]
        
        return [
            {
                "vuln_id": v.vuln_id,
                "dependency_name": v.dependency_name,
                "version_range": v.version_range,
                "severity": v.severity.value,
                "description": v.description,
                "cve_id": v.cve_id,
                "fixed_version": v.fixed_version,
            }
            for v in vulns
        ]
    
    def get_dependency_summary(self) -> Dict[str, Any]:
        """Obtener resumen de dependencias."""
        by_status: Dict[str, int] = defaultdict(int)
        by_severity: Dict[str, int] = defaultdict(int)
        
        for dep in self.dependencies.values():
            by_status[dep.status.value] += 1
        
        for vuln in self.vulnerabilities.values():
            by_severity[vuln.severity.value] += 1
        
        return {
            "total_dependencies": len(self.dependencies),
            "dependencies_by_status": dict(by_status),
            "total_vulnerabilities": len(self.vulnerabilities),
            "vulnerabilities_by_severity": dict(by_severity),
            "total_updates": len(self.update_history),
        }
















