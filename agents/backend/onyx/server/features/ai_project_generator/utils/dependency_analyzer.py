"""
Dependency Analyzer
===================

Sistema de análisis de dependencias de código.
"""

import ast
import re
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class DependencyType(Enum):
    """Tipos de dependencias."""
    STANDARD_LIBRARY = "standard_library"
    THIRD_PARTY = "third_party"
    LOCAL = "local"
    UNKNOWN = "unknown"


@dataclass
class Dependency:
    """Dependencia."""
    name: str
    type: DependencyType
    version: Optional[str] = None
    used_in: List[str] = None
    is_used: bool = True


class DependencyAnalyzer:
    """Analizador de dependencias."""
    
    def __init__(self):
        self.standard_library = {
            'os', 'sys', 'json', 're', 'datetime', 'time', 'random',
            'math', 'collections', 'itertools', 'functools', 'operator',
            'pathlib', 'shutil', 'glob', 'pickle', 'copy', 'enum',
            'dataclasses', 'typing', 'abc', 'contextlib', 'logging',
            'unittest', 'pytest', 'asyncio', 'threading', 'multiprocessing',
            'urllib', 'http', 'socket', 'ssl', 'hashlib', 'base64',
            'csv', 'xml', 'html', 'sqlite3', 'zlib', 'gzip', 'tarfile',
            'zipfile', 'io', 'tempfile', 'subprocess', 'signal', 'errno'
        }
        
        self.third_party_patterns = [
            r'^numpy$', r'^pandas$', r'^requests$', r'^flask$', r'^django$',
            r'^fastapi$', r'^pydantic$', r'^sqlalchemy$', r'^pytest$',
            r'^pytorch$', r'^tensorflow$', r'^sklearn$', r'^matplotlib$',
            r'^pillow$', r'^opencv$', r'^redis$', r'^celery$', r'^boto3$',
            r'^psycopg2$', r'^mysql$', r'^mongo$', r'^elasticsearch$'
        ]
    
    def analyze_file(self, code: str, file_path: str = "unknown") -> Dict[str, Any]:
        """Analiza dependencias de un archivo."""
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return {
                "file_path": file_path,
                "error": f"Syntax error: {str(e)}",
                "dependencies": []
            }
        
        imports = self._extract_imports(tree)
        dependencies = self._classify_dependencies(imports, code)
        
        return {
            "file_path": file_path,
            "total_dependencies": len(dependencies),
            "dependencies": [
                {
                    "name": d.name,
                    "type": d.type.value,
                    "version": d.version,
                    "used_in": d.used_in or [],
                    "is_used": d.is_used
                }
                for d in dependencies
            ],
            "by_type": self._group_by_type(dependencies),
            "unused": [d.name for d in dependencies if not d.is_used],
            "recommendations": self._generate_recommendations(dependencies)
        }
    
    def _extract_imports(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Extrae imports del AST."""
        imports = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append({
                        "name": alias.name,
                        "alias": alias.asname,
                        "line": node.lineno
                    })
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    imports.append({
                        "name": f"{module}.{alias.name}" if module else alias.name,
                        "alias": alias.asname or alias.name,
                        "module": module,
                        "line": node.lineno
                    })
        
        return imports
    
    def _classify_dependencies(self, imports: List[Dict], code: str) -> List[Dependency]:
        """Clasifica dependencias."""
        dependencies = []
        seen = set()
        
        for imp in imports:
            name = imp["name"]
            module_name = name.split('.')[0]
            
            if module_name in seen:
                continue
            seen.add(module_name)
            
            # Determinar tipo
            dep_type = self._determine_type(module_name)
            
            # Verificar si se usa
            is_used = self._is_dependency_used(imp, code)
            
            # Obtener version si está en requirements
            version = self._extract_version(module_name, code)
            
            # Obtener lugares donde se usa
            used_in = self._find_usage_locations(imp, code)
            
            dependencies.append(Dependency(
                name=module_name,
                type=dep_type,
                version=version,
                used_in=used_in,
                is_used=is_used
            ))
        
        return dependencies
    
    def _determine_type(self, module_name: str) -> DependencyType:
        """Determina tipo de dependencia."""
        if module_name in self.standard_library:
            return DependencyType.STANDARD_LIBRARY
        
        for pattern in self.third_party_patterns:
            if re.match(pattern, module_name, re.IGNORECASE):
                return DependencyType.THIRD_PARTY
        
        # Si no está en la librería estándar, probablemente es third-party
        if not module_name.startswith('.'):
            return DependencyType.THIRD_PARTY
        
        return DependencyType.LOCAL
    
    def _is_dependency_used(self, imp: Dict, code: str) -> bool:
        """Verifica si una dependencia se usa en el código."""
        name = imp.get("alias") or imp["name"].split('.')[-1]
        module_name = imp["name"].split('.')[0]
        
        # Buscar uso del módulo o alias
        pattern = rf'\b{re.escape(name)}\b'
        matches = list(re.finditer(pattern, code))
        
        # Filtrar el import mismo
        import_line = imp.get("line", 0)
        usage_matches = [m for m in matches if code[:m.start()].count('\n') + 1 != import_line]
        
        return len(usage_matches) > 0
    
    def _extract_version(self, module_name: str, code: str) -> Optional[str]:
        """Extrae versión de requirements.txt si existe."""
        # Buscar en requirements.txt o setup.py
        requirements_pattern = rf'{re.escape(module_name)}\s*[=<>!]+\s*([\d.]+)'
        match = re.search(requirements_pattern, code, re.IGNORECASE)
        
        if match:
            return match.group(1)
        
        return None
    
    def _find_usage_locations(self, imp: Dict, code: str) -> List[str]:
        """Encuentra lugares donde se usa la dependencia."""
        name = imp.get("alias") or imp["name"].split('.')[-1]
        pattern = rf'\b{re.escape(name)}\.'
        
        locations = []
        for match in re.finditer(pattern, code):
            line_num = code[:match.start()].count('\n') + 1
            line_content = code.split('\n')[line_num - 1].strip()[:50]
            locations.append(f"Line {line_num}: {line_content}")
        
        return locations[:5]  # Limitar a 5
    
    def _group_by_type(self, dependencies: List[Dependency]) -> Dict[str, int]:
        """Agrupa dependencias por tipo."""
        grouped = {}
        for dep in dependencies:
            dep_type = dep.type.value
            grouped[dep_type] = grouped.get(dep_type, 0) + 1
        return grouped
    
    def _generate_recommendations(self, dependencies: List[Dependency]) -> List[str]:
        """Genera recomendaciones sobre dependencias."""
        recommendations = []
        
        unused = [d for d in dependencies if not d.is_used]
        if unused:
            recommendations.append(
                f"Remove unused imports: {', '.join([d.name for d in unused[:5]])}"
            )
        
        third_party = [d for d in dependencies if d.type == DependencyType.THIRD_PARTY]
        if len(third_party) > 20:
            recommendations.append(
                "Consider reducing third-party dependencies to minimize security risks"
            )
        
        without_version = [d for d in third_party if not d.version]
        if without_version:
            recommendations.append(
                "Pin dependency versions in requirements.txt for reproducibility"
            )
        
        return recommendations
    
    def generate_requirements_txt(self, dependencies: List[Dependency]) -> str:
        """Genera requirements.txt."""
        third_party = [d for d in dependencies if d.type == DependencyType.THIRD_PARTY]
        
        requirements = []
        for dep in sorted(third_party, key=lambda x: x.name.lower()):
            if dep.version:
                requirements.append(f"{dep.name}=={dep.version}")
            else:
                requirements.append(f"{dep.name}")
        
        return '\n'.join(requirements)
    
    def check_vulnerabilities(self, dependencies: List[Dependency]) -> List[Dict[str, Any]]:
        """Verifica vulnerabilidades conocidas (simplificado)."""
        vulnerabilities = []
        
        # Lista simplificada de módulos con vulnerabilidades conocidas
        vulnerable_modules = {
            'pickle': 'Security risk - use json or safer alternatives',
            'eval': 'Security risk - never use eval()',
            'exec': 'Security risk - never use exec()',
        }
        
        for dep in dependencies:
            if dep.name in vulnerable_modules:
                vulnerabilities.append({
                    "module": dep.name,
                    "severity": "high",
                    "description": vulnerable_modules[dep.name],
                    "recommendation": "Use safer alternatives"
                })
        
        return vulnerabilities


# Factory function
_dependency_analyzer = None

def get_dependency_analyzer() -> DependencyAnalyzer:
    """Obtiene instancia global del analizador."""
    global _dependency_analyzer
    if _dependency_analyzer is None:
        _dependency_analyzer = DependencyAnalyzer()
    return _dependency_analyzer

