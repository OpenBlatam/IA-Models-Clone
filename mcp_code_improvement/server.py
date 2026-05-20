#!/usr/bin/env python3
"""
Servidor MCP para mejora de código - Blatam Academy
Implementa herramientas para análisis, refactorización y optimización de código
"""

import json
import sys
import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional
import subprocess
import ast
import re
from datetime import datetime
import traceback

# Configuración del proyecto
PROJECT_ROOT = Path(__file__).parent.parent
FEATURES_DIR = PROJECT_ROOT


class CodeImprovementServer:
    """Servidor MCP para mejora de código"""
    
    def __init__(self):
        self.tools = {
            "analyze_code_quality": self.analyze_code_quality,
            "detect_code_duplication": self.detect_code_duplication,
            "suggest_refactoring": self.suggest_refactoring,
            "analyze_architecture": self.analyze_architecture,
            "optimize_performance": self.optimize_performance,
            "check_security": self.check_security,
            "improve_documentation": self.improve_documentation,
            "analyze_dependencies": self.analyze_dependencies,
            "standardize_code_style": self.standardize_code_style,
            "detect_anti_patterns": self.detect_anti_patterns,
            "optimize_imports": self.optimize_imports,
            "generate_tests": self.generate_tests,
            "migrate_to_clean_architecture": self.migrate_to_clean_architecture,
            "analyze_feature_consistency": self.analyze_feature_consistency,
            "optimize_config_files": self.optimize_config_files,
            "detect_unified_patterns": self.detect_unified_patterns,
        }
    
    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Maneja una solicitud MCP"""
        method = request.get("method")
        
        if method == "tools/list":
            return await self.list_tools()
        elif method == "tools/call":
            return await self.call_tool(request.get("params", {}))
        elif method == "resources/list":
            return await self.list_resources()
        elif method == "resources/read":
            return await self.read_resource(request.get("params", {}))
        elif method == "prompts/list":
            return await self.list_prompts()
        else:
            return {"error": {"code": -32601, "message": f"Method not found: {method}"}}
    
    async def list_tools(self) -> Dict[str, Any]:
        """Lista todas las herramientas disponibles"""
        config_path = PROJECT_ROOT / "mcp_code_improvement_server.json"
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        
        return {
            "tools": config.get("tools", [])
        }
    
    async def call_tool(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Ejecuta una herramienta"""
        tool_name = params.get("name")
        arguments = params.get("arguments", {})
        
        if tool_name not in self.tools:
            return {
                "error": {
                    "code": -32601,
                    "message": f"Tool not found: {tool_name}"
                }
            }
        
        try:
            result = await self.tools[tool_name](arguments)
            
            # Formatear resultado para mejor legibilidad
            formatted_result = self._format_result(tool_name, result)
            
            return {
                "content": [
                    {
                        "type": "text",
                        "text": formatted_result
                    }
                ]
            }
        except Exception as e:
            error_details = {
                "error": {
                    "code": -32603,
                    "message": f"Internal error: {str(e)}",
                    "tool": tool_name,
                    "arguments": arguments,
                    "traceback": traceback.format_exc() if sys.stderr else None
                }
            }
            return {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(error_details, indent=2, ensure_ascii=False)
                    }
                ],
                "isError": True
            }
    
    def _format_result(self, tool_name: str, result: Dict[str, Any]) -> str:
        """Formatea el resultado para mejor legibilidad"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Crear formato markdown mejorado
        formatted = f"# 🔧 Resultado: {tool_name}\n\n"
        formatted += f"**Timestamp:** {timestamp}\n\n"
        formatted += "---\n\n"
        
        # Formatear según el tipo de resultado
        if isinstance(result, dict):
            if "issues" in result:
                formatted += self._format_issues(result)
            elif "suggestions" in result:
                formatted += self._format_suggestions(result)
            elif "vulnerabilities" in result:
                formatted += self._format_vulnerabilities(result)
            elif "optimizations" in result:
                formatted += self._format_optimizations(result)
            elif "duplications" in result:
                formatted += self._format_duplications(result)
            else:
                formatted += self._format_generic(result)
        else:
            formatted += f"```json\n{json.dumps(result, indent=2, ensure_ascii=False)}\n```\n"
        
        return formatted
    
    def _format_issues(self, result: Dict[str, Any]) -> str:
        """Formatea issues encontrados"""
        formatted = f"## 📊 Resumen\n\n"
        formatted += f"- **Total de issues:** {len(result.get('issues', []))}\n"
        formatted += f"- **Score de calidad:** {result.get('score', 'N/A')}/100\n\n"
        
        if result.get('issues'):
            formatted += "## 🐛 Issues Encontrados\n\n"
            for i, issue in enumerate(result.get('issues', [])[:20], 1):  # Limitar a 20
                formatted += f"### {i}. {issue.get('type', 'Unknown')}\n"
                formatted += f"- **Archivo:** `{issue.get('file', 'N/A')}`\n"
                formatted += f"- **Severidad:** {issue.get('severity', 'N/A')}\n"
                formatted += f"- **Mensaje:** {issue.get('message', 'N/A')}\n\n"
        
        if result.get('recommendations'):
            formatted += "## 💡 Recomendaciones\n\n"
            for rec in result.get('recommendations', [])[:10]:
                formatted += f"- {rec}\n"
        
        return formatted
    
    def _format_suggestions(self, result: Dict[str, Any]) -> str:
        """Formatea sugerencias de refactorización"""
        formatted = f"## 📝 Archivo: `{result.get('file', 'N/A')}`\n\n"
        
        suggestions = result.get('suggestions', [])
        formatted += f"**Total de sugerencias:** {len(suggestions)}\n\n"
        
        if suggestions:
            formatted += "## 🔄 Sugerencias de Refactorización\n\n"
            for i, sug in enumerate(suggestions[:15], 1):
                formatted += f"### {i}. {sug.get('type', 'Unknown')}\n"
                formatted += f"- **Ubicación:** {sug.get('location', 'N/A')}\n"
                formatted += f"- **Descripción:** {sug.get('description', 'N/A')}\n"
                formatted += f"- **Severidad:** {sug.get('severity', 'medium')}\n\n"
        
        return formatted
    
    def _format_vulnerabilities(self, result: Dict[str, Any]) -> str:
        """Formatea vulnerabilidades de seguridad"""
        formatted = f"## 🔒 Análisis de Seguridad\n\n"
        formatted += f"**Total de vulnerabilidades:** {result.get('total', 0)}\n\n"
        
        vulns = result.get('vulnerabilities', [])
        if vulns:
            formatted += "## ⚠️ Vulnerabilidades Encontradas\n\n"
            for i, vuln in enumerate(vulns[:20], 1):
                formatted += f"### {i}. {vuln.get('type', 'Unknown')}\n"
                formatted += f"- **Archivo:** `{vuln.get('file', 'N/A')}`\n"
                formatted += f"- **Severidad:** 🔴 {vuln.get('severity', 'N/A')}\n"
                formatted += f"- **Mensaje:** {vuln.get('message', 'N/A')}\n\n"
        else:
            formatted += "✅ No se encontraron vulnerabilidades críticas.\n\n"
        
        return formatted
    
    def _format_optimizations(self, result: Dict[str, Any]) -> str:
        """Formatea optimizaciones sugeridas"""
        formatted = f"## ⚡ Optimizaciones para: `{result.get('file', 'N/A')}`\n\n"
        
        optimizations = result.get('optimizations', [])
        formatted += f"**Total de optimizaciones:** {len(optimizations)}\n\n"
        
        if optimizations:
            for i, opt in enumerate(optimizations, 1):
                formatted += f"### {i}. {opt.get('type', 'Unknown')}\n"
                formatted += f"- **Sugerencia:** {opt.get('suggestion', 'N/A')}\n"
                formatted += f"- **Severidad:** {opt.get('severity', 'medium')}\n\n"
        
        return formatted
    
    def _format_duplications(self, result: Dict[str, Any]) -> str:
        """Formatea código duplicado"""
        formatted = "## 🔄 Análisis de Duplicación\n\n"
        formatted += f"**Total de duplicaciones:** {len(result.get('duplications', []))}\n"
        formatted += f"**Líneas duplicadas:** {result.get('total_duplicated_lines', 0)}\n"
        formatted += f"**Porcentaje:** {result.get('percentage', 0)}%\n\n"
        
        dups = result.get('duplications', [])
        if dups:
            formatted += "## 📋 Duplicaciones Encontradas\n\n"
            for i, dup in enumerate(dups[:15], 1):
                formatted += f"### {i}. Duplicación {i}\n"
                formatted += f"- **Archivo 1:** `{dup.get('file1', 'N/A')}`\n"
                formatted += f"- **Archivo 2:** `{dup.get('file2', 'N/A')}`\n"
                formatted += f"- **Similitud:** {dup.get('similarity', 0)*100:.1f}%\n"
                formatted += f"- **Líneas:** {dup.get('lines', 0)}\n\n"
        
        return formatted
    
    def _format_generic(self, result: Dict[str, Any]) -> str:
        """Formatea resultados genéricos"""
        formatted = "## 📊 Resultado\n\n"
        formatted += "```json\n"
        formatted += json.dumps(result, indent=2, ensure_ascii=False)
        formatted += "\n```\n"
        return formatted
    
    async def list_resources(self) -> Dict[str, Any]:
        """Lista todos los recursos disponibles"""
        config_path = PROJECT_ROOT / "mcp_code_improvement_server.json"
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        
        return {
            "resources": config.get("resources", [])
        }
    
    async def read_resource(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Lee un recurso"""
        uri = params.get("uri")
        
        # Mapeo de URIs a archivos
        resource_map = {
            "codebase://best-practices": PROJECT_ROOT / "BEST_PRACTICES_SUMMARY.md",
            "codebase://architecture-patterns": PROJECT_ROOT / "MSA_ARCHITECTURE.md",
            "codebase://refactoring-opportunities": PROJECT_ROOT / "REFACTORING_OPPORTUNITIES.md",
            "codebase://anti-patterns": PROJECT_ROOT / "ANTI_PATTERNS.md",
            "codebase://code-quality-standards": PROJECT_ROOT / "BEST_PRACTICES_SUMMARY.md",
        }
        
        if uri in resource_map:
            file_path = resource_map[uri]
            if file_path.exists():
                content = file_path.read_text(encoding="utf-8")
                return {
                    "contents": [
                        {
                            "uri": uri,
                            "mimeType": "text/markdown",
                            "text": content
                        }
                    ]
                }
        
        return {
            "error": {
                "code": -32602,
                "message": f"Resource not found: {uri}"
            }
        }
    
    async def list_prompts(self) -> Dict[str, Any]:
        """Lista todos los prompts disponibles"""
        config_path = PROJECT_ROOT / "mcp_code_improvement_server.json"
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        
        return {
            "prompts": config.get("prompts", [])
        }
    
    # Implementaciones de herramientas
    
    async def analyze_code_quality(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Analiza la calidad del código"""
        path = Path(args.get("path", "."))
        language = args.get("language", "all")
        checks = args.get("checks", ["all"])
        
        results = {
            "path": str(path),
            "language": language,
            "checks": checks,
            "issues": [],
            "score": 0,
            "recommendations": []
        }
        
        # Análisis básico de Python
        if language in ["python", "all"]:
            python_files = list(path.rglob("*.py"))
            for py_file in python_files[:10]:  # Limitar para demo
                try:
                    tree = ast.parse(py_file.read_text(encoding="utf-8"))
                    # Análisis básico de complejidad
                    complexity = self._calculate_complexity(tree)
                    if complexity > 10:
                        results["issues"].append({
                            "file": str(py_file),
                            "type": "high_complexity",
                            "severity": "medium",
                            "message": f"Complejidad ciclomática alta: {complexity}"
                        })
                except Exception as e:
                    results["issues"].append({
                        "file": str(py_file),
                        "type": "parse_error",
                        "severity": "low",
                        "message": f"Error al parsear: {str(e)}"
                    })
        
        results["score"] = max(0, 100 - len(results["issues"]) * 5)
        return results
    
    def _calculate_complexity(self, tree: ast.AST) -> int:
        """Calcula complejidad ciclomática básica"""
        complexity = 1
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.Try, ast.With)):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1
        return complexity
    
    async def detect_code_duplication(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Detecta código duplicado"""
        min_lines = args.get("minLines", 5)
        threshold = args.get("threshold", 0.8)
        exclude_patterns = args.get("excludePatterns", ["__pycache__", "node_modules", ".git"])
        
        results = {
            "duplications": [],
            "total_duplicated_lines": 0,
            "percentage": 0
        }
        
        # Implementación básica de detección
        python_files = list(FEATURES_DIR.rglob("*.py"))
        for i, file1 in enumerate(python_files[:20]):  # Limitar para demo
            if any(pattern in str(file1) for pattern in exclude_patterns):
                continue
            
            try:
                content1 = file1.read_text(encoding="utf-8")
                lines1 = content1.split('\n')
                
                for file2 in python_files[i+1:21]:
                    if any(pattern in str(file2) for pattern in exclude_patterns):
                        continue
                    
                    try:
                        content2 = file2.read_text(encoding="utf-8")
                        lines2 = content2.split('\n')
                        
                        # Comparación simple
                        similarity = self._calculate_similarity(lines1, lines2)
                        if similarity >= threshold and len(lines1) >= min_lines:
                            results["duplications"].append({
                                "file1": str(file1),
                                "file2": str(file2),
                                "similarity": similarity,
                                "lines": len(lines1)
                            })
                    except Exception:
                        continue
            except Exception:
                continue
        
        return results
    
    def _calculate_similarity(self, lines1: List[str], lines2: List[str]) -> float:
        """Calcula similitud entre dos listas de líneas"""
        if not lines1 or not lines2:
            return 0.0
        
        set1 = set(lines1)
        set2 = set(lines2)
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return intersection / union if union > 0 else 0.0
    
    async def suggest_refactoring(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Sugiere refactorizaciones"""
        file_path = Path(args.get("filePath"))
        refactoring_type = args.get("refactoringType", "all")
        apply = args.get("apply", False)
        
        suggestions = []
        
        if file_path.exists() and file_path.suffix == ".py":
            try:
                content = file_path.read_text(encoding="utf-8")
                tree = ast.parse(content)
                
                # Detectar funciones largas
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        func_lines = len(node.body) if node.body else 0
                        if func_lines > 50:
                            suggestions.append({
                                "type": "extract_method",
                                "location": f"{file_path}:{node.lineno}",
                                "description": f"Función '{node.name}' es muy larga ({func_lines} líneas). Considera extraer métodos.",
                                "severity": "medium"
                            })
            except Exception as e:
                return {"error": f"Error al analizar archivo: {str(e)}"}
        
        return {
            "file": str(file_path),
            "suggestions": suggestions,
            "applied": apply
        }
    
    async def analyze_architecture(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Analiza la arquitectura"""
        feature_path = Path(args.get("featurePath"))
        check_clean = args.get("checkCleanArchitecture", True)
        check_solid = args.get("checkSOLID", True)
        check_deps = args.get("checkDependencies", True)
        
        results = {
            "feature": str(feature_path),
            "clean_architecture": {"compliant": False, "issues": []},
            "solid": {"compliant": False, "violations": []},
            "dependencies": {"issues": []}
        }
        
        # Verificar estructura Clean Architecture
        if check_clean and feature_path.exists():
            required_dirs = ["domain", "application", "infrastructure", "interfaces"]
            existing_dirs = [d.name for d in feature_path.iterdir() if d.is_dir()]
            
            missing_dirs = [d for d in required_dirs if d not in existing_dirs]
            if missing_dirs:
                results["clean_architecture"]["issues"].append(
                    f"Directorios faltantes: {', '.join(missing_dirs)}"
                )
            else:
                results["clean_architecture"]["compliant"] = True
        
        return results
    
    async def optimize_performance(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Optimiza performance"""
        file_path = Path(args.get("filePath"))
        opt_types = args.get("optimizationTypes", ["all"])
        profile = args.get("profile", False)
        
        optimizations = []
        
        if file_path.exists() and file_path.suffix == ".py":
            content = file_path.read_text(encoding="utf-8")
            
            # Detectar oportunidades de cache
            if "all" in opt_types or "cache" in opt_types:
                if "def " in content and "@lru_cache" not in content:
                    optimizations.append({
                        "type": "cache",
                        "suggestion": "Considera usar @lru_cache para funciones costosas",
                        "severity": "medium"
                    })
            
            # Detectar operaciones síncronas que podrían ser async
            if "all" in opt_types or "async" in opt_types:
                if "requests.get" in content or "requests.post" in content:
                    optimizations.append({
                        "type": "async",
                        "suggestion": "Considera usar httpx o aiohttp para operaciones I/O asíncronas",
                        "severity": "high"
                    })
        
        return {
            "file": str(file_path),
            "optimizations": optimizations,
            "profiled": profile
        }
    
    async def check_security(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Verifica seguridad"""
        path = Path(args.get("path", "."))
        severity = args.get("severity", "medium")
        check_secrets = args.get("checkSecrets", True)
        check_injection = args.get("checkInjection", True)
        
        vulnerabilities = []
        
        if check_secrets:
            # Buscar posibles secrets hardcodeados
            python_files = list(path.rglob("*.py"))[:50]
            secret_patterns = [
                (r'password\s*=\s*["\']([^"\']+)["\']', "hardcoded_password"),
                (r'api_key\s*=\s*["\']([^"\']+)["\']', "hardcoded_api_key"),
                (r'secret\s*=\s*["\']([^"\']+)["\']', "hardcoded_secret"),
            ]
            
            for py_file in python_files:
                try:
                    content = py_file.read_text(encoding="utf-8")
                    for pattern, vuln_type in secret_patterns:
                        if re.search(pattern, content, re.IGNORECASE):
                            vulnerabilities.append({
                                "file": str(py_file),
                                "type": vuln_type,
                                "severity": "high",
                                "message": f"Posible secret hardcodeado encontrado"
                            })
                except Exception:
                    continue
        
        return {
            "path": str(path),
            "vulnerabilities": vulnerabilities,
            "total": len(vulnerabilities)
        }
    
    async def improve_documentation(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Mejora documentación"""
        file_path = Path(args.get("filePath"))
        add_docstrings = args.get("addDocstrings", True)
        improve_comments = args.get("improveComments", True)
        
        improvements = []
        
        if file_path.exists() and file_path.suffix == ".py":
            try:
                tree = ast.parse(file_path.read_text(encoding="utf-8"))
                
                if add_docstrings:
                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef) and not ast.get_docstring(node):
                            improvements.append({
                                "type": "missing_docstring",
                                "location": f"{file_path}:{node.lineno}",
                                "function": node.name,
                                "suggestion": f"Agregar docstring a función '{node.name}'"
                            })
            except Exception:
                pass
        
        return {
            "file": str(file_path),
            "improvements": improvements
        }
    
    async def analyze_dependencies(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Analiza dependencias"""
        results = {
            "vulnerabilities": [],
            "unused": [],
            "conflicts": [],
            "updates": []
        }
        
        # Buscar requirements.txt
        req_files = list(FEATURES_DIR.rglob("requirements.txt"))
        for req_file in req_files[:5]:
            try:
                content = req_file.read_text(encoding="utf-8")
                results["updates"].append({
                    "file": str(req_file),
                    "packages": len([l for l in content.split('\n') if l.strip() and not l.startswith('#')])
                })
            except Exception:
                continue
        
        return results
    
    async def standardize_code_style(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Estandariza estilo de código"""
        file_path = Path(args.get("filePath"))
        language = args.get("language")
        apply = args.get("apply", False)
        
        return {
            "file": str(file_path),
            "language": language,
            "suggestions": ["Usar black para formateo", "Usar isort para imports"],
            "applied": apply
        }
    
    async def detect_anti_patterns(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Detecta anti-patrones"""
        path = Path(args.get("path", "."))
        anti_patterns = args.get("antiPatterns", ["all"])
        
        detected = []
        
        python_files = list(path.rglob("*.py"))[:20]
        for py_file in python_files:
            try:
                content = py_file.read_text(encoding="utf-8")
                tree = ast.parse(content)
                
                # Detectar funciones muy largas
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        if len(node.body) > 100:
                            detected.append({
                                "pattern": "long_method",
                                "file": str(py_file),
                                "location": node.lineno,
                                "function": node.name
                            })
            except Exception:
                continue
        
        return {
            "path": str(path),
            "anti_patterns": detected
        }
    
    async def optimize_imports(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Optimiza imports"""
        file_path = Path(args.get("filePath"))
        remove_unused = args.get("removeUnused", True)
        organize = args.get("organize", True)
        apply = args.get("apply", False)
        
        return {
            "file": str(file_path),
            "suggestions": ["Organizar imports con isort", "Eliminar imports no usados"],
            "applied": apply
        }
    
    async def generate_tests(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Genera tests"""
        file_path = Path(args.get("filePath"))
        test_type = args.get("testType", "both")
        framework = args.get("framework", "auto")
        coverage = args.get("coverage", 80)
        
        return {
            "file": str(file_path),
            "test_type": test_type,
            "framework": framework,
            "coverage_target": coverage,
            "suggestion": "Usar pytest para generar tests automatizados"
        }
    
    async def migrate_to_clean_architecture(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Migra a Clean Architecture"""
        feature_path = Path(args.get("featurePath"))
        create_structure = args.get("createStructure", True)
        migrate_code = args.get("migrateCode", True)
        generate_interfaces = args.get("generateInterfaces", True)
        
        return {
            "feature": str(feature_path),
            "structure_created": create_structure,
            "migration_plan": "Crear estructura domain/application/infrastructure/interfaces"
        }
    
    async def analyze_feature_consistency(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Analiza consistencia entre features"""
        compare_with = args.get("compareWith", "instagram_captions")
        features = args.get("features", [])
        
        return {
            "reference": compare_with,
            "features_analyzed": features,
            "inconsistencies": []
        }
    
    async def optimize_config_files(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Optimiza archivos de configuración"""
        config_type = args.get("configType", "all")
        consolidate = args.get("consolidate", False)
        
        return {
            "config_type": config_type,
            "suggestions": ["Unificar configuraciones similares", "Usar variables de entorno"],
            "consolidated": consolidate
        }
    
    async def detect_unified_patterns(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Detecta patrones unificables"""
        pattern_type = args.get("patternType", "all")
        min_occurrences = args.get("minOccurrences", 3)
        
        patterns = []
        
        if pattern_type in ["optimizers", "all"]:
            # Buscar optimizadores en suno_clone_ai
            optimizers_path = FEATURES_DIR / "suno_clone_ai" / "core"
            if optimizers_path.exists():
                optimizer_files = list(optimizers_path.rglob("*optimizer*.py"))
                if len(optimizer_files) >= min_occurrences:
                    patterns.append({
                        "type": "optimizers",
                        "location": str(optimizers_path),
                        "count": len(optimizer_files),
                        "suggestion": "Unificar optimizadores en un sistema base común"
                    })
        
        return {
            "pattern_type": pattern_type,
            "patterns": patterns
        }


async def main():
    """Función principal del servidor"""
    server = CodeImprovementServer()
    
    # Leer desde stdin (protocolo MCP)
    while True:
        try:
            line = await asyncio.get_event_loop().run_in_executor(
                None, sys.stdin.readline
            )
            if not line:
                break
            
            request = json.loads(line.strip())
            response = await server.handle_request(request)
            print(json.dumps(response, ensure_ascii=False))
            sys.stdout.flush()
        except Exception as e:
            error_response = {
                "error": {
                    "code": -32603,
                    "message": f"Internal error: {str(e)}"
                }
            }
            print(json.dumps(error_response))
            sys.stdout.flush()


if __name__ == "__main__":
    asyncio.run(main())


