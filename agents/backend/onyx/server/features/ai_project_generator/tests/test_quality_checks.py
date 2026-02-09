"""
Quality checks for generated projects
"""

import pytest
from pathlib import Path
from typing import Dict, Any, List
import ast
import json


class QualityChecker:
    """Quality checks for generated projects"""
    
    @staticmethod
    def check_code_quality(file_path: Path) -> Dict[str, Any]:
        """Check code quality of a Python file"""
        if not file_path.exists() or file_path.suffix != ".py":
            return {"valid": False, "error": "Not a Python file"}
        
        try:
            content = file_path.read_text(encoding="utf-8")
            tree = ast.parse(content)
            
            issues = []
            
            # Check for long functions (more than 50 lines)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    lines = len(node.body) if hasattr(node, 'body') else 0
                    if lines > 50:
                        issues.append(f"Function {node.name} is too long ({lines} lines)")
            
            # Check for missing docstrings in classes
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    if not ast.get_docstring(node):
                        issues.append(f"Class {node.name} missing docstring")
            
            return {
                "valid": len(issues) == 0,
                "issues": issues,
                "syntax_valid": True
            }
        except SyntaxError as e:
            return {
                "valid": False,
                "error": f"Syntax error: {e}",
                "syntax_valid": False
            }
    
    @staticmethod
    def check_project_structure_quality(project_path: Path) -> Dict[str, Any]:
        """Check quality of project structure"""
        issues = []
        warnings = []
        
        # Check for required directories
        required_dirs = ["backend", "frontend"]
        for dir_name in required_dirs:
            dir_path = project_path / dir_name
            if not dir_path.exists():
                issues.append(f"Required directory {dir_name} not found")
        
        # Check for README
        readme = project_path / "README.md"
        if not readme.exists():
            warnings.append("README.md not found")
        
        # Check for .gitignore
        gitignore = project_path / ".gitignore"
        if not gitignore.exists():
            warnings.append(".gitignore not found")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings
        }
    
    @staticmethod
    def check_dependencies_quality(project_path: Path) -> Dict[str, Any]:
        """Check quality of dependencies"""
        issues = []
        
        # Check backend requirements.txt
        backend_req = project_path / "backend" / "requirements.txt"
        if backend_req.exists():
            content = backend_req.read_text(encoding="utf-8")
            if "fastapi" not in content.lower() and "flask" not in content.lower():
                issues.append("No web framework found in requirements.txt")
        
        # Check frontend package.json
        frontend_pkg = project_path / "frontend" / "package.json"
        if frontend_pkg.exists():
            try:
                pkg_data = json.loads(frontend_pkg.read_text(encoding="utf-8"))
                if "dependencies" not in pkg_data:
                    issues.append("No dependencies in package.json")
            except json.JSONDecodeError:
                issues.append("Invalid JSON in package.json")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues
        }
    
    @staticmethod
    def run_all_checks(project_path: Path) -> Dict[str, Any]:
        """Run all quality checks"""
        results = {
            "code_quality": {},
            "structure_quality": {},
            "dependencies_quality": {},
            "overall_valid": True
        }
        
        # Check Python files
        python_files = list(project_path.rglob("*.py"))
        if python_files:
            # Check first Python file as sample
            results["code_quality"] = QualityChecker.check_code_quality(python_files[0])
        
        # Check structure
        results["structure_quality"] = QualityChecker.check_project_structure_quality(project_path)
        
        # Check dependencies
        results["dependencies_quality"] = QualityChecker.check_dependencies_quality(project_path)
        
        # Overall validity
        results["overall_valid"] = (
            results["code_quality"].get("valid", True) and
            results["structure_quality"].get("valid", True) and
            results["dependencies_quality"].get("valid", True)
        )
        
        return results


@pytest.fixture
def quality_checker():
    """Fixture for quality checker"""
    return QualityChecker

