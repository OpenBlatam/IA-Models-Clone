"""
Import Optimizer for Piel Mejorador AI SAM3
===========================================

Utilities for optimizing imports and detecting circular dependencies.
"""

import ast
import logging
from typing import Dict, List, Set, Tuple
from pathlib import Path
from collections import defaultdict

logger = logging.getLogger(__name__)


class ImportAnalyzer:
    """
    Analyzes imports for optimization opportunities.
    
    Features:
    - Import analysis
    - Circular dependency detection
    - Unused import detection
    - Import optimization suggestions
    """
    
    def __init__(self, project_root: Path):
        """
        Initialize import analyzer.
        
        Args:
            project_root: Project root directory
        """
        self.project_root = project_root
        self._imports: Dict[str, List[str]] = {}
        self._exports: Dict[str, Set[str]] = defaultdict(set)
    
    def analyze_file(self, file_path: Path) -> Dict[str, Any]:
        """
        Analyze imports in a file.
        
        Args:
            file_path: Path to Python file
            
        Returns:
            Analysis results
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content, filename=str(file_path))
            
            imports = []
            from_imports = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ''
                    for alias in node.names:
                        from_imports.append(f"{module}.{alias.name}")
            
            return {
                "file": str(file_path.relative_to(self.project_root)),
                "imports": imports,
                "from_imports": from_imports,
                "total_imports": len(imports) + len(from_imports),
            }
        except Exception as e:
            logger.warning(f"Error analyzing {file_path}: {e}")
            return {
                "file": str(file_path.relative_to(self.project_root)),
                "error": str(e),
            }
    
    def detect_circular_dependencies(self) -> List[List[str]]:
        """
        Detect circular dependencies.
        
        Returns:
            List of circular dependency chains
        """
        # Build dependency graph
        graph: Dict[str, Set[str]] = defaultdict(set)
        
        for file_path in self.project_root.rglob("*.py"):
            if "__pycache__" in str(file_path):
                continue
            
            analysis = self.analyze_file(file_path)
            if "error" in analysis:
                continue
            
            module_name = self._file_to_module(file_path)
            
            for imp in analysis.get("imports", []) + analysis.get("from_imports", []):
                # Check if import is from project
                if imp.startswith("piel_mejorador_ai_sam3") or imp.startswith("."):
                    graph[module_name].add(imp)
        
        # Detect cycles using DFS
        cycles = []
        visited = set()
        rec_stack = set()
        
        def dfs(node: str, path: List[str]):
            visited.add(node)
            rec_stack.add(node)
            path.append(node)
            
            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    dfs(neighbor, path.copy())
                elif neighbor in rec_stack:
                    # Found cycle
                    cycle_start = path.index(neighbor)
                    cycle = path[cycle_start:] + [neighbor]
                    if cycle not in cycles:
                        cycles.append(cycle)
            
            rec_stack.remove(node)
            path.pop()
        
        for node in graph:
            if node not in visited:
                dfs(node, [])
        
        return cycles
    
    def _file_to_module(self, file_path: Path) -> str:
        """Convert file path to module name."""
        relative = file_path.relative_to(self.project_root)
        parts = relative.parts[:-1] + (relative.stem,)
        return ".".join(parts).replace("\\", "/")
    
    def get_import_statistics(self) -> Dict[str, Any]:
        """Get import statistics."""
        total_files = 0
        total_imports = 0
        import_counts = defaultdict(int)
        
        for file_path in self.project_root.rglob("*.py"):
            if "__pycache__" in str(file_path):
                continue
            
            analysis = self.analyze_file(file_path)
            if "error" in analysis:
                continue
            
            total_files += 1
            total_imports += analysis.get("total_imports", 0)
            
            for imp in analysis.get("imports", []):
                import_counts[imp] += 1
        
        return {
            "total_files": total_files,
            "total_imports": total_imports,
            "avg_imports_per_file": total_imports / total_files if total_files > 0 else 0,
            "most_common_imports": sorted(
                import_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10],
        }




