"""
Dependency Visualizer
Visualize test dependencies
"""

import json
from pathlib import Path
from typing import Dict, List, Set
import re

class DependencyVisualizer:
    """Visualize test dependencies"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.results_dir = project_root / "test_results"
    
    def analyze_dependencies(self) -> Dict:
        """Analyze test dependencies"""
        dependencies = {}
        test_files = {}
        
        # Scan test files for imports and dependencies
        tests_dir = self.project_root / "tests"
        
        if tests_dir.exists():
            for test_file in tests_dir.glob("test_*.py"):
                try:
                    with open(test_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Extract imports
                    imports = re.findall(r'^import\s+(\S+)|^from\s+(\S+)\s+import', content, re.MULTILINE)
                    module_imports = [imp[0] or imp[1] for imp in imports]
                    
                    # Extract test class/function names
                    test_classes = re.findall(r'class\s+(Test\w+)', content)
                    test_functions = re.findall(r'def\s+(test_\w+)', content)
                    
                    test_files[str(test_file.name)] = {
                        'imports': module_imports,
                        'test_classes': test_classes,
                        'test_functions': test_functions
                    }
                except Exception:
                    continue
        
        # Build dependency graph
        for file_name, file_data in test_files.items():
            dependencies[file_name] = {
                'depends_on': file_data['imports'],
                'test_count': len(file_data['test_functions']),
                'classes': file_data['test_classes']
            }
        
        return {
            'dependencies': dependencies,
            'total_files': len(test_files),
            'total_tests': sum(d['test_count'] for d in dependencies.values())
        }
    
    def generate_dependency_graph_html(self, analysis: Dict) -> Path:
        """Generate HTML dependency graph"""
        output_path = self.project_root / "dependency_graph.html"
        
        html = """<!DOCTYPE html>
<html>
<head>
    <title>Test Dependency Graph</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        body { font-family: Arial, sans-serif; padding: 20px; }
        .node { cursor: pointer; }
        .link { stroke: #999; stroke-opacity: 0.6; }
    </style>
</head>
<body>
    <h1>Test Dependency Graph</h1>
    <p>Total Files: {total_files} | Total Tests: {total_tests}</p>
    <svg id="graph" width="1200" height="800"></svg>
    <script>
        const data = {graph_data};
        const svg = d3.select("#graph");
        // Simple visualization code here
    </script>
</body>
</html>
""".format(
            total_files=analysis['total_files'],
            total_tests=analysis['total_tests'],
            graph_data=json.dumps(analysis['dependencies'])
        )
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)
        
        return output_path
    
    def generate_dependency_report(self, analysis: Dict) -> str:
        """Generate dependency report"""
        lines = []
        lines.append("=" * 80)
        lines.append("TEST DEPENDENCY ANALYSIS")
        lines.append("=" * 80)
        lines.append("")
        
        lines.append(f"Total Test Files: {analysis['total_files']}")
        lines.append(f"Total Tests: {analysis['total_tests']}")
        lines.append("")
        
        lines.append("📦 DEPENDENCIES")
        lines.append("-" * 80)
        
        for file_name, deps in analysis['dependencies'].items():
            lines.append(f"\n{file_name}")
            lines.append(f"  Tests: {deps['test_count']}")
            lines.append(f"  Classes: {', '.join(deps['classes'])}")
            lines.append(f"  Dependencies: {len(deps['depends_on'])}")
            for dep in deps['depends_on'][:5]:
                lines.append(f"    • {dep}")
            if len(deps['depends_on']) > 5:
                lines.append(f"    ... and {len(deps['depends_on']) - 5} more")
        
        return "\n".join(lines)

def main():
    """Main function"""
    from pathlib import Path
    project_root = Path(__file__).parent.parent
    
    visualizer = DependencyVisualizer(project_root)
    analysis = visualizer.analyze_dependencies()
    
    report = visualizer.generate_dependency_report(analysis)
    print(report)
    
    html_path = visualizer.generate_dependency_graph_html(analysis)
    print(f"\n✅ Dependency graph generated: {html_path}")

if __name__ == "__main__":
    main()







