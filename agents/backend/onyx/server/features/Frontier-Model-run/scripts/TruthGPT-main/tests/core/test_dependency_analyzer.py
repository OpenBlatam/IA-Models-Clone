"""
Test Dependency Analyzer
Analyzes dependencies between tests and test execution order
"""

import ast
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import defaultdict

class TestDependencyAnalyzer:
    """Analyze dependencies between tests"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.tests_dir = project_root / "tests"
        self.core_dir = project_root / "core"
        
    def analyze_test_dependencies(self) -> Dict:
        """Analyze dependencies between tests"""
        test_files = list(self.tests_dir.glob("test_*.py"))
        
        dependencies = defaultdict(set)
        test_classes = {}
        test_methods = {}
        
        # Analyze each test file
        for test_file in test_files:
            try:
                with open(test_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    tree = ast.parse(content, filename=str(test_file))
                
                # Find test classes and methods
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        class_name = node.name
                        test_classes[class_name] = {
                            'file': test_file.name,
                            'methods': []
                        }
                        
                        # Find test methods
                        for item in node.body:
                            if isinstance(item, ast.FunctionDef) and item.name.startswith('test_'):
                                method_name = item.name
                                full_name = f"{class_name}.{method_name}"
                                test_methods[full_name] = {
                                    'class': class_name,
                                    'file': test_file.name,
                                    'dependencies': set()
                                }
                                test_classes[class_name]['methods'].append(method_name)
                                
                                # Find dependencies (imports, function calls)
                                for subnode in ast.walk(item):
                                    if isinstance(subnode, ast.Call):
                                        if isinstance(subnode.func, ast.Name):
                                            dependencies[full_name].add(subnode.func.id)
                                    elif isinstance(subnode, ast.Attribute):
                                        if isinstance(subnode.value, ast.Name):
                                            dependencies[full_name].add(f"{subnode.value.id}.{subnode.attr}")
            except Exception as e:
                print(f"⚠️  Error analyzing {test_file}: {e}")
                continue
        
        # Analyze core module dependencies
        core_dependencies = self._analyze_core_dependencies()
        
        return {
            'test_classes': test_classes,
            'test_methods': test_methods,
            'dependencies': dict(dependencies),
            'core_dependencies': core_dependencies,
            'test_files': [f.name for f in test_files]
        }
    
    def _analyze_core_dependencies(self) -> Dict:
        """Analyze dependencies on core modules"""
        core_files = list(self.core_dir.glob("*.py")) if self.core_dir.exists() else []
        
        dependencies = {}
        for core_file in core_files:
            module_name = core_file.stem
            dependencies[module_name] = {
                'file': core_file.name,
                'used_by_tests': []
            }
        
        # Find which tests use which core modules
        test_files = list(self.tests_dir.glob("test_*.py"))
        for test_file in test_files:
            try:
                with open(test_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                    for module_name in dependencies.keys():
                        if module_name in content or f"core.{module_name}" in content:
                            dependencies[module_name]['used_by_tests'].append(test_file.name)
            except Exception:
                continue
        
        return dependencies
    
    def find_isolated_tests(self, analysis: Dict) -> List[str]:
        """Find tests that don't depend on other tests"""
        isolated = []
        
        for test_name, deps in analysis['dependencies'].items():
            # Check if dependencies are only to core modules or utilities
            test_deps = {d for d in deps if d.startswith('test_') or 'Test' in d}
            if not test_deps:
                isolated.append(test_name)
        
        return isolated
    
    def find_test_clusters(self, analysis: Dict) -> List[List[str]]:
        """Find groups of tests that depend on each other"""
        clusters = []
        visited = set()
        
        def dfs(test_name: str, cluster: List[str]):
            if test_name in visited:
                return
            visited.add(test_name)
            cluster.append(test_name)
            
            deps = analysis['dependencies'].get(test_name, set())
            for dep in deps:
                if dep in analysis['test_methods'] and dep not in visited:
                    dfs(dep, cluster)
        
        for test_name in analysis['test_methods'].keys():
            if test_name not in visited:
                cluster = []
                dfs(test_name, cluster)
                if len(cluster) > 1:
                    clusters.append(cluster)
        
        return clusters
    
    def generate_dependency_report(self, analysis: Dict) -> str:
        """Generate dependency analysis report"""
        lines = []
        lines.append("=" * 80)
        lines.append("TEST DEPENDENCY ANALYSIS")
        lines.append("=" * 80)
        lines.append("")
        
        lines.append("📊 SUMMARY")
        lines.append("-" * 80)
        lines.append(f"Test Files:        {len(analysis['test_files'])}")
        lines.append(f"Test Classes:      {len(analysis['test_classes'])}")
        lines.append(f"Test Methods:      {len(analysis['test_methods'])}")
        lines.append("")
        
        # Core dependencies
        lines.append("🔗 CORE MODULE DEPENDENCIES")
        lines.append("-" * 80)
        for module_name, info in sorted(analysis['core_dependencies'].items()):
            test_count = len(info['used_by_tests'])
            lines.append(f"  {module_name:20s} - Used by {test_count} test file(s)")
        lines.append("")
        
        # Isolated tests
        isolated = self.find_isolated_tests(analysis)
        lines.append(f"✅ ISOLATED TESTS ({len(isolated)})")
        lines.append("-" * 80)
        lines.append("Tests that don't depend on other tests (good for parallelization):")
        for test in isolated[:10]:  # Show first 10
            lines.append(f"  • {test}")
        if len(isolated) > 10:
            lines.append(f"  ... and {len(isolated) - 10} more")
        lines.append("")
        
        # Test clusters
        clusters = self.find_test_clusters(analysis)
        if clusters:
            lines.append(f"🔗 TEST CLUSTERS ({len(clusters)})")
            lines.append("-" * 80)
            lines.append("Groups of tests that depend on each other:")
            for i, cluster in enumerate(clusters[:5], 1):  # Show first 5
                lines.append(f"  Cluster {i} ({len(cluster)} tests):")
                for test in cluster[:5]:  # Show first 5 in cluster
                    lines.append(f"    • {test}")
                if len(cluster) > 5:
                    lines.append(f"    ... and {len(cluster) - 5} more")
            lines.append("")
        
        # Recommendations
        lines.append("💡 RECOMMENDATIONS")
        lines.append("-" * 80)
        lines.append("1. Isolated tests can run in parallel safely")
        lines.append("2. Test clusters should run sequentially or in order")
        lines.append("3. Reduce dependencies between tests for better parallelization")
        lines.append("4. Use fixtures and setup/teardown for shared state")
        
        return "\n".join(lines)

def main():
    """Main function"""
    from pathlib import Path
    project_root = Path(__file__).parent.parent
    
    analyzer = TestDependencyAnalyzer(project_root)
    analysis = analyzer.analyze_test_dependencies()
    
    report = analyzer.generate_dependency_report(analysis)
    print(report)
    
    # Save report
    report_file = project_root / "dependency_analysis_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\n📄 Dependency analysis report saved to: {report_file}")

if __name__ == "__main__":
    main()







