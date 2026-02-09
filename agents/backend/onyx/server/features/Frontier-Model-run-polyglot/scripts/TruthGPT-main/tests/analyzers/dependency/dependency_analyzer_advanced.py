"""
Advanced Dependency Analyzer
Advanced test dependency analysis
"""

import json
from pathlib import Path
from typing import Dict, List, Set
from datetime import datetime, timedelta
from collections import defaultdict

class AdvancedDependencyAnalyzer:
    """Advanced dependency analysis"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.history_file = project_root / "test_history.json"
        self.dependencies_file = project_root / "test_dependencies.json"
    
    def analyze_dependencies(self, lookback_days: int = 30) -> Dict:
        """Analyze test dependencies"""
        history = self._load_history()
        dependencies = self._load_dependencies()
        
        cutoff_date = (datetime.now() - timedelta(days=lookback_days)).isoformat()
        recent = [r for r in history if r.get('timestamp', '') >= cutoff_date]
        
        if not recent:
            return {'error': 'Insufficient data'}
        
        # Analyze dependency patterns
        dependency_analysis = {
            'period': f'Last {lookback_days} days',
            'total_runs': len(recent),
            'dependency_graph': self._build_dependency_graph(dependencies),
            'circular_dependencies': self._detect_circular_dependencies(dependencies),
            'critical_paths': self._find_critical_paths(dependencies),
            'dependency_metrics': self._calculate_dependency_metrics(dependencies),
            'recommendations': []
        }
        
        # Generate recommendations
        dependency_analysis['recommendations'] = self._generate_dependency_recommendations(dependency_analysis)
        
        return dependency_analysis
    
    def _build_dependency_graph(self, dependencies: Dict) -> Dict:
        """Build dependency graph"""
        graph = defaultdict(list)
        
        for test, deps in dependencies.items():
            graph[test] = deps
        
        return dict(graph)
    
    def _detect_circular_dependencies(self, dependencies: Dict) -> List[List[str]]:
        """Detect circular dependencies"""
        circular = []
        visited = set()
        rec_stack = set()
        
        def dfs(node: str, path: List[str]):
            if node in rec_stack:
                # Found cycle
                cycle_start = path.index(node)
                cycle = path[cycle_start:] + [node]
                if cycle not in circular:
                    circular.append(cycle)
                return
            
            if node in visited:
                return
            
            visited.add(node)
            rec_stack.add(node)
            
            for dep in dependencies.get(node, []):
                dfs(dep, path + [node])
            
            rec_stack.remove(node)
        
        for test in dependencies:
            if test not in visited:
                dfs(test, [])
        
        return circular
    
    def _find_critical_paths(self, dependencies: Dict) -> List[List[str]]:
        """Find critical dependency paths"""
        critical_paths = []
        
        # Find longest paths (simplified)
        def find_longest_path(node: str, path: List[str], visited: Set[str]) -> List[str]:
            if node in visited:
                return path
            
            visited.add(node)
            current_path = path + [node]
            
            longest = current_path
            for dep in dependencies.get(node, []):
                dep_path = find_longest_path(dep, current_path, visited.copy())
                if len(dep_path) > len(longest):
                    longest = dep_path
            
            return longest
        
        for test in dependencies:
            path = find_longest_path(test, [], set())
            if len(path) > 3:  # Critical if path length > 3
                critical_paths.append(path)
        
        return critical_paths[:10]  # Top 10
    
    def _calculate_dependency_metrics(self, dependencies: Dict) -> Dict:
        """Calculate dependency metrics"""
        if not dependencies:
            return {}
        
        total_tests = len(dependencies)
        total_dependencies = sum(len(deps) for deps in dependencies.values())
        
        # Calculate average dependencies per test
        avg_dependencies = total_dependencies / total_tests if total_tests > 0 else 0
        
        # Find most dependent tests
        most_dependent = sorted(
            dependencies.items(),
            key=lambda x: len(x[1]),
            reverse=True
        )[:5]
        
        # Find most depended upon tests
        depended_upon = defaultdict(int)
        for deps in dependencies.values():
            for dep in deps:
                depended_upon[dep] += 1
        
        most_depended_upon = sorted(
            depended_upon.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        return {
            'total_tests': total_tests,
            'total_dependencies': total_dependencies,
            'avg_dependencies_per_test': round(avg_dependencies, 2),
            'most_dependent_tests': [{'test': t, 'dependencies': len(d)} for t, d in most_dependent],
            'most_depended_upon_tests': [{'test': t, 'dependents': c} for t, c in most_depended_upon]
        }
    
    def _generate_dependency_recommendations(self, analysis: Dict) -> List[str]:
        """Generate dependency recommendations"""
        recommendations = []
        
        if analysis['circular_dependencies']:
            recommendations.append(f"Found {len(analysis['circular_dependencies'])} circular dependencies - break cycles")
        
        if analysis['critical_paths']:
            recommendations.append(f"Found {len(analysis['critical_paths'])} critical paths - consider refactoring")
        
        metrics = analysis['dependency_metrics']
        if metrics.get('avg_dependencies_per_test', 0) > 5:
            recommendations.append("High average dependencies per test - consider reducing coupling")
        
        if not recommendations:
            recommendations.append("Dependency structure is healthy - maintain current practices")
        
        return recommendations
    
    def generate_dependency_report(self, analysis: Dict) -> str:
        """Generate dependency report"""
        lines = []
        lines.append("=" * 80)
        lines.append("ADVANCED DEPENDENCY ANALYSIS REPORT")
        lines.append("=" * 80)
        lines.append("")
        
        if 'error' in analysis:
            lines.append(f"❌ {analysis['error']}")
            return "\n".join(lines)
        
        lines.append(f"Period: {analysis['period']}")
        lines.append(f"Total Runs: {analysis['total_runs']}")
        lines.append("")
        
        metrics = analysis['dependency_metrics']
        lines.append("📊 DEPENDENCY METRICS")
        lines.append("-" * 80)
        lines.append(f"Total Tests: {metrics.get('total_tests', 0)}")
        lines.append(f"Total Dependencies: {metrics.get('total_dependencies', 0)}")
        lines.append(f"Average Dependencies per Test: {metrics.get('avg_dependencies_per_test', 0)}")
        lines.append("")
        
        if metrics.get('most_dependent_tests'):
            lines.append("🔗 MOST DEPENDENT TESTS")
            lines.append("-" * 80)
            for item in metrics['most_dependent_tests']:
                lines.append(f"  {item['test']}: {item['dependencies']} dependencies")
            lines.append("")
        
        if metrics.get('most_depended_upon_tests'):
            lines.append("⭐ MOST DEPENDED UPON TESTS")
            lines.append("-" * 80)
            for item in metrics['most_depended_upon_tests']:
                lines.append(f"  {item['test']}: {item['dependents']} dependents")
            lines.append("")
        
        if analysis['circular_dependencies']:
            lines.append("⚠️ CIRCULAR DEPENDENCIES")
            lines.append("-" * 80)
            for i, cycle in enumerate(analysis['circular_dependencies'], 1):
                lines.append(f"{i}. {' → '.join(cycle)}")
            lines.append("")
        
        if analysis['critical_paths']:
            lines.append("🔴 CRITICAL PATHS")
            lines.append("-" * 80)
            for i, path in enumerate(analysis['critical_paths'][:5], 1):
                lines.append(f"{i}. {' → '.join(path)}")
            lines.append("")
        
        if analysis['recommendations']:
            lines.append("💡 RECOMMENDATIONS")
            lines.append("-" * 80)
            for rec in analysis['recommendations']:
                lines.append(f"• {rec}")
        
        return "\n".join(lines)
    
    def _load_history(self) -> List[Dict]:
        """Load test history"""
        if not self.history_file.exists():
            return []
        
        try:
            with open(self.history_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return []
    
    def _load_dependencies(self) -> Dict:
        """Load dependency data"""
        if not self.dependencies_file.exists():
            # Return sample structure
            return {
                'test_1': ['test_2', 'test_3'],
                'test_2': ['test_4'],
                'test_3': ['test_4'],
                'test_4': []
            }
        
        try:
            with open(self.dependencies_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return {}

def main():
    """Main function"""
    from pathlib import Path
    from collections import defaultdict
    
    project_root = Path(__file__).parent.parent
    
    analyzer = AdvancedDependencyAnalyzer(project_root)
    analysis = analyzer.analyze_dependencies(lookback_days=30)
    
    report = analyzer.generate_dependency_report(analysis)
    print(report)
    
    # Save report
    report_file = project_root / "advanced_dependency_analysis_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\n📄 Advanced dependency analysis report saved to: {report_file}")

if __name__ == "__main__":
    main()







