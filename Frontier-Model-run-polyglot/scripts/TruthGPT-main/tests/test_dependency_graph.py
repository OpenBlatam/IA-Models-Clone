"""
Test Dependency Graph Visualizer
Creates and visualizes dependencies between tests and source code
"""

import ast
import json
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple
from collections import defaultdict
import subprocess


class TestDependencyGraph:
    """Build and visualize test dependency graph"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.source_root = project_root.parent
        self.graph = defaultdict(set)  # test -> {dependencies}
        self.reverse_graph = defaultdict(set)  # source -> {tests}
        self._build_graph()
    
    def _build_graph(self):
        """Build dependency graph by analyzing imports"""
        # Find all test files
        test_files = list(self.project_root.rglob("test_*.py"))
        test_files.extend(list(self.project_root.rglob("*_test.py")))
        
        # Find all source files
        source_files = list(self.source_root.rglob("*.py"))
        source_files = [f for f in source_files if "test" not in str(f) and "__pycache__" not in str(f)]
        
        # Build module path mapping
        module_map = {}
        for source_file in source_files:
            rel_path = source_file.relative_to(self.source_root)
            module_path = str(rel_path).replace('/', '.').replace('\\', '.').replace('.py', '')
            module_map[module_path] = source_file
            # Also map without extension
            module_map[str(rel_path).replace('/', '.').replace('\\', '.')] = source_file
        
        # Analyze test files
        for test_file in test_files:
            try:
                with open(test_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    tree = ast.parse(content)
                    
                    # Extract imports
                    imports = []
                    for node in ast.walk(tree):
                        if isinstance(node, ast.Import):
                            for alias in node.names:
                                imports.append(alias.name)
                        elif isinstance(node, ast.ImportFrom):
                            if node.module:
                                imports.append(node.module)
                    
                    # Map imports to source files
                    for imp in imports:
                        # Try exact match
                        if imp in module_map:
                            source_file = module_map[imp]
                            self.graph[test_file].add(source_file)
                            self.reverse_graph[source_file].add(test_file)
                        else:
                            # Try partial matches
                            for module_path, source_file in module_map.items():
                                if imp in module_path or module_path.endswith(imp):
                                    self.graph[test_file].add(source_file)
                                    self.reverse_graph[source_file].add(test_file)
                                    break
            except Exception as e:
                print(f"Warning: Could not parse {test_file}: {e}")
    
    def get_test_dependencies(self, test_file: Path) -> Set[Path]:
        """Get all source files a test depends on"""
        return self.graph.get(test_file, set())
    
    def get_affected_tests(self, source_file: Path) -> Set[Path]:
        """Get all tests that depend on a source file"""
        return self.reverse_graph.get(source_file, set())
    
    def get_dependency_chain(self, test_file: Path, max_depth: int = 3) -> Dict:
        """Get full dependency chain for a test"""
        visited = set()
        chain = {
            'test': str(test_file.relative_to(self.project_root)),
            'dependencies': [],
            'depth': 0
        }
        
        def traverse(test: Path, depth: int):
            if depth > max_depth or test in visited:
                return
            
            visited.add(test)
            deps = self.get_test_dependencies(test)
            
            for dep in deps:
                dep_info = {
                    'file': str(dep.relative_to(self.source_root)),
                    'tests_affected': len(self.get_affected_tests(dep)),
                    'dependencies': []
                }
                
                # Recursively get dependencies of dependencies
                if depth < max_depth:
                    # Find tests that test this dependency
                    dependent_tests = self.get_affected_tests(dep)
                    for dep_test in dependent_tests:
                        if dep_test != test:
                            traverse(dep_test, depth + 1)
                
                chain['dependencies'].append(dep_info)
        
        traverse(test_file, 0)
        return chain
    
    def export_graph_json(self, output_file: Path = None) -> Dict:
        """Export graph as JSON"""
        if output_file is None:
            output_file = self.project_root / "test_dependency_graph.json"
        
        graph_data = {
            'nodes': [],
            'edges': []
        }
        
        # Add test nodes
        test_nodes = {}
        for i, test_file in enumerate(self.graph.keys()):
            node_id = f"test_{i}"
            test_nodes[test_file] = node_id
            graph_data['nodes'].append({
                'id': node_id,
                'label': str(test_file.relative_to(self.project_root)),
                'type': 'test'
            })
        
        # Add source nodes
        source_nodes = {}
        all_sources = set()
        for deps in self.graph.values():
            all_sources.update(deps)
        
        for i, source_file in enumerate(all_sources):
            node_id = f"source_{i}"
            source_nodes[source_file] = node_id
            graph_data['nodes'].append({
                'id': node_id,
                'label': str(source_file.relative_to(self.source_root)),
                'type': 'source'
            })
        
        # Add edges
        for test_file, deps in self.graph.items():
            test_id = test_nodes[test_file]
            for dep in deps:
                if dep in source_nodes:
                    dep_id = source_nodes[dep]
                    graph_data['edges'].append({
                        'from': test_id,
                        'to': dep_id
                    })
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(graph_data, f, indent=2)
        
        print(f"✅ Graph exported to {output_file}")
        return graph_data
    
    def generate_dot_file(self, output_file: Path = None) -> str:
        """Generate Graphviz DOT format"""
        if output_file is None:
            output_file = self.project_root / "test_dependency_graph.dot"
        
        lines = ["digraph TestDependencies {"]
        lines.append("  rankdir=LR;")
        lines.append("  node [shape=box];")
        lines.append("")
        
        # Add test nodes
        test_nodes = {}
        for i, test_file in enumerate(self.graph.keys()):
            node_id = f"test_{i}"
            test_nodes[test_file] = node_id
            label = str(test_file.relative_to(self.project_root)).replace('\\', '/')
            lines.append(f'  {node_id} [label="{label}", style=filled, fillcolor=lightblue];')
        
        # Add source nodes
        source_nodes = {}
        all_sources = set()
        for deps in self.graph.values():
            all_sources.update(deps)
        
        for i, source_file in enumerate(all_sources):
            node_id = f"source_{i}"
            source_nodes[source_file] = node_id
            label = str(source_file.relative_to(self.source_root)).replace('\\', '/')
            lines.append(f'  {node_id} [label="{label}", style=filled, fillcolor=lightgreen];')
        
        lines.append("")
        
        # Add edges
        for test_file, deps in self.graph.items():
            test_id = test_nodes[test_file]
            for dep in deps:
                if dep in source_nodes:
                    dep_id = source_nodes[dep]
                    lines.append(f"  {test_id} -> {dep_id};")
        
        lines.append("}")
        
        dot_content = "\n".join(lines)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(dot_content)
        
        print(f"✅ DOT file generated: {output_file}")
        print(f"   Render with: dot -Tpng {output_file} -o graph.png")
        
        return dot_content
    
    def get_statistics(self) -> Dict:
        """Get graph statistics"""
        total_tests = len(self.graph)
        total_sources = len(self.reverse_graph)
        
        # Calculate average dependencies per test
        if total_tests > 0:
            avg_deps = sum(len(deps) for deps in self.graph.values()) / total_tests
        else:
            avg_deps = 0
        
        # Find most tested files
        most_tested = sorted(
            self.reverse_graph.items(),
            key=lambda x: len(x[1]),
            reverse=True
        )[:10]
        
        # Find tests with most dependencies
        most_deps = sorted(
            self.graph.items(),
            key=lambda x: len(x[1]),
            reverse=True
        )[:10]
        
        return {
            'total_tests': total_tests,
            'total_source_files': total_sources,
            'average_dependencies_per_test': round(avg_deps, 2),
            'most_tested_files': [
                {
                    'file': str(f.relative_to(self.source_root)),
                    'test_count': len(tests)
                }
                for f, tests in most_tested
            ],
            'tests_with_most_dependencies': [
                {
                    'test': str(t.relative_to(self.project_root)),
                    'dependency_count': len(deps)
                }
                for t, deps in most_deps
            ]
        }


def main():
    """CLI interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Dependency Graph')
    parser.add_argument('--export-json', type=str, help='Export graph as JSON')
    parser.add_argument('--export-dot', type=str, help='Export graph as DOT')
    parser.add_argument('--stats', action='store_true', help='Show statistics')
    parser.add_argument('--test', type=str, help='Show dependencies for specific test')
    parser.add_argument('--source', type=str, help='Show tests for specific source file')
    parser.add_argument('--project-root', type=str, help='Project root directory')
    
    args = parser.parse_args()
    
    project_root = Path(args.project_root) if args.project_root else Path(__file__).parent
    
    print("🔍 Building dependency graph...")
    graph = TestDependencyGraph(project_root)
    
    if args.export_json:
        graph.export_graph_json(Path(args.export_json))
    elif args.export_dot:
        graph.generate_dot_file(Path(args.export_dot))
    elif args.stats:
        stats = graph.get_statistics()
        print(f"\n📊 Graph Statistics:")
        print(f"  Total Tests: {stats['total_tests']}")
        print(f"  Total Source Files: {stats['total_source_files']}")
        print(f"  Avg Dependencies/Test: {stats['average_dependencies_per_test']}")
        print(f"\n  Most Tested Files:")
        for item in stats['most_tested_files'][:5]:
            print(f"    {item['file']}: {item['test_count']} tests")
        print(f"\n  Tests with Most Dependencies:")
        for item in stats['tests_with_most_dependencies'][:5]:
            print(f"    {item['test']}: {item['dependency_count']} deps")
    elif args.test:
        test_file = project_root / args.test
        deps = graph.get_test_dependencies(test_file)
        print(f"\n📋 Dependencies for {args.test}:")
        for dep in sorted(deps):
            print(f"  - {dep.relative_to(project_root.parent)}")
    elif args.source:
        source_file = project_root.parent / args.source
        tests = graph.get_affected_tests(source_file)
        print(f"\n📋 Tests affected by {args.source}:")
        for test in sorted(tests):
            print(f"  - {test.relative_to(project_root)}")
    else:
        print("Use --help to see available options")


if __name__ == '__main__':
    main()

