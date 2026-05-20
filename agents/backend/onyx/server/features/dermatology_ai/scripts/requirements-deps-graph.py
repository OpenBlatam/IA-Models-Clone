#!/usr/bin/env python3
"""
Dependency Graph Generator
Generates a graph of package dependencies
"""

import sys
import subprocess
from pathlib import Path
from collections import defaultdict

try:
    import graphviz
    HAS_GRAPHVIZ = True
except ImportError:
    HAS_GRAPHVIZ = False
    print("graphviz not available, using text output")
    print("Install with: pip install graphviz")


def get_package_dependencies(package_name):
    """Get dependencies of a package"""
    try:
        result = subprocess.run(
            ['pip', 'show', package_name],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            requires = []
            in_requires = False
            for line in result.stdout.split('\n'):
                if line.startswith('Requires:'):
                    in_requires = True
                    req = line.split(':', 1)[1].strip()
                    if req:
                        requires.append(req.split()[0])
                elif in_requires and line.startswith(' '):
                    requires.append(line.strip().split()[0])
                elif in_requires and line and not line.startswith(' '):
                    break
            return requires
    except:
        pass
    return []


def build_dependency_graph(packages, max_depth=2):
    """Build dependency graph"""
    graph = defaultdict(set)
    
    for package in packages[:15]:  # Limit for performance
        deps = get_package_dependencies(package)
        for dep in deps:
            graph[package].add(dep)
    
    return graph


def create_graphviz_graph(graph, output_file='dependencies-graph'):
    """Create Graphviz graph"""
    if not HAS_GRAPHVIZ:
        return False
    
    dot = graphviz.Digraph(comment='Dependencies')
    dot.attr(rankdir='LR')
    dot.attr('node', shape='box')
    
    # Add nodes and edges
    for package, deps in graph.items():
        dot.node(package, package)
        for dep in deps:
            dot.node(dep, dep)
            dot.edge(package, dep)
    
    # Render
    dot.render(output_file, format='png', cleanup=True)
    return True


def print_text_graph(graph):
    """Print text-based graph"""
    print("Dependency Graph")
    print("=" * 60)
    print()
    
    for package, deps in sorted(graph.items()):
        if deps:
            print(f"{package}:")
            for dep in sorted(deps):
                print(f"  └─ {dep}")
            print()


def main():
    """Main function"""
    if len(sys.argv) < 2:
        filepath = Path('requirements.txt')
    else:
        filepath = Path(sys.argv[1])
    
    if not filepath.exists():
        print(f"Error: {filepath} not found")
        sys.exit(1)
    
    # Parse requirements
    packages = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and not line.startswith('-r'):
                package = line.split('>=')[0].split('==')[0].split('[')[0].strip()
                if package:
                    packages.append(package)
    
    print(f"Building dependency graph for {len(packages)} packages...")
    print("(This may take a while)")
    print()
    
    graph = build_dependency_graph(packages, max_depth=2)
    
    if HAS_GRAPHVIZ:
        if create_graphviz_graph(graph):
            print("✓ Graph saved to: dependencies-graph.png")
        else:
            print_text_graph(graph)
    else:
        print_text_graph(graph)
        print("\n💡 Install graphviz for visual graph: pip install graphviz")


if __name__ == '__main__':
    main()



