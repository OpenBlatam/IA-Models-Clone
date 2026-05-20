#!/usr/bin/env python3
"""
Dependency Tree Visualizer
Shows dependency relationships and tree structure
"""

import sys
from pathlib import Path
import subprocess
from collections import defaultdict


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
                        requires.append(req)
                elif in_requires and line.startswith(' '):
                    requires.append(line.strip())
                elif in_requires and line and not line.startswith(' '):
                    break
            return requires
    except:
        pass
    return []


def build_dependency_tree(packages, max_depth=3, current_depth=0):
    """Build dependency tree"""
    if current_depth >= max_depth:
        return {}
    
    tree = {}
    for package in packages[:10]:  # Limit to first 10 for performance
        deps = get_package_dependencies(package)
        if deps:
            tree[package] = build_dependency_tree(deps, max_depth, current_depth + 1)
        else:
            tree[package] = {}
    
    return tree


def print_tree(tree, prefix="", is_last=True):
    """Print dependency tree"""
    items = list(tree.items())
    for i, (package, deps) in enumerate(items):
        is_last_item = i == len(items) - 1
        current_prefix = "└── " if is_last_item else "├── "
        print(f"{prefix}{current_prefix}{package}")
        
        if deps:
            next_prefix = prefix + ("    " if is_last_item else "│   ")
            print_tree(deps, next_prefix, is_last_item)


def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("Usage: python dependency-tree.py <requirements-file>")
        sys.exit(1)
    
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
    
    print(f"Dependency Tree for {filepath.name}")
    print("=" * 60)
    print()
    
    # Build and print tree
    tree = build_dependency_tree(packages, max_depth=2)
    print_tree(tree)
    
    print()
    print("Note: Showing first 10 packages and 2 levels deep")
    print("Install packages first: pip install -r requirements.txt")


if __name__ == '__main__':
    main()



