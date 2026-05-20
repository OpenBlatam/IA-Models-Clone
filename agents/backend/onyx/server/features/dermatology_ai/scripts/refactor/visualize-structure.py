#!/usr/bin/env python3
"""
Visualize Project Structure
Creates a visual representation of the project structure
"""

import os
import sys
from pathlib import Path
from typing import Dict, List

try:
    from rich.console import Console
    from rich.tree import Tree
    from rich.text import Text
    HAS_RICH = True
except ImportError:
    HAS_RICH = False


def build_tree_structure(root_dir: Path, max_depth: int = 3) -> Dict:
    """Build tree structure"""
    structure = {}
    
    for root, dirs, files in os.walk(root_dir):
        # Skip hidden and common ignore dirs
        dirs[:] = [d for d in dirs if not d.startswith('.') and 
                  d not in ['__pycache__', 'node_modules', '.git', 'venv', '.venv']]
        
        rel_path = Path(root).relative_to(root_dir)
        depth = len(rel_path.parts)
        
        if depth > max_depth:
            continue
        
        # Count files
        py_files = [f for f in files if f.endswith('.py')]
        md_files = [f for f in files if f.endswith('.md')]
        other_files = [f for f in files if not f.endswith(('.py', '.md'))]
        
        if py_files or md_files or other_files or dirs:
            structure[str(rel_path)] = {
                'py_files': len(py_files),
                'md_files': len(md_files),
                'other_files': len(other_files),
                'dirs': dirs
            }
    
    return structure


def create_rich_tree(structure: Dict, root_dir: Path) -> Tree:
    """Create rich tree visualization"""
    console = Console()
    tree = Tree(f"[bold blue]{root_dir.name}[/bold blue]")
    
    # Organize by main directories
    main_dirs = ['services', 'utils', 'scripts', 'docs', 'config', 'api', 'core', 'tests']
    
    for main_dir in main_dirs:
        if main_dir in structure:
            branch = tree.add(f"[green]{main_dir}/[/green]")
            add_branch_content(branch, structure, main_dir, max_depth=2)
    
    # Add other directories
    for path, data in structure.items():
        if not any(path.startswith(md + '/') for md in main_dirs) and '/' not in path:
            if data['py_files'] > 0 or data['md_files'] > 0:
                tree.add(f"[yellow]{path}/[/yellow] ({data['py_files']} py, {data['md_files']} md)")
    
    return tree


def add_branch_content(branch, structure: Dict, prefix: str, max_depth: int = 2):
    """Add content to branch"""
    for path, data in structure.items():
        if path.startswith(prefix + '/') and path.count('/') <= max_depth:
            rel_path = path.replace(prefix + '/', '')
            if '/' not in rel_path:  # Direct child
                count_str = ""
                if data['py_files'] > 0:
                    count_str += f"{data['py_files']} py"
                if data['md_files'] > 0:
                    if count_str:
                        count_str += ", "
                    count_str += f"{data['md_files']} md"
                
                if count_str:
                    branch.add(f"[cyan]{rel_path}/[/cyan] ({count_str})")
                else:
                    branch.add(f"[cyan]{rel_path}/[/cyan]")


def create_text_tree(structure: Dict, root_dir: Path) -> str:
    """Create text tree visualization"""
    output = f"{root_dir.name}\n"
    output += "=" * 60 + "\n\n"
    
    main_dirs = ['services', 'utils', 'scripts', 'docs', 'config']
    
    for main_dir in main_dirs:
        if main_dir in structure:
            output += f"{main_dir}/\n"
            for path, data in structure.items():
                if path.startswith(main_dir + '/') and path.count('/') <= 2:
                    rel_path = path.replace(main_dir + '/', '')
                    indent = "  " * (path.count('/'))
                    count_str = ""
                    if data['py_files'] > 0:
                        count_str = f" ({data['py_files']} files)"
                    output += f"{indent}├── {rel_path}/{count_str}\n"
            output += "\n"
    
    return output


def main():
    """Main function"""
    if len(sys.argv) > 1:
        root_dir = Path(sys.argv[1])
    else:
        root_dir = Path(__file__).parent.parent.parent
    
    if not root_dir.exists():
        print(f"Error: {root_dir} does not exist")
        sys.exit(1)
    
    print("Analyzing project structure...")
    structure = build_tree_structure(root_dir, max_depth=3)
    
    if HAS_RICH:
        console = Console()
        tree = create_rich_tree(structure, root_dir)
        console.print(tree)
    else:
        text_tree = create_text_tree(structure, root_dir)
        print(text_tree)
        print("\n💡 Install rich for better visualization: pip install rich")


if __name__ == '__main__':
    main()



