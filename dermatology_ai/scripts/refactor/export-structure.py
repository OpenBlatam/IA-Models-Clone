#!/usr/bin/env python3
"""
Export Structure
Exports project structure to various formats
"""

import json
import sys
from pathlib import Path
from typing import Dict, List


def analyze_structure(root_dir: Path) -> Dict:
    """Analyze project structure"""
    structure = {
        'services': {},
        'utils': {},
        'scripts': {},
        'docs': {},
        'config': {}
    }
    
    # Services
    services_dir = root_dir / 'services'
    if services_dir.exists():
        for category_dir in services_dir.iterdir():
            if category_dir.is_dir():
                files = [f.name for f in category_dir.glob('*.py')]
                structure['services'][category_dir.name] = {
                    'count': len(files),
                    'files': files[:10]  # First 10
                }
    
    # Utils
    utils_dir = root_dir / 'utils'
    if utils_dir.exists():
        for category_dir in utils_dir.iterdir():
            if category_dir.is_dir():
                files = [f.name for f in category_dir.glob('*.py')]
                structure['utils'][category_dir.name] = {
                    'count': len(files),
                    'files': files[:10]
                }
    
    # Scripts
    scripts_dir = root_dir / 'scripts'
    if scripts_dir.exists():
        req_dir = scripts_dir / 'requirements'
        if req_dir.exists():
            for category_dir in req_dir.iterdir():
                if category_dir.is_dir():
                    files = [f.name for f in category_dir.iterdir() if f.is_file()]
                    structure['scripts'][category_dir.name] = {
                        'count': len(files),
                        'files': files[:10]
                    }
    
    # Docs
    docs_dir = root_dir / 'docs'
    if docs_dir.exists():
        for category_dir in docs_dir.iterdir():
            if category_dir.is_dir():
                files = [f.name for f in category_dir.glob('*.md')]
                structure['docs'][category_dir.name] = {
                    'count': len(files),
                    'files': files[:10]
                }
    
    return structure


def export_json(structure: Dict, output_file: Path):
    """Export to JSON"""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(structure, f, indent=2)
    print(f"✓ Exported to JSON: {output_file}")


def export_markdown(structure: Dict, output_file: Path):
    """Export to Markdown"""
    content = "# Project Structure Export\n\n"
    content += "## Services\n\n"
    for category, data in structure['services'].items():
        content += f"### {category}\n"
        content += f"- Files: {data['count']}\n"
        if data['files']:
            content += "- Sample files:\n"
            for file in data['files']:
                content += f"  - {file}\n"
        content += "\n"
    
    content += "## Utils\n\n"
    for category, data in structure['utils'].items():
        content += f"### {category}\n"
        content += f"- Files: {data['count']}\n"
        if data['files']:
            content += "- Sample files:\n"
            for file in data['files']:
                content += f"  - {file}\n"
        content += "\n"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"✓ Exported to Markdown: {output_file}")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Export project structure')
    parser.add_argument('--format', choices=['json', 'md', 'both'], default='both',
                       help='Export format')
    parser.add_argument('--output-dir', type=str, help='Output directory')
    args = parser.parse_args()
    
    root_dir = Path(__file__).parent.parent.parent
    
    if not root_dir.exists():
        print(f"Error: {root_dir} does not exist")
        sys.exit(1)
    
    output_dir = Path(args.output_dir) if args.output_dir else root_dir / 'docs'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Analyzing structure...")
    structure = analyze_structure(root_dir)
    
    if args.format in ['json', 'both']:
        export_json(structure, output_dir / 'structure.json')
    
    if args.format in ['md', 'both']:
        export_markdown(structure, output_dir / 'structure.md')
    
    print("\n✓ Export completed")


if __name__ == '__main__':
    main()



