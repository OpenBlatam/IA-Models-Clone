#!/usr/bin/env python3
"""
Create Documentation Index
Creates an index of all documentation files
"""

import os
import sys
from pathlib import Path
from typing import List, Dict


def find_documentation_files(root_dir: Path) -> Dict[str, List[str]]:
    """Find all documentation files"""
    docs = {
        'architecture': [],
        'dependencies': [],
        'features': [],
        'guides': [],
        'api': [],
        'other': []
    }
    
    doc_extensions = ['.md', '.rst', '.txt']
    
    # Check docs directory
    docs_dir = root_dir / 'docs'
    if docs_dir.exists():
        for category in docs.keys():
            category_dir = docs_dir / category
            if category_dir.exists():
                for file in category_dir.iterdir():
                    if file.suffix in doc_extensions:
                        docs[category].append(file.name)
    
    # Check root for unmoved docs
    for file in root_dir.iterdir():
        if file.is_file() and file.suffix in doc_extensions:
            name_lower = file.name.lower()
            if 'architect' in name_lower:
                docs['architecture'].append(f"{file.name} (root)")
            elif 'depend' in name_lower:
                docs['dependencies'].append(f"{file.name} (root)")
            elif 'feature' in name_lower:
                docs['features'].append(f"{file.name} (root)")
            elif 'guide' in name_lower or 'quick' in name_lower:
                docs['guides'].append(f"{file.name} (root)")
            else:
                docs['other'].append(f"{file.name} (root)")
    
    return docs


def generate_index(root_dir: Path) -> str:
    """Generate documentation index"""
    docs = find_documentation_files(root_dir)
    
    index_content = "# 📚 Documentation Index\n\n"
    index_content += "Complete index of all documentation files in the project.\n\n"
    
    categories = {
        'architecture': 'Architecture Documentation',
        'dependencies': 'Dependencies Documentation',
        'features': 'Features Documentation',
        'guides': 'Guides and Tutorials',
        'api': 'API Documentation',
        'other': 'Other Documentation'
    }
    
    for category, title in categories.items():
        if docs[category]:
            index_content += f"## {title}\n\n"
            for doc in sorted(docs[category]):
                index_content += f"- [{doc}](docs/{category}/{doc})\n"
            index_content += "\n"
    
    return index_content


def main():
    """Main function"""
    if len(sys.argv) > 1:
        root_dir = Path(sys.argv[1])
    else:
        root_dir = Path(__file__).parent.parent.parent
    
    if not root_dir.exists():
        print(f"Error: {root_dir} does not exist")
        sys.exit(1)
    
    index = generate_index(root_dir)
    
    # Write to file
    index_file = root_dir / 'docs' / 'INDEX.md'
    index_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(index_file, 'w', encoding='utf-8') as f:
        f.write(index)
    
    print(f"✓ Documentation index created: {index_file}")
    print(f"  Found {sum(len(docs) for docs in find_documentation_files(root_dir).values())} documentation files")


if __name__ == '__main__':
    main()



