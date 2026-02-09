#!/usr/bin/env python3
"""
Structure Analyzer
Analyzes project structure and provides refactoring recommendations
"""

import os
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple


def analyze_directory_structure(root_dir: Path) -> Dict:
    """Analyze directory structure"""
    stats = {
        'total_files': 0,
        'total_dirs': 0,
        'files_by_extension': defaultdict(int),
        'files_in_root': [],
        'large_directories': [],
        'potential_issues': []
    }
    
    # Analyze
    for root, dirs, files in os.walk(root_dir):
        # Skip hidden and common ignore dirs
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'node_modules']]
        
        rel_path = Path(root).relative_to(root_dir)
        
        # Count files
        for file in files:
            if file.startswith('.'):
                continue
            
            stats['total_files'] += 1
            ext = Path(file).suffix or 'no-extension'
            stats['files_by_extension'][ext] += 1
            
            # Check if in root
            if rel_path == Path('.'):
                stats['files_in_root'].append(file)
        
        # Count directories
        stats['total_dirs'] += len(dirs)
        
        # Check directory size
        file_count = len([f for f in files if not f.startswith('.')])
        if file_count > 20:
            stats['large_directories'].append((str(rel_path), file_count))
    
    return stats


def analyze_documentation(root_dir: Path) -> Dict:
    """Analyze documentation files"""
    doc_stats = {
        'total_docs': 0,
        'docs_in_root': [],
        'docs_by_category': defaultdict(int),
        'recommendations': []
    }
    
    doc_extensions = ['.md', '.rst', '.txt']
    
    for root, dirs, files in os.walk(root_dir):
        rel_path = Path(root).relative_to(root_dir)
        
        for file in files:
            if Path(file).suffix in doc_extensions:
                doc_stats['total_docs'] += 1
                
                if rel_path == Path('.'):
                    doc_stats['docs_in_root'].append(file)
                
                # Categorize
                if 'architect' in file.lower():
                    doc_stats['docs_by_category']['architecture'] += 1
                elif 'depend' in file.lower():
                    doc_stats['docs_by_category']['dependencies'] += 1
                elif 'feature' in file.lower():
                    doc_stats['docs_by_category']['features'] += 1
                elif 'guide' in file.lower() or 'quick' in file.lower():
                    doc_stats['docs_by_category']['guides'] += 1
                else:
                    doc_stats['docs_by_category']['other'] += 1
    
    # Recommendations
    if len(doc_stats['docs_in_root']) > 10:
        doc_stats['recommendations'].append(
            f"Move {len(doc_stats['docs_in_root'])} documentation files from root to docs/"
        )
    
    return doc_stats


def generate_report(root_dir: Path) -> str:
    """Generate analysis report"""
    print("Analyzing project structure...")
    print("=" * 60)
    print()
    
    # Analyze structure
    structure_stats = analyze_directory_structure(root_dir)
    doc_stats = analyze_documentation(root_dir)
    
    # Print structure stats
    print("📊 Structure Statistics")
    print("-" * 60)
    print(f"Total files: {structure_stats['total_files']}")
    print(f"Total directories: {structure_stats['total_dirs']}")
    print()
    
    # Print file types
    print("📁 Files by Extension (Top 10)")
    print("-" * 60)
    sorted_exts = sorted(structure_stats['files_by_extension'].items(), 
                        key=lambda x: x[1], reverse=True)[:10]
    for ext, count in sorted_exts:
        print(f"  {ext:20} {count:4} files")
    print()
    
    # Print root files
    if structure_stats['files_in_root']:
        print(f"⚠️  Files in Root ({len(structure_stats['files_in_root'])}):")
        print("-" * 60)
        for file in sorted(structure_stats['files_in_root'])[:20]:
            print(f"  - {file}")
        if len(structure_stats['files_in_root']) > 20:
            print(f"  ... and {len(structure_stats['files_in_root']) - 20} more")
        print()
    
    # Print large directories
    if structure_stats['large_directories']:
        print("📦 Large Directories (>20 files):")
        print("-" * 60)
        for dir_path, count in sorted(structure_stats['large_directories'], 
                                     key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {dir_path:40} {count:4} files")
        print()
    
    # Print documentation stats
    print("📚 Documentation Analysis")
    print("-" * 60)
    print(f"Total documentation files: {doc_stats['total_docs']}")
    print(f"Documentation in root: {len(doc_stats['docs_in_root'])}")
    print()
    
    print("Documentation by Category:")
    for category, count in doc_stats['docs_by_category'].items():
        print(f"  {category:20} {count:4} files")
    print()
    
    # Recommendations
    if doc_stats['recommendations']:
        print("💡 Recommendations:")
        print("-" * 60)
        for rec in doc_stats['recommendations']:
            print(f"  - {rec}")
        print()
    
    return "Analysis complete"


def main():
    """Main function"""
    if len(sys.argv) > 1:
        root_dir = Path(sys.argv[1])
    else:
        root_dir = Path(__file__).parent.parent.parent
    
    if not root_dir.exists():
        print(f"Error: {root_dir} does not exist")
        sys.exit(1)
    
    generate_report(root_dir)


if __name__ == '__main__':
    main()



