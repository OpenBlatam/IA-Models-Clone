#!/usr/bin/env python3
"""
Update Imports
Automatically updates imports in Python files to use new organized structure
"""

import os
import re
import sys
from pathlib import Path
from typing import List, Tuple, Dict


def load_import_mapping(mapping_file: Path) -> Dict[str, str]:
    """Load import mapping from file"""
    mapping = {}
    
    if not mapping_file.exists():
        print(f"Warning: {mapping_file} not found. Creating default mapping...")
        return create_default_mapping()
    
    with open(mapping_file, 'r', encoding='utf-8') as f:
        content = f.read()
        
        # Parse markdown table
        in_services = False
        in_utils = False
        
        for line in content.split('\n'):
            if '## Services' in line:
                in_services = True
                in_utils = False
                continue
            elif '## Utils' in line:
                in_services = False
                in_utils = True
                continue
            elif line.startswith('|') and 'Old Import' not in line and '---' not in line:
                parts = [p.strip() for p in line.split('|') if p.strip()]
                if len(parts) >= 2:
                    old = parts[0].strip('`')
                    new = parts[1].strip('`')
                    if old and new:
                        mapping[old] = new
    
    return mapping


def create_default_mapping() -> Dict[str, str]:
    """Create default import mapping"""
    mapping = {}
    
    # Services mapping
    service_categories = {
        'analysis': ['advanced_ml_analysis', 'age_analysis', 'ai_photo_analysis'],
        'recommendations': ['age_based_recommendations', 'budget_based_recommendations'],
        'tracking': ['allergy_tracker', 'budget_tracker'],
        'products': ['ingredient_analyzer', 'product_comparison'],
        'ml': ['advanced_texture_ml', 'condition_predictor'],
        'notifications': ['alert_system', 'enhanced_notifications'],
        'integrations': ['integration_service', 'iot_integration'],
        'reporting': ['advanced_reporting', 'report_generator'],
        'social': ['challenge_system', 'collaboration_service']
    }
    
    for category, services in service_categories.items():
        for service in services:
            old_pattern = f"from services.{service} import"
            new_pattern = f"from services.{category}.{service} import"
            mapping[old_pattern] = new_pattern
    
    # Utils mapping
    util_categories = {
        'logging': ['logger', 'advanced_logging'],
        'caching': ['cache', 'advanced_caching'],
        'validation': ['advanced_validator'],
        'security': ['oauth2', 'security_headers'],
        'performance': ['optimization', 'performance_profiler']
    }
    
    for category, utils in util_categories.items():
        for util in utils:
            old_pattern = f"from utils.{util} import"
            new_pattern = f"from utils.{category}.{util} import"
            mapping[old_pattern] = new_pattern
    
    return mapping


def update_file_imports(file_path: Path, mapping: Dict[str, str], dry_run: bool = False) -> Tuple[int, List[str]]:
    """Update imports in a file"""
    updated_count = 0
    changes = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Update imports
        for old_import, new_import in mapping.items():
            if old_import in content:
                content = content.replace(old_import, new_import)
                updated_count += content.count(new_import) - original_content.count(new_import)
                changes.append(f"  {old_import} → {new_import}")
        
        # Write back if changed
        if content != original_content and not dry_run:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
        
        return updated_count, changes
    
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return 0, []


def find_python_files(root_dir: Path) -> List[Path]:
    """Find all Python files"""
    python_files = []
    
    for root, dirs, files in os.walk(root_dir):
        # Skip certain directories
        dirs[:] = [d for d in dirs if d not in ['__pycache__', '.git', 'node_modules', 'venv', '.venv']]
        
        for file in files:
            if file.endswith('.py'):
                python_files.append(Path(root) / file)
    
    return python_files


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Update imports to new organized structure')
    parser.add_argument('--dry-run', action='store_true', help='Show changes without applying')
    parser.add_argument('--mapping', type=str, help='Path to import mapping file')
    parser.add_argument('--directory', type=str, help='Directory to process')
    args = parser.parse_args()
    
    if args.directory:
        root_dir = Path(args.directory)
    else:
        root_dir = Path(__file__).parent.parent.parent
    
    if not root_dir.exists():
        print(f"Error: {root_dir} does not exist")
        sys.exit(1)
    
    # Load mapping
    if args.mapping:
        mapping_file = Path(args.mapping)
    else:
        mapping_file = root_dir / 'docs' / 'IMPORT_MAPPING.md'
    
    mapping = load_import_mapping(mapping_file)
    
    if not mapping:
        print("Error: No import mapping found")
        sys.exit(1)
    
    print(f"Loaded {len(mapping)} import mappings")
    print(f"Processing Python files in {root_dir}")
    print("=" * 60)
    print()
    
    # Find Python files
    python_files = find_python_files(root_dir)
    print(f"Found {len(python_files)} Python files")
    print()
    
    # Process files
    total_updated = 0
    files_changed = 0
    
    for py_file in python_files:
        updated, changes = update_file_imports(py_file, mapping, dry_run=args.dry_run)
        
        if updated > 0:
            files_changed += 1
            total_updated += updated
            print(f"Updated {py_file.relative_to(root_dir)}:")
            for change in changes[:3]:  # Show first 3 changes
                print(change)
            if len(changes) > 3:
                print(f"  ... and {len(changes) - 3} more")
            print()
    
    print("=" * 60)
    print(f"Summary:")
    print(f"  Files processed: {len(python_files)}")
    print(f"  Files changed: {files_changed}")
    print(f"  Total imports updated: {total_updated}")
    
    if args.dry_run:
        print("\n(DRY RUN - No files were modified)")


if __name__ == '__main__':
    main()



