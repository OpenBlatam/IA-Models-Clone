#!/usr/bin/env python3
"""
Requirements Diff Tool
Shows detailed differences between requirements files
"""

import sys
from pathlib import Path
from collections import defaultdict


def parse_requirements(filepath):
    """Parse requirements file"""
    packages = {}
    
    with open(filepath, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('-r'):
                continue
            
            # Extract package and version
            parts = line.split('>=')
            if len(parts) == 1:
                parts = line.split('==')
            if len(parts) == 1:
                parts = line.split('<')
            
            package = parts[0].split('[')[0].strip()
            version = parts[1].strip() if len(parts) > 1 else 'any'
            
            packages[package] = {
                'version': version,
                'line': line,
                'line_num': line_num
            }
    
    return packages


def compare_files(file1_path, file2_path):
    """Compare two requirements files"""
    packages1 = parse_requirements(file1_path)
    packages2 = parse_requirements(file2_path)
    
    all_packages = set(packages1.keys()) | set(packages2.keys())
    
    only_in_1 = set(packages1.keys()) - set(packages2.keys())
    only_in_2 = set(packages2.keys()) - set(packages1.keys())
    in_both = set(packages1.keys()) & set(packages2.keys())
    
    # Check version differences
    version_diff = {}
    for pkg in in_both:
        v1 = packages1[pkg]['version']
        v2 = packages2[pkg]['version']
        if v1 != v2:
            version_diff[pkg] = (v1, v2)
    
    return {
        'only_in_1': only_in_1,
        'only_in_2': only_in_2,
        'in_both': in_both,
        'version_diff': version_diff,
        'total_1': len(packages1),
        'total_2': len(packages2)
    }


def print_diff(file1_path, file2_path, diff_result):
    """Print diff results"""
    print("=" * 80)
    print(f"Comparing: {file1_path.name} vs {file2_path.name}")
    print("=" * 80)
    print()
    
    print(f"📊 Summary:")
    print(f"  {file1_path.name}: {diff_result['total_1']} packages")
    print(f"  {file2_path.name}: {diff_result['total_2']} packages")
    print(f"  Common: {len(diff_result['in_both'])} packages")
    print()
    
    if diff_result['only_in_1']:
        print(f"📦 Only in {file1_path.name} ({len(diff_result['only_in_1'])}):")
        for pkg in sorted(diff_result['only_in_1'])[:20]:
            print(f"  + {pkg}")
        if len(diff_result['only_in_1']) > 20:
            print(f"  ... and {len(diff_result['only_in_1']) - 20} more")
        print()
    
    if diff_result['only_in_2']:
        print(f"📦 Only in {file2_path.name} ({len(diff_result['only_in_2'])}):")
        for pkg in sorted(diff_result['only_in_2'])[:20]:
            print(f"  - {pkg}")
        if len(diff_result['only_in_2']) > 20:
            print(f"  ... and {len(diff_result['only_in_2']) - 20} more")
        print()
    
    if diff_result['version_diff']:
        print(f"🔄 Version Differences ({len(diff_result['version_diff'])}):")
        for pkg, (v1, v2) in sorted(diff_result['version_diff'].items())[:10]:
            print(f"  {pkg}:")
            print(f"    {file1_path.name}: {v1}")
            print(f"    {file2_path.name}: {v2}")
        if len(diff_result['version_diff']) > 10:
            print(f"  ... and {len(diff_result['version_diff']) - 10} more")
        print()


def main():
    """Main function"""
    if len(sys.argv) < 3:
        print("Usage: python requirements-diff.py <file1> <file2>")
        sys.exit(1)
    
    file1_path = Path(sys.argv[1])
    file2_path = Path(sys.argv[2])
    
    if not file1_path.exists():
        print(f"Error: {file1_path} not found")
        sys.exit(1)
    
    if not file2_path.exists():
        print(f"Error: {file2_path} not found")
        sys.exit(1)
    
    diff_result = compare_files(file1_path, file2_path)
    print_diff(file1_path, file2_path, diff_result)


if __name__ == '__main__':
    main()



