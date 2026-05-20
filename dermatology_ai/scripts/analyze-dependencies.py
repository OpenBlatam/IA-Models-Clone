#!/usr/bin/env python3
"""
Dependency Analyzer
Analyzes requirements files and provides insights
"""

import re
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple
import json


def parse_requirements_file(filepath: Path) -> List[Tuple[str, str]]:
    """Parse requirements file and return list of (package, version) tuples"""
    requirements = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            
            # Skip comments and empty lines
            if not line or line.startswith('#'):
                continue
            
            # Skip -r includes
            if line.startswith('-r'):
                continue
            
            # Extract package name and version
            # Handle formats like: package>=1.0.0, package[extra]>=1.0.0
            match = re.match(r'^([a-zA-Z0-9_-]+(?:\[[^\]]+\])?)\s*([<>=!]+.*)?', line)
            if match:
                package = match.group(1).split('[')[0]  # Remove extras
                version = match.group(2) if match.group(2) else 'any'
                requirements.append((package, version))
    
    return requirements


def analyze_requirements() -> Dict:
    """Analyze all requirements files"""
    results = {
        'files': {},
        'statistics': {},
        'duplicates': [],
        'conflicts': [],
        'recommendations': []
    }
    
    requirements_files = list(Path('.').glob('requirements*.txt'))
    all_packages = defaultdict(list)
    
    # Parse each file
    for req_file in requirements_files:
        if req_file.name == 'requirements-lock.txt':
            continue
            
        packages = parse_requirements_file(req_file)
        results['files'][req_file.name] = {
            'packages': len(packages),
            'list': packages
        }
        
        # Track packages across files
        for package, version in packages:
            all_packages[package].append((req_file.name, version))
    
    # Find duplicates
    for package, occurrences in all_packages.items():
        if len(occurrences) > 1:
            results['duplicates'].append({
                'package': package,
                'files': [f[0] for f in occurrences],
                'versions': [f[1] for f in occurrences]
            })
    
    # Statistics
    total_packages = len(all_packages)
    total_files = len(requirements_files)
    
    results['statistics'] = {
        'total_packages': total_packages,
        'total_files': total_files,
        'packages_per_file': {name: data['packages'] 
                             for name, data in results['files'].items()}
    }
    
    # Recommendations
    if results['duplicates']:
        results['recommendations'].append(
            f"Found {len(results['duplicates'])} packages in multiple files. "
            "Consider consolidating."
        )
    
    # Check for common issues
    for req_file, data in results['files'].items():
        packages = [p[0] for p in data['list']]
        
        # Check for both torch and tensorflow
        if 'torch' in packages and 'tensorflow' in packages:
            results['conflicts'].append(
                f"{req_file}: Contains both PyTorch and TensorFlow"
            )
        
        # Check for multiple JSON libraries
        json_libs = [p for p in packages if 'json' in p.lower()]
        if len(json_libs) > 2:
            results['recommendations'].append(
                f"{req_file}: Multiple JSON libraries found. "
                "Consider using only orjson for best performance."
            )
    
    return results


def print_report(results: Dict):
    """Print analysis report"""
    print("=" * 60)
    print("Dependency Analysis Report")
    print("=" * 60)
    print()
    
    # Statistics
    print("📊 Statistics:")
    print(f"  Total unique packages: {results['statistics']['total_packages']}")
    print(f"  Requirements files: {results['statistics']['total_files']}")
    print()
    
    # Files breakdown
    print("📁 Files Breakdown:")
    for filename, data in results['files'].items():
        print(f"  {filename}: {data['packages']} packages")
    print()
    
    # Duplicates
    if results['duplicates']:
        print("⚠️  Duplicates Found:")
        for dup in results['duplicates'][:10]:  # Show first 10
            print(f"  {dup['package']} in: {', '.join(dup['files'])}")
        if len(results['duplicates']) > 10:
            print(f"  ... and {len(results['duplicates']) - 10} more")
        print()
    
    # Conflicts
    if results['conflicts']:
        print("❌ Conflicts:")
        for conflict in results['conflicts']:
            print(f"  {conflict}")
        print()
    
    # Recommendations
    if results['recommendations']:
        print("💡 Recommendations:")
        for rec in results['recommendations']:
            print(f"  • {rec}")
        print()


def main():
    """Main function"""
    if not Path('requirements.txt').exists():
        print("Error: requirements.txt not found")
        sys.exit(1)
    
    results = analyze_requirements()
    print_report(results)
    
    # Save JSON report
    json_file = Path('dependency-analysis.json')
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"📄 Full report saved to: {json_file}")


if __name__ == '__main__':
    main()



