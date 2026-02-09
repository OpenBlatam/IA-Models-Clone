#!/usr/bin/env python3
"""
Dependency Conflict Detector
Detects potential conflicts between packages
"""

import sys
import subprocess
from pathlib import Path
from collections import defaultdict


def get_package_info(package_name):
    """Get package information"""
    try:
        result = subprocess.run(
            ['pip', 'show', package_name],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            info = {}
            for line in result.stdout.split('\n'):
                if ':' in line:
                    key, value = line.split(':', 1)
                    info[key.strip()] = value.strip()
            return info
    except:
        pass
    return None


def parse_requirements(filepath):
    """Parse requirements file"""
    packages = []
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('-r'):
                continue
            
            package = line.split('>=')[0].split('==')[0].split('[')[0].strip()
            if package:
                packages.append(package)
    
    return packages


def check_conflicts(packages):
    """Check for known conflicts"""
    known_conflicts = {
        ('torch', 'tensorflow'): 'PyTorch and TensorFlow conflict',
        ('django', 'flask'): 'Django and Flask are different frameworks',
        ('requests', 'urllib3'): 'requests uses urllib3 internally',
    }
    
    conflicts = []
    package_set = set(packages)
    
    for (pkg1, pkg2), reason in known_conflicts.items():
        if pkg1 in package_set and pkg2 in package_set:
            conflicts.append({
                'packages': (pkg1, pkg2),
                'reason': reason
            })
    
    return conflicts


def main():
    """Main function"""
    if len(sys.argv) < 2:
        filepath = Path('requirements.txt')
    else:
        filepath = Path(sys.argv[1])
    
    if not filepath.exists():
        print(f"Error: {filepath} not found")
        sys.exit(1)
    
    print(f"Checking for conflicts in {filepath}...")
    print("=" * 60)
    print()
    
    packages = parse_requirements(filepath)
    print(f"Found {len(packages)} packages")
    print()
    
    # Check known conflicts
    conflicts = check_conflicts(packages)
    
    if conflicts:
        print("⚠️  Potential Conflicts Found:")
        print()
        for conflict in conflicts:
            pkg1, pkg2 = conflict['packages']
            print(f"  {pkg1} <-> {pkg2}")
            print(f"    Reason: {conflict['reason']}")
            print()
    else:
        print("✅ No known conflicts detected")
        print()
    
    # Check for packages that might conflict
    print("Checking for potential issues...")
    print()
    
    # Check for multiple JSON libraries
    json_libs = [p for p in packages if 'json' in p.lower() and p not in ['orjson', 'ujson']]
    if len(json_libs) > 2:
        print(f"⚠️  Multiple JSON libraries found: {', '.join(json_libs)}")
        print("   Consider using only orjson for best performance")
        print()
    
    # Check for multiple HTTP clients
    http_libs = [p for p in packages if any(x in p.lower() for x in ['httpx', 'requests', 'aiohttp'])]
    if len(http_libs) > 2:
        print(f"⚠️  Multiple HTTP clients found: {', '.join(http_libs)}")
        print("   Consider standardizing on one")
        print()


if __name__ == '__main__':
    main()



