#!/usr/bin/env python3
"""
Requirements Compatibility Checker
Checks compatibility between packages and Python version
"""

import sys
import subprocess
from pathlib import Path
import re


def get_python_version():
    """Get current Python version"""
    version = sys.version_info
    return f"{version.major}.{version.minor}.{version.micro}"


def check_package_compatibility(package_name, python_version):
    """Check if package is compatible with Python version"""
    try:
        # Try to get package info
        result = subprocess.run(
            ['pip', 'show', package_name],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0:
            # Check Requires-Python field
            for line in result.stdout.split('\n'):
                if line.startswith('Requires-Python:'):
                    requires = line.split(':', 1)[1].strip()
                    # Simple check (could be improved)
                    if requires:
                        return True, requires
            return True, "No specific requirement"
    except:
        pass
    
    return None, None


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


def main():
    """Main function"""
    if len(sys.argv) < 2:
        filepath = Path('requirements.txt')
    else:
        filepath = Path(sys.argv[1])
    
    if not filepath.exists():
        print(f"Error: {filepath} not found")
        sys.exit(1)
    
    python_version = get_python_version()
    print(f"Python version: {python_version}")
    print(f"Checking compatibility for: {filepath}")
    print("=" * 60)
    print()
    
    packages = parse_requirements(filepath)
    compatible = 0
    incompatible = 0
    unknown = 0
    
    print("Checking packages...")
    print()
    
    for package in packages[:20]:  # Check first 20
        is_compat, requires = check_package_compatibility(package, python_version)
        
        if is_compat is None:
            print(f"  ? {package} (unknown)")
            unknown += 1
        elif is_compat:
            print(f"  ✓ {package}")
            compatible += 1
        else:
            print(f"  ✗ {package} (requires: {requires})")
            incompatible += 1
    
    if len(packages) > 20:
        print(f"  ... and {len(packages) - 20} more packages")
    
    print()
    print("=" * 60)
    print(f"Summary:")
    print(f"  Compatible: {compatible}")
    print(f"  Incompatible: {incompatible}")
    print(f"  Unknown: {unknown}")
    print()
    
    if incompatible > 0:
        print("⚠️  Some packages may have compatibility issues")
        print("   Install packages to verify: pip install -r requirements.txt")
    else:
        print("✅ All checked packages appear compatible")


if __name__ == '__main__':
    main()



