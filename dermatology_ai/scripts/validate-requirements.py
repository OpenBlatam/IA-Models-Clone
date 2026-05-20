#!/usr/bin/env python3
"""
Requirements Validator
Validates requirements files for common issues
"""

import re
import sys
from pathlib import Path
from typing import List, Tuple, Dict


class RequirementsValidator:
    """Validates requirements files"""
    
    def __init__(self, filepath: Path):
        self.filepath = filepath
        self.errors = []
        self.warnings = []
        self.packages = []
    
    def validate(self) -> Tuple[List[str], List[str]]:
        """Run all validations"""
        self._parse_file()
        self._check_duplicates()
        self._check_version_specifiers()
        self._check_package_names()
        self._check_comments()
        self._check_includes()
        
        return self.errors, self.warnings
    
    def _parse_file(self):
        """Parse requirements file"""
        with open(self.filepath, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.rstrip()
                
                # Skip empty lines and comments
                if not line or line.startswith('#'):
                    continue
                
                # Check for includes
                if line.startswith('-r'):
                    self.packages.append(('include', line, line_num))
                    continue
                
                # Extract package
                match = re.match(r'^([a-zA-Z0-9_-]+(?:\[[^\]]+\])?)\s*(.*)$', line)
                if match:
                    package = match.group(1).split('[')[0]
                    version = match.group(2).strip()
                    self.packages.append((package, version, line_num))
    
    def _check_duplicates(self):
        """Check for duplicate packages"""
        seen = {}
        for package, version, line_num in self.packages:
            if package == 'include':
                continue
            if package in seen:
                self.warnings.append(
                    f"Line {line_num}: Duplicate package '{package}' "
                    f"(first seen at line {seen[package]})"
                )
            else:
                seen[package] = line_num
    
    def _check_version_specifiers(self):
        """Check version specifiers"""
        valid_specs = ['>=', '<=', '==', '!=', '~=', '>', '<']
        
        for package, version, line_num in self.packages:
            if package == 'include' or not version:
                continue
            
            # Check for invalid specifiers
            if not any(version.startswith(spec) for spec in valid_specs):
                if not version.startswith('#'):
                    self.errors.append(
                        f"Line {line_num}: Invalid version specifier for '{package}': {version}"
                    )
    
    def _check_package_names(self):
        """Check package name format"""
        for package, version, line_num in self.packages:
            if package == 'include':
                continue
            
            # Check for invalid characters
            if not re.match(r'^[a-zA-Z0-9_-]+$', package):
                self.errors.append(
                    f"Line {line_num}: Invalid package name '{package}'"
                )
            
            # Check for common typos
            common_typos = {
                'requets': 'requests',
                'pytho': 'python',
                'nump': 'numpy',
            }
            if package.lower() in common_typos:
                self.warnings.append(
                    f"Line {line_num}: Possible typo '{package}' "
                    f"(did you mean '{common_typos[package.lower()]}'?)"
                )
    
    def _check_comments(self):
        """Check comment format"""
        with open(self.filepath, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                # Check for inline comments without space
                if '#' in line and not re.match(r'.*#\s', line):
                    if line.strip() and not line.strip().startswith('#'):
                        self.warnings.append(
                            f"Line {line_num}: Consider adding space before comment"
                        )
    
    def _check_includes(self):
        """Check include statements"""
        for item, version, line_num in self.packages:
            if item == 'include':
                included_file = version.split()[1] if len(version.split()) > 1 else ''
                if included_file and not Path(included_file).exists():
                    self.errors.append(
                        f"Line {line_num}: Included file not found: {included_file}"
                    )


def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("Usage: python validate-requirements.py <requirements-file>")
        sys.exit(1)
    
    filepath = Path(sys.argv[1])
    if not filepath.exists():
        print(f"Error: {filepath} not found")
        sys.exit(1)
    
    print(f"Validating {filepath}...")
    validator = RequirementsValidator(filepath)
    errors, warnings = validator.validate()
    
    # Report results
    if errors:
        print(f"\n❌ Found {len(errors)} errors:")
        for error in errors:
            print(f"  {error}")
    
    if warnings:
        print(f"\n⚠️  Found {len(warnings)} warnings:")
        for warning in warnings:
            print(f"  {warning}")
    
    if not errors and not warnings:
        print(f"\n✅ {filepath} is valid!")
        sys.exit(0)
    else:
        sys.exit(1 if errors else 0)


if __name__ == '__main__':
    main()



