#!/usr/bin/env python3
"""
Requirements Migrator
Migrates between different requirements file formats
"""

import re
import sys
from pathlib import Path
from typing import List, Tuple


def parse_requirements_txt(filepath: Path) -> List[Tuple[str, str]]:
    """Parse requirements.txt format"""
    requirements = []
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('-r'):
                continue
            
            # Extract package and version
            parts = re.split(r'[>=<!=]+', line, 1)
            package = parts[0].split('[')[0].strip()
            version = parts[1].strip() if len(parts) > 1 else ''
            
            requirements.append((package, version, line))
    
    return requirements


def to_poetry_format(requirements: List[Tuple[str, str, str]]) -> str:
    """Convert to Poetry pyproject.toml format"""
    output = "[tool.poetry.dependencies]\n"
    output += "python = \"^3.10\"\n\n"
    
    for package, version, original in requirements:
        if version:
            # Convert version specifiers
            if version.startswith('>='):
                version_str = f'^{version[2:]}'
            elif version.startswith('=='):
                version_str = version[2:]
            else:
                version_str = version
            output += f'{package} = "{version_str}"\n'
        else:
            output += f'{package} = "*"\n'
    
    return output


def to_pipenv_format(requirements: List[Tuple[str, str, str]]) -> str:
    """Convert to Pipfile format"""
    output = "[[source]]\n"
    output += "url = \"https://pypi.org/simple\"\n"
    output += "verify_ssl = true\n"
    output += "name = \"pypi\"\n\n"
    output += "[packages]\n"
    
    for package, version, original in requirements:
        if version:
            output += f'{package} = "{version}"\n'
        else:
            output += f'{package} = "*"\n'
    
    return output


def to_conda_format(requirements: List[Tuple[str, str, str]]) -> str:
    """Convert to environment.yml format"""
    output = "name: dermatology-ai\n"
    output += "channels:\n"
    output += "  - defaults\n"
    output += "  - conda-forge\n"
    output += "dependencies:\n"
    output += "  - python>=3.10\n"
    output += "  - pip\n"
    output += "  - pip:\n"
    
    for package, version, original in requirements:
        if version:
            output += f"    - {package}{version}\n"
        else:
            output += f"    - {package}\n"
    
    return output


def main():
    """Main function"""
    if len(sys.argv) < 3:
        print("Usage: python migrate-requirements.py <input-file> <output-format>")
        print("Formats: poetry, pipenv, conda")
        sys.exit(1)
    
    input_file = Path(sys.argv[1])
    output_format = sys.argv[2].lower()
    
    if not input_file.exists():
        print(f"Error: {input_file} not found")
        sys.exit(1)
    
    # Parse requirements
    print(f"Parsing {input_file}...")
    requirements = parse_requirements_txt(input_file)
    print(f"Found {len(requirements)} packages")
    
    # Convert
    print(f"Converting to {output_format} format...")
    if output_format == 'poetry':
        output = to_poetry_format(requirements)
        output_file = 'pyproject-poetry.toml'
    elif output_format == 'pipenv':
        output = to_pipenv_format(requirements)
        output_file = 'Pipfile'
    elif output_format == 'conda':
        output = to_conda_format(requirements)
        output_file = 'environment.yml'
    else:
        print(f"Error: Unknown format '{output_format}'")
        print("Supported formats: poetry, pipenv, conda")
        sys.exit(1)
    
    # Write output
    with open(output_file, 'w') as f:
        f.write(output)
    
    print(f"✓ Converted to {output_file}")
    print(f"⚠️  Please review and test before using!")


if __name__ == '__main__':
    main()



