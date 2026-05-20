#!/usr/bin/env python3
"""
Legacy cleanup script for Export IA.
Removes old files and organizes the structure.
"""

import os
import shutil
import sys
from pathlib import Path


def cleanup_legacy_files():
    """Remove legacy files and organize structure."""
    
    # Files to remove (legacy versions)
    legacy_files = [
        "export_ia_engine.py",  # Original monolithic file
        "requirements.txt",     # Old requirements
        "requirements_advanced.txt",  # Old advanced requirements
        "requirements_refactored.txt",  # Old refactored requirements
        "README.md",  # Old README
    ]
    
    # Directories to remove (unused/legacy)
    legacy_dirs = [
        "core/",  # Old core directory
        "interfaces/",  # Old interfaces
        "enhanced/",  # Unused enhanced features
        "api_gateway/",  # Unused API gateway
        "microservices/",  # Unused microservices
        "analytics/",  # Unused analytics
        "content_optimization/",  # Unused content optimization
        "workflows/",  # Unused workflows
        "blockchain/",  # Unused blockchain
        "cosmic_transcendence/",  # Unused cosmic features
        "api/",  # Old API directory
        "training/",  # Unused training
        "gradio_interface/",  # Unused Gradio interface
        "ai_enhanced/",  # Unused AI enhanced
        "styling/",  # Unused styling
        "quality/",  # Unused quality
    ]
    
    print("🧹 Cleaning up legacy files and directories...")
    
    # Remove legacy files
    for file_name in legacy_files:
        file_path = Path(file_name)
        if file_path.exists():
            print(f"  ❌ Removing legacy file: {file_name}")
            file_path.unlink()
        else:
            print(f"  ⏭️  Legacy file not found: {file_name}")
    
    # Remove legacy directories
    for dir_name in legacy_dirs:
        dir_path = Path(dir_name)
        if dir_path.exists() and dir_path.is_dir():
            print(f"  ❌ Removing legacy directory: {dir_name}")
            shutil.rmtree(dir_path)
        else:
            print(f"  ⏭️  Legacy directory not found: {dir_name}")
    
    print("✅ Legacy cleanup completed!")


def organize_structure():
    """Organize the new structure."""
    
    print("\n📁 Organizing new structure...")
    
    # Ensure required directories exist
    required_dirs = [
        "src/core",
        "src/exporters", 
        "src/api",
        "src/cli",
        "src/plugins",
        "config",
        "examples",
        "tests",
        "docs"
    ]
    
    for dir_path in required_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"  ✅ Ensured directory exists: {dir_path}")
    
    # Create .gitignore if it doesn't exist
    gitignore_content = """# Export IA - Git Ignore

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
venv/
env/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo

# Export outputs
exports/
temp/
*.pdf
*.docx
*.html
*.md
*.rtf
*.txt
*.json
*.xml

# Logs
*.log
logs/

# Cache
.cache/
*.cache

# OS
.DS_Store
Thumbs.db

# Configuration (keep template)
config/export_config.yaml
!config/export_config.yaml.template
"""
    
    gitignore_path = Path(".gitignore")
    if not gitignore_path.exists():
        gitignore_path.write_text(gitignore_content)
        print("  ✅ Created .gitignore file")
    else:
        print("  ⏭️  .gitignore already exists")
    
    print("✅ Structure organization completed!")


def create_startup_scripts():
    """Create convenient startup scripts."""
    
    print("\n🚀 Creating startup scripts...")
    
    # Create Windows batch file
    windows_script = """@echo off
echo Starting Export IA API Server...
python run_api.py
pause
"""
    
    Path("start_api.bat").write_text(windows_script)
    print("  ✅ Created start_api.bat")
    
    # Create Unix shell script
    unix_script = """#!/bin/bash
echo "Starting Export IA API Server..."
python run_api.py
"""
    
    unix_script_path = Path("start_api.sh")
    unix_script_path.write_text(unix_script)
    unix_script_path.chmod(0o755)
    print("  ✅ Created start_api.sh")
    
    # Create CLI entry point
    cli_script = """#!/usr/bin/env python3
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from cli.main import cli

if __name__ == "__main__":
    cli()
"""
    
    cli_path = Path("export-ia")
    cli_path.write_text(cli_script)
    cli_path.chmod(0o755)
    print("  ✅ Created export-ia CLI entry point")
    
    print("✅ Startup scripts created!")


def main():
    """Main cleanup function."""
    print("🎯 Export IA - Legacy Cleanup and Organization")
    print("=" * 50)
    
    # Change to the script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    try:
        cleanup_legacy_files()
        organize_structure()
        create_startup_scripts()
        
        print("\n🎉 Cleanup and organization completed successfully!")
        print("\n📋 Next steps:")
        print("  1. Install dependencies: pip install -r requirements_refactored_v2.txt")
        print("  2. Start API server: python run_api.py")
        print("  3. Use CLI: python export-ia --help")
        print("  4. Run tests: pytest tests/")
        
    except Exception as e:
        print(f"\n❌ Error during cleanup: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()




