#!/usr/bin/env python3
"""
HeyGen AI Project Organizer

This script organizes the HeyGen AI project structure by:
1. Moving files to appropriate directories
2. Cleaning up duplicate files
3. Creating a clean, organized structure
4. Updating imports and references
"""

import os
import shutil
import sys
from pathlib import Path
from typing import List, Dict, Set
import re

class ProjectOrganizer:
    """Organizes the HeyGen AI project structure."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.organized_structure = {
            "src": {
                "core": ["core/"],
                "plugins": ["plugins/"],
                "api": ["api/"],
                "models": ["models/"],
                "utils": ["utils/"],
                "config": ["config/"]
            },
            "docs": {
                "guides": ["docs/"],
                "examples": ["examples/"],
                "api_docs": ["api_docs/"]
            },
            "scripts": {
                "setup": ["setup.py", "install_requirements.py"],
                "management": ["manage.py", "organize_project.py"]
            },
            "tests": {
                "unit": ["tests/unit/"],
                "integration": ["tests/integration/"],
                "fixtures": ["tests/fixtures/"]
            },
            "configs": {
                "main": ["config/"],
                "environments": ["config/environments/"]
            },
            "requirements": {
                "profiles": ["requirements/"],
                "main": ["requirements.txt"]
            }
        }
    
    def organize_project(self):
        """Main organization method."""
        print("🚀 Organizing HeyGen AI Project Structure")
        print("=" * 60)
        
        # Create organized directory structure
        self._create_directory_structure()
        
        # Move files to appropriate locations
        self._move_files()
        
        # Clean up duplicate and unnecessary files
        self._cleanup_files()
        
        # Update imports and references
        self._update_references()
        
        # Create new README and documentation
        self._create_documentation()
        
        print("✅ Project organization completed!")
    
    def _create_directory_structure(self):
        """Create the organized directory structure."""
        print("📁 Creating directory structure...")
        
        for category, subcategories in self.organized_structure.items():
            for subcategory, paths in subcategories.items():
                if isinstance(paths, list):
                    for path in paths:
                        full_path = self.project_root / category / subcategory / Path(path).name
                        full_path.mkdir(parents=True, exist_ok=True)
                        print(f"  Created: {full_path}")
    
    def _move_files(self):
        """Move files to their appropriate locations."""
        print("📦 Moving files to organized structure...")
        
        # Move core files
        if (self.project_root / "core").exists():
            shutil.move(str(self.project_root / "core"), 
                       str(self.project_root / "src" / "core"))
            print("  Moved: core/ → src/core/")
        
        # Move plugins
        if (self.project_root / "plugins").exists():
            shutil.move(str(self.project_root / "plugins"), 
                       str(self.project_root / "src" / "plugins"))
            print("  Moved: plugins/ → src/plugins/")
        
        # Move API files
        if (self.project_root / "api").exists():
            shutil.move(str(self.project_root / "api"), 
                       str(self.project_root / "src" / "api"))
            print("  Moved: api/ → src/api/")
        
        # Move configuration files
        if (self.project_root / "config").exists():
            shutil.move(str(self.project_root / "config"), 
                       str(self.project_root / "configs" / "main"))
            print("  Moved: config/ → configs/main/")
        
        # Move requirements
        if (self.project_root / "requirements").exists():
            shutil.move(str(self.project_root / "requirements"), 
                       str(self.project_root / "requirements" / "profiles"))
            print("  Moved: requirements/ → requirements/profiles/")
    
    def _cleanup_files(self):
        """Clean up duplicate and unnecessary files."""
        print("🧹 Cleaning up files...")
        
        # Remove duplicate requirements files
        duplicate_patterns = [
            "requirements_*.txt",
            "requirements-*.txt"
        ]
        
        for pattern in duplicate_patterns:
            for file_path in self.project_root.glob(pattern):
                if file_path.name != "requirements.txt":
                    file_path.unlink()
                    print(f"  Removed duplicate: {file_path.name}")
        
        # Remove old demo files (keep only essential ones)
        demo_files_to_keep = {
            "launch_demos.py",
            "plugin_demo.py",
            "comprehensive_demo_runner.py"
        }
        
        for demo_file in self.project_root.glob("run_*.py"):
            if demo_file.name not in demo_files_to_keep:
                demo_file.unlink()
                print(f"  Removed old demo: {demo_file.name}")
        
        # Remove old README files
        readme_files_to_keep = {
            "README.md",
            "README_ENTERPRISE.md"
        }
        
        for readme_file in self.project_root.glob("README_*.md"):
            if readme_file.name not in readme_files_to_keep:
                readme_file.unlink()
                print(f"  Removed old README: {readme_file.name}")
    
    def _update_references(self):
        """Update import statements and file references."""
        print("🔧 Updating file references...")
        
        # Update Python files with new import paths
        for py_file in self.project_root.rglob("*.py"):
            if py_file.is_file():
                self._update_file_imports(py_file)
    
    def _update_file_imports(self, file_path: Path):
        """Update import statements in a Python file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Update relative imports
            old_imports = [
                "from core.",
                "from plugins.",
                "from api.",
                "from config."
            ]
            
            new_imports = [
                "from src.core.",
                "from src.plugins.",
                "from src.api.",
                "from configs.main."
            ]
            
            for old_import, new_import in zip(old_imports, new_imports):
                content = content.replace(old_import, new_import)
            
            # Write updated content
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
                
        except Exception as e:
            print(f"  Warning: Could not update {file_path}: {e}")
    
    def _create_documentation(self):
        """Create new project documentation."""
        print("📚 Creating project documentation...")
        
        # Create main README
        main_readme = self.project_root / "README.md"
        if not main_readme.exists():
            self._create_main_readme(main_readme)
        
        # Create project structure documentation
        structure_doc = self.project_root / "docs" / "PROJECT_STRUCTURE.md"
        structure_doc.parent.mkdir(parents=True, exist_ok=True)
        self._create_structure_documentation(structure_doc)
        
        # Create setup guide
        setup_guide = self.project_root / "docs" / "SETUP.md"
        self._create_setup_guide(setup_guide)
    
    def _create_main_readme(self, readme_path: Path):
        """Create the main README file."""
        content = """# 🚀 HeyGen AI

A comprehensive AI platform with advanced machine learning capabilities, plugin architecture, and enterprise features.

## ✨ Features

- **🤖 Advanced AI Models**: Transformer, Diffusion, and custom models
- **🔌 Plugin System**: Extensible architecture for custom functionality
- **⚡ Ultra Performance**: Optimized for maximum speed and efficiency
- **🏢 Enterprise Ready**: Production-grade features and monitoring
- **🌐 Web Interface**: FastAPI backend with Gradio frontend

## 🚀 Quick Start

```bash
# Install requirements
python install_requirements.py basic

# Run demo launcher
python launch_demos.py

# Start the system
python src/main.py
```

## 📁 Project Structure

```
heygen_ai/
├── src/                    # Source code
│   ├── core/              # Core system components
│   ├── plugins/           # Plugin system
│   ├── api/               # API endpoints
│   └── models/            # AI models
├── configs/               # Configuration files
├── requirements/          # Dependency profiles
├── docs/                  # Documentation
├── tests/                 # Test suite
└── scripts/               # Management scripts
```

## 🔧 Installation

See [SETUP.md](docs/SETUP.md) for detailed installation instructions.

## 📚 Documentation

- [Project Structure](docs/PROJECT_STRUCTURE.md)
- [API Reference](docs/API.md)
- [Plugin Development](docs/PLUGINS.md)
- [Performance Guide](docs/PERFORMANCE.md)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
"""
        
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"  Created: {readme_path}")
    
    def _create_structure_documentation(self, doc_path: Path):
        """Create project structure documentation."""
        content = """# 📁 HeyGen AI Project Structure

This document describes the organized structure of the HeyGen AI project.

## 🏗️ Directory Organization

### Source Code (`src/`)

#### `core/`
Core system components and utilities:
- Plugin system
- Configuration management
- Performance optimization
- Model management

#### `plugins/`
Plugin implementations:
- Model plugins
- Optimization plugins
- Feature plugins
- Custom plugins

#### `api/`
API endpoints and web interface:
- FastAPI routes
- WebSocket handlers
- Middleware
- Authentication

#### `models/`
AI model implementations:
- Transformer models
- Diffusion models
- Custom architectures
- Model utilities

### Configuration (`configs/`)

#### `main/`
Main configuration files:
- System configuration
- Model settings
- Performance options
- Security settings

#### `environments/`
Environment-specific configurations:
- Development
- Staging
- Production

### Dependencies (`requirements/`)

#### `profiles/`
Modular requirement files:
- `base.txt` - Core dependencies
- `ml.txt` - Machine learning
- `web.txt` - Web framework
- `enterprise.txt` - Enterprise features
- `dev.txt` - Development tools

### Documentation (`docs/`)

- Setup guides
- API documentation
- Plugin development
- Performance optimization
- Troubleshooting

### Testing (`tests/`)

- Unit tests
- Integration tests
- Performance tests
- Test fixtures

### Scripts (`scripts/`)

- Project management
- Setup and installation
- Development tools
- Deployment scripts

## 🔄 Migration Notes

When migrating from the old structure:
1. Update import statements to use new paths
2. Update configuration file references
3. Test all functionality after migration
4. Update documentation references

## 📝 File Naming Conventions

- Use snake_case for Python files and directories
- Use PascalCase for class names
- Use UPPER_CASE for constants
- Use descriptive names that indicate purpose
"""
        
        with open(doc_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"  Created: {doc_path}")
    
    def _create_setup_guide(self, guide_path: Path):
        """Create setup guide documentation."""
        content = """# 🔧 HeyGen AI Setup Guide

Complete setup instructions for the HeyGen AI platform.

## 📋 Prerequisites

- Python 3.8 or higher
- pip package manager
- Git (for cloning the repository)
- 8GB+ RAM (16GB+ recommended)
- CUDA-compatible GPU (optional, for GPU acceleration)

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd heygen_ai
```

### 2. Install Dependencies

Choose an installation profile based on your needs:

```bash
# Minimal installation (core only)
python install_requirements.py minimal

# Basic installation (core + ML)
python install_requirements.py basic

# Web installation (core + ML + web framework)
python install_requirements.py web

# Enterprise installation (all features)
python install_requirements.py enterprise

# Development installation (includes dev tools)
python install_requirements.py dev

# Full installation (everything)
python install_requirements.py full
```

### 3. Verify Installation

```bash
# Check system requirements
python install_requirements.py check

# Run demo launcher
python launch_demos.py
```

## ⚙️ Configuration

### 1. Environment Setup

Copy the example environment file:
```bash
cp .env.example .env
```

Edit `.env` with your settings:
```env
HEYGEN_AI_ENVIRONMENT=development
HEYGEN_AI_DEBUG=true
HEYGEN_AI_LOG_LEVEL=INFO
HEYGEN_AI_API_HOST=0.0.0.0
HEYGEN_AI_API_PORT=8000
```

### 2. Configuration File

The main configuration is in `configs/main/heygen_ai_config.yaml`:
- System settings
- Plugin configuration
- Model settings
- Performance options
- Security settings

## 🧪 Testing

Run the test suite:
```bash
# Run all tests
python -m pytest

# Run with coverage
python -m pytest --cov=src

# Run specific test categories
python -m pytest tests/unit/
python -m pytest tests/integration/
```

## 🚀 Running the System

### Development Mode

```bash
# Start with auto-reload
python src/main.py --dev

# Start specific components
python src/api/main.py
python src/plugins/manager.py
```

### Production Mode

```bash
# Start production server
python src/main.py --production

# Use gunicorn for production
gunicorn src.api.main:app -w 4 -k uvicorn.workers.UvicornWorker
```

## 🔌 Plugin Development

### Creating a Plugin

1. Create plugin directory in `src/plugins/`
2. Implement required interfaces
3. Add metadata and configuration
4. Test plugin functionality
5. Register with plugin manager

### Plugin Structure

```
src/plugins/my_plugin/
├── __init__.py
├── plugin.py
├── metadata.json
└── requirements.txt
```

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**: Check that all dependencies are installed
2. **Memory Issues**: Reduce batch sizes or enable memory optimization
3. **GPU Issues**: Verify CUDA installation and compatibility
4. **Plugin Errors**: Check plugin configuration and dependencies

### Getting Help

- Check the logs in `logs/` directory
- Review configuration files
- Run system diagnostics
- Check GitHub issues

## 📚 Next Steps

After setup:
1. Explore the demo launcher
2. Try different plugin configurations
3. Customize model settings
4. Develop custom plugins
5. Deploy to production

## 🤝 Support

- Documentation: `docs/` directory
- Issues: GitHub Issues
- Discussions: GitHub Discussions
- Wiki: Project Wiki
"""
        
        with open(guide_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"  Created: {guide_path}")

def main():
    """Main function."""
    project_root = Path(__file__).parent
    
    print("🚀 HeyGen AI Project Organizer")
    print("=" * 40)
    print(f"Project root: {project_root}")
    
    # Confirm before proceeding
    response = input("\nThis will reorganize your project structure. Continue? (y/N): ")
    if response.lower() != 'y':
        print("❌ Organization cancelled")
        return
    
    # Create organizer and run
    organizer = ProjectOrganizer(project_root)
    organizer.organize_project()
    
    print("\n🎉 Project organization completed!")
    print("\nNext steps:")
    print("1. Review the new structure")
    print("2. Update any remaining import statements")
    print("3. Test the system functionality")
    print("4. Update your IDE project settings")

if __name__ == "__main__":
    main()
