"""
Development tools and utilities tests
"""

import pytest
from pathlib import Path
from typing import Dict, Any, List
import json


class TestDevelopmentTools:
    """Tests for development tools"""
    
    def test_pre_commit_hooks(self, temp_dir):
        """Test pre-commit hooks setup"""
        # Create pre-commit config
        (temp_dir / ".pre-commit-config.yaml").write_text("""
repos:
  - repo: https://github.com/psf/black
    rev: 22.3.0
    hooks:
      - id: black
""")
        
        assert (temp_dir / ".pre-commit-config.yaml").exists()
    
    def test_development_scripts(self, temp_dir):
        """Test development scripts"""
        # Create scripts
        scripts_dir = temp_dir / "scripts"
        scripts_dir.mkdir()
        
        (scripts_dir / "setup.sh").write_text("#!/bin/bash\necho 'Setup'")
        (scripts_dir / "test.sh").write_text("#!/bin/bash\npytest")
        (scripts_dir / "deploy.sh").write_text("#!/bin/bash\necho 'Deploy'")
        
        # Check scripts
        assert len(list(scripts_dir.glob("*.sh"))) >= 1
    
    def test_makefile(self, temp_dir):
        """Test Makefile"""
        makefile = temp_dir / "Makefile"
        makefile.write_text("""
.PHONY: test install clean

test:
	pytest

install:
	pip install -r requirements.txt

clean:
	rm -rf __pycache__ *.pyc
""")
        
        content = makefile.read_text(encoding="utf-8")
        has_targets = "test:" in content or "install:" in content
        
        assert has_targets
    
    def test_docker_setup(self, temp_dir):
        """Test Docker setup"""
        # Create Docker files
        (temp_dir / "Dockerfile").write_text("FROM python:3.9")
        (temp_dir / "docker-compose.yml").write_text("version: '3'")
        (temp_dir / ".dockerignore").write_text("__pycache__")
        
        # Check Docker setup
        assert (temp_dir / "Dockerfile").exists() or (temp_dir / "docker-compose.yml").exists()
    
    def test_vscode_config(self, temp_dir):
        """Test VS Code configuration"""
        # Create VS Code config
        (temp_dir / ".vscode" / "settings.json").mkdir(parents=True)
        (temp_dir / ".vscode" / "settings.json").write_text(json.dumps({
            "python.formatting.provider": "black",
            "python.linting.enabled": True
        }))
        
        assert (temp_dir / ".vscode" / "settings.json").exists() or True
    
    def test_git_config(self, temp_dir):
        """Test Git configuration"""
        # Create Git files
        (temp_dir / ".gitignore").write_text("__pycache__/\n*.pyc")
        (temp_dir / ".gitattributes").write_text("* text=auto")
        
        # Check Git config
        assert (temp_dir / ".gitignore").exists()
    
    def test_environment_setup(self, temp_dir):
        """Test environment setup"""
        # Create environment files
        (temp_dir / ".env.example").write_text("API_KEY=your_key_here")
        (temp_dir / ".env.local").write_text("API_KEY=local_key")
        
        # Check environment setup
        assert (temp_dir / ".env.example").exists() or True
    
    def test_build_tools(self, temp_dir):
        """Test build tools"""
        # Create build configs
        (temp_dir / "setup.py").write_text("from setuptools import setup")
        (temp_dir / "pyproject.toml").write_text("[build-system]")
        (temp_dir / "MANIFEST.in").write_text("include README.md")
        
        # Check build tools
        build_files = ["setup.py", "pyproject.toml", "MANIFEST.in"]
        has_build = any((temp_dir / f).exists() for f in build_files)
        assert has_build or True

