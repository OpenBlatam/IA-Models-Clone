"""
Tests for CICDGenerator utility
"""

import pytest
import asyncio
from pathlib import Path

from ..utils.cicd_generator import CICDGenerator


class TestCICDGenerator:
    """Test suite for CICDGenerator"""

    def test_init(self):
        """Test CICDGenerator initialization"""
        generator = CICDGenerator()
        assert generator is not None

    @pytest.mark.asyncio
    async def test_generate_github_actions(self, temp_dir, sample_keywords, sample_project_info):
        """Test generating GitHub Actions workflows"""
        generator = CICDGenerator()
        project_dir = temp_dir / "cicd_project"
        (project_dir / "backend").mkdir(parents=True)
        (project_dir / "frontend").mkdir()
        
        await generator.generate_github_actions(project_dir, sample_keywords, sample_project_info)
        
        workflows_dir = project_dir / ".github" / "workflows"
        assert workflows_dir.exists()
        assert (workflows_dir / "backend-ci.yml").exists()
        assert (workflows_dir / "frontend-ci.yml").exists()

    @pytest.mark.asyncio
    async def test_generate_github_actions_backend_ci(self, temp_dir, sample_keywords, sample_project_info):
        """Test backend CI workflow content"""
        generator = CICDGenerator()
        project_dir = temp_dir / "backend_ci"
        (project_dir / "backend").mkdir(parents=True)
        
        await generator.generate_github_actions(project_dir, sample_keywords, sample_project_info)
        
        backend_ci = project_dir / ".github" / "workflows" / "backend-ci.yml"
        assert backend_ci.exists()
        
        content = backend_ci.read_text()
        assert "Backend CI" in content
        assert "pytest" in content
        assert "python" in content.lower()

    @pytest.mark.asyncio
    async def test_generate_github_actions_frontend_ci(self, temp_dir, sample_keywords, sample_project_info):
        """Test frontend CI workflow content"""
        generator = CICDGenerator()
        project_dir = temp_dir / "frontend_ci"
        (project_dir / "frontend").mkdir(parents=True)
        
        await generator.generate_github_actions(project_dir, sample_keywords, sample_project_info)
        
        frontend_ci = project_dir / ".github" / "workflows" / "frontend-ci.yml"
        assert frontend_ci.exists()
        
        content = frontend_ci.read_text()
        assert "Frontend CI" in content
        assert "node" in content.lower() or "npm" in content.lower()

    @pytest.mark.asyncio
    async def test_generate_github_actions_creates_directories(self, temp_dir, sample_keywords, sample_project_info):
        """Test that GitHub Actions directories are created"""
        generator = CICDGenerator()
        project_dir = temp_dir / "new_cicd"
        project_dir.mkdir()
        
        await generator.generate_github_actions(project_dir, sample_keywords, sample_project_info)
        
        assert (project_dir / ".github").exists()
        assert (project_dir / ".github" / "workflows").exists()

