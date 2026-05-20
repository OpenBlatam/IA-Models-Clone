"""
Tests for DeploymentGenerator utility
"""

import pytest
import asyncio
import json
from pathlib import Path

from ..utils.deployment_generator import DeploymentGenerator


class TestDeploymentGenerator:
    """Test suite for DeploymentGenerator"""

    def test_init(self):
        """Test DeploymentGenerator initialization"""
        generator = DeploymentGenerator()
        assert generator is not None

    @pytest.mark.asyncio
    async def test_generate_vercel_config(self, temp_dir, sample_project_info):
        """Test generating Vercel configuration"""
        generator = DeploymentGenerator()
        project_dir = temp_dir / "vercel_project"
        (project_dir / "frontend").mkdir(parents=True)
        (project_dir / "backend").mkdir()
        
        await generator.generate_vercel_config(project_dir, sample_project_info)
        
        assert (project_dir / "vercel.json").exists()
        assert (project_dir / ".vercelignore").exists()
        
        # Verify vercel.json content
        vercel_config = json.loads((project_dir / "vercel.json").read_text())
        assert "version" in vercel_config
        assert "builds" in vercel_config
        assert "routes" in vercel_config

    @pytest.mark.asyncio
    async def test_generate_netlify_config(self, temp_dir, sample_project_info):
        """Test generating Netlify configuration"""
        generator = DeploymentGenerator()
        project_dir = temp_dir / "netlify_project"
        (project_dir / "frontend").mkdir(parents=True)
        
        await generator.generate_netlify_config(project_dir, sample_project_info)
        
        assert (project_dir / "netlify.toml").exists()
        
        content = (project_dir / "netlify.toml").read_text()
        assert "[build]" in content
        assert "redirects" in content

    @pytest.mark.asyncio
    async def test_generate_railway_config(self, temp_dir, sample_project_info):
        """Test generating Railway configuration"""
        generator = DeploymentGenerator()
        project_dir = temp_dir / "railway_project"
        project_dir.mkdir()
        
        await generator.generate_railway_config(project_dir, sample_project_info)
        
        assert (project_dir / "railway.json").exists()
        
        railway_config = json.loads((project_dir / "railway.json").read_text())
        assert isinstance(railway_config, dict)

    @pytest.mark.asyncio
    async def test_generate_heroku_config(self, temp_dir, sample_project_info):
        """Test generating Heroku configuration"""
        generator = DeploymentGenerator()
        project_dir = temp_dir / "heroku_project"
        (project_dir / "backend").mkdir(parents=True)
        (project_dir / "backend" / "requirements.txt").write_text("fastapi")
        
        await generator.generate_heroku_config(project_dir, sample_project_info)
        
        assert (project_dir / "Procfile").exists()
        assert (project_dir / "runtime.txt").exists()
        
        procfile = (project_dir / "Procfile").read_text()
        assert "web:" in procfile

    @pytest.mark.asyncio
    async def test_generate_all_platforms(self, temp_dir, sample_project_info):
        """Test generating configs for all platforms"""
        generator = DeploymentGenerator()
        project_dir = temp_dir / "all_platforms"
        (project_dir / "frontend").mkdir(parents=True)
        (project_dir / "backend").mkdir()
        (project_dir / "backend" / "requirements.txt").write_text("fastapi")
        
        await generator.generate_vercel_config(project_dir, sample_project_info)
        await generator.generate_netlify_config(project_dir, sample_project_info)
        await generator.generate_railway_config(project_dir, sample_project_info)
        await generator.generate_heroku_config(project_dir, sample_project_info)
        
        assert (project_dir / "vercel.json").exists()
        assert (project_dir / "netlify.toml").exists()
        assert (project_dir / "railway.json").exists()
        assert (project_dir / "Procfile").exists()

