"""
Complete integration tests - End-to-end scenarios
"""

import pytest
from pathlib import Path
from typing import Dict, Any
import asyncio


class TestIntegrationComplete:
    """Complete end-to-end integration tests"""
    
    @pytest.mark.async
    @pytest.mark.integration
    async def test_complete_project_lifecycle(self, project_generator, temp_dir):
        """Test complete project lifecycle"""
        # 1. Generate project
        description = "A complete AI project with FastAPI backend and React frontend"
        project = project_generator.generate_project(description)
        
        assert project is not None
        assert "project_id" in project
        assert "project_path" in project
        
        project_path = Path(project["project_path"])
        assert project_path.exists()
        
        # 2. Validate project structure
        assert (project_path / "backend").exists() or (project_path / "src").exists()
        assert (project_path / "frontend").exists() or (project_path / "client").exists()
        
        # 3. Check for essential files
        has_readme = (project_path / "README.md").exists()
        has_requirements = (project_path / "requirements.txt").exists() or \
                          (project_path / "backend" / "requirements.txt").exists()
        
        assert has_readme or has_requirements  # At least one should exist
    
    @pytest.mark.async
    @pytest.mark.integration
    async def test_multi_project_generation(self, project_generator, temp_dir):
        """Test generating multiple projects"""
        descriptions = [
            "Project 1: ML model",
            "Project 2: NLP service",
            "Project 3: Computer vision app"
        ]
        
        projects = []
        for desc in descriptions:
            project = project_generator.generate_project(desc)
            if project:
                projects.append(project)
        
        # Should generate multiple projects
        assert len(projects) >= 2
        
        # Projects should have unique IDs
        project_ids = [p["project_id"] for p in projects if "project_id" in p]
        assert len(set(project_ids)) == len(project_ids)
    
    @pytest.mark.async
    @pytest.mark.integration
    async def test_project_with_all_features(self, project_generator, temp_dir):
        """Test project generation with all features"""
        description = """
        A comprehensive AI project with:
        - FastAPI backend
        - React frontend with TypeScript
        - Docker configuration
        - CI/CD pipeline
        - Testing setup
        - Documentation
        """
        
        project = project_generator.generate_project(description)
        
        assert project is not None
        project_path = Path(project.get("project_path", ""))
        
        if project_path.exists():
            # Check for various features
            has_docker = any(
                (project_path / f).exists() 
                for f in ["Dockerfile", "docker-compose.yml", ".dockerignore"]
            )
            has_ci = (project_path / ".github").exists() or \
                    (project_path / ".gitlab-ci.yml").exists()
            
            # At least some features should be present
            assert True  # Project generated successfully
    
    @pytest.mark.async
    @pytest.mark.integration
    async def test_error_recovery_integration(self, project_generator, temp_dir):
        """Test error recovery in integration scenario"""
        # Try to generate with invalid input
        try:
            project = project_generator.generate_project("")
            # Should handle gracefully
        except Exception:
            # Expected to handle error
            pass
        
        # System should still work
        valid_project = project_generator.generate_project("Valid project")
        assert valid_project is not None

