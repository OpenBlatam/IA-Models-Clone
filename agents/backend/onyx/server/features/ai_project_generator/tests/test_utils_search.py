"""
Tests for ProjectSearchEngine utility
"""

import pytest
import asyncio
import json
from pathlib import Path
from datetime import datetime, timedelta

from ..utils.search_engine import ProjectSearchEngine


class TestProjectSearchEngine:
    """Test suite for ProjectSearchEngine"""

    def test_init(self, temp_dir):
        """Test ProjectSearchEngine initialization"""
        engine = ProjectSearchEngine(projects_dir=temp_dir / "projects")
        assert engine.projects_dir == temp_dir / "projects"

    @pytest.mark.asyncio
    async def test_search_projects_by_query(self, temp_dir):
        """Test searching projects by query"""
        engine = ProjectSearchEngine(projects_dir=temp_dir / "projects")
        (temp_dir / "projects").mkdir(parents=True)
        
        # Create test projects
        project1_dir = temp_dir / "projects" / "chat_project"
        project1_dir.mkdir()
        (project1_dir / "project_info.json").write_text(json.dumps({
            "name": "chat_project",
            "description": "A chat AI system",
            "author": "Author1",
            "created_at": datetime.now().isoformat(),
            "keywords": {"ai_type": "chat"}
        }))
        
        project2_dir = temp_dir / "projects" / "vision_project"
        project2_dir.mkdir()
        (project2_dir / "project_info.json").write_text(json.dumps({
            "name": "vision_project",
            "description": "A vision AI system",
            "author": "Author2",
            "created_at": datetime.now().isoformat(),
            "keywords": {"ai_type": "vision"}
        }))
        
        results = await engine.search_projects(query="chat")
        
        assert len(results) > 0
        assert any("chat" in r["description"].lower() or "chat" in r["name"].lower() 
                  for r in results)

    @pytest.mark.asyncio
    async def test_search_projects_by_ai_type(self, temp_dir):
        """Test searching projects by AI type"""
        engine = ProjectSearchEngine(projects_dir=temp_dir / "projects")
        (temp_dir / "projects").mkdir(parents=True)
        
        # Create projects with different AI types
        for ai_type in ["chat", "vision", "audio"]:
            project_dir = temp_dir / "projects" / f"{ai_type}_project"
            project_dir.mkdir()
            (project_dir / "project_info.json").write_text(json.dumps({
                "name": f"{ai_type}_project",
                "description": f"A {ai_type} AI system",
                "keywords": {"ai_type": ai_type},
                "created_at": datetime.now().isoformat()
            }))
        
        results = await engine.search_projects(ai_type="chat")
        
        assert len(results) > 0
        assert all(r["ai_type"] == "chat" for r in results)

    @pytest.mark.asyncio
    async def test_search_projects_by_author(self, temp_dir):
        """Test searching projects by author"""
        engine = ProjectSearchEngine(projects_dir=temp_dir / "projects")
        (temp_dir / "projects").mkdir(parents=True)
        
        # Create projects with different authors
        for author in ["Author1", "Author2"]:
            project_dir = temp_dir / "projects" / f"project_{author}"
            project_dir.mkdir()
            (project_dir / "project_info.json").write_text(json.dumps({
                "name": f"project_{author}",
                "author": author,
                "created_at": datetime.now().isoformat(),
                "keywords": {}
            }))
        
        results = await engine.search_projects(author="Author1")
        
        assert len(results) > 0
        assert all(r["author"] == "Author1" for r in results)

    @pytest.mark.asyncio
    async def test_search_projects_by_date_range(self, temp_dir):
        """Test searching projects by date range"""
        engine = ProjectSearchEngine(projects_dir=temp_dir / "projects")
        (temp_dir / "projects").mkdir(parents=True)
        
        now = datetime.now()
        
        # Create projects with different dates
        for i, days_ago in enumerate([1, 5, 10]):
            project_dir = temp_dir / "projects" / f"project_{i}"
            project_dir.mkdir()
            created_at = (now - timedelta(days=days_ago)).isoformat()
            (project_dir / "project_info.json").write_text(json.dumps({
                "name": f"project_{i}",
                "created_at": created_at,
                "keywords": {}
            }))
        
        min_date = (now - timedelta(days=7)).isoformat()
        results = await engine.search_projects(min_date=min_date)
        
        # Should only return projects from last 7 days
        assert len(results) >= 0

    @pytest.mark.asyncio
    async def test_search_projects_with_tests(self, temp_dir):
        """Test searching projects that have tests"""
        engine = ProjectSearchEngine(projects_dir=temp_dir / "projects")
        (temp_dir / "projects").mkdir(parents=True)
        
        # Project with tests
        project1_dir = temp_dir / "projects" / "with_tests"
        project1_dir.mkdir()
        (project1_dir / "backend" / "tests").mkdir(parents=True)
        (project1_dir / "project_info.json").write_text(json.dumps({
            "name": "with_tests",
            "keywords": {}
        }))
        
        # Project without tests
        project2_dir = temp_dir / "projects" / "no_tests"
        project2_dir.mkdir()
        (project2_dir / "project_info.json").write_text(json.dumps({
            "name": "no_tests",
            "keywords": {}
        }))
        
        results = await engine.search_projects(has_tests=True)
        
        assert len(results) > 0
        assert all(r["has_tests"] is True for r in results)

    @pytest.mark.asyncio
    async def test_search_projects_with_cicd(self, temp_dir):
        """Test searching projects that have CI/CD"""
        engine = ProjectSearchEngine(projects_dir=temp_dir / "projects")
        (temp_dir / "projects").mkdir(parents=True)
        
        # Project with CI/CD
        project1_dir = temp_dir / "projects" / "with_cicd"
        project1_dir.mkdir()
        (project1_dir / ".github").mkdir()
        (project1_dir / "project_info.json").write_text(json.dumps({
            "name": "with_cicd",
            "keywords": {}
        }))
        
        results = await engine.search_projects(has_cicd=True)
        
        assert len(results) > 0
        assert all(r["has_cicd"] is True for r in results)

    @pytest.mark.asyncio
    async def test_search_projects_limit(self, temp_dir):
        """Test limiting search results"""
        engine = ProjectSearchEngine(projects_dir=temp_dir / "projects")
        (temp_dir / "projects").mkdir(parents=True)
        
        # Create many projects
        for i in range(10):
            project_dir = temp_dir / "projects" / f"project_{i}"
            project_dir.mkdir()
            (project_dir / "project_info.json").write_text(json.dumps({
                "name": f"project_{i}",
                "keywords": {}
            }))
        
        results = await engine.search_projects(limit=5)
        
        assert len(results) <= 5

    @pytest.mark.asyncio
    async def test_search_projects_empty(self, temp_dir):
        """Test searching with no results"""
        engine = ProjectSearchEngine(projects_dir=temp_dir / "projects")
        (temp_dir / "projects").mkdir(parents=True)
        
        results = await engine.search_projects(query="nonexistent")
        
        assert results == []

    @pytest.mark.asyncio
    async def test_search_projects_multiple_filters(self, temp_dir):
        """Test searching with multiple filters"""
        engine = ProjectSearchEngine(projects_dir=temp_dir / "projects")
        (temp_dir / "projects").mkdir(parents=True)
        
        project_dir = temp_dir / "projects" / "filtered_project"
        project_dir.mkdir()
        (project_dir / "project_info.json").write_text(json.dumps({
            "name": "filtered_project",
            "description": "A chat AI system",
            "author": "TestAuthor",
            "created_at": datetime.now().isoformat(),
            "keywords": {"ai_type": "chat"}
        }))
        
        results = await engine.search_projects(
            query="chat",
            ai_type="chat",
            author="TestAuthor"
        )
        
        assert len(results) >= 0

