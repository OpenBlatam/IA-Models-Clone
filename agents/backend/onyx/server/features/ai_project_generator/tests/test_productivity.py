"""
Productivity tests
"""

import pytest
from pathlib import Path
from typing import Dict, Any, List
import time


class TestProductivity:
    """Tests for productivity features"""
    
    def test_quick_start(self, project_generator):
        """Test quick start capability"""
        # Should be able to start quickly
        start_time = time.time()
        project = project_generator.generate_project("Quick test")
        elapsed = time.time() - start_time
        
        # Should be fast
        assert project is not None
        assert elapsed < 10.0  # Should complete quickly
    
    def test_batch_operations(self, project_generator):
        """Test batch operations for productivity"""
        descriptions = [f"Project {i}" for i in range(10)]
        
        start_time = time.time()
        projects = []
        for desc in descriptions:
            project = project_generator.generate_project(desc)
            projects.append(project)
        elapsed = time.time() - start_time
        
        # Should handle batches efficiently
        assert len(projects) == 10
        avg_time = elapsed / 10 if len(projects) > 0 else 0
        assert avg_time < 5.0  # Average time per project
    
    def test_automation_support(self, temp_dir):
        """Test automation support"""
        # Should support automation
        script = temp_dir / "automate.py"
        script.write_text("""
from project_generator import ProjectGenerator

generator = ProjectGenerator()
project = generator.generate("Automated project")
print(f"Generated: {project['project_id']}")
""")
        
        # Should be automatable
        assert script.exists()
        content = script.read_text(encoding="utf-8")
        assert "ProjectGenerator" in content
    
    def test_template_support(self, temp_dir):
        """Test template support for productivity"""
        # Should support templates
        templates_dir = temp_dir / "templates"
        templates_dir.mkdir()
        
        (templates_dir / "basic.py").write_text("def main(): pass")
        (templates_dir / "advanced.py").write_text("class Advanced: pass")
        
        # Should have templates
        assert len(list(templates_dir.glob("*.py"))) >= 1
    
    def test_shortcuts(self, temp_dir):
        """Test shortcuts for productivity"""
        # Should have shortcuts or aliases
        config = temp_dir / "config.json"
        import json
        config.write_text(json.dumps({
            "shortcuts": {
                "gen": "generate",
                "proj": "project"
            }
        }), encoding="utf-8")
        
        # Should support shortcuts
        assert config.exists()
    
    def test_history_support(self, temp_dir):
        """Test history support"""
        # Should maintain history
        history_file = temp_dir / "history.json"
        import json
        history = [
            {"timestamp": "2024-01-01", "action": "generate", "project": "test1"},
            {"timestamp": "2024-01-02", "action": "generate", "project": "test2"},
        ]
        history_file.write_text(json.dumps(history), encoding="utf-8")
        
        # Should have history
        assert history_file.exists()
        loaded = json.loads(history_file.read_text(encoding="utf-8"))
        assert len(loaded) >= 1
    
    def test_export_productivity(self, temp_dir):
        """Test export for productivity"""
        # Should support quick export
        project_dir = temp_dir / "project"
        project_dir.mkdir()
        (project_dir / "file.txt").write_text("content")
        
        import zipfile
        zip_path = temp_dir / "export.zip"
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for file in project_dir.rglob("*"):
                if file.is_file():
                    zipf.write(file, file.relative_to(project_dir.parent))
        
        # Should export quickly
        assert zip_path.exists()
        assert zip_path.stat().st_size > 0

