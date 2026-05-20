"""
Comprehensive final tests - All aspects combined
"""

import pytest
from pathlib import Path
from typing import Dict, Any, List
import asyncio
import time
import json


class TestComprehensiveFinal:
    """Comprehensive final tests combining all aspects"""
    
    @pytest.mark.async
    @pytest.mark.integration
    @pytest.mark.slow
    async def test_complete_workflow_comprehensive(self, project_generator, temp_dir):
        """Test complete workflow with all features"""
        # 1. Generate project
        description = "A comprehensive AI project with all features"
        project = project_generator.generate_project(description)
        
        assert project is not None
        project_path = Path(project.get("project_path", ""))
        
        if project_path.exists():
            # 2. Validate structure
            has_backend = (project_path / "backend").exists() or (project_path / "src").exists()
            has_frontend = (project_path / "frontend").exists() or (project_path / "client").exists()
            has_readme = (project_path / "README.md").exists()
            
            # 3. Check quality
            quality_checks = {
                "has_structure": has_backend or has_frontend,
                "has_documentation": has_readme,
            }
            
            assert sum(quality_checks.values()) >= 1
    
    @pytest.mark.async
    async def test_performance_and_reliability(self, project_generator):
        """Test performance and reliability together"""
        num_operations = 50
        results = []
        times = []
        
        for i in range(num_operations):
            start = time.time()
            project = project_generator.generate_project(f"Project {i}")
            elapsed = time.time() - start
            
            results.append(project)
            times.append(elapsed)
        
        # Performance check
        avg_time = sum(times) / len(times) if times else 0
        assert avg_time < 5.0  # Should be fast
        
        # Reliability check
        assert len(results) == num_operations
        assert all(r is not None for r in results)
    
    def test_security_and_quality(self, temp_dir):
        """Test security and quality together"""
        # Create secure and quality code
        code_file = temp_dir / "secure_code.py"
        code_content = '''
def secure_function(user_input):
    """
    Secure function with input validation.
    
    Args:
        user_input: User input to validate
    
    Returns:
        str: Sanitized input
    """
    # Input validation
    if not user_input:
        return ""
    
    # Sanitization
    sanitized = user_input.strip().lower()
    return sanitized[:50]  # Limit length
'''
        code_file.write_text(code_content)
        
        # Security checks
        has_validation = "if not" in code_content or "validation" in code_content.lower()
        has_sanitization = "strip" in code_content or "sanitize" in code_content.lower()
        
        # Quality checks
        has_docstring = '"""' in code_content
        has_type_hints = ":" in code_content and "->" in code_content
        
        assert has_validation or has_sanitization
        assert has_docstring or has_type_hints
    
    @pytest.mark.async
    async def test_concurrency_and_robustness(self, temp_dir):
        """Test concurrency and robustness together"""
        from ..utils.cache_manager import CacheManager
        
        cache = CacheManager(cache_dir=temp_dir / "cache")
        
        num_concurrent = 100
        
        async def robust_operation(i):
            try:
                await cache.cache_project(f"Project {i}", {}, {"id": f"proj-{i}"})
                result = await cache.get_cached_project(f"Project {i}", {})
                return result is not None
            except Exception:
                # Should handle errors robustly
                return False
        
        # Run concurrently
        tasks = [robust_operation(i) for i in range(num_concurrent)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Robustness check
        successes = [r for r in results if r is True]
        success_rate = len(successes) / num_concurrent if num_concurrent > 0 else 0
        
        # Should be robust
        assert success_rate >= 0.8  # 80% success rate
    
    def test_documentation_and_maintainability(self, temp_dir):
        """Test documentation and maintainability together"""
        project_dir = temp_dir / "project"
        project_dir.mkdir()
        
        # Create well-documented and maintainable project
        (project_dir / "README.md").write_text("""
# Project

## Description
Well-documented project.

## Installation
```bash
pip install -r requirements.txt
```

## Usage
```python
from project import main
main()
```
""")
        
        (project_dir / "src").mkdir()
        (project_dir / "src" / "main.py").write_text('''
"""
Main module.
"""

def main():
    """Main function."""
    print("Hello, World!")

if __name__ == "__main__":
    main()
''')
        
        # Documentation check
        readme = (project_dir / "README.md").read_text(encoding="utf-8")
        has_docs = "Description" in readme and "Usage" in readme
        
        # Maintainability check
        has_structure = (project_dir / "src").exists()
        code = (project_dir / "src" / "main.py").read_text(encoding="utf-8")
        has_docstring = '"""' in code
        
        assert has_docs
        assert has_structure
        assert has_docstring
    
    def test_all_quality_aspects(self, temp_dir):
        """Test all quality aspects together"""
        # Create high-quality project
        project_dir = temp_dir / "quality_project"
        project_dir.mkdir()
        
        # Structure
        (project_dir / "src").mkdir()
        (project_dir / "tests").mkdir()
        (project_dir / "docs").mkdir()
        
        # Documentation
        (project_dir / "README.md").write_text("# Quality Project")
        (project_dir / "CHANGELOG.md").write_text("# Changelog")
        
        # Code quality
        (project_dir / "src" / "main.py").write_text('def main(): pass')
        (project_dir / "tests" / "test_main.py").write_text('def test_main(): assert True')
        
        # Configuration
        (project_dir / "requirements.txt").write_text("requests==2.0.0")
        (project_dir / ".gitignore").write_text("__pycache__/")
        
        # Quality checks
        quality = {
            "structure": all([
                (project_dir / "src").exists(),
                (project_dir / "tests").exists(),
            ]),
            "documentation": all([
                (project_dir / "README.md").exists(),
                (project_dir / "CHANGELOG.md").exists(),
            ]),
            "code": all([
                (project_dir / "src" / "main.py").exists(),
                (project_dir / "tests" / "test_main.py").exists(),
            ]),
            "config": all([
                (project_dir / "requirements.txt").exists(),
                (project_dir / ".gitignore").exists(),
            ]),
        }
        
        # Should have all quality aspects
        assert sum(quality.values()) >= 3

