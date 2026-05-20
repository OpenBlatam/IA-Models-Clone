"""
Contract testing - Testing contracts between components
"""

import pytest
from typing import Dict, Any, List, Protocol
from abc import ABC, abstractmethod


class ProjectGeneratorContract(Protocol):
    """Contract that ProjectGenerator must satisfy"""
    
    def generate_project(self, description: str) -> Dict[str, Any]:
        """Must return a dict with project information"""
        ...
    
    def _sanitize_name(self, name: str) -> str:
        """Must return a sanitized string"""
        ...


class TestContracts:
    """Tests for component contracts"""
    
    def test_project_generator_contract(self, project_generator):
        """Test that ProjectGenerator satisfies its contract"""
        # Contract: generate_project returns dict
        result = project_generator.generate_project("A test project")
        assert isinstance(result, dict), "generate_project must return dict"
        
        # Contract: _sanitize_name returns string
        result = project_generator._sanitize_name("Test")
        assert isinstance(result, str), "_sanitize_name must return string"
        assert len(result) <= 50, "_sanitize_name result must be <= 50 chars"
    
    def test_backend_generator_contract(self, backend_generator):
        """Test that BackendGenerator satisfies its contract"""
        # Contract: generate_backend returns dict
        import asyncio
        result = asyncio.run(backend_generator.generate_backend("fastapi", []))
        assert isinstance(result, dict), "generate_backend must return dict"
    
    def test_frontend_generator_contract(self, frontend_generator):
        """Test that FrontendGenerator satisfies its contract"""
        # Contract: generate_frontend returns dict
        import asyncio
        result = asyncio.run(frontend_generator.generate_frontend("react", []))
        assert isinstance(result, dict), "generate_frontend must return dict"

