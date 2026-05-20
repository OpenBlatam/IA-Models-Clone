"""
Tests for DashboardGenerator utility
"""

import pytest
import asyncio
from pathlib import Path

from ..utils.dashboard_generator import DashboardGenerator


class TestDashboardGenerator:
    """Test suite for DashboardGenerator"""

    def test_init(self):
        """Test DashboardGenerator initialization"""
        generator = DashboardGenerator()
        assert generator is not None

    @pytest.mark.asyncio
    async def test_generate_dashboard(self, temp_dir):
        """Test generating dashboard"""
        generator = DashboardGenerator()
        
        output_dir = temp_dir / "dashboard"
        
        await generator.generate_dashboard(output_dir, api_url="http://localhost:8020")
        
        assert output_dir.exists()
        assert (output_dir / "index.html").exists()

    @pytest.mark.asyncio
    async def test_generate_dashboard_files(self, temp_dir):
        """Test that dashboard generates all required files"""
        generator = DashboardGenerator()
        
        output_dir = temp_dir / "dashboard"
        
        await generator.generate_dashboard(output_dir)
        
        # Check main files
        assert (output_dir / "index.html").exists()
        
        # Check that HTML contains expected content
        index_content = (output_dir / "index.html").read_text()
        assert "AI Project Generator" in index_content or "Dashboard" in index_content

    @pytest.mark.asyncio
    async def test_dashboard_api_integration(self, temp_dir):
        """Test dashboard API integration"""
        generator = DashboardGenerator()
        
        output_dir = temp_dir / "dashboard"
        api_url = "http://localhost:8020"
        
        await generator.generate_dashboard(output_dir, api_url=api_url)
        
        index_content = (output_dir / "index.html").read_text()
        assert api_url in index_content

