"""
BUL Optimized Test Suite
========================

Comprehensive tests for the optimized BUL system.
"""

import asyncio
import pytest
import sys
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from modules.document_processor import DocumentProcessor
from modules.query_analyzer import QueryAnalyzer, Complexity
from modules.business_agents import BusinessAgentManager
from modules.api_handler import APIHandler, DocumentRequest
from config_optimized import BULConfig

class TestDocumentProcessor:
    """Test DocumentProcessor module."""
    
    def setup_method(self):
        """Setup test configuration."""
        self.config = {
            'supported_formats': ['markdown', 'html'],
            'output_directory': 'test_documents'
        }
        self.processor = DocumentProcessor(self.config)
    
    @pytest.mark.asyncio
    async def test_generate_document(self):
        """Test document generation."""
        document = await self.processor.generate_document(
            query="Test marketing strategy",
            business_area="marketing",
            document_type="strategy"
        )
        
        assert document['business_area'] == 'marketing'
        assert document['document_type'] == 'strategy'
        assert 'content' in document
        assert 'id' in document
    
    def test_supported_formats(self):
        """Test supported formats configuration."""
        assert 'markdown' in self.processor.supported_formats
        assert 'html' in self.processor.supported_formats

class TestQueryAnalyzer:
    """Test QueryAnalyzer module."""
    
    def setup_method(self):
        """Setup test analyzer."""
        self.analyzer = QueryAnalyzer()
    
    def test_marketing_query_analysis(self):
        """Test marketing query analysis."""
        analysis = self.analyzer.analyze("Create a marketing strategy for a new product")
        
        assert analysis.primary_area == 'marketing'
        assert 'strategy' in analysis.document_types
        assert analysis.complexity in [Complexity.SIMPLE, Complexity.MEDIUM, Complexity.COMPLEX]
        assert 0 <= analysis.confidence <= 1
    
    def test_sales_query_analysis(self):
        """Test sales query analysis."""
        analysis = self.analyzer.analyze("Develop a sales proposal for enterprise clients")
        
        assert analysis.primary_area == 'sales'
        assert 'proposal' in analysis.document_types
        assert analysis.priority >= 1
    
    def test_complexity_assessment(self):
        """Test complexity assessment."""
        simple_query = "Create a simple report"
        complex_query = "Develop a comprehensive strategic plan with detailed analysis"
        
        simple_analysis = self.analyzer.analyze(simple_query)
        complex_analysis = self.analyzer.analyze(complex_query)
        
        # Complex queries should generally have higher priority (lower number)
        assert complex_analysis.priority <= simple_analysis.priority

class TestBusinessAgentManager:
    """Test BusinessAgentManager module."""
    
    def setup_method(self):
        """Setup test agent manager."""
        self.config = {
            'business_areas': {
                'marketing': {'priority': 1},
                'sales': {'priority': 1}
            }
        }
        self.agent_manager = BusinessAgentManager(self.config)
    
    def test_available_areas(self):
        """Test available business areas."""
        areas = self.agent_manager.get_available_areas()
        assert 'marketing' in areas
        assert 'sales' in areas
    
    def test_get_agent(self):
        """Test getting specific agent."""
        marketing_agent = self.agent_manager.get_agent('marketing')
        assert marketing_agent is not None
        assert marketing_agent.area == 'marketing'
        
        nonexistent_agent = self.agent_manager.get_agent('nonexistent')
        assert nonexistent_agent is None
    
    @pytest.mark.asyncio
    async def test_process_with_agent(self):
        """Test processing with specific agent."""
        result = await self.agent_manager.process_with_agent(
            'marketing', 'Test query', 'strategy'
        )
        
        assert result is not None
        assert result['area'] == 'marketing'
        assert result['document_type'] == 'strategy'

class TestAPIHandler:
    """Test APIHandler module."""
    
    def setup_method(self):
        """Setup test API handler."""
        self.config = {
            'supported_formats': ['markdown'],
            'output_directory': 'test_documents'
        }
        self.processor = DocumentProcessor(self.config)
        self.analyzer = QueryAnalyzer()
        self.agent_manager = BusinessAgentManager(self.config)
        self.api_handler = APIHandler(
            self.processor, self.analyzer, self.agent_manager
        )
    
    @pytest.mark.asyncio
    async def test_generate_document_request(self):
        """Test document generation request."""
        request = DocumentRequest(
            query="Test marketing strategy",
            business_area="marketing",
            document_type="strategy",
            priority=1
        )
        
        response = await self.api_handler.generate_document(request)
        
        assert response.task_id is not None
        assert response.status == "queued"
        assert response.message == "Document generation started"
    
    def test_system_info(self):
        """Test system info endpoint."""
        info = self.api_handler.get_system_info()
        
        assert info['system'] == "BUL - Business Universal Language"
        assert info['version'] == "3.0.0"
        assert 'available_areas' in info
    
    def test_health_status(self):
        """Test health status endpoint."""
        health = self.api_handler.get_health_status()
        
        assert health['status'] == "healthy"
        assert 'components' in health
        assert health['components']['document_processor'] == "operational"

class TestConfiguration:
    """Test configuration management."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = BULConfig()
        
        assert config.api_host == "0.0.0.0"
        assert config.api_port == 8000
        assert config.debug_mode is False
        assert config.max_concurrent_tasks == 5
    
    def test_config_validation(self):
        """Test configuration validation."""
        config = BULConfig()
        errors = config.validate_config()
        
        # Should have error about missing API keys
        assert len(errors) > 0
        assert any("API_KEY" in error for error in errors)
    
    def test_business_area_config(self):
        """Test business area configuration."""
        config = BULConfig()
        area_config = config.get_business_area_config('marketing')
        
        assert area_config['enabled'] is True
        assert 'priority' in area_config

# Integration Tests
class TestIntegration:
    """Integration tests for the complete system."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_document_generation(self):
        """Test complete document generation flow."""
        # Setup
        config = {
            'supported_formats': ['markdown'],
            'output_directory': 'test_documents'
        }
        
        processor = DocumentProcessor(config)
        analyzer = QueryAnalyzer()
        agent_manager = BusinessAgentManager(config)
        api_handler = APIHandler(processor, analyzer, agent_manager)
        
        # Create request
        request = DocumentRequest(
            query="Create a comprehensive marketing strategy for a new restaurant",
            business_area="marketing",
            document_type="strategy",
            priority=1
        )
        
        # Process request
        response = await api_handler.generate_document(request)
        
        # Verify response
        assert response.task_id is not None
        assert response.status == "queued"
        
        # Wait for processing (in real scenario, this would be handled by background tasks)
        await asyncio.sleep(0.1)  # Small delay for async processing
        
        # Check task status
        status = await api_handler.get_task_status(response.task_id)
        assert status.task_id == response.task_id
        assert status.status in ["queued", "processing", "completed"]

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
