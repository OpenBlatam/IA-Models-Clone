"""
Optimization Tests
=================

Ultra-advanced optimization tests with pytest.
"""

import json
import pytest
from typing import Dict, Any
from flask import Flask
from app import db
from models import OptimizationSession, PerformanceMetric, User
from utils.exceptions import OptimizationError, ValidationError

class TestOptimizationSessions:
    """Optimization sessions test class."""
    
    def test_create_optimization_session_success(self, client, auth_headers):
        """Test successful optimization session creation."""
        if not auth_headers:
            pytest.skip("Authentication not available")
        
        response = client.post('/api/v1/optimization/sessions', 
                             json={
                                 'session_name': 'Test Optimization',
                                 'optimization_type': 'performance',
                                 'parameters': {'test': 'value'}
                             },
                             headers=auth_headers)
        
        assert response.status_code == 201
        data = json.loads(response.data)
        assert 'session_id' in data
        assert 'session_name' in data
        assert data['session_name'] == 'Test Optimization'
        assert data['optimization_type'] == 'performance'
    
    def test_create_optimization_session_validation_error(self, client, auth_headers):
        """Test optimization session creation with validation error."""
        if not auth_headers:
            pytest.skip("Authentication not available")
        
        response = client.post('/api/v1/optimization/sessions',
                             json={
                                 'session_name': 'Te',  # Too short
                                 'optimization_type': 'invalid_type',  # Invalid type
                                 'parameters': 'invalid'  # Should be dict
                             },
                             headers=auth_headers)
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data
    
    def test_create_optimization_session_unauthenticated(self, client):
        """Test optimization session creation without authentication."""
        response = client.post('/api/v1/optimization/sessions',
                             json={
                                 'session_name': 'Test Optimization',
                                 'optimization_type': 'performance'
                             })
        
        assert response.status_code == 401
        data = json.loads(response.data)
        assert 'error' in data
    
    def test_get_optimization_session_success(self, client, auth_headers, test_session):
        """Test successful optimization session retrieval."""
        if not auth_headers:
            pytest.skip("Authentication not available")
        
        response = client.get(f'/api/v1/optimization/sessions/{test_session.id}',
                            headers=auth_headers)
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'session_id' in data
        assert data['session_id'] == str(test_session.id)
        assert data['session_name'] == 'Test Session'
    
    def test_get_optimization_session_not_found(self, client, auth_headers):
        """Test optimization session retrieval for non-existent session."""
        if not auth_headers:
            pytest.skip("Authentication not available")
        
        response = client.get('/api/v1/optimization/sessions/non-existent-id',
                            headers=auth_headers)
        
        assert response.status_code == 404
        data = json.loads(response.data)
        assert 'error' in data
    
    def test_list_optimization_sessions_success(self, client, auth_headers, test_session):
        """Test successful optimization sessions listing."""
        if not auth_headers:
            pytest.skip("Authentication not available")
        
        response = client.get('/api/v1/optimization/sessions',
                            headers=auth_headers)
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'sessions' in data
        assert 'pagination' in data
        assert len(data['sessions']) >= 1
    
    def test_list_optimization_sessions_with_filters(self, client, auth_headers, test_session):
        """Test optimization sessions listing with filters."""
        if not auth_headers:
            pytest.skip("Authentication not available")
        
        response = client.get('/api/v1/optimization/sessions?type=performance&status=pending',
                            headers=auth_headers)
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'sessions' in data
        assert 'pagination' in data
    
    def test_execute_optimization_success(self, client, auth_headers, test_session):
        """Test successful optimization execution."""
        if not auth_headers:
            pytest.skip("Authentication not available")
        
        response = client.post(f'/api/v1/optimization/sessions/{test_session.id}/execute',
                             headers=auth_headers)
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'session_id' in data
        assert 'status' in data
        assert 'results' in data
        assert 'metrics' in data
    
    def test_execute_optimization_not_found(self, client, auth_headers):
        """Test optimization execution for non-existent session."""
        if not auth_headers:
            pytest.skip("Authentication not available")
        
        response = client.post('/api/v1/optimization/sessions/non-existent-id/execute',
                             headers=auth_headers)
        
        assert response.status_code == 404
        data = json.loads(response.data)
        assert 'error' in data
    
    def test_execute_optimization_already_executed(self, client, auth_headers, test_session):
        """Test optimization execution for already executed session."""
        if not auth_headers:
            pytest.skip("Authentication not available")
        
        # First execution
        client.post(f'/api/v1/optimization/sessions/{test_session.id}/execute',
                   headers=auth_headers)
        
        # Second execution should fail
        response = client.post(f'/api/v1/optimization/sessions/{test_session.id}/execute',
                             headers=auth_headers)
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data
    
    def test_get_optimization_metrics_success(self, client, auth_headers, test_session, test_metric):
        """Test successful optimization metrics retrieval."""
        if not auth_headers:
            pytest.skip("Authentication not available")
        
        response = client.get(f'/api/v1/optimization/sessions/{test_session.id}/metrics',
                            headers=auth_headers)
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'session_id' in data
        assert 'metrics' in data
        assert len(data['metrics']) >= 1
    
    def test_get_optimization_metrics_not_found(self, client, auth_headers):
        """Test optimization metrics retrieval for non-existent session."""
        if not auth_headers:
            pytest.skip("Authentication not available")
        
        response = client.get('/api/v1/optimization/sessions/non-existent-id/metrics',
                            headers=auth_headers)
        
        assert response.status_code == 404
        data = json.loads(response.data)
        assert 'error' in data

class TestOptimizationTypes:
    """Optimization types test class."""
    
    def test_performance_optimization(self, client, auth_headers):
        """Test performance optimization."""
        if not auth_headers:
            pytest.skip("Authentication not available")
        
        # Create session
        response = client.post('/api/v1/optimization/sessions',
                             json={
                                 'session_name': 'Performance Test',
                                 'optimization_type': 'performance'
                             },
                             headers=auth_headers)
        
        assert response.status_code == 201
        data = json.loads(response.data)
        session_id = data['session_id']
        
        # Execute optimization
        response = client.post(f'/api/v1/optimization/sessions/{session_id}/execute',
                             headers=auth_headers)
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['results']['optimization_type'] == 'performance'
        assert 'improvements' in data['results']
    
    def test_memory_optimization(self, client, auth_headers):
        """Test memory optimization."""
        if not auth_headers:
            pytest.skip("Authentication not available")
        
        # Create session
        response = client.post('/api/v1/optimization/sessions',
                             json={
                                 'session_name': 'Memory Test',
                                 'optimization_type': 'memory'
                             },
                             headers=auth_headers)
        
        assert response.status_code == 201
        data = json.loads(response.data)
        session_id = data['session_id']
        
        # Execute optimization
        response = client.post(f'/api/v1/optimization/sessions/{session_id}/execute',
                             headers=auth_headers)
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['results']['optimization_type'] == 'memory'
    
    def test_gpu_optimization(self, client, auth_headers):
        """Test GPU optimization."""
        if not auth_headers:
            pytest.skip("Authentication not available")
        
        # Create session
        response = client.post('/api/v1/optimization/sessions',
                             json={
                                 'session_name': 'GPU Test',
                                 'optimization_type': 'gpu'
                             },
                             headers=auth_headers)
        
        assert response.status_code == 201
        data = json.loads(response.data)
        session_id = data['session_id']
        
        # Execute optimization
        response = client.post(f'/api/v1/optimization/sessions/{session_id}/execute',
                             headers=auth_headers)
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['results']['optimization_type'] == 'gpu'
    
    def test_ml_optimization(self, client, auth_headers):
        """Test ML optimization."""
        if not auth_headers:
            pytest.skip("Authentication not available")
        
        # Create session
        response = client.post('/api/v1/optimization/sessions',
                             json={
                                 'session_name': 'ML Test',
                                 'optimization_type': 'ml'
                             },
                             headers=auth_headers)
        
        assert response.status_code == 201
        data = json.loads(response.data)
        session_id = data['session_id']
        
        # Execute optimization
        response = client.post(f'/api/v1/optimization/sessions/{session_id}/execute',
                             headers=auth_headers)
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['results']['optimization_type'] == 'ml'
    
    def test_quantum_optimization(self, client, auth_headers):
        """Test quantum optimization."""
        if not auth_headers:
            pytest.skip("Authentication not available")
        
        # Create session
        response = client.post('/api/v1/optimization/sessions',
                             json={
                                 'session_name': 'Quantum Test',
                                 'optimization_type': 'quantum'
                             },
                             headers=auth_headers)
        
        assert response.status_code == 201
        data = json.loads(response.data)
        session_id = data['session_id']
        
        # Execute optimization
        response = client.post(f'/api/v1/optimization/sessions/{session_id}/execute',
                             headers=auth_headers)
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['results']['optimization_type'] == 'quantum'
    
    def test_edge_optimization(self, client, auth_headers):
        """Test edge optimization."""
        if not auth_headers:
            pytest.skip("Authentication not available")
        
        # Create session
        response = client.post('/api/v1/optimization/sessions',
                             json={
                                 'session_name': 'Edge Test',
                                 'optimization_type': 'edge'
                             },
                             headers=auth_headers)
        
        assert response.status_code == 201
        data = json.loads(response.data)
        session_id = data['session_id']
        
        # Execute optimization
        response = client.post(f'/api/v1/optimization/sessions/{session_id}/execute',
                             headers=auth_headers)
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['results']['optimization_type'] == 'edge'

class TestOptimizationEdgeCases:
    """Optimization edge cases test class."""
    
    def test_invalid_optimization_type(self, client, auth_headers):
        """Test invalid optimization type."""
        if not auth_headers:
            pytest.skip("Authentication not available")
        
        response = client.post('/api/v1/optimization/sessions',
                             json={
                                 'session_name': 'Invalid Test',
                                 'optimization_type': 'invalid_type'
                             },
                             headers=auth_headers)
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data
    
    def test_missing_required_fields(self, client, auth_headers):
        """Test missing required fields."""
        if not auth_headers:
            pytest.skip("Authentication not available")
        
        response = client.post('/api/v1/optimization/sessions',
                             json={
                                 'session_name': 'Test'
                                 # Missing optimization_type
                             },
                             headers=auth_headers)
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data
    
    def test_large_parameters(self, client, auth_headers):
        """Test large parameters object."""
        if not auth_headers:
            pytest.skip("Authentication not available")
        
        large_params = {'param' + str(i): 'value' + str(i) for i in range(1000)}
        
        response = client.post('/api/v1/optimization/sessions',
                             json={
                                 'session_name': 'Large Params Test',
                                 'optimization_type': 'performance',
                                 'parameters': large_params
                             },
                             headers=auth_headers)
        
        assert response.status_code == 201
        data = json.loads(response.data)
        assert 'session_id' in data
    
    def test_unicode_parameters(self, client, auth_headers):
        """Test unicode parameters."""
        if not auth_headers:
            pytest.skip("Authentication not available")
        
        response = client.post('/api/v1/optimization/sessions',
                             json={
                                 'session_name': 'Unicode Test 测试',
                                 'optimization_type': 'performance',
                                 'parameters': {'unicode_param': '测试值', 'emoji': '🚀'}
                             },
                             headers=auth_headers)
        
        assert response.status_code == 201
        data = json.loads(response.data)
        assert 'session_id' in data









