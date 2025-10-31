"""
Test Suite
==========

Ultra-advanced test suite with pytest and Flask testing.
"""

import pytest
import json
import os
from typing import Dict, Any, Optional
from flask import Flask
from app import create_app, db
from models import User, OptimizationSession, PerformanceMetric
from utils.exceptions import AuthenticationError, ValidationError, OptimizationError

# Test configuration
TEST_CONFIG = {
    'TESTING': True,
    'SQLALCHEMY_DATABASE_URI': 'sqlite:///:memory:',
    'JWT_SECRET_KEY': 'test-secret-key',
    'WTF_CSRF_ENABLED': False,
    'LOGIN_DISABLED': False
}

@pytest.fixture
def app() -> Flask:
    """Create test application."""
    app = create_app('testing')
    app.config.update(TEST_CONFIG)
    
    with app.app_context():
        db.create_all()
        yield app
        db.drop_all()

@pytest.fixture
def client(app: Flask):
    """Create test client."""
    return app.test_client()

@pytest.fixture
def auth_headers(client) -> Dict[str, str]:
    """Get authentication headers."""
    # Create test user
    user = User(
        username='testuser',
        email='test@example.com',
        password_hash='hashed_password'
    )
    db.session.add(user)
    db.session.commit()
    
    # Login to get token
    response = client.post('/api/v1/auth/login', json={
        'username': 'testuser',
        'password': 'testpassword'
    })
    
    if response.status_code == 200:
        data = json.loads(response.data)
        return {'Authorization': f"Bearer {data['access_token']}"}
    
    return {}

@pytest.fixture
def test_user() -> User:
    """Create test user."""
    user = User(
        username='testuser',
        email='test@example.com',
        password_hash='hashed_password'
    )
    db.session.add(user)
    db.session.commit()
    return user

@pytest.fixture
def test_session(test_user: User) -> OptimizationSession:
    """Create test optimization session."""
    session = OptimizationSession(
        user_id=test_user.id,
        session_name='Test Session',
        optimization_type='performance',
        parameters={'test': 'value'}
    )
    db.session.add(session)
    db.session.commit()
    return session

@pytest.fixture
def test_metric(test_session: OptimizationSession) -> PerformanceMetric:
    """Create test performance metric."""
    metric = PerformanceMetric(
        session_id=test_session.id,
        metric_name='test_metric',
        metric_value=1.0,
        metric_unit='unit'
    )
    db.session.add(metric)
    db.session.commit()
    return metric









