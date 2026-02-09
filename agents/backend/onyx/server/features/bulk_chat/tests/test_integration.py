"""
Integration Tests
=================
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch


@pytest.mark.integration
@pytest.mark.asyncio
async def test_full_session_flow():
    """Test complete session flow."""
    from ..api.chat_api import create_chat_app
    
    app = create_chat_app()
    client = TestClient(app)
    
    # Create session
    response = client.post(
        "/api/v1/chat/sessions",
        json={"user_id": "test_user", "initial_message": "Hello"}
    )
    assert response.status_code == 200
    session_id = response.json()["session_id"]
    
    # Get session
    response = client.get(f"/api/v1/chat/sessions/{session_id}")
    assert response.status_code == 200
    
    # Add message
    response = client.post(
        f"/api/v1/chat/sessions/{session_id}/messages",
        json={"content": "Test message"}
    )
    assert response.status_code == 200
    
    # Pause
    response = client.post(
        f"/api/v1/chat/sessions/{session_id}/pause",
        json={"reason": "Test pause"}
    )
    assert response.status_code == 200
    
    # Resume
    response = client.post(f"/api/v1/chat/sessions/{session_id}/resume")
    assert response.status_code == 200
    
    # Stop
    response = client.post(f"/api/v1/chat/sessions/{session_id}/stop")
    assert response.status_code == 200


@pytest.mark.integration
@pytest.mark.asyncio
async def test_transaction_flow():
    """Test transaction flow."""
    from ..core.transaction_manager import TransactionManager
    
    manager = TransactionManager()
    
    # Start transaction
    transaction_id = await manager.start_transaction(
        operations=[{"type": "test", "data": "test"}]
    )
    assert transaction_id is not None
    
    # Get status
    status = manager.get_transaction_status(transaction_id)
    assert status is not None
    
    # Rollback
    result = await manager.rollback_transaction(transaction_id)
    assert result is True


@pytest.mark.integration
@pytest.mark.asyncio
async def test_saga_flow():
    """Test saga flow."""
    from ..core.saga_orchestrator import SagaOrchestrator
    
    orchestrator = SagaOrchestrator()
    
    # Create saga
    saga_id = orchestrator.create_saga(
        name="test_saga",
        steps=[{"name": "step1", "action": "test", "compensation": "rollback"}]
    )
    assert saga_id is not None
    
    # Get status
    status = orchestrator.get_saga_status(saga_id)
    assert status is not None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_service_mesh_flow():
    """Test service mesh flow."""
    from ..core.service_mesh import ServiceMesh, ServiceStatus
    
    mesh = ServiceMesh()
    
    # Register service
    service_name = mesh.register_service("test_service")
    assert service_name == "test_service"
    
    # Register instance
    instance_id = mesh.register_instance(
        "instance_1", "test_service", "127.0.0.1", 8000
    )
    assert instance_id == "instance_1"
    
    # Update status
    mesh.update_instance_status("instance_1", ServiceStatus.HEALTHY)
    
    # Get instance
    instance = mesh.get_instance("test_service")
    assert instance is not None


