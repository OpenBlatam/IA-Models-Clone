"""
Tests for Transaction Manager
=============================
"""

import pytest
import asyncio
from ..core.transaction_manager import TransactionManager, TransactionStatus


@pytest.fixture
def transaction_manager():
    """Create transaction manager for testing."""
    return TransactionManager()


@pytest.mark.asyncio
async def test_start_transaction(transaction_manager):
    """Test starting a transaction."""
    transaction_id = await transaction_manager.start_transaction(
        operations=[
            {"type": "create", "resource": "session", "data": {"user_id": "test"}},
            {"type": "update", "resource": "session", "data": {"status": "active"}},
        ]
    )
    
    assert transaction_id is not None
    assert transaction_id in transaction_manager.transactions


@pytest.mark.asyncio
async def test_commit_transaction(transaction_manager):
    """Test committing a transaction."""
    from unittest.mock import patch
    
    transaction_id = await transaction_manager.start_transaction(
        operations=[{"type": "create", "resource": "test"}]
    )
    
    # Mock operations
    with patch.object(transaction_manager, '_execute_operations', return_value=True):
        result = await transaction_manager.commit_transaction(transaction_id)
        
        assert result is True
        transaction = transaction_manager.transactions[transaction_id]
        assert transaction.status == TransactionStatus.COMMITTED


@pytest.mark.asyncio
async def test_rollback_transaction(transaction_manager):
    """Test rolling back a transaction."""
    transaction_id = await transaction_manager.start_transaction(
        operations=[{"type": "create", "resource": "test"}]
    )
    
    result = await transaction_manager.rollback_transaction(transaction_id)
    
    assert result is True
    transaction = transaction_manager.transactions[transaction_id]
    assert transaction.status == TransactionStatus.ROLLED_BACK


@pytest.mark.asyncio
async def test_get_transaction_status(transaction_manager):
    """Test getting transaction status."""
    transaction_id = await transaction_manager.start_transaction(
        operations=[{"type": "test"}]
    )
    
    status = transaction_manager.get_transaction_status(transaction_id)
    
    assert status == TransactionStatus.PENDING


@pytest.mark.asyncio
async def test_transaction_not_found(transaction_manager):
    """Test error handling for non-existent transaction."""
    with pytest.raises(ValueError):
        await transaction_manager.commit_transaction("non_existent")


@pytest.mark.asyncio
async def test_get_transaction_summary(transaction_manager):
    """Test getting transaction summary."""
    await transaction_manager.start_transaction(
        operations=[{"type": "test"}]
    )
    
    summary = transaction_manager.get_transaction_manager_summary()
    
    assert summary["total_transactions"] >= 1
    assert "pending" in summary["transactions_by_status"]

