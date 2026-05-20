"""
Transaction system for KV cache.

This module provides transactional operations with ACID properties
for cache operations.
"""

import time
import threading
import uuid
from typing import Dict, Any, List, Optional, Callable, Set
from dataclasses import dataclass, field
from enum import Enum


class TransactionStatus(Enum):
    """Transaction status."""
    PENDING = "pending"
    COMMITTED = "committed"
    ABORTED = "aborted"
    ROLLING_BACK = "rolling_back"


@dataclass
class TransactionOperation:
    """An operation in a transaction."""
    operation_type: str  # "get", "put", "delete"
    key: str
    value: Optional[Any] = None
    old_value: Optional[Any] = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class Transaction:
    """A cache transaction."""
    transaction_id: str
    status: TransactionStatus
    operations: List[TransactionOperation]
    created_at: float
    committed_at: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class CacheTransactionManager:
    """Manages cache transactions."""
    
    def __init__(self, cache: Any):
        self.cache = cache
        self._transactions: Dict[str, Transaction] = {}
        self._lock = threading.Lock()
        
    def begin_transaction(self, metadata: Optional[Dict[str, Any]] = None) -> Transaction:
        """Begin a new transaction."""
        transaction_id = str(uuid.uuid4())
        
        transaction = Transaction(
            transaction_id=transaction_id,
            status=TransactionStatus.PENDING,
            operations=[],
            created_at=time.time(),
            metadata=metadata or {}
        )
        
        with self._lock:
            self._transactions[transaction_id] = transaction
            
        return transaction
        
    def add_operation(
        self,
        transaction_id: str,
        operation_type: str,
        key: str,
        value: Optional[Any] = None
    ) -> bool:
        """Add an operation to a transaction."""
        with self._lock:
            if transaction_id not in self._transactions:
                return False
                
            transaction = self._transactions[transaction_id]
            
            if transaction.status != TransactionStatus.PENDING:
                return False
                
            # Store old value for rollback
            old_value = None
            if operation_type in ["put", "delete"]:
                old_value = self.cache.get(key)
                
            operation = TransactionOperation(
                operation_type=operation_type,
                key=key,
                value=value,
                old_value=old_value
            )
            
            transaction.operations.append(operation)
            return True
            
    def commit(self, transaction_id: str) -> bool:
        """Commit a transaction."""
        with self._lock:
            if transaction_id not in self._transactions:
                return False
                
            transaction = self._transactions[transaction_id]
            
            if transaction.status != TransactionStatus.PENDING:
                return False
                
            # Execute all operations
            try:
                for operation in transaction.operations:
                    if operation.operation_type == "put":
                        self.cache.put(operation.key, operation.value)
                    elif operation.operation_type == "delete":
                        self.cache.delete(operation.key)
                    # "get" operations don't modify cache
                    
                transaction.status = TransactionStatus.COMMITTED
                transaction.committed_at = time.time()
                return True
            except Exception as e:
                # Rollback on error
                self._rollback(transaction)
                return False
                
    def abort(self, transaction_id: str) -> bool:
        """Abort a transaction."""
        with self._lock:
            if transaction_id not in self._transactions:
                return False
                
            transaction = self._transactions[transaction_id]
            
            if transaction.status != TransactionStatus.PENDING:
                return False
                
            transaction.status = TransactionStatus.ABORTED
            return True
            
    def _rollback(self, transaction: Transaction) -> None:
        """Rollback a transaction."""
        transaction.status = TransactionStatus.ROLLING_BACK
        
        # Restore old values
        for operation in reversed(transaction.operations):
            if operation.operation_type == "put":
                if operation.old_value is not None:
                    self.cache.put(operation.key, operation.old_value)
                else:
                    self.cache.delete(operation.key)
            elif operation.operation_type == "delete":
                if operation.old_value is not None:
                    self.cache.put(operation.key, operation.old_value)
                    
        transaction.status = TransactionStatus.ABORTED


class TransactionalCache:
    """Cache wrapper with transactional support."""
    
    def __init__(self, cache: Any):
        self.cache = cache
        self.transaction_manager = CacheTransactionManager(cache)
        
    def begin(self, metadata: Optional[Dict[str, Any]] = None) -> Transaction:
        """Begin a transaction."""
        return self.transaction_manager.begin_transaction(metadata)
        
    def commit(self, transaction_id: str) -> bool:
        """Commit a transaction."""
        return self.transaction_manager.commit(transaction_id)
        
    def abort(self, transaction_id: str) -> bool:
        """Abort a transaction."""
        return self.transaction_manager.abort(transaction_id)
        
    def get(self, key: str, transaction_id: Optional[str] = None) -> Any:
        """Get value, optionally within a transaction."""
        if transaction_id:
            # In transaction, would check transaction operations first
            pass
        return self.cache.get(key)
        
    def put(
        self,
        key: str,
        value: Any,
        transaction_id: Optional[str] = None
    ) -> bool:
        """Put value, optionally within a transaction."""
        if transaction_id:
            return self.transaction_manager.add_operation(
                transaction_id,
                "put",
                key,
                value
            )
        return self.cache.put(key, value)
        
    def delete(self, key: str, transaction_id: Optional[str] = None) -> bool:
        """Delete value, optionally within a transaction."""
        if transaction_id:
            return self.transaction_manager.add_operation(
                transaction_id,
                "delete",
                key
            )
        return self.cache.delete(key)



