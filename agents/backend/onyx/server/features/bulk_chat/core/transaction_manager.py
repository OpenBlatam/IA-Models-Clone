"""
Transaction Manager - Gestor de Transacciones
==============================================

Sistema de gestión de transacciones con soporte para transacciones distribuidas, rollback y compensación.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable, Awaitable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict, deque
import uuid

logger = logging.getLogger(__name__)


class TransactionStatus(Enum):
    """Estado de transacción."""
    PENDING = "pending"
    COMMITTED = "committed"
    ROLLED_BACK = "rolled_back"
    FAILED = "failed"
    COMMITTING = "committing"
    ROLLING_BACK = "rolling_back"


@dataclass
class TransactionOperation:
    """Operación de transacción."""
    operation_id: str
    execute: Callable
    rollback: Optional[Callable] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Transaction:
    """Transacción."""
    transaction_id: str
    status: TransactionStatus = TransactionStatus.PENDING
    operations: List[TransactionOperation] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    committed_at: Optional[datetime] = None
    rolled_back_at: Optional[datetime] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class TransactionManager:
    """Gestor de transacciones."""
    
    def __init__(self):
        self.transactions: Dict[str, Transaction] = {}
        self.transaction_history: deque = deque(maxlen=100000)
        self._lock = asyncio.Lock()
    
    def begin_transaction(
        self,
        transaction_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Iniciar transacción."""
        tx_id = transaction_id or f"tx_{uuid.uuid4().hex[:12]}"
        
        transaction = Transaction(
            transaction_id=tx_id,
            metadata=metadata or {},
        )
        
        async def save_transaction():
            async with self._lock:
                self.transactions[tx_id] = transaction
        
        asyncio.create_task(save_transaction())
        
        logger.info(f"Began transaction: {tx_id}")
        return tx_id
    
    def add_operation(
        self,
        transaction_id: str,
        operation_id: str,
        execute: Callable,
        rollback: Optional[Callable] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Agregar operación a transacción."""
        operation = TransactionOperation(
            operation_id=operation_id,
            execute=execute,
            rollback=rollback,
            metadata=metadata or {},
        )
        
        async def save_operation():
            async with self._lock:
                transaction = self.transactions.get(transaction_id)
                if not transaction:
                    raise ValueError(f"Transaction {transaction_id} not found")
                if transaction.status != TransactionStatus.PENDING:
                    raise ValueError(f"Transaction {transaction_id} is not in PENDING state")
                transaction.operations.append(operation)
        
        asyncio.create_task(save_operation())
        
        logger.debug(f"Added operation {operation_id} to transaction {transaction_id}")
        return operation_id
    
    async def commit(self, transaction_id: str) -> bool:
        """Commit transacción."""
        transaction = self.transactions.get(transaction_id)
        if not transaction:
            return False
        
        if transaction.status != TransactionStatus.PENDING:
            return False
        
        transaction.status = TransactionStatus.COMMITTING
        
        executed_operations = []
        
        try:
            # Ejecutar todas las operaciones
            for operation in transaction.operations:
                try:
                    if asyncio.iscoroutinefunction(operation.execute):
                        await operation.execute()
                    else:
                        operation.execute()
                    
                    executed_operations.append(operation)
                
                except Exception as e:
                    # Fallo en ejecución, hacer rollback de operaciones ejecutadas
                    error_msg = f"Operation {operation.operation_id} failed: {str(e)}"
                    logger.error(error_msg)
                    
                    # Rollback en orden inverso
                    for op in reversed(executed_operations):
                        if op.rollback:
                            try:
                                if asyncio.iscoroutinefunction(op.rollback):
                                    await op.rollback()
                                else:
                                    op.rollback()
                            except Exception as rb_error:
                                logger.error(f"Rollback failed for {op.operation_id}: {rb_error}")
                    
                    transaction.status = TransactionStatus.FAILED
                    transaction.error = error_msg
                    transaction.rolled_back_at = datetime.now()
                    
                    async with self._lock:
                        self.transaction_history.append(transaction)
                        if transaction_id in self.transactions:
                            del self.transactions[transaction_id]
                    
                    return False
            
            # Todas las operaciones exitosas
            transaction.status = TransactionStatus.COMMITTED
            transaction.committed_at = datetime.now()
            
            async with self._lock:
                self.transaction_history.append(transaction)
                if transaction_id in self.transactions:
                    del self.transactions[transaction_id]
            
            logger.info(f"Committed transaction: {transaction_id}")
            return True
        
        except Exception as e:
            error_msg = f"Transaction commit failed: {str(e)}"
            logger.error(error_msg)
            transaction.status = TransactionStatus.FAILED
            transaction.error = error_msg
            
            async with self._lock:
                self.transaction_history.append(transaction)
                if transaction_id in self.transactions:
                    del self.transactions[transaction_id]
            
            return False
    
    async def rollback(self, transaction_id: str) -> bool:
        """Rollback transacción."""
        transaction = self.transactions.get(transaction_id)
        if not transaction:
            return False
        
        if transaction.status not in [TransactionStatus.PENDING, TransactionStatus.COMMITTING]:
            return False
        
        transaction.status = TransactionStatus.ROLLING_BACK
        
        try:
            # Ejecutar rollbacks en orden inverso
            for operation in reversed(transaction.operations):
                if operation.rollback:
                    try:
                        if asyncio.iscoroutinefunction(operation.rollback):
                            await operation.rollback()
                        else:
                            operation.rollback()
                    except Exception as e:
                        logger.error(f"Rollback failed for {operation.operation_id}: {e}")
            
            transaction.status = TransactionStatus.ROLLED_BACK
            transaction.rolled_back_at = datetime.now()
            
            async with self._lock:
                self.transaction_history.append(transaction)
                if transaction_id in self.transactions:
                    del self.transactions[transaction_id]
            
            logger.info(f"Rolled back transaction: {transaction_id}")
            return True
        
        except Exception as e:
            error_msg = f"Transaction rollback failed: {str(e)}"
            logger.error(error_msg)
            transaction.status = TransactionStatus.FAILED
            transaction.error = error_msg
            
            async with self._lock:
                self.transaction_history.append(transaction)
                if transaction_id in self.transactions:
                    del self.transactions[transaction_id]
            
            return False
    
    def get_transaction(self, transaction_id: str) -> Optional[Dict[str, Any]]:
        """Obtener información de transacción."""
        transaction = self.transactions.get(transaction_id)
        if not transaction:
            # Buscar en historial
            for tx in self.transaction_history:
                if tx.transaction_id == transaction_id:
                    transaction = tx
                    break
        
        if not transaction:
            return None
        
        return {
            "transaction_id": transaction.transaction_id,
            "status": transaction.status.value,
            "operations_count": len(transaction.operations),
            "created_at": transaction.created_at.isoformat(),
            "committed_at": transaction.committed_at.isoformat() if transaction.committed_at else None,
            "rolled_back_at": transaction.rolled_back_at.isoformat() if transaction.rolled_back_at else None,
            "error": transaction.error,
        }
    
    def get_transaction_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Obtener historial de transacciones."""
        history = list(self.transaction_history)[-limit:]
        
        return [
            {
                "transaction_id": tx.transaction_id,
                "status": tx.status.value,
                "operations_count": len(tx.operations),
                "created_at": tx.created_at.isoformat(),
                "committed_at": tx.committed_at.isoformat() if tx.committed_at else None,
                "rolled_back_at": tx.rolled_back_at.isoformat() if tx.rolled_back_at else None,
            }
            for tx in history
        ]
    
    def get_transaction_manager_summary(self) -> Dict[str, Any]:
        """Obtener resumen del gestor."""
        by_status: Dict[str, int] = defaultdict(int)
        
        for transaction in self.transactions.values():
            by_status[transaction.status.value] += 1
        
        # También contar historial
        for transaction in self.transaction_history:
            by_status[transaction.status.value] += 1
        
        return {
            "active_transactions": len(self.transactions),
            "transactions_by_status": dict(by_status),
            "total_history": len(self.transaction_history),
        }


