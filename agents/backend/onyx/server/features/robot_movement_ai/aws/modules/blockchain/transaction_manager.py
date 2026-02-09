"""
Transaction Manager
===================

Blockchain transaction management.
"""

import logging
import hashlib
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class TransactionStatus(Enum):
    """Transaction status."""
    PENDING = "pending"
    CONFIRMED = "confirmed"
    FAILED = "failed"


@dataclass
class Transaction:
    """Blockchain transaction."""
    id: str
    from_address: str
    to_address: str
    amount: float
    data: Optional[Any] = None
    status: TransactionStatus = TransactionStatus.PENDING
    created_at: datetime = None
    confirmed_at: Optional[datetime] = None
    block_hash: Optional[str] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        
        if not self.id:
            self.id = self._generate_id()
    
    def _generate_id(self) -> str:
        """Generate transaction ID."""
        content = f"{self.from_address}{self.to_address}{self.amount}{self.created_at.isoformat()}"
        return hashlib.sha256(content.encode()).hexdigest()


class TransactionManager:
    """Blockchain transaction manager."""
    
    def __init__(self):
        self._transactions: Dict[str, Transaction] = {}
        self._pending_transactions: List[str] = []
    
    def create_transaction(
        self,
        from_address: str,
        to_address: str,
        amount: float,
        data: Optional[Any] = None
    ) -> Transaction:
        """Create transaction."""
        transaction = Transaction(
            id="",
            from_address=from_address,
            to_address=to_address,
            amount=amount,
            data=data
        )
        
        self._transactions[transaction.id] = transaction
        self._pending_transactions.append(transaction.id)
        
        logger.info(f"Created transaction: {transaction.id}")
        return transaction
    
    def confirm_transaction(self, transaction_id: str, block_hash: str):
        """Confirm transaction."""
        if transaction_id not in self._transactions:
            return False
        
        transaction = self._transactions[transaction_id]
        transaction.status = TransactionStatus.CONFIRMED
        transaction.confirmed_at = datetime.now()
        transaction.block_hash = block_hash
        
        if transaction_id in self._pending_transactions:
            self._pending_transactions.remove(transaction_id)
        
        logger.info(f"Confirmed transaction: {transaction_id}")
        return True
    
    def fail_transaction(self, transaction_id: str):
        """Mark transaction as failed."""
        if transaction_id not in self._transactions:
            return False
        
        transaction = self._transactions[transaction_id]
        transaction.status = TransactionStatus.FAILED
        
        if transaction_id in self._pending_transactions:
            self._pending_transactions.remove(transaction_id)
        
        logger.warning(f"Transaction failed: {transaction_id}")
        return True
    
    def get_transaction(self, transaction_id: str) -> Optional[Transaction]:
        """Get transaction by ID."""
        return self._transactions.get(transaction_id)
    
    def get_pending_transactions(self) -> List[Transaction]:
        """Get pending transactions."""
        return [
            self._transactions[tid]
            for tid in self._pending_transactions
            if tid in self._transactions
        ]
    
    def get_transaction_stats(self) -> Dict[str, Any]:
        """Get transaction statistics."""
        return {
            "total_transactions": len(self._transactions),
            "pending": len(self._pending_transactions),
            "confirmed": sum(
                1 for t in self._transactions.values()
                if t.status == TransactionStatus.CONFIRMED
            ),
            "failed": sum(
                1 for t in self._transactions.values()
                if t.status == TransactionStatus.FAILED
            )
        }















