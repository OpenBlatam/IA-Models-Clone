"""
Consensus Manager
================

Distributed consensus management.
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ConsensusAlgorithm(Enum):
    """Consensus algorithms."""
    RAFT = "raft"
    PAXOS = "paxos"
    PBFT = "pbft"


@dataclass
class ConsensusNode:
    """Consensus node."""
    id: str
    endpoint: str
    role: str = "follower"  # leader, follower, candidate
    term: int = 0
    voted_for: Optional[str] = None


class ConsensusManager:
    """Consensus manager."""
    
    def __init__(self, algorithm: ConsensusAlgorithm = ConsensusAlgorithm.RAFT):
        self.algorithm = algorithm
        self._nodes: Dict[str, ConsensusNode] = {}
        self._leader: Optional[str] = None
    
    def add_node(self, node_id: str, endpoint: str):
        """Add consensus node."""
        node = ConsensusNode(
            id=node_id,
            endpoint=endpoint
        )
        
        self._nodes[node_id] = node
        logger.info(f"Added consensus node: {node_id}")
    
    def elect_leader(self, node_id: str, term: int) -> bool:
        """Elect leader."""
        if node_id not in self._nodes:
            return False
        
        node = self._nodes[node_id]
        
        if term > node.term:
            node.term = term
            node.role = "leader"
            node.voted_for = node_id
            self._leader = node_id
            
            # Update other nodes
            for other_node in self._nodes.values():
                if other_node.id != node_id:
                    other_node.role = "follower"
                    other_node.term = term
            
            logger.info(f"Elected leader: {node_id} for term {term}")
            return True
        
        return False
    
    def get_leader(self) -> Optional[str]:
        """Get current leader."""
        return self._leader
    
    def get_consensus_stats(self) -> Dict[str, Any]:
        """Get consensus statistics."""
        return {
            "algorithm": self.algorithm.value,
            "total_nodes": len(self._nodes),
            "leader": self._leader,
            "by_role": {
                role: sum(1 for n in self._nodes.values() if n.role == role)
                for role in ["leader", "follower", "candidate"]
            }
        }















