"""
Distributed Systems
===================

Distributed systems modules.
"""

from aws.modules.distributed.lock_manager import LockManager, Lock
from aws.modules.distributed.service_discovery import ServiceDiscovery, ServiceInstance
from aws.modules.distributed.consensus import ConsensusManager, ConsensusNode, ConsensusAlgorithm

__all__ = [
    "LockManager",
    "Lock",
    "ServiceDiscovery",
    "ServiceInstance",
    "ConsensusManager",
    "ConsensusNode",
    "ConsensusAlgorithm",
]

