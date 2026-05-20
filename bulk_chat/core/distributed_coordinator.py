"""
Distributed Coordinator - Coordinador Distribuido
==================================================

Sistema de coordinación distribuida con consenso, elección de líder y sincronización.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict, deque
import uuid

logger = logging.getLogger(__name__)


class NodeRole(Enum):
    """Rol de nodo."""
    LEADER = "leader"
    FOLLOWER = "follower"
    CANDIDATE = "candidate"


class ConsensusAlgorithm(Enum):
    """Algoritmo de consenso."""
    RAFT = "raft"
    PAXOS = "paxos"
    SIMPLE_MAJORITY = "simple_majority"


@dataclass
class Node:
    """Nodo."""
    node_id: str
    address: str
    role: NodeRole = NodeRole.FOLLOWER
    last_heartbeat: Optional[datetime] = None
    term: int = 0
    votes_received: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConsensusProposal:
    """Propuesta de consenso."""
    proposal_id: str
    value: Any
    proposer_id: str
    term: int
    votes: int = 0
    accepted: bool = False
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class DistributedCoordinator:
    """Coordinador distribuido."""
    
    def __init__(
        self,
        node_id: str,
        algorithm: ConsensusAlgorithm = ConsensusAlgorithm.SIMPLE_MAJORITY,
    ):
        self.node_id = node_id
        self.algorithm = algorithm
        self.nodes: Dict[str, Node] = {}
        self.current_term = 0
        self.leader_id: Optional[str] = None
        self.proposals: Dict[str, ConsensusProposal] = {}
        self.heartbeat_interval = 5.0
        self.election_timeout = 10.0
        self.last_heartbeat_received: Optional[datetime] = None
        self._lock = asyncio.Lock()
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._election_task: Optional[asyncio.Task] = None
    
    def register_node(
        self,
        node_id: str,
        address: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Registrar nodo."""
        node = Node(
            node_id=node_id,
            address=address,
            metadata=metadata or {},
        )
        
        async def save_node():
            async with self._lock:
                self.nodes[node_id] = node
        
        asyncio.create_task(save_node())
        
        logger.info(f"Registered node: {node_id} at {address}")
        return node_id
    
    def start_coordination(self):
        """Iniciar coordinación."""
        # Iniciar heartbeat si es líder
        if self.leader_id == self.node_id:
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        
        # Iniciar election timeout
        self._election_task = asyncio.create_task(self._election_timeout_loop())
        
        logger.info("Distributed coordination started")
    
    async def _heartbeat_loop(self):
        """Loop de heartbeat."""
        while True:
            try:
                if self.leader_id == self.node_id:
                    await self._send_heartbeat()
                await asyncio.sleep(self.heartbeat_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in heartbeat loop: {e}")
                await asyncio.sleep(self.heartbeat_interval)
    
    async def _send_heartbeat(self):
        """Enviar heartbeat a followers."""
        self.current_term += 1
        
        async with self._lock:
            now = datetime.now()
            for node_id, node in self.nodes.items():
                if node_id != self.node_id:
                    node.last_heartbeat = now
        
        # En producción, aquí se enviarían heartbeats reales a los nodos
        logger.debug(f"Sent heartbeat to followers (term: {self.current_term})")
    
    async def _election_timeout_loop(self):
        """Loop de timeout de elección."""
        while True:
            try:
                await asyncio.sleep(self.election_timeout)
                
                # Verificar si se necesita elección
                if self.leader_id != self.node_id:
                    if not self.last_heartbeat_received or \
                       (datetime.now() - self.last_heartbeat_received).total_seconds() > self.election_timeout:
                        await self._start_election()
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in election timeout loop: {e}")
    
    async def _start_election(self):
        """Iniciar elección de líder."""
        async with self._lock:
            self.current_term += 1
            my_node = self.nodes.get(self.node_id)
            if my_node:
                my_node.role = NodeRole.CANDIDATE
                my_node.term = self.current_term
                my_node.votes_received = 1  # Voto por sí mismo
        
        # Solicitar votos
        votes = 1  # Voto propio
        
        for node_id, node in self.nodes.items():
            if node_id != self.node_id:
                # Simular voto (en producción, se enviaría petición real)
                if node.term < self.current_term:
                    votes += 1
        
        # Verificar si se obtuvo mayoría
        total_nodes = len(self.nodes)
        if votes > total_nodes / 2:
            async with self._lock:
                self.leader_id = self.node_id
                my_node = self.nodes.get(self.node_id)
                if my_node:
                    my_node.role = NodeRole.LEADER
                    my_node.term = self.current_term
                
                # Actualizar otros nodos
                for node_id, node in self.nodes.items():
                    if node_id != self.node_id:
                        node.role = NodeRole.FOLLOWER
                        node.term = self.current_term
            
            logger.info(f"Elected as leader (term: {self.current_term}, votes: {votes}/{total_nodes})")
            
            # Iniciar heartbeat
            if not self._heartbeat_task or self._heartbeat_task.done():
                self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        else:
            logger.debug(f"Election failed (term: {self.current_term}, votes: {votes}/{total_nodes})")
    
    async def propose_value(
        self,
        value: Any,
        proposal_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Proponer valor para consenso."""
        if self.leader_id != self.node_id:
            raise ValueError("Only leader can propose values")
        
        prop_id = proposal_id or f"prop_{uuid.uuid4().hex[:12]}"
        
        proposal = ConsensusProposal(
            proposal_id=prop_id,
            value=value,
            proposer_id=self.node_id,
            term=self.current_term,
            metadata=metadata or {},
        )
        
        async with self._lock:
            self.proposals[prop_id] = proposal
        
        # Solicitar aceptación (simplificado)
        votes = 1  # Voto propio
        
        for node_id in self.nodes.keys():
            if node_id != self.node_id:
                votes += 1
        
        # Verificar mayoría
        total_nodes = len(self.nodes)
        if votes > total_nodes / 2:
            async with self._lock:
                proposal.accepted = True
                proposal.votes = votes
        
        logger.info(f"Proposed value: {prop_id} (accepted: {proposal.accepted})")
        return prop_id
    
    def get_leader(self) -> Optional[Dict[str, Any]]:
        """Obtener información del líder."""
        if not self.leader_id:
            return None
        
        leader = self.nodes.get(self.leader_id)
        if not leader:
            return None
        
        return {
            "node_id": leader.node_id,
            "address": leader.address,
            "term": leader.term,
            "role": leader.role.value,
        }
    
    def get_coordination_status(self) -> Dict[str, Any]:
        """Obtener estado de coordinación."""
        by_role: Dict[str, int] = defaultdict(int)
        
        for node in self.nodes.values():
            by_role[node.role.value] += 1
        
        return {
            "node_id": self.node_id,
            "current_term": self.current_term,
            "leader_id": self.leader_id,
            "total_nodes": len(self.nodes),
            "nodes_by_role": dict(by_role),
            "total_proposals": len(self.proposals),
            "accepted_proposals": len([p for p in self.proposals.values() if p.accepted]),
        }
    
    def get_distributed_coordinator_summary(self) -> Dict[str, Any]:
        """Obtener resumen del coordinador."""
        return {
            "node_id": self.node_id,
            "leader_id": self.leader_id,
            "current_term": self.current_term,
            "total_nodes": len(self.nodes),
            "total_proposals": len(self.proposals),
        }


