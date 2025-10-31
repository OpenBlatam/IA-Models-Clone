"""
Ultimate BUL System - Temporal Document Management & Time Travel
Advanced temporal document management with time travel capabilities for document versioning and timeline manipulation
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import aiohttp
import redis
from prometheus_client import Counter, Histogram, Gauge
import time
import uuid
import numpy as np

logger = logging.getLogger(__name__)

class TemporalOperation(str, Enum):
    """Temporal operations"""
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    RESTORE = "restore"
    BRANCH = "branch"
    MERGE = "merge"
    REVERT = "revert"
    FORWARD = "forward"

class TimelineType(str, Enum):
    """Timeline types"""
    LINEAR = "linear"
    BRANCHED = "branched"
    PARALLEL = "parallel"
    CYCLICAL = "cyclical"
    QUANTUM = "quantum"
    MULTIVERSE = "multiverse"

class TemporalStatus(str, Enum):
    """Temporal status"""
    STABLE = "stable"
    FLUX = "flux"
    PARADOX = "paradox"
    CONVERGED = "converged"
    DIVERGED = "diverged"
    LOCKED = "locked"
    UNSTABLE = "unstable"

@dataclass
class TemporalDocument:
    """Temporal document with time travel capabilities"""
    id: str
    content: str
    version: int
    timeline_id: str
    branch_id: str
    created_at: datetime
    modified_at: datetime
    temporal_coordinates: Tuple[float, float, float]  # (time, space, dimension)
    causality_chain: List[str] = field(default_factory=list)
    paradox_risk: float = 0.0
    stability_score: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Timeline:
    """Document timeline"""
    id: str
    name: str
    timeline_type: TimelineType
    status: TemporalStatus
    created_at: datetime
    last_accessed: datetime
    document_count: int = 0
    branch_count: int = 0
    paradox_count: int = 0
    stability_score: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TemporalEvent:
    """Temporal event in document history"""
    id: str
    document_id: str
    timeline_id: str
    operation: TemporalOperation
    timestamp: datetime
    temporal_coordinates: Tuple[float, float, float]
    causality_impact: float
    paradox_risk: float
    description: str
    metadata: Dict[str, Any] = field(default_factory=dict)

class TemporalDocumentManagement:
    """Temporal document management with time travel capabilities"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.temporal_documents = {}
        self.timelines = {}
        self.temporal_events = {}
        self.causality_chains = {}
        self.paradox_detector = {}
        self.temporal_locks = {}
        
        # Redis for temporal data caching
        self.redis_client = redis.Redis(
            host=config.get("redis_host", "localhost"),
            port=config.get("redis_port", 6379),
            db=config.get("redis_db", 8)
        )
        
        # Prometheus metrics
        self.prometheus_metrics = self._initialize_prometheus_metrics()
        
        # Monitoring active
        self.monitoring_active = False
        
        # Start monitoring
        self.start_monitoring()
    
    def _initialize_prometheus_metrics(self) -> Dict[str, Any]:
        """Initialize Prometheus metrics"""
        return {
            "temporal_operations": Counter(
                "bul_temporal_operations_total",
                "Total temporal operations",
                ["operation", "timeline_type", "status"]
            ),
            "temporal_operation_duration": Histogram(
                "bul_temporal_operation_duration_seconds",
                "Temporal operation duration in seconds",
                ["operation", "timeline_type"]
            ),
            "timeline_stability": Gauge(
                "bul_timeline_stability",
                "Timeline stability score",
                ["timeline_id", "timeline_type"]
            ),
            "paradox_risk": Gauge(
                "bul_paradox_risk",
                "Paradox risk level",
                ["document_id", "timeline_id"]
            ),
            "temporal_documents": Gauge(
                "bul_temporal_documents",
                "Number of temporal documents",
                ["timeline_id", "status"]
            ),
            "causality_impact": Gauge(
                "bul_causality_impact",
                "Causality impact score",
                ["document_id", "operation"]
            ),
            "time_travel_events": Counter(
                "bul_time_travel_events_total",
                "Total time travel events",
                ["operation", "direction"]
            ),
            "temporal_flux": Gauge(
                "bul_temporal_flux",
                "Temporal flux level",
                ["timeline_id"]
            )
        }
    
    async def start_monitoring(self):
        """Start temporal monitoring"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        logger.info("Starting temporal monitoring")
        
        # Start monitoring tasks
        asyncio.create_task(self._monitor_temporal_stability())
        asyncio.create_task(self._detect_paradoxes())
        asyncio.create_task(self._update_metrics())
    
    async def stop_monitoring(self):
        """Stop temporal monitoring"""
        self.monitoring_active = False
        logger.info("Stopping temporal monitoring")
    
    async def _monitor_temporal_stability(self):
        """Monitor temporal stability"""
        while self.monitoring_active:
            try:
                current_time = datetime.utcnow()
                
                for timeline_id, timeline in self.timelines.items():
                    # Calculate timeline stability
                    stability = await self._calculate_timeline_stability(timeline_id)
                    timeline.stability_score = stability
                    
                    # Update timeline status based on stability
                    if stability < 0.3:
                        timeline.status = TemporalStatus.UNSTABLE
                    elif stability < 0.6:
                        timeline.status = TemporalStatus.FLUX
                    elif stability > 0.9:
                        timeline.status = TemporalStatus.STABLE
                    else:
                        timeline.status = TemporalStatus.CONVERGED
                    
                    # Update metrics
                    self.prometheus_metrics["timeline_stability"].labels(
                        timeline_id=timeline_id,
                        timeline_type=timeline.timeline_type.value
                    ).set(stability)
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error monitoring temporal stability: {e}")
                await asyncio.sleep(60)
    
    async def _detect_paradoxes(self):
        """Detect temporal paradoxes"""
        while self.monitoring_active:
            try:
                for document_id, document in self.temporal_documents.items():
                    # Calculate paradox risk
                    paradox_risk = await self._calculate_paradox_risk(document_id)
                    document.paradox_risk = paradox_risk
                    
                    # Update metrics
                    self.prometheus_metrics["paradox_risk"].labels(
                        document_id=document_id,
                        timeline_id=document.timeline_id
                    ).set(paradox_risk)
                    
                    # Detect paradoxes
                    if paradox_risk > 0.8:
                        await self._handle_paradox(document_id, paradox_risk)
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Error detecting paradoxes: {e}")
                await asyncio.sleep(30)
    
    async def _update_metrics(self):
        """Update Prometheus metrics"""
        while self.monitoring_active:
            try:
                # Update document counts
                for timeline_id, timeline in self.timelines.items():
                    doc_count = len([d for d in self.temporal_documents.values() if d.timeline_id == timeline_id])
                    self.prometheus_metrics["temporal_documents"].labels(
                        timeline_id=timeline_id,
                        status=timeline.status.value
                    ).set(doc_count)
                
                # Update temporal flux
                for timeline_id, timeline in self.timelines.items():
                    flux_level = 1.0 - timeline.stability_score
                    self.prometheus_metrics["temporal_flux"].labels(
                        timeline_id=timeline_id
                    ).set(flux_level)
                
                await asyncio.sleep(60)  # Update every minute
                
            except Exception as e:
                logger.error(f"Error updating metrics: {e}")
                await asyncio.sleep(60)
    
    async def _calculate_timeline_stability(self, timeline_id: str) -> float:
        """Calculate timeline stability score"""
        try:
            timeline = self.timelines.get(timeline_id)
            if not timeline:
                return 0.0
            
            # Get documents in timeline
            documents = [d for d in self.temporal_documents.values() if d.timeline_id == timeline_id]
            
            if not documents:
                return 1.0
            
            # Calculate stability based on various factors
            stability_factors = []
            
            # Factor 1: Document consistency
            consistency_score = self._calculate_document_consistency(documents)
            stability_factors.append(consistency_score)
            
            # Factor 2: Causality chain integrity
            causality_score = self._calculate_causality_integrity(timeline_id)
            stability_factors.append(causality_score)
            
            # Factor 3: Paradox risk
            paradox_score = 1.0 - (timeline.paradox_count / max(timeline.document_count, 1))
            stability_factors.append(paradox_score)
            
            # Factor 4: Temporal flux
            flux_score = self._calculate_temporal_flux(timeline_id)
            stability_factors.append(flux_score)
            
            # Calculate weighted average
            weights = [0.3, 0.3, 0.2, 0.2]
            stability = sum(factor * weight for factor, weight in zip(stability_factors, weights))
            
            return max(0.0, min(1.0, stability))
            
        except Exception as e:
            logger.error(f"Error calculating timeline stability: {e}")
            return 0.5
    
    def _calculate_document_consistency(self, documents: List[TemporalDocument]) -> float:
        """Calculate document consistency score"""
        if not documents:
            return 1.0
        
        # Check for content consistency across versions
        consistency_scores = []
        for i in range(len(documents) - 1):
            doc1 = documents[i]
            doc2 = documents[i + 1]
            
            # Calculate content similarity
            similarity = self._calculate_content_similarity(doc1.content, doc2.content)
            consistency_scores.append(similarity)
        
        return np.mean(consistency_scores) if consistency_scores else 1.0
    
    def _calculate_content_similarity(self, content1: str, content2: str) -> float:
        """Calculate content similarity between two documents"""
        # Simple similarity calculation based on common words
        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())
        
        if not words1 and not words2:
            return 1.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_causality_integrity(self, timeline_id: str) -> float:
        """Calculate causality chain integrity"""
        try:
            # Get events for timeline
            events = [e for e in self.temporal_events.values() if e.timeline_id == timeline_id]
            
            if not events:
                return 1.0
            
            # Check for causality violations
            violations = 0
            for event in events:
                if event.causality_impact < 0:
                    violations += 1
            
            integrity_score = 1.0 - (violations / len(events))
            return max(0.0, min(1.0, integrity_score))
            
        except Exception as e:
            logger.error(f"Error calculating causality integrity: {e}")
            return 0.5
    
    def _calculate_temporal_flux(self, timeline_id: str) -> float:
        """Calculate temporal flux level"""
        try:
            # Get recent events
            recent_events = [
                e for e in self.temporal_events.values()
                if e.timeline_id == timeline_id and
                (datetime.utcnow() - e.timestamp).total_seconds() < 3600  # Last hour
            ]
            
            if not recent_events:
                return 1.0
            
            # Calculate flux based on event frequency and impact
            flux_score = 1.0
            for event in recent_events:
                flux_score -= abs(event.causality_impact) * 0.1
            
            return max(0.0, min(1.0, flux_score))
            
        except Exception as e:
            logger.error(f"Error calculating temporal flux: {e}")
            return 0.5
    
    async def _calculate_paradox_risk(self, document_id: str) -> float:
        """Calculate paradox risk for document"""
        try:
            document = self.temporal_documents.get(document_id)
            if not document:
                return 0.0
            
            # Get document events
            events = [e for e in self.temporal_events.values() if e.document_id == document_id]
            
            if not events:
                return 0.0
            
            # Calculate paradox risk based on various factors
            risk_factors = []
            
            # Factor 1: Causality violations
            causality_violations = len([e for e in events if e.causality_impact < 0])
            causality_risk = causality_violations / len(events)
            risk_factors.append(causality_risk)
            
            # Factor 2: Temporal loops
            loop_risk = self._detect_temporal_loops(document_id)
            risk_factors.append(loop_risk)
            
            # Factor 3: Content contradictions
            contradiction_risk = self._detect_content_contradictions(document_id)
            risk_factors.append(contradiction_risk)
            
            # Factor 4: Timeline divergence
            divergence_risk = self._detect_timeline_divergence(document.timeline_id)
            risk_factors.append(divergence_risk)
            
            # Calculate weighted average
            weights = [0.3, 0.3, 0.2, 0.2]
            total_risk = sum(factor * weight for factor, weight in zip(risk_factors, weights))
            
            return max(0.0, min(1.0, total_risk))
            
        except Exception as e:
            logger.error(f"Error calculating paradox risk: {e}")
            return 0.0
    
    def _detect_temporal_loops(self, document_id: str) -> float:
        """Detect temporal loops in document history"""
        # Simulate temporal loop detection
        return np.random.uniform(0.0, 0.3)
    
    def _detect_content_contradictions(self, document_id: str) -> float:
        """Detect content contradictions in document versions"""
        # Simulate contradiction detection
        return np.random.uniform(0.0, 0.2)
    
    def _detect_timeline_divergence(self, timeline_id: str) -> float:
        """Detect timeline divergence"""
        timeline = self.timelines.get(timeline_id)
        if not timeline:
            return 0.0
        
        # Simulate divergence detection based on branch count
        divergence_risk = min(timeline.branch_count / 10.0, 1.0)
        return divergence_risk
    
    async def _handle_paradox(self, document_id: str, paradox_risk: float):
        """Handle temporal paradox"""
        try:
            logger.warning(f"Paradox detected in document {document_id} with risk {paradox_risk}")
            
            # Implement paradox resolution strategies
            if paradox_risk > 0.9:
                # Critical paradox - lock document
                await self._lock_temporal_document(document_id, "critical_paradox")
            elif paradox_risk > 0.7:
                # High risk - attempt auto-resolution
                await self._attempt_paradox_resolution(document_id)
            else:
                # Medium risk - monitor closely
                await self._monitor_paradox(document_id)
            
        except Exception as e:
            logger.error(f"Error handling paradox: {e}")
    
    async def _lock_temporal_document(self, document_id: str, reason: str):
        """Lock temporal document to prevent further modifications"""
        self.temporal_locks[document_id] = {
            "locked_at": datetime.utcnow(),
            "reason": reason,
            "locked_by": "paradox_detector"
        }
        
        logger.info(f"Locked temporal document {document_id} due to {reason}")
    
    async def _attempt_paradox_resolution(self, document_id: str):
        """Attempt to resolve paradox automatically"""
        # Simulate paradox resolution
        await asyncio.sleep(1)
        logger.info(f"Attempted paradox resolution for document {document_id}")
    
    async def _monitor_paradox(self, document_id: str):
        """Monitor paradox without intervention"""
        logger.info(f"Monitoring paradox in document {document_id}")
    
    def create_timeline(self, name: str, timeline_type: TimelineType = TimelineType.LINEAR) -> str:
        """Create new timeline"""
        try:
            timeline_id = f"timeline_{uuid.uuid4().hex[:8]}"
            
            timeline = Timeline(
                id=timeline_id,
                name=name,
                timeline_type=timeline_type,
                status=TemporalStatus.STABLE,
                created_at=datetime.utcnow(),
                last_accessed=datetime.utcnow()
            )
            
            self.timelines[timeline_id] = timeline
            
            logger.info(f"Created timeline: {timeline_id}")
            return timeline_id
            
        except Exception as e:
            logger.error(f"Error creating timeline: {e}")
            raise
    
    async def create_temporal_document(self, content: str, timeline_id: str,
                                     branch_id: str = "main") -> str:
        """Create temporal document"""
        try:
            document_id = f"temporal_doc_{uuid.uuid4().hex[:8]}"
            
            # Get timeline
            timeline = self.timelines.get(timeline_id)
            if not timeline:
                raise ValueError(f"Timeline {timeline_id} not found")
            
            # Create document
            document = TemporalDocument(
                id=document_id,
                content=content,
                version=1,
                timeline_id=timeline_id,
                branch_id=branch_id,
                created_at=datetime.utcnow(),
                modified_at=datetime.utcnow(),
                temporal_coordinates=(time.time(), 0.0, 0.0)
            )
            
            self.temporal_documents[document_id] = document
            
            # Create temporal event
            await self._create_temporal_event(
                document_id=document_id,
                timeline_id=timeline_id,
                operation=TemporalOperation.CREATE,
                description="Document created"
            )
            
            # Update timeline
            timeline.document_count += 1
            timeline.last_accessed = datetime.utcnow()
            
            # Update metrics
            self.prometheus_metrics["temporal_operations"].labels(
                operation="create",
                timeline_type=timeline.timeline_type.value,
                status=timeline.status.value
            ).inc()
            
            logger.info(f"Created temporal document: {document_id}")
            return document_id
            
        except Exception as e:
            logger.error(f"Error creating temporal document: {e}")
            raise
    
    async def time_travel_to_document(self, document_id: str, target_time: datetime,
                                    operation: TemporalOperation = TemporalOperation.REVERT) -> str:
        """Time travel to specific document version"""
        try:
            start_time = time.time()
            
            # Get document
            document = self.temporal_documents.get(document_id)
            if not document:
                raise ValueError(f"Document {document_id} not found")
            
            # Check if document is locked
            if document_id in self.temporal_locks:
                raise ValueError(f"Document {document_id} is locked")
            
            # Create new version at target time
            new_document_id = f"temporal_doc_{uuid.uuid4().hex[:8]}"
            
            new_document = TemporalDocument(
                id=new_document_id,
                content=document.content,
                version=document.version + 1,
                timeline_id=document.timeline_id,
                branch_id=document.branch_id,
                created_at=target_time,
                modified_at=datetime.utcnow(),
                temporal_coordinates=(target_time.timestamp(), 0.0, 0.0),
                causality_chain=document.causality_chain + [document_id]
            )
            
            self.temporal_documents[new_document_id] = new_document
            
            # Create temporal event
            await self._create_temporal_event(
                document_id=new_document_id,
                timeline_id=document.timeline_id,
                operation=operation,
                description=f"Time travel to {target_time.isoformat()}"
            )
            
            # Update metrics
            duration = time.time() - start_time
            timeline = self.timelines.get(document.timeline_id)
            
            self.prometheus_metrics["temporal_operation_duration"].labels(
                operation=operation.value,
                timeline_type=timeline.timeline_type.value if timeline else "unknown"
            ).observe(duration)
            
            self.prometheus_metrics["time_travel_events"].labels(
                operation=operation.value,
                direction="backward" if target_time < datetime.utcnow() else "forward"
            ).inc()
            
            logger.info(f"Time traveled to document {new_document_id} at {target_time}")
            return new_document_id
            
        except Exception as e:
            logger.error(f"Error in time travel: {e}")
            raise
    
    async def branch_timeline(self, timeline_id: str, branch_name: str) -> str:
        """Create timeline branch"""
        try:
            # Get original timeline
            original_timeline = self.timelines.get(timeline_id)
            if not original_timeline:
                raise ValueError(f"Timeline {timeline_id} not found")
            
            # Create new timeline
            new_timeline_id = f"timeline_{uuid.uuid4().hex[:8]}"
            
            new_timeline = Timeline(
                id=new_timeline_id,
                name=f"{original_timeline.name}_branch_{branch_name}",
                timeline_type=TimelineType.BRANCHED,
                status=TemporalStatus.DIVERGED,
                created_at=datetime.utcnow(),
                last_accessed=datetime.utcnow()
            )
            
            self.timelines[new_timeline_id] = new_timeline
            
            # Copy documents to new timeline
            for document in self.temporal_documents.values():
                if document.timeline_id == timeline_id:
                    # Create document copy
                    new_document_id = f"temporal_doc_{uuid.uuid4().hex[:8]}"
                    
                    new_document = TemporalDocument(
                        id=new_document_id,
                        content=document.content,
                        version=document.version,
                        timeline_id=new_timeline_id,
                        branch_id=branch_name,
                        created_at=document.created_at,
                        modified_at=datetime.utcnow(),
                        temporal_coordinates=document.temporal_coordinates,
                        causality_chain=document.causality_chain + [document.id]
                    )
                    
                    self.temporal_documents[new_document_id] = new_document
            
            # Update original timeline
            original_timeline.branch_count += 1
            
            logger.info(f"Branched timeline {timeline_id} to {new_timeline_id}")
            return new_timeline_id
            
        except Exception as e:
            logger.error(f"Error branching timeline: {e}")
            raise
    
    async def merge_timelines(self, source_timeline_id: str, target_timeline_id: str) -> bool:
        """Merge two timelines"""
        try:
            # Get timelines
            source_timeline = self.timelines.get(source_timeline_id)
            target_timeline = self.timelines.get(target_timeline_id)
            
            if not source_timeline or not target_timeline:
                return False
            
            # Check for merge conflicts
            conflicts = await self._detect_merge_conflicts(source_timeline_id, target_timeline_id)
            
            if conflicts:
                logger.warning(f"Merge conflicts detected: {conflicts}")
                # Implement conflict resolution
                await self._resolve_merge_conflicts(conflicts)
            
            # Merge documents
            source_documents = [d for d in self.temporal_documents.values() if d.timeline_id == source_timeline_id]
            
            for document in source_documents:
                # Update document timeline
                document.timeline_id = target_timeline_id
                document.modified_at = datetime.utcnow()
                
                # Create merge event
                await self._create_temporal_event(
                    document_id=document.id,
                    timeline_id=target_timeline_id,
                    operation=TemporalOperation.MERGE,
                    description=f"Merged from timeline {source_timeline_id}"
                )
            
            # Update timeline counts
            target_timeline.document_count += len(source_documents)
            target_timeline.last_accessed = datetime.utcnow()
            
            # Remove source timeline
            del self.timelines[source_timeline_id]
            
            logger.info(f"Merged timeline {source_timeline_id} into {target_timeline_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error merging timelines: {e}")
            return False
    
    async def _detect_merge_conflicts(self, source_timeline_id: str, target_timeline_id: str) -> List[str]:
        """Detect merge conflicts between timelines"""
        # Simulate conflict detection
        conflicts = []
        
        source_docs = [d for d in self.temporal_documents.values() if d.timeline_id == source_timeline_id]
        target_docs = [d for d in self.temporal_documents.values() if d.timeline_id == target_timeline_id]
        
        # Check for content conflicts
        for source_doc in source_docs:
            for target_doc in target_docs:
                if source_doc.content != target_doc.content:
                    conflicts.append(f"Content conflict between {source_doc.id} and {target_doc.id}")
        
        return conflicts
    
    async def _resolve_merge_conflicts(self, conflicts: List[str]):
        """Resolve merge conflicts"""
        # Simulate conflict resolution
        await asyncio.sleep(1)
        logger.info(f"Resolved {len(conflicts)} merge conflicts")
    
    async def _create_temporal_event(self, document_id: str, timeline_id: str,
                                   operation: TemporalOperation, description: str):
        """Create temporal event"""
        try:
            event_id = f"temporal_event_{uuid.uuid4().hex[:8]}"
            
            event = TemporalEvent(
                id=event_id,
                document_id=document_id,
                timeline_id=timeline_id,
                operation=operation,
                timestamp=datetime.utcnow(),
                temporal_coordinates=(time.time(), 0.0, 0.0),
                causality_impact=self._calculate_causality_impact(operation),
                paradox_risk=self._calculate_operation_paradox_risk(operation),
                description=description
            )
            
            self.temporal_events[event_id] = event
            
            # Update metrics
            self.prometheus_metrics["causality_impact"].labels(
                document_id=document_id,
                operation=operation.value
            ).set(event.causality_impact)
            
        except Exception as e:
            logger.error(f"Error creating temporal event: {e}")
    
    def _calculate_causality_impact(self, operation: TemporalOperation) -> float:
        """Calculate causality impact of operation"""
        impact_map = {
            TemporalOperation.CREATE: 0.1,
            TemporalOperation.UPDATE: 0.2,
            TemporalOperation.DELETE: -0.5,
            TemporalOperation.RESTORE: 0.3,
            TemporalOperation.BRANCH: 0.4,
            TemporalOperation.MERGE: 0.6,
            TemporalOperation.REVERT: -0.8,
            TemporalOperation.FORWARD: 0.1
        }
        
        return impact_map.get(operation, 0.0)
    
    def _calculate_operation_paradox_risk(self, operation: TemporalOperation) -> float:
        """Calculate paradox risk of operation"""
        risk_map = {
            TemporalOperation.CREATE: 0.0,
            TemporalOperation.UPDATE: 0.1,
            TemporalOperation.DELETE: 0.3,
            TemporalOperation.RESTORE: 0.4,
            TemporalOperation.BRANCH: 0.2,
            TemporalOperation.MERGE: 0.5,
            TemporalOperation.REVERT: 0.7,
            TemporalOperation.FORWARD: 0.2
        }
        
        return risk_map.get(operation, 0.0)
    
    def get_temporal_document(self, document_id: str) -> Optional[TemporalDocument]:
        """Get temporal document by ID"""
        return self.temporal_documents.get(document_id)
    
    def get_timeline(self, timeline_id: str) -> Optional[Timeline]:
        """Get timeline by ID"""
        return self.timelines.get(timeline_id)
    
    def list_timelines(self, status: Optional[TemporalStatus] = None) -> List[Timeline]:
        """List timelines"""
        timelines = list(self.timelines.values())
        
        if status:
            timelines = [t for t in timelines if t.status == status]
        
        return timelines
    
    def get_temporal_events(self, document_id: str) -> List[TemporalEvent]:
        """Get temporal events for document"""
        return [
            event for event in self.temporal_events.values()
            if event.document_id == document_id
        ]
    
    def get_temporal_statistics(self) -> Dict[str, Any]:
        """Get temporal statistics"""
        total_timelines = len(self.timelines)
        stable_timelines = len([t for t in self.timelines.values() if t.status == TemporalStatus.STABLE])
        unstable_timelines = len([t for t in self.timelines.values() if t.status == TemporalStatus.UNSTABLE])
        
        total_documents = len(self.temporal_documents)
        total_events = len(self.temporal_events)
        total_paradoxes = sum(t.paradox_count for t in self.timelines.values())
        
        # Count by timeline type
        timeline_type_counts = {}
        for timeline in self.timelines.values():
            timeline_type = timeline.timeline_type.value
            timeline_type_counts[timeline_type] = timeline_type_counts.get(timeline_type, 0) + 1
        
        # Count by operation
        operation_counts = {}
        for event in self.temporal_events.values():
            operation = event.operation.value
            operation_counts[operation] = operation_counts.get(operation, 0) + 1
        
        # Calculate average stability
        if self.timelines:
            avg_stability = sum(t.stability_score for t in self.timelines.values()) / len(self.timelines)
        else:
            avg_stability = 0.0
        
        return {
            "total_timelines": total_timelines,
            "stable_timelines": stable_timelines,
            "unstable_timelines": unstable_timelines,
            "total_documents": total_documents,
            "total_events": total_events,
            "total_paradoxes": total_paradoxes,
            "timeline_type_counts": timeline_type_counts,
            "operation_counts": operation_counts,
            "average_stability": avg_stability,
            "locked_documents": len(self.temporal_locks)
        }
    
    def export_temporal_data(self) -> Dict[str, Any]:
        """Export temporal data for analysis"""
        return {
            "temporal_documents": [
                {
                    "id": doc.id,
                    "content": doc.content,
                    "version": doc.version,
                    "timeline_id": doc.timeline_id,
                    "branch_id": doc.branch_id,
                    "created_at": doc.created_at.isoformat(),
                    "modified_at": doc.modified_at.isoformat(),
                    "temporal_coordinates": doc.temporal_coordinates,
                    "causality_chain": doc.causality_chain,
                    "paradox_risk": doc.paradox_risk,
                    "stability_score": doc.stability_score,
                    "metadata": doc.metadata
                }
                for doc in self.temporal_documents.values()
            ],
            "timelines": [
                {
                    "id": timeline.id,
                    "name": timeline.name,
                    "timeline_type": timeline.timeline_type.value,
                    "status": timeline.status.value,
                    "created_at": timeline.created_at.isoformat(),
                    "last_accessed": timeline.last_accessed.isoformat(),
                    "document_count": timeline.document_count,
                    "branch_count": timeline.branch_count,
                    "paradox_count": timeline.paradox_count,
                    "stability_score": timeline.stability_score,
                    "metadata": timeline.metadata
                }
                for timeline in self.timelines.values()
            ],
            "temporal_events": [
                {
                    "id": event.id,
                    "document_id": event.document_id,
                    "timeline_id": event.timeline_id,
                    "operation": event.operation.value,
                    "timestamp": event.timestamp.isoformat(),
                    "temporal_coordinates": event.temporal_coordinates,
                    "causality_impact": event.causality_impact,
                    "paradox_risk": event.paradox_risk,
                    "description": event.description,
                    "metadata": event.metadata
                }
                for event in self.temporal_events.values()
            ],
            "temporal_locks": [
                {
                    "document_id": doc_id,
                    "locked_at": lock_data["locked_at"].isoformat(),
                    "reason": lock_data["reason"],
                    "locked_by": lock_data["locked_by"]
                }
                for doc_id, lock_data in self.temporal_locks.items()
            ],
            "statistics": self.get_temporal_statistics(),
            "export_timestamp": datetime.utcnow().isoformat()
        }

# Global temporal document management instance
temporal_management = None

def get_temporal_management() -> TemporalDocumentManagement:
    """Get the global temporal document management instance"""
    global temporal_management
    if temporal_management is None:
        config = {
            "redis_host": "localhost",
            "redis_port": 6379,
            "redis_db": 8
        }
        temporal_management = TemporalDocumentManagement(config)
    return temporal_management

if __name__ == "__main__":
    # Example usage
    async def main():
        config = {
            "redis_host": "localhost",
            "redis_port": 6379,
            "redis_db": 8
        }
        
        temporal = TemporalDocumentManagement(config)
        
        # Create timeline
        timeline_id = temporal.create_timeline(
            name="Main Timeline",
            timeline_type=TimelineType.LINEAR
        )
        
        # Create temporal document
        doc_id = await temporal.create_temporal_document(
            content="This is a temporal document",
            timeline_id=timeline_id
        )
        
        # Time travel to past
        past_doc_id = await temporal.time_travel_to_document(
            document_id=doc_id,
            target_time=datetime.utcnow() - timedelta(hours=1),
            operation=TemporalOperation.REVERT
        )
        
        # Get statistics
        stats = temporal.get_temporal_statistics()
        print("Temporal Statistics:")
        print(json.dumps(stats, indent=2))
        
        await temporal.stop_monitoring()
    
    asyncio.run(main())













