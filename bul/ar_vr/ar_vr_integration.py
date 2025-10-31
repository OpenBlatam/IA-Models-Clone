"""
Ultimate BUL System - AR/VR Integration
Advanced augmented reality and virtual reality capabilities for immersive document generation and collaboration
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

class ARVRPlatform(str, Enum):
    """AR/VR platforms"""
    OCULUS = "oculus"
    HTC_VIVE = "htc_vive"
    MICROSOFT_HOLOLENS = "microsoft_hololens"
    MAGIC_LEAP = "magic_leap"
    APPLE_VISION_PRO = "apple_vision_pro"
    GOOGLE_CARDBOARD = "google_cardboard"
    SAMSUNG_GEAR_VR = "samsung_gear_vr"
    PLAYSTATION_VR = "playstation_vr"

class ARVRSessionType(str, Enum):
    """AR/VR session types"""
    DOCUMENT_VIEWING = "document_viewing"
    COLLABORATIVE_EDITING = "collaborative_editing"
    PRESENTATION = "presentation"
    TRAINING = "training"
    MEETING = "meeting"
    DESIGN_REVIEW = "design_review"
    VIRTUAL_OFFICE = "virtual_office"

class ARVRStatus(str, Enum):
    """AR/VR session status"""
    ACTIVE = "active"
    PAUSED = "paused"
    ENDED = "ended"
    ERROR = "error"
    CONNECTING = "connecting"

@dataclass
class ARVRSession:
    """AR/VR session"""
    id: str
    session_type: ARVRSessionType
    platform: ARVRPlatform
    status: ARVRStatus
    participants: List[str]
    document_id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ARVRObject:
    """AR/VR 3D object"""
    id: str
    name: str
    object_type: str
    position: Tuple[float, float, float]
    rotation: Tuple[float, float, float]
    scale: Tuple[float, float, float]
    mesh_data: Optional[Dict[str, Any]] = None
    texture_data: Optional[Dict[str, Any]] = None
    created_at: datetime = field(default_factory=datetime.utcnow)

@dataclass
class ARVRInteraction:
    """AR/VR user interaction"""
    id: str
    session_id: str
    user_id: str
    interaction_type: str
    position: Tuple[float, float, float]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    data: Dict[str, Any] = field(default_factory=dict)

class ARVRIntegration:
    """AR/VR integration system"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.ar_vr_sessions = {}
        self.ar_vr_objects = {}
        self.ar_vr_interactions = {}
        self.spatial_anchors = {}
        
        # Redis for AR/VR data caching
        self.redis_client = redis.Redis(
            host=config.get("redis_host", "localhost"),
            port=config.get("redis_port", 6379),
            db=config.get("redis_db", 5)
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
            "ar_vr_sessions": Counter(
                "bul_ar_vr_sessions_total",
                "Total AR/VR sessions",
                ["platform", "session_type", "status"]
            ),
            "ar_vr_session_duration": Histogram(
                "bul_ar_vr_session_duration_seconds",
                "AR/VR session duration in seconds",
                ["platform", "session_type"]
            ),
            "ar_vr_interactions": Counter(
                "bul_ar_vr_interactions_total",
                "Total AR/VR interactions",
                ["session_type", "interaction_type"]
            ),
            "ar_vr_objects": Gauge(
                "bul_ar_vr_objects",
                "Number of AR/VR objects",
                ["object_type"]
            ),
            "ar_vr_latency": Histogram(
                "bul_ar_vr_latency_seconds",
                "AR/VR processing latency in seconds",
                ["platform", "operation"]
            ),
            "active_ar_vr_sessions": Gauge(
                "bul_active_ar_vr_sessions",
                "Number of active AR/VR sessions"
            ),
            "ar_vr_platform_usage": Gauge(
                "bul_ar_vr_platform_usage",
                "AR/VR platform usage",
                ["platform"]
            )
        }
    
    async def start_monitoring(self):
        """Start AR/VR monitoring"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        logger.info("Starting AR/VR monitoring")
        
        # Start monitoring tasks
        asyncio.create_task(self._monitor_ar_vr_sessions())
        asyncio.create_task(self._monitor_ar_vr_objects())
        asyncio.create_task(self._update_metrics())
    
    async def stop_monitoring(self):
        """Stop AR/VR monitoring"""
        self.monitoring_active = False
        logger.info("Stopping AR/VR monitoring")
    
    async def _monitor_ar_vr_sessions(self):
        """Monitor AR/VR sessions"""
        while self.monitoring_active:
            try:
                current_time = datetime.utcnow()
                
                for session_id, session in self.ar_vr_sessions.items():
                    if session.status == ARVRStatus.ACTIVE:
                        # Check if session is still active
                        if session.started_at:
                            duration = (current_time - session.started_at).total_seconds()
                            
                            # Auto-end sessions after 2 hours
                            if duration > 7200:
                                session.status = ARVRStatus.ENDED
                                session.ended_at = current_time
                                
                                # Update metrics
                                self.prometheus_metrics["ar_vr_sessions"].labels(
                                    platform=session.platform.value,
                                    session_type=session.session_type.value,
                                    status="ended"
                                ).inc()
                
                # Update active sessions count
                active_sessions = len([s for s in self.ar_vr_sessions.values() if s.status == ARVRStatus.ACTIVE])
                self.prometheus_metrics["active_ar_vr_sessions"].set(active_sessions)
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error monitoring AR/VR sessions: {e}")
                await asyncio.sleep(60)
    
    async def _monitor_ar_vr_objects(self):
        """Monitor AR/VR objects"""
        while self.monitoring_active:
            try:
                # Update object counts by type
                object_type_counts = {}
                for obj in self.ar_vr_objects.values():
                    object_type = obj.object_type
                    object_type_counts[object_type] = object_type_counts.get(object_type, 0) + 1
                
                for object_type, count in object_type_counts.items():
                    self.prometheus_metrics["ar_vr_objects"].labels(
                        object_type=object_type
                    ).set(count)
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error monitoring AR/VR objects: {e}")
                await asyncio.sleep(60)
    
    async def _update_metrics(self):
        """Update Prometheus metrics"""
        while self.monitoring_active:
            try:
                # Update platform usage
                platform_counts = {}
                for session in self.ar_vr_sessions.values():
                    platform = session.platform.value
                    platform_counts[platform] = platform_counts.get(platform, 0) + 1
                
                for platform, count in platform_counts.items():
                    self.prometheus_metrics["ar_vr_platform_usage"].labels(
                        platform=platform
                    ).set(count)
                
                await asyncio.sleep(60)  # Update every minute
                
            except Exception as e:
                logger.error(f"Error updating metrics: {e}")
                await asyncio.sleep(60)
    
    async def create_ar_vr_session(self, session_type: ARVRSessionType, 
                                 platform: ARVRPlatform, participants: List[str],
                                 document_id: Optional[str] = None) -> str:
        """Create AR/VR session"""
        try:
            session_id = f"ar_vr_session_{uuid.uuid4().hex[:8]}"
            
            session = ARVRSession(
                id=session_id,
                session_type=session_type,
                platform=platform,
                status=ARVRStatus.CONNECTING,
                participants=participants,
                document_id=document_id
            )
            
            self.ar_vr_sessions[session_id] = session
            
            # Update metrics
            self.prometheus_metrics["ar_vr_sessions"].labels(
                platform=platform.value,
                session_type=session_type.value,
                status="connecting"
            ).inc()
            
            logger.info(f"Created AR/VR session: {session_id}")
            return session_id
            
        except Exception as e:
            logger.error(f"Error creating AR/VR session: {e}")
            raise
    
    async def start_ar_vr_session(self, session_id: str) -> bool:
        """Start AR/VR session"""
        try:
            session = self.ar_vr_sessions.get(session_id)
            if not session:
                return False
            
            session.status = ARVRStatus.ACTIVE
            session.started_at = datetime.utcnow()
            
            # Update metrics
            self.prometheus_metrics["ar_vr_sessions"].labels(
                platform=session.platform.value,
                session_type=session.session_type.value,
                status="active"
            ).inc()
            
            logger.info(f"Started AR/VR session: {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error starting AR/VR session: {e}")
            return False
    
    async def end_ar_vr_session(self, session_id: str) -> bool:
        """End AR/VR session"""
        try:
            session = self.ar_vr_sessions.get(session_id)
            if not session:
                return False
            
            session.status = ARVRStatus.ENDED
            session.ended_at = datetime.utcnow()
            
            # Calculate session duration
            if session.started_at:
                duration = (session.ended_at - session.started_at).total_seconds()
                
                # Update metrics
                self.prometheus_metrics["ar_vr_session_duration"].labels(
                    platform=session.platform.value,
                    session_type=session.session_type.value
                ).observe(duration)
            
            # Update metrics
            self.prometheus_metrics["ar_vr_sessions"].labels(
                platform=session.platform.value,
                session_type=session.session_type.value,
                status="ended"
            ).inc()
            
            logger.info(f"Ended AR/VR session: {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error ending AR/VR session: {e}")
            return False
    
    async def create_ar_vr_object(self, name: str, object_type: str,
                                position: Tuple[float, float, float],
                                rotation: Tuple[float, float, float] = (0, 0, 0),
                                scale: Tuple[float, float, float] = (1, 1, 1),
                                mesh_data: Optional[Dict[str, Any]] = None,
                                texture_data: Optional[Dict[str, Any]] = None) -> str:
        """Create AR/VR 3D object"""
        try:
            object_id = f"ar_vr_object_{uuid.uuid4().hex[:8]}"
            
            obj = ARVRObject(
                id=object_id,
                name=name,
                object_type=object_type,
                position=position,
                rotation=rotation,
                scale=scale,
                mesh_data=mesh_data,
                texture_data=texture_data
            )
            
            self.ar_vr_objects[object_id] = obj
            
            # Update metrics
            self.prometheus_metrics["ar_vr_objects"].labels(
                object_type=object_type
            ).inc()
            
            logger.info(f"Created AR/VR object: {object_id}")
            return object_id
            
        except Exception as e:
            logger.error(f"Error creating AR/VR object: {e}")
            raise
    
    async def update_ar_vr_object(self, object_id: str, position: Optional[Tuple[float, float, float]] = None,
                                rotation: Optional[Tuple[float, float, float]] = None,
                                scale: Optional[Tuple[float, float, float]] = None) -> bool:
        """Update AR/VR object"""
        try:
            obj = self.ar_vr_objects.get(object_id)
            if not obj:
                return False
            
            if position:
                obj.position = position
            if rotation:
                obj.rotation = rotation
            if scale:
                obj.scale = scale
            
            logger.info(f"Updated AR/VR object: {object_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating AR/VR object: {e}")
            return False
    
    async def record_ar_vr_interaction(self, session_id: str, user_id: str,
                                     interaction_type: str, position: Tuple[float, float, float],
                                     data: Optional[Dict[str, Any]] = None) -> str:
        """Record AR/VR user interaction"""
        try:
            interaction_id = f"ar_vr_interaction_{uuid.uuid4().hex[:8]}"
            
            interaction = ARVRInteraction(
                id=interaction_id,
                session_id=session_id,
                user_id=user_id,
                interaction_type=interaction_type,
                position=position,
                data=data or {}
            )
            
            self.ar_vr_interactions[interaction_id] = interaction
            
            # Update metrics
            session = self.ar_vr_sessions.get(session_id)
            if session:
                self.prometheus_metrics["ar_vr_interactions"].labels(
                    session_type=session.session_type.value,
                    interaction_type=interaction_type
                ).inc()
            
            logger.info(f"Recorded AR/VR interaction: {interaction_id}")
            return interaction_id
            
        except Exception as e:
            logger.error(f"Error recording AR/VR interaction: {e}")
            raise
    
    async def render_document_in_ar_vr(self, session_id: str, document_id: str,
                                     position: Tuple[float, float, float],
                                     scale: Tuple[float, float, float] = (1, 1, 1)) -> str:
        """Render document in AR/VR space"""
        try:
            start_time = time.time()
            
            # Create document object in AR/VR space
            object_id = await self.create_ar_vr_object(
                name=f"Document_{document_id}",
                object_type="document",
                position=position,
                scale=scale,
                mesh_data={
                    "type": "plane",
                    "width": 2.0,
                    "height": 1.5,
                    "material": "document_material"
                },
                texture_data={
                    "type": "document_texture",
                    "document_id": document_id,
                    "resolution": "high"
                }
            )
            
            # Record interaction
            await self.record_ar_vr_interaction(
                session_id=session_id,
                user_id="system",
                interaction_type="document_render",
                position=position,
                data={"document_id": document_id, "object_id": object_id}
            )
            
            # Update metrics
            duration = time.time() - start_time
            session = self.ar_vr_sessions.get(session_id)
            if session:
                self.prometheus_metrics["ar_vr_latency"].labels(
                    platform=session.platform.value,
                    operation="document_render"
                ).observe(duration)
            
            logger.info(f"Rendered document {document_id} in AR/VR session {session_id}")
            return object_id
            
        except Exception as e:
            logger.error(f"Error rendering document in AR/VR: {e}")
            raise
    
    async def create_collaborative_space(self, session_id: str, space_type: str,
                                       position: Tuple[float, float, float],
                                       dimensions: Tuple[float, float, float]) -> str:
        """Create collaborative AR/VR space"""
        try:
            # Create collaborative space object
            space_id = await self.create_ar_vr_object(
                name=f"CollaborativeSpace_{space_type}",
                object_type="collaborative_space",
                position=position,
                scale=dimensions,
                mesh_data={
                    "type": "room",
                    "width": dimensions[0],
                    "height": dimensions[1],
                    "depth": dimensions[2],
                    "material": "collaborative_space_material"
                },
                texture_data={
                    "type": "collaborative_space_texture",
                    "space_type": space_type,
                    "interactive": True
                }
            )
            
            # Store spatial anchor
            self.spatial_anchors[space_id] = {
                "position": position,
                "dimensions": dimensions,
                "space_type": space_type,
                "created_at": datetime.utcnow()
            }
            
            logger.info(f"Created collaborative space: {space_id}")
            return space_id
            
        except Exception as e:
            logger.error(f"Error creating collaborative space: {e}")
            raise
    
    async def track_user_gaze(self, session_id: str, user_id: str,
                            gaze_direction: Tuple[float, float, float],
                            gaze_origin: Tuple[float, float, float]) -> str:
        """Track user gaze in AR/VR"""
        try:
            # Calculate gaze position
            gaze_position = (
                gaze_origin[0] + gaze_direction[0] * 10,
                gaze_origin[1] + gaze_direction[1] * 10,
                gaze_origin[2] + gaze_direction[2] * 10
            )
            
            # Record gaze interaction
            interaction_id = await self.record_ar_vr_interaction(
                session_id=session_id,
                user_id=user_id,
                interaction_type="gaze_tracking",
                position=gaze_position,
                data={
                    "gaze_direction": gaze_direction,
                    "gaze_origin": gaze_origin,
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
            
            logger.info(f"Tracked user gaze: {interaction_id}")
            return interaction_id
            
        except Exception as e:
            logger.error(f"Error tracking user gaze: {e}")
            raise
    
    async def process_hand_gesture(self, session_id: str, user_id: str,
                                 gesture_type: str, hand_position: Tuple[float, float, float],
                                 gesture_data: Dict[str, Any]) -> str:
        """Process hand gesture in AR/VR"""
        try:
            # Record hand gesture interaction
            interaction_id = await self.record_ar_vr_interaction(
                session_id=session_id,
                user_id=user_id,
                interaction_type="hand_gesture",
                position=hand_position,
                data={
                    "gesture_type": gesture_type,
                    "hand_position": hand_position,
                    "gesture_data": gesture_data,
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
            
            # Process gesture based on type
            if gesture_type == "point":
                await self._process_point_gesture(session_id, hand_position, gesture_data)
            elif gesture_type == "grab":
                await self._process_grab_gesture(session_id, hand_position, gesture_data)
            elif gesture_type == "swipe":
                await self._process_swipe_gesture(session_id, hand_position, gesture_data)
            elif gesture_type == "pinch":
                await self._process_pinch_gesture(session_id, hand_position, gesture_data)
            
            logger.info(f"Processed hand gesture: {interaction_id}")
            return interaction_id
            
        except Exception as e:
            logger.error(f"Error processing hand gesture: {e}")
            raise
    
    async def _process_point_gesture(self, session_id: str, position: Tuple[float, float, float], data: Dict[str, Any]):
        """Process point gesture"""
        # Simulate point gesture processing
        await asyncio.sleep(0.1)
        
        # Find objects at position
        nearby_objects = []
        for obj in self.ar_vr_objects.values():
            distance = np.sqrt(sum((a - b) ** 2 for a, b in zip(obj.position, position)))
            if distance < 0.5:  # Within 0.5 units
                nearby_objects.append(obj.id)
        
        # Store interaction result
        await self.redis_client.setex(
            f"gesture_result:{session_id}",
            300,  # 5 minutes TTL
            json.dumps({
                "gesture_type": "point",
                "position": position,
                "nearby_objects": nearby_objects,
                "timestamp": datetime.utcnow().isoformat()
            })
        )
    
    async def _process_grab_gesture(self, session_id: str, position: Tuple[float, float, float], data: Dict[str, Any]):
        """Process grab gesture"""
        # Simulate grab gesture processing
        await asyncio.sleep(0.1)
        
        # Find objects at position
        grabbed_objects = []
        for obj in self.ar_vr_objects.values():
            distance = np.sqrt(sum((a - b) ** 2 for a, b in zip(obj.position, position)))
            if distance < 0.3:  # Within 0.3 units
                grabbed_objects.append(obj.id)
        
        # Store interaction result
        await self.redis_client.setex(
            f"gesture_result:{session_id}",
            300,  # 5 minutes TTL
            json.dumps({
                "gesture_type": "grab",
                "position": position,
                "grabbed_objects": grabbed_objects,
                "timestamp": datetime.utcnow().isoformat()
            })
        )
    
    async def _process_swipe_gesture(self, session_id: str, position: Tuple[float, float, float], data: Dict[str, Any]):
        """Process swipe gesture"""
        # Simulate swipe gesture processing
        await asyncio.sleep(0.1)
        
        # Store interaction result
        await self.redis_client.setex(
            f"gesture_result:{session_id}",
            300,  # 5 minutes TTL
            json.dumps({
                "gesture_type": "swipe",
                "position": position,
                "swipe_direction": data.get("direction", "unknown"),
                "timestamp": datetime.utcnow().isoformat()
            })
        )
    
    async def _process_pinch_gesture(self, session_id: str, position: Tuple[float, float, float], data: Dict[str, Any]):
        """Process pinch gesture"""
        # Simulate pinch gesture processing
        await asyncio.sleep(0.1)
        
        # Store interaction result
        await self.redis_client.setex(
            f"gesture_result:{session_id}",
            300,  # 5 minutes TTL
            json.dumps({
                "gesture_type": "pinch",
                "position": position,
                "pinch_scale": data.get("scale", 1.0),
                "timestamp": datetime.utcnow().isoformat()
            })
        )
    
    def get_ar_vr_session(self, session_id: str) -> Optional[ARVRSession]:
        """Get AR/VR session by ID"""
        return self.ar_vr_sessions.get(session_id)
    
    def list_ar_vr_sessions(self, status: Optional[ARVRStatus] = None) -> List[ARVRSession]:
        """List AR/VR sessions"""
        sessions = list(self.ar_vr_sessions.values())
        
        if status:
            sessions = [s for s in sessions if s.status == status]
        
        return sessions
    
    def get_ar_vr_object(self, object_id: str) -> Optional[ARVRObject]:
        """Get AR/VR object by ID"""
        return self.ar_vr_objects.get(object_id)
    
    def list_ar_vr_objects(self, object_type: Optional[str] = None) -> List[ARVRObject]:
        """List AR/VR objects"""
        objects = list(self.ar_vr_objects.values())
        
        if object_type:
            objects = [o for o in objects if o.object_type == object_type]
        
        return objects
    
    def get_ar_vr_interactions(self, session_id: str) -> List[ARVRInteraction]:
        """Get AR/VR interactions for session"""
        return [
            interaction for interaction in self.ar_vr_interactions.values()
            if interaction.session_id == session_id
        ]
    
    def get_ar_vr_statistics(self) -> Dict[str, Any]:
        """Get AR/VR statistics"""
        total_sessions = len(self.ar_vr_sessions)
        active_sessions = len([s for s in self.ar_vr_sessions.values() if s.status == ARVRStatus.ACTIVE])
        ended_sessions = len([s for s in self.ar_vr_sessions.values() if s.status == ARVRStatus.ENDED])
        
        # Count by platform
        platform_counts = {}
        for session in self.ar_vr_sessions.values():
            platform = session.platform.value
            platform_counts[platform] = platform_counts.get(platform, 0) + 1
        
        # Count by session type
        session_type_counts = {}
        for session in self.ar_vr_sessions.values():
            session_type = session.session_type.value
            session_type_counts[session_type] = session_type_counts.get(session_type, 0) + 1
        
        # Count by object type
        object_type_counts = {}
        for obj in self.ar_vr_objects.values():
            object_type = obj.object_type
            object_type_counts[object_type] = object_type_counts.get(object_type, 0) + 1
        
        # Count interactions by type
        interaction_type_counts = {}
        for interaction in self.ar_vr_interactions.values():
            interaction_type = interaction.interaction_type
            interaction_type_counts[interaction_type] = interaction_type_counts.get(interaction_type, 0) + 1
        
        return {
            "total_sessions": total_sessions,
            "active_sessions": active_sessions,
            "ended_sessions": ended_sessions,
            "platform_counts": platform_counts,
            "session_type_counts": session_type_counts,
            "total_objects": len(self.ar_vr_objects),
            "object_type_counts": object_type_counts,
            "total_interactions": len(self.ar_vr_interactions),
            "interaction_type_counts": interaction_type_counts,
            "spatial_anchors": len(self.spatial_anchors)
        }
    
    def export_ar_vr_data(self) -> Dict[str, Any]:
        """Export AR/VR data for analysis"""
        return {
            "ar_vr_sessions": [
                {
                    "id": session.id,
                    "session_type": session.session_type.value,
                    "platform": session.platform.value,
                    "status": session.status.value,
                    "participants": session.participants,
                    "document_id": session.document_id,
                    "created_at": session.created_at.isoformat(),
                    "started_at": session.started_at.isoformat() if session.started_at else None,
                    "ended_at": session.ended_at.isoformat() if session.ended_at else None,
                    "metadata": session.metadata
                }
                for session in self.ar_vr_sessions.values()
            ],
            "ar_vr_objects": [
                {
                    "id": obj.id,
                    "name": obj.name,
                    "object_type": obj.object_type,
                    "position": obj.position,
                    "rotation": obj.rotation,
                    "scale": obj.scale,
                    "created_at": obj.created_at.isoformat()
                }
                for obj in self.ar_vr_objects.values()
            ],
            "ar_vr_interactions": [
                {
                    "id": interaction.id,
                    "session_id": interaction.session_id,
                    "user_id": interaction.user_id,
                    "interaction_type": interaction.interaction_type,
                    "position": interaction.position,
                    "timestamp": interaction.timestamp.isoformat(),
                    "data": interaction.data
                }
                for interaction in self.ar_vr_interactions.values()
            ],
            "spatial_anchors": [
                {
                    "anchor_id": anchor_id,
                    "position": anchor_data["position"],
                    "dimensions": anchor_data["dimensions"],
                    "space_type": anchor_data["space_type"],
                    "created_at": anchor_data["created_at"].isoformat()
                }
                for anchor_id, anchor_data in self.spatial_anchors.items()
            ],
            "statistics": self.get_ar_vr_statistics(),
            "export_timestamp": datetime.utcnow().isoformat()
        }

# Global AR/VR integration instance
ar_vr_integration = None

def get_ar_vr_integration() -> ARVRIntegration:
    """Get the global AR/VR integration instance"""
    global ar_vr_integration
    if ar_vr_integration is None:
        config = {
            "redis_host": "localhost",
            "redis_port": 6379,
            "redis_db": 5
        }
        ar_vr_integration = ARVRIntegration(config)
    return ar_vr_integration

if __name__ == "__main__":
    # Example usage
    async def main():
        config = {
            "redis_host": "localhost",
            "redis_port": 6379,
            "redis_db": 5
        }
        
        ar_vr = ARVRIntegration(config)
        
        # Create AR/VR session
        session_id = await ar_vr.create_ar_vr_session(
            session_type=ARVRSessionType.DOCUMENT_VIEWING,
            platform=ARVRPlatform.OCULUS,
            participants=["user1", "user2"],
            document_id="doc_123"
        )
        
        # Start session
        await ar_vr.start_ar_vr_session(session_id)
        
        # Render document in AR/VR
        object_id = await ar_vr.render_document_in_ar_vr(
            session_id=session_id,
            document_id="doc_123",
            position=(0, 0, -2),
            scale=(1, 1, 1)
        )
        
        # Track user gaze
        await ar_vr.track_user_gaze(
            session_id=session_id,
            user_id="user1",
            gaze_direction=(0, 0, -1),
            gaze_origin=(0, 0, 0)
        )
        
        # Process hand gesture
        await ar_vr.process_hand_gesture(
            session_id=session_id,
            user_id="user1",
            gesture_type="point",
            hand_position=(0.5, 0.5, -1.5),
            gesture_data={"confidence": 0.95}
        )
        
        # Get statistics
        stats = ar_vr.get_ar_vr_statistics()
        print("AR/VR Statistics:")
        print(json.dumps(stats, indent=2))
        
        await ar_vr.stop_monitoring()
    
    asyncio.run(main())













