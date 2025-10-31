"""
BUL Metaverse Integration
========================

Metaverse integration for immersive document experiences and virtual collaboration.
"""

import asyncio
import json
import hashlib
import time
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, asdict
from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field
import aiohttp
import websockets
import numpy as np
import base64
from PIL import Image
import io
import uuid

from ..utils import get_logger, get_cache_manager
from ..config import get_config
from ..core import BusinessArea, DocumentType

logger = get_logger(__name__)

class MetaversePlatform(str, Enum):
    """Supported metaverse platforms"""
    DECENTRALAND = "decentraland"
    SANDBOX = "sandbox"
    VRChat = "vrchat"
    HORIZON_WORLDS = "horizon_worlds"
    SPATIAL = "spatial"
    GATHER = "gather"
    CUSTOM_VR = "custom_vr"
    WEBXR = "webxr"

class AvatarType(str, Enum):
    """Avatar types"""
    HUMAN = "human"
    ROBOT = "robot"
    ANIMAL = "animal"
    ABSTRACT = "abstract"
    CORPORATE = "corporate"
    CUSTOM = "custom"

class VirtualObjectType(str, Enum):
    """Virtual object types"""
    DOCUMENT = "document"
    PRESENTATION = "presentation"
    WHITEBOARD = "whiteboard"
    SCREEN = "screen"
    HOLOGRAM = "hologram"
    INTERACTIVE_PANEL = "interactive_panel"
    DATA_VISUALIZATION = "data_visualization"
    VIRTUAL_ASSISTANT = "virtual_assistant"

class InteractionType(str, Enum):
    """Interaction types"""
    VIEW = "view"
    EDIT = "edit"
    ANNOTATE = "annotate"
    COLLABORATE = "collaborate"
    PRESENT = "present"
    DISCUSS = "discuss"
    VOTE = "vote"
    GESTURE = "gesture"

@dataclass
class VirtualAvatar:
    """Virtual avatar representation"""
    id: str
    user_id: str
    name: str
    avatar_type: AvatarType
    appearance: Dict[str, Any]
    position: Dict[str, float]  # x, y, z coordinates
    rotation: Dict[str, float]  # x, y, z rotation
    animation_state: str
    current_activity: str
    permissions: List[str]
    metadata: Dict[str, Any] = None

@dataclass
class VirtualObject:
    """Virtual object in metaverse"""
    id: str
    object_type: VirtualObjectType
    name: str
    position: Dict[str, float]
    rotation: Dict[str, float]
    scale: Dict[str, float]
    content: Dict[str, Any]
    interactions: List[InteractionType]
    permissions: List[str]
    physics_enabled: bool = False
    collider_enabled: bool = True
    metadata: Dict[str, Any] = None

@dataclass
class VirtualSpace:
    """Virtual space/room in metaverse"""
    id: str
    name: str
    platform: MetaversePlatform
    description: str
    capacity: int
    current_users: List[str]
    objects: List[VirtualObject]
    environment: Dict[str, Any]
    permissions: List[str]
    created_at: datetime
    metadata: Dict[str, Any] = None

@dataclass
class MetaverseEvent:
    """Metaverse event/action"""
    id: str
    event_type: str
    user_id: str
    object_id: Optional[str]
    position: Dict[str, float]
    data: Dict[str, Any]
    timestamp: datetime
    space_id: str

class MetaverseIntegration:
    """Metaverse integration system"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.cache_manager = get_cache_manager()
        self.config = get_config()
        
        # Metaverse data
        self.virtual_spaces: Dict[str, VirtualSpace] = {}
        self.avatars: Dict[str, VirtualAvatar] = {}
        self.virtual_objects: Dict[str, VirtualObject] = {}
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        
        # WebSocket connections
        self.websocket_connections: Dict[str, WebSocket] = {}
        
        # Platform integrations
        self.platform_clients: Dict[MetaversePlatform, Any] = {}
        
        # Initialize metaverse services
        self._initialize_metaverse_services()
    
    def _initialize_metaverse_services(self):
        """Initialize metaverse platform services"""
        try:
            # Initialize platform clients
            self._initialize_platform_clients()
            
            # Create default virtual spaces
            self._create_default_spaces()
            
            # Start background tasks
            asyncio.create_task(self._metaverse_sync_worker())
            asyncio.create_task(self._avatar_animation_worker())
            
            self.logger.info("Metaverse integration initialized successfully")
        
        except Exception as e:
            self.logger.error(f"Failed to initialize metaverse services: {e}")
    
    def _initialize_platform_clients(self):
        """Initialize platform-specific clients"""
        try:
            # Initialize platform clients (simplified)
            for platform in MetaversePlatform:
                self.platform_clients[platform] = {
                    'connected': False,
                    'api_key': None,
                    'endpoint': None
                }
            
            self.logger.info(f"Initialized {len(self.platform_clients)} platform clients")
        
        except Exception as e:
            self.logger.error(f"Error initializing platform clients: {e}")
    
    def _create_default_spaces(self):
        """Create default virtual spaces"""
        try:
            # Create document collaboration space
            doc_space = VirtualSpace(
                id="doc_collaboration_space",
                name="Document Collaboration Hub",
                platform=MetaversePlatform.WEBXR,
                description="A virtual space for collaborative document editing and review",
                capacity=20,
                current_users=[],
                objects=[],
                environment={
                    'theme': 'modern_office',
                    'lighting': 'bright',
                    'ambient_sound': 'office_ambient',
                    'background': 'city_skyline'
                },
                permissions=['view', 'edit', 'collaborate'],
                created_at=datetime.now()
            )
            
            self.virtual_spaces[doc_space.id] = doc_space
            
            # Create presentation theater
            pres_space = VirtualSpace(
                id="presentation_theater",
                name="Presentation Theater",
                platform=MetaversePlatform.WEBXR,
                description="A virtual theater for document presentations and meetings",
                capacity=50,
                current_users=[],
                objects=[],
                environment={
                    'theme': 'theater',
                    'lighting': 'stage',
                    'ambient_sound': 'audience_ambient',
                    'background': 'theater_stage'
                },
                permissions=['view', 'present', 'discuss'],
                created_at=datetime.now()
            )
            
            self.virtual_spaces[pres_space.id] = pres_space
            
            self.logger.info(f"Created {len(self.virtual_spaces)} default virtual spaces")
        
        except Exception as e:
            self.logger.error(f"Error creating default spaces: {e}")
    
    async def create_virtual_space(
        self,
        name: str,
        platform: MetaversePlatform,
        description: str,
        capacity: int,
        environment: Dict[str, Any],
        permissions: List[str]
    ) -> VirtualSpace:
        """Create a new virtual space"""
        try:
            space_id = str(uuid.uuid4())
            
            space = VirtualSpace(
                id=space_id,
                name=name,
                platform=platform,
                description=description,
                capacity=capacity,
                current_users=[],
                objects=[],
                environment=environment,
                permissions=permissions,
                created_at=datetime.now()
            )
            
            self.virtual_spaces[space_id] = space
            
            # Create space on platform
            await self._create_space_on_platform(space)
            
            self.logger.info(f"Created virtual space: {name} ({platform.value})")
            return space
        
        except Exception as e:
            self.logger.error(f"Error creating virtual space: {e}")
            raise
    
    async def _create_space_on_platform(self, space: VirtualSpace):
        """Create space on metaverse platform"""
        try:
            # Platform-specific space creation
            platform_client = self.platform_clients.get(space.platform)
            
            if platform_client:
                # Simulate platform API call
                await asyncio.sleep(0.1)
                platform_client['connected'] = True
                
                self.logger.info(f"Space created on {space.platform.value}")
        
        except Exception as e:
            self.logger.error(f"Error creating space on platform: {e}")
    
    async def create_avatar(
        self,
        user_id: str,
        name: str,
        avatar_type: AvatarType,
        appearance: Dict[str, Any]
    ) -> VirtualAvatar:
        """Create a virtual avatar"""
        try:
            avatar_id = str(uuid.uuid4())
            
            avatar = VirtualAvatar(
                id=avatar_id,
                user_id=user_id,
                name=name,
                avatar_type=avatar_type,
                appearance=appearance,
                position={'x': 0, 'y': 0, 'z': 0},
                rotation={'x': 0, 'y': 0, 'z': 0},
                animation_state='idle',
                current_activity='none',
                permissions=['move', 'interact', 'speak']
            )
            
            self.avatars[avatar_id] = avatar
            
            self.logger.info(f"Created avatar: {name} for user {user_id}")
            return avatar
        
        except Exception as e:
            self.logger.error(f"Error creating avatar: {e}")
            raise
    
    async def create_virtual_document(
        self,
        document_content: str,
        document_metadata: Dict[str, Any],
        position: Dict[str, float],
        space_id: str
    ) -> VirtualObject:
        """Create a virtual document object"""
        try:
            object_id = str(uuid.uuid4())
            
            # Create document visualization
            document_visualization = await self._create_document_visualization(
                document_content, document_metadata
            )
            
            virtual_doc = VirtualObject(
                id=object_id,
                object_type=VirtualObjectType.DOCUMENT,
                name=document_metadata.get('title', 'Untitled Document'),
                position=position,
                rotation={'x': 0, 'y': 0, 'z': 0},
                scale={'x': 1, 'y': 1, 'z': 0.1},
                content={
                    'text_content': document_content,
                    'metadata': document_metadata,
                    'visualization': document_visualization,
                    'pages': self._split_document_into_pages(document_content),
                    'interactive_elements': self._extract_interactive_elements(document_content)
                },
                interactions=[
                    InteractionType.VIEW,
                    InteractionType.EDIT,
                    InteractionType.ANNOTATE,
                    InteractionType.COLLABORATE
                ],
                permissions=['view', 'edit', 'annotate'],
                physics_enabled=True,
                collider_enabled=True
            )
            
            self.virtual_objects[object_id] = virtual_doc
            
            # Add to virtual space
            if space_id in self.virtual_spaces:
                self.virtual_spaces[space_id].objects.append(virtual_doc)
            
            self.logger.info(f"Created virtual document: {virtual_doc.name}")
            return virtual_doc
        
        except Exception as e:
            self.logger.error(f"Error creating virtual document: {e}")
            raise
    
    async def _create_document_visualization(
        self,
        content: str,
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create 3D visualization of document"""
        try:
            # Analyze document structure
            word_count = len(content.split())
            paragraph_count = len([p for p in content.split('\n\n') if p.strip()])
            section_count = content.count('#')
            
            # Create visualization data
            visualization = {
                'type': '3d_document',
                'dimensions': {
                    'width': min(2.0, max(0.5, word_count / 1000.0)),
                    'height': min(3.0, max(0.5, paragraph_count / 10.0)),
                    'depth': 0.1
                },
                'structure': {
                    'sections': section_count,
                    'paragraphs': paragraph_count,
                    'words': word_count
                },
                'visual_elements': {
                    'background_color': self._get_document_theme_color(metadata),
                    'text_color': '#000000',
                    'border_style': 'modern',
                    'shadow_enabled': True
                },
                'interactive_features': {
                    'page_turning': True,
                    'zoom': True,
                    'highlight': True,
                    'annotations': True
                }
            }
            
            return visualization
        
        except Exception as e:
            self.logger.error(f"Error creating document visualization: {e}")
            return {}
    
    def _get_document_theme_color(self, metadata: Dict[str, Any]) -> str:
        """Get theme color based on document metadata"""
        try:
            doc_type = metadata.get('document_type', 'unknown')
            
            color_map = {
                'contract': '#FF6B6B',      # Red
                'report': '#4ECDC4',        # Teal
                'proposal': '#45B7D1',      # Blue
                'manual': '#96CEB4',        # Green
                'policy': '#FFEAA7',        # Yellow
                'agreement': '#DDA0DD',     # Plum
                'unknown': '#D3D3D3'        # Light Gray
            }
            
            return color_map.get(doc_type, color_map['unknown'])
        
        except Exception:
            return '#D3D3D3'
    
    def _split_document_into_pages(self, content: str) -> List[Dict[str, Any]]:
        """Split document into virtual pages"""
        try:
            # Simple page splitting (in practice, this would be more sophisticated)
            words_per_page = 250
            words = content.split()
            
            pages = []
            for i in range(0, len(words), words_per_page):
                page_words = words[i:i + words_per_page]
                page_content = ' '.join(page_words)
                
                page = {
                    'page_number': len(pages) + 1,
                    'content': page_content,
                    'word_count': len(page_words),
                    'position': {'x': 0, 'y': len(pages) * 0.3, 'z': 0}
                }
                
                pages.append(page)
            
            return pages
        
        except Exception as e:
            self.logger.error(f"Error splitting document into pages: {e}")
            return []
    
    def _extract_interactive_elements(self, content: str) -> List[Dict[str, Any]]:
        """Extract interactive elements from document"""
        try:
            elements = []
            
            # Find headings
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if line.startswith('#'):
                    elements.append({
                        'type': 'heading',
                        'text': line.strip('# '),
                        'level': len(line) - len(line.lstrip('#')),
                        'line_number': i,
                        'interactive': True
                    })
            
            # Find links
            import re
            links = re.findall(r'\[([^\]]+)\]\(([^)]+)\)', content)
            for link_text, link_url in links:
                elements.append({
                    'type': 'link',
                    'text': link_text,
                    'url': link_url,
                    'interactive': True
                })
            
            # Find lists
            list_items = re.findall(r'^\s*[-*+]\s+(.+)$', content, re.MULTILINE)
            for item in list_items:
                elements.append({
                    'type': 'list_item',
                    'text': item,
                    'interactive': True
                })
            
            return elements
        
        except Exception as e:
            self.logger.error(f"Error extracting interactive elements: {e}")
            return []
    
    async def join_virtual_space(
        self,
        space_id: str,
        user_id: str,
        avatar_id: str
    ) -> Dict[str, Any]:
        """Join a virtual space"""
        try:
            if space_id not in self.virtual_spaces:
                raise ValueError(f"Virtual space {space_id} not found")
            
            space = self.virtual_spaces[space_id]
            
            if len(space.current_users) >= space.capacity:
                raise ValueError("Virtual space is at capacity")
            
            if avatar_id not in self.avatars:
                raise ValueError(f"Avatar {avatar_id} not found")
            
            avatar = self.avatars[avatar_id]
            
            # Add user to space
            if user_id not in space.current_users:
                space.current_users.append(user_id)
            
            # Create session
            session_id = str(uuid.uuid4())
            session = {
                'session_id': session_id,
                'user_id': user_id,
                'avatar_id': avatar_id,
                'space_id': space_id,
                'joined_at': datetime.now(),
                'position': avatar.position,
                'permissions': avatar.permissions
            }
            
            self.active_sessions[session_id] = session
            
            # Broadcast user joined event
            await self._broadcast_space_event(space_id, {
                'type': 'user_joined',
                'user_id': user_id,
                'avatar_id': avatar_id,
                'position': avatar.position
            })
            
            self.logger.info(f"User {user_id} joined space {space_id}")
            
            return {
                'session_id': session_id,
                'space': asdict(space),
                'avatar': asdict(avatar),
                'success': True
            }
        
        except Exception as e:
            self.logger.error(f"Error joining virtual space: {e}")
            raise
    
    async def interact_with_object(
        self,
        session_id: str,
        object_id: str,
        interaction_type: InteractionType,
        interaction_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Interact with a virtual object"""
        try:
            if session_id not in self.active_sessions:
                raise ValueError("Invalid session")
            
            if object_id not in self.virtual_objects:
                raise ValueError("Object not found")
            
            session = self.active_sessions[session_id]
            virtual_object = self.virtual_objects[object_id]
            
            # Check permissions
            if interaction_type.value not in virtual_object.interactions:
                raise ValueError(f"Interaction {interaction_type.value} not allowed")
            
            # Process interaction
            result = await self._process_interaction(
                session, virtual_object, interaction_type, interaction_data
            )
            
            # Broadcast interaction event
            await self._broadcast_space_event(session['space_id'], {
                'type': 'object_interaction',
                'user_id': session['user_id'],
                'object_id': object_id,
                'interaction_type': interaction_type.value,
                'data': interaction_data,
                'result': result
            })
            
            return result
        
        except Exception as e:
            self.logger.error(f"Error interacting with object: {e}")
            raise
    
    async def _process_interaction(
        self,
        session: Dict[str, Any],
        virtual_object: VirtualObject,
        interaction_type: InteractionType,
        interaction_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process object interaction"""
        try:
            if interaction_type == InteractionType.VIEW:
                return await self._process_view_interaction(virtual_object, interaction_data)
            elif interaction_type == InteractionType.EDIT:
                return await self._process_edit_interaction(virtual_object, interaction_data)
            elif interaction_type == InteractionType.ANNOTATE:
                return await self._process_annotate_interaction(virtual_object, interaction_data)
            elif interaction_type == InteractionType.COLLABORATE:
                return await self._process_collaborate_interaction(virtual_object, interaction_data)
            else:
                return {'success': False, 'message': 'Unsupported interaction type'}
        
        except Exception as e:
            self.logger.error(f"Error processing interaction: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _process_view_interaction(
        self,
        virtual_object: VirtualObject,
        interaction_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process view interaction"""
        try:
            return {
                'success': True,
                'content': virtual_object.content,
                'view_mode': interaction_data.get('view_mode', 'full'),
                'zoom_level': interaction_data.get('zoom_level', 1.0)
            }
        
        except Exception as e:
            self.logger.error(f"Error processing view interaction: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _process_edit_interaction(
        self,
        virtual_object: VirtualObject,
        interaction_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process edit interaction"""
        try:
            # Simulate document editing
            new_content = interaction_data.get('content', '')
            if new_content:
                virtual_object.content['text_content'] = new_content
            
            return {
                'success': True,
                'updated_content': virtual_object.content,
                'edit_timestamp': datetime.now().isoformat()
            }
        
        except Exception as e:
            self.logger.error(f"Error processing edit interaction: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _process_annotate_interaction(
        self,
        virtual_object: VirtualObject,
        interaction_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process annotation interaction"""
        try:
            annotation = {
                'id': str(uuid.uuid4()),
                'text': interaction_data.get('annotation_text', ''),
                'position': interaction_data.get('position', {}),
                'color': interaction_data.get('color', '#FFFF00'),
                'timestamp': datetime.now().isoformat()
            }
            
            if 'annotations' not in virtual_object.content:
                virtual_object.content['annotations'] = []
            
            virtual_object.content['annotations'].append(annotation)
            
            return {
                'success': True,
                'annotation': annotation,
                'total_annotations': len(virtual_object.content['annotations'])
            }
        
        except Exception as e:
            self.logger.error(f"Error processing annotation interaction: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _process_collaborate_interaction(
        self,
        virtual_object: VirtualObject,
        interaction_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process collaboration interaction"""
        try:
            collaboration_action = interaction_data.get('action', 'discuss')
            
            if collaboration_action == 'discuss':
                return await self._process_discussion(virtual_object, interaction_data)
            elif collaboration_action == 'vote':
                return await self._process_voting(virtual_object, interaction_data)
            else:
                return {'success': False, 'message': 'Unsupported collaboration action'}
        
        except Exception as e:
            self.logger.error(f"Error processing collaboration interaction: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _process_discussion(
        self,
        virtual_object: VirtualObject,
        interaction_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process discussion interaction"""
        try:
            discussion = {
                'id': str(uuid.uuid4()),
                'user_id': interaction_data.get('user_id'),
                'message': interaction_data.get('message', ''),
                'timestamp': datetime.now().isoformat(),
                'position': interaction_data.get('position', {})
            }
            
            if 'discussions' not in virtual_object.content:
                virtual_object.content['discussions'] = []
            
            virtual_object.content['discussions'].append(discussion)
            
            return {
                'success': True,
                'discussion': discussion,
                'total_discussions': len(virtual_object.content['discussions'])
            }
        
        except Exception as e:
            self.logger.error(f"Error processing discussion: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _process_voting(
        self,
        virtual_object: VirtualObject,
        interaction_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process voting interaction"""
        try:
            vote = {
                'id': str(uuid.uuid4()),
                'user_id': interaction_data.get('user_id'),
                'vote': interaction_data.get('vote', 'approve'),
                'timestamp': datetime.now().isoformat()
            }
            
            if 'votes' not in virtual_object.content:
                virtual_object.content['votes'] = []
            
            virtual_object.content['votes'].append(vote)
            
            # Calculate vote results
            votes = virtual_object.content['votes']
            approve_count = len([v for v in votes if v['vote'] == 'approve'])
            reject_count = len([v for v in votes if v['vote'] == 'reject'])
            
            return {
                'success': True,
                'vote': vote,
                'results': {
                    'approve': approve_count,
                    'reject': reject_count,
                    'total': len(votes)
                }
            }
        
        except Exception as e:
            self.logger.error(f"Error processing voting: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _broadcast_space_event(self, space_id: str, event_data: Dict[str, Any]):
        """Broadcast event to all users in space"""
        try:
            if space_id not in self.virtual_spaces:
                return
            
            space = self.virtual_spaces[space_id]
            
            # Send to all active sessions in this space
            for session_id, session in self.active_sessions.items():
                if session['space_id'] == space_id:
                    # In a real implementation, this would send via WebSocket
                    self.logger.debug(f"Broadcasting to session {session_id}: {event_data}")
        
        except Exception as e:
            self.logger.error(f"Error broadcasting space event: {e}")
    
    async def _metaverse_sync_worker(self):
        """Background worker for metaverse synchronization"""
        while True:
            try:
                # Sync with metaverse platforms
                await self._sync_with_platforms()
                
                # Update avatar positions and states
                await self._update_avatar_states()
                
                # Clean up inactive sessions
                await self._cleanup_inactive_sessions()
                
                await asyncio.sleep(5)  # Sync every 5 seconds
            
            except Exception as e:
                self.logger.error(f"Error in metaverse sync worker: {e}")
                await asyncio.sleep(5)
    
    async def _sync_with_platforms(self):
        """Sync with metaverse platforms"""
        try:
            for platform, client in self.platform_clients.items():
                if client.get('connected'):
                    # Simulate platform sync
                    await asyncio.sleep(0.01)
        
        except Exception as e:
            self.logger.error(f"Error syncing with platforms: {e}")
    
    async def _update_avatar_states(self):
        """Update avatar states and animations"""
        try:
            for avatar in self.avatars.values():
                # Update animation state based on activity
                if avatar.current_activity == 'walking':
                    avatar.animation_state = 'walking'
                elif avatar.current_activity == 'typing':
                    avatar.animation_state = 'typing'
                else:
                    avatar.animation_state = 'idle'
        
        except Exception as e:
            self.logger.error(f"Error updating avatar states: {e}")
    
    async def _cleanup_inactive_sessions(self):
        """Clean up inactive sessions"""
        try:
            current_time = datetime.now()
            inactive_sessions = []
            
            for session_id, session in self.active_sessions.items():
                # Remove sessions inactive for more than 30 minutes
                if (current_time - session['joined_at']).total_seconds() > 1800:
                    inactive_sessions.append(session_id)
            
            for session_id in inactive_sessions:
                session = self.active_sessions[session_id]
                space_id = session['space_id']
                
                # Remove user from space
                if space_id in self.virtual_spaces:
                    space = self.virtual_spaces[space_id]
                    if session['user_id'] in space.current_users:
                        space.current_users.remove(session['user_id'])
                
                # Remove session
                del self.active_sessions[session_id]
                
                self.logger.info(f"Cleaned up inactive session: {session_id}")
        
        except Exception as e:
            self.logger.error(f"Error cleaning up inactive sessions: {e}")
    
    async def _avatar_animation_worker(self):
        """Background worker for avatar animations"""
        while True:
            try:
                # Update avatar animations
                for avatar in self.avatars.values():
                    # Simulate animation updates
                    pass
                
                await asyncio.sleep(0.1)  # 10 FPS animation updates
            
            except Exception as e:
                self.logger.error(f"Error in avatar animation worker: {e}")
                await asyncio.sleep(0.1)
    
    async def get_metaverse_status(self) -> Dict[str, Any]:
        """Get metaverse integration status"""
        try:
            total_spaces = len(self.virtual_spaces)
            total_avatars = len(self.avatars)
            total_objects = len(self.virtual_objects)
            active_sessions = len(self.active_sessions)
            
            # Calculate space occupancy
            space_occupancy = {}
            for space_id, space in self.virtual_spaces.items():
                occupancy = len(space.current_users) / max(space.capacity, 1)
                space_occupancy[space_id] = {
                    'name': space.name,
                    'current_users': len(space.current_users),
                    'capacity': space.capacity,
                    'occupancy_rate': round(occupancy * 100, 1)
                }
            
            return {
                'total_virtual_spaces': total_spaces,
                'total_avatars': total_avatars,
                'total_virtual_objects': total_objects,
                'active_sessions': active_sessions,
                'space_occupancy': space_occupancy,
                'platform_connections': {
                    platform.value: client.get('connected', False)
                    for platform, client in self.platform_clients.items()
                }
            }
        
        except Exception as e:
            self.logger.error(f"Error getting metaverse status: {e}")
            return {}

# Global metaverse integration
_metaverse_integration: Optional[MetaverseIntegration] = None

def get_metaverse_integration() -> MetaverseIntegration:
    """Get the global metaverse integration"""
    global _metaverse_integration
    if _metaverse_integration is None:
        _metaverse_integration = MetaverseIntegration()
    return _metaverse_integration

# Metaverse router
metaverse_router = APIRouter(prefix="/metaverse", tags=["Metaverse Integration"])

@metaverse_router.post("/create-space")
async def create_virtual_space_endpoint(
    name: str = Field(..., description="Virtual space name"),
    platform: MetaversePlatform = Field(..., description="Metaverse platform"),
    description: str = Field(..., description="Space description"),
    capacity: int = Field(..., description="Maximum capacity"),
    environment: Dict[str, Any] = Field(..., description="Environment settings"),
    permissions: List[str] = Field(..., description="Space permissions")
):
    """Create a new virtual space"""
    try:
        integration = get_metaverse_integration()
        space = await integration.create_virtual_space(
            name, platform, description, capacity, environment, permissions
        )
        return {"space": asdict(space), "success": True}
    
    except Exception as e:
        logger.error(f"Error creating virtual space: {e}")
        raise HTTPException(status_code=500, detail="Failed to create virtual space")

@metaverse_router.post("/create-avatar")
async def create_avatar_endpoint(
    user_id: str = Field(..., description="User ID"),
    name: str = Field(..., description="Avatar name"),
    avatar_type: AvatarType = Field(..., description="Avatar type"),
    appearance: Dict[str, Any] = Field(..., description="Avatar appearance")
):
    """Create a virtual avatar"""
    try:
        integration = get_metaverse_integration()
        avatar = await integration.create_avatar(user_id, name, avatar_type, appearance)
        return {"avatar": asdict(avatar), "success": True}
    
    except Exception as e:
        logger.error(f"Error creating avatar: {e}")
        raise HTTPException(status_code=500, detail="Failed to create avatar")

@metaverse_router.post("/create-document")
async def create_virtual_document_endpoint(
    document_content: str = Field(..., description="Document content"),
    document_metadata: Dict[str, Any] = Field(..., description="Document metadata"),
    position: Dict[str, float] = Field(..., description="3D position"),
    space_id: str = Field(..., description="Virtual space ID")
):
    """Create a virtual document object"""
    try:
        integration = get_metaverse_integration()
        virtual_doc = await integration.create_virtual_document(
            document_content, document_metadata, position, space_id
        )
        return {"virtual_document": asdict(virtual_doc), "success": True}
    
    except Exception as e:
        logger.error(f"Error creating virtual document: {e}")
        raise HTTPException(status_code=500, detail="Failed to create virtual document")

@metaverse_router.post("/join-space")
async def join_virtual_space_endpoint(
    space_id: str = Field(..., description="Virtual space ID"),
    user_id: str = Field(..., description="User ID"),
    avatar_id: str = Field(..., description="Avatar ID")
):
    """Join a virtual space"""
    try:
        integration = get_metaverse_integration()
        result = await integration.join_virtual_space(space_id, user_id, avatar_id)
        return result
    
    except Exception as e:
        logger.error(f"Error joining virtual space: {e}")
        raise HTTPException(status_code=500, detail="Failed to join virtual space")

@metaverse_router.post("/interact")
async def interact_with_object_endpoint(
    session_id: str = Field(..., description="Session ID"),
    object_id: str = Field(..., description="Object ID"),
    interaction_type: InteractionType = Field(..., description="Interaction type"),
    interaction_data: Dict[str, Any] = Field(..., description="Interaction data")
):
    """Interact with a virtual object"""
    try:
        integration = get_metaverse_integration()
        result = await integration.interact_with_object(
            session_id, object_id, interaction_type, interaction_data
        )
        return result
    
    except Exception as e:
        logger.error(f"Error interacting with object: {e}")
        raise HTTPException(status_code=500, detail="Failed to interact with object")

@metaverse_router.get("/status")
async def get_metaverse_status_endpoint():
    """Get metaverse integration status"""
    try:
        integration = get_metaverse_integration()
        status = await integration.get_metaverse_status()
        return {"status": status}
    
    except Exception as e:
        logger.error(f"Error getting metaverse status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get metaverse status")

@metaverse_router.get("/spaces")
async def get_virtual_spaces_endpoint():
    """Get all virtual spaces"""
    try:
        integration = get_metaverse_integration()
        spaces = [asdict(space) for space in integration.virtual_spaces.values()]
        return {"spaces": spaces, "count": len(spaces)}
    
    except Exception as e:
        logger.error(f"Error getting virtual spaces: {e}")
        raise HTTPException(status_code=500, detail="Failed to get virtual spaces")

@metaverse_router.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time metaverse communication"""
    try:
        await websocket.accept()
        integration = get_metaverse_integration()
        integration.websocket_connections[session_id] = websocket
        
        try:
            while True:
                # Receive message from client
                data = await websocket.receive_text()
                message = json.loads(data)
                
                # Process message
                await integration._process_websocket_message(session_id, message)
                
        except WebSocketDisconnect:
            # Clean up connection
            if session_id in integration.websocket_connections:
                del integration.websocket_connections[session_id]
    
    except Exception as e:
        logger.error(f"Error in WebSocket connection: {e}")
        await websocket.close()


