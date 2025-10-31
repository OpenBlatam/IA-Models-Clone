#!/usr/bin/env python3
"""
👤 HeyGen AI - Advanced Avatar Management System
===============================================

This module implements a comprehensive avatar management system that provides
avatar creation, customization, animation, and management capabilities
for the HeyGen AI system.
"""

import asyncio
import logging
import time
import json
import uuid
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
import threading
import queue
import hashlib
import secrets
import base64
import hmac
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend
import aiohttp
import asyncio
from aiohttp import web, WSMsgType
import ssl
import certifi
import cv2
import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import os
import sys
import threading
import queue
import asyncio
from concurrent.futures import ThreadPoolExecutor
import random
import string
import re
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AvatarType(str, Enum):
    """Avatar types"""
    HUMAN = "human"
    CARTOON = "cartoon"
    ANIME = "anime"
    REALISTIC = "realistic"
    ABSTRACT = "abstract"
    ANIMAL = "animal"
    ROBOT = "robot"
    FANTASY = "fantasy"

class AvatarStyle(str, Enum):
    """Avatar styles"""
    PROFESSIONAL = "professional"
    CASUAL = "casual"
    FORMAL = "formal"
    CREATIVE = "creative"
    MODERN = "modern"
    VINTAGE = "vintage"
    MINIMALIST = "minimalist"
    VIBRANT = "vibrant"

class AnimationType(str, Enum):
    """Animation types"""
    IDLE = "idle"
    TALKING = "talking"
    GESTURING = "gesturing"
    WALKING = "walking"
    WAVING = "waving"
    NODDING = "nodding"
    SHAKING_HEAD = "shaking_head"
    POINTING = "pointing"

@dataclass
class Avatar:
    """Avatar representation"""
    avatar_id: str
    name: str
    avatar_type: AvatarType
    style: AvatarStyle
    gender: str = "neutral"
    age_range: str = "adult"
    appearance: Dict[str, Any] = field(default_factory=dict)
    animations: List[AnimationType] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AvatarRequest:
    """Avatar generation request"""
    request_id: str
    avatar_type: AvatarType
    style: AvatarStyle
    gender: str = "neutral"
    age_range: str = "adult"
    appearance_preferences: Dict[str, Any] = field(default_factory=dict)
    animation_requirements: List[AnimationType] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

class AvatarGenerator:
    """Advanced avatar generation system"""
    
    def __init__(self):
        self.initialized = False
    
    async def initialize(self):
        """Initialize avatar generator"""
        self.initialized = True
        logger.info("✅ Avatar Generator initialized")
    
    async def generate_avatar(self, request: AvatarRequest) -> Optional[Avatar]:
        """Generate avatar from request"""
        if not self.initialized:
            return None
        
        try:
            # Create avatar ID
            avatar_id = str(uuid.uuid4())
            
            # Generate avatar appearance
            appearance = await self._generate_appearance(request)
            
            # Create avatar object
            avatar = Avatar(
                avatar_id=avatar_id,
                name=f"Avatar_{avatar_id[:8]}",
                avatar_type=request.avatar_type,
                style=request.style,
                gender=request.gender,
                age_range=request.age_range,
                appearance=appearance,
                animations=request.animation_requirements,
                metadata=request.metadata.copy()
            )
            
            logger.info(f"✅ Avatar generated: {avatar_id}")
            return avatar
            
        except Exception as e:
            logger.error(f"❌ Avatar generation failed: {e}")
            return None
    
    async def _generate_appearance(self, request: AvatarRequest) -> Dict[str, Any]:
        """Generate avatar appearance"""
        try:
            appearance = {
                'skin_tone': self._generate_skin_tone(request),
                'hair_color': self._generate_hair_color(request),
                'hair_style': self._generate_hair_style(request),
                'eye_color': self._generate_eye_color(request),
                'facial_features': self._generate_facial_features(request),
                'body_type': self._generate_body_type(request),
                'clothing_style': self._generate_clothing_style(request),
                'accessories': self._generate_accessories(request)
            }
            
            return appearance
            
        except Exception as e:
            logger.error(f"❌ Appearance generation failed: {e}")
            return {}
    
    def _generate_skin_tone(self, request: AvatarRequest) -> str:
        """Generate skin tone"""
        skin_tones = ['light', 'medium', 'dark', 'olive', 'tan']
        return random.choice(skin_tones)
    
    def _generate_hair_color(self, request: AvatarRequest) -> str:
        """Generate hair color"""
        hair_colors = ['black', 'brown', 'blonde', 'red', 'gray', 'white', 'blue', 'green', 'purple']
        return random.choice(hair_colors)
    
    def _generate_hair_style(self, request: AvatarRequest) -> str:
        """Generate hair style"""
        hair_styles = ['short', 'long', 'curly', 'straight', 'wavy', 'bald', 'ponytail', 'bun']
        return random.choice(hair_styles)
    
    def _generate_eye_color(self, request: AvatarRequest) -> str:
        """Generate eye color"""
        eye_colors = ['brown', 'blue', 'green', 'hazel', 'gray', 'amber']
        return random.choice(eye_colors)
    
    def _generate_facial_features(self, request: AvatarRequest) -> Dict[str, Any]:
        """Generate facial features"""
        return {
            'nose_shape': random.choice(['small', 'medium', 'large', 'pointed', 'rounded']),
            'lip_shape': random.choice(['thin', 'medium', 'full', 'wide']),
            'eyebrow_shape': random.choice(['straight', 'arched', 'thick', 'thin']),
            'jaw_shape': random.choice(['square', 'round', 'oval', 'pointed'])
        }
    
    def _generate_body_type(self, request: AvatarRequest) -> str:
        """Generate body type"""
        body_types = ['slim', 'average', 'athletic', 'curvy', 'muscular']
        return random.choice(body_types)
    
    def _generate_clothing_style(self, request: AvatarRequest) -> str:
        """Generate clothing style"""
        clothing_styles = ['casual', 'formal', 'business', 'sporty', 'elegant', 'trendy']
        return random.choice(clothing_styles)
    
    def _generate_accessories(self, request: AvatarRequest) -> List[str]:
        """Generate accessories"""
        accessories = ['glasses', 'hat', 'jewelry', 'watch', 'scarf', 'tie']
        return random.sample(accessories, random.randint(0, 3))

class AvatarAnimator:
    """Advanced avatar animation system"""
    
    def __init__(self):
        self.initialized = False
    
    async def initialize(self):
        """Initialize avatar animator"""
        self.initialized = True
        logger.info("✅ Avatar Animator initialized")
    
    async def create_animation(self, avatar: Avatar, animation_type: AnimationType, 
                             duration: float = 2.0) -> str:
        """Create avatar animation"""
        if not self.initialized:
            return ""
        
        try:
            # Create animation file path
            animation_id = str(uuid.uuid4())
            animation_path = f"avatar_animation_{animation_id}.mp4"
            
            # Generate animation based on type
            if animation_type == AnimationType.IDLE:
                animation_frames = await self._create_idle_animation(avatar, duration)
            elif animation_type == AnimationType.TALKING:
                animation_frames = await self._create_talking_animation(avatar, duration)
            elif animation_type == AnimationType.GESTURING:
                animation_frames = await self._create_gesturing_animation(avatar, duration)
            elif animation_type == AnimationType.WALKING:
                animation_frames = await self._create_walking_animation(avatar, duration)
            elif animation_type == AnimationType.WAVING:
                animation_frames = await self._create_waving_animation(avatar, duration)
            elif animation_type == AnimationType.NODDING:
                animation_frames = await self._create_nodding_animation(avatar, duration)
            elif animation_type == AnimationType.SHAKING_HEAD:
                animation_frames = await self._create_shaking_head_animation(avatar, duration)
            elif animation_type == AnimationType.POINTING:
                animation_frames = await self._create_pointing_animation(avatar, duration)
            else:
                animation_frames = await self._create_idle_animation(avatar, duration)
            
            # Save animation as video
            await self._save_animation_frames(animation_frames, animation_path, 30)
            
            logger.info(f"✅ Animation created: {animation_type.value}")
            return animation_path
            
        except Exception as e:
            logger.error(f"❌ Animation creation failed: {e}")
            return ""
    
    async def _create_idle_animation(self, avatar: Avatar, duration: float) -> List[np.ndarray]:
        """Create idle animation frames"""
        frames = []
        fps = 30
        total_frames = int(duration * fps)
        
        for i in range(total_frames):
            # Create base avatar image
            frame = await self._create_avatar_frame(avatar)
            
            # Add subtle breathing animation
            breath_offset = int(2 * np.sin(2 * np.pi * i / fps))
            frame = np.roll(frame, breath_offset, axis=0)
            
            frames.append(frame)
        
        return frames
    
    async def _create_talking_animation(self, avatar: Avatar, duration: float) -> List[np.ndarray]:
        """Create talking animation frames"""
        frames = []
        fps = 30
        total_frames = int(duration * fps)
        
        for i in range(total_frames):
            # Create base avatar image
            frame = await self._create_avatar_frame(avatar)
            
            # Add mouth movement
            mouth_open = int(5 * np.sin(4 * np.pi * i / fps))
            frame = self._modify_mouth(frame, mouth_open)
            
            frames.append(frame)
        
        return frames
    
    async def _create_gesturing_animation(self, avatar: Avatar, duration: float) -> List[np.ndarray]:
        """Create gesturing animation frames"""
        frames = []
        fps = 30
        total_frames = int(duration * fps)
        
        for i in range(total_frames):
            # Create base avatar image
            frame = await self._create_avatar_frame(avatar)
            
            # Add arm movement
            arm_angle = 30 * np.sin(2 * np.pi * i / fps)
            frame = self._modify_arm_position(frame, arm_angle)
            
            frames.append(frame)
        
        return frames
    
    async def _create_walking_animation(self, avatar: Avatar, duration: float) -> List[np.ndarray]:
        """Create walking animation frames"""
        frames = []
        fps = 30
        total_frames = int(duration * fps)
        
        for i in range(total_frames):
            # Create base avatar image
            frame = await self._create_avatar_frame(avatar)
            
            # Add walking movement
            walk_offset = int(10 * np.sin(4 * np.pi * i / fps))
            frame = np.roll(frame, walk_offset, axis=1)
            
            frames.append(frame)
        
        return frames
    
    async def _create_waving_animation(self, avatar: Avatar, duration: float) -> List[np.ndarray]:
        """Create waving animation frames"""
        frames = []
        fps = 30
        total_frames = int(duration * fps)
        
        for i in range(total_frames):
            # Create base avatar image
            frame = await self._create_avatar_frame(avatar)
            
            # Add waving motion
            wave_angle = 45 * np.sin(4 * np.pi * i / fps)
            frame = self._modify_arm_position(frame, wave_angle)
            
            frames.append(frame)
        
        return frames
    
    async def _create_nodding_animation(self, avatar: Avatar, duration: float) -> List[np.ndarray]:
        """Create nodding animation frames"""
        frames = []
        fps = 30
        total_frames = int(duration * fps)
        
        for i in range(total_frames):
            # Create base avatar image
            frame = await self._create_avatar_frame(avatar)
            
            # Add nodding motion
            nod_offset = int(5 * np.sin(4 * np.pi * i / fps))
            frame = np.roll(frame, nod_offset, axis=0)
            
            frames.append(frame)
        
        return frames
    
    async def _create_shaking_head_animation(self, avatar: Avatar, duration: float) -> List[np.ndarray]:
        """Create shaking head animation frames"""
        frames = []
        fps = 30
        total_frames = int(duration * fps)
        
        for i in range(total_frames):
            # Create base avatar image
            frame = await self._create_avatar_frame(avatar)
            
            # Add head shaking motion
            shake_angle = 10 * np.sin(8 * np.pi * i / fps)
            frame = self._rotate_head(frame, shake_angle)
            
            frames.append(frame)
        
        return frames
    
    async def _create_pointing_animation(self, avatar: Avatar, duration: float) -> List[np.ndarray]:
        """Create pointing animation frames"""
        frames = []
        fps = 30
        total_frames = int(duration * fps)
        
        for i in range(total_frames):
            # Create base avatar image
            frame = await self._create_avatar_frame(avatar)
            
            # Add pointing motion
            point_angle = 30 * np.sin(2 * np.pi * i / fps)
            frame = self._modify_arm_position(frame, point_angle)
            
            frames.append(frame)
        
        return frames
    
    async def _create_avatar_frame(self, avatar: Avatar) -> np.ndarray:
        """Create base avatar frame"""
        # Create placeholder avatar image
        width, height = 400, 600
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Apply avatar appearance
        skin_color = self._get_skin_color(avatar.appearance.get('skin_tone', 'medium'))
        frame[:] = skin_color
        
        # Add basic avatar features
        self._add_face_features(frame, avatar)
        self._add_hair(frame, avatar)
        self._add_clothing(frame, avatar)
        
        return frame
    
    def _get_skin_color(self, skin_tone: str) -> Tuple[int, int, int]:
        """Get skin color based on tone"""
        skin_colors = {
            'light': (255, 220, 177),
            'medium': (205, 170, 125),
            'dark': (139, 90, 43),
            'olive': (198, 142, 72),
            'tan': (222, 184, 135)
        }
        return skin_colors.get(skin_tone, (205, 170, 125))
    
    def _add_face_features(self, frame: np.ndarray, avatar: Avatar):
        """Add face features to avatar"""
        height, width = frame.shape[:2]
        
        # Add eyes
        eye_color = self._get_eye_color(avatar.appearance.get('eye_color', 'brown'))
        cv2.circle(frame, (width//2 - 30, height//2 - 50), 15, eye_color, -1)
        cv2.circle(frame, (width//2 + 30, height//2 - 50), 15, eye_color, -1)
        
        # Add nose
        cv2.circle(frame, (width//2, height//2), 8, (200, 150, 100), -1)
        
        # Add mouth
        cv2.ellipse(frame, (width//2, height//2 + 30), (20, 10), 0, 0, 180, (150, 50, 50), -1)
    
    def _add_hair(self, frame: np.ndarray, avatar: Avatar):
        """Add hair to avatar"""
        height, width = frame.shape[:2]
        hair_color = self._get_hair_color(avatar.appearance.get('hair_color', 'brown'))
        
        # Add hair
        cv2.ellipse(frame, (width//2, height//2 - 80), (60, 40), 0, 0, 180, hair_color, -1)
    
    def _add_clothing(self, frame: np.ndarray, avatar: Avatar):
        """Add clothing to avatar"""
        height, width = frame.shape[:2]
        clothing_color = (100, 150, 200)  # Blue shirt
        
        # Add shirt
        cv2.rectangle(frame, (width//2 - 40, height//2 + 50), (width//2 + 40, height - 50), clothing_color, -1)
    
    def _get_eye_color(self, eye_color: str) -> Tuple[int, int, int]:
        """Get eye color"""
        eye_colors = {
            'brown': (101, 67, 33),
            'blue': (0, 100, 200),
            'green': (0, 150, 0),
            'hazel': (139, 69, 19),
            'gray': (128, 128, 128),
            'amber': (255, 191, 0)
        }
        return eye_colors.get(eye_color, (101, 67, 33))
    
    def _get_hair_color(self, hair_color: str) -> Tuple[int, int, int]:
        """Get hair color"""
        hair_colors = {
            'black': (0, 0, 0),
            'brown': (101, 67, 33),
            'blonde': (255, 215, 0),
            'red': (255, 0, 0),
            'gray': (128, 128, 128),
            'white': (255, 255, 255),
            'blue': (0, 0, 255),
            'green': (0, 255, 0),
            'purple': (128, 0, 128)
        }
        return hair_colors.get(hair_color, (101, 67, 33))
    
    def _modify_mouth(self, frame: np.ndarray, mouth_open: int) -> np.ndarray:
        """Modify mouth opening"""
        height, width = frame.shape[:2]
        
        # Modify mouth based on opening
        mouth_height = max(5, 10 + mouth_open)
        cv2.ellipse(frame, (width//2, height//2 + 30), (20, mouth_height), 0, 0, 180, (150, 50, 50), -1)
        
        return frame
    
    def _modify_arm_position(self, frame: np.ndarray, angle: float) -> np.ndarray:
        """Modify arm position"""
        # Simple arm movement simulation
        height, width = frame.shape[:2]
        
        # Add arm line
        arm_x = int(width//2 + 30 + angle)
        arm_y = int(height//2 + 50)
        cv2.line(frame, (width//2 + 30, height//2 + 50), (arm_x, arm_y), (200, 150, 100), 5)
        
        return frame
    
    def _rotate_head(self, frame: np.ndarray, angle: float) -> np.ndarray:
        """Rotate head"""
        height, width = frame.shape[:2]
        center = (width//2, height//2)
        
        # Create rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Apply rotation
        rotated = cv2.warpAffine(frame, rotation_matrix, (width, height))
        
        return rotated
    
    async def _save_animation_frames(self, frames: List[np.ndarray], output_path: str, fps: int):
        """Save animation frames as video"""
        if not frames:
            return
        
        height, width = frames[0].shape[:2]
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Write frames
        for frame in frames:
            out.write(frame)
        
        out.release()

class AdvancedAvatarManagementSystem:
    """Main avatar management system"""
    
    def __init__(self):
        self.avatar_generator = AvatarGenerator()
        self.avatar_animator = AvatarAnimator()
        self.avatars: Dict[str, Avatar] = {}
        self.initialized = False
    
    async def initialize(self):
        """Initialize avatar management system"""
        try:
            logger.info("👤 Initializing Advanced Avatar Management System...")
            
            # Initialize components
            await self.avatar_generator.initialize()
            await self.avatar_animator.initialize()
            
            self.initialized = True
            logger.info("✅ Advanced Avatar Management System initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize avatar management system: {e}")
            raise
    
    async def create_avatar(self, request: AvatarRequest) -> Optional[Avatar]:
        """Create avatar from request"""
        if not self.initialized:
            return None
        
        try:
            avatar = await self.avatar_generator.generate_avatar(request)
            
            if avatar:
                # Store avatar
                self.avatars[avatar.avatar_id] = avatar
                
                logger.info(f"✅ Avatar created: {avatar.avatar_id}")
            
            return avatar
            
        except Exception as e:
            logger.error(f"❌ Avatar creation failed: {e}")
            return None
    
    async def create_animation(self, avatar_id: str, animation_type: AnimationType, 
                             duration: float = 2.0) -> Optional[str]:
        """Create animation for avatar"""
        if not self.initialized or avatar_id not in self.avatars:
            return None
        
        try:
            avatar = self.avatars[avatar_id]
            animation_path = await self.avatar_animator.create_animation(
                avatar, animation_type, duration
            )
            
            logger.info(f"✅ Animation created for avatar: {avatar_id}")
            return animation_path
            
        except Exception as e:
            logger.error(f"❌ Animation creation failed: {e}")
            return None
    
    async def get_avatar(self, avatar_id: str) -> Optional[Avatar]:
        """Get avatar by ID"""
        return self.avatars.get(avatar_id)
    
    async def list_avatars(self, avatar_type: AvatarType = None) -> List[Avatar]:
        """List avatars with optional filtering"""
        avatars = list(self.avatars.values())
        
        if avatar_type:
            avatars = [a for a in avatars if a.avatar_type == avatar_type]
        
        return avatars
    
    async def update_avatar(self, avatar_id: str, updates: Dict[str, Any]) -> bool:
        """Update avatar properties"""
        if not self.initialized or avatar_id not in self.avatars:
            return False
        
        try:
            avatar = self.avatars[avatar_id]
            
            # Update properties
            for key, value in updates.items():
                if hasattr(avatar, key):
                    setattr(avatar, key, value)
            
            avatar.updated_at = datetime.now()
            
            logger.info(f"✅ Avatar updated: {avatar_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Avatar update failed: {e}")
            return False
    
    async def delete_avatar(self, avatar_id: str) -> bool:
        """Delete avatar"""
        if not self.initialized or avatar_id not in self.avatars:
            return False
        
        try:
            del self.avatars[avatar_id]
            
            logger.info(f"✅ Avatar deleted: {avatar_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Avatar deletion failed: {e}")
            return False
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get system status"""
        return {
            'initialized': self.initialized,
            'avatar_generator_ready': self.avatar_generator.initialized,
            'avatar_animator_ready': self.avatar_animator.initialized,
            'total_avatars': len(self.avatars),
            'timestamp': datetime.now().isoformat()
        }
    
    async def shutdown(self):
        """Shutdown avatar management system"""
        self.initialized = False
        logger.info("✅ Advanced Avatar Management System shutdown complete")

# Example usage and demonstration
async def main():
    """Demonstrate the advanced avatar management system"""
    print("👤 HeyGen AI - Advanced Avatar Management System Demo")
    print("=" * 70)
    
    # Initialize system
    avatar_system = AdvancedAvatarManagementSystem()
    
    try:
        # Initialize the system
        print("\n🚀 Initializing Advanced Avatar Management System...")
        await avatar_system.initialize()
        print("✅ Advanced Avatar Management System initialized successfully")
        
        # Get system status
        print("\n📊 System Status:")
        status = await avatar_system.get_system_status()
        for key, value in status.items():
            print(f"  {key}: {value}")
        
        # Create different types of avatars
        print("\n👤 Creating Different Types of Avatars...")
        
        # Human avatar
        human_request = AvatarRequest(
            request_id="human_001",
            avatar_type=AvatarType.HUMAN,
            style=AvatarStyle.PROFESSIONAL,
            gender="male",
            age_range="adult",
            appearance_preferences={
                'skin_tone': 'medium',
                'hair_color': 'brown',
                'eye_color': 'blue'
            },
            animation_requirements=[AnimationType.TALKING, AnimationType.GESTURING]
        )
        
        human_avatar = await avatar_system.create_avatar(human_request)
        if human_avatar:
            print(f"  ✅ Human Avatar: {human_avatar.avatar_id}")
            print(f"    Type: {human_avatar.avatar_type.value}")
            print(f"    Style: {human_avatar.style.value}")
            print(f"    Gender: {human_avatar.gender}")
            print(f"    Age Range: {human_avatar.age_range}")
            print(f"    Appearance: {human_avatar.appearance}")
        
        # Cartoon avatar
        cartoon_request = AvatarRequest(
            request_id="cartoon_001",
            avatar_type=AvatarType.CARTOON,
            style=AvatarStyle.CREATIVE,
            gender="female",
            age_range="young_adult",
            appearance_preferences={
                'hair_color': 'pink',
                'eye_color': 'green'
            },
            animation_requirements=[AnimationType.WAVING, AnimationType.NODDING]
        )
        
        cartoon_avatar = await avatar_system.create_avatar(cartoon_request)
        if cartoon_avatar:
            print(f"  ✅ Cartoon Avatar: {cartoon_avatar.avatar_id}")
            print(f"    Type: {cartoon_avatar.avatar_type.value}")
            print(f"    Style: {cartoon_avatar.style.value}")
            print(f"    Gender: {cartoon_avatar.gender}")
            print(f"    Age Range: {cartoon_avatar.age_range}")
            print(f"    Appearance: {cartoon_avatar.appearance}")
        
        # Robot avatar
        robot_request = AvatarRequest(
            request_id="robot_001",
            avatar_type=AvatarType.ROBOT,
            style=AvatarStyle.MODERN,
            gender="neutral",
            age_range="adult",
            appearance_preferences={
                'skin_tone': 'metallic',
                'eye_color': 'red'
            },
            animation_requirements=[AnimationType.POINTING, AnimationType.SHAKING_HEAD]
        )
        
        robot_avatar = await avatar_system.create_avatar(robot_request)
        if robot_avatar:
            print(f"  ✅ Robot Avatar: {robot_avatar.avatar_id}")
            print(f"    Type: {robot_avatar.avatar_type.value}")
            print(f"    Style: {robot_avatar.style.value}")
            print(f"    Gender: {robot_avatar.gender}")
            print(f"    Age Range: {robot_avatar.age_range}")
            print(f"    Appearance: {robot_avatar.appearance}")
        
        # Create animations for avatars
        print("\n🎬 Creating Animations for Avatars...")
        
        if human_avatar:
            # Create talking animation
            talking_animation = await avatar_system.create_animation(
                human_avatar.avatar_id, AnimationType.TALKING, 3.0
            )
            if talking_animation:
                print(f"  ✅ Talking Animation: {talking_animation}")
            
            # Create gesturing animation
            gesturing_animation = await avatar_system.create_animation(
                human_avatar.avatar_id, AnimationType.GESTURING, 2.0
            )
            if gesturing_animation:
                print(f"  ✅ Gesturing Animation: {gesturing_animation}")
        
        if cartoon_avatar:
            # Create waving animation
            waving_animation = await avatar_system.create_animation(
                cartoon_avatar.avatar_id, AnimationType.WAVING, 2.0
            )
            if waving_animation:
                print(f"  ✅ Waving Animation: {waving_animation}")
            
            # Create nodding animation
            nodding_animation = await avatar_system.create_animation(
                cartoon_avatar.avatar_id, AnimationType.NODDING, 2.0
            )
            if nodding_animation:
                print(f"  ✅ Nodding Animation: {nodding_animation}")
        
        if robot_avatar:
            # Create pointing animation
            pointing_animation = await avatar_system.create_animation(
                robot_avatar.avatar_id, AnimationType.POINTING, 2.0
            )
            if pointing_animation:
                print(f"  ✅ Pointing Animation: {pointing_animation}")
            
            # Create shaking head animation
            shaking_animation = await avatar_system.create_animation(
                robot_avatar.avatar_id, AnimationType.SHAKING_HEAD, 2.0
            )
            if shaking_animation:
                print(f"  ✅ Shaking Head Animation: {shaking_animation}")
        
        # Test avatar updates
        print("\n🔄 Testing Avatar Updates...")
        
        if human_avatar:
            update_success = await avatar_system.update_avatar(
                human_avatar.avatar_id,
                {'name': 'Updated Human Avatar', 'gender': 'female'}
            )
            if update_success:
                print(f"  ✅ Avatar updated: {human_avatar.avatar_id}")
        
        # List all avatars
        print("\n📋 Avatar Summary:")
        all_avatars = await avatar_system.list_avatars()
        
        print(f"  Total Avatars Created: {len(all_avatars)}")
        
        for avatar in all_avatars:
            print(f"    {avatar.name} ({avatar.avatar_type.value}): {avatar.avatar_id}")
            print(f"      Style: {avatar.style.value}")
            print(f"      Gender: {avatar.gender}")
            print(f"      Age Range: {avatar.age_range}")
            print(f"      Animations: {[a.value for a in avatar.animations]}")
        
        # List avatars by type
        print("\n📊 Avatars by Type:")
        for avatar_type in AvatarType:
            type_avatars = await avatar_system.list_avatars(avatar_type)
            if type_avatars:
                print(f"  {avatar_type.value}: {len(type_avatars)} avatars")
        
    except Exception as e:
        print(f"❌ Demo Error: {e}")
        logger.error(f"Demo failed: {e}")
    
    finally:
        # Shutdown
        await avatar_system.shutdown()
        print("\n✅ Demo completed")

if __name__ == "__main__":
    asyncio.run(main())


