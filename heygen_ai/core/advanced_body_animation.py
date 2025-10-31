#!/usr/bin/env python3
"""
Advanced Body Animation Service
==============================

Provides sophisticated body animation capabilities including:
- Natural gesture generation based on script analysis
- Dynamic posture control and body language
- Hand and arm movement coordination
- Full-body animation sequences
"""

import asyncio
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import json
import time

logger = logging.getLogger(__name__)

class GestureType(Enum):
    """Types of gestures that can be generated."""
    POINTING = "pointing"
    WAVING = "waving"
    COUNTING = "counting"
    EMPHASIS = "emphasis"
    GREETING = "greeting"
    THINKING = "thinking"
    EXPLANATION = "explanation"
    AGREEMENT = "agreement"
    DISAGREEMENT = "disagreement"
    EXCITEMENT = "excitement"
    CALM = "calm"
    CONFIDENCE = "confidence"
    UNCERTAINTY = "uncertainty"

class BodyPart(Enum):
    """Body parts that can be animated."""
    HEAD = "head"
    NECK = "neck"
    SHOULDERS = "shoulders"
    ARMS = "arms"
    HANDS = "hands"
    TORSO = "torso"
    HIPS = "hips"
    LEGS = "legs"

@dataclass
class GestureConfig:
    """Configuration for gesture generation."""
    gesture_type: GestureType
    intensity: float = 1.0  # 0.0 to 2.0
    duration: float = 2.0   # seconds
    body_parts: List[BodyPart] = field(default_factory=list)
    emotion_context: Optional[str] = None
    script_context: Optional[str] = None

@dataclass
class BodyPose:
    """Represents a body pose configuration."""
    timestamp: float
    head_rotation: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # pitch, yaw, roll
    neck_rotation: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    shoulder_position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    arm_rotation: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    hand_position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    torso_rotation: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    hip_position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    weight_shift: float = 0.0  # -1.0 (left) to 1.0 (right)

@dataclass
class AnimationSequence:
    """Complete animation sequence for body movement."""
    sequence_id: str
    poses: List[BodyPose]
    total_duration: float
    easing_type: str = "ease_in_out"
    loop: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

class AdvancedBodyAnimationService:
    """
    Advanced service for generating natural body animations and gestures.
    """
    
    def __init__(self):
        self.gesture_patterns = self._initialize_gesture_patterns()
        self.emotion_gesture_mapping = self._initialize_emotion_mapping()
        self.script_analysis_cache = {}
        self.animation_cache = {}
        
    def _initialize_gesture_patterns(self) -> Dict[GestureType, Dict[str, Any]]:
        """Initialize predefined gesture patterns."""
        return {
            GestureType.POINTING: {
                "arm_extension": 0.8,
                "hand_shape": "point",
                "head_alignment": 0.6,
                "torso_rotation": 0.3
            },
            GestureType.WAVING: {
                "arm_swing": 0.7,
                "hand_rotation": 0.9,
                "head_tilt": 0.4,
                "body_sway": 0.2
            },
            GestureType.EMPHASIS: {
                "arm_gesture": 0.9,
                "hand_clench": 0.8,
                "torso_lean": 0.5,
                "head_nod": 0.6
            },
            GestureType.THINKING: {
                "hand_chin": 0.7,
                "head_tilt": 0.8,
                "torso_lean": 0.4,
                "arm_rest": 0.6
            }
        }
    
    def _initialize_emotion_mapping(self) -> Dict[str, Dict[str, Any]]:
        """Initialize emotion to gesture mapping."""
        return {
            "excited": {
                "gesture_intensity": 1.5,
                "movement_speed": 1.3,
                "body_openness": 0.8,
                "preferred_gestures": [GestureType.EMPHASIS, GestureType.WAVING]
            },
            "calm": {
                "gesture_intensity": 0.7,
                "movement_speed": 0.8,
                "body_openness": 0.5,
                "preferred_gestures": [GestureType.THINKING, GestureType.EXPLANATION]
            },
            "confident": {
                "gesture_intensity": 1.2,
                "movement_speed": 1.0,
                "body_openness": 0.9,
                "preferred_gestures": [GestureType.POINTING, GestureType.EMPHASIS]
            },
            "uncertain": {
                "gesture_intensity": 0.5,
                "movement_speed": 0.6,
                "body_openness": 0.3,
                "preferred_gestures": [GestureType.THINKING, GestureType.CALM]
            }
        }
    
    async def analyze_script_for_gestures(self, script_text: str) -> List[GestureConfig]:
        """
        Analyze script text to determine appropriate gestures.
        
        Args:
            script_text: The script text to analyze
            
        Returns:
            List of gesture configurations with timing
        """
        try:
            # Check cache first
            cache_key = hash(script_text)
            if cache_key in self.script_analysis_cache:
                return self.script_analysis_cache[cache_key]
            
            gestures = []
            
            # Simple keyword-based analysis (can be enhanced with NLP)
            sentences = script_text.split('.')
            current_time = 0.0
            
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                
                # Analyze sentence content for gesture opportunities
                gesture_config = self._analyze_sentence_for_gesture(sentence)
                if gesture_config:
                    gesture_config.duration = len(sentence.split()) * 0.3  # Rough timing
                    gestures.append(gesture_config)
                
                current_time += gesture_config.duration if gesture_config else 1.0
            
            # Cache the result
            self.script_analysis_cache[cache_key] = gestures
            
            logger.info(f"Generated {len(gestures)} gestures for script analysis")
            return gestures
            
        except Exception as e:
            logger.error(f"Error analyzing script for gestures: {e}")
            return []
    
    def _analyze_sentence_for_gesture(self, sentence: str) -> Optional[GestureConfig]:
        """Analyze a single sentence to determine appropriate gesture."""
        sentence_lower = sentence.lower()
        
        # Question marks indicate thinking gestures
        if '?' in sentence:
            return GestureConfig(
                gesture_type=GestureType.THINKING,
                intensity=0.8,
                body_parts=[BodyPart.HEAD, BodyPart.HANDS, BodyPart.TORSO]
            )
        
        # Exclamation marks indicate emphasis
        if '!' in sentence:
            return GestureConfig(
                gesture_type=GestureType.EMPHASIS,
                intensity=1.2,
                body_parts=[BodyPart.ARMS, BodyPart.HANDS, BodyPart.TORSO]
            )
        
        # Keywords for specific gestures
        if any(word in sentence_lower for word in ['here', 'there', 'this', 'that']):
            return GestureConfig(
                gesture_type=GestureType.POINTING,
                intensity=1.0,
                body_parts=[BodyPart.ARMS, BodyPart.HANDS, BodyPart.HEAD]
            )
        
        if any(word in sentence_lower for word in ['hello', 'hi', 'welcome', 'goodbye']):
            return GestureConfig(
                gesture_type=GestureType.GREETING,
                intensity=0.9,
                body_parts=[BodyPart.ARMS, BodyPart.HANDS, BodyPart.HEAD]
            )
        
        if any(word in sentence_lower for word in ['first', 'second', 'third', '1', '2', '3']):
            return GestureConfig(
                gesture_type=GestureType.COUNTING,
                intensity=0.8,
                body_parts=[BodyPart.HANDS, BodyPart.ARMS]
            )
        
        # Default to explanation gesture
        return GestureConfig(
            gesture_type=GestureType.EXPLANATION,
            intensity=0.7,
            body_parts=[BodyPart.ARMS, BodyPart.HANDS]
        )
    
    async def generate_body_animation_sequence(
        self, 
        gesture_configs: List[GestureConfig],
        base_emotion: str = "neutral",
        avatar_style: str = "realistic"
    ) -> AnimationSequence:
        """
        Generate a complete body animation sequence from gesture configurations.
        
        Args:
            gesture_configs: List of gesture configurations
            base_emotion: Base emotion for the avatar
            avatar_style: Style of the avatar (realistic, cartoon, etc.)
            
        Returns:
            Complete animation sequence
        """
        try:
            sequence_id = f"body_anim_{int(time.time())}"
            poses = []
            current_time = 0.0
            
            # Get emotion modifiers
            emotion_modifiers = self.emotion_gesture_mapping.get(base_emotion, {})
            
            for gesture_config in gesture_configs:
                # Apply emotion modifiers
                adjusted_intensity = gesture_config.intensity * emotion_modifiers.get("gesture_intensity", 1.0)
                movement_speed = emotion_modifiers.get("movement_speed", 1.0)
                
                # Generate poses for this gesture
                gesture_poses = self._generate_gesture_poses(
                    gesture_config, 
                    adjusted_intensity, 
                    movement_speed,
                    avatar_style
                )
                
                # Adjust timing
                for pose in gesture_poses:
                    pose.timestamp = current_time
                    current_time += gesture_config.duration / len(gesture_poses)
                    poses.append(pose)
            
            # Add transition poses between gestures
            poses = self._add_transition_poses(poses)
            
            sequence = AnimationSequence(
                sequence_id=sequence_id,
                poses=poses,
                total_duration=current_time,
                metadata={
                    "emotion": base_emotion,
                    "avatar_style": avatar_style,
                    "gesture_count": len(gesture_configs)
                }
            )
            
            # Cache the sequence
            self.animation_cache[sequence_id] = sequence
            
            logger.info(f"Generated body animation sequence: {sequence_id} with {len(poses)} poses")
            return sequence
            
        except Exception as e:
            logger.error(f"Error generating body animation sequence: {e}")
            return AnimationSequence(
                sequence_id="error_sequence",
                poses=[],
                total_duration=0.0
            )
    
    def _generate_gesture_poses(
        self, 
        gesture_config: GestureConfig, 
        intensity: float, 
        speed: float,
        avatar_style: str
    ) -> List[BodyPose]:
        """Generate specific poses for a gesture."""
        poses = []
        pattern = self.gesture_patterns.get(gesture_config.gesture_type, {})
        
        # Generate key poses based on gesture type
        if gesture_config.gesture_type == GestureType.POINTING:
            poses = self._generate_pointing_poses(intensity, speed, pattern)
        elif gesture_config.gesture_type == GestureType.WAVING:
            poses = self._generate_waving_poses(intensity, speed, pattern)
        elif gesture_config.gesture_type == GestureType.EMPHASIS:
            poses = self._generate_emphasis_poses(intensity, speed, pattern)
        elif gesture_config.gesture_type == GestureType.THINKING:
            poses = self._generate_thinking_poses(intensity, speed, pattern)
        else:
            poses = self._generate_default_poses(intensity, speed, pattern)
        
        # Apply avatar style modifications
        poses = self._apply_avatar_style_modifications(poses, avatar_style)
        
        return poses
    
    def _generate_pointing_poses(self, intensity: float, speed: float, pattern: Dict[str, Any]) -> List[BodyPose]:
        """Generate poses for pointing gesture."""
        poses = []
        
        # Initial pose
        poses.append(BodyPose(
            timestamp=0.0,
            arm_rotation=(0.0, 0.0, 0.0),
            hand_position=(0.0, 0.0, 0.0)
        ))
        
        # Extension pose
        poses.append(BodyPose(
            timestamp=0.3,
            arm_rotation=(0.0, 0.0, 45.0 * intensity),
            hand_position=(0.0, 0.0, 0.8 * intensity),
            head_rotation=(0.0, 15.0 * intensity, 0.0)
        ))
        
        # Hold pose
        poses.append(BodyPose(
            timestamp=0.6,
            arm_rotation=(0.0, 0.0, 45.0 * intensity),
            hand_position=(0.0, 0.0, 0.8 * intensity),
            head_rotation=(0.0, 15.0 * intensity, 0.0)
        ))
        
        # Return pose
        poses.append(BodyPose(
            timestamp=1.0,
            arm_rotation=(0.0, 0.0, 0.0),
            hand_position=(0.0, 0.0, 0.0)
        ))
        
        return poses
    
    def _generate_waving_poses(self, intensity: float, speed: float, pattern: Dict[str, Any]) -> List[BodyPose]:
        """Generate poses for waving gesture."""
        poses = []
        
        # Initial pose
        poses.append(BodyPose(
            timestamp=0.0,
            arm_rotation=(0.0, 0.0, 0.0),
            hand_position=(0.0, 0.0, 0.0)
        ))
        
        # Wave poses (multiple)
        for i in range(3):
            wave_intensity = intensity * (0.5 + 0.5 * np.sin(i * np.pi / 2))
            poses.append(BodyPose(
                timestamp=0.2 + i * 0.2,
                arm_rotation=(0.0, 0.0, 30.0 * wave_intensity),
                hand_position=(0.0, 0.0, 0.6 * wave_intensity),
                head_rotation=(0.0, 10.0 * wave_intensity, 0.0)
            ))
        
        # Return pose
        poses.append(BodyPose(
            timestamp=1.0,
            arm_rotation=(0.0, 0.0, 0.0),
            hand_position=(0.0, 0.0, 0.0)
        ))
        
        return poses
    
    def _generate_emphasis_poses(self, intensity: float, speed: float, pattern: Dict[str, Any]) -> List[BodyPose]:
        """Generate poses for emphasis gesture."""
        poses = []
        
        # Initial pose
        poses.append(BodyPose(
            timestamp=0.0,
            arm_rotation=(0.0, 0.0, 0.0),
            hand_position=(0.0, 0.0, 0.0)
        ))
        
        # Emphasis pose
        poses.append(BodyPose(
            timestamp=0.3,
            arm_rotation=(0.0, 0.0, 60.0 * intensity),
            hand_position=(0.0, 0.0, 1.0 * intensity),
            torso_rotation=(0.0, 0.0, 15.0 * intensity),
            head_rotation=(0.0, 20.0 * intensity, 0.0)
        ))
        
        # Hold pose
        poses.append(BodyPose(
            timestamp=0.6,
            arm_rotation=(0.0, 0.0, 60.0 * intensity),
            hand_position=(0.0, 0.0, 1.0 * intensity),
            torso_rotation=(0.0, 0.0, 15.0 * intensity),
            head_rotation=(0.0, 20.0 * intensity, 0.0)
        ))
        
        # Return pose
        poses.append(BodyPose(
            timestamp=1.0,
            arm_rotation=(0.0, 0.0, 0.0),
            hand_position=(0.0, 0.0, 0.0)
        ))
        
        return poses
    
    def _generate_thinking_poses(self, intensity: float, speed: float, pattern: Dict[str, Any]) -> List[BodyPose]:
        """Generate poses for thinking gesture."""
        poses = []
        
        # Initial pose
        poses.append(BodyPose(
            timestamp=0.0,
            arm_rotation=(0.0, 0.0, 0.0),
            hand_position=(0.0, 0.0, 0.0)
        ))
        
        # Thinking pose
        poses.append(BodyPose(
            timestamp=0.4,
            arm_rotation=(0.0, 0.0, 30.0 * intensity),
            hand_position=(0.0, 0.0, 0.7 * intensity),
            head_rotation=(0.0, 0.0, 25.0 * intensity),
            torso_rotation=(0.0, 0.0, 10.0 * intensity)
        ))
        
        # Hold pose
        poses.append(BodyPose(
            timestamp=0.7,
            arm_rotation=(0.0, 0.0, 30.0 * intensity),
            hand_position=(0.0, 0.0, 0.7 * intensity),
            head_rotation=(0.0, 0.0, 25.0 * intensity),
            torso_rotation=(0.0, 0.0, 10.0 * intensity)
        ))
        
        # Return pose
        poses.append(BodyPose(
            timestamp=1.0,
            arm_rotation=(0.0, 0.0, 0.0),
            hand_position=(0.0, 0.0, 0.0)
        ))
        
        return poses
    
    def _generate_default_poses(self, intensity: float, speed: float, pattern: Dict[str, Any]) -> List[BodyPose]:
        """Generate default poses for unknown gesture types."""
        return [
            BodyPose(timestamp=0.0),
            BodyPose(timestamp=0.5, arm_rotation=(0.0, 0.0, 20.0 * intensity)),
            BodyPose(timestamp=1.0)
        ]
    
    def _apply_avatar_style_modifications(self, poses: List[BodyPose], avatar_style: str) -> List[BodyPose]:
        """Apply modifications based on avatar style."""
        if avatar_style == "cartoon":
            # Exaggerate movements for cartoon style
            for pose in poses:
                pose.arm_rotation = tuple(x * 1.5 for x in pose.arm_rotation)
                pose.head_rotation = tuple(x * 1.3 for x in pose.head_rotation)
        elif avatar_style == "minimalist":
            # Reduce movements for minimalist style
            for pose in poses:
                pose.arm_rotation = tuple(x * 0.7 for x in pose.arm_rotation)
                pose.head_rotation = tuple(x * 0.8 for x in pose.head_rotation)
        
        return poses
    
    def _add_transition_poses(self, poses: List[BodyPose]) -> List[BodyPose]:
        """Add smooth transition poses between gestures."""
        if len(poses) < 2:
            return poses
        
        transition_poses = []
        for i in range(len(poses) - 1):
            current_pose = poses[i]
            next_pose = poses[i + 1]
            
            # Add current pose
            transition_poses.append(current_pose)
            
            # Add intermediate transition pose
            if i < len(poses) - 1:
                transition_time = (current_pose.timestamp + next_pose.timestamp) / 2
                transition_pose = BodyPose(
                    timestamp=transition_time,
                    head_rotation=tuple((a + b) / 2 for a, b in zip(current_pose.head_rotation, next_pose.head_rotation)),
                    arm_rotation=tuple((a + b) / 2 for a, b in zip(current_pose.arm_rotation, next_pose.arm_rotation)),
                    hand_position=tuple((a + b) / 2 for a, b in zip(current_pose.hand_position, next_pose.hand_position))
                )
                transition_poses.append(transition_pose)
        
        # Add final pose
        transition_poses.append(poses[-1])
        
        return transition_poses
    
    async def export_animation_sequence(self, sequence: AnimationSequence, format: str = "json") -> str:
        """
        Export animation sequence to various formats.
        
        Args:
            sequence: Animation sequence to export
            format: Export format (json, fbx, bvh, etc.)
            
        Returns:
            Exported animation data
        """
        try:
            if format.lower() == "json":
                return self._export_to_json(sequence)
            elif format.lower() == "bvh":
                return self._export_to_bvh(sequence)
            else:
                logger.warning(f"Unsupported export format: {format}, defaulting to JSON")
                return self._export_to_json(sequence)
                
        except Exception as e:
            logger.error(f"Error exporting animation sequence: {e}")
            return ""
    
    def _export_to_json(self, sequence: AnimationSequence) -> str:
        """Export animation sequence to JSON format."""
        export_data = {
            "sequence_id": sequence.sequence_id,
            "total_duration": sequence.total_duration,
            "easing_type": sequence.easing_type,
            "loop": sequence.loop,
            "metadata": sequence.metadata,
            "poses": []
        }
        
        for pose in sequence.poses:
            pose_data = {
                "timestamp": pose.timestamp,
                "head_rotation": pose.head_rotation,
                "neck_rotation": pose.neck_rotation,
                "shoulder_position": pose.shoulder_position,
                "arm_rotation": pose.arm_rotation,
                "hand_position": pose.hand_position,
                "torso_rotation": pose.torso_rotation,
                "hip_position": pose.hip_position,
                "weight_shift": pose.weight_shift
            }
            export_data["poses"].append(pose_data)
        
        return json.dumps(export_data, indent=2)
    
    def _export_to_bvh(self, sequence: AnimationSequence) -> str:
        """Export animation sequence to BVH format."""
        # BVH header
        bvh_content = [
            "HIERARCHY",
            "ROOT root",
            "{",
            "  OFFSET 0.000000 0.000000 0.000000",
            "  CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation",
            "  JOINT head",
            "  {",
            "    OFFSET 0.000000 0.000000 0.000000",
            "    CHANNELS 3 Zrotation Xrotation Yrotation",
            "    End Site",
            "    {",
            "      OFFSET 0.000000 0.000000 0.000000",
            "    }",
            "  }",
            "  JOINT leftArm",
            "  {",
            "    OFFSET 0.000000 0.000000 0.000000",
            "    CHANNELS 3 Zrotation Xrotation Yrotation",
            "    End Site",
            "    {",
            "      OFFSET 0.000000 0.000000 0.000000",
            "    }",
            "  }",
            "  JOINT rightArm",
            "  {",
            "    OFFSET 0.000000 0.000000 0.000000",
            "    CHANNELS 3 Zrotation Xrotation Yrotation",
            "    End Site",
            "    {",
            "      OFFSET 0.000000 0.000000 0.000000",
            "    }",
            "  }",
            "}",
            "",
            "MOTION",
            f"Frames: {len(sequence.poses)}",
            f"Frame Time: {sequence.total_duration / len(sequence.poses):.6f}"
        ]
        
        # Add motion data
        for pose in sequence.poses:
            frame_data = [
                "0.000000 0.000000 0.000000",  # root position
                f"{pose.head_rotation[2]:.6f} {pose.head_rotation[0]:.6f} {pose.head_rotation[1]:.6f}",  # head
                f"{pose.arm_rotation[2]:.6f} {pose.arm_rotation[0]:.6f} {pose.arm_rotation[1]:.6f}",  # left arm
                f"{pose.arm_rotation[2]:.6f} {pose.arm_rotation[0]:.6f} {pose.arm_rotation[1]:.6f}"   # right arm
            ]
            bvh_content.extend(frame_data)
        
        return "\n".join(bvh_content)
    
    async def get_animation_statistics(self) -> Dict[str, Any]:
        """Get statistics about generated animations."""
        return {
            "total_sequences": len(self.animation_cache),
            "total_poses": sum(len(seq.poses) for seq in self.animation_cache.values()),
            "cache_size": len(self.script_analysis_cache),
            "supported_gestures": [gesture.value for gesture in GestureType],
            "supported_emotions": list(self.emotion_gesture_mapping.keys())
        }
    
    async def clear_cache(self):
        """Clear all cached data."""
        self.script_analysis_cache.clear()
        self.animation_cache.clear()
        logger.info("Animation service cache cleared")

