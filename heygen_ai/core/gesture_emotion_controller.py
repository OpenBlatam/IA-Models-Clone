"""
Gesture and Emotion Controller for HeyGen AI
Manages body gestures, facial expressions, and real-time emotion control for avatars.
"""

import json
import logging
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
import cv2
import mediapipe as mp
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class GestureType(str, Enum):
    """Types of body gestures"""
    WAVE = "wave"
    POINT = "point"
    THUMBS_UP = "thumbs_up"
    THUMBS_DOWN = "thumbs_down"
    CLAP = "clap"
    SHRUG = "shrug"
    NOD = "nod"
    SHAKE_HEAD = "shake_head"
    HANDS_ON_HIPS = "hands_on_hips"
    CROSSED_ARMS = "crossed_arms"
    OPEN_PALMS = "open_palms"
    FIST_PUMP = "fist_pump"
    PEACE_SIGN = "peace_sign"
    SALUTE = "salute"
    BOW = "bow"
    CUSTOM = "custom"


class EmotionType(str, Enum):
    """Types of emotions"""
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    SURPRISED = "surprised"
    FEARFUL = "fearful"
    DISGUSTED = "disgusted"
    NEUTRAL = "neutral"
    EXCITED = "excited"
    CONFUSED = "confused"
    CONFIDENT = "confident"
    THOUGHTFUL = "thoughtful"
    DETERMINED = "determined"
    RELAXED = "relaxed"
    NERVOUS = "nervous"
    ENTHUSIASTIC = "enthusiastic"
    SERIOUS = "serious"


class ExpressionIntensity(str, Enum):
    """Expression intensity levels"""
    SUBTLE = "subtle"
    MODERATE = "moderate"
    STRONG = "strong"
    EXTREME = "extreme"


@dataclass
class GestureConfig:
    """Configuration for a body gesture"""
    gesture_type: GestureType
    duration: float = 2.0
    intensity: float = 1.0
    start_time: float = 0.0
    end_time: Optional[float] = None
    body_parts: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EmotionConfig:
    """Configuration for an emotion"""
    emotion_type: EmotionType
    intensity: ExpressionIntensity = ExpressionIntensity.MODERATE
    duration: float = 3.0
    start_time: float = 0.0
    end_time: Optional[float] = None
    facial_features: Dict[str, float] = field(default_factory=dict)
    body_language: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GestureSequence:
    """Sequence of gestures"""
    id: str
    name: str
    description: str
    gestures: List[GestureConfig] = field(default_factory=list)
    total_duration: float = 0.0
    loop: bool = False
    tags: List[str] = field(default_factory=list)


@dataclass
class EmotionSequence:
    """Sequence of emotions"""
    id: str
    name: str
    description: str
    emotions: List[EmotionConfig] = field(default_factory=list)
    total_duration: float = 0.0
    loop: bool = False
    tags: List[str] = field(default_factory=list)


class GestureEmotionController:
    """Controller for managing gestures and emotions in real-time"""
    
    def __init__(self, config_path: str = "./data/gesture_emotion"):
        self.config_path = Path(config_path)
        self.gesture_sequences: Dict[str, GestureSequence] = {}
        self.emotion_sequences: Dict[str, EmotionSequence] = {}
        self.current_gesture: Optional[GestureConfig] = None
        self.current_emotion: Optional[EmotionConfig] = None
        self.is_active = False
        
        # MediaPipe setup for gesture detection
        self.mp_pose = mp.solutions.pose
        self.mp_face_mesh = mp.solutions.face_mesh
        self.pose_detector = None
        self.face_detector = None
        
        self._initialize_controller()
        self._load_default_sequences()
    
    def _initialize_controller(self):
        """Initialize the gesture and emotion controller"""
        try:
            self.config_path.mkdir(parents=True, exist_ok=True)
            
            # Create subdirectories
            (self.config_path / "gestures").mkdir(exist_ok=True)
            (self.config_path / "emotions").mkdir(exist_ok=True)
            (self.config_path / "sequences").mkdir(exist_ok=True)
            
            # Initialize MediaPipe detectors
            self.pose_detector = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                smooth_landmarks=True,
                enable_segmentation=False,
                smooth_segmentation=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            
            self.face_detector = self.mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            
            logger.info(f"Gesture and emotion controller initialized at {self.config_path}")
        except Exception as e:
            logger.error(f"Failed to initialize gesture and emotion controller: {e}")
            raise
    
    def _load_default_sequences(self):
        """Load default gesture and emotion sequences"""
        # Default gesture sequences
        default_gestures = [
            GestureSequence(
                id="greeting",
                name="Greeting Gesture",
                description="Friendly greeting with wave and smile",
                gestures=[
                    GestureConfig(
                        gesture_type=GestureType.WAVE,
                        duration=2.0,
                        intensity=0.8,
                        start_time=0.0,
                        body_parts=["right_arm", "right_hand"]
                    ),
                    GestureConfig(
                        gesture_type=GestureType.OPEN_PALMS,
                        duration=1.5,
                        intensity=0.6,
                        start_time=2.5,
                        body_parts=["both_arms", "both_hands"]
                    )
                ],
                total_duration=4.0,
                tags=["greeting", "friendly", "welcome"]
            ),
            GestureSequence(
                id="presentation",
                name="Presentation Gestures",
                description="Professional presentation gestures",
                gestures=[
                    GestureConfig(
                        gesture_type=GestureType.POINT,
                        duration=1.0,
                        intensity=0.7,
                        start_time=0.0,
                        body_parts=["right_arm", "right_hand"]
                    ),
                    GestureConfig(
                        gesture_type=GestureType.OPEN_PALMS,
                        duration=2.0,
                        intensity=0.8,
                        start_time=1.5,
                        body_parts=["both_arms", "both_hands"]
                    ),
                    GestureConfig(
                        gesture_type=GestureType.THUMBS_UP,
                        duration=1.5,
                        intensity=0.6,
                        start_time=4.0,
                        body_parts=["right_arm", "right_hand"]
                    )
                ],
                total_duration=6.0,
                tags=["presentation", "professional", "business"]
            ),
            GestureSequence(
                id="celebration",
                name="Celebration Gestures",
                description="Excited celebration gestures",
                gestures=[
                    GestureConfig(
                        gesture_type=GestureType.FIST_PUMP,
                        duration=1.0,
                        intensity=0.9,
                        start_time=0.0,
                        body_parts=["right_arm", "right_hand"]
                    ),
                    GestureConfig(
                        gesture_type=GestureType.CLAP,
                        duration=2.0,
                        intensity=0.8,
                        start_time=1.5,
                        body_parts=["both_arms", "both_hands"]
                    ),
                    GestureConfig(
                        gesture_type=GestureType.THUMBS_UP,
                        duration=1.5,
                        intensity=0.7,
                        start_time=4.0,
                        body_parts=["both_arms", "both_hands"]
                    )
                ],
                total_duration=6.0,
                tags=["celebration", "excited", "happy"]
            )
        ]
        
        # Default emotion sequences
        default_emotions = [
            EmotionSequence(
                id="friendly_greeting",
                name="Friendly Greeting",
                description="Warm and friendly greeting emotion",
                emotions=[
                    EmotionConfig(
                        emotion_type=EmotionType.HAPPY,
                        intensity=ExpressionIntensity.MODERATE,
                        duration=3.0,
                        start_time=0.0,
                        facial_features={
                            "smile": 0.7,
                            "eye_brightness": 0.8,
                            "cheek_raise": 0.6
                        }
                    ),
                    EmotionConfig(
                        emotion_type=EmotionType.CONFIDENT,
                        intensity=ExpressionIntensity.SUBTLE,
                        duration=2.0,
                        start_time=3.5,
                        facial_features={
                            "posture": 0.8,
                            "eye_contact": 0.9,
                            "head_tilt": 0.3
                        }
                    )
                ],
                total_duration=5.5,
                tags=["greeting", "friendly", "warm"]
            ),
            EmotionSequence(
                id="professional_presentation",
                name="Professional Presentation",
                description="Confident and professional presentation emotion",
                emotions=[
                    EmotionConfig(
                        emotion_type=EmotionType.CONFIDENT,
                        intensity=ExpressionIntensity.STRONG,
                        duration=4.0,
                        start_time=0.0,
                        facial_features={
                            "posture": 0.9,
                            "eye_contact": 0.95,
                            "jaw_set": 0.7
                        }
                    ),
                    EmotionConfig(
                        emotion_type=EmotionType.ENTHUSIASTIC,
                        intensity=ExpressionIntensity.MODERATE,
                        duration=3.0,
                        start_time=4.5,
                        facial_features={
                            "smile": 0.6,
                            "eye_brightness": 0.8,
                            "gesture_energy": 0.7
                        }
                    )
                ],
                total_duration=7.5,
                tags=["presentation", "professional", "confident"]
            ),
            EmotionSequence(
                id="excited_celebration",
                name="Excited Celebration",
                description="High-energy celebration emotion",
                emotions=[
                    EmotionConfig(
                        emotion_type=EmotionType.EXCITED,
                        intensity=ExpressionIntensity.STRONG,
                        duration=3.0,
                        start_time=0.0,
                        facial_features={
                            "smile": 0.9,
                            "eye_brightness": 1.0,
                            "cheek_raise": 0.8,
                            "mouth_open": 0.6
                        }
                    ),
                    EmotionConfig(
                        emotion_type=EmotionType.ENTHUSIASTIC,
                        intensity=ExpressionIntensity.EXTREME,
                        duration=2.5,
                        start_time=3.5,
                        facial_features={
                            "smile": 1.0,
                            "eye_brightness": 1.0,
                            "gesture_energy": 0.9
                        }
                    )
                ],
                total_duration=6.0,
                tags=["celebration", "excited", "energetic"]
            )
        ]
        
        # Add to dictionaries
        for gesture_seq in default_gestures:
            self.gesture_sequences[gesture_seq.id] = gesture_seq
        
        for emotion_seq in default_emotions:
            self.emotion_sequences[emotion_seq.id] = emotion_seq
        
        # Save to files
        self._save_gesture_sequences()
        self._save_emotion_sequences()
    
    def start_gesture_sequence(self, sequence_id: str, start_time: float = 0.0) -> bool:
        """Start a gesture sequence"""
        try:
            if sequence_id not in self.gesture_sequences:
                logger.error(f"Gesture sequence not found: {sequence_id}")
                return False
            
            sequence = self.gesture_sequences[sequence_id]
            if sequence.gestures:
                self.current_gesture = sequence.gestures[0]
                self.current_gesture.start_time = start_time
                self.is_active = True
                logger.info(f"Started gesture sequence: {sequence_id}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to start gesture sequence: {e}")
            return False
    
    def start_emotion_sequence(self, sequence_id: str, start_time: float = 0.0) -> bool:
        """Start an emotion sequence"""
        try:
            if sequence_id not in self.emotion_sequences:
                logger.error(f"Emotion sequence not found: {sequence_id}")
                return False
            
            sequence = self.emotion_sequences[sequence_id]
            if sequence.emotions:
                self.current_emotion = sequence.emotions[0]
                self.current_emotion.start_time = start_time
                self.is_active = True
                logger.info(f"Started emotion sequence: {sequence_id}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to start emotion sequence: {e}")
            return False
    
    def update_gesture(self, current_time: float) -> Optional[Dict[str, Any]]:
        """Update current gesture based on time"""
        if not self.current_gesture or not self.is_active:
            return None
        
        gesture = self.current_gesture
        elapsed_time = current_time - gesture.start_time
        
        if elapsed_time >= gesture.duration:
            # Gesture completed, move to next or stop
            return self._complete_current_gesture()
        
        # Calculate gesture progress and parameters
        progress = elapsed_time / gesture.duration
        gesture_data = {
            "type": gesture.gesture_type.value,
            "progress": progress,
            "intensity": gesture.intensity,
            "body_parts": gesture.body_parts,
            "parameters": self._calculate_gesture_parameters(gesture, progress)
        }
        
        return gesture_data
    
    def update_emotion(self, current_time: float) -> Optional[Dict[str, Any]]:
        """Update current emotion based on time"""
        if not self.current_emotion or not self.is_active:
            return None
        
        emotion = self.current_emotion
        elapsed_time = current_time - emotion.start_time
        
        if elapsed_time >= emotion.duration:
            # Emotion completed, move to next or stop
            return self._complete_current_emotion()
        
        # Calculate emotion progress and parameters
        progress = elapsed_time / emotion.duration
        emotion_data = {
            "type": emotion.emotion_type.value,
            "intensity": emotion.intensity.value,
            "progress": progress,
            "facial_features": self._calculate_emotion_parameters(emotion, progress),
            "body_language": emotion.body_language
        }
        
        return emotion_data
    
    def _complete_current_gesture(self) -> Optional[Dict[str, Any]]:
        """Complete current gesture and move to next"""
        # This would be implemented to move to the next gesture in the sequence
        # For now, just stop the current gesture
        self.current_gesture = None
        return {"type": "completed", "gesture": "finished"}
    
    def _complete_current_emotion(self) -> Optional[Dict[str, Any]]:
        """Complete current emotion and move to next"""
        # This would be implemented to move to the next emotion in the sequence
        # For now, just stop the current emotion
        self.current_emotion = None
        return {"type": "completed", "emotion": "finished"}
    
    def _calculate_gesture_parameters(self, gesture: GestureConfig, progress: float) -> Dict[str, Any]:
        """Calculate gesture parameters based on progress"""
        # This would implement specific gesture animations
        # For now, return basic parameters
        return {
            "arm_angle": 45 * progress,
            "hand_position": [0.5, 0.3 + 0.2 * progress],
            "body_rotation": 10 * progress if gesture.gesture_type == GestureType.WAVE else 0
        }
    
    def _calculate_emotion_parameters(self, emotion: EmotionConfig, progress: float) -> Dict[str, float]:
        """Calculate emotion parameters based on progress"""
        # This would implement specific facial expression parameters
        # For now, return basic parameters
        base_features = emotion.facial_features.copy()
        
        # Apply progress-based interpolation
        for key, value in base_features.items():
            if key in ["smile", "eye_brightness", "cheek_raise"]:
                base_features[key] = value * progress
        
        return base_features
    
    def detect_gesture_from_video(self, video_frame: np.ndarray) -> Optional[GestureType]:
        """Detect gesture from video frame using MediaPipe"""
        try:
            if self.pose_detector is None:
                return None
            
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(video_frame, cv2.COLOR_BGR2RGB)
            results = self.pose_detector.process(rgb_frame)
            
            if not results.pose_landmarks:
                return None
            
            # Analyze pose landmarks to detect gestures
            gesture = self._analyze_pose_landmarks(results.pose_landmarks)
            return gesture
        except Exception as e:
            logger.error(f"Failed to detect gesture from video: {e}")
            return None
    
    def detect_emotion_from_video(self, video_frame: np.ndarray) -> Optional[EmotionType]:
        """Detect emotion from video frame using MediaPipe"""
        try:
            if self.face_detector is None:
                return None
            
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(video_frame, cv2.COLOR_BGR2RGB)
            results = self.face_detector.process(rgb_frame)
            
            if not results.multi_face_landmarks:
                return None
            
            # Analyze face landmarks to detect emotions
            emotion = self._analyze_face_landmarks(results.multi_face_landmarks[0])
            return emotion
        except Exception as e:
            logger.error(f"Failed to detect emotion from video: {e}")
            return None
    
    def _analyze_pose_landmarks(self, landmarks) -> Optional[GestureType]:
        """Analyze pose landmarks to detect gestures"""
        # This would implement gesture detection logic
        # For now, return None
        return None
    
    def _analyze_face_landmarks(self, landmarks) -> Optional[EmotionType]:
        """Analyze face landmarks to detect emotions"""
        # This would implement emotion detection logic
        # For now, return None
        return None
    
    def create_custom_gesture_sequence(self, sequence: GestureSequence) -> str:
        """Create a custom gesture sequence"""
        try:
            if sequence.id in self.gesture_sequences:
                raise ValueError("Sequence ID must be unique")
            
            self.gesture_sequences[sequence.id] = sequence
            self._save_gesture_sequences()
            logger.info(f"Created custom gesture sequence: {sequence.id}")
            return sequence.id
        except Exception as e:
            logger.error(f"Failed to create custom gesture sequence: {e}")
            raise
    
    def create_custom_emotion_sequence(self, sequence: EmotionSequence) -> str:
        """Create a custom emotion sequence"""
        try:
            if sequence.id in self.emotion_sequences:
                raise ValueError("Sequence ID must be unique")
            
            self.emotion_sequences[sequence.id] = sequence
            self._save_emotion_sequences()
            logger.info(f"Created custom emotion sequence: {sequence.id}")
            return sequence.id
        except Exception as e:
            logger.error(f"Failed to create custom emotion sequence: {e}")
            raise
    
    def get_gesture_sequence(self, sequence_id: str) -> Optional[GestureSequence]:
        """Get a gesture sequence by ID"""
        return self.gesture_sequences.get(sequence_id)
    
    def get_emotion_sequence(self, sequence_id: str) -> Optional[EmotionSequence]:
        """Get an emotion sequence by ID"""
        return self.emotion_sequences.get(sequence_id)
    
    def get_all_gesture_sequences(self) -> Dict[str, GestureSequence]:
        """Get all gesture sequences"""
        return self.gesture_sequences.copy()
    
    def get_all_emotion_sequences(self) -> Dict[str, EmotionSequence]:
        """Get all emotion sequences"""
        return self.emotion_sequences.copy()
    
    def search_gesture_sequences(self, query: str) -> List[GestureSequence]:
        """Search gesture sequences by name, description, or tags"""
        query = query.lower()
        results = []
        
        for sequence in self.gesture_sequences.values():
            if (query in sequence.name.lower() or 
                query in sequence.description.lower() or
                any(query in tag.lower() for tag in sequence.tags)):
                results.append(sequence)
        
        return results
    
    def search_emotion_sequences(self, query: str) -> List[EmotionSequence]:
        """Search emotion sequences by name, description, or tags"""
        query = query.lower()
        results = []
        
        for sequence in self.emotion_sequences.values():
            if (query in sequence.name.lower() or 
                query in sequence.description.lower() or
                any(query in tag.lower() for tag in sequence.tags)):
                results.append(sequence)
        
        return results
    
    def stop_current_sequences(self):
        """Stop all current gesture and emotion sequences"""
        self.current_gesture = None
        self.current_emotion = None
        self.is_active = False
        logger.info("Stopped all current sequences")
    
    def _save_gesture_sequences(self):
        """Save gesture sequences to JSON file"""
        try:
            sequences_data = {}
            for seq_id, sequence in self.gesture_sequences.items():
                sequences_data[seq_id] = {
                    "id": sequence.id,
                    "name": sequence.name,
                    "description": sequence.description,
                    "gestures": [
                        {
                            "gesture_type": g.gesture_type.value,
                            "duration": g.duration,
                            "intensity": g.intensity,
                            "start_time": g.start_time,
                            "end_time": g.end_time,
                            "body_parts": g.body_parts,
                            "parameters": g.parameters
                        } for g in sequence.gestures
                    ],
                    "total_duration": sequence.total_duration,
                    "loop": sequence.loop,
                    "tags": sequence.tags
                }
            
            with open(self.config_path / "gestures" / "gesture_sequences.json", "w") as f:
                json.dump(sequences_data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save gesture sequences: {e}")
    
    def _save_emotion_sequences(self):
        """Save emotion sequences to JSON file"""
        try:
            sequences_data = {}
            for seq_id, sequence in self.emotion_sequences.items():
                sequences_data[seq_id] = {
                    "id": sequence.id,
                    "name": sequence.name,
                    "description": sequence.description,
                    "emotions": [
                        {
                            "emotion_type": e.emotion_type.value,
                            "intensity": e.intensity.value,
                            "duration": e.duration,
                            "start_time": e.start_time,
                            "end_time": e.end_time,
                            "facial_features": e.facial_features,
                            "body_language": e.body_language
                        } for e in sequence.emotions
                    ],
                    "total_duration": sequence.total_duration,
                    "loop": sequence.loop,
                    "tags": sequence.tags
                }
            
            with open(self.config_path / "emotions" / "emotion_sequences.json", "w") as f:
                json.dump(sequences_data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save emotion sequences: {e}")
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on the gesture and emotion controller"""
        try:
            stats = {
                "status": "healthy",
                "total_gesture_sequences": len(self.gesture_sequences),
                "total_emotion_sequences": len(self.emotion_sequences),
                "is_active": self.is_active,
                "current_gesture": self.current_gesture.gesture_type.value if self.current_gesture else None,
                "current_emotion": self.current_emotion.emotion_type.value if self.current_emotion else None,
                "pose_detector_available": self.pose_detector is not None,
                "face_detector_available": self.face_detector is not None,
                "errors": []
            }
            
            # Check file system
            if not self.config_path.exists():
                stats["status"] = "error"
                stats["errors"].append("Config directory does not exist")
            
            return stats
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "errors": [str(e)]
            }


# Example usage
if __name__ == "__main__":
    # Initialize controller
    controller = GestureEmotionController()
    
    # Get all sequences
    gesture_sequences = controller.get_all_gesture_sequences()
    emotion_sequences = controller.get_all_emotion_sequences()
    print(f"Available gesture sequences: {len(gesture_sequences)}")
    print(f"Available emotion sequences: {len(emotion_sequences)}")
    
    # Start a gesture sequence
    controller.start_gesture_sequence("greeting")
    
    # Simulate time progression
    for i in range(10):
        current_time = i * 0.5
        gesture_data = controller.update_gesture(current_time)
        emotion_data = controller.update_emotion(current_time)
        
        if gesture_data:
            print(f"Time {current_time}s - Gesture: {gesture_data}")
        if emotion_data:
            print(f"Time {current_time}s - Emotion: {emotion_data}")
    
    # Health check
    health = controller.health_check()
    print(f"Controller health: {health['status']}")


