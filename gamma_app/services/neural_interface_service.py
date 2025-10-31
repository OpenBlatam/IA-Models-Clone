"""
Neural Interface Service for Gamma App
======================================

Advanced service for Neural Interface capabilities including brain-computer
interfaces, neural signal processing, and direct neural communication.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import uuid
import json
import numpy as np
from enum import Enum

logger = logging.getLogger(__name__)

class NeuralSignalType(str, Enum):
    """Types of neural signals."""
    EEG = "eeg"
    ECoG = "ecog"
    LFP = "lfp"
    SPIKE = "spike"
    FMRI = "fmri"
    NIRS = "nirs"
    MEG = "meg"
    OPTICAL = "optical"

class NeuralInterfaceType(str, Enum):
    """Types of neural interfaces."""
    INVASIVE = "invasive"
    NON_INVASIVE = "non_invasive"
    PARTIALLY_INVASIVE = "partially_invasive"
    OPTICAL = "optical"
    ELECTRICAL = "electrical"
    MAGNETIC = "magnetic"
    ULTRASOUND = "ultrasound"

class NeuralCommandType(str, Enum):
    """Types of neural commands."""
    MOVEMENT = "movement"
    SPEECH = "speech"
    VISION = "vision"
    COGNITION = "cognition"
    EMOTION = "emotion"
    MEMORY = "memory"
    ATTENTION = "attention"
    CREATIVITY = "creativity"

@dataclass
class NeuralDevice:
    """Neural interface device."""
    device_id: str
    name: str
    interface_type: NeuralInterfaceType
    signal_types: List[NeuralSignalType]
    num_channels: int
    sampling_rate: float
    resolution: float
    is_connected: bool = False
    battery_level: Optional[int] = None
    last_calibration: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class NeuralSession:
    """Neural interface session."""
    session_id: str
    device_id: str
    user_id: str
    session_type: str
    start_time: datetime
    end_time: Optional[datetime] = None
    is_active: bool = True
    signal_quality: float = 0.0
    artifacts_detected: List[str] = field(default_factory=list)
    calibration_data: Dict[str, Any] = field(default_factory=dict)

@dataclass
class NeuralSignal:
    """Neural signal data."""
    signal_id: str
    session_id: str
    signal_type: NeuralSignalType
    channel: int
    timestamp: datetime
    data: List[float]
    frequency_bands: Dict[str, float]
    artifacts: List[str] = field(default_factory=list)
    quality_score: float = 0.0

@dataclass
class NeuralCommand:
    """Neural command."""
    command_id: str
    session_id: str
    command_type: NeuralCommandType
    intent: str
    confidence: float
    parameters: Dict[str, Any]
    timestamp: datetime
    executed: bool = False
    execution_time: Optional[float] = None

@dataclass
class NeuralPattern:
    """Neural pattern recognition."""
    pattern_id: str
    user_id: str
    pattern_type: str
    features: Dict[str, Any]
    classification: str
    confidence: float
    training_data: List[Dict[str, Any]]
    model_accuracy: float
    created_at: datetime = field(default_factory=datetime.now)

class NeuralInterfaceService:
    """Service for Neural Interface capabilities."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.neural_devices: Dict[str, NeuralDevice] = {}
        self.neural_sessions: Dict[str, NeuralSession] = {}
        self.neural_signals: List[NeuralSignal] = []
        self.neural_commands: List[NeuralCommand] = []
        self.neural_patterns: Dict[str, NeuralPattern] = {}
        self.active_sessions: Dict[str, str] = {}  # user_id -> session_id
        
        # Initialize default devices
        self._initialize_default_devices()
        
        logger.info("NeuralInterfaceService initialized")
    
    async def register_neural_device(self, device_info: Dict[str, Any]) -> str:
        """Register a neural interface device."""
        try:
            device_id = str(uuid.uuid4())
            device = NeuralDevice(
                device_id=device_id,
                name=device_info.get("name", "Unknown Neural Device"),
                interface_type=NeuralInterfaceType(device_info.get("interface_type", "non_invasive")),
                signal_types=[NeuralSignalType(st) for st in device_info.get("signal_types", ["eeg"])],
                num_channels=device_info.get("num_channels", 64),
                sampling_rate=device_info.get("sampling_rate", 1000.0),
                resolution=device_info.get("resolution", 16.0),
                is_connected=True
            )
            
            self.neural_devices[device_id] = device
            logger.info(f"Neural device registered: {device_id}")
            return device_id
            
        except Exception as e:
            logger.error(f"Error registering neural device: {e}")
            raise
    
    async def start_neural_session(self, session_info: Dict[str, Any]) -> str:
        """Start a neural interface session."""
        try:
            session_id = str(uuid.uuid4())
            session = NeuralSession(
                session_id=session_id,
                device_id=session_info.get("device_id", ""),
                user_id=session_info.get("user_id", ""),
                session_type=session_info.get("session_type", "general"),
                start_time=datetime.now()
            )
            
            self.neural_sessions[session_id] = session
            self.active_sessions[session.user_id] = session_id
            
            # Start signal processing in background
            asyncio.create_task(self._process_neural_signals(session_id))
            
            logger.info(f"Neural session started: {session_id}")
            return session_id
            
        except Exception as e:
            logger.error(f"Error starting neural session: {e}")
            raise
    
    async def process_neural_signal(self, signal_data: Dict[str, Any]) -> str:
        """Process neural signal data."""
        try:
            signal_id = str(uuid.uuid4())
            signal = NeuralSignal(
                signal_id=signal_id,
                session_id=signal_data.get("session_id", ""),
                signal_type=NeuralSignalType(signal_data.get("signal_type", "eeg")),
                channel=signal_data.get("channel", 0),
                timestamp=datetime.now(),
                data=signal_data.get("data", []),
                frequency_bands=self._analyze_frequency_bands(signal_data.get("data", [])),
                quality_score=self._calculate_signal_quality(signal_data.get("data", []))
            )
            
            # Detect artifacts
            signal.artifacts = self._detect_artifacts(signal.data)
            
            self.neural_signals.append(signal)
            
            # Process signal for commands
            await self._process_signal_for_commands(signal)
            
            logger.info(f"Neural signal processed: {signal_id}")
            return signal_id
            
        except Exception as e:
            logger.error(f"Error processing neural signal: {e}")
            raise
    
    async def generate_neural_command(self, command_info: Dict[str, Any]) -> str:
        """Generate a neural command from brain signals."""
        try:
            command_id = str(uuid.uuid4())
            command = NeuralCommand(
                command_id=command_id,
                session_id=command_info.get("session_id", ""),
                command_type=NeuralCommandType(command_info.get("command_type", "movement")),
                intent=command_info.get("intent", ""),
                confidence=command_info.get("confidence", 0.0),
                parameters=command_info.get("parameters", {}),
                timestamp=datetime.now()
            )
            
            self.neural_commands.append(command)
            
            # Execute command if confidence is high enough
            if command.confidence > 0.7:
                await self._execute_neural_command(command)
            
            logger.info(f"Neural command generated: {command_id}")
            return command_id
            
        except Exception as e:
            logger.error(f"Error generating neural command: {e}")
            raise
    
    async def train_neural_pattern(self, pattern_info: Dict[str, Any]) -> str:
        """Train a neural pattern recognition model."""
        try:
            pattern_id = str(uuid.uuid4())
            pattern = NeuralPattern(
                pattern_id=pattern_id,
                user_id=pattern_info.get("user_id", ""),
                pattern_type=pattern_info.get("pattern_type", "movement"),
                features=pattern_info.get("features", {}),
                classification=pattern_info.get("classification", ""),
                confidence=0.0,
                training_data=pattern_info.get("training_data", []),
                model_accuracy=0.0
            )
            
            # Train the pattern recognition model
            await self._train_pattern_model(pattern)
            
            self.neural_patterns[pattern_id] = pattern
            logger.info(f"Neural pattern trained: {pattern_id}")
            return pattern_id
            
        except Exception as e:
            logger.error(f"Error training neural pattern: {e}")
            raise
    
    async def recognize_neural_pattern(self, pattern_data: Dict[str, Any]) -> Dict[str, Any]:
        """Recognize neural patterns in real-time."""
        try:
            user_id = pattern_data.get("user_id", "")
            signal_features = pattern_data.get("features", {})
            
            # Find matching patterns for the user
            user_patterns = [p for p in self.neural_patterns.values() if p.user_id == user_id]
            
            best_match = None
            best_confidence = 0.0
            
            for pattern in user_patterns:
                confidence = self._calculate_pattern_similarity(signal_features, pattern.features)
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_match = pattern
            
            if best_match and best_confidence > 0.6:
                return {
                    "pattern_id": best_match.pattern_id,
                    "classification": best_match.classification,
                    "confidence": best_confidence,
                    "pattern_type": best_match.pattern_type,
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return {
                    "pattern_id": None,
                    "classification": "unknown",
                    "confidence": best_confidence,
                    "pattern_type": "unknown",
                    "timestamp": datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Error recognizing neural pattern: {e}")
            return {"error": str(e)}
    
    async def get_neural_session_status(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get neural session status."""
        try:
            if session_id not in self.neural_sessions:
                return None
            
            session = self.neural_sessions[session_id]
            device = self.neural_devices.get(session.device_id)
            
            return {
                "session_id": session.session_id,
                "device_id": session.device_id,
                "user_id": session.user_id,
                "session_type": session.session_type,
                "start_time": session.start_time.isoformat(),
                "end_time": session.end_time.isoformat() if session.end_time else None,
                "is_active": session.is_active,
                "signal_quality": session.signal_quality,
                "artifacts_detected": session.artifacts_detected,
                "device_info": {
                    "name": device.name if device else "Unknown",
                    "interface_type": device.interface_type.value if device else "unknown",
                    "num_channels": device.num_channels if device else 0
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting neural session status: {e}")
            return None
    
    async def get_neural_statistics(self) -> Dict[str, Any]:
        """Get neural interface service statistics."""
        try:
            total_devices = len(self.neural_devices)
            connected_devices = len([d for d in self.neural_devices.values() if d.is_connected])
            total_sessions = len(self.neural_sessions)
            active_sessions = len([s for s in self.neural_sessions.values() if s.is_active])
            total_signals = len(self.neural_signals)
            total_commands = len(self.neural_commands)
            executed_commands = len([c for c in self.neural_commands if c.executed])
            total_patterns = len(self.neural_patterns)
            
            # Signal type distribution
            signal_type_stats = {}
            for signal in self.neural_signals:
                signal_type = signal.signal_type.value
                signal_type_stats[signal_type] = signal_type_stats.get(signal_type, 0) + 1
            
            # Command type distribution
            command_type_stats = {}
            for command in self.neural_commands:
                command_type = command.command_type.value
                command_type_stats[command_type] = command_type_stats.get(command_type, 0) + 1
            
            # Interface type distribution
            interface_type_stats = {}
            for device in self.neural_devices.values():
                interface_type = device.interface_type.value
                interface_type_stats[interface_type] = interface_type_stats.get(interface_type, 0) + 1
            
            return {
                "total_neural_devices": total_devices,
                "connected_devices": connected_devices,
                "total_neural_sessions": total_sessions,
                "active_sessions": active_sessions,
                "total_neural_signals": total_signals,
                "total_neural_commands": total_commands,
                "executed_commands": executed_commands,
                "command_success_rate": (executed_commands / total_commands * 100) if total_commands > 0 else 0,
                "total_neural_patterns": total_patterns,
                "signal_type_distribution": signal_type_stats,
                "command_type_distribution": command_type_stats,
                "interface_type_distribution": interface_type_stats,
                "last_updated": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting neural statistics: {e}")
            return {}
    
    async def _process_neural_signals(self, session_id: str):
        """Process neural signals in background."""
        try:
            session = self.neural_sessions[session_id]
            device = self.neural_devices.get(session.device_id)
            
            if not device:
                return
            
            # Simulate continuous signal processing
            while session.is_active:
                await asyncio.sleep(0.1)  # 100ms processing interval
                
                # Simulate signal quality monitoring
                session.signal_quality = np.random.uniform(0.7, 0.95)
                
                # Simulate artifact detection
                if np.random.random() < 0.1:  # 10% chance of artifacts
                    artifacts = ["eye_blink", "muscle_artifact", "electrode_drift"]
                    session.artifacts_detected = [np.random.choice(artifacts)]
                
        except Exception as e:
            logger.error(f"Error processing neural signals for session {session_id}: {e}")
    
    async def _process_signal_for_commands(self, signal: NeuralSignal):
        """Process neural signal for command generation."""
        try:
            # Analyze signal for command patterns
            if signal.signal_type == NeuralSignalType.EEG:
                # Analyze EEG for movement intentions
                if self._detect_movement_intention(signal.data):
                    await self.generate_neural_command({
                        "session_id": signal.session_id,
                        "command_type": "movement",
                        "intent": "move_forward",
                        "confidence": 0.8,
                        "parameters": {"direction": "forward", "speed": 1.0}
                    })
                
                # Analyze EEG for attention patterns
                if self._detect_attention_pattern(signal.data):
                    await self.generate_neural_command({
                        "session_id": signal.session_id,
                        "command_type": "attention",
                        "intent": "focus",
                        "confidence": 0.7,
                        "parameters": {"intensity": 0.8}
                    })
                
        except Exception as e:
            logger.error(f"Error processing signal for commands: {e}")
    
    async def _execute_neural_command(self, command: NeuralCommand):
        """Execute a neural command."""
        try:
            command.executed = True
            command.execution_time = 0.1  # Simulate execution time
            
            # Simulate command execution based on type
            if command.command_type == NeuralCommandType.MOVEMENT:
                logger.info(f"Executing movement command: {command.intent}")
            elif command.command_type == NeuralCommandType.SPEECH:
                logger.info(f"Executing speech command: {command.intent}")
            elif command.command_type == NeuralCommandType.VISION:
                logger.info(f"Executing vision command: {command.intent}")
            elif command.command_type == NeuralCommandType.COGNITION:
                logger.info(f"Executing cognition command: {command.intent}")
            
        except Exception as e:
            logger.error(f"Error executing neural command: {e}")
    
    async def _train_pattern_model(self, pattern: NeuralPattern):
        """Train neural pattern recognition model."""
        try:
            # Simulate training process
            await asyncio.sleep(1)  # Simulate training time
            
            # Calculate model accuracy based on training data
            if pattern.training_data:
                pattern.model_accuracy = np.random.uniform(0.8, 0.95)
                pattern.confidence = np.random.uniform(0.7, 0.9)
            else:
                pattern.model_accuracy = 0.5
                pattern.confidence = 0.5
            
        except Exception as e:
            logger.error(f"Error training pattern model: {e}")
    
    def _analyze_frequency_bands(self, data: List[float]) -> Dict[str, float]:
        """Analyze frequency bands in neural signal."""
        if not data:
            return {}
        
        # Simulate frequency band analysis
        return {
            "delta": np.random.uniform(0.1, 0.3),    # 0.5-4 Hz
            "theta": np.random.uniform(0.1, 0.4),    # 4-8 Hz
            "alpha": np.random.uniform(0.2, 0.6),    # 8-13 Hz
            "beta": np.random.uniform(0.1, 0.5),     # 13-30 Hz
            "gamma": np.random.uniform(0.05, 0.3)    # 30-100 Hz
        }
    
    def _calculate_signal_quality(self, data: List[float]) -> float:
        """Calculate signal quality score."""
        if not data:
            return 0.0
        
        # Simulate signal quality calculation
        # In real implementation, this would analyze SNR, artifacts, etc.
        return np.random.uniform(0.6, 0.95)
    
    def _detect_artifacts(self, data: List[float]) -> List[str]:
        """Detect artifacts in neural signal."""
        artifacts = []
        
        # Simulate artifact detection
        if np.random.random() < 0.1:  # 10% chance of eye blink
            artifacts.append("eye_blink")
        if np.random.random() < 0.05:  # 5% chance of muscle artifact
            artifacts.append("muscle_artifact")
        if np.random.random() < 0.02:  # 2% chance of electrode drift
            artifacts.append("electrode_drift")
        
        return artifacts
    
    def _detect_movement_intention(self, data: List[float]) -> bool:
        """Detect movement intention in EEG data."""
        # Simulate movement intention detection
        return np.random.random() < 0.1  # 10% chance of movement intention
    
    def _detect_attention_pattern(self, data: List[float]) -> bool:
        """Detect attention patterns in EEG data."""
        # Simulate attention pattern detection
        return np.random.random() < 0.15  # 15% chance of attention pattern
    
    def _calculate_pattern_similarity(self, features1: Dict[str, Any], features2: Dict[str, Any]) -> float:
        """Calculate similarity between neural patterns."""
        # Simulate pattern similarity calculation
        return np.random.uniform(0.0, 1.0)
    
    def _initialize_default_devices(self):
        """Initialize default neural devices."""
        try:
            # Default EEG device
            eeg_device = NeuralDevice(
                device_id=str(uuid.uuid4()),
                name="Standard EEG Headset",
                interface_type=NeuralInterfaceType.NON_INVASIVE,
                signal_types=[NeuralSignalType.EEG],
                num_channels=64,
                sampling_rate=1000.0,
                resolution=16.0,
                is_connected=False
            )
            self.neural_devices[eeg_device.device_id] = eeg_device
            
            # Default ECoG device
            ecog_device = NeuralDevice(
                device_id=str(uuid.uuid4()),
                name="ECoG Grid Array",
                interface_type=NeuralInterfaceType.INVASIVE,
                signal_types=[NeuralSignalType.ECoG],
                num_channels=128,
                sampling_rate=2000.0,
                resolution=24.0,
                is_connected=False
            )
            self.neural_devices[ecog_device.device_id] = ecog_device
            
            logger.info("Default neural devices initialized")
            
        except Exception as e:
            logger.error(f"Error initializing default devices: {e}")


