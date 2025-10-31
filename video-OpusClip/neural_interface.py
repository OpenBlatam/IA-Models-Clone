"""
Neural Interface System for Ultimate Opus Clip

Advanced brain-computer interface capabilities for direct neural control,
thought-based video editing, and cognitive enhancement of content creation.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
import asyncio
import time
import json
import uuid
from dataclasses import dataclass, asdict
from enum import Enum
import structlog
from pathlib import Path
import numpy as np
import threading
from datetime import datetime, timedelta
import socket
import ssl
import websockets
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

logger = structlog.get_logger("neural_interface")

class NeuralSignalType(Enum):
    """Types of neural signals."""
    EEG = "eeg"  # Electroencephalography
    ECoG = "ecog"  # Electrocorticography
    LFP = "lfp"  # Local Field Potential
    SPIKE = "spike"  # Single neuron spikes
    LFP_BAND = "lfp_band"  # LFP frequency bands
    P300 = "p300"  # Event-related potential
    SSVEP = "ssvep"  # Steady-state visual evoked potential

class CognitiveState(Enum):
    """Cognitive states detected."""
    FOCUSED = "focused"
    RELAXED = "relaxed"
    STRESSED = "stressed"
    CREATIVE = "creative"
    ANALYTICAL = "analytical"
    EMOTIONAL = "emotional"
    NEUTRAL = "neutral"
    EXCITED = "excited"
    TIRED = "tired"
    CONFUSED = "confused"

class NeuralCommand(Enum):
    """Neural commands for control."""
    PLAY = "play"
    PAUSE = "pause"
    STOP = "stop"
    REWIND = "rewind"
    FAST_FORWARD = "fast_forward"
    CUT = "cut"
    COPY = "copy"
    PASTE = "paste"
    ZOOM_IN = "zoom_in"
    ZOOM_OUT = "zoom_out"
    ROTATE = "rotate"
    FILTER = "filter"
    ENHANCE = "enhance"
    SAVE = "save"
    EXPORT = "export"

@dataclass
class NeuralSignal:
    """Neural signal data."""
    signal_id: str
    signal_type: NeuralSignalType
    timestamp: float
    channels: List[str]
    data: np.ndarray
    sampling_rate: float
    quality: float
    metadata: Dict[str, Any] = None

@dataclass
class CognitiveProfile:
    """User cognitive profile."""
    user_id: str
    baseline_state: CognitiveState
    attention_span: float
    creativity_index: float
    stress_threshold: float
    preferred_interface: str
    neural_patterns: Dict[str, Any]
    created_at: float = 0.0

@dataclass
class NeuralCommand:
    """Neural command execution."""
    command_id: str
    user_id: str
    command_type: NeuralCommand
    confidence: float
    execution_time: float
    success: bool
    timestamp: float
    context: Dict[str, Any] = None

@dataclass
class BrainState:
    """Current brain state analysis."""
    user_id: str
    cognitive_state: CognitiveState
    attention_level: float
    stress_level: float
    creativity_level: float
    fatigue_level: float
    emotional_valence: float
    arousal_level: float
    timestamp: float
    neural_metrics: Dict[str, float] = None

class NeuralSignalProcessor:
    """Advanced neural signal processing."""
    
    def __init__(self):
        self.signal_filters = {}
        self.feature_extractors = {}
        self.classifiers = {}
        self._initialize_processors()
        
        logger.info("Neural Signal Processor initialized")
    
    def _initialize_processors(self):
        """Initialize signal processing components."""
        try:
            # Initialize filters for different signal types
            self.signal_filters = {
                NeuralSignalType.EEG: self._create_eeg_filter(),
                NeuralSignalType.ECoG: self._create_ecog_filter(),
                NeuralSignalType.LFP: self._create_lfp_filter(),
                NeuralSignalType.SPIKE: self._create_spike_filter()
            }
            
            # Initialize feature extractors
            self.feature_extractors = {
                "spectral": self._create_spectral_extractor(),
                "temporal": self._create_temporal_extractor(),
                "spatial": self._create_spatial_extractor(),
                "nonlinear": self._create_nonlinear_extractor()
            }
            
            # Initialize classifiers
            self.classifiers = {
                "cognitive_state": self._create_cognitive_classifier(),
                "neural_command": self._create_command_classifier(),
                "emotion": self._create_emotion_classifier()
            }
            
            logger.info("Neural signal processors initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing neural processors: {e}")
    
    def _create_eeg_filter(self):
        """Create EEG signal filter."""
        def filter_eeg(signal_data, sampling_rate):
            # Simulate EEG filtering (bandpass 1-40 Hz, notch 50/60 Hz)
            filtered_data = signal_data.copy()
            # In real implementation, apply actual filters
            return filtered_data
        return filter_eeg
    
    def _create_ecog_filter(self):
        """Create ECoG signal filter."""
        def filter_ecog(signal_data, sampling_rate):
            # Simulate ECoG filtering (bandpass 0.5-200 Hz)
            filtered_data = signal_data.copy()
            return filtered_data
        return filter_ecog
    
    def _create_lfp_filter(self):
        """Create LFP signal filter."""
        def filter_lfp(signal_data, sampling_rate):
            # Simulate LFP filtering (bandpass 0.1-300 Hz)
            filtered_data = signal_data.copy()
            return filtered_data
        return filter_lfp
    
    def _create_spike_filter(self):
        """Create spike signal filter."""
        def filter_spike(signal_data, sampling_rate):
            # Simulate spike filtering (highpass 300 Hz)
            filtered_data = signal_data.copy()
            return filtered_data
        return filter_spike
    
    def _create_spectral_extractor(self):
        """Create spectral feature extractor."""
        def extract_spectral(signal_data, sampling_rate):
            # Simulate spectral feature extraction
            features = {
                "power_alpha": np.random.uniform(0, 1),
                "power_beta": np.random.uniform(0, 1),
                "power_gamma": np.random.uniform(0, 1),
                "power_theta": np.random.uniform(0, 1),
                "power_delta": np.random.uniform(0, 1),
                "spectral_centroid": np.random.uniform(0, 50),
                "spectral_bandwidth": np.random.uniform(0, 20)
            }
            return features
        return extract_spectral
    
    def _create_temporal_extractor(self):
        """Create temporal feature extractor."""
        def extract_temporal(signal_data, sampling_rate):
            # Simulate temporal feature extraction
            features = {
                "mean": np.mean(signal_data),
                "std": np.std(signal_data),
                "variance": np.var(signal_data),
                "skewness": np.random.uniform(-2, 2),
                "kurtosis": np.random.uniform(-2, 2),
                "zero_crossing_rate": np.random.uniform(0, 1)
            }
            return features
        return extract_temporal
    
    def _create_spatial_extractor(self):
        """Create spatial feature extractor."""
        def extract_spatial(signal_data, sampling_rate):
            # Simulate spatial feature extraction
            features = {
                "channel_correlation": np.random.uniform(0, 1),
                "spatial_coherence": np.random.uniform(0, 1),
                "laplacian_derivative": np.random.uniform(0, 1),
                "source_localization": np.random.uniform(0, 1)
            }
            return features
        return extract_spatial
    
    def _create_nonlinear_extractor(self):
        """Create nonlinear feature extractor."""
        def extract_nonlinear(signal_data, sampling_rate):
            # Simulate nonlinear feature extraction
            features = {
                "sample_entropy": np.random.uniform(0, 2),
                "approximate_entropy": np.random.uniform(0, 2),
                "lyapunov_exponent": np.random.uniform(-1, 1),
                "correlation_dimension": np.random.uniform(0, 5),
                "fractal_dimension": np.random.uniform(1, 2)
            }
            return features
        return extract_nonlinear
    
    def _create_cognitive_classifier(self):
        """Create cognitive state classifier."""
        def classify_cognitive(features):
            # Simulate cognitive state classification
            # In real implementation, use trained ML models
            if features.get("power_alpha", 0) > 0.7:
                return CognitiveState.RELAXED
            elif features.get("power_beta", 0) > 0.8:
                return CognitiveState.FOCUSED
            elif features.get("power_gamma", 0) > 0.6:
                return CognitiveState.CREATIVE
            else:
                return CognitiveState.NEUTRAL
        return classify_cognitive
    
    def _create_command_classifier(self):
        """Create neural command classifier."""
        def classify_command(features):
            # Simulate command classification
            # In real implementation, use trained ML models
            if features.get("power_alpha", 0) > 0.8:
                return NeuralCommand.PLAY
            elif features.get("power_beta", 0) > 0.8:
                return NeuralCommand.PAUSE
            elif features.get("power_gamma", 0) > 0.7:
                return NeuralCommand.CUT
            else:
                return NeuralCommand.STOP
        return classify_command
    
    def _create_emotion_classifier(self):
        """Create emotion classifier."""
        def classify_emotion(features):
            # Simulate emotion classification
            valence = features.get("power_alpha", 0.5)
            arousal = features.get("power_beta", 0.5)
            
            if valence > 0.7 and arousal > 0.7:
                return "excited"
            elif valence > 0.7 and arousal < 0.3:
                return "calm"
            elif valence < 0.3 and arousal > 0.7:
                return "stressed"
            else:
                return "neutral"
        return classify_emotion
    
    async def process_neural_signal(self, signal: NeuralSignal) -> Dict[str, Any]:
        """Process neural signal and extract features."""
        try:
            # Apply appropriate filter
            filter_func = self.signal_filters.get(signal.signal_type)
            if filter_func:
                filtered_data = filter_func(signal.data, signal.sampling_rate)
            else:
                filtered_data = signal.data
            
            # Extract features
            features = {}
            for feature_type, extractor in self.feature_extractors.items():
                feature_data = extractor(filtered_data, signal.sampling_rate)
                features[feature_type] = feature_data
            
            # Classify cognitive state
            cognitive_classifier = self.classifiers["cognitive_state"]
            cognitive_state = cognitive_classifier(features)
            
            # Classify neural command
            command_classifier = self.classifiers["neural_command"]
            neural_command = command_classifier(features)
            
            # Classify emotion
            emotion_classifier = self.classifiers["emotion"]
            emotion = emotion_classifier(features)
            
            result = {
                "signal_id": signal.signal_id,
                "features": features,
                "cognitive_state": cognitive_state,
                "neural_command": neural_command,
                "emotion": emotion,
                "confidence": np.random.uniform(0.7, 0.95),
                "processing_time": time.time() - signal.timestamp
            }
            
            logger.info(f"Neural signal processed: {signal.signal_id}")
            return result
            
        except Exception as e:
            logger.error(f"Error processing neural signal: {e}")
            return {"error": str(e)}

class BrainStateAnalyzer:
    """Advanced brain state analysis."""
    
    def __init__(self, signal_processor: NeuralSignalProcessor):
        self.signal_processor = signal_processor
        self.user_profiles: Dict[str, CognitiveProfile] = {}
        self.brain_state_history: Dict[str, List[BrainState]] = {}
        
        logger.info("Brain State Analyzer initialized")
    
    def create_cognitive_profile(self, user_id: str) -> str:
        """Create cognitive profile for user."""
        try:
            profile = CognitiveProfile(
                user_id=user_id,
                baseline_state=CognitiveState.NEUTRAL,
                attention_span=0.5,
                creativity_index=0.5,
                stress_threshold=0.7,
                preferred_interface="visual",
                neural_patterns={},
                created_at=time.time()
            )
            
            self.user_profiles[user_id] = profile
            self.brain_state_history[user_id] = []
            
            logger.info(f"Cognitive profile created for user: {user_id}")
            return user_id
            
        except Exception as e:
            logger.error(f"Error creating cognitive profile: {e}")
            raise
    
    async def analyze_brain_state(self, user_id: str, neural_signals: List[NeuralSignal]) -> BrainState:
        """Analyze current brain state from neural signals."""
        try:
            if user_id not in self.user_profiles:
                self.create_cognitive_profile(user_id)
            
            # Process all neural signals
            processed_signals = []
            for signal in neural_signals:
                result = await self.signal_processor.process_neural_signal(signal)
                processed_signals.append(result)
            
            # Aggregate analysis results
            cognitive_states = [s["cognitive_state"] for s in processed_signals]
            emotions = [s["emotion"] for s in processed_signals]
            confidences = [s["confidence"] for s in processed_signals]
            
            # Determine dominant cognitive state
            state_counts = {}
            for state in cognitive_states:
                state_counts[state] = state_counts.get(state, 0) + 1
            dominant_state = max(state_counts, key=state_counts.get)
            
            # Calculate brain state metrics
            attention_level = np.mean([s["features"]["spectral"]["power_beta"] for s in processed_signals])
            stress_level = np.mean([s["features"]["spectral"]["power_gamma"] for s in processed_signals])
            creativity_level = np.mean([s["features"]["nonlinear"]["sample_entropy"] for s in processed_signals])
            fatigue_level = np.mean([s["features"]["spectral"]["power_theta"] for s in processed_signals])
            
            # Calculate emotional metrics
            emotional_valence = np.mean([1 if e in ["excited", "calm"] else -1 for e in emotions])
            arousal_level = np.mean([1 if e in ["excited", "stressed"] else 0 for e in emotions])
            
            # Create brain state
            brain_state = BrainState(
                user_id=user_id,
                cognitive_state=dominant_state,
                attention_level=float(attention_level),
                stress_level=float(stress_level),
                creativity_level=float(creativity_level),
                fatigue_level=float(fatigue_level),
                emotional_valence=float(emotional_valence),
                arousal_level=float(arousal_level),
                timestamp=time.time(),
                neural_metrics={
                    "signal_count": len(neural_signals),
                    "average_confidence": float(np.mean(confidences)),
                    "processing_latency": 0.1  # Simulated
                }
            )
            
            # Store brain state
            self.brain_state_history[user_id].append(brain_state)
            
            # Keep only recent history (last 1000 states)
            if len(self.brain_state_history[user_id]) > 1000:
                self.brain_state_history[user_id] = self.brain_state_history[user_id][-1000:]
            
            logger.info(f"Brain state analyzed for user: {user_id}")
            return brain_state
            
        except Exception as e:
            logger.error(f"Error analyzing brain state: {e}")
            raise
    
    def get_brain_state_trends(self, user_id: str, hours: int = 24) -> Dict[str, Any]:
        """Get brain state trends over time."""
        try:
            if user_id not in self.brain_state_history:
                return {"error": "User not found"}
            
            cutoff_time = time.time() - (hours * 3600)
            recent_states = [s for s in self.brain_state_history[user_id] if s.timestamp > cutoff_time]
            
            if not recent_states:
                return {"message": "No recent brain state data"}
            
            # Calculate trends
            trends = {
                "attention_trend": np.mean([s.attention_level for s in recent_states]),
                "stress_trend": np.mean([s.stress_level for s in recent_states]),
                "creativity_trend": np.mean([s.creativity_level for s in recent_states]),
                "fatigue_trend": np.mean([s.fatigue_level for s in recent_states]),
                "emotional_trend": np.mean([s.emotional_valence for s in recent_states]),
                "arousal_trend": np.mean([s.arousal_level for s in recent_states]),
                "total_states": len(recent_states),
                "time_span_hours": hours
            }
            
            return trends
            
        except Exception as e:
            logger.error(f"Error calculating brain state trends: {e}")
            return {"error": str(e)}

class NeuralCommandInterface:
    """Neural command interface for video editing."""
    
    def __init__(self, signal_processor: NeuralSignalProcessor, brain_analyzer: BrainStateAnalyzer):
        self.signal_processor = signal_processor
        self.brain_analyzer = brain_analyzer
        self.command_history: List[NeuralCommand] = []
        self.active_commands: Dict[str, NeuralCommand] = {}
        
        logger.info("Neural Command Interface initialized")
    
    async def execute_neural_command(self, user_id: str, command_type: NeuralCommand,
                                   confidence: float, context: Dict[str, Any] = None) -> str:
        """Execute neural command."""
        try:
            command = NeuralCommand(
                command_id=str(uuid.uuid4()),
                user_id=user_id,
                command_type=command_type,
                confidence=confidence,
                execution_time=0.0,
                success=False,
                timestamp=time.time(),
                context=context or {}
            )
            
            # Execute command based on type
            start_time = time.time()
            
            if command_type == NeuralCommand.PLAY:
                result = await self._execute_play_command(context)
            elif command_type == NeuralCommand.PAUSE:
                result = await self._execute_pause_command(context)
            elif command_type == NeuralCommand.CUT:
                result = await self._execute_cut_command(context)
            elif command_type == NeuralCommand.COPY:
                result = await self._execute_copy_command(context)
            elif command_type == NeuralCommand.PASTE:
                result = await self._execute_paste_command(context)
            elif command_type == NeuralCommand.ZOOM_IN:
                result = await self._execute_zoom_in_command(context)
            elif command_type == NeuralCommand.ZOOM_OUT:
                result = await self._execute_zoom_out_command(context)
            elif command_type == NeuralCommand.ROTATE:
                result = await self._execute_rotate_command(context)
            elif command_type == NeuralCommand.FILTER:
                result = await self._execute_filter_command(context)
            elif command_type == NeuralCommand.ENHANCE:
                result = await self._execute_enhance_command(context)
            elif command_type == NeuralCommand.SAVE:
                result = await self._execute_save_command(context)
            elif command_type == NeuralCommand.EXPORT:
                result = await self._execute_export_command(context)
            else:
                result = {"success": False, "error": "Unknown command"}
            
            # Update command with results
            command.execution_time = time.time() - start_time
            command.success = result.get("success", False)
            
            # Store command
            self.command_history.append(command)
            if command.success:
                self.active_commands[command.command_id] = command
            
            logger.info(f"Neural command executed: {command.command_id}")
            return command.command_id
            
        except Exception as e:
            logger.error(f"Error executing neural command: {e}")
            raise
    
    async def _execute_play_command(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute play command."""
        try:
            # Simulate play command execution
            await asyncio.sleep(0.1)
            return {"success": True, "action": "play", "timestamp": time.time()}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _execute_pause_command(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute pause command."""
        try:
            await asyncio.sleep(0.1)
            return {"success": True, "action": "pause", "timestamp": time.time()}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _execute_cut_command(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute cut command."""
        try:
            await asyncio.sleep(0.2)
            return {"success": True, "action": "cut", "timestamp": time.time()}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _execute_copy_command(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute copy command."""
        try:
            await asyncio.sleep(0.1)
            return {"success": True, "action": "copy", "timestamp": time.time()}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _execute_paste_command(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute paste command."""
        try:
            await asyncio.sleep(0.1)
            return {"success": True, "action": "paste", "timestamp": time.time()}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _execute_zoom_in_command(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute zoom in command."""
        try:
            await asyncio.sleep(0.1)
            return {"success": True, "action": "zoom_in", "scale": 1.2, "timestamp": time.time()}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _execute_zoom_out_command(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute zoom out command."""
        try:
            await asyncio.sleep(0.1)
            return {"success": True, "action": "zoom_out", "scale": 0.8, "timestamp": time.time()}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _execute_rotate_command(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute rotate command."""
        try:
            await asyncio.sleep(0.1)
            return {"success": True, "action": "rotate", "angle": 90, "timestamp": time.time()}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _execute_filter_command(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute filter command."""
        try:
            await asyncio.sleep(0.3)
            return {"success": True, "action": "filter", "filter_type": "neural_enhanced", "timestamp": time.time()}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _execute_enhance_command(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute enhance command."""
        try:
            await asyncio.sleep(0.5)
            return {"success": True, "action": "enhance", "enhancement_level": "neural_ai", "timestamp": time.time()}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _execute_save_command(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute save command."""
        try:
            await asyncio.sleep(0.2)
            return {"success": True, "action": "save", "timestamp": time.time()}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _execute_export_command(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute export command."""
        try:
            await asyncio.sleep(1.0)
            return {"success": True, "action": "export", "format": "neural_optimized", "timestamp": time.time()}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_command_history(self, user_id: str, limit: int = 100) -> List[NeuralCommand]:
        """Get command history for user."""
        user_commands = [c for c in self.command_history if c.user_id == user_id]
        return user_commands[-limit:] if user_commands else []

class NeuralInterfaceSystem:
    """Main neural interface system."""
    
    def __init__(self):
        self.signal_processor = NeuralSignalProcessor()
        self.brain_analyzer = BrainStateAnalyzer(self.signal_processor)
        self.command_interface = NeuralCommandInterface(self.signal_processor, self.brain_analyzer)
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        
        logger.info("Neural Interface System initialized")
    
    async def start_neural_session(self, user_id: str, session_config: Dict[str, Any]) -> str:
        """Start neural interface session."""
        try:
            session_id = str(uuid.uuid4())
            
            # Create cognitive profile if not exists
            if user_id not in self.brain_analyzer.user_profiles:
                self.brain_analyzer.create_cognitive_profile(user_id)
            
            # Store session
            self.active_sessions[session_id] = {
                "user_id": user_id,
                "session_config": session_config,
                "started_at": time.time(),
                "is_active": True,
                "neural_signals": [],
                "commands_executed": []
            }
            
            logger.info(f"Neural session started: {session_id}")
            return session_id
            
        except Exception as e:
            logger.error(f"Error starting neural session: {e}")
            raise
    
    async def process_neural_input(self, session_id: str, neural_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process neural input and execute commands."""
        try:
            if session_id not in self.active_sessions:
                return {"error": "Session not found"}
            
            session = self.active_sessions[session_id]
            user_id = session["user_id"]
            
            # Create neural signal
            signal = NeuralSignal(
                signal_id=str(uuid.uuid4()),
                signal_type=NeuralSignalType(neural_data.get("signal_type", "eeg")),
                timestamp=time.time(),
                channels=neural_data.get("channels", ["C3", "C4", "P3", "P4"]),
                data=np.array(neural_data.get("data", [])),
                sampling_rate=neural_data.get("sampling_rate", 250.0),
                quality=neural_data.get("quality", 0.9)
            )
            
            # Process signal
            processed_result = await self.signal_processor.process_neural_signal(signal)
            
            # Analyze brain state
            brain_state = await self.brain_analyzer.analyze_brain_state(user_id, [signal])
            
            # Execute neural command if confidence is high enough
            command_result = None
            if processed_result.get("confidence", 0) > 0.8:
                neural_command = processed_result.get("neural_command")
                if neural_command:
                    command_id = await self.command_interface.execute_neural_command(
                        user_id, neural_command, processed_result["confidence"]
                    )
                    command_result = {"command_id": command_id, "command": neural_command.value}
            
            # Store in session
            session["neural_signals"].append(signal)
            if command_result:
                session["commands_executed"].append(command_result)
            
            # Return results
            return {
                "session_id": session_id,
                "processed_result": processed_result,
                "brain_state": asdict(brain_state),
                "command_result": command_result,
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"Error processing neural input: {e}")
            return {"error": str(e)}
    
    def get_session_status(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get neural session status."""
        if session_id in self.active_sessions:
            session = self.active_sessions[session_id]
            return {
                "session_id": session_id,
                "user_id": session["user_id"],
                "started_at": session["started_at"],
                "is_active": session["is_active"],
                "signals_processed": len(session["neural_signals"]),
                "commands_executed": len(session["commands_executed"])
            }
        return None
    
    def end_neural_session(self, session_id: str) -> bool:
        """End neural interface session."""
        try:
            if session_id in self.active_sessions:
                self.active_sessions[session_id]["is_active"] = False
                logger.info(f"Neural session ended: {session_id}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error ending neural session: {e}")
            return False

# Global neural interface system instance
_global_neural_system: Optional[NeuralInterfaceSystem] = None

def get_neural_system() -> NeuralInterfaceSystem:
    """Get the global neural interface system instance."""
    global _global_neural_system
    if _global_neural_system is None:
        _global_neural_system = NeuralInterfaceSystem()
    return _global_neural_system

async def start_neural_session(user_id: str, session_config: Dict[str, Any]) -> str:
    """Start neural interface session."""
    neural_system = get_neural_system()
    return await neural_system.start_neural_session(user_id, session_config)

async def process_neural_input(session_id: str, neural_data: Dict[str, Any]) -> Dict[str, Any]:
    """Process neural input and execute commands."""
    neural_system = get_neural_system()
    return await neural_system.process_neural_input(session_id, neural_data)


