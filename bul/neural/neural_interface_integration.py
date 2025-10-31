"""
Ultimate BUL System - Neural Interface & Brain-Computer Integration
Advanced neural interface and brain-computer integration for direct thought-to-document generation
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

class NeuralSignalType(str, Enum):
    """Neural signal types"""
    EEG = "eeg"
    ECoG = "ecog"
    LFP = "lfp"
    SPIKE = "spike"
    FMRI = "fmri"
    NIRS = "nirs"
    MEG = "meg"
    OPTICAL = "optical"

class BrainRegion(str, Enum):
    """Brain regions"""
    PREFRONTAL_CORTEX = "prefrontal_cortex"
    MOTOR_CORTEX = "motor_cortex"
    VISUAL_CORTEX = "visual_cortex"
    AUDITORY_CORTEX = "auditory_cortex"
    LANGUAGE_CENTERS = "language_centers"
    MEMORY_CENTERS = "memory_centers"
    EMOTIONAL_CENTERS = "emotional_centers"
    DECISION_CENTERS = "decision_centers"

class NeuralInterfaceType(str, Enum):
    """Neural interface types"""
    INVASIVE = "invasive"
    NON_INVASIVE = "non_invasive"
    SEMI_INVASIVE = "semi_invasive"
    OPTICAL = "optical"
    MAGNETIC = "magnetic"
    ELECTRICAL = "electrical"

class NeuralStatus(str, Enum):
    """Neural interface status"""
    CONNECTED = "connected"
    CALIBRATING = "calibrating"
    RECORDING = "recording"
    PROCESSING = "processing"
    LEARNING = "learning"
    DISCONNECTED = "disconnected"
    ERROR = "error"

@dataclass
class NeuralInterface:
    """Neural interface device"""
    id: str
    name: str
    interface_type: NeuralInterfaceType
    signal_types: List[NeuralSignalType]
    brain_regions: List[BrainRegion]
    status: NeuralStatus
    sampling_rate: int
    resolution: int
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_calibration: Optional[datetime] = None
    calibration_data: Dict[str, Any] = field(default_factory=dict)

@dataclass
class NeuralSignal:
    """Neural signal data"""
    id: str
    interface_id: str
    signal_type: NeuralSignalType
    brain_region: BrainRegion
    timestamp: datetime
    data: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ThoughtPattern:
    """Thought pattern recognition"""
    id: str
    pattern_type: str
    confidence: float
    brain_regions: List[BrainRegion]
    signal_characteristics: Dict[str, Any]
    created_at: datetime = field(default_factory=datetime.utcnow)

@dataclass
class NeuralCommand:
    """Neural command for document generation"""
    id: str
    interface_id: str
    command_type: str
    thought_pattern_id: str
    parameters: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    executed: bool = False

class NeuralInterfaceIntegration:
    """Neural interface and brain-computer integration"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.neural_interfaces = {}
        self.neural_signals = {}
        self.thought_patterns = {}
        self.neural_commands = {}
        self.brain_state_models = {}
        self.neural_decoders = {}
        
        # Redis for neural data caching
        self.redis_client = redis.Redis(
            host=config.get("redis_host", "localhost"),
            port=config.get("redis_port", 6379),
            db=config.get("redis_db", 7)
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
            "neural_signals": Counter(
                "bul_neural_signals_total",
                "Total neural signals processed",
                ["interface_id", "signal_type", "brain_region"]
            ),
            "neural_signal_processing_time": Histogram(
                "bul_neural_signal_processing_seconds",
                "Neural signal processing time in seconds",
                ["interface_id", "signal_type"]
            ),
            "thought_patterns": Counter(
                "bul_thought_patterns_total",
                "Total thought patterns recognized",
                ["pattern_type", "confidence_level"]
            ),
            "neural_commands": Counter(
                "bul_neural_commands_total",
                "Total neural commands executed",
                ["interface_id", "command_type"]
            ),
            "neural_interface_status": Gauge(
                "bul_neural_interface_status",
                "Neural interface status",
                ["interface_id", "status"]
            ),
            "brain_activity": Gauge(
                "bul_brain_activity",
                "Brain activity level",
                ["interface_id", "brain_region"]
            ),
            "neural_decoder_accuracy": Gauge(
                "bul_neural_decoder_accuracy",
                "Neural decoder accuracy",
                ["interface_id", "decoder_type"]
            ),
            "thought_to_document_latency": Histogram(
                "bul_thought_to_document_latency_seconds",
                "Thought to document generation latency",
                ["interface_id", "document_type"]
            )
        }
    
    async def start_monitoring(self):
        """Start neural interface monitoring"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        logger.info("Starting neural interface monitoring")
        
        # Start monitoring tasks
        asyncio.create_task(self._monitor_neural_interfaces())
        asyncio.create_task(self._process_neural_signals())
        asyncio.create_task(self._update_metrics())
    
    async def stop_monitoring(self):
        """Stop neural interface monitoring"""
        self.monitoring_active = False
        logger.info("Stopping neural interface monitoring")
    
    async def _monitor_neural_interfaces(self):
        """Monitor neural interfaces"""
        while self.monitoring_active:
            try:
                current_time = datetime.utcnow()
                
                for interface_id, interface in self.neural_interfaces.items():
                    # Update interface status
                    if interface.status == NeuralStatus.CONNECTED:
                        # Simulate brain activity monitoring
                        brain_activity = self._simulate_brain_activity(interface)
                        
                        # Update brain activity metrics
                        for brain_region in interface.brain_regions:
                            self.prometheus_metrics["brain_activity"].labels(
                                interface_id=interface_id,
                                brain_region=brain_region.value
                            ).set(brain_activity.get(brain_region.value, 0.0))
                    
                    # Update interface status metrics
                    self.prometheus_metrics["neural_interface_status"].labels(
                        interface_id=interface_id,
                        status=interface.status.value
                    ).set(1 if interface.status == NeuralStatus.CONNECTED else 0)
                
                await asyncio.sleep(1)  # Monitor every second
                
            except Exception as e:
                logger.error(f"Error monitoring neural interfaces: {e}")
                await asyncio.sleep(5)
    
    async def _process_neural_signals(self):
        """Process neural signals"""
        while self.monitoring_active:
            try:
                # Process signals from all connected interfaces
                for interface_id, interface in self.neural_interfaces.items():
                    if interface.status == NeuralStatus.RECORDING:
                        # Simulate neural signal generation
                        signals = await self._generate_neural_signals(interface)
                        
                        for signal in signals:
                            # Store signal
                            self.neural_signals[signal.id] = signal
                            
                            # Process signal for thought patterns
                            thought_pattern = await self._analyze_neural_signal(signal)
                            
                            if thought_pattern:
                                self.thought_patterns[thought_pattern.id] = thought_pattern
                                
                                # Generate neural command if pattern matches
                                command = await self._generate_neural_command(interface_id, thought_pattern)
                                if command:
                                    self.neural_commands[command.id] = command
                                    
                                    # Execute command
                                    await self._execute_neural_command(command)
                
                await asyncio.sleep(0.1)  # Process every 100ms
                
            except Exception as e:
                logger.error(f"Error processing neural signals: {e}")
                await asyncio.sleep(1)
    
    async def _update_metrics(self):
        """Update Prometheus metrics"""
        while self.monitoring_active:
            try:
                # Update signal counts
                for signal in self.neural_signals.values():
                    self.prometheus_metrics["neural_signals"].labels(
                        interface_id=signal.interface_id,
                        signal_type=signal.signal_type.value,
                        brain_region=signal.brain_region.value
                    ).inc()
                
                # Update thought pattern counts
                for pattern in self.thought_patterns.values():
                    confidence_level = "high" if pattern.confidence > 0.8 else "medium" if pattern.confidence > 0.6 else "low"
                    self.prometheus_metrics["thought_patterns"].labels(
                        pattern_type=pattern.pattern_type,
                        confidence_level=confidence_level
                    ).inc()
                
                # Update command counts
                for command in self.neural_commands.values():
                    if command.executed:
                        self.prometheus_metrics["neural_commands"].labels(
                            interface_id=command.interface_id,
                            command_type=command.command_type
                        ).inc()
                
                await asyncio.sleep(10)  # Update every 10 seconds
                
            except Exception as e:
                logger.error(f"Error updating metrics: {e}")
                await asyncio.sleep(30)
    
    def _simulate_brain_activity(self, interface: NeuralInterface) -> Dict[str, float]:
        """Simulate brain activity for interface"""
        activity = {}
        for brain_region in interface.brain_regions:
            # Simulate realistic brain activity levels
            base_activity = np.random.normal(0.5, 0.1)
            activity[brain_region.value] = max(0.0, min(1.0, base_activity))
        return activity
    
    async def _generate_neural_signals(self, interface: NeuralInterface) -> List[NeuralSignal]:
        """Generate neural signals for interface"""
        signals = []
        
        for signal_type in interface.signal_types:
            for brain_region in interface.brain_regions:
                # Generate realistic neural signal data
                signal_data = np.random.normal(0, 1, (interface.sampling_rate // 10,))  # 100ms of data
                
                signal = NeuralSignal(
                    id=f"neural_signal_{uuid.uuid4().hex[:8]}",
                    interface_id=interface.id,
                    signal_type=signal_type,
                    brain_region=brain_region,
                    timestamp=datetime.utcnow(),
                    data=signal_data,
                    metadata={
                        "sampling_rate": interface.sampling_rate,
                        "resolution": interface.resolution,
                        "amplitude": float(np.mean(np.abs(signal_data))),
                        "frequency": float(np.argmax(np.abs(np.fft.fft(signal_data))))
                    }
                )
                
                signals.append(signal)
        
        return signals
    
    async def _analyze_neural_signal(self, signal: NeuralSignal) -> Optional[ThoughtPattern]:
        """Analyze neural signal for thought patterns"""
        try:
            # Simulate thought pattern analysis
            signal_features = self._extract_signal_features(signal)
            
            # Check for known thought patterns
            pattern_type, confidence = self._classify_thought_pattern(signal_features)
            
            if confidence > 0.6:  # Threshold for pattern recognition
                thought_pattern = ThoughtPattern(
                    id=f"thought_pattern_{uuid.uuid4().hex[:8]}",
                    pattern_type=pattern_type,
                    confidence=confidence,
                    brain_regions=[signal.brain_region],
                    signal_characteristics=signal_features
                )
                
                return thought_pattern
            
            return None
            
        except Exception as e:
            logger.error(f"Error analyzing neural signal: {e}")
            return None
    
    def _extract_signal_features(self, signal: NeuralSignal) -> Dict[str, Any]:
        """Extract features from neural signal"""
        data = signal.data
        
        return {
            "mean_amplitude": float(np.mean(np.abs(data))),
            "std_amplitude": float(np.std(data)),
            "max_amplitude": float(np.max(np.abs(data))),
            "min_amplitude": float(np.min(np.abs(data))),
            "frequency_peak": float(np.argmax(np.abs(np.fft.fft(data)))),
            "power_spectrum": float(np.sum(data ** 2)),
            "zero_crossings": int(np.sum(np.diff(np.sign(data)) != 0)),
            "signal_entropy": float(-np.sum(data * np.log(np.abs(data) + 1e-10)))
        }
    
    def _classify_thought_pattern(self, features: Dict[str, Any]) -> Tuple[str, float]:
        """Classify thought pattern from signal features"""
        # Simulate pattern classification
        patterns = [
            ("document_generation", 0.85),
            ("content_editing", 0.78),
            ("formatting", 0.72),
            ("reviewing", 0.68),
            ("planning", 0.75),
            ("creativity", 0.82),
            ("analysis", 0.70),
            ("decision_making", 0.77)
        ]
        
        # Select pattern based on features
        pattern_scores = []
        for pattern_type, base_confidence in patterns:
            # Adjust confidence based on signal features
            adjusted_confidence = base_confidence * (1 + features["mean_amplitude"] * 0.1)
            adjusted_confidence = min(1.0, adjusted_confidence)
            pattern_scores.append((pattern_type, adjusted_confidence))
        
        # Return pattern with highest confidence
        return max(pattern_scores, key=lambda x: x[1])
    
    async def _generate_neural_command(self, interface_id: str, thought_pattern: ThoughtPattern) -> Optional[NeuralCommand]:
        """Generate neural command from thought pattern"""
        try:
            # Map thought patterns to document generation commands
            command_mapping = {
                "document_generation": "create_document",
                "content_editing": "edit_content",
                "formatting": "apply_formatting",
                "reviewing": "review_document",
                "planning": "create_outline",
                "creativity": "enhance_creativity",
                "analysis": "analyze_content",
                "decision_making": "make_decision"
            }
            
            command_type = command_mapping.get(thought_pattern.pattern_type)
            if not command_type:
                return None
            
            # Generate command parameters based on thought pattern
            parameters = {
                "confidence": thought_pattern.confidence,
                "brain_regions": [region.value for region in thought_pattern.brain_regions],
                "signal_characteristics": thought_pattern.signal_characteristics,
                "timestamp": thought_pattern.created_at.isoformat()
            }
            
            command = NeuralCommand(
                id=f"neural_command_{uuid.uuid4().hex[:8]}",
                interface_id=interface_id,
                command_type=command_type,
                thought_pattern_id=thought_pattern.id,
                parameters=parameters
            )
            
            return command
            
        except Exception as e:
            logger.error(f"Error generating neural command: {e}")
            return None
    
    async def _execute_neural_command(self, command: NeuralCommand):
        """Execute neural command"""
        try:
            start_time = time.time()
            
            # Simulate command execution based on type
            if command.command_type == "create_document":
                result = await self._execute_create_document_command(command)
            elif command.command_type == "edit_content":
                result = await self._execute_edit_content_command(command)
            elif command.command_type == "apply_formatting":
                result = await self._execute_apply_formatting_command(command)
            elif command.command_type == "review_document":
                result = await self._execute_review_document_command(command)
            elif command.command_type == "create_outline":
                result = await self._execute_create_outline_command(command)
            elif command.command_type == "enhance_creativity":
                result = await self._execute_enhance_creativity_command(command)
            elif command.command_type == "analyze_content":
                result = await self._execute_analyze_content_command(command)
            elif command.command_type == "make_decision":
                result = await self._execute_make_decision_command(command)
            else:
                result = {"status": "unknown_command", "message": "Command type not recognized"}
            
            # Mark command as executed
            command.executed = True
            
            # Update metrics
            duration = time.time() - start_time
            self.prometheus_metrics["thought_to_document_latency"].labels(
                interface_id=command.interface_id,
                document_type=command.command_type
            ).observe(duration)
            
            logger.info(f"Executed neural command: {command.id}")
            
        except Exception as e:
            logger.error(f"Error executing neural command: {e}")
    
    async def _execute_create_document_command(self, command: NeuralCommand) -> Dict[str, Any]:
        """Execute create document command"""
        # Simulate document creation from neural command
        await asyncio.sleep(0.5)
        
        return {
            "document_id": f"neural_doc_{uuid.uuid4().hex[:8]}",
            "content": "Neural interface generated document content",
            "confidence": command.parameters.get("confidence", 0.8),
            "brain_regions": command.parameters.get("brain_regions", []),
            "neural_enhanced": True
        }
    
    async def _execute_edit_content_command(self, command: NeuralCommand) -> Dict[str, Any]:
        """Execute edit content command"""
        await asyncio.sleep(0.3)
        
        return {
            "edits_applied": 3,
            "content_improved": True,
            "confidence": command.parameters.get("confidence", 0.8),
            "neural_enhanced": True
        }
    
    async def _execute_apply_formatting_command(self, command: NeuralCommand) -> Dict[str, Any]:
        """Execute apply formatting command"""
        await asyncio.sleep(0.2)
        
        return {
            "formatting_applied": True,
            "style_consistency": 0.95,
            "confidence": command.parameters.get("confidence", 0.8),
            "neural_enhanced": True
        }
    
    async def _execute_review_document_command(self, command: NeuralCommand) -> Dict[str, Any]:
        """Execute review document command"""
        await asyncio.sleep(0.4)
        
        return {
            "review_completed": True,
            "quality_score": 0.92,
            "suggestions": ["Improve clarity", "Add examples", "Enhance structure"],
            "confidence": command.parameters.get("confidence", 0.8),
            "neural_enhanced": True
        }
    
    async def _execute_create_outline_command(self, command: NeuralCommand) -> Dict[str, Any]:
        """Execute create outline command"""
        await asyncio.sleep(0.6)
        
        return {
            "outline_created": True,
            "sections": 5,
            "structure_quality": 0.88,
            "confidence": command.parameters.get("confidence", 0.8),
            "neural_enhanced": True
        }
    
    async def _execute_enhance_creativity_command(self, command: NeuralCommand) -> Dict[str, Any]:
        """Execute enhance creativity command"""
        await asyncio.sleep(0.7)
        
        return {
            "creativity_enhanced": True,
            "creativity_score": 0.94,
            "innovative_elements": 3,
            "confidence": command.parameters.get("confidence", 0.8),
            "neural_enhanced": True
        }
    
    async def _execute_analyze_content_command(self, command: NeuralCommand) -> Dict[str, Any]:
        """Execute analyze content command"""
        await asyncio.sleep(0.8)
        
        return {
            "analysis_completed": True,
            "insights": ["High engagement potential", "Clear structure", "Strong call-to-action"],
            "confidence": command.parameters.get("confidence", 0.8),
            "neural_enhanced": True
        }
    
    async def _execute_make_decision_command(self, command: NeuralCommand) -> Dict[str, Any]:
        """Execute make decision command"""
        await asyncio.sleep(0.9)
        
        return {
            "decision_made": True,
            "decision_confidence": 0.87,
            "reasoning": "Based on neural pattern analysis and content requirements",
            "confidence": command.parameters.get("confidence", 0.8),
            "neural_enhanced": True
        }
    
    def create_neural_interface(self, name: str, interface_type: NeuralInterfaceType,
                              signal_types: List[NeuralSignalType],
                              brain_regions: List[BrainRegion],
                              sampling_rate: int = 1000, resolution: int = 16) -> str:
        """Create neural interface"""
        try:
            interface_id = f"neural_interface_{uuid.uuid4().hex[:8]}"
            
            interface = NeuralInterface(
                id=interface_id,
                name=name,
                interface_type=interface_type,
                signal_types=signal_types,
                brain_regions=brain_regions,
                status=NeuralStatus.DISCONNECTED,
                sampling_rate=sampling_rate,
                resolution=resolution
            )
            
            self.neural_interfaces[interface_id] = interface
            
            logger.info(f"Created neural interface: {interface_id}")
            return interface_id
            
        except Exception as e:
            logger.error(f"Error creating neural interface: {e}")
            raise
    
    async def connect_neural_interface(self, interface_id: str) -> bool:
        """Connect neural interface"""
        try:
            interface = self.neural_interfaces.get(interface_id)
            if not interface:
                return False
            
            interface.status = NeuralStatus.CONNECTED
            
            # Start calibration
            await self._calibrate_neural_interface(interface)
            
            logger.info(f"Connected neural interface: {interface_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error connecting neural interface: {e}")
            return False
    
    async def _calibrate_neural_interface(self, interface: NeuralInterface):
        """Calibrate neural interface"""
        try:
            interface.status = NeuralStatus.CALIBRATING
            
            # Simulate calibration process
            await asyncio.sleep(5)
            
            # Generate calibration data
            calibration_data = {
                "baseline_signals": {},
                "noise_levels": {},
                "signal_quality": {},
                "calibration_timestamp": datetime.utcnow().isoformat()
            }
            
            for signal_type in interface.signal_types:
                for brain_region in interface.brain_regions:
                    key = f"{signal_type.value}_{brain_region.value}"
                    calibration_data["baseline_signals"][key] = np.random.normal(0, 0.1, 100).tolist()
                    calibration_data["noise_levels"][key] = np.random.uniform(0.01, 0.05)
                    calibration_data["signal_quality"][key] = np.random.uniform(0.8, 0.95)
            
            interface.calibration_data = calibration_data
            interface.last_calibration = datetime.utcnow()
            interface.status = NeuralStatus.RECORDING
            
            logger.info(f"Calibrated neural interface: {interface.id}")
            
        except Exception as e:
            logger.error(f"Error calibrating neural interface: {e}")
            interface.status = NeuralStatus.ERROR
    
    async def start_neural_recording(self, interface_id: str) -> bool:
        """Start neural recording"""
        try:
            interface = self.neural_interfaces.get(interface_id)
            if not interface or interface.status != NeuralStatus.CONNECTED:
                return False
            
            interface.status = NeuralStatus.RECORDING
            
            logger.info(f"Started neural recording: {interface_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error starting neural recording: {e}")
            return False
    
    async def stop_neural_recording(self, interface_id: str) -> bool:
        """Stop neural recording"""
        try:
            interface = self.neural_interfaces.get(interface_id)
            if not interface:
                return False
            
            interface.status = NeuralStatus.CONNECTED
            
            logger.info(f"Stopped neural recording: {interface_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error stopping neural recording: {e}")
            return False
    
    def get_neural_interface(self, interface_id: str) -> Optional[NeuralInterface]:
        """Get neural interface by ID"""
        return self.neural_interfaces.get(interface_id)
    
    def list_neural_interfaces(self, status: Optional[NeuralStatus] = None) -> List[NeuralInterface]:
        """List neural interfaces"""
        interfaces = list(self.neural_interfaces.values())
        
        if status:
            interfaces = [i for i in interfaces if i.status == status]
        
        return interfaces
    
    def get_neural_signals(self, interface_id: str) -> List[NeuralSignal]:
        """Get neural signals for interface"""
        return [
            signal for signal in self.neural_signals.values()
            if signal.interface_id == interface_id
        ]
    
    def get_thought_patterns(self, interface_id: str) -> List[ThoughtPattern]:
        """Get thought patterns for interface"""
        # Get thought patterns from neural commands
        command_ids = [
            command.thought_pattern_id for command in self.neural_commands.values()
            if command.interface_id == interface_id
        ]
        
        return [
            pattern for pattern in self.thought_patterns.values()
            if pattern.id in command_ids
        ]
    
    def get_neural_commands(self, interface_id: str) -> List[NeuralCommand]:
        """Get neural commands for interface"""
        return [
            command for command in self.neural_commands.values()
            if command.interface_id == interface_id
        ]
    
    def get_neural_statistics(self) -> Dict[str, Any]:
        """Get neural interface statistics"""
        total_interfaces = len(self.neural_interfaces)
        connected_interfaces = len([i for i in self.neural_interfaces.values() if i.status == NeuralStatus.CONNECTED])
        recording_interfaces = len([i for i in self.neural_interfaces.values() if i.status == NeuralStatus.RECORDING])
        
        total_signals = len(self.neural_signals)
        total_patterns = len(self.thought_patterns)
        total_commands = len(self.neural_commands)
        executed_commands = len([c for c in self.neural_commands.values() if c.executed])
        
        # Count by interface type
        interface_type_counts = {}
        for interface in self.neural_interfaces.values():
            interface_type = interface.interface_type.value
            interface_type_counts[interface_type] = interface_type_counts.get(interface_type, 0) + 1
        
        # Count by signal type
        signal_type_counts = {}
        for signal in self.neural_signals.values():
            signal_type = signal.signal_type.value
            signal_type_counts[signal_type] = signal_type_counts.get(signal_type, 0) + 1
        
        # Count by brain region
        brain_region_counts = {}
        for signal in self.neural_signals.values():
            brain_region = signal.brain_region.value
            brain_region_counts[brain_region] = brain_region_counts.get(brain_region, 0) + 1
        
        # Count by pattern type
        pattern_type_counts = {}
        for pattern in self.thought_patterns.values():
            pattern_type = pattern.pattern_type
            pattern_type_counts[pattern_type] = pattern_type_counts.get(pattern_type, 0) + 1
        
        return {
            "total_interfaces": total_interfaces,
            "connected_interfaces": connected_interfaces,
            "recording_interfaces": recording_interfaces,
            "total_signals": total_signals,
            "total_patterns": total_patterns,
            "total_commands": total_commands,
            "executed_commands": executed_commands,
            "command_success_rate": (executed_commands / total_commands * 100) if total_commands > 0 else 0,
            "interface_type_counts": interface_type_counts,
            "signal_type_counts": signal_type_counts,
            "brain_region_counts": brain_region_counts,
            "pattern_type_counts": pattern_type_counts
        }
    
    def export_neural_data(self) -> Dict[str, Any]:
        """Export neural data for analysis"""
        return {
            "neural_interfaces": [
                {
                    "id": interface.id,
                    "name": interface.name,
                    "interface_type": interface.interface_type.value,
                    "signal_types": [st.value for st in interface.signal_types],
                    "brain_regions": [br.value for br in interface.brain_regions],
                    "status": interface.status.value,
                    "sampling_rate": interface.sampling_rate,
                    "resolution": interface.resolution,
                    "created_at": interface.created_at.isoformat(),
                    "last_calibration": interface.last_calibration.isoformat() if interface.last_calibration else None,
                    "calibration_data": interface.calibration_data
                }
                for interface in self.neural_interfaces.values()
            ],
            "neural_signals": [
                {
                    "id": signal.id,
                    "interface_id": signal.interface_id,
                    "signal_type": signal.signal_type.value,
                    "brain_region": signal.brain_region.value,
                    "timestamp": signal.timestamp.isoformat(),
                    "data": signal.data.tolist(),
                    "metadata": signal.metadata
                }
                for signal in self.neural_signals.values()
            ],
            "thought_patterns": [
                {
                    "id": pattern.id,
                    "pattern_type": pattern.pattern_type,
                    "confidence": pattern.confidence,
                    "brain_regions": [br.value for br in pattern.brain_regions],
                    "signal_characteristics": pattern.signal_characteristics,
                    "created_at": pattern.created_at.isoformat()
                }
                for pattern in self.thought_patterns.values()
            ],
            "neural_commands": [
                {
                    "id": command.id,
                    "interface_id": command.interface_id,
                    "command_type": command.command_type,
                    "thought_pattern_id": command.thought_pattern_id,
                    "parameters": command.parameters,
                    "timestamp": command.timestamp.isoformat(),
                    "executed": command.executed
                }
                for command in self.neural_commands.values()
            ],
            "statistics": self.get_neural_statistics(),
            "export_timestamp": datetime.utcnow().isoformat()
        }

# Global neural interface integration instance
neural_integration = None

def get_neural_integration() -> NeuralInterfaceIntegration:
    """Get the global neural interface integration instance"""
    global neural_integration
    if neural_integration is None:
        config = {
            "redis_host": "localhost",
            "redis_port": 6379,
            "redis_db": 7
        }
        neural_integration = NeuralInterfaceIntegration(config)
    return neural_integration

if __name__ == "__main__":
    # Example usage
    async def main():
        config = {
            "redis_host": "localhost",
            "redis_port": 6379,
            "redis_db": 7
        }
        
        neural = NeuralInterfaceIntegration(config)
        
        # Create neural interface
        interface_id = neural.create_neural_interface(
            name="Brain-Computer Interface",
            interface_type=NeuralInterfaceType.NON_INVASIVE,
            signal_types=[NeuralSignalType.EEG, NeuralSignalType.FMRI],
            brain_regions=[BrainRegion.PREFRONTAL_CORTEX, BrainRegion.LANGUAGE_CENTERS],
            sampling_rate=1000,
            resolution=16
        )
        
        # Connect interface
        await neural.connect_neural_interface(interface_id)
        
        # Start recording
        await neural.start_neural_recording(interface_id)
        
        # Get statistics
        stats = neural.get_neural_statistics()
        print("Neural Interface Statistics:")
        print(json.dumps(stats, indent=2))
        
        await neural.stop_monitoring()
    
    asyncio.run(main())













