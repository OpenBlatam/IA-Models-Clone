"""
Design of Seamless Multi-modal Interaction Framework
=====================================================

Paper: "Design of Seamless Multi-modal Interaction Framework for..."

Key concepts:
- Seamless multi-modal interactions
- Cross-modal understanding
- Unified interaction framework
- Multi-modal fusion
- Context-aware modality switching
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

from ..common.agent_base import BaseAgent, AgentState, AgentStatus
from ..common.memory import EpisodicMemory, SemanticMemory


class Modality(Enum):
    """Input/output modalities."""
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    GESTURE = "gesture"
    TOUCH = "touch"


class FusionStrategy(Enum):
    """Multi-modal fusion strategies."""
    EARLY_FUSION = "early_fusion"
    LATE_FUSION = "late_fusion"
    ATTENTION_FUSION = "attention_fusion"
    HIERARCHICAL_FUSION = "hierarchical_fusion"


@dataclass
class MultimodalInput:
    """Multi-modal input representation."""
    input_id: str
    modalities: List[Modality]
    content: Dict[Modality, Any]
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class MultimodalOutput:
    """Multi-modal output representation."""
    output_id: str
    modalities: List[Modality]
    content: Dict[Modality, Any]
    fusion_strategy: FusionStrategy
    timestamp: datetime = field(default_factory=datetime.now)


class SeamlessMultimodalAgent(BaseAgent):
    """
    Agent with seamless multi-modal interaction capabilities.
    
    Handles multiple input/output modalities with seamless
    transitions and cross-modal understanding.
    """
    
    def __init__(
        self,
        name: str,
        supported_modalities: Optional[List[Modality]] = None,
        fusion_strategy: FusionStrategy = FusionStrategy.ATTENTION_FUSION,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize seamless multimodal agent.
        
        Args:
            name: Agent name
            supported_modalities: List of supported modalities
            fusion_strategy: Multi-modal fusion strategy
            config: Additional configuration
        """
        super().__init__(name, config)
        
        # Modality support
        self.supported_modalities = supported_modalities or [
            Modality.TEXT, Modality.IMAGE, Modality.AUDIO
        ]
        self.fusion_strategy = fusion_strategy
        
        # Interaction tracking
        self.multimodal_inputs: List[MultimodalInput] = []
        self.multimodal_outputs: List[MultimodalOutput] = []
        self.modality_switches: List[Dict[str, Any]] = []
        
        # Metrics
        self.interactions_processed = 0
        self.modality_switches_count = 0
        self.fusion_operations = 0
        
        # Memory systems
        self.episodic_memory = EpisodicMemory()
        self.semantic_memory = SemanticMemory()
        
        # Cross-modal understanding
        self.cross_modal_mappings: Dict[str, Any] = {}
        self.context_history: List[Dict[str, Any]] = []
    
    def think(self, task: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Think about task with multi-modal awareness.
        
        Args:
            task: Task description
            context: Additional context (may contain multimodal data)
            
        Returns:
            Thinking result with modality analysis
        """
        self.state.status = AgentStatus.THINKING
        self.state.current_task = task
        
        # Extract modalities from context
        modalities_detected = self._detect_modalities(context)
        
        # Analyze cross-modal relationships
        cross_modal_analysis = self._analyze_cross_modal(context)
        
        # Determine fusion approach
        fusion_approach = self._determine_fusion_approach(modalities_detected)
        
        result = {
            "task": task,
            "modalities_detected": [m.value for m in modalities_detected],
            "cross_modal_analysis": cross_modal_analysis,
            "fusion_approach": fusion_approach,
            "fusion_strategy": self.fusion_strategy.value
        }
        
        self.state.add_step("think", result)
        return result
    
    def _detect_modalities(self, context: Optional[Dict[str, Any]]) -> List[Modality]:
        """Detect modalities present in context."""
        modalities = []
        
        if context:
            if context.get("text"):
                modalities.append(Modality.TEXT)
            if context.get("image") or context.get("image_data"):
                modalities.append(Modality.IMAGE)
            if context.get("audio") or context.get("audio_data"):
                modalities.append(Modality.AUDIO)
            if context.get("video") or context.get("video_data"):
                modalities.append(Modality.VIDEO)
            if context.get("gesture"):
                modalities.append(Modality.GESTURE)
            if context.get("touch"):
                modalities.append(Modality.TOUCH)
        
        # Default to text if no modalities detected
        if not modalities:
            modalities.append(Modality.TEXT)
        
        return modalities
    
    def _analyze_cross_modal(self, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze cross-modal relationships."""
        if not context:
            return {"relationships": []}
        
        relationships = []
        
        # Check for text-image relationships
        if context.get("text") and context.get("image"):
            relationships.append({
                "type": "text_image",
                "relationship": "complementary",
                "strength": 0.8
            })
        
        # Check for audio-text relationships
        if context.get("audio") and context.get("text"):
            relationships.append({
                "type": "audio_text",
                "relationship": "transcript",
                "strength": 0.9
            })
        
        return {
            "relationships": relationships,
            "cross_modal_understanding": len(relationships) > 0
        }
    
    def _determine_fusion_approach(self, modalities: List[Modality]) -> Dict[str, Any]:
        """Determine fusion approach based on modalities."""
        if len(modalities) == 1:
            return {
                "fusion_needed": False,
                "approach": "single_modality"
            }
        
        return {
            "fusion_needed": True,
            "approach": self.fusion_strategy.value,
            "modalities_count": len(modalities)
        }
    
    def act(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute action with multi-modal output.
        
        Args:
            action: Action to execute
            
        Returns:
            Action result with multimodal output
        """
        self.state.status = AgentStatus.ACTING
        
        # Determine output modalities
        output_modalities = self._determine_output_modalities(action)
        
        # Generate multimodal output
        multimodal_output = self._generate_multimodal_output(action, output_modalities)
        
        # Track interaction
        self.interactions_processed += 1
        self.fusion_operations += 1
        
        self.state.add_step("act", {
            "action": action,
            "multimodal_output": {
                "modalities": [m.value for m in output_modalities],
                "fusion_strategy": self.fusion_strategy.value
            }
        })
        
        return {
            "status": "executed",
            "action": action,
            "multimodal_output": multimodal_output,
            "timestamp": datetime.now().isoformat()
        }
    
    def _determine_output_modalities(self, action: Dict[str, Any]) -> List[Modality]:
        """Determine appropriate output modalities for action."""
        action_type = action.get("type", "execute")
        
        # Default output modalities
        output_modalities = [Modality.TEXT]
        
        # Add modalities based on action type
        if action_type == "visualize" or action_type == "show":
            output_modalities.append(Modality.IMAGE)
        elif action_type == "explain_audio" or action_type == "speak":
            output_modalities.append(Modality.AUDIO)
        elif action_type == "demonstrate":
            output_modalities.extend([Modality.VIDEO, Modality.AUDIO])
        
        # Filter to supported modalities
        return [m for m in output_modalities if m in self.supported_modalities]
    
    def _generate_multimodal_output(
        self,
        action: Dict[str, Any],
        modalities: List[Modality]
    ) -> Dict[str, Any]:
        """Generate multimodal output."""
        output_content = {}
        
        for modality in modalities:
            if modality == Modality.TEXT:
                output_content[modality] = f"Text response for {action.get('type', 'action')}"
            elif modality == Modality.IMAGE:
                output_content[modality] = {"image_description": "Generated image"}
            elif modality == Modality.AUDIO:
                output_content[modality] = {"audio_description": "Generated audio"}
            elif modality == Modality.VIDEO:
                output_content[modality] = {"video_description": "Generated video"}
        
        # Create multimodal output object
        multimodal_output = MultimodalOutput(
            output_id=f"output_{datetime.now().timestamp()}",
            modalities=modalities,
            content=output_content,
            fusion_strategy=self.fusion_strategy
        )
        
        self.multimodal_outputs.append(multimodal_output)
        
        return {
            "modalities": [m.value for m in modalities],
            "content": {k.value: v for k, v in output_content.items()},
            "fused": len(modalities) > 1
        }
    
    def observe(self, observation: Any) -> Dict[str, Any]:
        """
        Process multimodal observation.
        
        Args:
            observation: Observation data (may be multimodal)
            
        Returns:
            Processed observation
        """
        from ..common.agent_utils import standard_observe_pattern
        
        self.state.status = AgentStatus.OBSERVING
        
        # Extract modalities from observation
        if isinstance(observation, dict):
            modalities = self._detect_modalities(observation)
            
            # Track modality switches
            if len(modalities) > 1:
                self.modality_switches_count += 1
                self.modality_switches.append({
                    "modalities": [m.value for m in modalities],
                    "timestamp": datetime.now().isoformat()
                })
            
            # Create multimodal input
            if modalities:
                multimodal_input = MultimodalInput(
                    input_id=f"input_{datetime.now().timestamp()}",
                    modalities=modalities,
                    content={m: observation.get(m.value, {}) for m in modalities}
                )
                self.multimodal_inputs.append(multimodal_input)
        
        # Use standard observe pattern
        return standard_observe_pattern(
            self,
            observation,
            importance=0.7,
            additional_data={
                "supported_modalities": [m.value for m in self.supported_modalities],
                "modality_switches": self.modality_switches_count,
                "interactions_processed": self.interactions_processed
            }
        )
    
    def run(self, task: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Run task with seamless multimodal interaction.
        
        Args:
            task: Task description
            context: Optional context (may contain multimodal data)
            
        Returns:
            Final result
        """
        from ..common.agent_utils import standard_run_pattern
        
        # Prepare context
        if context is None:
            context = {}
        
        context["multimodal_enabled"] = True
        context["supported_modalities"] = [m.value for m in self.supported_modalities]
        
        # Use standard pattern
        result = standard_run_pattern(self, task, context)
        
        # Add multimodal information
        result["multimodal_summary"] = {
            "supported_modalities": [m.value for m in self.supported_modalities],
            "interactions_processed": self.interactions_processed,
            "modality_switches": self.modality_switches_count,
            "fusion_operations": self.fusion_operations,
            "fusion_strategy": self.fusion_strategy.value
        }
        
        return result
    
    def get_status(self) -> Dict[str, Any]:
        """Get current agent status."""
        from ..common.agent_utils import create_status_dict
        return create_status_dict(self, {
            "supported_modalities": [m.value for m in self.supported_modalities],
            "interactions_processed": self.interactions_processed,
            "modality_switches": self.modality_switches_count,
            "fusion_operations": self.fusion_operations,
            "fusion_strategy": self.fusion_strategy.value
        })
    
    def switch_modality(self, from_modality: Modality, to_modality: Modality) -> bool:
        """Switch between modalities."""
        if to_modality not in self.supported_modalities:
            return False
        
        self.modality_switches.append({
            "from": from_modality.value,
            "to": to_modality.value,
            "timestamp": datetime.now().isoformat()
        })
        self.modality_switches_count += 1
        
        return True


