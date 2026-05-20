"""
Creating Multimodal Interactive Agents
========================================

Paper: "Creating Multimodal Interactive Agents with"

Key concepts:
- Multi-modal input processing (text, image, audio, video)
- Interactive conversation and task execution
- Cross-modal understanding and generation
- Real-time interaction
- Context preservation across modalities
"""

from typing import Dict, Any, Optional, List
from datetime import datetime

from ..common.agent_base import BaseAgent, AgentState, AgentStatus
from ..common.tools import ToolRegistry

from .models import (
    ModalityType,
    InteractionType,
    MultimodalInput,
    MultimodalOutput,
    Interaction
)
from .modality_processors import ModalityProcessorRegistry
from .modality_generators import ModalityGeneratorRegistry
from .context_analyzer import ContextAnalyzer
from .classifiers import InteractionClassifier


class MultimodalInteractiveAgent(BaseAgent):
    """
    Multimodal interactive agent.
    
    Processes inputs from multiple modalities and generates
    appropriate responses across modalities.
    """
    
    def __init__(
        self,
        name: str,
        tool_registry: Optional[ToolRegistry] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize multimodal interactive agent.
        
        Args:
            name: Agent name
            tool_registry: Registry of available tools
            config: Additional configuration
        """
        super().__init__(name, config)
        self.tool_registry = tool_registry or ToolRegistry()
        
        # Specialized components
        self.processor_registry = ModalityProcessorRegistry()
        self.generator_registry = ModalityGeneratorRegistry()
        self.context_analyzer = ContextAnalyzer()
        self.interaction_classifier = InteractionClassifier()
        
        # Interaction history
        self.interaction_history: List[Interaction] = []
        
        # Multimodal context
        self.multimodal_context: Dict[ModalityType, List[Any]] = {
            modality: [] for modality in ModalityType
        }
        
        # Current conversation context
        self.conversation_context: List[Dict[str, Any]] = []
    
    def think(self, task: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Think about multimodal task.
        
        Args:
            task: Task description
            context: Additional context (may include multimodal data)
            
        Returns:
            Thinking result
        """
        self.state.status = AgentStatus.THINKING
        self.state.current_task = task
        
        # Process multimodal context
        context_analysis = self.context_analyzer.analyze(context)
        
        # Determine required modalities
        required_modalities = self.context_analyzer.determine_required_modalities(
            task, context
        )
        
        # Generate reasoning
        reasoning = self.context_analyzer.generate_reasoning(task, context_analysis)
        
        result = {
            "task": task,
            "reasoning": reasoning,
            "context_analysis": context_analysis,
            "required_modalities": [m.value for m in required_modalities],
            "available_modalities": [
                m.value for m, content in self.multimodal_context.items()
                if content
            ]
        }
        
        self.state.add_step("thinking", result)
        return result
    
    def act(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute action (may involve multiple modalities).
        
        Args:
            action: Action to execute
            
        Returns:
            Action result
        """
        self.state.status = AgentStatus.ACTING
        
        action_type = action.get("type", "process")
        
        # Execute action based on type
        if action_type == "process":
            result = self._process_multimodal(action)
        elif action_type == "generate":
            result = self._generate_multimodal(action)
        elif action_type == "tool_call":
            result = self._execute_tool_action(action)
        else:
            result = {"status": "unknown_action", "action": action}
        
        action_record = {
            "action": action,
            "result": result,
            "timestamp": datetime.now().isoformat()
        }
        self.state.add_step("action", action_record)
        
        return result
    
    def _process_multimodal(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Process multimodal input."""
        modality_str = action.get("modality", "text")
        try:
            modality = ModalityType(modality_str)
        except ValueError:
            modality = ModalityType.TEXT
        
        content = action.get("content")
        
        # Process using registry
        processed = self.processor_registry.process(modality, content)
        
        # Store in context
        self.multimodal_context[modality].append(processed)
        
        return {
            "status": "processed",
            "modality": modality.value,
            "processed_content": processed
        }
    
    def _generate_multimodal(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Generate multimodal output."""
        output_modality = action.get("modality", ModalityType.TEXT)
        prompt = action.get("prompt", "")
        
        # Convert string to ModalityType if needed
        if isinstance(output_modality, str):
            try:
                output_modality = ModalityType(output_modality)
            except ValueError:
                output_modality = ModalityType.TEXT
        
        # Generate using registry
        generated = self.generator_registry.generate(output_modality, prompt)
        
        return {
            "status": "generated",
            "modality": output_modality.value,
            "generated_content": generated
        }
    
    def _execute_tool_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Execute tool action."""
        tool_name = action.get("tool")
        tool_args = action.get("args", {})
        
        if self.tool_registry.has_tool(tool_name):
            tool = self.tool_registry.get_tool(tool_name)
            result = tool(**tool_args)
        else:
            result = {"error": f"Tool {tool_name} not found"}
        
        return {
            "status": "tool_executed",
            "tool": tool_name,
            "result": result
        }
    
    def observe(self, observation: Any) -> Dict[str, Any]:
        """
        Process observation (may be multimodal).
        
        Args:
            observation: Observation data
            
        Returns:
            Processed observation
        """
        self.state.status = AgentStatus.OBSERVING
        
        # Determine observation modality
        if isinstance(observation, dict):
            modality = self.context_analyzer.detect_modality(observation)
        else:
            modality = ModalityType.TEXT
        
        # Process observation
        processed = {
            "observation": observation,
            "modality": modality.value,
            "timestamp": datetime.now().isoformat()
        }
        
        # Update context
        self.multimodal_context[modality].append(observation)
        
        self.state.add_step("observation", processed)
        return processed
    
    def interact(
        self,
        input_data: MultimodalInput
    ) -> MultimodalOutput:
        """
        Interact with the agent using multimodal input.
        
        Args:
            input_data: Multimodal input
            
        Returns:
            Multimodal output
        """
        # Determine interaction type
        interaction_type = self.interaction_classifier.classify(input_data)
        
        # Process input
        self._process_input(input_data)
        
        # Generate response
        output = self._generate_response(input_data, interaction_type)
        
        # Create interaction record
        interaction = Interaction(
            interaction_id=f"interaction_{datetime.now().timestamp()}",
            interaction_type=interaction_type,
            input=input_data,
            output=output,
            context={"conversation_turn": len(self.conversation_context)}
        )
        
        self.interaction_history.append(interaction)
        self.conversation_context.append({
            "input": input_data.content,
            "output": output.content,
            "modality": input_data.modality.value
        })
        
        return output
    
    def _process_input(self, input_data: MultimodalInput):
        """Process and store input."""
        self.multimodal_context[input_data.modality].append(input_data.content)
    
    def _generate_response(
        self,
        input_data: MultimodalInput,
        interaction_type: InteractionType
    ) -> MultimodalOutput:
        """Generate response to input."""
        # Generate response based on interaction type and modality
        if interaction_type == InteractionType.QUESTION:
            response_content = f"Answer to question: {input_data.content}"
        elif interaction_type == InteractionType.COMMAND:
            response_content = f"Executing command: {input_data.content}"
        else:
            response_content = f"Response to: {input_data.content}"
        
        output = MultimodalOutput(
            output_id=f"output_{datetime.now().timestamp()}",
            modality=input_data.modality,  # Respond in same modality
            content=response_content,
            metadata={"interaction_type": interaction_type.value}
        )
        
        return output
    
    def run(self, task: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Run multimodal task.
        
        Args:
            task: Task description
            
        Returns:
            Final result
        """
        # Think about task
        thinking = self.think(task)
        
        # Process task as multimodal input
        input_data = MultimodalInput(
            input_id=f"input_{datetime.now().timestamp()}",
            modality=ModalityType.TEXT,
            content=task
        )
        
        # Interact
        output = self.interact(input_data)
        
        self.state.status = AgentStatus.COMPLETED
        
        return {
            "task": task,
            "thinking": thinking,
            "input_modality": input_data.modality.value,
            "output_modality": output.modality.value,
            "output": output.content,
            "interactions_count": len(self.interaction_history)
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get current agent status."""
        from ..common.agent_utils import create_status_dict
        return create_status_dict(self, {
            "interactions": len(self.interaction_history),
            "modalities_available": [
                m.value for m, content in self.multimodal_context.items()
                if content
            ],
            "conversation_turns": len(self.conversation_context)
        })



