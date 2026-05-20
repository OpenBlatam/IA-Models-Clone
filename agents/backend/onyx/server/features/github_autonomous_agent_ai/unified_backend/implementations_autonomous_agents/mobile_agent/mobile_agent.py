"""
MOBILE-AGENT: Autonomous Multi-Modal Mobile Agent
================================================

Paper: "MOBILE-AGENT: AUTONOMOUS MULTI-MODAL MOBILE"

Key concepts:
- Multi-modal perception (vision, text, audio)
- Mobile device interaction
- Screen understanding and navigation
- Cross-platform compatibility
- Autonomous task execution on mobile devices
"""

from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

from ..common.agent_base import BaseAgent, AgentState, AgentStatus
from ..common.tools import ToolRegistry


class ModalityType(Enum):
    """Types of input modalities."""
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    SCREEN = "screen"
    TOUCH = "touch"


class ActionType(Enum):
    """Types of mobile actions."""
    TAP = "tap"
    SWIPE = "swipe"
    TYPE = "type"
    SCROLL = "scroll"
    LONG_PRESS = "long_press"
    BACK = "back"
    HOME = "home"
    SCREENSHOT = "screenshot"


@dataclass
class ScreenElement:
    """Represents an element on the screen."""
    element_id: str
    element_type: str  # button, text, image, etc.
    bounds: Tuple[int, int, int, int]  # x, y, width, height
    text: Optional[str] = None
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ScreenState:
    """Current state of the mobile screen."""
    screenshot_path: Optional[str] = None
    elements: List[ScreenElement] = field(default_factory=list)
    app_name: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class MobileAction:
    """A mobile device action."""
    action_id: str
    action_type: ActionType
    target: Optional[ScreenElement] = None
    coordinates: Optional[Tuple[int, int]] = None
    text: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


class MobileAgent(BaseAgent):
    """
    Autonomous multi-modal mobile agent.
    
    Can interact with mobile devices through various modalities
    and execute tasks autonomously.
    """
    
    def __init__(
        self,
        name: str,
        tool_registry: Optional[ToolRegistry] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize mobile agent.
        
        Args:
            name: Agent name
            tool_registry: Registry of available tools
            config: Additional configuration
        """
        super().__init__(name, config)
        self.tool_registry = tool_registry or ToolRegistry()
        
        # Screen state
        self.current_screen: Optional[ScreenState] = None
        self.screen_history: List[ScreenState] = []
        
        # Action history
        self.action_history: List[MobileAction] = []
        
        # Multi-modal inputs
        self.multimodal_context: Dict[ModalityType, Any] = {}
        
        # Task tracking
        self.current_task: Optional[str] = None
        self.task_steps: List[Dict[str, Any]] = []
    
    def think(self, task: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Think about a mobile task.
        
        Args:
            task: Task description
            context: Additional context
            
        Returns:
            Thinking result
        """
        self.state.status = AgentStatus.THINKING
        self.state.current_task = task
        self.current_task = task
        
        # Analyze task and break into steps
        task_steps = self._decompose_task(task)
        self.task_steps = task_steps
        
        # Consider current screen state
        screen_analysis = self._analyze_screen() if self.current_screen else None
        
        result = {
            "task": task,
            "task_steps": task_steps,
            "screen_analysis": screen_analysis,
            "multimodal_context": {
                modality.value: bool(content)
                for modality, content in self.multimodal_context.items()
            }
        }
        
        self.state.add_step("thinking", result)
        return result
    
    def _decompose_task(self, task: str) -> List[Dict[str, Any]]:
        """Decompose task into actionable steps."""
        steps = []
        
        # Simple task decomposition (in production, use LLM)
        task_lower = task.lower()
        
        if "open" in task_lower:
            steps.append({
                "step": "open_app",
                "description": f"Open application mentioned in task",
                "action_type": ActionType.TAP
            })
        
        if "search" in task_lower or "find" in task_lower:
            steps.append({
                "step": "search",
                "description": "Perform search operation",
                "action_type": ActionType.TYPE
            })
        
        if "click" in task_lower or "tap" in task_lower:
            steps.append({
                "step": "interact",
                "description": "Tap on target element",
                "action_type": ActionType.TAP
            })
        
        if not steps:
            steps.append({
                "step": "execute",
                "description": task,
                "action_type": ActionType.TAP
            })
        
        return steps
    
    def _analyze_screen(self) -> Dict[str, Any]:
        """Analyze current screen state."""
        if not self.current_screen:
            return {}
        
        return {
            "app_name": self.current_screen.app_name,
            "elements_count": len(self.current_screen.elements),
            "interactable_elements": [
                e.element_id for e in self.current_screen.elements
                if e.element_type in ["button", "link", "input"]
            ],
            "text_elements": [
                e.text for e in self.current_screen.elements
                if e.text
            ]
        }
    
    def act(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a mobile action.
        
        Args:
            action: Action to execute
            
        Returns:
            Action result
        """
        self.state.status = AgentStatus.ACTING
        
        # Create mobile action
        mobile_action = self._create_mobile_action(action)
        self.action_history.append(mobile_action)
        
        # Execute action (placeholder - in production, use ADB or similar)
        result = self._execute_mobile_action(mobile_action)
        
        # Update screen state after action
        self._update_screen_after_action(mobile_action)
        
        action_result = {
            "action": mobile_action,
            "result": result,
            "timestamp": datetime.now().isoformat()
        }
        
        self.state.add_step("action", action_result)
        return action_result
    
    def _create_mobile_action(self, action: Dict[str, Any]) -> MobileAction:
        """Create a MobileAction from action dict."""
        action_type_str = action.get("action_type", "tap")
        try:
            action_type = ActionType(action_type_str)
        except ValueError:
            action_type = ActionType.TAP
        
        # Find target element if specified
        target = None
        if "target_element_id" in action:
            target = self._find_element(action["target_element_id"])
        
        return MobileAction(
            action_id=f"action_{datetime.now().timestamp()}",
            action_type=action_type,
            target=target,
            coordinates=action.get("coordinates"),
            text=action.get("text"),
            parameters=action.get("parameters", {})
        )
    
    def _find_element(self, element_id: str) -> Optional[ScreenElement]:
        """Find element by ID in current screen."""
        if not self.current_screen:
            return None
        
        for element in self.current_screen.elements:
            if element.element_id == element_id:
                return element
        
        return None
    
    def _execute_mobile_action(self, action: MobileAction) -> Dict[str, Any]:
        """Execute mobile action (placeholder)."""
        # In production, this would use ADB, Appium, or similar
        return {
            "status": "executed",
            "action_type": action.action_type.value,
            "success": True,
            "message": f"Executed {action.action_type.value} action"
        }
    
    def _update_screen_after_action(self, action: MobileAction):
        """Update screen state after action execution."""
        # In production, capture new screenshot and parse elements
        # For now, create placeholder screen state
        if action.action_type == ActionType.SCREENSHOT:
            self.capture_screen()
    
    def observe(self, observation: Any) -> Dict[str, Any]:
        """
        Process observation from mobile device.
        
        Args:
            observation: Observation data (can be screen, text, image, etc.)
            
        Returns:
            Processed observation
        """
        self.state.status = AgentStatus.OBSERVING
        
        # Determine observation type
        if isinstance(observation, dict):
            if "screen" in observation:
                self._process_screen_observation(observation["screen"])
            if "text" in observation:
                self.multimodal_context[ModalityType.TEXT] = observation["text"]
            if "image" in observation:
                self.multimodal_context[ModalityType.IMAGE] = observation["image"]
            if "audio" in observation:
                self.multimodal_context[ModalityType.AUDIO] = observation["audio"]
        
        processed = {
            "observation": observation,
            "screen_state": self.current_screen.app_name if self.current_screen else None,
            "timestamp": datetime.now().isoformat()
        }
        
        self.state.add_step("observation", processed)
        return processed
    
    def _process_screen_observation(self, screen_data: Dict[str, Any]):
        """Process screen observation and update screen state."""
        elements = []
        for elem_data in screen_data.get("elements", []):
            element = ScreenElement(
                element_id=elem_data.get("id", ""),
                element_type=elem_data.get("type", "unknown"),
                bounds=tuple(elem_data.get("bounds", (0, 0, 0, 0))),
                text=elem_data.get("text"),
                attributes=elem_data.get("attributes", {})
            )
            elements.append(element)
        
        self.current_screen = ScreenState(
            screenshot_path=screen_data.get("screenshot"),
            elements=elements,
            app_name=screen_data.get("app_name")
        )
        self.screen_history.append(self.current_screen)
    
    def capture_screen(self) -> ScreenState:
        """
        Capture current screen state.
        
        Returns:
            Screen state
        """
        # In production, use ADB or similar to capture screenshot
        # and parse UI elements using OCR or accessibility services
        
        # Placeholder
        self.current_screen = ScreenState(
            screenshot_path=f"screenshot_{datetime.now().timestamp()}.png",
            elements=[],
            app_name="unknown"
        )
        
        return self.current_screen
    
    def find_element_by_text(self, text: str) -> Optional[ScreenElement]:
        """
        Find element on screen by text.
        
        Args:
            text: Text to search for
            
        Returns:
            Screen element if found
        """
        if not self.current_screen:
            return None
        
        for element in self.current_screen.elements:
            if element.text and text.lower() in element.text.lower():
                return element
        
        return None
    
    def tap_element(self, element: ScreenElement) -> Dict[str, Any]:
        """
        Tap on a screen element.
        
        Args:
            element: Element to tap
            
        Returns:
            Action result
        """
        action = {
            "action_type": ActionType.TAP.value,
            "target_element_id": element.element_id,
            "coordinates": (
                element.bounds[0] + element.bounds[2] // 2,
                element.bounds[1] + element.bounds[3] // 2
            )
        }
        
        return self.act(action)
    
    def type_text(self, text: str, target_element: Optional[ScreenElement] = None) -> Dict[str, Any]:
        """
        Type text on screen.
        
        Args:
            text: Text to type
            target_element: Target element (if None, uses focused element)
            
        Returns:
            Action result
        """
        action = {
            "action_type": ActionType.TYPE.value,
            "text": text,
            "target_element_id": target_element.element_id if target_element else None
        }
        
        return self.act(action)
    
    def run(self, task: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Run complete mobile task execution.
        
        Args:
            task: Task to execute
            context: Optional context
            
        Returns:
            Final result
        """
        # Step 1: Capture initial screen
        initial_screen = self.capture_screen()
        self.observe({"screen": {"app_name": initial_screen.app_name}})
        
        # Step 2: Think about task
        thinking = self.think(task, context)
        
        # Step 3: Execute task steps
        results = []
        for step in self.task_steps:
            action = {
                "action_type": step["action_type"].value if isinstance(step["action_type"], ActionType) else step["action_type"],
                "description": step["description"]
            }
            
            result = self.act(action)
            results.append(result)
            
            # Observe after each action
            new_screen = self.capture_screen()
            self.observe({"screen": {"app_name": new_screen.app_name}})
        
        self.state.status = AgentStatus.COMPLETED
        
        return {
            "task": task,
            "initial_screen": initial_screen.app_name,
            "thinking": thinking,
            "steps_executed": len(results),
            "results": results,
            "final_screen": self.current_screen.app_name if self.current_screen else None
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get current agent status."""
        from ..common.agent_utils import create_status_dict
        return create_status_dict(self, {
            "current_app": self.current_screen.app_name if self.current_screen else None,
            "actions_executed": len(self.action_history),
            "screens_captured": len(self.screen_history)
        })



