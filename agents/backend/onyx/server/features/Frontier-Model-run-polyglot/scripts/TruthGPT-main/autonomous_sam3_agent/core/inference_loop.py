"""
Agent Inference Loop
====================

Manages the agent inference loop with tool call handling.

Refactored with:
- Handler registry pattern
- State management dataclass
- Plugin hooks integration
- Improved error handling
"""

import json
import logging
from typing import Dict, Any, List, Set, Optional, Callable, Awaitable
from pathlib import Path
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

from ..infrastructure.openrouter_client import OpenRouterClient
from ..infrastructure.sam3_client import SAM3Client
from .message_preparer import MessagePreparer
from .tool_call_parser import ToolCallParser
from .tool_call_handlers import (
    SegmentPhraseHandler,
    ExamineEachMaskHandler,
    SelectMasksHandler,
    ReportNoMaskHandler,
)

logger = logging.getLogger(__name__)


@dataclass
class InferenceState:
    """
    Encapsulates the state of an inference loop run.
    
    This makes state management explicit and easier to track.
    """
    image_path: str
    initial_text_prompt: str
    task_id: str
    messages: List[Dict] = field(default_factory=list)
    used_text_prompts: Set[str] = field(default_factory=set)
    latest_sam3_text_prompt: str = ""
    path_to_latest_output_json: str = ""
    generation_count: int = 0
    max_generations: int = 100
    
    @property
    def has_exceeded_limit(self) -> bool:
        """Check if generation limit exceeded."""
        return self.generation_count >= self.max_generations
    
    def increment_generation(self):
        """Increment generation counter."""
        self.generation_count += 1


class ToolHandler(ABC):
    """Abstract base class for tool handlers."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Tool name this handler handles."""
        pass
    
    @abstractmethod
    async def handle(
        self,
        tool_call: Dict[str, Any],
        state: InferenceState,
        **kwargs
    ) -> Dict[str, Any]:
        """Handle a tool call."""
        pass


class ToolHandlerRegistry:
    """
    Registry for tool handlers.
    
    Provides a clean way to register and lookup handlers.
    """
    
    def __init__(self):
        self._handlers: Dict[str, Callable] = {}
    
    def register(self, name: str, handler: Callable):
        """Register a handler for a tool name."""
        self._handlers[name] = handler
        logger.debug(f"Registered tool handler: {name}")
    
    def get(self, name: str) -> Optional[Callable]:
        """Get handler for a tool name."""
        return self._handlers.get(name)
    
    def has(self, name: str) -> bool:
        """Check if handler exists."""
        return name in self._handlers
    
    @property
    def available_tools(self) -> List[str]:
        """List of available tool names."""
        return list(self._handlers.keys())


class AgentInferenceLoop:
    """
    Manages the agent inference loop.
    
    Responsibilities:
    - Run agent inference loop
    - Manage conversation state
    - Coordinate tool calls via handler registry
    - Handle generation limits
    - Support plugin hooks
    
    Refactored to use:
    - Handler registry pattern for cleaner tool dispatch
    - State dataclass for explicit state management
    - Pre/post hooks for extensibility
    """
    
    def __init__(
        self,
        openrouter_client: OpenRouterClient,
        sam3_client: SAM3Client,
        output_dir: Path,
        model: str,
        message_preparer: MessagePreparer,
    ):
        """
        Initialize inference loop.
        
        Args:
            openrouter_client: OpenRouter client
            sam3_client: SAM3 client
            output_dir: Output directory
            model: Model to use
            message_preparer: Message preparer
        """
        self.openrouter_client = openrouter_client
        self.sam3_client = sam3_client
        self.output_dir = output_dir
        self.model = model
        self.message_preparer = message_preparer
        self.tool_call_parser = ToolCallParser()
        
        # Initialize tool handlers
        self._segment_handler = SegmentPhraseHandler(
            sam3_client, output_dir, message_preparer
        )
        self._examine_handler = ExamineEachMaskHandler(
            openrouter_client, model, message_preparer
        )
        self._select_handler = SelectMasksHandler()
        self._report_handler = ReportNoMaskHandler()
        
        # Setup handler registry
        self._handler_registry = ToolHandlerRegistry()
        self._register_handlers()
        
        # Hooks for extensibility
        self._pre_generation_hooks: List[Callable] = []
        self._post_generation_hooks: List[Callable] = []
        self._pre_tool_hooks: List[Callable] = []
        self._post_tool_hooks: List[Callable] = []
        
        # Cache for system prompts
        self._system_prompts: Optional[Dict[str, str]] = None
    
    def _register_handlers(self):
        """Register all tool handlers."""
        self._handler_registry.register("segment_phrase", self._handle_segment_phrase)
        self._handler_registry.register("examine_each_mask", self._handle_examine_mask)
        self._handler_registry.register("select_masks_and_return", self._handle_select_masks)
        self._handler_registry.register("report_no_mask", self._handle_report_no_mask)
    
    # === Hook Registration ===
    
    def add_pre_generation_hook(self, hook: Callable):
        """Add hook called before each generation."""
        self._pre_generation_hooks.append(hook)
    
    def add_post_generation_hook(self, hook: Callable):
        """Add hook called after each generation."""
        self._post_generation_hooks.append(hook)
    
    def add_pre_tool_hook(self, hook: Callable):
        """Add hook called before tool execution."""
        self._pre_tool_hooks.append(hook)
    
    def add_post_tool_hook(self, hook: Callable):
        """Add hook called after tool execution."""
        self._post_tool_hooks.append(hook)
    
    async def _run_hooks(self, hooks: List[Callable], *args, **kwargs):
        """Run all hooks in a list."""
        for hook in hooks:
            try:
                if callable(hook):
                    result = hook(*args, **kwargs)
                    if hasattr(result, '__await__'):
                        await result
            except Exception as e:
                logger.warning(f"Hook error: {e}")
    
    # === Main Run Method ===
    
    async def run(
        self,
        image_path: str,
        initial_text_prompt: str,
        task_id: str,
        max_generations: int = 100,
    ) -> Dict[str, Any]:
        """
        Run agent inference loop.
        
        Args:
            image_path: Path to input image
            initial_text_prompt: Initial user prompt
            task_id: Task ID
            max_generations: Maximum number of generations
            
        Returns:
            Final result dictionary
            
        Raises:
            ValueError: If max generations exceeded or invalid tool call
        """
        # Load system prompts (cached)
        system_prompts = self._get_system_prompts()
        
        # Initialize state
        state = InferenceState(
            image_path=image_path,
            initial_text_prompt=initial_text_prompt,
            task_id=task_id,
            messages=self._create_initial_messages(
                system_prompts["system"],
                image_path,
                initial_text_prompt
            ),
            max_generations=max_generations,
        )
        
        # Store iterative prompt for handlers
        iterative_prompt = system_prompts["iterative"]
        
        # Main agent loop
        while not state.has_exceeded_limit:
            try:
                result = await self._run_single_generation(state, iterative_prompt)
                
                # Check if we have a final result
                if result and "pred_masks" in result:
                    return result
                    
            except Exception as e:
                logger.error(f"Generation error: {e}", exc_info=True)
                raise
        
        raise ValueError(f"Exceeded maximum generations ({max_generations})")
    
    async def _run_single_generation(
        self,
        state: InferenceState,
        iterative_prompt: str,
    ) -> Optional[Dict[str, Any]]:
        """Run a single generation cycle."""
        # Pre-generation hooks
        await self._run_hooks(
            self._pre_generation_hooks,
            state=state,
            generation=state.generation_count
        )
        
        # Get response from OpenRouter
        response = await self.openrouter_client.chat_completion(
            model=self.model,
            messages=self.message_preparer.prepare_messages_for_openrouter(state.messages),
            max_tokens=4096,
            temperature=0.7,
        )
        
        generated_text = response.get("response", "")
        if not generated_text:
            raise ValueError("Empty response from OpenRouter")
        
        logger.debug(f"Generation {state.generation_count + 1}: {generated_text[:200]}...")
        
        # Parse tool call
        tool_call = self.tool_call_parser.parse_tool_call(generated_text)
        
        # Pre-tool hooks
        await self._run_hooks(
            self._pre_tool_hooks,
            tool_call=tool_call,
            state=state
        )
        
        # Handle tool call
        result = await self._dispatch_tool_call(tool_call, state, iterative_prompt)
        
        # Post-tool hooks
        await self._run_hooks(
            self._post_tool_hooks,
            tool_call=tool_call,
            result=result,
            state=state
        )
        
        # Update state from result
        self._update_state_from_result(state, result)
        
        # Post-generation hooks
        await self._run_hooks(
            self._post_generation_hooks,
            state=state,
            result=result,
            generation=state.generation_count
        )
        
        state.increment_generation()
        return result
    
    async def _dispatch_tool_call(
        self,
        tool_call: Dict[str, Any],
        state: InferenceState,
        iterative_prompt: str,
    ) -> Dict[str, Any]:
        """Dispatch tool call to appropriate handler."""
        tool_name = tool_call.get("name", "")
        
        handler = self._handler_registry.get(tool_name)
        if handler is None:
            available = self._handler_registry.available_tools
            raise ValueError(
                f"Unknown tool: '{tool_name}'. "
                f"Available tools: {available}"
            )
        
        return await handler(tool_call, state, iterative_prompt)
    
    def _update_state_from_result(self, state: InferenceState, result: Dict[str, Any]):
        """Update state with results from tool execution."""
        if "text_prompt" in result:
            state.latest_sam3_text_prompt = result["text_prompt"]
        if "output_json_path" in result:
            state.path_to_latest_output_json = result["output_json_path"]
    
    # === Tool Handler Methods ===
    
    async def _handle_segment_phrase(
        self,
        tool_call: Dict[str, Any],
        state: InferenceState,
        iterative_prompt: str,
    ) -> Dict[str, Any]:
        """Handle segment_phrase tool call."""
        return await self._segment_handler.handle(
            tool_call,
            state.image_path,
            state.initial_text_prompt,
            state.messages,
            state.used_text_prompts,
            state.latest_sam3_text_prompt,
            state.path_to_latest_output_json,
        )
    
    async def _handle_examine_mask(
        self,
        tool_call: Dict[str, Any],
        state: InferenceState,
        iterative_prompt: str,
    ) -> Dict[str, Any]:
        """Handle examine_each_mask tool call."""
        return await self._examine_handler.handle(
            tool_call,
            state.image_path,
            state.initial_text_prompt,
            state.messages,
            state.path_to_latest_output_json,
            state.latest_sam3_text_prompt,
            iterative_prompt,
        )
    
    async def _handle_select_masks(
        self,
        tool_call: Dict[str, Any],
        state: InferenceState,
        iterative_prompt: str,
    ) -> Dict[str, Any]:
        """Handle select_masks_and_return tool call."""
        return self._select_handler.handle(
            tool_call,
            state.path_to_latest_output_json,
            state.image_path,
        )
    
    async def _handle_report_no_mask(
        self,
        tool_call: Dict[str, Any],
        state: InferenceState,
        iterative_prompt: str,
    ) -> Dict[str, Any]:
        """Handle report_no_mask tool call."""
        return self._report_handler.handle(state.image_path)
    
    # === Helper Methods ===
    
    def _get_system_prompts(self) -> Dict[str, str]:
        """Get system prompts (cached)."""
        if self._system_prompts is None:
            self._system_prompts = self._load_system_prompts()
        return self._system_prompts
    
    def _load_system_prompts(self) -> Dict[str, str]:
        """Load system prompts from files."""
        current_dir = Path(__file__).parent.parent
        system_prompt_path = current_dir / "system_prompts" / "system_prompt.txt"
        iterative_prompt_path = current_dir / "system_prompts" / "system_prompt_iterative_checking.txt"
        
        try:
            with open(system_prompt_path, "r", encoding="utf-8") as f:
                system_prompt = f.read().strip()
            with open(iterative_prompt_path, "r", encoding="utf-8") as f:
                iterative_prompt = f.read().strip()
        except FileNotFoundError as e:
            logger.error(f"System prompt file not found: {e}")
            raise
        
        return {
            "system": system_prompt,
            "iterative": iterative_prompt,
        }
    
    def _create_initial_messages(
        self,
        system_prompt: str,
        image_path: str,
        initial_text_prompt: str,
    ) -> List[Dict]:
        """Create initial conversation messages."""
        return [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {
                        "type": "text",
                        "text": (
                            f"The above image is the raw input image. "
                            f"The initial user input query is: '{initial_text_prompt}'."
                        ),
                    },
                ],
            },
        ]
