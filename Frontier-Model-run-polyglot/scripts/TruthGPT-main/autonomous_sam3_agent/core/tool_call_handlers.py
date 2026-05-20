"""
Tool Call Handlers
==================

Specialized handlers for SAM3 agent tool calls.

Refactored with:
- Abstract base class pattern
- Handler registry integration
- State-based interface
- Cleaner separation of concerns
"""

import json
import logging
from typing import Dict, Any, List, Set, Optional, TYPE_CHECKING
from pathlib import Path
from abc import ABC, abstractmethod
from dataclasses import dataclass

if TYPE_CHECKING:
    from .inference_loop import InferenceState

logger = logging.getLogger(__name__)


@dataclass
class HandlerContext:
    """
    Context passed to tool handlers.
    
    Encapsulates all information a handler needs.
    """
    tool_call: Dict[str, Any]
    image_path: str
    initial_text_prompt: str
    messages: List[Dict]
    used_text_prompts: Set[str]
    latest_sam3_text_prompt: str
    path_to_latest_output_json: str
    iterative_checking_prompt: str = ""
    
    @property
    def tool_name(self) -> str:
        """Get tool name from tool call."""
        return self.tool_call.get("name", "")
    
    @property
    def parameters(self) -> Dict[str, Any]:
        """Get parameters from tool call."""
        return self.tool_call.get("parameters", {})


class BaseToolHandler(ABC):
    """
    Abstract base class for tool handlers.
    
    All tool handlers should inherit from this class.
    """
    
    @property
    @abstractmethod
    def tool_name(self) -> str:
        """Tool name this handler handles."""
        pass
    
    @abstractmethod
    async def handle(self, context: HandlerContext) -> Dict[str, Any]:
        """
        Handle a tool call.
        
        Args:
            context: Handler context with all necessary data
            
        Returns:
            Result dictionary
        """
        pass
    
    @staticmethod
    def format_tool_call(tool_call: Dict) -> str:
        """Format tool call as string for messages."""
        return f"<tool>{json.dumps(tool_call)}</tool>"
    
    @staticmethod
    def create_assistant_message(tool_call: Dict) -> Dict:
        """Create assistant message with tool call."""
        return {
            "role": "assistant",
            "content": [{"type": "text", "text": BaseToolHandler.format_tool_call(tool_call)}],
        }
    
    @staticmethod
    def create_user_message(text: str, image_path: Optional[str] = None) -> Dict:
        """Create user message, optionally with image."""
        content = [{"type": "text", "text": text}]
        if image_path:
            content.append({"type": "image", "image": image_path})
        return {"role": "user", "content": content}
    
    @staticmethod
    def load_json(path: str) -> Dict[str, Any]:
        """Load JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    
    @staticmethod
    def save_json(data: Dict[str, Any], path: str):
        """Save data to JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)


class SegmentPhraseHandler(BaseToolHandler):
    """
    Handler for segment_phrase tool call.
    
    Responsibility: Execute SAM3 segmentation with given text prompt.
    """
    
    tool_name = "segment_phrase"
    
    def __init__(self, sam3_client, output_dir: Path, message_preparer):
        """
        Initialize segment phrase handler.
        
        Args:
            sam3_client: SAM3 client instance
            output_dir: Output directory for results
            message_preparer: Message preparer instance
        """
        self.sam3_client = sam3_client
        self.output_dir = output_dir
        self.message_preparer = message_preparer
    
    async def handle(self, context: HandlerContext) -> Dict[str, Any]:
        """Handle segment_phrase tool call."""
        text_prompt = context.parameters.get("text_prompt", "")
        
        # Check if prompt was already used
        if text_prompt in context.used_text_prompts:
            return self._handle_duplicate_prompt(context, text_prompt)
        
        context.used_text_prompts.add(text_prompt)
        
        # Call SAM3 service
        output_json_path = await self.sam3_client.call_sam_service(
            image_path=context.image_path,
            text_prompt=text_prompt,
            output_folder_path=str(self.output_dir / "sam_out"),
        )
        
        # Load and process results
        sam3_outputs = self.load_json(output_json_path)
        num_masks = len(sam3_outputs.get("pred_boxes", []))
        
        # Update conversation
        self._add_messages(context, text_prompt, num_masks, sam3_outputs)
        
        return {
            "text_prompt": text_prompt,
            "output_json_path": output_json_path,
        }
    
    def _handle_duplicate_prompt(
        self, 
        context: HandlerContext, 
        text_prompt: str
    ) -> Dict[str, Any]:
        """Handle case when prompt was already used."""
        context.messages.append(self.create_assistant_message(context.tool_call))
        context.messages.append(self.create_user_message(
            f"You have previously used '{text_prompt}'. Please use a different prompt."
        ))
        return {
            "text_prompt": text_prompt,
            "output_json_path": context.path_to_latest_output_json,
        }
    
    def _add_messages(
        self,
        context: HandlerContext,
        text_prompt: str,
        num_masks: int,
        sam3_outputs: Dict[str, Any],
    ):
        """Add messages to conversation."""
        context.messages.append(self.create_assistant_message(context.tool_call))
        
        if num_masks == 0:
            context.messages.append(self.create_user_message(
                f"No masks generated. Please try a different prompt. "
                f"Original query: '{context.initial_text_prompt}'."
            ))
        else:
            context.messages.append(self.create_user_message(
                f"Generated {num_masks} masks. Analyze them carefully. "
                f"Original query: '{context.initial_text_prompt}'.",
                image_path=sam3_outputs.get("output_image_path")
            ))


class ExamineEachMaskHandler(BaseToolHandler):
    """
    Handler for examine_each_mask tool call.
    
    Responsibility: Examine each mask and filter based on LLM verdict.
    """
    
    tool_name = "examine_each_mask"
    
    def __init__(self, openrouter_client, model: str, message_preparer):
        """
        Initialize examine each mask handler.
        
        Args:
            openrouter_client: OpenRouter client instance
            model: Model to use for checking
            message_preparer: Message preparer instance
        """
        self.openrouter_client = openrouter_client
        self.model = model
        self.message_preparer = message_preparer
    
    async def handle(self, context: HandlerContext) -> Dict[str, Any]:
        """Handle examine_each_mask tool call."""
        current_outputs = self.load_json(context.path_to_latest_output_json)
        
        num_masks = len(current_outputs.get("pred_masks", []))
        masks_to_keep = await self._evaluate_masks(context, num_masks)
        
        # Filter outputs
        updated_outputs = self._filter_outputs(current_outputs, masks_to_keep)
        
        # Save updated outputs
        updated_path = self._generate_output_path(
            context.path_to_latest_output_json,
            masks_to_keep
        )
        self.save_json(updated_outputs, updated_path)
        
        # Update conversation
        self._add_messages(context, len(masks_to_keep))
        
        return {"output_json_path": updated_path}
    
    async def _evaluate_masks(
        self, 
        context: HandlerContext, 
        num_masks: int
    ) -> List[int]:
        """Evaluate each mask and return indices to keep."""
        masks_to_keep = []
        
        for i in range(num_masks):
            verdict = await self._get_mask_verdict(context, i)
            if verdict == "Accept":
                masks_to_keep.append(i)
        
        return masks_to_keep
    
    async def _get_mask_verdict(
        self, 
        context: HandlerContext, 
        mask_index: int
    ) -> str:
        """Get LLM verdict for a specific mask."""
        checking_messages = [
            {"role": "system", "content": context.iterative_checking_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Raw input image:"},
                    {"type": "image", "image": context.image_path},
                    {"type": "text", "text": f"Initial query: '{context.initial_text_prompt}'"},
                    {"type": "text", "text": f"Analyze mask {mask_index + 1} and provide <verdict>Accept</verdict> or <verdict>Reject</verdict>"},
                ],
            },
        ]
        
        response = await self.openrouter_client.chat_completion(
            model=self.model,
            messages=self.message_preparer.prepare_messages_for_openrouter(checking_messages),
            max_tokens=500,
        )
        
        verdict_text = response.get("response", "")
        if "Accept" in verdict_text and "Reject" not in verdict_text:
            return "Accept"
        return "Reject"
    
    @staticmethod
    def _filter_outputs(
        outputs: Dict[str, Any], 
        indices: List[int]
    ) -> Dict[str, Any]:
        """Filter outputs to keep only specified indices."""
        return {
            "original_image_path": outputs["original_image_path"],
            "orig_img_h": outputs["orig_img_h"],
            "orig_img_w": outputs["orig_img_w"],
            "pred_boxes": [outputs["pred_boxes"][i] for i in indices],
            "pred_scores": [outputs["pred_scores"][i] for i in indices],
            "pred_masks": [outputs["pred_masks"][i] for i in indices],
        }
    
    @staticmethod
    def _generate_output_path(base_path: str, mask_indices: List[int]) -> str:
        """Generate output path with mask indices."""
        suffix = "_".join(map(str, mask_indices)) if mask_indices else "none"
        return base_path.replace(".json", f"_masks_{suffix}.json")
    
    def _add_messages(self, context: HandlerContext, kept_count: int):
        """Add messages to conversation."""
        context.messages.append(self.create_assistant_message(context.tool_call))
        
        if kept_count == 0:
            context.messages.append(self.create_user_message(
                f"All masks rejected. Try a different prompt. "
                f"Original query: '{context.initial_text_prompt}'."
            ))
        else:
            context.messages.append(self.create_user_message(
                f"After examination, {kept_count} masks remain. "
                f"Original query: '{context.initial_text_prompt}'."
            ))


class SelectMasksHandler(BaseToolHandler):
    """
    Handler for select_masks_and_return tool call.
    
    Responsibility: Select final masks and return result.
    """
    
    tool_name = "select_masks_and_return"
    
    async def handle(self, context: HandlerContext) -> Dict[str, Any]:
        """Handle select_masks_and_return tool call."""
        current_outputs = self.load_json(context.path_to_latest_output_json)
        
        # Get masks to keep (1-indexed from LLM)
        masks_to_keep = context.parameters.get("final_answer_masks", [])
        
        # Validate indices
        available_masks = set(range(1, len(current_outputs["pred_masks"]) + 1))
        valid_masks = sorted({i for i in masks_to_keep if i in available_masks})
        
        # Convert to 0-indexed and filter
        return {
            "original_image_path": current_outputs["original_image_path"],
            "orig_img_h": current_outputs["orig_img_h"],
            "orig_img_w": current_outputs["orig_img_w"],
            "pred_boxes": [current_outputs["pred_boxes"][i - 1] for i in valid_masks],
            "pred_scores": [current_outputs["pred_scores"][i - 1] for i in valid_masks],
            "pred_masks": [current_outputs["pred_masks"][i - 1] for i in valid_masks],
        }


class ReportNoMaskHandler(BaseToolHandler):
    """
    Handler for report_no_mask tool call.
    
    Responsibility: Report that no valid masks were found.
    """
    
    tool_name = "report_no_mask"
    
    async def handle(self, context: HandlerContext) -> Dict[str, Any]:
        """Handle report_no_mask tool call."""
        import cv2
        
        image = cv2.imread(context.image_path)
        height, width = image.shape[:2]
        
        return {
            "original_image_path": context.image_path,
            "orig_img_h": height,
            "orig_img_w": width,
            "pred_boxes": [],
            "pred_scores": [],
            "pred_masks": [],
        }


class ToolHandlerFactory:
    """
    Factory for creating tool handlers.
    
    Provides centralized handler creation and registration.
    """
    
    _handler_classes = {
        "segment_phrase": SegmentPhraseHandler,
        "examine_each_mask": ExamineEachMaskHandler,
        "select_masks_and_return": SelectMasksHandler,
        "report_no_mask": ReportNoMaskHandler,
    }
    
    @classmethod
    def create_all(
        cls,
        sam3_client,
        openrouter_client,
        output_dir: Path,
        model: str,
        message_preparer,
    ) -> Dict[str, BaseToolHandler]:
        """
        Create all handlers.
        
        Returns:
            Dictionary mapping tool names to handler instances
        """
        return {
            "segment_phrase": SegmentPhraseHandler(
                sam3_client, output_dir, message_preparer
            ),
            "examine_each_mask": ExamineEachMaskHandler(
                openrouter_client, model, message_preparer
            ),
            "select_masks_and_return": SelectMasksHandler(),
            "report_no_mask": ReportNoMaskHandler(),
        }
    
    @classmethod
    def get_available_tools(cls) -> List[str]:
        """Get list of available tool names."""
        return list(cls._handler_classes.keys())
