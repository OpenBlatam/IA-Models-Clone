"""
Common helper functions for autonomous SAM3 agent.

This module consolidates frequently used operations to eliminate
duplication across the codebase.

Refactored to:
- Consolidate message construction
- Centralize JSON operations
- Provide reusable output builders
"""

import json
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


def create_message(role: str, content: Any) -> Dict[str, Any]:
    """
    Create a message dictionary for OpenRouter API.
    
    Args:
        role: Message role ("system", "user", "assistant")
        content: Message content (string or list of content items)
    
    Returns:
        Message dictionary
    """
    return {"role": role, "content": content}


def create_text_content(text: str) -> Dict[str, str]:
    """
    Create a text content item for messages.
    
    Args:
        text: Text content
    
    Returns:
        Text content dictionary
    """
    return {"type": "text", "text": text}


def create_image_content(image_path: str) -> Dict[str, str]:
    """
    Create an image content item for messages.
    
    Args:
        image_path: Path to image file
    
    Returns:
        Image content dictionary
    """
    return {"type": "image", "image": image_path}


def create_user_message_with_image(
    text: str,
    image_path: str
) -> Dict[str, Any]:
    """
    Create a user message with text and image.
    
    Args:
        text: Text content
        image_path: Path to image file
    
    Returns:
        User message dictionary
    """
    return create_message(
        role="user",
        content=[
            create_text_content(text),
            create_image_content(image_path)
        ]
    )


def create_tool_message(tool_call: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create an assistant message with tool call.
    
    Args:
        tool_call: Tool call dictionary
    
    Returns:
        Assistant message dictionary
    """
    return create_message(
        role="assistant",
        content=[create_text_content(f"<tool>{json.dumps(tool_call)}</tool>")]
    )


def load_json_file(file_path: str) -> Dict[str, Any]:
    """
    Load JSON data from file.
    
    Args:
        file_path: Path to JSON file
    
    Returns:
        Loaded JSON data
    
    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If file contains invalid JSON
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"JSON file not found: {file_path}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in file {file_path}: {e}")
        raise


def save_json_file(data: Dict[str, Any], file_path: str, indent: int = 4) -> None:
    """
    Save data to JSON file.
    
    Args:
        data: Data to save
        file_path: Path to output file
        indent: JSON indentation (default: 4)
    
    Raises:
        OSError: If file cannot be written
    """
    try:
        # Ensure directory exists
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
    except OSError as e:
        logger.error(f"Error saving JSON file {file_path}: {e}")
        raise


def create_output_structure(
    original_image_path: str,
    orig_img_h: int,
    orig_img_w: int,
    pred_boxes: Optional[List] = None,
    pred_scores: Optional[List] = None,
    pred_masks: Optional[List] = None,
) -> Dict[str, Any]:
    """
    Create a standardized output structure for SAM3 results.
    
    Args:
        original_image_path: Path to original image
        orig_img_h: Original image height
        orig_img_w: Original image width
        pred_boxes: Predicted boxes (default: empty list)
        pred_scores: Predicted scores (default: empty list)
        pred_masks: Predicted masks (default: empty list)
    
    Returns:
        Output structure dictionary
    """
    return {
        "original_image_path": original_image_path,
        "orig_img_h": orig_img_h,
        "orig_img_w": orig_img_w,
        "pred_boxes": pred_boxes or [],
        "pred_scores": pred_scores or [],
        "pred_masks": pred_masks or [],
    }


def filter_outputs_by_indices(
    outputs: Dict[str, Any],
    indices: List[int],
    offset: int = 0
) -> Dict[str, Any]:
    """
    Filter outputs by indices (for selecting specific masks).
    
    Args:
        outputs: Original outputs dictionary
        indices: List of indices to keep (1-based if offset=0)
        offset: Index offset (0 for 1-based, -1 for 0-based)
    
    Returns:
        Filtered outputs dictionary
    """
    adjusted_indices = [i + offset for i in indices]
    
    return {
        "original_image_path": outputs["original_image_path"],
        "orig_img_h": outputs["orig_img_h"],
        "orig_img_w": outputs["orig_img_w"],
        "pred_boxes": [outputs["pred_boxes"][i] for i in adjusted_indices if 0 <= i < len(outputs["pred_boxes"])],
        "pred_scores": [outputs["pred_scores"][i] for i in adjusted_indices if 0 <= i < len(outputs["pred_scores"])],
        "pred_masks": [outputs["pred_masks"][i] for i in adjusted_indices if 0 <= i < len(outputs["pred_masks"])],
    }


def extract_tool_call_from_text(generated_text: str) -> Dict[str, Any]:
    """
    Extract tool call JSON from generated text.
    
    Args:
        generated_text: Text containing <tool>...</tool> tags
    
    Returns:
        Parsed tool call dictionary
    
    Raises:
        ValueError: If tool tag not found or JSON is invalid
    """
    if "<tool>" not in generated_text:
        raise ValueError(f"Generated text does not contain <tool> tag: {generated_text}")
    
    tool_call_json_str = (
        generated_text.split("<tool>")[-1]
        .split("</tool>")[0]
        .strip()
        .replace(r"}}}", r"}}")
    )
    
    try:
        return json.loads(tool_call_json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in tool call: {tool_call_json_str}") from e

