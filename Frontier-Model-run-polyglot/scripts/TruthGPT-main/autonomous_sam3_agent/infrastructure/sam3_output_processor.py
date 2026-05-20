"""
SAM3 Output Processor
====================

Helper utilities for processing SAM3 inference outputs.
Centralizes logic for sorting, filtering, and formatting outputs.

Single Responsibility: Process and format SAM3 inference outputs.
"""

import logging
from typing import Dict, Any, List, Tuple
from pathlib import Path
import os

logger = logging.getLogger(__name__)


def sort_outputs_by_scores(outputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sort outputs by prediction scores (descending).
    
    Args:
        outputs: Dictionary with pred_scores, pred_boxes, pred_masks
        
    Returns:
        Dictionary with sorted outputs
    """
    if not outputs.get("pred_scores"):
        return outputs
    
    score_indices = sorted(
        range(len(outputs["pred_scores"])),
        key=lambda i: outputs["pred_scores"][i],
        reverse=True,
    )
    
    outputs["pred_scores"] = [outputs["pred_scores"][i] for i in score_indices]
    outputs["pred_boxes"] = [outputs["pred_boxes"][i] for i in score_indices]
    outputs["pred_masks"] = [outputs["pred_masks"][i] for i in score_indices]
    
    return outputs


def filter_valid_masks(
    outputs: Dict[str, Any],
    min_rle_length: int = 4
) -> Dict[str, Any]:
    """
    Filter out invalid masks based on RLE length.
    
    Args:
        outputs: Dictionary with pred_masks, pred_boxes, pred_scores
        min_rle_length: Minimum RLE length to consider valid
        
    Returns:
        Dictionary with filtered outputs
    """
    valid_masks = []
    valid_boxes = []
    valid_scores = []
    
    for i, rle in enumerate(outputs.get("pred_masks", [])):
        if len(rle) > min_rle_length:
            valid_masks.append(rle)
            valid_boxes.append(outputs["pred_boxes"][i])
            valid_scores.append(outputs["pred_scores"][i])
    
    outputs["pred_masks"] = valid_masks
    outputs["pred_boxes"] = valid_boxes
    outputs["pred_scores"] = valid_scores
    
    return outputs


def process_sam3_outputs(
    outputs: Dict[str, Any],
    sort_by_scores: bool = True,
    filter_invalid: bool = True,
    min_rle_length: int = 4
) -> Dict[str, Any]:
    """
    Process SAM3 outputs: sort by scores and filter invalid masks.
    
    Args:
        outputs: Raw SAM3 outputs dictionary
        sort_by_scores: Whether to sort by scores (descending)
        filter_invalid: Whether to filter invalid masks
        min_rle_length: Minimum RLE length for valid masks
        
    Returns:
        Processed outputs dictionary
    """
    if sort_by_scores:
        outputs = sort_outputs_by_scores(outputs)
    
    if filter_invalid:
        outputs = filter_valid_masks(outputs, min_rle_length)
    
    return outputs


def build_output_paths(
    image_path: str,
    text_prompt: str,
    output_folder_path: str,
    file_extension: str = ".json"
) -> Tuple[str, str]:
    """
    Build output paths for JSON and image files.
    
    Args:
        image_path: Path to input image
        text_prompt: Text prompt used for segmentation
        output_folder_path: Base output folder
        file_extension: Extension for JSON file (default: .json)
        
    Returns:
        Tuple of (output_json_path, output_image_path)
    """
    # Sanitize text prompt for filename
    text_prompt_for_save_path = text_prompt.replace("/", "_") if "/" in text_prompt else text_prompt
    
    # Ensure output directory exists
    os.makedirs(output_folder_path, exist_ok=True)
    
    # Build paths
    image_stem = Path(image_path).stem
    output_json_path = os.path.join(
        output_folder_path,
        f"{image_stem}_{text_prompt_for_save_path}{file_extension}"
    )
    output_image_path = os.path.join(
        output_folder_path,
        f"{image_stem}_{text_prompt_for_save_path}.png"
    )
    
    return output_json_path, output_image_path


def filter_outputs_by_indices(
    outputs: Dict[str, Any],
    indices: List[int],
    one_indexed: bool = False
) -> Dict[str, Any]:
    """
    Filter outputs to keep only specified indices.
    
    Args:
        outputs: Dictionary with pred_masks, pred_boxes, pred_scores
        indices: List of indices to keep
        one_indexed: Whether indices are 1-indexed (default: False, 0-indexed)
        
    Returns:
        Filtered outputs dictionary
    """
    if one_indexed:
        # Convert 1-indexed to 0-indexed
        indices = [i - 1 for i in indices]
    
    # Filter to valid indices
    max_index = len(outputs.get("pred_masks", [])) - 1
    valid_indices = [i for i in indices if 0 <= i <= max_index]
    
    if not valid_indices:
        return {
            "original_image_path": outputs.get("original_image_path"),
            "orig_img_h": outputs.get("orig_img_h"),
            "orig_img_w": outputs.get("orig_img_w"),
            "pred_boxes": [],
            "pred_scores": [],
            "pred_masks": [],
        }
    
    filtered_outputs = {
        "original_image_path": outputs.get("original_image_path"),
        "orig_img_h": outputs.get("orig_img_h"),
        "orig_img_w": outputs.get("orig_img_w"),
        "pred_boxes": [outputs["pred_boxes"][i] for i in valid_indices],
        "pred_scores": [outputs["pred_scores"][i] for i in valid_indices],
        "pred_masks": [outputs["pred_masks"][i] for i in valid_indices],
    }
    
    return filtered_outputs

