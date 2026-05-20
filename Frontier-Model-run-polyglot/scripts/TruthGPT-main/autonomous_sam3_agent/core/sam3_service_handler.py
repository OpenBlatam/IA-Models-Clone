"""
SAM3 Service Handler
====================

Encapsulates SAM3 logic and OpenRouter interaction.
"""

import logging
import os
import base64
from typing import Dict, Any, List, Optional, Set
from pathlib import Path

from ..infrastructure.openrouter_client import OpenRouterClient
from ..infrastructure.sam3_client import SAM3Client
from .helpers import (
    create_message,
    create_text_content,
    create_image_content,
    create_user_message_with_image,
    create_tool_message,
    load_json_file,
    save_json_file,
    create_output_structure,
    filter_outputs_by_indices,
    extract_tool_call_from_text,
)

logger = logging.getLogger(__name__)


class SAM3ServiceHandler:
    """
    Handles SAM3 agent inference logic.
    """
    
    def __init__(
        self,
        openrouter_client: OpenRouterClient,
        sam3_client: SAM3Client,
        model: str,
        output_dir: Path,
    ):
        self.openrouter_client = openrouter_client
        self.sam3_client = sam3_client
        self.model = model
        self.output_dir = output_dir
        
        # Load system prompts
        current_dir = Path(__file__).parent.parent
        system_prompt_path = current_dir / "system_prompts" / "system_prompt.txt"
        iterative_prompt_path = current_dir / "system_prompts" / "system_prompt_iterative_checking.txt"
        
        try:
            with open(system_prompt_path, "r") as f:
                self.system_prompt = f.read().strip()
            with open(iterative_prompt_path, "r") as f:
                self.iterative_checking_system_prompt = f.read().strip()
        except FileNotFoundError:
            logger.warning("System prompts not found, using defaults")
            self.system_prompt = "You are an AI agent that uses SAM3 for image segmentation."
            self.iterative_checking_system_prompt = "Verify the masks."
    
    async def run_inference(
        self,
        image_path: str,
        initial_text_prompt: str,
        task_id: str,
    ) -> Dict[str, Any]:
        """
        Run SAM3 agent inference.
        """
        messages = [
            create_message("system", self.system_prompt),
            create_user_message_with_image(
                f"The above image is the raw input image. The initial user input query is: '{initial_text_prompt}'.",
                image_path
            ),
        ]
        
        used_text_prompts = set()
        latest_sam3_text_prompt = ""
        path_to_latest_output_json = ""
        generation_count = 0
        max_generations = 100
        
        while generation_count < max_generations:
            # Send request to OpenRouter
            response = await self.openrouter_client.chat_completion(
                model=self.model,
                messages=self._prepare_messages_for_openrouter(messages),
                max_tokens=4096,
                temperature=0.7,
            )
            
            generated_text = response.get("response", "")
            if not generated_text:
                raise ValueError("Empty response from OpenRouter")
            
            logger.debug(f"Generation {generation_count + 1}: {generated_text[:200]}...")
            
            # Parse tool call
            tool_call = extract_tool_call_from_text(generated_text)
            
            # Handle tool calls
            if tool_call["name"] == "segment_phrase":
                result = await self._handle_segment_phrase(
                    tool_call,
                    image_path,
                    initial_text_prompt,
                    messages,
                    used_text_prompts,
                    path_to_latest_output_json,
                )
                latest_sam3_text_prompt = result["text_prompt"]
                path_to_latest_output_json = result["output_json_path"]
                
            elif tool_call["name"] == "examine_each_mask":
                result = await self._handle_examine_each_mask(
                    tool_call,
                    image_path,
                    initial_text_prompt,
                    messages,
                    path_to_latest_output_json,
                )
                path_to_latest_output_json = result["output_json_path"]
                
            elif tool_call["name"] == "select_masks_and_return":
                result = await self._handle_select_masks_and_return(
                    tool_call,
                    path_to_latest_output_json,
                )
                result["generations"] = generation_count + 1
                return result
                
            elif tool_call["name"] == "report_no_mask":
                result = await self._handle_report_no_mask(image_path)
                result["generations"] = generation_count + 1
                return result
            
            else:
                raise ValueError(f"Unknown tool call: {tool_call['name']}")
            
            generation_count += 1
        
        raise ValueError(f"Exceeded maximum generations ({max_generations})")
    
    def _prepare_messages_for_openrouter(self, messages: List[Dict]) -> List[Dict]:
        """Prepare messages for OpenRouter API format."""
        processed = []
        for msg in messages:
            if msg["role"] == "user" and isinstance(msg.get("content"), list):
                content = []
                for item in msg["content"]:
                    if isinstance(item, dict) and item.get("type") == "image":
                        image_path = item["image"]
                        base64_image, mime_type = self._get_image_base64(image_path)
                        content.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{base64_image}",
                                "detail": "high"
                            }
                        })
                    else:
                        content.append(item)
                processed.append({"role": msg["role"], "content": content})
            else:
                processed.append(msg)
        return processed
    
    def _get_image_base64(self, image_path: str) -> tuple:
        """Convert image to base64."""
        ext = os.path.splitext(image_path)[1].lower()
        mime_types = {
            ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".png": "image/png", ".gif": "image/gif", ".webp": "image/webp",
        }
        mime_type = mime_types.get(ext, "image/jpeg")
        
        with open(image_path, "rb") as f:
            base64_data = base64.b64encode(f.read()).decode("utf-8")
        
        return base64_data, mime_type
    
    async def _handle_segment_phrase(
        self,
        tool_call: Dict,
        image_path: str,
        initial_text_prompt: str,
        messages: List[Dict],
        used_text_prompts: Set[str],
        path_to_latest_output_json: str,
    ) -> Dict[str, Any]:
        """Handle segment_phrase tool call."""
        text_prompt = tool_call["parameters"]["text_prompt"]
        
        if text_prompt in used_text_prompts:
            messages.append(create_tool_message(tool_call))
            messages.append(create_message(
                "user",
                [create_text_content(f"You have previously used '{text_prompt}'. Please use a different prompt.")]
            ))
            return {"text_prompt": text_prompt, "output_json_path": path_to_latest_output_json}
        
        used_text_prompts.add(text_prompt)
        
        output_json_path = await self.sam3_client.call_sam_service(
            image_path=image_path,
            text_prompt=text_prompt,
            output_folder_path=str(self.output_dir / "sam_out"),
        )
        
        sam3_outputs = load_json_file(output_json_path)
        num_masks = len(sam3_outputs.get("pred_boxes", []))
        
        messages.append(create_tool_message(tool_call))
        
        if num_masks == 0:
            messages.append(create_message(
                "user",
                [create_text_content(f"No masks generated. Please try a different prompt. Original query: '{initial_text_prompt}'.")]
            ))
        else:
            messages.append(create_message(
                "user",
                [
                    create_text_content(f"Generated {num_masks} masks. Analyze them carefully. Original query: '{initial_text_prompt}'."),
                    create_image_content(sam3_outputs["output_image_path"])
                ]
            ))
        
        return {"text_prompt": text_prompt, "output_json_path": output_json_path}
    
    async def _handle_examine_each_mask(
        self,
        tool_call: Dict,
        image_path: str,
        initial_text_prompt: str,
        messages: List[Dict],
        path_to_latest_output_json: str,
    ) -> Dict[str, Any]:
        """Handle examine_each_mask tool call."""
        current_outputs = load_json_file(path_to_latest_output_json)
        num_masks = len(current_outputs.get("pred_masks", []))
        masks_to_keep = []
        
        for i in range(num_masks):
            checking_messages = [
                {"role": "system", "content": self.iterative_checking_system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Raw input image:"},
                        {"type": "image", "image": image_path},
                        {"type": "text", "text": f"Initial query: '{initial_text_prompt}'"},
                        {"type": "text", "text": f"Analyze mask {i+1} and provide <verdict>Accept</verdict> or <verdict>Reject</verdict>"},
                    ],
                },
            ]
            
            response = await self.openrouter_client.chat_completion(
                model=self.model,
                messages=self._prepare_messages_for_openrouter(checking_messages),
                max_tokens=500,
            )
            
            verdict_text = response.get("response", "")
            if "Accept" in verdict_text and "Reject" not in verdict_text:
                masks_to_keep.append(i)
        
        updated_outputs = filter_outputs_by_indices(current_outputs, masks_to_keep, offset=0)
        updated_path = path_to_latest_output_json.replace(".json", f"_masks_{'_'.join(map(str, masks_to_keep))}.json")
        save_json_file(updated_outputs, updated_path)
        
        messages.append(create_tool_message(tool_call))
        
        if len(masks_to_keep) == 0:
            messages.append(create_message(
                "user",
                [create_text_content(f"All masks rejected. Try a different prompt. Original query: '{initial_text_prompt}'.")]
            ))
        else:
            messages.append(create_message(
                "user",
                [create_text_content(f"After examination, {len(masks_to_keep)} masks remain. Original query: '{initial_text_prompt}'.")]
            ))
        
        return {"output_json_path": updated_path}
    
    async def _handle_select_masks_and_return(
        self,
        tool_call: Dict,
        path_to_latest_output_json: str,
    ) -> Dict[str, Any]:
        """Handle select_masks_and_return tool call."""
        current_outputs = load_json_file(path_to_latest_output_json)
        masks_to_keep = tool_call["parameters"]["final_answer_masks"]
        available_masks = set(range(1, len(current_outputs["pred_masks"]) + 1))
        masks_to_keep = sorted({i for i in masks_to_keep if i in available_masks})
        
        return filter_outputs_by_indices(current_outputs, masks_to_keep, offset=-1)
    
    async def _handle_report_no_mask(self, image_path: str) -> Dict[str, Any]:
        """Handle report_no_mask tool call."""
        import cv2
        height, width = cv2.imread(image_path).shape[:2]
        return create_output_structure(
            original_image_path=image_path,
            orig_img_h=height,
            orig_img_w=width,
        )
