"""
Workflow Node Manager
=====================

Manages workflow node operations including finding, updating, and validating nodes.
"""

import logging
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)


class WorkflowNodeManager:
    """Manages workflow node operations."""
    
    def __init__(self, node_ids: Dict[str, int]):
        """
        Initialize Workflow Node Manager.
        
        Args:
            node_ids: Dictionary mapping node names to IDs
        """
        self.node_ids = node_ids
    
    def find_node(self, workflow: Dict[str, Any], node_id: int) -> Optional[Dict[str, Any]]:
        """
        Find a node by ID in the workflow.
        
        Args:
            workflow: Workflow dictionary
            node_id: Node ID to find
            
        Returns:
            Node dictionary if found, None otherwise
        """
        nodes = workflow.get("nodes", [])
        for node in nodes:
            if node.get("id") == node_id:
                return node
        return None
    
    def update_node_widget(
        self,
        node: Dict[str, Any],
        widget_index: int,
        value: Any
    ) -> bool:
        """
        Update a widget value in a node.
        
        Args:
            node: Node dictionary to update
            widget_index: Index of widget to update
            value: New value for the widget
            
        Returns:
            True if update was successful, False otherwise
        """
        if widget_index < 0:
            logger.warning(f"Invalid widget index: {widget_index}")
            return False
        
        if "widgets_values" not in node:
            node["widgets_values"] = []
        
        # Extend list if needed
        while len(node["widgets_values"]) <= widget_index:
            node["widgets_values"].append(None)
        
        node["widgets_values"][widget_index] = value
        return True
    
    def update_image_node(self, workflow: Dict[str, Any], image_url: str) -> bool:
        """
        Update image node with image URL.
        
        Args:
            workflow: Workflow dictionary
            image_url: Image URL or path
            
        Returns:
            True if update was successful
        """
        node = self.find_node(workflow, self.node_ids["LOAD_IMAGE"])
        if node:
            return self.update_node_widget(node, 0, image_url)
        return False
    
    def update_prompt_node(self, workflow: Dict[str, Any], prompt: str) -> bool:
        """
        Update prompt node with prompt text.
        
        Args:
            workflow: Workflow dictionary
            prompt: Prompt text
            
        Returns:
            True if update was successful
        """
        node = self.find_node(workflow, self.node_ids["CLIP_TEXT_ENCODE"])
        if node:
            return self.update_node_widget(node, 0, prompt)
        return False
    
    def update_sampler_node(
        self,
        workflow: Dict[str, Any],
        seed: int,
        num_steps: int
    ) -> bool:
        """
        Update sampler node with seed and steps.
        
        Args:
            workflow: Workflow dictionary
            seed: Random seed
            num_steps: Number of inference steps
            
        Returns:
            True if update was successful
        """
        node = self.find_node(workflow, self.node_ids["KSAMPLER"])
        if node:
            # Update seed (typically index 0) and steps (typically index 1)
            self.update_node_widget(node, 0, seed)
            self.update_node_widget(node, 1, num_steps)
            return True
        return False
    
    def update_guidance_node(self, workflow: Dict[str, Any], guidance_scale: float) -> bool:
        """
        Update guidance node with guidance scale.
        
        Args:
            workflow: Workflow dictionary
            guidance_scale: Guidance scale value
            
        Returns:
            True if update was successful
        """
        node = self.find_node(workflow, self.node_ids["FLUX_GUIDANCE"])
        if node:
            return self.update_node_widget(node, 0, guidance_scale)
        return False
    
    def update_face_node(self, workflow: Dict[str, Any], face_url: str) -> bool:
        """
        Update face node with face URL.
        
        Args:
            workflow: Workflow dictionary
            face_url: Face image URL or path
            
        Returns:
            True if update was successful
        """
        node = self.find_node(workflow, self.node_ids["LOAD_NEW_FACE"])
        if node:
            return self.update_node_widget(node, 0, face_url)
        return False

