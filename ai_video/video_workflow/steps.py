from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
TIMEOUT_SECONDS = 60

from typing import List
from .core.models import OnyxVideoStep
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Onyx Video Workflow - Workflow Steps

Definition of workflow steps for video generation process.
"""



def get_default_workflow_steps() -> List[OnyxVideoStep]:
    """Get the default workflow steps for video generation."""
    return [
        OnyxVideoStep(
            name="content_analysis",
            description="Analyze input content and extract key themes",
            llm_prompt="Analyze the following content and extract key themes, emotions, and visual elements: {input_text}",
            timeout=30
        ),
        OnyxVideoStep(
            name="script_generation",
            description="Generate video script based on content analysis",
            llm_prompt="Based on the content analysis, generate a compelling video script with scenes, dialogue, and visual descriptions: {content_analysis}",
            dependencies=["content_analysis"],
            timeout=60
        ),
        OnyxVideoStep(
            name="storyboard_creation",
            description="Create storyboard from script",
            llm_prompt="Convert the video script into a detailed storyboard with scene descriptions, camera angles, and visual elements: {script}",
            dependencies=["script_generation"],
            timeout=90
        ),
        OnyxVideoStep(
            name="visual_style_definition",
            description="Define visual style and aesthetic",
            llm_prompt="Based on the content and storyboard, define the visual style, color palette, and aesthetic direction: {storyboard}",
            dependencies=["storyboard_creation"],
            timeout=45
        ),
        OnyxVideoStep(
            name="video_generation",
            description="Generate final video",
            llm_prompt="Generate the final video based on the storyboard and visual style: {storyboard} {visual_style}",
            dependencies=["storyboard_creation", "visual_style_definition"],
            timeout=300
        )
    ]


def get_vision_workflow_steps() -> List[OnyxVideoStep]:
    """Get workflow steps for vision-based video generation."""
    return [
        OnyxVideoStep(
            name="image_analysis",
            description="Analyze input image and extract visual elements",
            llm_prompt="Analyze this image and extract key visual elements, style, and composition: {input_text}",
            timeout=45
        ),
        OnyxVideoStep(
            name="vision_script_generation",
            description="Generate script based on image analysis",
            llm_prompt="Based on the image analysis, create a video script that incorporates the visual elements: {image_analysis}",
            dependencies=["image_analysis"],
            timeout=60
        ),
        OnyxVideoStep(
            name="vision_video_generation",
            description="Generate video with vision capabilities",
            llm_prompt="Generate a video that matches the analyzed image style and incorporates the script: {vision_script}",
            dependencies=["vision_script_generation"],
            timeout=300
        )
    ]


def get_quick_workflow_steps() -> List[OnyxVideoStep]:
    """Get simplified workflow steps for quick video generation."""
    return [
        OnyxVideoStep(
            name="quick_analysis",
            description="Quick content analysis",
            llm_prompt="Quickly analyze this content: {input_text}",
            timeout=15
        ),
        OnyxVideoStep(
            name="quick_generation",
            description="Quick video generation",
            llm_prompt="Generate a quick video based on: {quick_analysis}",
            dependencies=["quick_analysis"],
            timeout=120
        )
    ]


def get_advanced_workflow_steps() -> List[OnyxVideoStep]:
    """Get advanced workflow steps with additional processing."""
    return [
        OnyxVideoStep(
            name="content_analysis",
            description="Analyze input content and extract key themes",
            llm_prompt="Analyze the following content and extract key themes, emotions, and visual elements: {input_text}",
            timeout=30
        ),
        OnyxVideoStep(
            name="audience_analysis",
            description="Analyze target audience and preferences",
            llm_prompt="Based on the content, analyze the target audience and their preferences: {content_analysis}",
            dependencies=["content_analysis"],
            timeout=30
        ),
        OnyxVideoStep(
            name="script_generation",
            description="Generate video script based on content and audience analysis",
            llm_prompt="Generate a compelling video script considering the content analysis and audience preferences: {content_analysis} {audience_analysis}",
            dependencies=["content_analysis", "audience_analysis"],
            timeout=60
        ),
        OnyxVideoStep(
            name="storyboard_creation",
            description="Create detailed storyboard from script",
            llm_prompt="Convert the video script into a detailed storyboard with scene descriptions, camera angles, and visual elements: {script}",
            dependencies=["script_generation"],
            timeout=90
        ),
        OnyxVideoStep(
            name="visual_style_definition",
            description="Define visual style and aesthetic",
            llm_prompt="Based on the content and storyboard, define the visual style, color palette, and aesthetic direction: {storyboard}",
            dependencies=["storyboard_creation"],
            timeout=45
        ),
        OnyxVideoStep(
            name="audio_script_generation",
            description="Generate audio script and music direction",
            llm_prompt="Create an audio script and music direction for the video: {script} {visual_style}",
            dependencies=["script_generation", "visual_style_definition"],
            timeout=45
        ),
        OnyxVideoStep(
            name="video_generation",
            description="Generate final video with audio",
            llm_prompt="Generate the final video with audio based on all previous steps: {storyboard} {visual_style} {audio_script}",
            dependencies=["storyboard_creation", "visual_style_definition", "audio_script_generation"],
            timeout=300
        ),
        OnyxVideoStep(
            name="quality_assurance",
            description="Perform quality assurance checks",
            llm_prompt="Review the generated video for quality and consistency: {video_generation}",
            dependencies=["video_generation"],
            required=False,
            timeout=30
        )
    ]


def get_workflow_steps_by_type(workflow_type: str) -> List[OnyxVideoStep]:
    """Get workflow steps based on the specified type."""
    workflow_types = {
        "default": get_default_workflow_steps,
        "vision": get_vision_workflow_steps,
        "quick": get_quick_workflow_steps,
        "advanced": get_advanced_workflow_steps
    }
    
    if workflow_type not in workflow_types:
        raise ValueError(f"Unknown workflow type: {workflow_type}")
    
    return workflow_types[workflow_type]() 