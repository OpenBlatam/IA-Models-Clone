"""
Context Analyzer

Analyzes multimodal context and determines required modalities.
"""

from typing import Dict, Any, Optional, List
import logging

from .models import ModalityType

logger = logging.getLogger(__name__)


class ContextAnalyzer:
    """Analyzes multimodal context."""
    
    def analyze(self, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze multimodal context.
        
        Args:
            context: Context dictionary with potential multimodal data
            
        Returns:
            Analysis result with modalities present and content summary
        """
        analysis = {
            "modalities_present": [],
            "content_summary": {}
        }
        
        if not context:
            return analysis
        
        # Check for different modalities
        modality_checks = {
            "text": ModalityType.TEXT,
            "image": ModalityType.IMAGE,
            "audio": ModalityType.AUDIO,
            "video": ModalityType.VIDEO,
        }
        
        for key, modality in modality_checks.items():
            if key in context:
                analysis["modalities_present"].append(modality.value)
                content = context[key]
                analysis["content_summary"][key] = self._summarize_content(key, content)
        
        return analysis
    
    def _summarize_content(self, modality: str, content: Any) -> Dict[str, Any]:
        """Summarize content for a modality."""
        if modality == "text":
            return {
                "length": len(content) if isinstance(content, str) else 0,
                "type": type(content).__name__
            }
        else:
            return {
                "type": type(content).__name__,
                "path": content if isinstance(content, str) else "in_memory"
            }
    
    def determine_required_modalities(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None
    ) -> List[ModalityType]:
        """
        Determine which modalities are required for the task.
        
        Args:
            task: Task description
            context: Optional context
            
        Returns:
            List of required modalities
        """
        required = []
        task_lower = task.lower()
        
        # Text is usually required
        required.append(ModalityType.TEXT)
        
        # Check for image-related tasks
        image_keywords = ["image", "picture", "photo", "visual", "see"]
        if any(word in task_lower for word in image_keywords):
            required.append(ModalityType.IMAGE)
        
        # Check for audio-related tasks
        audio_keywords = ["audio", "sound", "listen", "hear", "speech"]
        if any(word in task_lower for word in audio_keywords):
            required.append(ModalityType.AUDIO)
        
        # Check for video-related tasks
        video_keywords = ["video", "movie", "clip", "watch"]
        if any(word in task_lower for word in video_keywords):
            required.append(ModalityType.VIDEO)
        
        return required
    
    def detect_modality(self, data: Dict[str, Any]) -> ModalityType:
        """
        Detect modality of input data.
        
        Args:
            data: Input data dictionary
            
        Returns:
            Detected modality type
        """
        if "image" in data or "image_path" in data:
            return ModalityType.IMAGE
        elif "audio" in data or "audio_path" in data:
            return ModalityType.AUDIO
        elif "video" in data or "video_path" in data:
            return ModalityType.VIDEO
        elif "text" in data:
            return ModalityType.TEXT
        else:
            return ModalityType.MULTIMODAL
    
    def generate_reasoning(
        self,
        task: str,
        context_analysis: Dict[str, Any]
    ) -> str:
        """
        Generate reasoning considering multimodal context.
        
        Args:
            task: Task description
            context_analysis: Context analysis result
            
        Returns:
            Reasoning string
        """
        reasoning_parts = []
        modalities = context_analysis.get("modalities_present", [])
        
        modality_reasoning = {
            ModalityType.TEXT.value: "I have text input to process.",
            ModalityType.IMAGE.value: "I need to analyze the image content.",
            ModalityType.AUDIO.value: "I need to process the audio input.",
            ModalityType.VIDEO.value: "I need to analyze the video content.",
        }
        
        for modality_value in modalities:
            if modality_value in modality_reasoning:
                reasoning_parts.append(modality_reasoning[modality_value])
        
        if len(modalities) > 1:
            reasoning_parts.append(
                "I'll integrate information across multiple modalities."
            )
        
        reasoning_parts.append(f"Task: {task}")
        
        return " ".join(reasoning_parts) if reasoning_parts else f"Processing: {task}"



