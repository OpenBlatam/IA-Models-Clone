"""
Video Transitions Service
Creates smooth transitions between video segments
"""

import asyncio
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class TransitionService:
    """Creates transitions between video segments"""
    
    TRANSITION_TYPES = {
        "fade": "fade",
        "crossfade": "xfade",
        "slide": "slide",
        "zoom": "zoom",
        "rotate": "rotate",
        "none": None,
    }
    
    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = Path(output_dir) if output_dir else Path("/tmp/faceless_video/transitions")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    async def add_transitions(
        self,
        video_path: Path,
        segments: List[Dict[str, Any]],
        transition_type: str = "fade",
        duration: float = 0.5,
        output_path: Optional[Path] = None
    ) -> Path:
        """
        Add transitions between video segments
        
        Args:
            video_path: Input video path
            segments: List of segments with timing
            transition_type: Type of transition
            duration: Transition duration in seconds
            output_path: Output video path
            
        Returns:
            Path to video with transitions
        """
        if transition_type == "none" or not transition_type:
            return video_path
        
        if output_path is None:
            output_path = self.output_dir / f"with_transitions_{video_path.stem}.mp4"
        
        # Build FFmpeg filter for transitions
        filter_complex = self._build_transition_filter(
            segments=segments,
            transition_type=transition_type,
            duration=duration
        )
        
        if not filter_complex:
            return video_path
        
        cmd = [
            "ffmpeg",
            "-i", str(video_path),
            "-vf", filter_complex,
            "-c:a", "copy",
            "-y",
            str(output_path)
        ]
        
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await process.communicate()
            
            if process.returncode != 0:
                logger.warning("Transition application failed, returning original video")
                return video_path
            
            logger.info(f"Transitions added: {output_path}")
            return output_path
            
        except Exception as e:
            logger.warning(f"Transition error: {str(e)}, returning original video")
            return video_path
    
    def _build_transition_filter(
        self,
        segments: List[Dict[str, Any]],
        transition_type: str,
        duration: float
    ) -> Optional[str]:
        """Build FFmpeg filter complex for transitions"""
        if transition_type == "fade":
            # Simple fade in/out
            return f"fade=t=in:st=0:d={duration},fade=t=out:st={{duration-{duration}}}:d={duration}"
        
        elif transition_type == "crossfade":
            # Crossfade between segments
            # This is more complex and requires multiple inputs
            # For now, return a simple fade
            return f"fade=t=in:st=0:d={duration}"
        
        elif transition_type == "slide":
            # Slide transition
            return f"xfade=transition=slideleft:duration={duration}:offset={{duration-{duration}}}"
        
        elif transition_type == "zoom":
            # Zoom transition
            return f"zoompan=z='if(lte(zoom,1.0),1.5,max(1.001,zoom-0.0015))':d=25"
        
        return None
    
    def get_available_transitions(self) -> List[str]:
        """Get list of available transition types"""
        return list(self.TRANSITION_TYPES.keys())

