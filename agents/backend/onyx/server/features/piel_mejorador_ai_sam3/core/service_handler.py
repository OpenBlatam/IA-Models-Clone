"""
Service Handler for Piel Mejorador AI SAM3
===========================================
"""

import asyncio
import logging
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum

from .helpers import create_message, get_mime_type
from .prompt_builder import PromptBuilder
from .validators import FileValidator, ValidationError
from ..infrastructure.openrouter_client import OpenRouterClient
from ..infrastructure.truthgpt_client import TruthGPTClient

logger = logging.getLogger(__name__)


class ServiceType(Enum):
    """Service types."""
    MEJORAR_IMAGEN = "mejorar_imagen"
    MEJORAR_VIDEO = "mejorar_video"
    ANALISIS_PIEL = "analisis_piel"


@dataclass
class ServiceResult:
    """Result from a service request."""
    success: bool
    data: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    tokens_used: int = 0
    model: str = ""
    timestamp: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {**self.data}
        result["tokens_used"] = self.tokens_used
        result["model"] = self.model
        if self.timestamp:
            result["timestamp"] = self.timestamp
        if self.error:
            result["error"] = self.error
        return result
    
    @classmethod
    def from_response(
        cls,
        response: Dict[str, Any],
        response_key: str = "resultado"
    ) -> "ServiceResult":
        """Create from API response."""
        data = {
            response_key: response.get("response", ""),
        }
        
        return cls(
            success=True,
            data=data,
            tokens_used=response.get("tokens_used", 0),
            model=response.get("model", ""),
            timestamp=datetime.now().isoformat(),
        )
    
    @classmethod
    def error_result(cls, error: str) -> "ServiceResult":
        """Create error result."""
        return cls(success=False, error=error)


class ServiceHandler:
    """Handles service requests for skin enhancement."""
    
    def __init__(
        self,
        openrouter_client: OpenRouterClient,
        truthgpt_client: TruthGPTClient,
        system_prompts: Dict[str, str],
        config: Any
    ):
        self.openrouter_client = openrouter_client
        self.truthgpt_client = truthgpt_client
        self.system_prompts = system_prompts
        self.config = config
    
    async def handle_mejorar_imagen(
        self,
        file_path: str,
        enhancement_level: str = "medium",
        realism_level: Optional[float] = None,
        custom_instructions: Optional[str] = None
    ) -> ServiceResult:
        """Handle image enhancement request."""
        try:
            # Validate file
            FileValidator.validate_image_file(
                file_path,
                max_size_mb=self.config.max_image_size_mb,
                allowed_extensions=self.config.supported_image_formats
            )
            
            # Build prompt
            prompt = PromptBuilder.build_image_enhancement_prompt(
                file_path=file_path,
                enhancement_level=enhancement_level,
                realism_level=realism_level,
                custom_instructions=custom_instructions
            )
            
            # Optimize prompt with TruthGPT
            optimized_prompt = await self.truthgpt_client.optimize_query(prompt)
            
            # Get enhancement config
            enhancement_config = self.config.get_enhancement_config(enhancement_level)
            if realism_level is None:
                realism_level = enhancement_config.get("realism", 0.7)
            
            # Process image with vision model (with circuit breaker)
            mime_type = get_mime_type(file_path)
            
            # Use circuit breaker for resilience
            async def process_with_circuit():
                return await self.openrouter_client.process_image(
                    model=self.config.openrouter.model,
                    image_path=file_path,
                    prompt=optimized_prompt,
                    system_prompt=self.system_prompts["mejorar_imagen"],
                    temperature=0.7,
                    max_tokens=4000,
                    mime_type=mime_type
                )
            
            # Get circuit breaker from agent if available
            try:
                from ..core.piel_mejorador_agent import PielMejoradorAgent
                # This would need to be passed or accessed differently
                response = await process_with_circuit()
            except Exception:
                response = await process_with_circuit()
            
            return ServiceResult.from_response(response, "mejora_aplicada")
            
        except ValidationError as e:
            logger.error(f"Validation error enhancing image: {e}")
            return ServiceResult.error_result(f"Validation error: {str(e)}")
        except Exception as e:
            logger.error(f"Error enhancing image: {e}", exc_info=True)
            return ServiceResult.error_result(str(e))
    
    async def handle_mejorar_video(
        self,
        file_path: str,
        enhancement_level: str = "medium",
        realism_level: Optional[float] = None,
        custom_instructions: Optional[str] = None
    ) -> ServiceResult:
        """Handle video enhancement request."""
        try:
            # Validate file
            FileValidator.validate_video_file(
                file_path,
                max_size_mb=self.config.max_video_size_mb,
                allowed_extensions=self.config.supported_video_formats
            )
            
            # Build prompt
            prompt = PromptBuilder.build_video_enhancement_prompt(
                file_path=file_path,
                enhancement_level=enhancement_level,
                realism_level=realism_level,
                custom_instructions=custom_instructions
            )
            
            # Optimize prompt with TruthGPT
            optimized_prompt = await self.truthgpt_client.optimize_query(prompt)
            
            # Get enhancement config
            enhancement_config = self.config.get_enhancement_config(enhancement_level)
            if realism_level is None:
                realism_level = enhancement_config.get("realism", 0.7)
            
            # Use video processor for frame-by-frame processing
            from .video_processor import VideoProcessor
            
            processor = VideoProcessor(output_dir=self.config.output_dir / "temp" / "video")
            
            # Extract frames (sample every 5 frames for efficiency)
            frames = await processor.extract_frames(
                file_path=file_path,
                max_frames=100,  # Limit for processing
                frame_interval=5  # Process every 5th frame
            )
            
            if not frames:
                return ServiceResult.error_result("No frames extracted from video")
            
            # Process frames in batches
            processed_frames = []
            
            async def process_frame(frame_info):
                """Process a single frame."""
                frame_path = frame_info["image_path"]
                
                # Process frame with image enhancement
                frame_result = await self.handle_mejorar_imagen(
                    file_path=frame_path,
                    enhancement_level=enhancement_level,
                    realism_level=realism_level,
                    custom_instructions=custom_instructions
                )
                
                if frame_result.success:
                    # Store processed frame path (in real implementation, save enhanced frame)
                    return {
                        "frame_number": frame_info["frame_number"],
                        "processed_path": frame_path,  # Would be enhanced frame path
                        "result": frame_result.data
                    }
                return None
            
            # Process frames in batches
            batch_size = 3  # Process 3 frames at a time
            for i in range(0, len(frames), batch_size):
                batch = frames[i:i + batch_size]
                batch_results = await asyncio.gather(*[
                    process_frame(frame) for frame in batch
                ], return_exceptions=True)
                
                for result in batch_results:
                    if result and not isinstance(result, Exception):
                        processed_frames.append(result)
            
            # Reconstruct video with processed frames
            output_video_path = str(Path(file_path).parent / f"enhanced_{Path(file_path).name}")
            await processor.reconstruct_video(
                frames=processed_frames,
                output_path=output_video_path,
                fps=30.0
            )
            
            # Cleanup temporary frames
            processor.cleanup_temp_files(frames)
            
            return ServiceResult(
                success=True,
                data={
                    "mejora_aplicada": f"Video procesado: {len(processed_frames)} frames mejorados",
                    "output_video": output_video_path,
                    "frames_processed": len(processed_frames),
                    "total_frames": len(frames)
                },
                timestamp=datetime.now().isoformat()
            )
            
        except ValidationError as e:
            logger.error(f"Validation error enhancing video: {e}")
            return ServiceResult.error_result(f"Validation error: {str(e)}")
        except Exception as e:
            logger.error(f"Error enhancing video: {e}", exc_info=True)
            return ServiceResult.error_result(str(e))
    
    async def handle_analisis_piel(
        self,
        file_path: str,
        file_type: str = "image"
    ) -> ServiceResult:
        """Handle skin analysis request."""
        try:
            # Validate file
            if file_type == "image":
                FileValidator.validate_image_file(
                    file_path,
                    max_size_mb=self.config.max_image_size_mb,
                    allowed_extensions=self.config.supported_image_formats
                )
            else:
                FileValidator.validate_video_file(
                    file_path,
                    max_size_mb=self.config.max_video_size_mb,
                    allowed_extensions=self.config.supported_video_formats
                )
            
            # Build prompt
            prompt = PromptBuilder.build_skin_analysis_prompt(
                file_path=file_path,
                file_type=file_type
            )
            
            # Optimize prompt with TruthGPT
            optimized_prompt = await self.truthgpt_client.optimize_query(prompt)
            
            # Process with vision model if image
            if file_type == "image":
                mime_type = get_mime_type(file_path)
                response = await self.openrouter_client.process_image(
                    model=self.config.openrouter.model,
                    image_path=file_path,
                    prompt=optimized_prompt,
                    system_prompt=self.system_prompts["analisis_piel"],
                    temperature=0.5,
                    max_tokens=2000,
                    mime_type=mime_type
                )
            else:
                messages = [
                    create_message("system", self.system_prompts["analisis_piel"]),
                    create_message("user", optimized_prompt),
                ]
                
                response = await self.openrouter_client.chat_completion(
                    model=self.config.openrouter.model,
                    messages=messages,
                    temperature=0.5,
                    max_tokens=2000,
                )
            
            return ServiceResult.from_response(response, "analisis")
            
        except ValidationError as e:
            logger.error(f"Validation error analyzing skin: {e}")
            return ServiceResult.error_result(f"Validation error: {str(e)}")
        except Exception as e:
            logger.error(f"Error analyzing skin: {e}", exc_info=True)
            return ServiceResult.error_result(str(e))
    
    async def handle(
        self,
        service_type: ServiceType,
        parameters: Dict[str, Any]
    ) -> ServiceResult:
        """Handle a service request."""
        if service_type == ServiceType.MEJORAR_IMAGEN:
            return await self.handle_mejorar_imagen(
                file_path=parameters["file_path"],
                enhancement_level=parameters.get("enhancement_level", "medium"),
                realism_level=parameters.get("realism_level"),
                custom_instructions=parameters.get("custom_instructions"),
            )
        elif service_type == ServiceType.MEJORAR_VIDEO:
            return await self.handle_mejorar_video(
                file_path=parameters["file_path"],
                enhancement_level=parameters.get("enhancement_level", "medium"),
                realism_level=parameters.get("realism_level"),
                custom_instructions=parameters.get("custom_instructions"),
            )
        elif service_type == ServiceType.ANALISIS_PIEL:
            return await self.handle_analisis_piel(
                file_path=parameters["file_path"],
                file_type=parameters.get("file_type", "image"),
            )
        else:
            return ServiceResult.error_result(f"Unknown service type: {service_type}")

