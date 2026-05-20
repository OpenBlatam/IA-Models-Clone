"""
Task Creation Utilities for Enhancer Agent
===========================================
"""

from typing import Dict, Any, Optional


class TaskCreator:
    """
    Creates tasks with consistent patterns.
    """
    
    @staticmethod
    async def create_enhance_image_task(
        task_manager: Any,
        file_path: str,
        enhancement_type: str = "general",
        options: Optional[Dict[str, Any]] = None,
        priority: int = 0
    ) -> str:
        """Create image enhancement task."""
        return await task_manager.create_task(
            service_type="enhance_image",
            parameters={
                "file_path": file_path,
                "enhancement_type": enhancement_type,
                "options": options or {},
            },
            priority=priority,
        )
    
    @staticmethod
    async def create_enhance_video_task(
        task_manager: Any,
        file_path: str,
        enhancement_type: str = "general",
        options: Optional[Dict[str, Any]] = None,
        priority: int = 0
    ) -> str:
        """Create video enhancement task."""
        return await task_manager.create_task(
            service_type="enhance_video",
            parameters={
                "file_path": file_path,
                "enhancement_type": enhancement_type,
                "options": options or {},
            },
            priority=priority,
        )
    
    @staticmethod
    async def create_upscale_task(
        task_manager: Any,
        file_path: str,
        scale_factor: int = 2,
        options: Optional[Dict[str, Any]] = None,
        priority: int = 0
    ) -> str:
        """Create upscale task."""
        return await task_manager.create_task(
            service_type="upscale",
            parameters={
                "file_path": file_path,
                "scale_factor": scale_factor,
                "options": options or {},
            },
            priority=priority,
        )
    
    @staticmethod
    async def create_denoise_task(
        task_manager: Any,
        file_path: str,
        noise_level: str = "medium",
        options: Optional[Dict[str, Any]] = None,
        priority: int = 0
    ) -> str:
        """Create denoise task."""
        return await task_manager.create_task(
            service_type="denoise",
            parameters={
                "file_path": file_path,
                "noise_level": noise_level,
                "options": options or {},
            },
            priority=priority,
        )
    
    @staticmethod
    async def create_restore_task(
        task_manager: Any,
        file_path: str,
        damage_type: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
        priority: int = 0
    ) -> str:
        """Create restoration task."""
        return await task_manager.create_task(
            service_type="restore",
            parameters={
                "file_path": file_path,
                "damage_type": damage_type,
                "options": options or {},
            },
            priority=priority,
        )
    
    @staticmethod
    async def create_color_correction_task(
        task_manager: Any,
        file_path: str,
        correction_type: str = "auto",
        options: Optional[Dict[str, Any]] = None,
        priority: int = 0
    ) -> str:
        """Create color correction task."""
        return await task_manager.create_task(
            service_type="color_correction",
            parameters={
                "file_path": file_path,
                "correction_type": correction_type,
                "options": options or {},
            },
            priority=priority,
        )




