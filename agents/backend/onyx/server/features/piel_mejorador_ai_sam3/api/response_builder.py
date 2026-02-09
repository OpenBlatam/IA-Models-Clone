"""
Response building utilities for Piel Mejorador AI SAM3 API.
"""

from typing import Dict, Any


class ResponseBuilder:
    """
    Builds consistent API responses.
    """
    
    @staticmethod
    def task_submitted(task_id: str) -> Dict[str, Any]:
        """
        Create a task submission response.
        
        Args:
            task_id: Task identifier
            
        Returns:
            Response dictionary with task_id and status
        """
        return {
            "task_id": task_id,
            "status": "submitted"
        }
    
    @staticmethod
    def health_check(agent_running: bool) -> Dict[str, Any]:
        """
        Create a health check response.
        
        Args:
            agent_running: Whether the agent is running
            
        Returns:
            Response dictionary with status and agent state
        """
        return {
            "status": "healthy",
            "agent_running": agent_running
        }




