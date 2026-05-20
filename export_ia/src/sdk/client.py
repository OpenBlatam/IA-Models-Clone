"""
Export IA SDK Client for developers.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Union
import httpx
from datetime import datetime
import json

from .models import ExportRequest, ExportResponse, TaskStatus, QualityMetrics, SystemInfo
from .exceptions import ExportIAException, ValidationError, ExportError, ServiceUnavailableError

logger = logging.getLogger(__name__)


class ExportIAClient:
    """Synchronous client for Export IA API."""
    
    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: Optional[str] = None,
        timeout: int = 30
    ):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.timeout = timeout
        self.session = httpx.Client(
            base_url=self.base_url,
            timeout=self.timeout,
            headers=self._get_headers()
        )
    
    def _get_headers(self) -> Dict[str, str]:
        """Get request headers."""
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "ExportIA-SDK/2.0.0"
        }
        
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        return headers
    
    def export_document(
        self,
        content: Dict[str, Any],
        format: str = "pdf",
        document_type: str = "report",
        quality_level: str = "professional",
        **kwargs
    ) -> ExportResponse:
        """Export a document."""
        try:
            request = ExportRequest(
                content=content,
                format=format,
                document_type=document_type,
                quality_level=quality_level,
                **kwargs
            )
            
            response = self.session.post(
                "/export",
                json=request.dict()
            )
            response.raise_for_status()
            
            data = response.json()
            return ExportResponse(**data)
            
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 400:
                raise ValidationError(f"Validation error: {e.response.text}")
            elif e.response.status_code == 503:
                raise ServiceUnavailableError("Service temporarily unavailable")
            else:
                raise ExportError(f"Export failed: {e.response.text}")
        except httpx.RequestError as e:
            raise ExportIAException(f"Request failed: {e}")
    
    def get_task_status(self, task_id: str) -> TaskStatus:
        """Get task status."""
        try:
            response = self.session.get(f"/export/{task_id}/status")
            response.raise_for_status()
            
            data = response.json()
            return TaskStatus(**data)
            
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                raise ExportIAException(f"Task not found: {task_id}")
            else:
                raise ExportIAException(f"Failed to get task status: {e.response.text}")
        except httpx.RequestError as e:
            raise ExportIAException(f"Request failed: {e}")
    
    def download_export(self, task_id: str, save_path: Optional[str] = None) -> str:
        """Download exported file."""
        try:
            response = self.session.get(f"/export/{task_id}/download")
            response.raise_for_status()
            
            if save_path is None:
                # Generate filename from task_id
                save_path = f"export_{task_id}.pdf"
            
            with open(save_path, 'wb') as f:
                f.write(response.content)
            
            return save_path
            
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                raise ExportIAException(f"File not found for task: {task_id}")
            else:
                raise ExportIAException(f"Download failed: {e.response.text}")
        except httpx.RequestError as e:
            raise ExportIAException(f"Request failed: {e}")
    
    def validate_content(
        self,
        content: Dict[str, Any],
        format: str = "pdf",
        document_type: str = "report",
        quality_level: str = "professional"
    ) -> QualityMetrics:
        """Validate document content."""
        try:
            request = {
                "content": content,
                "config": {
                    "format": format,
                    "document_type": document_type,
                    "quality_level": quality_level
                }
            }
            
            response = self.session.post("/validate", json=request)
            response.raise_for_status()
            
            data = response.json()
            return QualityMetrics(**data)
            
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 400:
                raise ValidationError(f"Validation error: {e.response.text}")
            else:
                raise ExportIAException(f"Validation failed: {e.response.text}")
        except httpx.RequestError as e:
            raise ExportIAException(f"Request failed: {e}")
    
    def get_system_info(self) -> SystemInfo:
        """Get system information."""
        try:
            response = self.session.get("/")
            response.raise_for_status()
            
            data = response.json()
            return SystemInfo(**data)
            
        except httpx.RequestError as e:
            raise ExportIAException(f"Request failed: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get system statistics."""
        try:
            response = self.session.get("/statistics")
            response.raise_for_status()
            
            return response.json()
            
        except httpx.RequestError as e:
            raise ExportIAException(f"Request failed: {e}")
    
    def list_supported_formats(self) -> List[Dict[str, Any]]:
        """List supported export formats."""
        try:
            response = self.session.get("/formats")
            response.raise_for_status()
            
            return response.json()
            
        except httpx.RequestError as e:
            raise ExportIAException(f"Request failed: {e}")
    
    def close(self):
        """Close the client session."""
        self.session.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class AsyncExportIAClient:
    """Asynchronous client for Export IA API."""
    
    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: Optional[str] = None,
        timeout: int = 30
    ):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.timeout = timeout
        self.session = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=self.timeout,
            headers=self._get_headers()
        )
    
    def _get_headers(self) -> Dict[str, str]:
        """Get request headers."""
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "ExportIA-SDK/2.0.0"
        }
        
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        return headers
    
    async def export_document(
        self,
        content: Dict[str, Any],
        format: str = "pdf",
        document_type: str = "report",
        quality_level: str = "professional",
        **kwargs
    ) -> ExportResponse:
        """Export a document asynchronously."""
        try:
            request = ExportRequest(
                content=content,
                format=format,
                document_type=document_type,
                quality_level=quality_level,
                **kwargs
            )
            
            response = await self.session.post(
                "/export",
                json=request.dict()
            )
            response.raise_for_status()
            
            data = response.json()
            return ExportResponse(**data)
            
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 400:
                raise ValidationError(f"Validation error: {e.response.text}")
            elif e.response.status_code == 503:
                raise ServiceUnavailableError("Service temporarily unavailable")
            else:
                raise ExportError(f"Export failed: {e.response.text}")
        except httpx.RequestError as e:
            raise ExportIAException(f"Request failed: {e}")
    
    async def get_task_status(self, task_id: str) -> TaskStatus:
        """Get task status asynchronously."""
        try:
            response = await self.session.get(f"/export/{task_id}/status")
            response.raise_for_status()
            
            data = response.json()
            return TaskStatus(**data)
            
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                raise ExportIAException(f"Task not found: {task_id}")
            else:
                raise ExportIAException(f"Failed to get task status: {e.response.text}")
        except httpx.RequestError as e:
            raise ExportIAException(f"Request failed: {e}")
    
    async def download_export(self, task_id: str, save_path: Optional[str] = None) -> str:
        """Download exported file asynchronously."""
        try:
            response = await self.session.get(f"/export/{task_id}/download")
            response.raise_for_status()
            
            if save_path is None:
                # Generate filename from task_id
                save_path = f"export_{task_id}.pdf"
            
            with open(save_path, 'wb') as f:
                f.write(response.content)
            
            return save_path
            
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                raise ExportIAException(f"File not found for task: {task_id}")
            else:
                raise ExportIAException(f"Download failed: {e.response.text}")
        except httpx.RequestError as e:
            raise ExportIAException(f"Request failed: {e}")
    
    async def validate_content(
        self,
        content: Dict[str, Any],
        format: str = "pdf",
        document_type: str = "report",
        quality_level: str = "professional"
    ) -> QualityMetrics:
        """Validate document content asynchronously."""
        try:
            request = {
                "content": content,
                "config": {
                    "format": format,
                    "document_type": document_type,
                    "quality_level": quality_level
                }
            }
            
            response = await self.session.post("/validate", json=request)
            response.raise_for_status()
            
            data = response.json()
            return QualityMetrics(**data)
            
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 400:
                raise ValidationError(f"Validation error: {e.response.text}")
            else:
                raise ExportIAException(f"Validation failed: {e.response.text}")
        except httpx.RequestError as e:
            raise ExportIAException(f"Request failed: {e}")
    
    async def get_system_info(self) -> SystemInfo:
        """Get system information asynchronously."""
        try:
            response = await self.session.get("/")
            response.raise_for_status()
            
            data = response.json()
            return SystemInfo(**data)
            
        except httpx.RequestError as e:
            raise ExportIAException(f"Request failed: {e}")
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get system statistics asynchronously."""
        try:
            response = await self.session.get("/statistics")
            response.raise_for_status()
            
            return response.json()
            
        except httpx.RequestError as e:
            raise ExportIAException(f"Request failed: {e}")
    
    async def list_supported_formats(self) -> List[Dict[str, Any]]:
        """List supported export formats asynchronously."""
        try:
            response = await self.session.get("/formats")
            response.raise_for_status()
            
            return response.json()
            
        except httpx.RequestError as e:
            raise ExportIAException(f"Request failed: {e}")
    
    async def wait_for_completion(
        self,
        task_id: str,
        poll_interval: int = 2,
        timeout: int = 300
    ) -> TaskStatus:
        """Wait for task completion."""
        start_time = datetime.now()
        
        while True:
            status = await self.get_task_status(task_id)
            
            if status.status in ["completed", "failed", "cancelled"]:
                return status
            
            # Check timeout
            elapsed = (datetime.now() - start_time).total_seconds()
            if elapsed > timeout:
                raise ExportIAException(f"Task {task_id} timed out after {timeout} seconds")
            
            await asyncio.sleep(poll_interval)
    
    async def close(self):
        """Close the client session."""
        await self.session.aclose()
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()


# Convenience functions
def create_client(
    base_url: str = "http://localhost:8000",
    api_key: Optional[str] = None,
    timeout: int = 30
) -> ExportIAClient:
    """Create a synchronous Export IA client."""
    return ExportIAClient(base_url, api_key, timeout)


def create_async_client(
    base_url: str = "http://localhost:8000",
    api_key: Optional[str] = None,
    timeout: int = 30
) -> AsyncExportIAClient:
    """Create an asynchronous Export IA client."""
    return AsyncExportIAClient(base_url, api_key, timeout)




