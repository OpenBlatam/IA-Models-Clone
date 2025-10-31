"""
BUL Integration Service for Gamma App
====================================

Integrates the BUL (Business Universal Language) system with Gamma App
to provide advanced document generation capabilities for enterprises.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import uuid
import json
import requests
from enum import Enum

logger = logging.getLogger(__name__)

class DocumentType(str, Enum):
    """Types of documents that can be generated."""
    STRATEGY = "estrategia"
    PROPOSAL = "propuesta"
    MANUAL = "manual"
    POLICY = "política"
    REPORT = "reporte"
    TEMPLATE = "plantilla"

class BusinessArea(str, Enum):
    """Business areas supported by BUL."""
    MARKETING = "marketing"
    SALES = "ventas"
    OPERATIONS = "operaciones"
    HR = "rrhh"
    FINANCE = "finanzas"
    LEGAL = "legal"
    TECHNICAL = "técnico"
    CONTENT = "contenido"
    STRATEGY = "estrategia"
    CUSTOMER_SERVICE = "atencion_cliente"

class TaskStatus(str, Enum):
    """Status of BUL tasks."""
    QUEUED = "en_cola"
    PROCESSING = "procesando"
    COMPLETED = "completado"
    FAILED = "fallido"
    CANCELLED = "cancelado"

@dataclass
class BULDocumentRequest:
    """Request for document generation."""
    query: str
    business_area: Optional[BusinessArea] = None
    document_type: Optional[DocumentType] = None
    priority: int = 1
    user_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class BULDocumentResponse:
    """Response from document generation."""
    task_id: str
    status: TaskStatus
    message: str
    estimated_time: Optional[int] = None
    document_id: Optional[str] = None

@dataclass
class BULTask:
    """BUL task information."""
    task_id: str
    status: TaskStatus
    progress: int
    request: BULDocumentRequest
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None

@dataclass
class BULDocument:
    """Generated document information."""
    document_id: str
    title: str
    content: str
    format: str
    word_count: int
    business_area: BusinessArea
    document_type: DocumentType
    query: str
    generated_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

class BULIntegrationService:
    """Service for integrating BUL system with Gamma App."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.bul_api_url = self.config.get("bul_api_url", "http://localhost:8000")
        self.api_key = self.config.get("bul_api_key")
        self.tasks: Dict[str, BULTask] = {}
        self.documents: Dict[str, BULDocument] = {}
        self.session = requests.Session()
        
        # Configure session
        if self.api_key:
            self.session.headers.update({"Authorization": f"Bearer {self.api_key}"})
        
        logger.info("BULIntegrationService initialized")
    
    async def generate_document(self, request: BULDocumentRequest) -> BULDocumentResponse:
        """Generate a document using BUL system."""
        try:
            # Prepare request data
            request_data = {
                "query": request.query,
                "business_area": request.business_area.value if request.business_area else None,
                "document_type": request.document_type.value if request.document_type else None,
                "priority": request.priority
            }
            
            # Make API call to BUL system
            response = self.session.post(
                f"{self.bul_api_url}/documents/generate",
                json=request_data,
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            
            # Create task tracking
            task_id = result["task_id"]
            task = BULTask(
                task_id=task_id,
                status=TaskStatus(result["status"]),
                progress=0,
                request=request
            )
            self.tasks[task_id] = task
            
            return BULDocumentResponse(
                task_id=task_id,
                status=TaskStatus(result["status"]),
                message=result["message"],
                estimated_time=result.get("estimated_time")
            )
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error calling BUL API: {e}")
            raise Exception(f"Failed to generate document: {e}")
        except Exception as e:
            logger.error(f"Error generating document: {e}")
            raise
    
    async def get_task_status(self, task_id: str) -> Optional[BULTask]:
        """Get the status of a BUL task."""
        try:
            # Check local cache first
            if task_id in self.tasks:
                local_task = self.tasks[task_id]
                
                # If task is still processing, check with BUL API
                if local_task.status in [TaskStatus.QUEUED, TaskStatus.PROCESSING]:
                    response = self.session.get(
                        f"{self.bul_api_url}/tasks/{task_id}/status",
                        timeout=10
                    )
                    response.raise_for_status()
                    
                    result = response.json()
                    
                    # Update local task
                    local_task.status = TaskStatus(result["status"])
                    local_task.progress = result["progress"]
                    local_task.error = result.get("error")
                    
                    # If completed, store the document
                    if result["status"] == "completado" and result.get("result"):
                        await self._store_document(result["result"])
                
                return local_task
            
            return None
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error getting task status: {e}")
            return None
        except Exception as e:
            logger.error(f"Error getting task status: {e}")
            return None
    
    async def get_document(self, document_id: str) -> Optional[BULDocument]:
        """Get a generated document."""
        return self.documents.get(document_id)
    
    async def list_documents(self, 
                           business_area: Optional[BusinessArea] = None,
                           document_type: Optional[DocumentType] = None,
                           limit: int = 50,
                           offset: int = 0) -> List[BULDocument]:
        """List generated documents with optional filtering."""
        documents = list(self.documents.values())
        
        # Apply filters
        if business_area:
            documents = [d for d in documents if d.business_area == business_area]
        
        if document_type:
            documents = [d for d in documents if d.document_type == document_type]
        
        # Sort by generation date (newest first)
        documents.sort(key=lambda x: x.generated_at, reverse=True)
        
        # Apply pagination
        return documents[offset:offset + limit]
    
    async def search_documents(self, query: str, limit: int = 20) -> List[BULDocument]:
        """Search documents by content or title."""
        query_lower = query.lower()
        matching_documents = []
        
        for document in self.documents.values():
            if (query_lower in document.title.lower() or 
                query_lower in document.content.lower() or
                query_lower in document.query.lower()):
                matching_documents.append(document)
        
        # Sort by relevance (simple implementation)
        matching_documents.sort(key=lambda x: x.generated_at, reverse=True)
        
        return matching_documents[:limit]
    
    async def get_business_areas(self) -> Dict[str, List[str]]:
        """Get available business areas from BUL system."""
        try:
            response = self.session.get(
                f"{self.bul_api_url}/business-areas",
                timeout=10
            )
            response.raise_for_status()
            
            result = response.json()
            return result.get("business_areas", {})
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error getting business areas: {e}")
            return {}
        except Exception as e:
            logger.error(f"Error getting business areas: {e}")
            return {}
    
    async def get_document_types(self) -> Dict[str, Dict[str, str]]:
        """Get available document types from BUL system."""
        try:
            response = self.session.get(
                f"{self.bul_api_url}/document-types",
                timeout=10
            )
            response.raise_for_status()
            
            result = response.json()
            return result.get("document_types", {})
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error getting document types: {e}")
            return {}
        except Exception as e:
            logger.error(f"Error getting document types: {e}")
            return {}
    
    async def delete_task(self, task_id: str) -> bool:
        """Delete a BUL task."""
        try:
            response = self.session.delete(
                f"{self.bul_api_url}/tasks/{task_id}",
                timeout=10
            )
            response.raise_for_status()
            
            # Remove from local cache
            if task_id in self.tasks:
                del self.tasks[task_id]
            
            return True
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error deleting task: {e}")
            return False
        except Exception as e:
            logger.error(f"Error deleting task: {e}")
            return False
    
    async def get_system_health(self) -> Dict[str, Any]:
        """Get BUL system health status."""
        try:
            response = self.session.get(
                f"{self.bul_api_url}/health",
                timeout=10
            )
            response.raise_for_status()
            
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error getting system health: {e}")
            return {"status": "unhealthy", "error": str(e)}
        except Exception as e:
            logger.error(f"Error getting system health: {e}")
            return {"status": "unhealthy", "error": str(e)}
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get BUL integration statistics."""
        total_tasks = len(self.tasks)
        completed_tasks = len([t for t in self.tasks.values() if t.status == TaskStatus.COMPLETED])
        failed_tasks = len([t for t in self.tasks.values() if t.status == TaskStatus.FAILED])
        total_documents = len(self.documents)
        
        # Business area distribution
        business_area_stats = {}
        for document in self.documents.values():
            area = document.business_area.value
            business_area_stats[area] = business_area_stats.get(area, 0) + 1
        
        # Document type distribution
        document_type_stats = {}
        for document in self.documents.values():
            doc_type = document.document_type.value
            document_type_stats[doc_type] = document_type_stats.get(doc_type, 0) + 1
        
        return {
            "total_tasks": total_tasks,
            "completed_tasks": completed_tasks,
            "failed_tasks": failed_tasks,
            "success_rate": (completed_tasks / total_tasks * 100) if total_tasks > 0 else 0,
            "total_documents": total_documents,
            "business_area_distribution": business_area_stats,
            "document_type_distribution": document_type_stats,
            "last_updated": datetime.now().isoformat()
        }
    
    async def _store_document(self, result: Dict[str, Any]) -> None:
        """Store a completed document."""
        try:
            document = BULDocument(
                document_id=result["document_id"],
                title=result["title"],
                content=result["content"],
                format=result["format"],
                word_count=result["word_count"],
                business_area=BusinessArea(result["business_area"]),
                document_type=DocumentType(result["document_type"]),
                query=result["query"],
                generated_at=datetime.fromisoformat(result["generated_at"]),
                metadata=result.get("metadata", {})
            )
            
            self.documents[document.document_id] = document
            logger.info(f"Stored document: {document.document_id}")
            
        except Exception as e:
            logger.error(f"Error storing document: {e}")
    
    async def cleanup_old_tasks(self, max_age_hours: int = 24) -> int:
        """Clean up old completed tasks."""
        cutoff_time = datetime.now().timestamp() - (max_age_hours * 3600)
        cleaned_count = 0
        
        tasks_to_remove = []
        for task_id, task in self.tasks.items():
            if (task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED] and
                task.created_at.timestamp() < cutoff_time):
                tasks_to_remove.append(task_id)
        
        for task_id in tasks_to_remove:
            del self.tasks[task_id]
            cleaned_count += 1
        
        logger.info(f"Cleaned up {cleaned_count} old tasks")
        return cleaned_count





















