"""
Task Creator - Creates document generation tasks
================================================

Handles creation of document generation tasks from bulk requests.
"""

import uuid
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class TaskCreator:
    """Creates document generation tasks from bulk requests."""
    
    @staticmethod
    def create_initial_tasks(
        request_id: str,
        query: str,
        document_types: List[str],
        business_areas: List[str],
        max_documents: int,
        priority: int,
        task_class: type
    ) -> List[Any]:
        """
        Create initial tasks for a bulk request.
        
        Args:
            request_id: ID of the bulk request
            query: Query/topic for document generation
            document_types: List of document types to generate
            business_areas: List of business areas to focus on
            max_documents: Maximum number of documents to generate
            priority: Priority level
            task_class: Class to use for creating tasks
            
        Returns:
            List of created tasks
        """
        tasks = []
        tasks_created = 0
        
        for doc_type in document_types:
            for business_area in business_areas:
                if tasks_created >= max_documents:
                    break
                
                task = task_class(
                    id=str(uuid.uuid4()),
                    request_id=request_id,
                    document_type=doc_type,
                    business_area=business_area,
                    query=query,
                    priority=priority
                )
                
                tasks.append(task)
                tasks_created += 1
        
        logger.debug(f"Created {len(tasks)} initial tasks for request {request_id}")
        return tasks
    
    @staticmethod
    def create_additional_tasks(
        request_id: str,
        query: str,
        document_types: List[str],
        business_areas: List[str],
        current_count: int,
        max_documents: int,
        priority: int,
        task_class: type
    ) -> List[Any]:
        """
        Create additional tasks for continuous generation.
        
        Args:
            request_id: ID of the bulk request
            query: Query/topic for document generation
            document_types: List of document types to generate
            business_areas: List of business areas to focus on
            current_count: Current number of tasks created
            max_documents: Maximum number of documents to generate
            priority: Priority level
            task_class: Class to use for creating tasks
            
        Returns:
            List of created tasks
        """
        tasks = []
        remaining = max_documents - current_count
        
        for i in range(remaining):
            doc_type = document_types[i % len(document_types)]
            business_area = business_areas[i % len(business_areas)]
            variation_number = (i // len(document_types)) + 1
            
            task = task_class(
                id=str(uuid.uuid4()),
                request_id=request_id,
                document_type=doc_type,
                business_area=business_area,
                query=query,
                priority=priority
            )
            
            if hasattr(task, 'metadata'):
                if task.metadata is None:
                    task.metadata = {}
                task.metadata["variation_number"] = variation_number
            
            tasks.append(task)
        
        logger.debug(f"Created {len(tasks)} additional tasks for request {request_id}")
        return tasks

