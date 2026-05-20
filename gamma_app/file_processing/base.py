"""
File Processing Base Classes and Interfaces
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from datetime import datetime
from uuid import uuid4
from enum import Enum


class FileType(str, Enum):
    """File types"""
    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"
    IMAGE = "image"
    HTML = "html"
    MARKDOWN = "markdown"


class ProcessedFile:
    """Processed file model"""
    
    def __init__(
        self,
        file_id: str,
        file_type: FileType,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.id = str(uuid4())
        self.file_id = file_id
        self.file_type = file_type
        self.content = content
        self.metadata = metadata or {}
        self.processed_at = datetime.utcnow()


class ProcessingResult:
    """Processing result"""
    
    def __init__(
        self,
        success: bool,
        content: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None
    ):
        self.success = success
        self.content = content
        self.metadata = metadata or {}
        self.error = error


class FileProcessor:
    """File processor definition"""
    
    def __init__(self, file_type: FileType, processor_name: str):
        self.file_type = file_type
        self.processor_name = processor_name


class FileProcessorBase(ABC):
    """Base interface for file processing"""
    
    @abstractmethod
    async def process_file(
        self,
        file_path: str,
        file_type: FileType
    ) -> ProcessingResult:
        """Process a file"""
        pass
    
    @abstractmethod
    async def extract_text(
        self,
        file_path: str,
        file_type: FileType
    ) -> str:
        """Extract text from file"""
        pass
    
    @abstractmethod
    async def convert_format(
        self,
        file_path: str,
        from_format: FileType,
        to_format: FileType
    ) -> str:
        """Convert file format"""
        pass

