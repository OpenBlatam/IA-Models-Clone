"""
File upload and management service
"""

import os
import uuid
import aiofiles
from typing import Optional, List, Dict, Any
from pathlib import Path
from fastapi import UploadFile, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from ..models.database import FileUpload, User
from ..models.schemas import FileUploadResponse
from ..core.exceptions import FileUploadError, NotFoundError, DatabaseError
from ..config.settings import get_settings


class FileService:
    """Service for file upload and management operations."""
    
    def __init__(self, session: AsyncSession):
        self.session = session
        self.settings = get_settings()
        self.upload_dir = Path("uploads")
        self.upload_dir.mkdir(exist_ok=True)
    
    async def upload_file(
        self,
        file: UploadFile,
        uploaded_by: str,
        category: str = "general"
    ) -> FileUploadResponse:
        """Upload a file and save metadata to database."""
        try:
            # Validate file
            await self._validate_file(file)
            
            # Generate unique filename
            file_extension = Path(file.filename).suffix
            unique_filename = f"{uuid.uuid4()}{file_extension}"
            file_path = self.upload_dir / unique_filename
            
            # Save file to disk
            async with aiofiles.open(file_path, 'wb') as f:
                content = await file.read()
                await f.write(content)
            
            # Get file metadata
            file_size = len(content)
            mime_type = file.content_type or "application/octet-stream"
            
            # Create database record
            db_file = FileUpload(
                filename=unique_filename,
                original_filename=file.filename,
                file_path=str(file_path),
                file_size=file_size,
                mime_type=mime_type,
                uploaded_by=uploaded_by
            )
            
            self.session.add(db_file)
            await self.session.commit()
            await self.session.refresh(db_file)
            
            # Generate URL
            file_url = f"/files/{db_file.uuid}"
            
            return FileUploadResponse(
                id=db_file.id,
                uuid=str(db_file.uuid),
                filename=db_file.filename,
                original_filename=db_file.original_filename,
                file_size=db_file.file_size,
                mime_type=db_file.mime_type,
                url=file_url,
                created_at=db_file.created_at
            )
            
        except Exception as e:
            # Clean up file if database operation fails
            if 'file_path' in locals() and file_path.exists():
                file_path.unlink()
            
            await self.session.rollback()
            if isinstance(e, (FileUploadError,)):
                raise
            raise DatabaseError(f"Failed to upload file: {str(e)}")
    
    async def get_file(self, file_uuid: str) -> FileUploadResponse:
        """Get file metadata by UUID."""
        query = select(FileUpload).where(FileUpload.uuid == file_uuid)
        result = await self.session.execute(query)
        file_record = result.scalar_one_or_none()
        
        if not file_record:
            raise NotFoundError("File", file_uuid)
        
        file_url = f"/files/{file_record.uuid}"
        
        return FileUploadResponse(
            id=file_record.id,
            uuid=str(file_record.uuid),
            filename=file_record.filename,
            original_filename=file_record.original_filename,
            file_size=file_record.file_size,
            mime_type=file_record.mime_type,
            url=file_url,
            created_at=file_record.created_at
        )
    
    async def delete_file(self, file_uuid: str, user_id: str) -> bool:
        """Delete a file and its database record."""
        try:
            query = select(FileUpload).where(FileUpload.uuid == file_uuid)
            result = await self.session.execute(query)
            file_record = result.scalar_one_or_none()
            
            if not file_record:
                raise NotFoundError("File", file_uuid)
            
            # Check authorization (owner or admin)
            if file_record.uploaded_by != user_id:
                # In a real implementation, you would check user roles here
                pass
            
            # Delete file from disk
            file_path = Path(file_record.file_path)
            if file_path.exists():
                file_path.unlink()
            
            # Delete database record
            await self.session.delete(file_record)
            await self.session.commit()
            
            return True
            
        except Exception as e:
            await self.session.rollback()
            if isinstance(e, (NotFoundError,)):
                raise
            raise DatabaseError(f"Failed to delete file: {str(e)}")
    
    async def list_user_files(
        self,
        user_id: str,
        skip: int = 0,
        limit: int = 20
    ) -> List[FileUploadResponse]:
        """List files uploaded by a user."""
        try:
            query = select(FileUpload).where(FileUpload.uploaded_by == user_id)
            query = query.offset(skip).limit(limit).order_by(FileUpload.created_at.desc())
            
            result = await self.session.execute(query)
            files = result.scalars().all()
            
            return [
                FileUploadResponse(
                    id=file.id,
                    uuid=str(file.uuid),
                    filename=file.filename,
                    original_filename=file.original_filename,
                    file_size=file.file_size,
                    mime_type=file.mime_type,
                    url=f"/files/{file.uuid}",
                    created_at=file.created_at
                )
                for file in files
            ]
            
        except Exception as e:
            raise DatabaseError(f"Failed to list user files: {str(e)}")
    
    async def get_file_path(self, file_uuid: str) -> Path:
        """Get the file path for serving files."""
        query = select(FileUpload).where(FileUpload.uuid == file_uuid)
        result = await self.session.execute(query)
        file_record = result.scalar_one_or_none()
        
        if not file_record:
            raise NotFoundError("File", file_uuid)
        
        return Path(file_record.file_path)
    
    async def _validate_file(self, file: UploadFile) -> None:
        """Validate uploaded file."""
        # Check file size
        if file.size and file.size > self.settings.max_file_size:
            raise FileUploadError(
                f"File size exceeds maximum allowed size of {self.settings.max_file_size} bytes",
                file_name=file.filename
            )
        
        # Check file type
        if file.content_type and file.content_type not in self.settings.allowed_file_types:
            raise FileUploadError(
                f"File type {file.content_type} is not allowed",
                file_name=file.filename
            )
        
        # Check filename
        if not file.filename or len(file.filename) > 255:
            raise FileUploadError(
                "Invalid filename",
                file_name=file.filename
            )
    
    async def get_file_stats(self, user_id: str) -> Dict[str, Any]:
        """Get file upload statistics for a user."""
        try:
            from sqlalchemy import func
            
            # Get total files and size
            query = select(
                func.count(FileUpload.id),
                func.sum(FileUpload.file_size)
            ).where(FileUpload.uploaded_by == user_id)
            
            result = await self.session.execute(query)
            total_files, total_size = result.first()
            
            # Get files by type
            type_query = select(
                FileUpload.mime_type,
                func.count(FileUpload.id)
            ).where(FileUpload.uploaded_by == user_id).group_by(FileUpload.mime_type)
            
            type_result = await self.session.execute(type_query)
            files_by_type = dict(type_result.all())
            
            return {
                "total_files": total_files or 0,
                "total_size": total_size or 0,
                "files_by_type": files_by_type
            }
            
        except Exception as e:
            raise DatabaseError(f"Failed to get file stats: {str(e)}")






























