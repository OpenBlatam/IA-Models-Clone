"""
Embedding Handlers
==================

Request handlers for embedding-related endpoints.
"""

import logging
from typing import List, Optional, Dict, Any
from fastapi import UploadFile, HTTPException
from fastapi.responses import JSONResponse

from ..utils.image_utils import process_uploaded_images
from ..utils.metadata_utils import parse_metadata
from ..utils.error_handlers import handle_api_error

logger = logging.getLogger(__name__)


class EmbeddingHandlers:
    """Handlers for embedding-related operations."""
    
    def __init__(self, service):
        """
        Initialize handlers.
        
        Args:
            service: CharacterConsistencyService instance
        """
        self.service = service
    
    async def generate_embedding(
        self,
        images: List[UploadFile],
        character_name: Optional[str],
        save_tensor: bool,
        metadata: Optional[str],
    ) -> JSONResponse:
        """
        Handle generate embedding request.
        
        Args:
            images: Uploaded image files
            character_name: Optional character name
            save_tensor: Whether to save tensor
            metadata: Optional JSON metadata string
            
        Returns:
            JSONResponse with embedding info
        """
        try:
            parsed_metadata = parse_metadata(metadata)
            image_list = await process_uploaded_images(images)
            
            result = self.service.generate_character_embedding(
                images=image_list,
                character_name=character_name,
                metadata=parsed_metadata,
                save_tensor=save_tensor,
            )
            
            return JSONResponse(content=result)
        
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise handle_api_error("generate_embedding", e)
    
    async def list_embeddings(self) -> JSONResponse:
        """
        Handle list embeddings request.
        
        Returns:
            JSONResponse with list of embeddings
        """
        try:
            embeddings = self.service.list_embeddings()
            return JSONResponse(content=embeddings)
        except Exception as e:
            raise handle_api_error("list_embeddings", e)
    
    async def get_embedding(self, embedding_id: str):
        """
        Handle get embedding request.
        
        Args:
            embedding_id: Embedding ID or filename
            
        Returns:
            FileResponse with safe tensor file
        """
        try:
            from fastapi.responses import FileResponse
            
            embedding = self._find_embedding(embedding_id)
            if embedding is None:
                raise HTTPException(status_code=404, detail="Embedding not found")
            
            return FileResponse(
                path=embedding["path"],
                filename=embedding["filename"],
                media_type="application/octet-stream",
            )
        
        except HTTPException:
            raise
        except Exception as e:
            raise handle_api_error("get_embedding", e)
    
    def _find_embedding(self, embedding_id: str) -> Optional[Dict[str, Any]]:
        """Find embedding by ID or filename."""
        embeddings = self.service.list_embeddings()
        return next(
            (emb for emb in embeddings 
             if emb["filename"] == embedding_id or embedding_id in emb["path"]),
            None
        )

