"""
Vector Store
===========

Advanced vector store for embeddings and similarity search.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

logger = logging.getLogger(__name__)

class VectorStore:
    """
    Advanced vector store with ChromaDB backend.
    
    Features:
    - Embedding generation
    - Similarity search
    - Vector indexing
    - Metadata filtering
    - Batch operations
    """
    
    def __init__(self, collection_name: str = "bulk_truthgpt"):
        self.collection_name = collection_name
        self.client = None
        self.collection = None
        self.embedder = None
        
    async def initialize(self):
        """Initialize vector store."""
        logger.info("Initializing Vector Store...")
        
        try:
            # Initialize ChromaDB client
            self.client = chromadb.Client(Settings(
                persist_directory="./chroma_db",
                anonymized_telemetry=False
            ))
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "Bulk TruthGPT knowledge base"}
            )
            
            # Initialize sentence transformer
            self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
            
            logger.info("Vector Store initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Vector Store: {str(e)}")
            raise
    
    async def add_document(
        self, 
        content: str, 
        metadata: Optional[Dict[str, Any]] = None,
        document_id: Optional[str] = None
    ) -> str:
        """Add document to vector store."""
        try:
            # Generate embedding
            embedding = self.embedder.encode(content).tolist()
            
            # Generate ID if not provided
            if not document_id:
                import uuid
                document_id = str(uuid.uuid4())
            
            # Add to collection
            self.collection.add(
                documents=[content],
                embeddings=[embedding],
                metadatas=[metadata or {}],
                ids=[document_id]
            )
            
            logger.info(f"Added document {document_id} to vector store")
            return document_id
            
        except Exception as e:
            logger.error(f"Failed to add document: {str(e)}")
            raise
    
    async def similarity_search(
        self, 
        query: str, 
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar documents."""
        try:
            # Generate query embedding
            query_embedding = self.embedder.encode(query).tolist()
            
            # Search collection
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=filter_metadata
            )
            
            # Format results
            documents = []
            if results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    documents.append({
                        'content': doc,
                        'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                        'distance': results['distances'][0][i] if results['distances'] else 0.0,
                        'id': results['ids'][0][i] if results['ids'] else None
                    })
            
            return documents
            
        except Exception as e:
            logger.error(f"Failed to search documents: {str(e)}")
            return []
    
    async def get_all_embeddings(self) -> List[Dict[str, Any]]:
        """Get all embeddings from the collection."""
        try:
            # Get all documents
            results = self.collection.get()
            
            embeddings = []
            if results['documents']:
                for i, doc in enumerate(results['documents']):
                    embeddings.append({
                        'id': results['ids'][i],
                        'content': doc,
                        'metadata': results['metadatas'][i] if results['metadatas'] else {},
                        'embedding': results['embeddings'][i] if results['embeddings'] else None
                    })
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to get all embeddings: {str(e)}")
            return []
    
    async def delete_document(self, document_id: str) -> bool:
        """Delete document from vector store."""
        try:
            self.collection.delete(ids=[document_id])
            logger.info(f"Deleted document {document_id} from vector store")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete document {document_id}: {str(e)}")
            return False
    
    async def update_document(
        self, 
        document_id: str, 
        content: str, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Update document in vector store."""
        try:
            # Generate new embedding
            embedding = self.embedder.encode(content).tolist()
            
            # Update document
            self.collection.update(
                ids=[document_id],
                documents=[content],
                embeddings=[embedding],
                metadatas=[metadata or {}]
            )
            
            logger.info(f"Updated document {document_id} in vector store")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update document {document_id}: {str(e)}")
            return False
    
    async def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get document by ID."""
        try:
            results = self.collection.get(ids=[document_id])
            
            if results['documents'] and results['documents'][0]:
                return {
                    'id': results['ids'][0],
                    'content': results['documents'][0],
                    'metadata': results['metadatas'][0] if results['metadatas'] else {},
                    'embedding': results['embeddings'][0] if results['embeddings'] else None
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get document {document_id}: {str(e)}")
            return None
    
    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        try:
            count = self.collection.count()
            
            return {
                'total_documents': count,
                'collection_name': self.collection_name,
                'embedder_model': self.embedder.get_sentence_embedding_dimension() if self.embedder else 0
            }
            
        except Exception as e:
            logger.error(f"Failed to get collection stats: {str(e)}")
            return {}
    
    async def cleanup(self):
        """Cleanup vector store resources."""
        try:
            if self.client:
                self.client = None
            
            logger.info("Vector Store cleaned up successfully")
            
        except Exception as e:
            logger.error(f"Failed to cleanup Vector Store: {str(e)}")











