"""
Blockchain integration service for content verification and NFT functionality
"""

import asyncio
import hashlib
import json
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, desc
from web3 import Web3
from eth_account import Account
import ipfshttpclient

from ..models.database import BlogPost, User, ContentHash
from ..core.exceptions import DatabaseError, ExternalServiceError


class BlockchainService:
    """Service for blockchain integration and content verification."""
    
    def __init__(self, session: AsyncSession):
        self.session = session
        self.web3 = None
        self.ipfs_client = None
        self.contract_address = None
        self.private_key = None
        
        # Initialize blockchain connections
        self._initialize_blockchain()
    
    def _initialize_blockchain(self):
        """Initialize blockchain and IPFS connections."""
        try:
            # Initialize Web3 connection (Ethereum)
            # In production, you would use actual RPC URLs
            self.web3 = Web3(Web3.HTTPProvider('https://mainnet.infura.io/v3/YOUR_PROJECT_ID'))
            
            # Initialize IPFS client
            self.ipfs_client = ipfshttpclient.connect('/ip4/127.0.0.1/tcp/5001/http')
            
            # Load contract ABI and address
            self.contract_address = "0x..."  # Your contract address
            self.private_key = "0x..."  # Your private key (should be in environment)
            
        except Exception as e:
            print(f"Warning: Could not initialize blockchain services: {e}")
    
    async def create_content_hash(
        self,
        post_id: int,
        content: str,
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create a hash of content for blockchain verification."""
        try:
            # Create content hash
            content_hash = hashlib.sha256(content.encode()).hexdigest()
            
            # Create metadata hash
            metadata_json = json.dumps(metadata, sort_keys=True)
            metadata_hash = hashlib.sha256(metadata_json.encode()).hexdigest()
            
            # Create combined hash
            combined_data = f"{content_hash}{metadata_hash}{post_id}"
            combined_hash = hashlib.sha256(combined_data.encode()).hexdigest()
            
            # Store hash in database
            content_hash_record = ContentHash(
                post_id=post_id,
                content_hash=content_hash,
                metadata_hash=metadata_hash,
                combined_hash=combined_hash,
                blockchain_tx_hash=None,
                ipfs_hash=None,
                created_at=datetime.utcnow()
            )
            
            self.session.add(content_hash_record)
            await self.session.commit()
            
            return {
                "content_hash": content_hash,
                "metadata_hash": metadata_hash,
                "combined_hash": combined_hash,
                "post_id": post_id,
                "created_at": content_hash_record.created_at
            }
            
        except Exception as e:
            await self.session.rollback()
            raise DatabaseError(f"Failed to create content hash: {str(e)}")
    
    async def store_content_on_ipfs(
        self,
        post_id: int,
        content: str,
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Store content on IPFS for decentralized storage."""
        try:
            if not self.ipfs_client:
                raise ExternalServiceError("IPFS not available", service_name="ipfs")
            
            # Prepare content for IPFS
            ipfs_content = {
                "post_id": post_id,
                "content": content,
                "metadata": metadata,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Add content to IPFS
            ipfs_result = await asyncio.to_thread(
                self.ipfs_client.add,
                json.dumps(ipfs_content)
            )
            
            ipfs_hash = ipfs_result['Hash']
            
            # Update content hash record
            content_hash_query = select(ContentHash).where(ContentHash.post_id == post_id)
            content_hash_result = await self.session.execute(content_hash_query)
            content_hash_record = content_hash_result.scalar_one_or_none()
            
            if content_hash_record:
                content_hash_record.ipfs_hash = ipfs_hash
                await self.session.commit()
            
            return {
                "ipfs_hash": ipfs_hash,
                "post_id": post_id,
                "ipfs_url": f"https://ipfs.io/ipfs/{ipfs_hash}",
                "success": True
            }
            
        except Exception as e:
            raise ExternalServiceError(f"Failed to store content on IPFS: {str(e)}", service_name="ipfs")
    
    async def create_nft_for_post(
        self,
        post_id: int,
        author_id: str,
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create an NFT for a blog post."""
        try:
            if not self.web3:
                raise ExternalServiceError("Blockchain not available", service_name="blockchain")
            
            # Get content hash record
            content_hash_query = select(ContentHash).where(ContentHash.post_id == post_id)
            content_hash_result = await self.session.execute(content_hash_query)
            content_hash_record = content_hash_result.scalar_one_or_none()
            
            if not content_hash_record:
                raise ValidationError("Content hash not found")
            
            # Prepare NFT metadata
            nft_metadata = {
                "name": metadata.get("title", f"Blog Post #{post_id}"),
                "description": metadata.get("excerpt", ""),
                "image": metadata.get("featured_image_url", ""),
                "attributes": [
                    {"trait_type": "Author", "value": author_id},
                    {"trait_type": "Content Hash", "value": content_hash_record.content_hash},
                    {"trait_type": "IPFS Hash", "value": content_hash_record.ipfs_hash or ""},
                    {"trait_type": "Created", "value": content_hash_record.created_at.isoformat()}
                ]
            }
            
            # Store NFT metadata on IPFS
            ipfs_result = await self.store_content_on_ipfs(post_id, "", nft_metadata)
            metadata_uri = ipfs_result["ipfs_url"]
            
            # Create NFT transaction (mock implementation)
            # In production, you would interact with a real NFT contract
            tx_hash = f"0x{hashlib.sha256(f'{post_id}{author_id}{metadata_uri}'.encode()).hexdigest()}"
            
            # Update content hash record with transaction hash
            content_hash_record.blockchain_tx_hash = tx_hash
            await self.session.commit()
            
            return {
                "nft_id": post_id,
                "tx_hash": tx_hash,
                "metadata_uri": metadata_uri,
                "author_id": author_id,
                "success": True
            }
            
        except Exception as e:
            raise ExternalServiceError(f"Failed to create NFT: {str(e)}", service_name="blockchain")
    
    async def verify_content_integrity(
        self,
        post_id: int,
        content: str
    ) -> Dict[str, Any]:
        """Verify content integrity using blockchain hash."""
        try:
            # Get stored content hash
            content_hash_query = select(ContentHash).where(ContentHash.post_id == post_id)
            content_hash_result = await self.session.execute(content_hash_query)
            content_hash_record = content_hash_result.scalar_one_or_none()
            
            if not content_hash_record:
                return {
                    "verified": False,
                    "error": "No content hash found"
                }
            
            # Calculate current content hash
            current_hash = hashlib.sha256(content.encode()).hexdigest()
            stored_hash = content_hash_record.content_hash
            
            # Verify integrity
            is_verified = current_hash == stored_hash
            
            return {
                "verified": is_verified,
                "current_hash": current_hash,
                "stored_hash": stored_hash,
                "post_id": post_id,
                "blockchain_tx_hash": content_hash_record.blockchain_tx_hash,
                "ipfs_hash": content_hash_record.ipfs_hash
            }
            
        except Exception as e:
            raise DatabaseError(f"Failed to verify content integrity: {str(e)}")
    
    async def get_blockchain_stats(self) -> Dict[str, Any]:
        """Get blockchain integration statistics."""
        try:
            # Get total content hashes
            total_hashes_query = select(func.count(ContentHash.id))
            total_hashes_result = await self.session.execute(total_hashes_query)
            total_hashes = total_hashes_result.scalar()
            
            # Get hashes with blockchain transactions
            blockchain_hashes_query = select(func.count(ContentHash.id)).where(
                ContentHash.blockchain_tx_hash.isnot(None)
            )
            blockchain_hashes_result = await self.session.execute(blockchain_hashes_query)
            blockchain_hashes = blockchain_hashes_result.scalar()
            
            # Get hashes with IPFS storage
            ipfs_hashes_query = select(func.count(ContentHash.id)).where(
                ContentHash.ipfs_hash.isnot(None)
            )
            ipfs_hashes_result = await self.session.execute(ipfs_hashes_query)
            ipfs_hashes = ipfs_hashes_result.scalar()
            
            return {
                "total_content_hashes": total_hashes,
                "blockchain_verified": blockchain_hashes,
                "ipfs_stored": ipfs_hashes,
                "blockchain_percentage": (blockchain_hashes / max(total_hashes, 1)) * 100,
                "ipfs_percentage": (ipfs_hashes / max(total_hashes, 1)) * 100
            }
            
        except Exception as e:
            raise DatabaseError(f"Failed to get blockchain stats: {str(e)}")
    
    async def get_nft_collection(
        self,
        author_id: Optional[str] = None,
        limit: int = 20,
        offset: int = 0
    ) -> Tuple[List[Dict[str, Any]], int]:
        """Get NFT collection for posts."""
        try:
            # Build query
            query = select(ContentHash).join(BlogPost, ContentHash.post_id == BlogPost.id).where(
                ContentHash.blockchain_tx_hash.isnot(None)
            )
            
            if author_id:
                query = query.where(BlogPost.author_id == author_id)
            
            # Get total count
            count_query = select(func.count(ContentHash.id))
            if author_id:
                count_query = count_query.join(BlogPost, ContentHash.post_id == BlogPost.id).where(
                    and_(
                        ContentHash.blockchain_tx_hash.isnot(None),
                        BlogPost.author_id == author_id
                    )
                )
            else:
                count_query = count_query.where(ContentHash.blockchain_tx_hash.isnot(None))
            
            total_result = await self.session.execute(count_query)
            total = total_result.scalar()
            
            # Get NFTs
            query = query.order_by(desc(ContentHash.created_at)).offset(offset).limit(limit)
            nfts_result = await self.session.execute(query)
            nfts = nfts_result.scalars().all()
            
            # Format NFTs
            nft_list = []
            for nft in nfts:
                # Get post details
                post_query = select(BlogPost).where(BlogPost.id == nft.post_id)
                post_result = await self.session.execute(post_query)
                post = post_result.scalar_one_or_none()
                
                if post:
                    nft_list.append({
                        "nft_id": nft.post_id,
                        "title": post.title,
                        "author_id": str(post.author_id),
                        "tx_hash": nft.blockchain_tx_hash,
                        "ipfs_hash": nft.ipfs_hash,
                        "content_hash": nft.content_hash,
                        "created_at": nft.created_at,
                        "post_slug": post.slug
                    })
            
            return nft_list, total
            
        except Exception as e:
            raise DatabaseError(f"Failed to get NFT collection: {str(e)}")
    
    async def transfer_nft_ownership(
        self,
        post_id: int,
        from_address: str,
        to_address: str
    ) -> Dict[str, Any]:
        """Transfer NFT ownership (mock implementation)."""
        try:
            if not self.web3:
                raise ExternalServiceError("Blockchain not available", service_name="blockchain")
            
            # Get content hash record
            content_hash_query = select(ContentHash).where(ContentHash.post_id == post_id)
            content_hash_result = await self.session.execute(content_hash_query)
            content_hash_record = content_hash_result.scalar_one_or_none()
            
            if not content_hash_record or not content_hash_record.blockchain_tx_hash:
                raise ValidationError("NFT not found")
            
            # Create transfer transaction (mock)
            transfer_tx_hash = f"0x{hashlib.sha256(f'{post_id}{from_address}{to_address}'.encode()).hexdigest()}"
            
            return {
                "post_id": post_id,
                "from_address": from_address,
                "to_address": to_address,
                "transfer_tx_hash": transfer_tx_hash,
                "success": True
            }
            
        except Exception as e:
            raise ExternalServiceError(f"Failed to transfer NFT: {str(e)}", service_name="blockchain")
    
    async def get_blockchain_health(self) -> Dict[str, Any]:
        """Get blockchain service health status."""
        try:
            health_status = {
                "web3_connected": self.web3 is not None and self.web3.is_connected(),
                "ipfs_connected": self.ipfs_client is not None,
                "contract_configured": self.contract_address is not None,
                "private_key_configured": self.private_key is not None
            }
            
            # Test Web3 connection
            if self.web3:
                try:
                    latest_block = self.web3.eth.block_number
                    health_status["latest_block"] = latest_block
                except Exception:
                    health_status["web3_connected"] = False
            
            # Test IPFS connection
            if self.ipfs_client:
                try:
                    await asyncio.to_thread(self.ipfs_client.id)
                    health_status["ipfs_connected"] = True
                except Exception:
                    health_status["ipfs_connected"] = False
            
            overall_health = all([
                health_status["web3_connected"],
                health_status["ipfs_connected"],
                health_status["contract_configured"]
            ])
            
            return {
                "healthy": overall_health,
                "services": health_status,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "healthy": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }






























