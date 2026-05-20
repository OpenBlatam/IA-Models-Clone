"""Digital signature utilities"""
from typing import Dict, Any, Optional
from pathlib import Path
import hashlib
import base64
from datetime import datetime
import json
import logging

logger = logging.getLogger(__name__)


class DigitalSignatureManager:
    """Manage digital signatures for documents"""
    
    def __init__(self, keys_dir: Optional[str] = None):
        """
        Initialize signature manager
        
        Args:
            keys_dir: Directory for storing keys
        """
        if keys_dir is None:
            from config import settings
            keys_dir = settings.temp_dir + "/signatures"
        
        self.keys_dir = Path(keys_dir)
        self.keys_dir.mkdir(parents=True, exist_ok=True)
    
    def sign_document(
        self,
        document_path: str,
        signer_name: str,
        private_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Sign a document
        
        Args:
            document_path: Path to document
            signer_name: Name of signer
            private_key: Optional private key (uses default if None)
            
        Returns:
            Signature information
        """
        try:
            path = Path(document_path)
            if not path.exists():
                raise FileNotFoundError(f"Document not found: {document_path}")
            
            # Calculate document hash
            doc_hash = self._calculate_document_hash(path)
            
            # Create signature
            signature_data = {
                "document_path": str(document_path),
                "document_hash": doc_hash,
                "signer": signer_name,
                "signed_at": datetime.now().isoformat(),
                "signature": self._create_signature(doc_hash, signer_name, private_key)
            }
            
            # Save signature
            signature_file = self.keys_dir / f"{path.stem}_signature.json"
            with open(signature_file, 'w') as f:
                json.dump(signature_data, f, indent=2)
            
            return signature_data
        except Exception as e:
            logger.error(f"Error signing document: {e}")
            raise
    
    def verify_signature(
        self,
        document_path: str,
        signature_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Verify document signature
        
        Args:
            document_path: Path to document
            signature_data: Optional signature data (loads from file if None)
            
        Returns:
            Verification result
        """
        try:
            path = Path(document_path)
            if not path.exists():
                return {
                    "valid": False,
                    "error": "Document not found"
                }
            
            # Load signature if not provided
            if signature_data is None:
                signature_file = self.keys_dir / f"{path.stem}_signature.json"
                if signature_file.exists():
                    with open(signature_file, 'r') as f:
                        signature_data = json.load(f)
                else:
                    return {
                        "valid": False,
                        "error": "No signature found"
                    }
            
            # Calculate current hash
            current_hash = self._calculate_document_hash(path)
            
            # Verify hash matches
            if current_hash != signature_data.get("document_hash"):
                return {
                    "valid": False,
                    "error": "Document has been modified",
                    "original_hash": signature_data.get("document_hash"),
                    "current_hash": current_hash
                }
            
            # Verify signature
            expected_signature = self._create_signature(
                current_hash,
                signature_data.get("signer", ""),
                None
            )
            
            signature_valid = expected_signature == signature_data.get("signature")
            
            return {
                "valid": signature_valid,
                "signer": signature_data.get("signer"),
                "signed_at": signature_data.get("signed_at"),
                "document_hash": current_hash,
                "signature_match": signature_valid
            }
        except Exception as e:
            logger.error(f"Error verifying signature: {e}")
            return {
                "valid": False,
                "error": str(e)
            }
    
    def _calculate_document_hash(self, path: Path) -> str:
        """Calculate SHA256 hash of document"""
        sha256 = hashlib.sha256()
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256.update(chunk)
        return sha256.hexdigest()
    
    def _create_signature(
        self,
        document_hash: str,
        signer_name: str,
        private_key: Optional[str]
    ) -> str:
        """
        Create signature for document
        
        Args:
            document_hash: Document hash
            signer_name: Signer name
            private_key: Optional private key
            
        Returns:
            Signature string
        """
        # Simple signature (in production, would use proper cryptographic signing)
        signature_string = f"{document_hash}:{signer_name}:{datetime.now().isoformat()}"
        
        if private_key:
            # Use private key for signing
            import hmac
            signature = hmac.new(
                private_key.encode(),
                signature_string.encode(),
                hashlib.sha256
            ).hexdigest()
        else:
            # Simple hash-based signature
            signature = hashlib.sha256(signature_string.encode()).hexdigest()
        
        return base64.b64encode(signature.encode()).decode()


# Global signature manager
_signature_manager: Optional[DigitalSignatureManager] = None


def get_signature_manager() -> DigitalSignatureManager:
    """Get global signature manager"""
    global _signature_manager
    if _signature_manager is None:
        _signature_manager = DigitalSignatureManager()
    return _signature_manager

