"""
Document Security Service
========================

Advanced security features for document protection and encryption.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
from dataclasses import dataclass
from uuid import uuid4
import hashlib
import secrets
import json
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import os

logger = logging.getLogger(__name__)


class SecurityLevel(str, Enum):
    """Security level."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AccessType(str, Enum):
    """Access type."""
    READ = "read"
    WRITE = "write"
    EDIT = "edit"
    DELETE = "delete"
    SHARE = "share"
    EXPORT = "export"
    PRINT = "print"


class EncryptionType(str, Enum):
    """Encryption type."""
    SYMMETRIC = "symmetric"
    ASYMMETRIC = "asymmetric"
    HYBRID = "hybrid"


@dataclass
class SecurityPolicy:
    """Security policy."""
    policy_id: str
    name: str
    description: str
    security_level: SecurityLevel
    encryption_required: bool
    access_controls: List[Dict[str, Any]]
    audit_required: bool
    retention_period: Optional[int] = None  # days
    watermark_required: bool = False
    digital_signature_required: bool = False
    created_at: datetime = None


@dataclass
class DocumentAccess:
    """Document access record."""
    access_id: str
    document_id: str
    user_id: str
    access_type: AccessType
    granted_at: datetime
    expires_at: Optional[datetime] = None
    granted_by: str = None
    reason: str = None
    ip_address: str = None
    user_agent: str = None


@dataclass
class SecurityAudit:
    """Security audit record."""
    audit_id: str
    document_id: str
    user_id: str
    action: str
    timestamp: datetime
    ip_address: str
    user_agent: str
    success: bool
    details: Dict[str, Any] = None


@dataclass
class DocumentEncryption:
    """Document encryption info."""
    encryption_id: str
    document_id: str
    encryption_type: EncryptionType
    key_id: str
    encrypted_content: bytes
    iv: bytes
    created_at: datetime
    expires_at: Optional[datetime] = None


class DocumentSecurityService:
    """Document security service."""
    
    def __init__(self, master_key: Optional[str] = None):
        self.master_key = master_key or self._generate_master_key()
        self.security_policies: Dict[str, SecurityPolicy] = {}
        self.document_access: Dict[str, List[DocumentAccess]] = {}
        self.security_audits: Dict[str, List[SecurityAudit]] = {}
        self.encryption_keys: Dict[str, bytes] = {}
        self.document_encryption: Dict[str, DocumentEncryption] = {}
        self._initialize_default_policies()
    
    def _generate_master_key(self) -> str:
        """Generate master encryption key."""
        
        return Fernet.generate_key().decode()
    
    def _initialize_default_policies(self):
        """Initialize default security policies."""
        
        # Low security policy
        low_policy = SecurityPolicy(
            policy_id="low_security",
            name="Low Security",
            description="Basic security for non-sensitive documents",
            security_level=SecurityLevel.LOW,
            encryption_required=False,
            access_controls=[
                {"access_type": AccessType.READ, "requires_auth": True},
                {"access_type": AccessType.WRITE, "requires_auth": True}
            ],
            audit_required=False,
            retention_period=365,
            watermark_required=False,
            digital_signature_required=False,
            created_at=datetime.now()
        )
        
        # Medium security policy
        medium_policy = SecurityPolicy(
            policy_id="medium_security",
            name="Medium Security",
            description="Standard security for business documents",
            security_level=SecurityLevel.MEDIUM,
            encryption_required=True,
            access_controls=[
                {"access_type": AccessType.READ, "requires_auth": True, "requires_permission": True},
                {"access_type": AccessType.WRITE, "requires_auth": True, "requires_permission": True},
                {"access_type": AccessType.SHARE, "requires_auth": True, "requires_approval": True}
            ],
            audit_required=True,
            retention_period=2555,  # 7 years
            watermark_required=True,
            digital_signature_required=False,
            created_at=datetime.now()
        )
        
        # High security policy
        high_policy = SecurityPolicy(
            policy_id="high_security",
            name="High Security",
            description="Enhanced security for sensitive documents",
            security_level=SecurityLevel.HIGH,
            encryption_required=True,
            access_controls=[
                {"access_type": AccessType.READ, "requires_auth": True, "requires_permission": True, "requires_2fa": True},
                {"access_type": AccessType.WRITE, "requires_auth": True, "requires_permission": True, "requires_2fa": True},
                {"access_type": AccessType.SHARE, "requires_auth": True, "requires_approval": True, "requires_2fa": True},
                {"access_type": AccessType.EXPORT, "requires_auth": True, "requires_approval": True}
            ],
            audit_required=True,
            retention_period=3650,  # 10 years
            watermark_required=True,
            digital_signature_required=True,
            created_at=datetime.now()
        )
        
        # Critical security policy
        critical_policy = SecurityPolicy(
            policy_id="critical_security",
            name="Critical Security",
            description="Maximum security for confidential documents",
            security_level=SecurityLevel.CRITICAL,
            encryption_required=True,
            access_controls=[
                {"access_type": AccessType.READ, "requires_auth": True, "requires_permission": True, "requires_2fa": True, "requires_justification": True},
                {"access_type": AccessType.WRITE, "requires_auth": True, "requires_permission": True, "requires_2fa": True, "requires_approval": True},
                {"access_type": AccessType.SHARE, "requires_auth": True, "requires_approval": True, "requires_2fa": True, "requires_justification": True},
                {"access_type": AccessType.EXPORT, "requires_auth": True, "requires_approval": True, "requires_2fa": True},
                {"access_type": AccessType.PRINT, "requires_auth": True, "requires_approval": True}
            ],
            audit_required=True,
            retention_period=7300,  # 20 years
            watermark_required=True,
            digital_signature_required=True,
            created_at=datetime.now()
        )
        
        # Store policies
        self.security_policies[low_policy.policy_id] = low_policy
        self.security_policies[medium_policy.policy_id] = medium_policy
        self.security_policies[high_policy.policy_id] = high_policy
        self.security_policies[critical_policy.policy_id] = critical_policy
    
    async def apply_security_policy(
        self,
        document_id: str,
        policy_id: str,
        content: str,
        metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Apply security policy to document."""
        
        try:
            if policy_id not in self.security_policies:
                raise ValueError(f"Security policy {policy_id} not found")
            
            policy = self.security_policies[policy_id]
            security_result = {
                "document_id": document_id,
                "policy_id": policy_id,
                "security_level": policy.security_level.value,
                "applied_at": datetime.now().isoformat(),
                "measures": []
            }
            
            # Apply encryption if required
            if policy.encryption_required:
                encrypted_content = await self._encrypt_content(content, policy.security_level)
                security_result["measures"].append("content_encrypted")
                security_result["encryption_applied"] = True
            else:
                security_result["encryption_applied"] = False
            
            # Apply watermark if required
            if policy.watermark_required:
                watermarked_content = await self._apply_watermark(content, document_id)
                security_result["measures"].append("watermark_applied")
                security_result["watermark_applied"] = True
            else:
                security_result["watermark_applied"] = False
            
            # Apply digital signature if required
            if policy.digital_signature_required:
                signature = await self._create_digital_signature(content, document_id)
                security_result["measures"].append("digital_signature_applied")
                security_result["digital_signature"] = signature
            else:
                security_result["digital_signature"] = None
            
            # Set up access controls
            security_result["access_controls"] = policy.access_controls
            
            # Set up audit logging
            if policy.audit_required:
                security_result["measures"].append("audit_logging_enabled")
                security_result["audit_enabled"] = True
            else:
                security_result["audit_enabled"] = False
            
            # Set retention period
            if policy.retention_period:
                security_result["retention_period_days"] = policy.retention_period
                security_result["expires_at"] = (datetime.now() + timedelta(days=policy.retention_period)).isoformat()
            
            logger.info(f"Applied security policy {policy_id} to document {document_id}")
            
            return security_result
            
        except Exception as e:
            logger.error(f"Error applying security policy: {str(e)}")
            raise
    
    async def _encrypt_content(self, content: str, security_level: SecurityLevel) -> bytes:
        """Encrypt document content."""
        
        try:
            # Generate encryption key based on security level
            if security_level == SecurityLevel.LOW:
                key = Fernet.generate_key()
            elif security_level == SecurityLevel.MEDIUM:
                key = self._generate_medium_security_key()
            elif security_level == SecurityLevel.HIGH:
                key = self._generate_high_security_key()
            else:  # CRITICAL
                key = self._generate_critical_security_key()
            
            # Encrypt content
            fernet = Fernet(key)
            encrypted_content = fernet.encrypt(content.encode())
            
            # Store key (in production, use proper key management)
            key_id = str(uuid4())
            self.encryption_keys[key_id] = key
            
            return encrypted_content
            
        except Exception as e:
            logger.error(f"Error encrypting content: {str(e)}")
            raise
    
    def _generate_medium_security_key(self) -> bytes:
        """Generate medium security encryption key."""
        
        # Use PBKDF2 with higher iteration count
        password = secrets.token_bytes(32)
        salt = secrets.token_bytes(16)
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password))
        return key
    
    def _generate_high_security_key(self) -> bytes:
        """Generate high security encryption key."""
        
        # Use PBKDF2 with very high iteration count
        password = secrets.token_bytes(32)
        salt = secrets.token_bytes(16)
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=500000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password))
        return key
    
    def _generate_critical_security_key(self) -> bytes:
        """Generate critical security encryption key."""
        
        # Use PBKDF2 with maximum iteration count
        password = secrets.token_bytes(32)
        salt = secrets.token_bytes(16)
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=1000000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password))
        return key
    
    async def _apply_watermark(self, content: str, document_id: str) -> str:
        """Apply watermark to document content."""
        
        watermark = f"CONFIDENTIAL - Document ID: {document_id} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        # Add watermark as invisible text (in production, use proper watermarking)
        watermarked_content = f"{content}\n\n<!-- {watermark} -->"
        
        return watermarked_content
    
    async def _create_digital_signature(self, content: str, document_id: str) -> str:
        """Create digital signature for document."""
        
        try:
            # Generate RSA key pair
            private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048,
            )
            public_key = private_key.public_key()
            
            # Create signature
            content_hash = hashlib.sha256(content.encode()).digest()
            signature = private_key.sign(
                content_hash,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            
            # Encode signature
            signature_b64 = base64.b64encode(signature).decode()
            
            # Store public key (in production, use proper key management)
            key_id = str(uuid4())
            self.encryption_keys[key_id] = public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            )
            
            return {
                "signature": signature_b64,
                "key_id": key_id,
                "algorithm": "RSA-PSS-SHA256",
                "created_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error creating digital signature: {str(e)}")
            raise
    
    async def grant_access(
        self,
        document_id: str,
        user_id: str,
        access_type: AccessType,
        granted_by: str,
        expires_at: Optional[datetime] = None,
        reason: str = None,
        ip_address: str = None,
        user_agent: str = None
    ) -> DocumentAccess:
        """Grant access to document."""
        
        try:
            # Check if access is allowed
            if not await self._check_access_permission(document_id, user_id, access_type):
                raise PermissionError(f"Access denied for user {user_id} to document {document_id}")
            
            # Create access record
            access = DocumentAccess(
                access_id=str(uuid4()),
                document_id=document_id,
                user_id=user_id,
                access_type=access_type,
                granted_at=datetime.now(),
                expires_at=expires_at,
                granted_by=granted_by,
                reason=reason,
                ip_address=ip_address,
                user_agent=user_agent
            )
            
            # Store access record
            if document_id not in self.document_access:
                self.document_access[document_id] = []
            self.document_access[document_id].append(access)
            
            # Log audit event
            await self._log_audit_event(
                document_id=document_id,
                user_id=user_id,
                action=f"access_granted_{access_type.value}",
                success=True,
                details={"granted_by": granted_by, "reason": reason},
                ip_address=ip_address,
                user_agent=user_agent
            )
            
            logger.info(f"Granted {access_type.value} access to user {user_id} for document {document_id}")
            
            return access
            
        except Exception as e:
            logger.error(f"Error granting access: {str(e)}")
            raise
    
    async def revoke_access(
        self,
        document_id: str,
        user_id: str,
        access_type: AccessType,
        revoked_by: str,
        reason: str = None,
        ip_address: str = None,
        user_agent: str = None
    ) -> bool:
        """Revoke access to document."""
        
        try:
            if document_id not in self.document_access:
                return False
            
            # Find and remove access record
            access_records = self.document_access[document_id]
            for i, access in enumerate(access_records):
                if (access.user_id == user_id and 
                    access.access_type == access_type and 
                    (access.expires_at is None or access.expires_at > datetime.now())):
                    
                    # Mark as revoked
                    access.expires_at = datetime.now()
                    
                    # Log audit event
                    await self._log_audit_event(
                        document_id=document_id,
                        user_id=user_id,
                        action=f"access_revoked_{access_type.value}",
                        success=True,
                        details={"revoked_by": revoked_by, "reason": reason},
                        ip_address=ip_address,
                        user_agent=user_agent
                    )
                    
                    logger.info(f"Revoked {access_type.value} access for user {user_id} from document {document_id}")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error revoking access: {str(e)}")
            raise
    
    async def check_access(
        self,
        document_id: str,
        user_id: str,
        access_type: AccessType,
        ip_address: str = None,
        user_agent: str = None
    ) -> bool:
        """Check if user has access to document."""
        
        try:
            # Check access records
            if document_id not in self.document_access:
                await self._log_audit_event(
                    document_id=document_id,
                    user_id=user_id,
                    action=f"access_check_{access_type.value}",
                    success=False,
                    details={"reason": "no_access_records"},
                    ip_address=ip_address,
                    user_agent=user_agent
                )
                return False
            
            access_records = self.document_access[document_id]
            for access in access_records:
                if (access.user_id == user_id and 
                    access.access_type == access_type and 
                    (access.expires_at is None or access.expires_at > datetime.now())):
                    
                    # Log successful access check
                    await self._log_audit_event(
                        document_id=document_id,
                        user_id=user_id,
                        action=f"access_check_{access_type.value}",
                        success=True,
                        details={"access_id": access.access_id},
                        ip_address=ip_address,
                        user_agent=user_agent
                    )
                    return True
            
            # Log failed access check
            await self._log_audit_event(
                document_id=document_id,
                user_id=user_id,
                action=f"access_check_{access_type.value}",
                success=False,
                details={"reason": "access_denied"},
                ip_address=ip_address,
                user_agent=user_agent
            )
            return False
            
        except Exception as e:
            logger.error(f"Error checking access: {str(e)}")
            return False
    
    async def _check_access_permission(
        self,
        document_id: str,
        user_id: str,
        access_type: AccessType
    ) -> bool:
        """Check if access permission is allowed by policy."""
        
        # This would typically check against security policies
        # For now, return True (in production, implement proper permission checking)
        return True
    
    async def _log_audit_event(
        self,
        document_id: str,
        user_id: str,
        action: str,
        success: bool,
        details: Dict[str, Any] = None,
        ip_address: str = None,
        user_agent: str = None
    ):
        """Log security audit event."""
        
        audit = SecurityAudit(
            audit_id=str(uuid4()),
            document_id=document_id,
            user_id=user_id,
            action=action,
            timestamp=datetime.now(),
            ip_address=ip_address or "unknown",
            user_agent=user_agent or "unknown",
            success=success,
            details=details or {}
        )
        
        if document_id not in self.security_audits:
            self.security_audits[document_id] = []
        self.security_audits[document_id].append(audit)
    
    async def get_access_log(
        self,
        document_id: str,
        limit: int = 100,
        offset: int = 0
    ) -> List[SecurityAudit]:
        """Get access log for document."""
        
        if document_id not in self.security_audits:
            return []
        
        audits = self.security_audits[document_id]
        audits.sort(key=lambda x: x.timestamp, reverse=True)
        
        return audits[offset:offset + limit]
    
    async def get_user_access_history(
        self,
        user_id: str,
        limit: int = 100,
        offset: int = 0
    ) -> List[SecurityAudit]:
        """Get user access history."""
        
        all_audits = []
        for document_audits in self.security_audits.values():
            for audit in document_audits:
                if audit.user_id == user_id:
                    all_audits.append(audit)
        
        all_audits.sort(key=lambda x: x.timestamp, reverse=True)
        return all_audits[offset:offset + limit]
    
    async def verify_digital_signature(
        self,
        content: str,
        signature_data: Dict[str, Any]
    ) -> bool:
        """Verify digital signature."""
        
        try:
            # Get public key
            key_id = signature_data["key_id"]
            if key_id not in self.encryption_keys:
                return False
            
            public_key_pem = self.encryption_keys[key_id]
            public_key = serialization.load_pem_public_key(public_key_pem)
            
            # Decode signature
            signature = base64.b64decode(signature_data["signature"])
            
            # Verify signature
            content_hash = hashlib.sha256(content.encode()).digest()
            public_key.verify(
                signature,
                content_hash,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Error verifying digital signature: {str(e)}")
            return False
    
    async def decrypt_content(
        self,
        encrypted_content: bytes,
        key_id: str
    ) -> str:
        """Decrypt document content."""
        
        try:
            if key_id not in self.encryption_keys:
                raise ValueError(f"Encryption key {key_id} not found")
            
            key = self.encryption_keys[key_id]
            fernet = Fernet(key)
            decrypted_content = fernet.decrypt(encrypted_content)
            
            return decrypted_content.decode()
            
        except Exception as e:
            logger.error(f"Error decrypting content: {str(e)}")
            raise
    
    async def get_security_summary(self, document_id: str) -> Dict[str, Any]:
        """Get security summary for document."""
        
        summary = {
            "document_id": document_id,
            "access_records": len(self.document_access.get(document_id, [])),
            "audit_events": len(self.security_audits.get(document_id, [])),
            "is_encrypted": document_id in self.document_encryption,
            "has_digital_signature": False,
            "active_access": 0,
            "last_access": None
        }
        
        # Count active access
        if document_id in self.document_access:
            active_access = 0
            last_access = None
            
            for access in self.document_access[document_id]:
                if access.expires_at is None or access.expires_at > datetime.now():
                    active_access += 1
                
                if last_access is None or access.granted_at > last_access:
                    last_access = access.granted_at
            
            summary["active_access"] = active_access
            summary["last_access"] = last_access.isoformat() if last_access else None
        
        # Check for digital signature
        if document_id in self.security_audits:
            for audit in self.security_audits[document_id]:
                if "digital_signature" in audit.action:
                    summary["has_digital_signature"] = True
                    break
        
        return summary
    
    async def create_security_policy(
        self,
        name: str,
        description: str,
        security_level: SecurityLevel,
        encryption_required: bool = False,
        access_controls: List[Dict[str, Any]] = None,
        audit_required: bool = False,
        retention_period: Optional[int] = None,
        watermark_required: bool = False,
        digital_signature_required: bool = False
    ) -> SecurityPolicy:
        """Create custom security policy."""
        
        policy = SecurityPolicy(
            policy_id=str(uuid4()),
            name=name,
            description=description,
            security_level=security_level,
            encryption_required=encryption_required,
            access_controls=access_controls or [],
            audit_required=audit_required,
            retention_period=retention_period,
            watermark_required=watermark_required,
            digital_signature_required=digital_signature_required,
            created_at=datetime.now()
        )
        
        self.security_policies[policy.policy_id] = policy
        
        logger.info(f"Created security policy: {policy.name}")
        
        return policy
    
    async def get_security_policies(self) -> List[SecurityPolicy]:
        """Get all security policies."""
        
        return list(self.security_policies.values())
    
    async def get_security_analytics(self) -> Dict[str, Any]:
        """Get security analytics."""
        
        total_documents = len(self.document_access)
        total_access_records = sum(len(access_list) for access_list in self.document_access.values())
        total_audit_events = sum(len(audit_list) for audit_list in self.security_audits.values())
        
        # Count by security level
        security_levels = {}
        for policy in self.security_policies.values():
            security_levels[policy.security_level.value] = security_levels.get(policy.security_level.value, 0) + 1
        
        # Count access types
        access_types = {}
        for access_list in self.document_access.values():
            for access in access_list:
                access_types[access.access_type.value] = access_types.get(access.access_type.value, 0) + 1
        
        return {
            "total_documents": total_documents,
            "total_access_records": total_access_records,
            "total_audit_events": total_audit_events,
            "security_levels": security_levels,
            "access_types": access_types,
            "policies_count": len(self.security_policies),
            "encrypted_documents": len(self.document_encryption)
        }



























