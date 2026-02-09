from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, field_validator
import structlog
import socket
import ssl
import OpenSSL
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
            from dateutil import parser
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
SSL certificate utilities for cybersecurity tools.
"""

logger = structlog.get_logger(__name__)

class CertificateStatus(str, Enum):
    """Certificate status enumeration."""
    VALID = "valid"
    EXPIRED = "expired"
    NOT_YET_VALID = "not_yet_valid"
    REVOKED = "revoked"
    UNKNOWN = "unknown"

@dataclass
class CertificateInfo:
    """SSL certificate information."""
    subject: Dict[str, str]
    issuer: Dict[str, str]
    serial_number: str
    not_before: datetime
    not_after: datetime
    version: int
    signature_algorithm: str
    public_key_algorithm: str
    public_key_size: int
    san_domains: List[str]
    status: CertificateStatus

class SSLCertificateInput(BaseModel):
    """Input model for SSL certificate retrieval."""
    host: str
    port: int = 443
    timeout: float = 10.0
    verify_ssl: bool = True
    check_revocation: bool = False
    
    @field_validator('host')
    def validate_host(cls, v) -> bool:
        if not v:
            raise ValueError("Host cannot be empty")
        return v
    
    @field_validator('port')
    def validate_port(cls, v) -> bool:
        if v < 1 or v > 65535:
            raise ValueError("Port must be between 1 and 65535")
        return v
    
    @field_validator('timeout')
    def validate_timeout(cls, v) -> bool:
        if v <= 0:
            raise ValueError("Timeout must be positive")
        return v

class SSLCertificateResult(BaseModel):
    """Result model for SSL certificate operations."""
    host: str
    port: int
    certificate_info: Optional[Dict[str, Any]] = None
    ssl_version: Optional[str] = None
    cipher_suite: Optional[str] = None
    is_ssl_enabled: bool
    certificate_status: CertificateStatus
    is_successful: bool
    error_message: Optional[str] = None

def get_ssl_certificate(input_data: SSLCertificateInput) -> SSLCertificateResult:
    """
    RORO: Receive SSLCertificateInput, return SSLCertificateResult
    
    Get SSL certificate information from a host.
    """
    try:
        # Create SSL context
        context = ssl.create_default_context()
        
        if not input_data.verify_ssl:
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE
        
        # Create socket and wrap with SSL
        with socket.create_connection((input_data.host, input_data.port), timeout=input_data.timeout) as sock:
            with context.wrap_socket(sock, server_hostname=input_data.host) as ssl_sock:
                # Get certificate
                cert = ssl_sock.getpeercert()
                
                # Get SSL info
                ssl_version = ssl_sock.version()
                cipher_suite = ssl_sock.cipher()[0]
                
                # Parse certificate information
                certificate_info = parse_certificate_info(cert)
                
                # Check certificate status
                certificate_status = check_certificate_status(certificate_info)
                
                return SSLCertificateResult(
                    host=input_data.host,
                    port=input_data.port,
                    certificate_info=certificate_info,
                    ssl_version=ssl_version,
                    cipher_suite=cipher_suite,
                    is_ssl_enabled=True,
                    certificate_status=certificate_status,
                    is_successful=True
                )
        
    except ssl.SSLError as e:
        logger.warning(f"SSL error for {input_data.host}:{input_data.port}", error=str(e))
        return SSLCertificateResult(
            host=input_data.host,
            port=input_data.port,
            is_ssl_enabled=False,
            certificate_status=CertificateStatus.UNKNOWN,
            is_successful=False,
            error_message=f"SSL error: {str(e)}"
        )
    except socket.timeout:
        logger.warning(f"Timeout connecting to {input_data.host}:{input_data.port}")
        return SSLCertificateResult(
            host=input_data.host,
            port=input_data.port,
            is_ssl_enabled=False,
            certificate_status=CertificateStatus.UNKNOWN,
            is_successful=False,
            error_message="Connection timeout"
        )
    except Exception as e:
        logger.error(f"SSL certificate retrieval failed for {input_data.host}:{input_data.port}", error=str(e))
        return SSLCertificateResult(
            host=input_data.host,
            port=input_data.port,
            is_ssl_enabled=False,
            certificate_status=CertificateStatus.UNKNOWN,
            is_successful=False,
            error_message=str(e)
        )

def parse_certificate_info(cert: Dict[str, Any]) -> Dict[str, Any]:
    """Parse certificate information from SSL certificate."""
    try:
        info = {}
        
        # Subject information
        if 'subject' in cert:
            subject = {}
            for item in cert['subject']:
                for key, value in item:
                    subject[key] = value
            info['subject'] = subject
        
        # Issuer information
        if 'issuer' in cert:
            issuer = {}
            for item in cert['issuer']:
                for key, value in item:
                    issuer[key] = value
            info['issuer'] = issuer
        
        # Validity dates
        if 'notBefore' in cert:
            info['not_before'] = parse_cert_date(cert['notBefore'])
        
        if 'notAfter' in cert:
            info['not_after'] = parse_cert_date(cert['notAfter'])
        
        # Serial number
        if 'serialNumber' in cert:
            info['serial_number'] = cert['serialNumber']
        
        # Version
        if 'version' in cert:
            info['version'] = cert['version']
        
        # Signature algorithm
        if 'signatureAlgorithm' in cert:
            info['signature_algorithm'] = cert['signatureAlgorithm']
        
        # Subject Alternative Names
        if 'subjectAltName' in cert:
            san_domains = []
            for san_type, san_value in cert['subjectAltName']:
                if san_type == 'DNS':
                    san_domains.append(san_value)
            info['san_domains'] = san_domains
        
        return info
        
    except Exception as e:
        logger.error("Certificate parsing failed", error=str(e))
        return {}

def parse_cert_date(date_str: str) -> datetime:
    """Parse certificate date string."""
    try:
        # Handle different date formats
        formats = [
            "%b %d %H:%M:%S %Y %Z",
            "%b %d %H:%M:%S %Y GMT",
            "%Y%m%d%H%M%SZ",
            "%Y-%m-%d %H:%M:%S"
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        
        # If no format matches, try to parse with dateutil
        try:
            return parser.parse(date_str)
        except ImportError:
            pass
        
        # Fallback to current time
        return datetime.now()
        
    except Exception as e:
        logger.warning(f"Date parsing failed for {date_str}", error=str(e))
        return datetime.now()

def check_certificate_status(cert_info: Dict[str, Any]) -> CertificateStatus:
    """Check the status of a certificate."""
    try:
        now = datetime.now()
        
        # Check if certificate is not yet valid
        if 'not_before' in cert_info:
            if now < cert_info['not_before']:
                return CertificateStatus.NOT_YET_VALID
        
        # Check if certificate is expired
        if 'not_after' in cert_info:
            if now > cert_info['not_after']:
                return CertificateStatus.EXPIRED
        
        # Certificate appears valid
        return CertificateStatus.VALID
        
    except Exception as e:
        logger.error("Certificate status check failed", error=str(e))
        return CertificateStatus.UNKNOWN

def get_certificate_chain(host: str, port: int = 443, timeout: float = 10.0) -> List[Dict[str, Any]]:
    """
    Get the full certificate chain.
    
    Args:
        host: Target hostname
        port: Target port
        timeout: Connection timeout
        
    Returns:
        List of certificates in the chain
    """
    try:
        # Create SSL context
        context = ssl.create_default_context()
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
        
        # Create socket and wrap with SSL
        with socket.create_connection((host, port), timeout=timeout) as sock:
            with context.wrap_socket(sock, server_hostname=host) as ssl_sock:
                # Get certificate chain
                cert_chain = ssl_sock.getpeercert(binary_form=True)
                
                if cert_chain:
                    # Parse the certificate
                    cert = OpenSSL.crypto.load_certificate(OpenSSL.crypto.FILETYPE_ASN1, cert_chain)
                    
                    # Get certificate info
                    cert_info = {
                        'subject': dict(cert.get_subject().get_components()),
                        'issuer': dict(cert.get_issuer().get_components()),
                        'serial_number': str(cert.get_serial_number()),
                        'not_before': datetime.strptime(cert.get_notBefore().decode(), '%Y%m%d%H%M%SZ'),
                        'not_after': datetime.strptime(cert.get_notAfter().decode(), '%Y%m%d%H%M%SZ'),
                        'version': cert.get_version(),
                        'signature_algorithm': cert.get_signature_algorithm().decode(),
                        'public_key_algorithm': cert.get_pubkey().type().__name__,
                        'public_key_size': cert.get_pubkey().bits()
                    }
                    
                    return [cert_info]
        
        return []
        
    except Exception as e:
        logger.error(f"Certificate chain retrieval failed for {host}:{port}", error=str(e))
        return []

def check_certificate_revocation(cert_info: Dict[str, Any]) -> bool:
    """
    Check if a certificate is revoked.
    
    Args:
        cert_info: Certificate information
        
    Returns:
        True if revoked, False otherwise
    """
    try:
        # This is a simplified check - in practice, you would:
        # 1. Check OCSP (Online Certificate Status Protocol)
        # 2. Check CRL (Certificate Revocation List)
        # 3. Use a certificate validation service
        
        # For now, we'll assume the certificate is not revoked
        # In a real implementation, you would implement proper revocation checking
        return False
        
    except Exception as e:
        logger.error("Certificate revocation check failed", error=str(e))
        return False

def validate_ssl_configuration(host: str, port: int = 443, timeout: float = 10.0) -> Dict[str, Any]:
    """
    Validate SSL configuration and security settings.
    
    Args:
        host: Target hostname
        port: Target port
        timeout: Connection timeout
        
    Returns:
        Dictionary with validation results
    """
    try:
        results = {
            'host': host,
            'port': port,
            'ssl_enabled': False,
            'tls_version': None,
            'cipher_suite': None,
            'certificate_valid': False,
            'weak_ciphers': [],
            'security_issues': []
        }
        
        # Test different SSL/TLS versions
        ssl_versions = [
            (ssl.PROTOCOL_TLSv1_3, "TLSv1.3"),
            (ssl.PROTOCOL_TLSv1_2, "TLSv1.2"),
            (ssl.PROTOCOL_TLSv1_1, "TLSv1.1"),
            (ssl.PROTOCOL_TLSv1, "TLSv1.0")
        ]
        
        for ssl_version, version_name in ssl_versions:
            try:
                context = ssl.SSLContext(ssl_version)
                context.check_hostname = False
                context.verify_mode = ssl.CERT_NONE
                
                with socket.create_connection((host, port), timeout=timeout) as sock:
                    with context.wrap_socket(sock, server_hostname=host) as ssl_sock:
                        results['ssl_enabled'] = True
                        results['tls_version'] = version_name
                        results['cipher_suite'] = ssl_sock.cipher()[0]
                        
                        # Check for weak ciphers
                        if 'RC4' in results['cipher_suite'] or 'DES' in results['cipher_suite']:
                            results['weak_ciphers'].append(results['cipher_suite'])
                        
                        # Check certificate
                        cert = ssl_sock.getpeercert()
                        if cert:
                            cert_info = parse_certificate_info(cert)
                            cert_status = check_certificate_status(cert_info)
                            results['certificate_valid'] = (cert_status == CertificateStatus.VALID)
                        
                        break
                        
            except Exception:
                continue
        
        # Check for security issues
        if not results['ssl_enabled']:
            results['security_issues'].append("SSL/TLS not supported")
        
        if results['tls_version'] in ["TLSv1.0", "TLSv1.1"]:
            results['security_issues'].append(f"Weak TLS version: {results['tls_version']}")
        
        if results['weak_ciphers']:
            results['security_issues'].append(f"Weak ciphers: {', '.join(results['weak_ciphers'])}")
        
        if not results['certificate_valid']:
            results['security_issues'].append("Invalid certificate")
        
        return results
        
    except Exception as e:
        logger.error(f"SSL configuration validation failed for {host}:{port}", error=str(e))
        return {
            'host': host,
            'port': port,
            'ssl_enabled': False,
            'security_issues': [f"Validation failed: {str(e)}"]
        } 