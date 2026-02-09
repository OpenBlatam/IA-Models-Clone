"""
Security Validator for Flux2 Clothing Changer
=============================================

Security validation and threat detection.
"""

import hashlib
import re
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import logging
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SecurityCheck:
    """Security check result."""
    check_name: str
    passed: bool
    severity: str  # "low", "medium", "high", "critical"
    message: str
    details: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.details is None:
            self.details = {}


class SecurityValidator:
    """Security validation system."""
    
    def __init__(self):
        """Initialize security validator."""
        # Known malicious patterns
        self.malicious_patterns = [
            r"\.\./",  # Path traversal
            r"<script",  # XSS attempts
            r"javascript:",  # JavaScript injection
            r"onerror=",  # Event handler injection
            r"eval\(",  # Code execution
            r"exec\(",  # Code execution
        ]
        
        # File size limits (in bytes)
        self.max_file_size = 50 * 1024 * 1024  # 50MB
        self.max_image_dimension = 8192  # 8K
        
        # Allowed image formats
        self.allowed_formats = {"JPEG", "PNG", "WEBP"}
        
        # Suspicious content patterns
        self.suspicious_content = [
            "malware",
            "virus",
            "exploit",
            "payload",
        ]
    
    def validate_input(
        self,
        image: Optional[Any] = None,
        text: Optional[str] = None,
        file_path: Optional[Path] = None,
    ) -> Tuple[bool, List[SecurityCheck]]:
        """
        Validate input for security threats.
        
        Args:
            image: Image input
            text: Text input (prompts, descriptions)
            file_path: File path input
            
        Returns:
            Tuple of (is_safe, security_checks)
        """
        checks = []
        
        # Validate text input
        if text:
            text_checks = self._validate_text(text)
            checks.extend(text_checks)
        
        # Validate file path
        if file_path:
            path_checks = self._validate_file_path(file_path)
            checks.extend(path_checks)
        
        # Validate image
        if image:
            image_checks = self._validate_image(image)
            checks.extend(image_checks)
        
        # Determine overall safety
        critical_failures = any(
            c.severity == "critical" and not c.passed
            for c in checks
        )
        high_failures = any(
            c.severity == "high" and not c.passed
            for c in checks
        )
        
        is_safe = not (critical_failures or high_failures)
        
        return is_safe, checks
    
    def _validate_text(self, text: str) -> List[SecurityCheck]:
        """Validate text input."""
        checks = []
        
        # Check for malicious patterns
        for pattern in self.malicious_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                checks.append(SecurityCheck(
                    check_name="malicious_pattern",
                    passed=False,
                    severity="high",
                    message=f"Malicious pattern detected: {pattern}",
                    details={"pattern": pattern, "text_snippet": text[:100]},
                ))
        
        # Check for suspicious content
        text_lower = text.lower()
        for suspicious in self.suspicious_content:
            if suspicious in text_lower:
                checks.append(SecurityCheck(
                    check_name="suspicious_content",
                    passed=False,
                    severity="medium",
                    message=f"Suspicious content detected: {suspicious}",
                    details={"content": suspicious},
                ))
        
        # Check length (prevent DoS)
        if len(text) > 10000:
            checks.append(SecurityCheck(
                check_name="text_length",
                passed=False,
                severity="medium",
                message=f"Text too long: {len(text)} characters",
                details={"length": len(text), "max_length": 10000},
            ))
        
        # If no issues found
        if not checks:
            checks.append(SecurityCheck(
                check_name="text_validation",
                passed=True,
                severity="low",
                message="Text validation passed",
            ))
        
        return checks
    
    def _validate_file_path(self, file_path: Path) -> List[SecurityCheck]:
        """Validate file path."""
        checks = []
        
        # Check for path traversal
        path_str = str(file_path)
        if ".." in path_str or path_str.startswith("/"):
            checks.append(SecurityCheck(
                check_name="path_traversal",
                passed=False,
                severity="critical",
                message="Path traversal attempt detected",
                details={"path": path_str},
            ))
        
        # Check file size
        if file_path.exists():
            file_size = file_path.stat().st_size
            if file_size > self.max_file_size:
                checks.append(SecurityCheck(
                    check_name="file_size",
                    passed=False,
                    severity="high",
                    message=f"File too large: {file_size / (1024*1024):.2f}MB",
                    details={"size": file_size, "max_size": self.max_file_size},
                ))
        
        # If no issues found
        if not checks:
            checks.append(SecurityCheck(
                check_name="file_path_validation",
                passed=True,
                severity="low",
                message="File path validation passed",
            ))
        
        return checks
    
    def _validate_image(self, image: Any) -> List[SecurityCheck]:
        """Validate image input."""
        checks = []
        
        try:
            # Convert to PIL if needed
            if isinstance(image, (str, Path)):
                pil_image = Image.open(image)
            elif isinstance(image, np.ndarray):
                pil_image = Image.fromarray(image)
            elif isinstance(image, Image.Image):
                pil_image = image
            else:
                checks.append(SecurityCheck(
                    check_name="image_type",
                    passed=False,
                    severity="high",
                    message=f"Unsupported image type: {type(image)}",
                ))
                return checks
            
            # Check dimensions
            width, height = pil_image.size
            if width > self.max_image_dimension or height > self.max_image_dimension:
                checks.append(SecurityCheck(
                    check_name="image_dimensions",
                    passed=False,
                    severity="medium",
                    message=f"Image too large: {width}x{height}",
                    details={
                        "width": width,
                        "height": height,
                        "max_dimension": self.max_image_dimension,
                    },
                ))
            
            # Check format
            if pil_image.format not in self.allowed_formats:
                checks.append(SecurityCheck(
                    check_name="image_format",
                    passed=False,
                    severity="medium",
                    message=f"Unsupported image format: {pil_image.format}",
                    details={"format": pil_image.format, "allowed": list(self.allowed_formats)},
                ))
            
            # Check for suspicious patterns in image data
            # (This is a simplified check - in production, use more sophisticated methods)
            img_array = np.array(pil_image)
            if img_array.size > 100000000:  # 100M pixels
                checks.append(SecurityCheck(
                    check_name="image_size",
                    passed=False,
                    severity="medium",
                    message="Image data too large",
                    details={"pixel_count": img_array.size},
                ))
            
            # If no issues found
            if not any(not c.passed for c in checks):
                checks.append(SecurityCheck(
                    check_name="image_validation",
                    passed=True,
                    severity="low",
                    message="Image validation passed",
                ))
        
        except Exception as e:
            checks.append(SecurityCheck(
                check_name="image_validation_error",
                passed=False,
                severity="high",
                message=f"Image validation error: {e}",
                details={"error": str(e)},
            ))
        
        return checks
    
    def sanitize_text(self, text: str) -> str:
        """
        Sanitize text input.
        
        Args:
            text: Input text
            
        Returns:
            Sanitized text
        """
        # Remove malicious patterns
        sanitized = text
        for pattern in self.malicious_patterns:
            sanitized = re.sub(pattern, "", sanitized, flags=re.IGNORECASE)
        
        # Remove control characters
        sanitized = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", sanitized)
        
        # Limit length
        if len(sanitized) > 10000:
            sanitized = sanitized[:10000]
        
        return sanitized.strip()
    
    def calculate_file_hash(self, file_path: Path) -> str:
        """
        Calculate SHA256 hash of file.
        
        Args:
            file_path: Path to file
            
        Returns:
            SHA256 hash
        """
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256.update(chunk)
        return sha256.hexdigest()
    
    def get_security_report(self, checks: List[SecurityCheck]) -> Dict[str, Any]:
        """
        Generate security report.
        
        Args:
            checks: List of security checks
            
        Returns:
            Security report
        """
        total = len(checks)
        passed = sum(1 for c in checks if c.passed)
        failed = total - passed
        
        by_severity = {}
        for check in checks:
            if not check.passed:
                by_severity[check.severity] = by_severity.get(check.severity, 0) + 1
        
        return {
            "total_checks": total,
            "passed": passed,
            "failed": failed,
            "pass_rate": passed / total if total > 0 else 0.0,
            "failures_by_severity": by_severity,
            "checks": [
                {
                    "name": c.check_name,
                    "passed": c.passed,
                    "severity": c.severity,
                    "message": c.message,
                    "details": c.details,
                }
                for c in checks
            ],
        }


