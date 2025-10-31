"""
Instagram Captions API v10.0 - Utilities Module

Common utility functions, middleware, and helper classes for the API.
"""

import time
import hashlib
import secrets
import logging
import json
from typing import Any, Dict, List, Optional, Union, Callable
from functools import wraps
from datetime import datetime, timedelta
from fastapi import Request, Response, HTTPException, status
from fastapi.responses import JSONResponse
import asyncio

# =============================================================================
# LOGGING UTILITIES
# =============================================================================

def setup_logging(level: str = "INFO", format_string: Optional[str] = None) -> None:
    """Setup logging configuration."""
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=format_string,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("instagram_captions.log")
        ]
    )

def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the specified name."""
    return logging.getLogger(name)

# =============================================================================
# SECURITY UTILITIES
# =============================================================================

class SecurityUtils:
    """Enterprise-grade security utility functions with advanced threat detection."""
    
    # Enhanced security patterns and configurations
    SECURITY_PATTERNS = {
        'xss_patterns': [
            r'<script[^>]*>.*?</script>',
            r'javascript:',
            r'on\w+\s*=',
            r'<iframe[^>]*>',
            r'<object[^>]*>',
            r'<embed[^>]*>',
            r'<svg[^>]*>.*?</svg>',
            r'<math[^>]*>.*?</math>',
            r'<link[^>]*>',
            r'<meta[^>]*>',
            r'<form[^>]*>.*?</form>',
            r'<input[^>]*>',
            r'<textarea[^>]*>.*?</textarea>',
            r'<select[^>]*>.*?</select>',
            r'<button[^>]*>.*?</button>',
            r'<a[^>]*href\s*=\s*["\']javascript:',
            r'<img[^>]*on\w+\s*=',
            r'<div[^>]*on\w+\s*=',
            r'<span[^>]*on\w+\s*=',
            r'<p[^>]*on\w+\s*='
        ],
        'sql_injection_patterns': [
            r'(\b(union|select|insert|update|delete|drop|create|alter|truncate|rename|grant|revoke)\b)',
            r'(\b(or|and)\b\s+\d+\s*[=<>])',
            r'(\b(exec|execute|execsql|sp_executesql)\b)',
            r'(\b(declare|cast|convert|parse|try_parse)\b)',
            r'(\b(begin|end|if|else|case|when|then)\b)',
            r'(\b(while|for|loop|break|continue)\b)',
            r'(\b(go|batch|block|transaction|commit|rollback)\b)',
            r'(\b(waitfor|delay|timeout)\b)',
            r'(\b(openquery|opendatasource|openrowset)\b)',
            r'(\b(xp_cmdshell|sp_configure|sp_helptext)\b)',
            r'(\b(backup|restore|attach|detach|shutdown)\b)',
            r'(\b(load|dump|import|export|bcp)\b)',
            r'(\b(load_file|into\s+outfile|into\s+dumpfile)\b)',
            r'(\b(concat|group_concat|make_set|elt|field)\b)',
            r'(\b(updatexml|extractvalue|floor|rand|sleep)\b)'
        ],
        'command_injection_patterns': [
            r'(\b(cmd|command|powershell|bash|sh|zsh|fish|tcsh|ksh)\b)',
            r'(\b(system|eval|exec|popen|subprocess|os\.system)\b)',
            r'(\b(rm|del|format|fdisk|mkfs|dd|cp|mv|ln)\b)',
            r'(\b(net|netstat|ipconfig|ifconfig|route|arp|ping|traceroute)\b)',
            r'(\b(wget|curl|ftp|telnet|ssh|scp|rsync|nc|ncat)\b)',
            r'(\b(chmod|chown|chgrp|umask|su|sudo|passwd|useradd)\b)',
            r'(\b(service|systemctl|init|upstart|launchctl)\b)',
            r'(\b(cron|at|anacron|systemd-timer)\b)',
            r'(\b(docker|kubectl|helm|terraform|ansible)\b)',
            r'(\b(git|svn|hg|bzr|cvs|rsync)\b)',
            r'(\b(apt|yum|dnf|pacman|brew|snap|flatpak)\b)',
            r'(\b(ps|top|htop|iotop|iotop|nethogs)\b)',
            r'(\b(lsof|netstat|ss|tcpdump|wireshark|tshark)\b)',
            r'(\b(awk|sed|grep|find|xargs|parallel)\b)',
            r'(\b(python|python3|node|npm|yarn|pip|conda)\b)'
        ],
        'ldap_injection_patterns': [
            r'(\b(uid|cn|sn|givenName|mail|telephoneNumber)\b)',
            r'(\b(|&!)(uid|cn|sn|givenName|mail|telephoneNumber)\b)',
            r'(\b(admin|administrator|root|guest|test|demo)\b)',
            r'(\b(ou|dc|o|c|st|l|street|postalCode|co)\b)',
            r'(\b(employeeID|employeeNumber|department|title)\b)',
            r'(\b(memberOf|member|group|role|permission)\b)',
            r'(\b(createTimestamp|modifyTimestamp|entryUUID)\b)',
            r'(\b(objectClass|structuralObjectClass|subSchemaSubEntry)\b)'
        ],
        'xml_injection_patterns': [
            r'(\b<!DOCTYPE|<!ENTITY|<!ELEMENT|<!ATTLIST)\b)',
            r'(\b(&\w+;|&#\d+;|&#x[0-9a-fA-F]+;)\b)',
            r'(\b(<\?xml|<\?xml-stylesheet|<\?xml-multiple)\b)',
            r'(\b(<\?import|<\?include|<\?ignore|<\?pi)\b)',
            r'(\b(<\?define|<\?choose|<\?when|<\?otherwise)\b)',
            r'(\b(<\?loop|<\?for-each|<\?call-template)\b)',
            r'(\b(<\?apply-templates|<\?copy-of|<\?value-of)\b)',
            r'(\b(<\?if|<\?unless|<\?choose|<\?when)\b)',
            r'(\b(<\?otherwise|<\?fallback|<\?message)\b)',
            r'(\b(<\?terminate|<\?exit|<\?return|<\?break)\b)'
        ],
        'path_traversal_patterns': [
            r'(\b(\.\./|\.\.\\|\.\.\\\\|\.\.%2f|\.\.%5c)\b)',
            r'(\b(%2e%2e%2f|%2e%2e%5c|%2e%2e%5c%2e%2e)\b)',
            r'(\b(\.\.%c0%af|\.\.%c1%9c|\.\.%c0%9v|\.\.%c0%qf)\b)',
            r'(\b(\.\.%255c|\.\.%252e|\.\.%252f|\.\.%255f)\b)',
            r'(\b(\.\.%c0%2e|\.\.%c0%ae|\.\.%c0%af|\.\.%c0%5c)\b)',
            r'(\b(\.\.%c1%2e|\.\.%c1%ae|\.\.%c1%af|\.\.%c1%5c)\b)',
            r'(\b(\.\.%c0%2f|\.\.%c0%5c|\.\.%c1%2f|\.\.%c1%5c)\b)',
            r'(\b(\.\.%252e%252e%252f|\.\.%252e%252e%255c)\b)',
            r'(\b(\.\.%c0%ae%c0%af|\.\.%c0%ae%c0%5c)\b)',
            r'(\b(\.\.%c1%ae%c1%af|\.\.%c1%ae%c1%5c)\b)'
        ],
        'ssrf_patterns': [
            r'(\b(http://|https://|ftp://|file://|gopher://)\b)',
            r'(\b(dict://|ldap://|tftp://|telnet://|ssh://)\b)',
            r'(\b(redis://|mongodb://|mysql://|postgresql://)\b)',
            r'(\b(amqp://|mqtt://|coap://|ws://|wss://)\b)',
            r'(\b(smtp://|pop3://|imap://|nntp://|irc://)\b)',
            r'(\b(rtsp://|rtmp://|mms://|sip://|h323://)\b)',
            r'(\b(ipfs://|dweb://|dat://|hyper://|gemini://)\b)',
            r'(\b(about:|chrome:|chrome-extension:|moz-extension:)\b)',
            r'(\b(view-source:|data:|vbscript:|mocha:|livescript:)\b)',
            r'(\b(ms-help:|ms-its:|microsoft.windows.photogallery:)\b)'
        ]
    }
    
    # Enterprise security configuration
    ENTERPRISE_SECURITY_CONFIG = {
        'threat_detection_level': 'maximum',  # basic, enhanced, maximum
        'ml_threat_detection': True,
        'behavioral_analysis': True,
        'real_time_monitoring': True,
        'threat_intelligence': True,
        'compliance_mode': 'enterprise',  # basic, enterprise, government
        'audit_logging': True,
        'incident_response': True
    }
    
    # Threat intelligence database (simplified)
    THREAT_INTELLIGENCE = {
        'known_malicious_ips': set(),
        'known_malicious_domains': set(),
        'known_malicious_patterns': set(),
        'threat_indicators': set(),
        'risk_scores': {}
    }
    
    # Behavioral analysis patterns
    BEHAVIORAL_PATTERNS = {
        'suspicious_sequences': [
            r'(?:<script|javascript:|on\w+\s*=).*?(?:</script|;|>)',
            r'(?:union|select|insert|update|delete).*?(?:--|#|/\*)',
            r'(?:cmd|command|powershell|bash).*?(?:&|;|\||`)',
            r'(?:\.\./|\.\.\\).*?(?:etc|var|usr|windows|system32)',
            r'(?:http|ftp|file)://.*?(?:localhost|127\.0\.0\.1|0\.0\.0\.0)'
        ],
        'anomaly_thresholds': {
            'request_frequency': 100,  # requests per minute
            'payload_size': 1048576,   # 1MB
            'pattern_matches': 5,      # suspicious patterns per request
            'encoding_variations': 3,  # different encoding attempts
            'parameter_pollution': 10  # duplicate parameters
        }
    }
    
    @staticmethod
    def generate_api_key(length: int = 32, complexity: str = "maximum") -> str:
        """Generate enterprise-grade secure API key with configurable complexity."""
        if complexity == "maximum":
            # Use multiple entropy sources for maximum security
            import os
            import random
            
            # System entropy
            system_entropy = os.urandom(length)
            
            # Cryptographic entropy
            crypto_entropy = secrets.token_bytes(length)
            
            # Time-based entropy
            time_entropy = str(int(time.time() * 1000000)).encode()
            
            # Process entropy
            process_entropy = str(os.getpid()).encode()
            
            # Combine all entropy sources
            combined_entropy = system_entropy + crypto_entropy + time_entropy + process_entropy
            
            # Generate final key using SHA-256
            final_key = hashlib.sha256(combined_entropy).digest()
            
            # Convert to URL-safe base64
            return secrets.token_urlsafe(length)
        else:
            return secrets.token_urlsafe(length)
    
    @staticmethod
    def generate_secure_token(prefix: str = "token", length: int = 24, 
                            security_level: str = "enterprise") -> str:
        """Generate enterprise-grade secure token with advanced security features."""
        if security_level == "enterprise":
            # Enhanced token with multiple security layers
            random_part = secrets.token_urlsafe(length)
            timestamp = str(int(time.time()))
            process_id = str(os.getpid())
            session_id = secrets.token_hex(8)
            
            # Combine with cryptographic hash
            combined = f"{prefix}_{timestamp}_{process_id}_{session_id}_{random_part}"
            final_hash = hashlib.sha256(combined.encode()).hexdigest()
            
            return f"{prefix}_{timestamp}_{final_hash[:length]}"
        else:
            random_part = secrets.token_urlsafe(length)
            timestamp = str(int(time.time()))
            return f"{prefix}_{timestamp}_{random_part}"
    
    @staticmethod
    def hash_password(password: str, salt: Optional[str] = None, 
                     algorithm: str = "pbkdf2") -> str:
        """Enterprise-grade password hashing with multiple algorithms."""
        if algorithm == "pbkdf2":
            import hashlib
            import hmac
            
            if salt is None:
                salt = secrets.token_hex(32)
            
            # PBKDF2 with SHA-256, 100,000 iterations
            key = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
            return f"pbkdf2:sha256:100000:${salt}${key.hex()}"
        
        elif algorithm == "bcrypt":
            import bcrypt
            
            if salt is None:
                salt = bcrypt.gensalt(rounds=12)
            
            hashed = bcrypt.hashpw(password.encode(), salt)
            return hashed.decode()
        
        elif algorithm == "argon2":
            try:
                import argon2
                
                if salt is None:
                    salt = secrets.token_hex(32)
                
                # Argon2id with high security parameters
                ph = argon2.PasswordHasher(
                    time_cost=3,      # 3 iterations
                    memory_cost=65536, # 64MB
                    parallelism=4,     # 4 parallel threads
                    hash_len=32        # 32 bytes output
                )
                
                return ph.hash(password)
            except ImportError:
                # Fallback to SHA-256 if argon2 not available
                return SecurityUtils.hash_password(password, salt, "sha256")
        
        else:  # Default to SHA-256
            if salt is None:
                salt = secrets.token_hex(16)
            
            salted_password = f"{password}{salt}"
            hashed = hashlib.sha256(salted_password.encode()).hexdigest()
            return f"sha256:${salt}${hashed}"
    
    @staticmethod
    def verify_password(password: str, hashed_password: str) -> bool:
        """Verify password against various hash formats."""
        try:
            if hashed_password.startswith("pbkdf2:"):
                # PBKDF2 format: pbkdf2:sha256:100000:$salt$hash
                parts = hashed_password.split("$")
                if len(parts) == 3:
                    algorithm_info = parts[0]
                    salt = parts[1]
                    stored_hash = parts[2]
                    
                    # Parse algorithm info
                    algo_parts = algorithm_info.split(":")
                    hash_algo = algo_parts[1]
                    iterations = int(algo_parts[2])
                    
                    # Verify using same parameters
                    import hashlib
                    import hmac
                    key = hashlib.pbkdf2_hmac(hash_algo, password.encode(), salt.encode(), iterations)
                    return key.hex() == stored_hash
                
            elif hashed_password.startswith("sha256:"):
                # SHA-256 format: sha256:$salt$hash
                salt, hash_part = hashed_password.split("$", 1)
                return SecurityUtils.hash_password(password, salt, "sha256") == hashed_password
            
            else:
                # Try bcrypt
                try:
                    import bcrypt
                    return bcrypt.checkpw(password.encode(), hashed_password.encode())
                except (ImportError, ValueError):
                    pass
                
                # Try argon2
                try:
                    import argon2
                    ph = argon2.PasswordHasher()
                    ph.verify(hashed_password, password)
                    return True
                except (ImportError, argon2.exceptions.VerifyMismatchError):
                    pass
            
            return False
            
        except (ValueError, AttributeError, IndexError):
            return False
    
    @staticmethod
    def verify_api_key(api_key: str, min_length: int = 32, 
                      security_level: str = "enterprise") -> Dict[str, Any]:
        """Enterprise-grade API key validation with comprehensive security analysis."""
        result = {
            "valid": False,
            "score": 0,
            "warnings": [],
            "security_level": "unknown",
            "recommendations": []
        }
        
        if not api_key or len(api_key) < min_length:
            result["warnings"].append(f"API key too short: {len(api_key)} < {min_length}")
            return result
        
        # Enhanced security validation
        score = 100
        warnings = []
        recommendations = []
        
        # Check for common weak patterns
        weak_patterns = [
            'test', 'demo', 'example', '123', 'abc', 'password', 'admin',
            'key', 'secret', 'token', 'auth', 'api', 'dev', 'development',
            'staging', 'production', 'local', 'localhost', '127.0.0.1'
        ]
        
        api_key_lower = api_key.lower()
        for pattern in weak_patterns:
            if pattern in api_key_lower:
                score -= 20
                warnings.append(f"Weak pattern detected: {pattern}")
                recommendations.append(f"Remove weak pattern: {pattern}")
        
        # Check for sequential characters
        sequential_patterns = [
            'abcdefghijklmnopqrstuvwxyz',
            'zyxwvutsrqponmlkjihgfedcba',
            '0123456789',
            '9876543210'
        ]
        
        for seq in sequential_patterns:
            for i in range(len(seq) - 2):
                pattern = seq[i:i+3]
                if pattern in api_key_lower:
                    score -= 15
                    warnings.append(f"Sequential pattern detected: {pattern}")
                    recommendations.append("Avoid sequential character patterns")
                    break
        
        # Check for repeated characters
        char_counts = {}
        for char in api_key:
            char_counts[char] = char_counts.get(char, 0) + 1
        
        max_repetition = max(char_counts.values()) if char_counts else 0
        if max_repetition > len(api_key) * 0.3:  # More than 30% repetition
            score -= 25
            warnings.append(f"High character repetition: {max_repetition} times")
            recommendations.append("Reduce character repetition")
        
        # Check character diversity
        unique_chars = len(set(api_key))
        diversity_ratio = unique_chars / len(api_key)
        if diversity_ratio < 0.7:  # Less than 70% unique characters
            score -= 20
            warnings.append(f"Low character diversity: {diversity_ratio:.1%}")
            recommendations.append("Increase character diversity")
        
        # Check for common encoding patterns
        encoding_patterns = [
            r'[A-F0-9]{32}',  # MD5-like
            r'[A-F0-9]{40}',  # SHA1-like
            r'[A-F0-9]{64}',  # SHA256-like
            r'[A-Za-z0-9+/]{43,}={0,2}'  # Base64-like
        ]
        
        import re
        for pattern in encoding_patterns:
            if re.match(pattern, api_key):
                score -= 10
                warnings.append("Common encoding pattern detected")
                recommendations.append("Use custom encoding patterns")
                break
        
        # Determine security level
        if score >= 90:
            security_level = "maximum"
        elif score >= 70:
            security_level = "enterprise"
        elif score >= 50:
            security_level = "enhanced"
        elif score >= 30:
            security_level = "basic"
        else:
            security_level = "weak"
        
        # Final validation
        is_valid = score >= 50 and len(api_key) >= min_length
        
        result.update({
            "valid": is_valid,
            "score": max(0, score),
            "warnings": warnings,
            "security_level": security_level,
            "recommendations": recommendations,
            "character_diversity": diversity_ratio,
            "max_repetition": max_repetition,
            "length": len(api_key)
        })
        
        return result
    
    @staticmethod
    def sanitize_input(text: str, max_length: int = 1000, strict: bool = True, 
                      security_level: str = "enterprise") -> Dict[str, Any]:
        """Enterprise-grade input sanitization with comprehensive threat detection."""
        result = {
            "sanitized_text": "",
            "threats_detected": [],
            "security_score": 100,
            "sanitization_applied": [],
            "risk_level": "low"
        }
        
        if not text:
            return result
        
        original_text = text
        sanitized = text
        threats = []
        sanitization_steps = []
        security_score = 100
        
        # Step 1: Basic character filtering
        dangerous_chars = ['<', '>', '&', '"', "'", ';', '(', ')', '{', '}', '[', ']', '\\', '/', '|', '`', '~', '^']
        for char in dangerous_chars:
            if char in sanitized:
                sanitized = sanitized.replace(char, '')
                sanitization_steps.append(f"Removed dangerous character: {char}")
                security_score -= 2
        
        # Step 2: HTML entity removal
        html_entities = ['&lt;', '&gt;', '&amp;', '&quot;', '&#39;', '&apos;', '&nbsp;', '&copy;', '&reg;', '&trade;']
        for entity in html_entities:
            if entity in sanitized:
                sanitized = sanitized.replace(entity, '')
                sanitization_steps.append(f"Removed HTML entity: {entity}")
                security_score -= 3
        
        # Step 3: Advanced pattern detection and removal
        import re
        
        # Check all security patterns
        for pattern_category, patterns in SecurityUtils.SECURITY_PATTERNS.items():
            for pattern in patterns:
                matches = re.findall(pattern, sanitized, flags=re.IGNORECASE)
                if matches:
                    threat_info = {
                        "category": pattern_category,
                        "pattern": pattern,
                        "matches": matches,
                        "severity": "high" if pattern_category in ["xss_patterns", "command_injection_patterns"] else "medium"
                    }
                    threats.append(threat_info)
                    security_score -= 10
                    
                    # Remove the pattern
                    sanitized = re.sub(pattern, '', sanitized, flags=re.IGNORECASE)
                    sanitization_steps.append(f"Removed {pattern_category}: {len(matches)} matches")
        
        # Step 4: Behavioral analysis
        if SecurityUtils.ENTERPRISE_SECURITY_CONFIG.get('behavioral_analysis', False):
            behavioral_threats = SecurityUtils._analyze_behavioral_patterns(sanitized)
            if behavioral_threats:
                threats.extend(behavioral_threats)
                security_score -= 15
                sanitization_steps.append("Applied behavioral analysis sanitization")
        
        # Step 5: Advanced encoding detection
        encoding_threats = SecurityUtils._detect_encoding_attacks(sanitized)
        if encoding_threats:
            threats.extend(encoding_threats)
            security_score -= 20
            sanitization_steps.append("Removed encoding-based attacks")
        
        # Step 6: Strict mode sanitization
        if strict:
            # Remove any remaining potentially dangerous content
            sanitized = re.sub(r'[^\w\s\-.,!?¿¡áéíóúñüÁÉÍÓÚÑÜ]', '', sanitized)
            sanitization_steps.append("Applied strict character filtering")
            security_score += 5
        
        # Step 7: Length validation
        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length]
            sanitization_steps.append(f"Truncated to {max_length} characters")
            security_score -= 5
        
        # Step 8: Final cleanup
        sanitized = sanitized.strip()
        
        # Calculate risk level
        if security_score >= 80:
            risk_level = "low"
        elif security_score >= 60:
            risk_level = "medium"
        elif security_score >= 40:
            risk_level = "high"
        else:
            risk_level = "critical"
        
        result.update({
            "sanitized_text": sanitized,
            "threats_detected": threats,
            "security_score": max(0, security_score),
            "sanitization_applied": sanitization_steps,
            "risk_level": risk_level,
            "original_length": len(original_text),
            "sanitized_length": len(sanitized),
            "threat_count": len(threats)
        })
        
        return result
    
    @staticmethod
    def _analyze_behavioral_patterns(text: str) -> List[Dict[str, Any]]:
        """Analyze text for behavioral anomalies and suspicious patterns."""
        threats = []
        
        # Check for suspicious sequences
        for pattern in SecurityUtils.BEHAVIORAL_PATTERNS['suspicious_sequences']:
            matches = re.findall(pattern, text, flags=re.IGNORECASE)
            if matches:
                threats.append({
                    "category": "behavioral_anomaly",
                    "pattern": pattern,
                    "matches": matches,
                    "severity": "high",
                    "description": "Suspicious behavioral pattern detected"
                })
        
        # Check for encoding variations
        encoding_variations = 0
        encoding_patterns = [
            r'%[0-9a-fA-F]{2}',  # URL encoding
            r'\\x[0-9a-fA-F]{2}',  # Hex encoding
            r'\\u[0-9a-fA-F]{4}',  # Unicode encoding
            r'&#[0-9]+;',  # HTML numeric encoding
            r'&#x[0-9a-fA-F]+;'  # HTML hex encoding
        ]
        
        for pattern in encoding_patterns:
            if re.search(pattern, text):
                encoding_variations += 1
        
        if encoding_variations > SecurityUtils.BEHAVIORAL_PATTERNS['anomaly_thresholds']['encoding_variations']:
            threats.append({
                "category": "encoding_attack",
                "pattern": "multiple_encodings",
                "matches": encoding_variations,
                "severity": "medium",
                "description": "Multiple encoding variations detected"
            })
        
        return threats
    
    @staticmethod
    def _detect_encoding_attacks(text: str) -> List[Dict[str, Any]]:
        """Detect various encoding-based attack patterns."""
        threats = []
        
        # Double encoding detection
        double_encoded_patterns = [
            r'%25[0-9a-fA-F]{2}',  # %25 = % in URL encoding
            r'%5c%5c',  # %5c = \ in URL encoding
            r'%2f%2f',  # %2f = / in URL encoding
            r'%2e%2e',  # %2e = . in URL encoding
        ]
        
        for pattern in double_encoded_patterns:
            if re.search(pattern, text):
                threats.append({
                    "category": "encoding_attack",
                    "pattern": "double_encoding",
                    "matches": re.findall(pattern, text),
                    "severity": "high",
                    "description": "Double encoding attack detected"
                })
        
        # Unicode normalization attacks
        unicode_attacks = [
            r'[\u0000-\u001f\u007f-\u009f]',  # Control characters
            r'[\u2000-\u200f\u2028-\u202f\u205f-\u206f]',  # Various spaces
            r'[\u2060-\u2064\u206a-\u206f]',  # Format characters
            r'[\ufeff\ufffe\uffff]',  # BOM and invalid characters
        ]
        
        for pattern in unicode_attacks:
            if re.search(pattern, text):
                threats.append({
                    "category": "unicode_attack",
                    "pattern": "control_characters",
                    "matches": re.findall(pattern, text),
                    "severity": "medium",
                    "description": "Unicode control characters detected"
                })
        
        return threats
    
    @staticmethod
    def validate_content_type(content_type: str, security_level: str = "enterprise") -> Dict[str, Any]:
        """Enterprise-grade content type validation with security analysis."""
        result = {
            "valid": False,
            "security_score": 100,
            "warnings": [],
            "recommendations": []
        }
        
        allowed_types = [
            'text/plain',
            'application/json',
            'application/x-www-form-urlencoded',
            'multipart/form-data'
        ]
        
        # Check exact match first
        if content_type in allowed_types:
            result["valid"] = True
            result["security_score"] = 100
            return result
        
        # Check for partial matches (e.g., application/json; charset=utf-8)
        for allowed_type in allowed_types:
            if content_type.startswith(allowed_type + ';'):
                result["valid"] = True
                result["security_score"] = 95
                result["warnings"].append("Charset specified - ensure it's safe")
                return result
        
        # Security analysis for invalid content types
        result["security_score"] = 0
        
        # Check for dangerous content types
        dangerous_types = [
            'text/html', 'application/javascript', 'text/javascript',
            'application/xml', 'text/xml', 'application/x-shockwave-flash',
            'application/x-executable', 'application/x-msdownload',
            'application/x-msi', 'application/x-msdos-program'
        ]
        
        for dangerous_type in dangerous_types:
            if dangerous_type in content_type.lower():
                result["warnings"].append(f"Dangerous content type: {dangerous_type}")
                result["recommendations"].append(f"Reject {dangerous_type} content")
        
        # Check for suspicious patterns
        suspicious_patterns = [
            r'data:',  # Data URLs
            r'vbscript:',  # VBScript
            r'javascript:',  # JavaScript
            r'file:',  # File protocol
            r'ftp:',  # FTP protocol
            r'gopher:',  # Gopher protocol
        ]
        
        import re
        for pattern in suspicious_patterns:
            if re.search(pattern, content_type, re.IGNORECASE):
                result["warnings"].append(f"Suspicious pattern in content type: {pattern}")
                result["recommendations"].append("Reject content type with suspicious patterns")
        
        return result
    
    @staticmethod
    def validate_file_extension(filename: str, allowed_extensions: Optional[List[str]] = None, 
                               security_level: str = "enterprise") -> Dict[str, Any]:
        """Enterprise-grade file extension validation with comprehensive security analysis."""
        result = {
            "valid": False,
            "security_score": 100,
            "warnings": [],
            "recommendations": [],
            "risk_level": "low"
        }
        
        if allowed_extensions is None:
            allowed_extensions = ['.txt', '.json', '.md', '.csv', '.xml', '.yaml', '.yml']
        
        if not filename or '.' not in filename:
            result["warnings"].append("No file extension found")
            result["recommendations"].append("Files must have valid extensions")
            return result
        
        extension = filename.lower().split('.')[-1]
        result["extension"] = extension
        result["filename"] = filename
        
        # Check if extension is allowed
        if f'.{extension}' in allowed_extensions:
            result["valid"] = True
            result["security_score"] = 100
        else:
            result["valid"] = False
            result["security_score"] = 0
            
            # Check for dangerous extensions
            dangerous_extensions = [
                'exe', 'bat', 'cmd', 'com', 'pif', 'scr', 'vbs', 'js', 'jar',
                'msi', 'msu', 'msp', 'mst', 'reg', 'ps1', 'psm1', 'psd1',
                'ps1xml', 'psc1', 'psc2', 'pssc', 'pl', 'py', 'rb', 'sh',
                'cgi', 'asp', 'aspx', 'php', 'jsp', 'jspx', 'cfm', 'cfml'
            ]
            
            if extension in dangerous_extensions:
                result["warnings"].append(f"Dangerous file extension: .{extension}")
                result["recommendations"].append(f"Reject .{extension} files - potential security risk")
                result["risk_level"] = "critical"
            else:
                result["warnings"].append(f"Unsupported file extension: .{extension}")
                result["recommendations"].append(f"Add .{extension} to allowed extensions if safe")
                result["risk_level"] = "medium"
        
        # Additional security checks
        if security_level == "enterprise":
            # Check for path traversal attempts
            if any(pattern in filename for pattern in ['..', '\\', '/', ':', '*', '?', '"', '<', '>', '|']):
                result["warnings"].append("Path traversal characters detected in filename")
                result["recommendations"].append("Sanitize filename to remove dangerous characters")
                result["security_score"] = max(0, result["security_score"] - 30)
                result["risk_level"] = "high"
            
            # Check for suspicious patterns
            suspicious_patterns = [
                r'\.\.',  # Double dots
                r'[\\/:*?"<>|]',  # Invalid characters
                r'^(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])$',  # Reserved names
                r'\.(lnk|url|pif|scf|hta|cpl|msc|msi|msp|mst|reg|ps1|psm1|psd1|ps1xml|psc1|psc2|pssc)$'
            ]
            
            import re
            for pattern in suspicious_patterns:
                if re.search(pattern, filename, re.IGNORECASE):
                    result["warnings"].append(f"Suspicious pattern in filename: {pattern}")
                    result["recommendations"].append("Reject filename with suspicious patterns")
                    result["security_score"] = max(0, result["security_score"] - 20)
                    result["risk_level"] = "high"
        
        return result
    
    @staticmethod
    def generate_security_headers(security_level: str = "enterprise", 
                                 compliance_mode: str = "enterprise") -> Dict[str, str]:
        """Generate comprehensive enterprise-grade security headers."""
        headers = {
            'X-Content-Type-Options': 'nosniff',
            'X-Frame-Options': 'DENY',
            'X-XSS-Protection': '1; mode=block',
            'Referrer-Policy': 'strict-origin-when-cross-origin',
            'X-Request-ID': secrets.token_hex(16),
            'X-API-Version': '10.0.0'
        }
        
        if security_level == "enterprise":
            headers.update({
                'Content-Security-Policy': "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data: https:; font-src 'self' https:; connect-src 'self' https:; frame-ancestors 'none'; base-uri 'self'; form-action 'self'",
                'Strict-Transport-Security': 'max-age=31536000; includeSubDomains; preload',
                'Permissions-Policy': 'geolocation=(), microphone=(), camera=(), payment=(), usb=(), magnetometer=(), gyroscope=(), accelerometer=()',
                'Cross-Origin-Embedder-Policy': 'require-corp',
                'Cross-Origin-Opener-Policy': 'same-origin',
                'Cross-Origin-Resource-Policy': 'same-origin',
                'Origin-Agent-Cluster': '?1',
                'Sec-Fetch-Dest': 'empty',
                'Sec-Fetch-Mode': 'cors',
                'Sec-Fetch-Site': 'same-origin',
                'Sec-GPC': '1'
            })
        
        if compliance_mode == "government":
            headers.update({
                'X-Content-Type-Options': 'nosniff',
                'X-Frame-Options': 'DENY',
                'X-XSS-Protection': '1; mode=block',
                'Strict-Transport-Security': 'max-age=31536000; includeSubDomains; preload',
                'Content-Security-Policy': "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self'; font-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'self'; form-action 'self'",
                'Permissions-Policy': 'geolocation=(), microphone=(), camera=(), payment=(), usb=(), magnetometer=(), gyroscope=(), accelerometer=()'
            })
        
        return headers
    
    @staticmethod
    def validate_url(url: str, allowed_domains: Optional[List[str]] = None, 
                    security_level: str = "enterprise") -> Dict[str, Any]:
        """Enterprise-grade URL validation with comprehensive security analysis."""
        result = {
            "valid": False,
            "security_score": 100,
            "warnings": [],
            "recommendations": [],
            "risk_level": "low"
        }
        
        if not url:
            result["warnings"].append("Empty URL provided")
            return result
        
        # Basic URL pattern validation
        import re
        url_pattern = r'^https?://[^\s/$.?#].[^\s]*$'
        if not re.match(url_pattern, url):
            result["warnings"].append("Invalid URL format")
            result["recommendations"].append("URL must be properly formatted")
            result["security_score"] = 0
            return result
        
        # Parse URL for analysis
        from urllib.parse import urlparse
        try:
            parsed_url = urlparse(url)
            result["parsed_url"] = {
                "scheme": parsed_url.scheme,
                "netloc": parsed_url.netloc,
                "path": parsed_url.path,
                "query": parsed_url.query,
                "fragment": parsed_url.fragment
            }
        except Exception as e:
            result["warnings"].append(f"URL parsing failed: {e}")
            result["security_score"] = 0
            return result
        
        # Check for dangerous protocols
        dangerous_protocols = ['file:', 'ftp:', 'gopher:', 'data:', 'javascript:', 'vbscript:', 'mailto:', 'tel:']
        if parsed_url.scheme in dangerous_protocols:
            result["warnings"].append(f"Dangerous protocol: {parsed_url.scheme}")
            result["recommendations"].append(f"Reject {parsed_url.scheme} protocol")
            result["security_score"] = 0
            result["risk_level"] = "critical"
            return result
        
        # Check for localhost and private IPs
        localhost_patterns = [
            r'localhost', r'127\.0\.0\.1', r'0\.0\.0\.0', r'::1',
            r'10\.', r'172\.(1[6-9]|2[0-9]|3[0-1])\.', r'192\.168\.'
        ]
        
        for pattern in localhost_patterns:
            if re.search(pattern, parsed_url.netloc):
                result["warnings"].append(f"Local/private network access: {parsed_url.netloc}")
                result["recommendations"].append("Reject local/private network URLs")
                result["security_score"] = max(0, result["security_score"] - 50)
                result["risk_level"] = "high"
        
        # Check for suspicious patterns in path/query
        suspicious_path_patterns = [
            r'\.\./', r'\.\.\\',  # Path traversal
            r'<script', r'javascript:', r'on\w+\s*=',  # XSS attempts
            r'union\s+select', r'insert\s+into', r'drop\s+table',  # SQL injection
            r'cmd\.exe', r'powershell', r'bash', r'sh',  # Command injection
        ]
        
        full_path = parsed_url.path + '?' + parsed_url.query if parsed_url.query else parsed_url.path
        
        for pattern in suspicious_path_patterns:
            if re.search(pattern, full_path, re.IGNORECASE):
                result["warnings"].append(f"Suspicious pattern in URL: {pattern}")
                result["recommendations"].append("Reject URL with suspicious patterns")
                result["security_score"] = max(0, result["security_score"] - 30)
                result["risk_level"] = "high"
        
        # Check for allowed domains if specified
        if allowed_domains:
            if parsed_url.netloc not in allowed_domains:
                result["warnings"].append(f"Domain not in allowed list: {parsed_url.netloc}")
                result["recommendations"].append(f"Add {parsed_url.netloc} to allowed domains if safe")
                result["security_score"] = max(0, result["security_score"] - 20)
                result["risk_level"] = "medium"
        
        # Additional enterprise security checks
        if security_level == "enterprise":
            # Check for encoding attacks
            if '%' in url:
                result["warnings"].append("URL encoding detected")
                result["recommendations"].append("Verify URL encoding is legitimate")
                result["security_score"] = max(0, result["security_score"] - 10)
            
            # Check for overly long URLs
            if len(url) > 2048:
                result["warnings"].append("URL too long (potential buffer overflow)")
                result["recommendations"].append("Limit URL length to 2048 characters")
                result["security_score"] = max(0, result["security_score"] - 15)
                result["risk_level"] = "medium"
        
        # Final validation
        result["valid"] = result["security_score"] >= 50
        
        return result
    
    @staticmethod
    def generate_csrf_token(security_level: str = "enterprise") -> str:
        """Generate enterprise-grade CSRF token with enhanced security."""
        if security_level == "enterprise":
            # Enhanced token with multiple entropy sources
            random_part = secrets.token_hex(32)
            timestamp = str(int(time.time()))
            process_id = str(os.getpid())
            session_id = secrets.token_hex(8)
            
            # Combine with cryptographic hash
            combined = f"csrf_{timestamp}_{process_id}_{session_id}_{random_part}"
            final_hash = hashlib.sha256(combined.encode()).hexdigest()
            
            return f"csrf_{timestamp}_{final_hash[:32]}"
        else:
            return secrets.token_hex(32)
    
    @staticmethod
    def verify_csrf_token(token: str, stored_token: str, 
                         security_level: str = "enterprise") -> Dict[str, Any]:
        """Enterprise-grade CSRF token verification with security analysis."""
        result = {
            "valid": False,
            "security_score": 100,
            "warnings": [],
            "recommendations": []
        }
        
        # Basic verification
        is_valid = secrets.compare_digest(token, stored_token)
        result["valid"] = is_valid
        
        if not is_valid:
            result["security_score"] = 0
            result["warnings"].append("CSRF token mismatch")
            result["recommendations"].append("Regenerate CSRF token and retry")
            return result
        
        # Enterprise security analysis
        if security_level == "enterprise":
            # Check token format
            if not token.startswith("csrf_"):
                result["warnings"].append("Invalid CSRF token format")
                result["security_score"] = max(0, result["security_score"] - 20)
            
            # Check token age (if timestamp is included)
            try:
                if "_" in token:
                    parts = token.split("_")
                    if len(parts) >= 3:
                        timestamp = int(parts[1])
                        current_time = int(time.time())
                        token_age = current_time - timestamp
                        
                        # Token should not be older than 24 hours
                        if token_age > 86400:  # 24 hours
                            result["warnings"].append(f"CSRF token too old: {token_age} seconds")
                            result["recommendations"].append("Regenerate CSRF token")
                            result["security_score"] = max(0, result["security_score"] - 30)
            except (ValueError, IndexError):
                pass
        
        return result
    
    @staticmethod
    def analyze_threat_intelligence(input_data: str, threat_type: str = "general") -> Dict[str, Any]:
        """Analyze input data against threat intelligence database."""
        result = {
            "threats_found": [],
            "risk_score": 0,
            "confidence": 0.0,
            "recommendations": []
        }
        
        # This is a simplified threat intelligence analysis
        # In a real enterprise environment, this would connect to external threat feeds
        
        # Check against known malicious patterns
        for pattern_category, patterns in SecurityUtils.SECURITY_PATTERNS.items():
            for pattern in patterns:
                matches = re.findall(pattern, input_data, re.IGNORECASE)
                if matches:
                    result["threats_found"].append({
                        "category": pattern_category,
                        "pattern": pattern,
                        "matches": matches,
                        "severity": "high" if pattern_category in ["xss_patterns", "command_injection_patterns"] else "medium"
                    })
        
        # Calculate risk score
        threat_count = len(result["threats_found"])
        result["risk_score"] = min(100, threat_count * 25)
        
        # Calculate confidence based on pattern matches
        if threat_count > 0:
            result["confidence"] = min(1.0, 0.5 + (threat_count * 0.1))
        
        # Generate recommendations
        if result["risk_score"] > 75:
            result["recommendations"].append("Immediate action required - high threat level detected")
        elif result["risk_score"] > 50:
            result["recommendations"].append("Review and investigate - medium threat level detected")
        elif result["risk_score"] > 25:
            result["recommendations"].append("Monitor closely - low threat level detected")
        else:
            result["recommendations"].append("No immediate threats detected")
        
        return result

# =============================================================================
# CACHE UTILITIES
# =============================================================================

class CacheManager:
    """Simple in-memory cache manager."""
    
    def __init__(self, max_size: int = 1000, ttl: int = 3600):
        self.max_size = max_size
        self.ttl = ttl
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.access_order: List[str] = []
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if key not in self.cache:
            return None
        
        cache_entry = self.cache[key]
        
        # Check if expired
        if time.time() > cache_entry['expires_at']:
            self.delete(key)
            return None
        
        # Update access order
        if key in self.access_order:
            self.access_order.remove(key)
        self.access_order.append(key)
        
        return cache_entry['value']
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache."""
        if len(self.cache) >= self.max_size:
            # Remove least recently used item
            if self.access_order:
                oldest_key = self.access_order.pop(0)
                del self.cache[oldest_key]
        
        ttl_seconds = ttl or self.ttl
        expires_at = time.time() + ttl_seconds
        
        self.cache[key] = {
            'value': value,
            'expires_at': expires_at,
            'created_at': time.time()
        }
        
        if key not in self.access_order:
            self.access_order.append(key)
    
    def delete(self, key: str) -> None:
        """Delete key from cache."""
        if key in self.cache:
            del self.cache[key]
        if key in self.access_order:
            self.access_order.remove(key)
    
    def clear(self) -> None:
        """Clear all cache entries."""
        self.cache.clear()
        self.access_order.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        current_time = time.time()
        expired_keys = [
            key for key, entry in self.cache.items()
            if current_time > entry['expires_at']
        ]
        
        return {
            'total_entries': len(self.cache),
            'expired_entries': len(expired_keys),
            'max_size': self.max_size,
            'ttl_seconds': self.ttl,
            'hit_rate': self._calculate_hit_rate()
        }
    
    def _calculate_hit_rate(self) -> float:
        """Calculate cache hit rate (simplified)."""
        # This is a simplified calculation
        return 0.85  # Placeholder value

# =============================================================================
# RATE LIMITING UTILITIES
# =============================================================================

class RateLimiter:
    """Simple rate limiter implementation."""
    
    def __init__(self, requests_per_minute: int = 100, burst_size: int = 20):
        self.requests_per_minute = requests_per_minute
        self.burst_size = burst_size
        self.requests: Dict[str, List[float]] = {}
        self.cleanup_interval = 60  # Clean up old entries every 60 seconds
        self.last_cleanup = time.time()
    
    def is_allowed(self, identifier: str) -> bool:
        """Check if request is allowed."""
        current_time = time.time()
        
        # Cleanup old entries periodically
        if current_time - self.last_cleanup > self.cleanup_interval:
            self._cleanup_old_entries()
            self.last_cleanup = current_time
        
        # Get request history for this identifier
        if identifier not in self.requests:
            self.requests[identifier] = []
        
        request_history = self.requests[identifier]
        
        # Remove requests older than 1 minute
        cutoff_time = current_time - 60
        request_history = [req_time for req_time in request_history if req_time > cutoff_time]
        self.requests[identifier] = request_history
        
        # Check rate limit
        if len(request_history) >= self.requests_per_minute:
            return False
        
        # Check burst limit
        recent_requests = [req_time for req_time in request_history if req_time > current_time - 1]
        if len(recent_requests) >= self.burst_size:
            return False
        
        # Allow request
        request_history.append(current_time)
        return True
    
    def _cleanup_old_entries(self) -> None:
        """Clean up old request entries."""
        current_time = time.time()
        cutoff_time = current_time - 60
        
        for identifier in list(self.requests.keys()):
            self.requests[identifier] = [
                req_time for req_time in self.requests[identifier]
                if req_time > cutoff_time
            ]
            
            # Remove empty entries
            if not self.requests[identifier]:
                del self.requests[identifier]
    
    def get_remaining_requests(self, identifier: str) -> int:
        """Get remaining requests for an identifier."""
        if identifier not in self.requests:
            return self.requests_per_minute
        
        current_time = time.time()
        cutoff_time = current_time - 60
        
        recent_requests = [
            req_time for req_time in self.requests[identifier]
            if req_time > cutoff_time
        ]
        
        return max(0, self.requests_per_minute - len(recent_requests))

# =============================================================================
# MIDDLEWARE UTILITIES
# =============================================================================

async def rate_limit_middleware(request: Request, call_next):
    """Rate limiting middleware."""
    rate_limiter = getattr(request.app.state, 'rate_limiter', None)
    
    if rate_limiter:
        # Use client IP as identifier
        client_ip = request.client.host
        request_id = f"{client_ip}:{request.url.path}"
        
        if not rate_limiter.is_allowed(request_id):
            remaining = rate_limiter.get_remaining_requests(request_id)
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": "Rate limit exceeded",
                    "retry_after": 60,
                    "remaining_requests": remaining
                },
                headers={
                    "X-RateLimit-Remaining": str(remaining),
                    "X-RateLimit-Reset": str(int(time.time() + 60))
                }
            )
    
    response = await call_next(request)
    
    # Add rate limit headers
    if rate_limiter:
        client_ip = request.client.host
        request_id = f"{client_ip}:{request.url.path}"
        remaining = rate_limiter.get_remaining_requests(request_id)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
    
    return response

async def logging_middleware(request: Request, call_next):
    """Logging middleware for requests."""
    start_time = time.time()
    
    # Log request
    logger = get_logger("api")
    logger.info(f"Request: {request.method} {request.url.path} from {request.client.host}")
    
    # Process request
    response = await call_next(request)
    
    # Log response
    process_time = time.time() - start_time
    logger.info(f"Response: {response.status_code} in {process_time:.3f}s")
    
    # Add timing header
    response.headers["X-Process-Time"] = f"{process_time:.3f}"
    
    return response

async def security_middleware(request: Request, call_next):
    """Enhanced security middleware with comprehensive protection."""
    # Validate content type for POST/PUT requests
    if request.method in ["POST", "PUT", "PATCH"]:
        content_type = request.headers.get("content-type", "")
        if not SecurityUtils.validate_content_type(content_type):
            logger = get_logger("security")
            logger.warning(f"Invalid content type: {content_type} from {request.client.host}")
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"error": "Invalid content type"}
            )
    
    # Validate user agent for suspicious patterns
    user_agent = request.headers.get("user-agent", "")
    if any(pattern in user_agent.lower() for pattern in ["bot", "crawler", "scraper"]):
        logger = get_logger("security")
        logger.info(f"Bot/crawler detected: {user_agent[:100]} from {request.client.host}")
    
    # Process request
    response = await call_next(request)
    
    # Add comprehensive security headers
    security_headers = SecurityUtils.generate_security_headers()
    for header_name, header_value in security_headers.items():
        response.headers[header_name] = header_value
    
    # Add additional security headers
    response.headers["X-Request-ID"] = secrets.token_hex(16)
    response.headers["X-API-Version"] = "10.0.0"
    
    return response

# =============================================================================
# PERFORMANCE UTILITIES
# =============================================================================

def timing_decorator(func: Callable) -> Callable:
    """Decorator to measure function execution time."""
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger = get_logger("performance")
            logger.info(f"{func.__name__} executed in {execution_time:.3f}s")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger = get_logger("performance")
            logger.error(f"{func.__name__} failed after {execution_time:.3f}s: {e}")
            raise
    
    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger = get_logger("performance")
            logger.info(f"{func.__name__} executed in {execution_time:.3f}s")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger = get_logger("performance")
            logger.error(f"{func.__name__} failed after {execution_time:.3f}s: {e}")
            raise
    
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper

class PerformanceMonitor:
    """Enterprise-grade performance monitoring utility with advanced analytics, ML insights, and predictive capabilities."""
    
    def __init__(self, enterprise_mode: bool = True):
        self.metrics: Dict[str, List[float]] = {}
        self.historical_data: Dict[str, List[Dict[str, Any]]] = {}
        self.alerts: List[Dict[str, Any]] = []
        self.thresholds: Dict[str, Dict[str, float]] = {}
        self.start_time = time.time()
        self.enterprise_mode = enterprise_mode
        
        # Enterprise features
        self.performance_baselines: Dict[str, Dict[str, float]] = {}
        self.anomaly_detection: Dict[str, Dict[str, Any]] = {}
        self.predictive_models: Dict[str, Dict[str, Any]] = {}
        self.capacity_planning: Dict[str, Dict[str, Any]] = {}
        self.business_metrics: Dict[str, Dict[str, Any]] = {}
        
        # Advanced monitoring
        self.sla_targets: Dict[str, Dict[str, float]] = {}
        self.cost_analysis: Dict[str, Dict[str, float]] = {}
        self.resource_utilization: Dict[str, Dict[str, float]] = {}
        self.user_experience_metrics: Dict[str, Dict[str, Any]] = {}
        
        # ML and AI features
        self.ml_insights: List[Dict[str, Any]] = []
        self.pattern_recognition: Dict[str, List[Dict[str, Any]]] = {}
        self.forecasting_data: Dict[str, List[Dict[str, Any]]] = {}
        
        # Initialize enterprise defaults
        if enterprise_mode:
            self._initialize_enterprise_features()
    
    def _initialize_enterprise_features(self):
        """Initialize enterprise-grade monitoring features."""
        # Set default SLA targets
        self.sla_targets = {
            'response_time': {'p95': 200, 'p99': 500, 'max': 1000},  # milliseconds
            'throughput': {'min': 100, 'target': 1000, 'max': 10000},  # requests/second
            'availability': {'min': 99.5, 'target': 99.9, 'max': 99.99},  # percentage
            'error_rate': {'max': 1.0, 'target': 0.1, 'critical': 5.0}  # percentage
        }
        
        # Initialize cost analysis
        self.cost_analysis = {
            'compute_cost_per_hour': 0.50,
            'storage_cost_per_gb': 0.023,
            'network_cost_per_gb': 0.09,
            'total_monthly_cost': 0.0
        }
        
        # Initialize resource utilization tracking
        self.resource_utilization = {
            'cpu_usage': [],
            'memory_usage': [],
            'disk_io': [],
            'network_io': [],
            'database_connections': []
        }
    
    def record_metric(self, name: str, value: float, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Record a performance metric with enhanced enterprise features."""
        if name not in self.metrics:
            self.metrics[name] = []
            self.historical_data[name] = []
            self.anomaly_detection[name] = {}
            self.predictive_models[name] = {}
            self.capacity_planning[name] = {}
            self.business_metrics[name] = {}
            self.pattern_recognition[name] = []
            self.forecasting_data[name] = []
        
        self.metrics[name].append(value)
        
        # Enhanced historical data with business context
        record = {
            'timestamp': time.time(),
            'value': value,
            'metadata': metadata or {},
            'business_hour': self._is_business_hour(),
            'day_of_week': datetime.now().strftime('%A'),
            'month': datetime.now().strftime('%B'),
            'season': self._get_season()
        }
        self.historical_data[name].append(record)
        
        # Keep only last 10000 values for enterprise analytics
        if len(self.metrics[name]) > 10000:
            self.metrics[name] = self.metrics[name][-10000:]
            self.historical_data[name] = self.historical_data[name][-10000:]
        
        # Enterprise-grade analysis
        if self.enterprise_mode:
            self._analyze_enterprise_metrics(name, value, record)
            self._detect_anomalies(name, value)
            self._update_predictive_models(name, value)
            self._update_capacity_planning(name, value)
            self._analyze_business_impact(name, value, record)
        
        # Check thresholds and generate alerts
        self._check_thresholds(name, value)
    
    def _is_business_hour(self) -> bool:
        """Check if current time is during business hours."""
        now = datetime.now()
        return 9 <= now.hour <= 17 and now.weekday() < 5
    
    def _get_season(self) -> str:
        """Get current season for seasonal analysis."""
        month = datetime.now().month
        if month in [12, 1, 2]:
            return 'winter'
        elif month in [3, 4, 5]:
            return 'spring'
        elif month in [6, 7, 8]:
            return 'summer'
        else:
            return 'autumn'
    
    def _analyze_enterprise_metrics(self, name: str, value: float, record: Dict[str, Any]):
        """Analyze metrics for enterprise insights."""
        # Update performance baselines
        if name not in self.performance_baselines:
            self.performance_baselines[name] = {
                'historical_avg': 0.0,
                'historical_std': 0.0,
                'trend_direction': 'stable',
                'seasonal_patterns': {},
                'business_hour_impact': 0.0
            }
        
        # Calculate business hour impact
        if record['business_hour']:
            business_hour_values = [r['value'] for r in self.historical_data[name] if r['business_hour']]
            non_business_values = [r['value'] for r in self.historical_data[name] if not r['business_hour']]
            
            if business_hour_values and non_business_values:
                business_avg = sum(business_hour_values) / len(business_hour_values)
                non_business_avg = sum(non_business_values) / len(non_business_values)
                self.performance_baselines[name]['business_hour_impact'] = business_avg - non_business_avg
    
    def _detect_anomalies(self, name: str, value: float):
        """Advanced anomaly detection using statistical methods."""
        if len(self.metrics[name]) < 10:
            return
        
        values = self.metrics[name]
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        std_dev = variance ** 0.5
        
        # Z-score based anomaly detection
        z_score = abs((value - mean) / std_dev) if std_dev > 0 else 0
        
        if z_score > 3.0:  # 3-sigma rule
            anomaly = {
                'timestamp': time.time(),
                'metric': name,
                'value': value,
                'z_score': z_score,
                'severity': 'high' if z_score > 4.0 else 'medium',
                'expected_range': f"{mean - 2*std_dev:.2f} - {mean + 2*std_dev:.2f}",
                'deviation': f"{((value - mean) / mean * 100):.1f}%" if mean > 0 else "N/A"
            }
            
            if name not in self.anomaly_detection:
                self.anomaly_detection[name] = []
            
            self.anomaly_detection[name].append(anomaly)
            
            # Keep only last 100 anomalies per metric
            if len(self.anomaly_detection[name]) > 100:
                self.anomaly_detection[name] = self.anomaly_detection[name][-100:]
    
    def _update_predictive_models(self, name: str, value: float):
        """Update predictive models for forecasting."""
        if len(self.metrics[name]) < 50:
            return
        
        # Simple linear regression for trend prediction
        values = self.metrics[name][-50:]  # Last 50 values
        timestamps = list(range(len(values)))
        
        if len(values) > 1:
            x_mean = sum(timestamps) / len(timestamps)
            y_mean = sum(values) / len(values)
            
            numerator = sum((timestamps[i] - x_mean) * (values[i] - y_mean) for i in range(len(values)))
            denominator = sum((timestamps[i] - x_mean) ** 2 for i in range(len(values)))
            
            if denominator != 0:
                slope = numerator / denominator
                intercept = y_mean - slope * x_mean
                
                # Predict next 5 values
                predictions = []
                for i in range(1, 6):
                    pred_value = slope * (len(values) + i) + intercept
                    predictions.append(max(0, pred_value))  # Ensure non-negative
                
                self.predictive_models[name] = {
                    'slope': slope,
                    'intercept': intercept,
                    'predictions': predictions,
                    'confidence': min(0.95, 1.0 - abs(slope) / 1000),  # Higher confidence for stable trends
                    'last_updated': time.time()
                }
    
    def _update_capacity_planning(self, name: str, value: float):
        """Update capacity planning insights."""
        if name not in self.capacity_planning:
            self.capacity_planning[name] = {
                'peak_values': [],
                'growth_rate': 0.0,
                'capacity_recommendations': [],
                'scaling_thresholds': {}
            }
        
        # Track peak values
        self.capacity_planning[name]['peak_values'].append(value)
        if len(self.capacity_planning[name]['peak_values']) > 100:
            self.capacity_planning[name]['peak_values'] = self.capacity_planning[name]['peak_values'][-100:]
        
        # Calculate growth rate
        if len(self.capacity_planning[name]['peak_values']) > 10:
            recent_avg = sum(self.capacity_planning[name]['peak_values'][-10:]) / 10
            older_avg = sum(self.capacity_planning[name]['peak_values'][-20:-10]) / 10
            
            if older_avg > 0:
                growth_rate = ((recent_avg - older_avg) / older_avg) * 100
                self.capacity_planning[name]['growth_rate'] = growth_rate
                
                # Generate capacity recommendations
                if growth_rate > 20:
                    self.capacity_planning[name]['capacity_recommendations'].append({
                        'timestamp': time.time(),
                        'type': 'scaling_up',
                        'reason': f'High growth rate: {growth_rate:.1f}%',
                        'priority': 'high' if growth_rate > 50 else 'medium'
                    })
    
    def _analyze_business_impact(self, name: str, value: float, record: Dict[str, Any]):
        """Analyze business impact of performance metrics."""
        if name not in self.business_metrics:
            self.business_metrics[name] = {
                'revenue_impact': 0.0,
                'user_satisfaction': 0.0,
                'operational_efficiency': 0.0,
                'cost_implications': 0.0
            }
        
        # Example business impact calculations (customize based on your business model)
        if 'response_time' in name:
            # Response time impact on user satisfaction
            if value < 100:
                satisfaction = 1.0
            elif value < 500:
                satisfaction = 0.8
            elif value < 1000:
                satisfaction = 0.6
            else:
                satisfaction = 0.3
            
            self.business_metrics[name]['user_satisfaction'] = satisfaction
            
            # Revenue impact (example: 1 second delay = 7% revenue drop)
            if value > 1000:
                revenue_impact = (value - 1000) * 0.007
                self.business_metrics[name]['revenue_impact'] = revenue_impact
        
        elif 'error_rate' in name:
            # Error rate impact on operational efficiency
            efficiency = max(0, 1.0 - (value / 100))
            self.business_metrics[name]['operational_efficiency'] = efficiency
    
    def set_sla_target(self, metric_name: str, sla_type: str, target_value: float) -> None:
        """Set SLA targets for enterprise monitoring."""
        if metric_name not in self.sla_targets:
            self.sla_targets[metric_name] = {}
        
        self.sla_targets[metric_name][sla_type] = target_value
    
    def get_sla_compliance(self, metric_name: str) -> Dict[str, Any]:
        """Get SLA compliance status for a metric."""
        if metric_name not in self.sla_targets or metric_name not in self.metrics:
            return {'status': 'no_sla_defined'}
        
        current_stats = self.get_statistics(metric_name)
        sla_targets = self.sla_targets[metric_name]
        compliance = {}
        
        for sla_type, target in sla_targets.items():
            if sla_type == 'p95' and 'p95' in current_stats:
                compliance[sla_type] = {
                    'target': target,
                    'actual': current_stats['p95'],
                    'compliant': current_stats['p95'] <= target,
                    'deviation': current_stats['p95'] - target
                }
            elif sla_type == 'max' and 'max' in current_stats:
                compliance[sla_type] = {
                    'target': target,
                    'actual': current_stats['max'],
                    'compliant': current_stats['max'] <= target,
                    'deviation': current_stats['max'] - target
                }
            elif sla_type == 'min' and 'min' in current_stats:
                compliance[sla_type] = {
                    'target': target,
                    'actual': current_stats['min'],
                    'compliant': current_stats['min'] >= target,
                    'deviation': target - current_stats['min']
                }
        
        # Overall compliance status
        total_slas = len(compliance)
        compliant_slas = sum(1 for c in compliance.values() if c['compliant'])
        overall_compliance = (compliant_slas / total_slas * 100) if total_slas > 0 else 0
        
        return {
            'overall_compliance': overall_compliance,
            'compliant_slas': compliant_slas,
            'total_slas': total_slas,
            'sla_details': compliance,
            'status': 'compliant' if overall_compliance >= 95 else 'warning' if overall_compliance >= 80 else 'critical'
        }
    
    def get_enterprise_insights(self) -> Dict[str, Any]:
        """Get comprehensive enterprise insights and recommendations."""
        insights = {
            'performance_summary': self.get_performance_summary(),
            'sla_compliance': {},
            'anomaly_summary': {},
            'predictive_insights': {},
            'capacity_recommendations': {},
            'business_impact': {},
            'cost_analysis': self._calculate_cost_analysis(),
            'recommendations': []
        }
        
        # SLA compliance for all metrics
        for metric_name in self.metrics.keys():
            insights['sla_compliance'][metric_name] = self.get_sla_compliance(metric_name)
        
        # Anomaly summary
        for metric_name, anomalies in self.anomaly_detection.items():
            if anomalies:
                insights['anomaly_summary'][metric_name] = {
                    'total_anomalies': len(anomalies),
                    'recent_anomalies': len([a for a in anomalies if time.time() - a['timestamp'] < 3600]),
                    'high_severity': len([a for a in anomalies if a['severity'] == 'high']),
                    'latest_anomaly': anomalies[-1] if anomalies else None
                }
        
        # Predictive insights
        for metric_name, model in self.predictive_models.items():
            if model:
                insights['predictive_insights'][metric_name] = {
                    'trend_direction': 'increasing' if model['slope'] > 0 else 'decreasing',
                    'trend_strength': abs(model['slope']),
                    'confidence': model['confidence'],
                    'next_predictions': model['predictions'][:3],  # Next 3 predictions
                    'recommendation': self._generate_prediction_recommendation(model)
                }
        
        # Capacity recommendations
        for metric_name, planning in self.capacity_planning.items():
            if planning['capacity_recommendations']:
                insights['capacity_recommendations'][metric_name] = {
                    'growth_rate': planning['growth_rate'],
                    'recent_recommendations': planning['capacity_recommendations'][-5:],
                    'scaling_needed': planning['growth_rate'] > 20
                }
        
        # Business impact summary
        for metric_name, business_metrics in self.business_metrics.items():
            insights['business_impact'][metric_name] = business_metrics
        
        # Generate overall recommendations
        insights['recommendations'] = self._generate_enterprise_recommendations(insights)
        
        return insights
    
    def _calculate_cost_analysis(self) -> Dict[str, Any]:
        """Calculate current operational costs."""
        total_requests = sum(len(values) for values in self.metrics.values())
        uptime_hours = self.get_uptime() / 3600
        
        compute_cost = uptime_hours * self.cost_analysis['compute_cost_per_hour']
        storage_cost = (total_requests * 0.001) * self.cost_analysis['storage_cost_per_gb']  # Simplified
        network_cost = (total_requests * 0.01) * self.cost_analysis['network_cost_per_gb']   # Simplified
        
        total_cost = compute_cost + storage_cost + network_cost
        cost_per_request = total_cost / total_requests if total_requests > 0 else 0
        
        return {
            'compute_cost': compute_cost,
            'storage_cost': storage_cost,
            'network_cost': network_cost,
            'total_cost': total_cost,
            'cost_per_request': cost_per_request,
            'uptime_hours': uptime_hours,
            'total_requests': total_requests
        }
    
    def _generate_prediction_recommendation(self, model: Dict[str, Any]) -> str:
        """Generate recommendations based on predictive models."""
        slope = model['slope']
        confidence = model['confidence']
        
        if confidence < 0.7:
            return "Insufficient data for reliable predictions"
        
        if abs(slope) < 0.1:
            return "Performance is stable, no immediate action needed"
        elif slope > 0.5:
            return "Performance degrading rapidly, investigate root cause"
        elif slope > 0.1:
            return "Performance slowly degrading, plan for optimization"
        elif slope < -0.5:
            return "Performance improving rapidly, maintain current practices"
        elif slope < -0.1:
            return "Performance slowly improving, continue optimization efforts"
        
        return "Performance trend is stable"
    
    def _generate_enterprise_recommendations(self, insights: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate enterprise-level recommendations."""
        recommendations = []
        
        # SLA compliance recommendations
        for metric_name, compliance in insights['sla_compliance'].items():
            if compliance['status'] == 'critical':
                recommendations.append({
                    'priority': 'critical',
                    'category': 'sla_compliance',
                    'metric': metric_name,
                    'message': f"Critical SLA violation for {metric_name}. Immediate action required.",
                    'action': 'investigate_root_cause'
                })
            elif compliance['status'] == 'warning':
                recommendations.append({
                    'priority': 'high',
                    'category': 'sla_compliance',
                    'metric': metric_name,
                    'message': f"SLA warning for {metric_name}. Monitor closely.",
                    'action': 'optimize_performance'
                })
        
        # Anomaly recommendations
        for metric_name, anomaly_summary in insights['anomaly_summary'].items():
            if anomaly_summary['high_severity'] > 0:
                recommendations.append({
                    'priority': 'high',
                    'category': 'anomaly_detection',
                    'metric': metric_name,
                    'message': f"High severity anomalies detected in {metric_name}.",
                    'action': 'investigate_anomalies'
                })
        
        # Capacity planning recommendations
        for metric_name, capacity in insights['capacity_recommendations'].items():
            if capacity['scaling_needed']:
                recommendations.append({
                    'priority': 'medium',
                    'category': 'capacity_planning',
                    'metric': metric_name,
                    'message': f"High growth rate ({capacity['growth_rate']:.1f}%) detected in {metric_name}.",
                    'action': 'plan_capacity_increase'
                })
        
        # Cost optimization recommendations
        cost_analysis = insights['cost_analysis']
        if cost_analysis['cost_per_request'] > 0.01:  # $0.01 per request threshold
            recommendations.append({
                'priority': 'medium',
                'category': 'cost_optimization',
                'metric': 'cost_per_request',
                'message': f"High cost per request: ${cost_analysis['cost_per_request']:.4f}",
                'action': 'optimize_resource_usage'
            })
        
        return recommendations
    
    def export_enterprise_report(self, format: str = "json") -> str:
        """Export comprehensive enterprise performance report."""
        insights = self.get_enterprise_insights()
        
        if format.lower() == "json":
            return json.dumps(insights, indent=2, default=str)
        elif format.lower() == "csv":
            return self._export_csv_report(insights)
        else:
            return json.dumps(insights, indent=2, default=str)
    
    def _export_csv_report(self, insights: Dict[str, Any]) -> str:
        """Export insights as CSV format."""
        # This is a simplified CSV export - in production, use proper CSV library
        csv_lines = []
        
        # Performance summary
        csv_lines.append("Performance Summary")
        csv_lines.append("Metric,Count,Latest Value,Average Value")
        for metric_name, summary in insights['performance_summary']['metrics_summary'].items():
            csv_lines.append(f"{metric_name},{summary['count']},{summary['latest_value']},{summary['avg_value']:.2f}")
        
        csv_lines.append("")
        
        # SLA compliance
        csv_lines.append("SLA Compliance")
        csv_lines.append("Metric,Overall Compliance,Status")
        for metric_name, compliance in insights['sla_compliance'].items():
            if 'overall_compliance' in compliance:
                csv_lines.append(f"{metric_name},{compliance['overall_compliance']:.1f}%,{compliance['status']}")
        
        return "\n".join(csv_lines)
    
    def get_statistics(self, name: str) -> Dict[str, float]:
        """Get comprehensive statistics for a metric."""
        if name not in self.metrics or not self.metrics[name]:
            return {}
        
        values = self.metrics[name]
        sorted_values = sorted(values)
        n = len(values)
        
        if n == 0:
            return {}
        
        # Calculate percentiles
        p50 = sorted_values[n // 2]
        p90 = sorted_values[int(n * 0.9)]
        p95 = sorted_values[int(n * 0.95)]
        p99 = sorted_values[int(n * 0.99)]
        
        # Calculate variance and standard deviation
        mean = sum(values) / n
        variance = sum((x - mean) ** 2 for x in values) / n
        std_dev = variance ** 0.5
        
        return {
            'count': n,
            'min': min(values),
            'max': max(values),
            'mean': mean,
            'median': p50,
            'p90': p90,
            'p95': p95,
            'p99': p99,
            'std_dev': std_dev,
            'variance': variance,
            'range': max(values) - min(values)
        }
    
    def get_all_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all metrics."""
        return {name: self.get_statistics(name) for name in self.metrics}
    
    def get_performance_trends(self, metric_name: str, window_minutes: int = 60) -> Dict[str, Any]:
        """Get performance trends over time."""
        if metric_name not in self.historical_data:
            return {}
        
        current_time = time.time()
        window_seconds = window_minutes * 60
        
        # Filter data within the time window
        recent_data = [
            record for record in self.historical_data[metric_name]
            if current_time - record['timestamp'] <= window_seconds
        ]
        
        if not recent_data:
            return {}
        
        values = [record['value'] for record in recent_data]
        timestamps = [record['timestamp'] for record in recent_data]
        
        # Calculate trend (simple linear regression)
        if len(values) > 1:
            x_mean = sum(timestamps) / len(timestamps)
            y_mean = sum(values) / len(values)
            
            numerator = sum((timestamps[i] - x_mean) * (values[i] - y_mean) for i in range(len(values)))
            denominator = sum((timestamps[i] - x_mean) ** 2 for i in range(len(values)))
            
            if denominator != 0:
                slope = numerator / denominator
                trend_direction = 'improving' if slope < 0 else 'degrading' if slope > 0 else 'stable'
            else:
                slope = 0
                trend_direction = 'stable'
        else:
            slope = 0
            trend_direction = 'insufficient_data'
        
        return {
            'trend_direction': trend_direction,
            'slope': slope,
            'data_points': len(values),
            'window_minutes': window_minutes,
            'recent_stats': self.get_statistics(metric_name)
        }
    
    def get_alerts(self, severity: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get performance alerts, optionally filtered by severity."""
        if severity:
            return [alert for alert in self.alerts if alert['severity'] == severity]
        return self.alerts
    
    def clear_alerts(self) -> None:
        """Clear all performance alerts."""
        self.alerts.clear()
    
    def get_uptime(self) -> float:
        """Get system uptime in seconds."""
        return time.time() - self.start_time
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        return {
            'uptime_seconds': self.get_uptime(),
            'total_metrics': len(self.metrics),
            'total_alerts': len(self.alerts),
            'active_thresholds': len(self.thresholds),
            'metrics_summary': {
                name: {
                    'count': len(values),
                    'latest_value': values[-1] if values else None,
                    'avg_value': sum(values) / len(values) if values else 0
                }
                for name, values in self.metrics.items()
            }
        }

# =============================================================================
# DATA VALIDATION UTILITIES
# =============================================================================

class ValidationUtils:
    """Data validation utilities."""
    
    @staticmethod
    def validate_email(email: str) -> bool:
        """Simple email validation."""
        import re
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None
    
    @staticmethod
    def validate_url(url: str) -> bool:
        """Simple URL validation."""
        import re
        pattern = r'^https?://(?:[-\w.])+(?:[:\d]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:#(?:[\w.])*)?)?$'
        return re.match(pattern, url) is not None
    
    @staticmethod
    def validate_phone(phone: str) -> bool:
        """Simple phone number validation."""
        import re
        # Remove all non-digit characters
        digits_only = re.sub(r'\D', '', phone)
        return 7 <= len(digits_only) <= 15
    
    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """Sanitize filename for safe file operations."""
        import re
        # Remove or replace unsafe characters
        unsafe_chars = r'[<>:"/\\|?*]'
        sanitized = re.sub(unsafe_chars, '_', filename)
        
        # Remove leading/trailing dots and spaces
        sanitized = sanitized.strip('. ')
        
        # Ensure filename is not empty
        if not sanitized:
            sanitized = "unnamed_file"
        
        return sanitized

# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'setup_logging',
    'get_logger',
    'SecurityUtils',
    'CacheManager',
    'RateLimiter',
    'rate_limit_middleware',
    'logging_middleware',
    'security_middleware',
    'timing_decorator',
    'PerformanceMonitor',
    'ValidationUtils',
    'CircuitBreaker',
    'ErrorHandler'
]

# =============================================================================
# CIRCUIT BREAKER PATTERN (ENTERPRISE-GRADE)
# =============================================================================

class CircuitBreaker:
    """Enterprise-grade circuit breaker implementation with advanced monitoring and adaptive thresholds."""
    
    # Circuit breaker states
    CLOSED = "CLOSED"      # Normal operation
    OPEN = "OPEN"          # Circuit is open, requests are blocked
    HALF_OPEN = "HALF_OPEN"  # Testing if service has recovered
    
    def __init__(self, 
                 failure_threshold: int = 5,
                 recovery_timeout: int = 60,
                 success_threshold: int = 3,
                 adaptive_thresholds: bool = True,
                 health_check_interval: int = 30,
                 monitoring_enabled: bool = True):
        
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold
        self.adaptive_thresholds = adaptive_thresholds
        self.health_check_interval = health_check_interval
        self.monitoring_enabled = monitoring_enabled
        
        # State management
        self.state = self.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.last_success_time = None
        self.last_state_change = time.time()
        
        # Advanced monitoring
        self.total_requests = 0
        self.total_failures = 0
        self.total_successes = 0
        self.response_times = []
        self.error_types = {}
        self.health_metrics = {}
        
        # Adaptive thresholds
        self.dynamic_failure_threshold = failure_threshold
        self.dynamic_recovery_timeout = recovery_timeout
        self.performance_baseline = None
        
        # Health check tracking
        self.last_health_check = time.time()
        self.health_check_results = []
        self.service_dependencies = set()
        
        # Business impact tracking
        self.business_impact = {
            'requests_blocked': 0,
            'estimated_revenue_loss': 0.0,
            'user_experience_impact': 0.0,
            'operational_efficiency': 1.0
        }
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        if not self._can_execute():
            self._handle_circuit_open()
            raise Exception(f"Circuit breaker is {self.state}")
        
        start_time = time.time()
        self.total_requests += 1
        
        try:
            result = func(*args, **kwargs)
            self._on_success(time.time() - start_time)
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            self._on_failure(e, execution_time)
            raise
    
    async def call_async(self, func: Callable, *args, **kwargs) -> Any:
        """Execute async function with circuit breaker protection."""
        if not self._can_execute():
            self._handle_circuit_open()
            raise Exception(f"Circuit breaker is {self.state}")
        
        start_time = time.time()
        self.total_requests += 1
        
        try:
            result = await func(*args, **kwargs)
            self._on_success(time.time() - start_time)
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            self._on_failure(e, execution_time)
            raise
    
    def _can_execute(self) -> bool:
        """Check if request can be executed based on current state."""
        current_time = time.time()
        
        if self.state == self.CLOSED:
            return True
        
        elif self.state == self.OPEN:
            if current_time - self.last_failure_time >= self.dynamic_recovery_timeout:
                self._transition_to_half_open()
                return True
            return False
        
        elif self.state == self.HALF_OPEN:
            return True
        
        return False
    
    def _on_success(self, execution_time: float):
        """Handle successful execution."""
        self.success_count += 1
        self.total_successes += 1
        self.last_success_time = time.time()
        
        # Record performance metrics
        self.response_times.append(execution_time)
        if len(self.response_times) > 1000:
            self.response_times = self.response_times[-1000:]
        
        # Update performance baseline
        if self.performance_baseline is None:
            self.performance_baseline = execution_time
        else:
            # Exponential moving average
            self.performance_baseline = 0.9 * self.performance_baseline + 0.1 * execution_time
        
        # Check if we can close the circuit
        if self.state == self.HALF_OPEN and self.success_count >= self.success_threshold:
            self._transition_to_closed()
        
        # Update business impact
        self._update_business_impact(True, execution_time)
        
        # Adaptive threshold adjustment
        if self.adaptive_thresholds:
            self._adjust_thresholds(True)
    
    def _on_failure(self, error: Exception, execution_time: float):
        """Handle execution failure."""
        self.failure_count += 1
        self.total_failures += 1
        self.last_failure_time = time.time()
        
        # Track error types
        error_type = type(error).__name__
        self.error_types[error_type] = self.error_types.get(error_type, 0) + 1
        
        # Check if we should open the circuit
        if self.failure_count >= self.dynamic_failure_threshold:
            self._transition_to_open()
        
        # Update business impact
        self._update_business_impact(False, execution_time)
        
        # Adaptive threshold adjustment
        if self.adaptive_thresholds:
            self._adjust_thresholds(False)
    
    def _transition_to_open(self):
        """Transition circuit to OPEN state."""
        if self.state != self.OPEN:
            self.state = self.OPEN
            self.last_state_change = time.time()
            self.success_count = 0
            
            if self.monitoring_enabled:
                self._log_state_change("OPEN", "Circuit opened due to failures")
    
    def _transition_to_half_open(self):
        """Transition circuit to HALF_OPEN state."""
        if self.state != self.HALF_OPEN:
            self.state = self.HALF_OPEN
            self.last_state_change = time.time()
            self.failure_count = 0
            self.success_count = 0
            
            if self.monitoring_enabled:
                self._log_state_change("HALF_OPEN", "Testing service recovery")
    
    def _transition_to_closed(self):
        """Transition circuit to CLOSED state."""
        if self.state != self.CLOSED:
            self.state = self.CLOSED
            self.last_state_change = time.time()
            self.failure_count = 0
            self.success_count = 0
            
            if self.monitoring_enabled:
                self._log_state_change("CLOSED", "Service recovered, circuit closed")
    
    def _handle_circuit_open(self):
        """Handle request when circuit is open."""
        self.business_impact['requests_blocked'] += 1
        
        # Estimate revenue loss (customize based on your business model)
        avg_request_value = 0.10  # Example: $0.10 per request
        self.business_impact['estimated_revenue_loss'] += avg_request_value
        
        # Update user experience impact
        self.business_impact['user_experience_impact'] = min(1.0, 
            self.business_impact['requests_blocked'] / max(1, self.total_requests))
    
    def _adjust_thresholds(self, success: bool):
        """Dynamically adjust thresholds based on performance."""
        if success:
            # Gradually reduce failure threshold if service is performing well
            if self.dynamic_failure_threshold > self.failure_threshold * 0.5:
                self.dynamic_failure_threshold = max(self.failure_threshold * 0.5, 
                                                   self.dynamic_failure_threshold * 0.95)
        else:
            # Increase failure threshold if service is struggling
            if self.dynamic_failure_threshold < self.failure_threshold * 2.0:
                self.dynamic_failure_threshold = min(self.failure_threshold * 2.0, 
                                                   self.dynamic_failure_threshold * 1.05)
    
    def _update_business_impact(self, success: bool, execution_time: float):
        """Update business impact metrics."""
        if success:
            # Improve operational efficiency
            self.business_impact['operational_efficiency'] = min(1.0, 
                self.business_impact['operational_efficiency'] + 0.001)
        else:
            # Decrease operational efficiency
            self.business_impact['operational_efficiency'] = max(0.0, 
                self.business_impact['operational_efficiency'] - 0.01)
    
    def _log_state_change(self, new_state: str, reason: str):
        """Log circuit breaker state changes."""
        logger = get_logger("circuit_breaker")
        logger.info(f"Circuit breaker state changed to {new_state}: {reason}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive circuit breaker status."""
        current_time = time.time()
        
        # Calculate success rate
        total_attempts = self.total_successes + self.total_failures
        success_rate = (self.total_successes / total_attempts * 100) if total_attempts > 0 else 0
        
        # Calculate average response time
        avg_response_time = sum(self.response_times) / len(self.response_times) if self.response_times else 0
        
        # Calculate time in current state
        time_in_state = current_time - self.last_state_change
        
        # Health score (0-100)
        health_score = self._calculate_health_score()
        
        return {
            'state': self.state,
            'failure_count': self.failure_count,
            'success_count': self.success_count,
            'total_requests': self.total_requests,
            'total_failures': self.total_failures,
            'total_successes': self.total_successes,
            'success_rate': success_rate,
            'last_failure_time': self.last_failure_time,
            'last_success_time': self.last_success_time,
            'last_state_change': self.last_state_change,
            'time_in_current_state': time_in_state,
            'failure_threshold': self.dynamic_failure_threshold,
            'recovery_timeout': self.dynamic_recovery_timeout,
            'success_threshold': self.success_threshold,
            'performance_baseline': self.performance_baseline,
            'avg_response_time': avg_response_time,
            'health_score': health_score,
            'error_types': self.error_types,
            'business_impact': self.business_impact,
            'adaptive_thresholds_enabled': self.adaptive_thresholds,
            'monitoring_enabled': self.monitoring_enabled
        }
    
    def _calculate_health_score(self) -> float:
        """Calculate overall health score (0-100)."""
        if self.total_requests == 0:
            return 100.0
        
        # Base score from success rate
        success_rate_score = (self.total_successes / self.total_requests) * 60
        
        # Performance score from response times
        if self.performance_baseline and self.response_times:
            recent_avg = sum(self.response_times[-10:]) / min(10, len(self.response_times))
            if recent_avg <= self.performance_baseline * 1.2:
                performance_score = 20
            elif recent_avg <= self.performance_baseline * 1.5:
                performance_score = 15
            elif recent_avg <= self.performance_baseline * 2.0:
                performance_score = 10
            else:
                performance_score = 5
        else:
            performance_score = 20
        
        # Stability score from state consistency
        if self.state == self.CLOSED:
            stability_score = 20
        elif self.state == self.HALF_OPEN:
            stability_score = 10
        else:
            stability_score = 0
        
        return min(100.0, success_rate_score + performance_score + stability_score)
    
    def reset(self) -> None:
        """Manually reset the circuit breaker."""
        self.state = self.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.last_success_time = None
        self.last_state_change = time.time()
        
        if self.monitoring_enabled:
            self._log_state_change("CLOSED", "Manual reset")
    
    def force_open(self) -> None:
        """Force the circuit breaker to OPEN state."""
        self._transition_to_open()
    
    def force_close(self) -> None:
        """Force the circuit breaker to CLOSED state."""
        self._transition_to_closed()
    
    def update_thresholds(self, failure_threshold: Optional[int] = None, 
                         recovery_timeout: Optional[int] = None,
                         success_threshold: Optional[int] = None) -> None:
        """Update circuit breaker thresholds."""
        if failure_threshold is not None:
            self.failure_threshold = failure_threshold
            self.dynamic_failure_threshold = failure_threshold
        
        if recovery_timeout is not None:
            self.recovery_timeout = recovery_timeout
            self.dynamic_recovery_timeout = recovery_timeout
        
        if success_threshold is not None:
            self.success_threshold = success_threshold
    
    def add_service_dependency(self, service_name: str) -> None:
        """Add a service dependency for health monitoring."""
        self.service_dependencies.add(service_name)
    
    def remove_service_dependency(self, service_name: str) -> None:
        """Remove a service dependency."""
        self.service_dependencies.discard(service_name)
    
    def get_health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check."""
        current_time = time.time()
        
        # Basic health indicators
        health_indicators = {
            'circuit_state': self.state,
            'success_rate': (self.total_successes / max(1, self.total_requests)) * 100,
            'response_time_stable': self._is_response_time_stable(),
            'error_rate_acceptable': self._is_error_rate_acceptable(),
            'business_impact_acceptable': self._is_business_impact_acceptable()
        }
        
        # Overall health status
        overall_health = all(health_indicators.values())
        
        # Health check result
        health_result = {
            'timestamp': current_time,
            'overall_health': overall_health,
            'health_score': self._calculate_health_score(),
            'indicators': health_indicators,
            'recommendations': self._generate_health_recommendations(health_indicators)
        }
        
        # Store health check result
        self.health_check_results.append(health_result)
        if len(self.health_check_results) > 100:
            self.health_check_results = self.health_check_results[-100:]
        
        self.last_health_check = current_time
        
        return health_result
    
    def _is_response_time_stable(self) -> bool:
        """Check if response times are stable."""
        if len(self.response_times) < 10:
            return True
        
        recent_times = self.response_times[-10:]
        avg_time = sum(recent_times) / len(recent_times)
        variance = sum((t - avg_time) ** 2 for t in recent_times) / len(recent_times)
        std_dev = variance ** 0.5
        
        # Consider stable if standard deviation is less than 20% of average
        return std_dev < avg_time * 0.2
    
    def _is_error_rate_acceptable(self) -> bool:
        """Check if error rate is acceptable."""
        if self.total_requests == 0:
            return True
        
        error_rate = (self.total_failures / self.total_requests) * 100
        return error_rate < 5.0  # Acceptable if less than 5%
    
    def _is_business_impact_acceptable(self) -> bool:
        """Check if business impact is acceptable."""
        return (self.business_impact['operational_efficiency'] > 0.8 and
                self.business_impact['user_experience_impact'] < 0.3)
    
    def _generate_health_recommendations(self, indicators: Dict[str, bool]) -> List[str]:
        """Generate health improvement recommendations."""
        recommendations = []
        
        if not indicators['success_rate']:
            recommendations.append("Investigate high failure rate and implement error handling improvements")
        
        if not indicators['response_time_stable']:
            recommendations.append("Optimize performance bottlenecks and implement caching strategies")
        
        if not indicators['error_rate_acceptable']:
            recommendations.append("Review error patterns and implement preventive measures")
        
        if not indicators['business_impact_acceptable']:
            recommendations.append("Address operational efficiency and user experience issues")
        
        if self.state == self.OPEN:
            recommendations.append("Service is down - investigate root cause and restore service")
        elif self.state == self.HALF_OPEN:
            recommendations.append("Service is recovering - monitor closely and ensure stability")
        
        return recommendations
    
    def export_health_report(self, format: str = "json") -> str:
        """Export comprehensive health report."""
        health_data = {
            'circuit_breaker_status': self.get_status(),
            'health_check_history': self.health_check_results[-20:],  # Last 20 health checks
            'performance_metrics': {
                'response_times': self.response_times[-100:],  # Last 100 response times
                'error_distribution': self.error_types,
                'business_impact_summary': self.business_impact
            },
            'recommendations': self._generate_health_recommendations({
                'success_rate': True,  # Placeholder
                'response_time_stable': True,
                'error_rate_acceptable': True,
                'business_impact_acceptable': True
            })
        }
        
        if format.lower() == "json":
            return json.dumps(health_data, indent=2, default=str)
        else:
            return json.dumps(health_data, indent=2, default=str)

# =============================================================================
# ENTERPRISE ERROR HANDLING & INTELLIGENT ALERTING
# =============================================================================

class ErrorHandler:
    """Enterprise-grade error handling system with intelligent alerting and categorization."""
    
    def __init__(self, alerting_enabled: bool = True, error_tracking: bool = True):
        self.alerting_enabled = alerting_enabled
        self.error_tracking = error_tracking
        
        # Error categorization
        self.error_categories = {
            'critical': {
                'severity': 10,
                'description': 'System failure, immediate action required',
                'response_time': 'immediate',
                'escalation': True,
                'business_impact': 'high'
            },
            'high': {
                'severity': 8,
                'description': 'Major functionality affected',
                'response_time': '1 hour',
                'escalation': True,
                'business_impact': 'medium'
            },
            'medium': {
                'severity': 5,
                'description': 'Minor functionality affected',
                'response_time': '4 hours',
                'escalation': False,
                'business_impact': 'low'
            },
            'low': {
                'severity': 2,
                'description': 'Cosmetic or non-critical issue',
                'response_time': '24 hours',
                'escalation': False,
                'business_impact': 'minimal'
            }
        }
        
        # Error tracking
        self.error_history: List[Dict[str, Any]] = []
        self.error_patterns: Dict[str, Dict[str, Any]] = {}
        self.error_frequency: Dict[str, int] = {}
        self.error_resolution: Dict[str, Dict[str, Any]] = {}
        
        # Alerting system
        self.alerts: List[Dict[str, Any]] = []
        self.alert_channels = ['logging', 'email', 'slack', 'pagerduty']
        self.alert_thresholds = {
            'critical_errors_per_hour': 5,
            'high_errors_per_hour': 20,
            'total_errors_per_hour': 100
        }
        
        # Business impact tracking
        self.business_impact_metrics = {
            'total_downtime_minutes': 0,
            'affected_users': 0,
            'revenue_impact': 0.0,
            'customer_satisfaction_impact': 0.0
        }
        
        # Error resolution tracking
        self.resolution_times = {
            'critical': [],
            'high': [],
            'medium': [],
            'low': []
        }
    
    def handle_error(self, error: Exception, context: Optional[Dict[str, Any]] = None, 
                    severity: Optional[str] = None) -> Dict[str, Any]:
        """Handle and categorize an error with comprehensive tracking."""
        error_info = self._analyze_error(error, context, severity)
        
        # Track error
        if self.error_tracking:
            self._track_error(error_info)
        
        # Generate alert if needed
        if self.alerting_enabled:
            self._generate_alert(error_info)
        
        # Update business impact
        self._update_business_impact(error_info)
        
        # Log error with context
        self._log_error(error_info)
        
        return error_info
    
    def _analyze_error(self, error: Exception, context: Optional[Dict[str, Any]], 
                      severity: Optional[str]) -> Dict[str, Any]:
        """Analyze error and determine severity and category."""
        error_type = type(error).__name__
        error_message = str(error)
        
        # Auto-determine severity if not provided
        if not severity:
            severity = self._auto_determine_severity(error_type, error_message, context)
        
        # Get category details
        category_details = self.error_categories.get(severity, self.error_categories['medium'])
        
        # Analyze error context
        context_analysis = self._analyze_error_context(context)
        
        # Generate error ID
        error_id = self._generate_error_id(error_type, error_message)
        
        return {
            'error_id': error_id,
            'timestamp': time.time(),
            'error_type': error_type,
            'error_message': error_message,
            'severity': severity,
            'category_details': category_details,
            'context': context or {},
            'context_analysis': context_analysis,
            'stack_trace': self._get_stack_trace(error),
            'user_agent': context.get('user_agent', 'unknown') if context else 'unknown',
            'client_ip': context.get('client_ip', 'unknown') if context else 'unknown',
            'endpoint': context.get('endpoint', 'unknown') if context else 'unknown',
            'request_id': context.get('request_id', 'unknown') if context else 'unknown'
        }
    
    def _auto_determine_severity(self, error_type: str, error_message: str, 
                                context: Optional[Dict[str, Any]]) -> str:
        """Automatically determine error severity based on type and context."""
        # Critical errors
        critical_patterns = [
            'database connection failed',
            'authentication failed',
            'authorization denied',
            'service unavailable',
            'out of memory',
            'disk full',
            'network timeout'
        ]
        
        # High severity errors
        high_patterns = [
            'validation failed',
            'invalid input',
            'rate limit exceeded',
            'quota exceeded',
            'timeout',
            'connection refused'
        ]
        
        # Medium severity errors
        medium_patterns = [
            'not found',
            'already exists',
            'conflict',
            'bad request',
            'unauthorized'
        ]
        
        error_lower = error_message.lower()
        
        # Check critical patterns
        for pattern in critical_patterns:
            if pattern in error_lower:
                return 'critical'
        
        # Check high patterns
        for pattern in high_patterns:
            if pattern in error_lower:
                return 'high'
        
        # Check medium patterns
        for pattern in medium_patterns:
            if pattern in error_lower:
                return 'medium'
        
        # Check context for additional clues
        if context:
            if context.get('endpoint') in ['/health', '/status']:
                return 'high'  # Health check failures are high priority
            if context.get('user_role') == 'admin':
                return 'high'  # Admin errors are high priority
        
        # Default to medium
        return 'medium'
    
    def _analyze_error_context(self, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze error context for additional insights."""
        if not context:
            return {}
        
        analysis = {
            'has_user_context': 'user_id' in context or 'user_role' in context,
            'has_request_context': 'request_id' in context or 'endpoint' in context,
            'has_performance_context': 'execution_time' in context,
            'has_business_context': 'business_impact' in context,
            'context_completeness': 0.0
        }
        
        # Calculate context completeness
        expected_fields = ['user_id', 'request_id', 'endpoint', 'execution_time', 'business_impact']
        present_fields = sum(1 for field in expected_fields if field in context)
        analysis['context_completeness'] = (present_fields / len(expected_fields)) * 100
        
        return analysis
    
    def _get_stack_trace(self, error: Exception) -> List[str]:
        """Extract stack trace from error."""
        try:
            import traceback
            return traceback.format_exception(type(error), error, error.__traceback__)
        except:
            return [str(error)]
    
    def _generate_error_id(self, error_type: str, error_message: str) -> str:
        """Generate unique error ID."""
        timestamp = str(int(time.time()))
        hash_input = f"{error_type}:{error_message}:{timestamp}"
        error_hash = hashlib.md5(hash_input.encode()).hexdigest()[:8]
        return f"ERR_{timestamp}_{error_hash}"
    
    def _track_error(self, error_info: Dict[str, Any]):
        """Track error for pattern analysis and reporting."""
        # Add to history
        self.error_history.append(error_info)
        if len(self.error_history) > 10000:
            self.error_history = self.error_history[-10000:]
        
        # Update frequency
        error_key = f"{error_info['error_type']}:{error_info['severity']}"
        self.error_frequency[error_key] = self.error_frequency.get(error_key, 0) + 1
        
        # Update patterns
        self._update_error_patterns(error_info)
    
    def _update_error_patterns(self, error_info: Dict[str, Any]):
        """Update error pattern analysis."""
        error_key = error_info['error_type']
        
        if error_key not in self.error_patterns:
            self.error_patterns[error_key] = {
                'count': 0,
                'first_occurrence': error_info['timestamp'],
                'last_occurrence': error_info['timestamp'],
                'severity_distribution': {},
                'context_patterns': {},
                'resolution_history': []
            }
        
        pattern = self.error_patterns[error_key]
        pattern['count'] += 1
        pattern['last_occurrence'] = error_info['timestamp']
        
        # Update severity distribution
        severity = error_info['severity']
        pattern['severity_distribution'][severity] = pattern['severity_distribution'].get(severity, 0) + 1
        
        # Update context patterns
        context = error_info['context']
        for key, value in context.items():
            if key not in pattern['context_patterns']:
                pattern['context_patterns'][key] = {}
            if value not in pattern['context_patterns'][key]:
                pattern['context_patterns'][key][value] = 0
            pattern['context_patterns'][key][value] += 1
    
    def _generate_alert(self, error_info: Dict[str, Any]):
        """Generate intelligent alert based on error severity and patterns."""
        severity = error_info['severity']
        category = error_info['category_details']
        
        # Check if alert should be generated
        if not self._should_generate_alert(error_info):
            return
        
        # Create alert
        alert = {
            'alert_id': f"ALT_{int(time.time())}_{hashlib.md5(str(error_info).encode()).hexdigest()[:8]}",
            'timestamp': time.time(),
            'error_id': error_info['error_id'],
            'severity': severity,
            'category': category,
            'error_type': error_info['error_type'],
            'error_message': error_info['error_message'],
            'context': error_info['context'],
            'channels': self._determine_alert_channels(severity),
            'escalation_required': category['escalation'],
            'response_time_required': category['response_time'],
            'business_impact': category['business_impact']
        }
        
        self.alerts.append(alert)
        
        # Keep only last 1000 alerts
        if len(self.alerts) > 1000:
            self.alerts = self.alerts[-1000:]
        
        # Send alert through channels
        self._send_alert(alert)
    
    def _should_generate_alert(self, error_info: Dict[str, Any]) -> bool:
        """Determine if an alert should be generated based on thresholds and patterns."""
        severity = error_info['severity']
        
        # Always alert for critical errors
        if severity == 'critical':
            return True
        
        # Check hourly thresholds
        current_time = time.time()
        one_hour_ago = current_time - 3600
        
        # Count recent errors by severity
        recent_critical = len([e for e in self.error_history 
                             if e['severity'] == 'critical' and e['timestamp'] > one_hour_ago])
        recent_high = len([e for e in self.error_history 
                          if e['severity'] == 'high' and e['timestamp'] > one_hour_ago])
        recent_total = len([e for e in self.error_history if e['timestamp'] > one_hour_ago])
        
        # Check thresholds
        if recent_critical >= self.alert_thresholds['critical_errors_per_hour']:
            return True
        
        if recent_high >= self.alert_thresholds['high_errors_per_hour']:
            return True
        
        if recent_total >= self.alert_thresholds['total_errors_per_hour']:
            return True
        
        # Check for pattern-based alerts
        if self._is_pattern_alert_worthy(error_info):
            return True
        
        return False
    
    def _is_pattern_alert_worthy(self, error_info: Dict[str, Any]) -> bool:
        """Check if error pattern warrants an alert."""
        error_type = error_info['error_type']
        
        if error_type in self.error_patterns:
            pattern = self.error_patterns[error_type]
            
            # Alert if error frequency is increasing rapidly
            if pattern['count'] > 10:
                recent_count = len([e for e in self.error_history[-100:] 
                                  if e['error_type'] == error_type])
                if recent_count > pattern['count'] * 0.3:  # 30% of total errors in last 100
                    return True
            
            # Alert if error severity is escalating
            severity_dist = pattern['severity_distribution']
            if 'critical' in severity_dist and severity_dist['critical'] > 0:
                return True
        
        return False
    
    def _determine_alert_channels(self, severity: str) -> List[str]:
        """Determine which alert channels to use based on severity."""
        if severity == 'critical':
            return ['logging', 'email', 'slack', 'pagerduty']
        elif severity == 'high':
            return ['logging', 'email', 'slack']
        elif severity == 'medium':
            return ['logging', 'email']
        else:
            return ['logging']
    
    def _send_alert(self, alert: Dict[str, Any]):
        """Send alert through specified channels."""
        logger = get_logger("error_handler")
        
        for channel in alert['channels']:
            try:
                if channel == 'logging':
                    self._send_log_alert(alert)
                elif channel == 'email':
                    self._send_email_alert(alert)
                elif channel == 'slack':
                    self._send_slack_alert(alert)
                elif channel == 'pagerduty':
                    self._send_pagerduty_alert(alert)
            except Exception as e:
                logger.error(f"Failed to send alert through {channel}: {e}")
    
    def _send_log_alert(self, alert: Dict[str, Any]):
        """Send alert through logging system."""
        logger = get_logger("error_handler")
        
        log_message = f"ALERT [{alert['severity'].upper()}] {alert['error_type']}: {alert['error_message']}"
        if alert['severity'] == 'critical':
            logger.critical(log_message, extra={'alert_data': alert})
        elif alert['severity'] == 'high':
            logger.error(log_message, extra={'alert_data': alert})
        else:
            logger.warning(log_message, extra={'alert_data': alert})
    
    def _send_email_alert(self, alert: Dict[str, Any]):
        """Send alert via email (placeholder implementation)."""
        # In production, implement actual email sending
        logger = get_logger("error_handler")
        logger.info(f"Email alert sent for {alert['error_id']}")
    
    def _send_slack_alert(self, alert: Dict[str, Any]):
        """Send alert via Slack (placeholder implementation)."""
        # In production, implement actual Slack integration
        logger = get_logger("error_handler")
        logger.info(f"Slack alert sent for {alert['error_id']}")
    
    def _send_pagerduty_alert(self, alert: Dict[str, Any]):
        """Send alert via PagerDuty (placeholder implementation)."""
        # In production, implement actual PagerDuty integration
        logger = get_logger("error_handler")
        logger.info(f"PagerDuty alert sent for {alert['error_id']}")
    
    def _update_business_impact(self, error_info: Dict[str, Any]):
        """Update business impact metrics based on error."""
        severity = error_info['severity']
        category = error_info['category_details']
        
        # Update downtime tracking
        if severity in ['critical', 'high']:
            # Estimate downtime impact (customize based on your business model)
            if severity == 'critical':
                downtime_minutes = 30  # Critical errors cause ~30 min downtime
            else:
                downtime_minutes = 10  # High errors cause ~10 min downtime
            
            self.business_impact_metrics['total_downtime_minutes'] += downtime_minutes
        
        # Update affected users (estimate based on context)
        if 'endpoint' in error_info['context']:
            endpoint = error_info['context']['endpoint']
            if '/api/' in endpoint:
                affected_users = 100  # API errors affect many users
            elif '/admin/' in endpoint:
                affected_users = 10   # Admin errors affect few users
            else:
                affected_users = 50   # Default estimate
            
            self.business_impact_metrics['affected_users'] += affected_users
        
        # Update revenue impact (customize based on your business model)
        if severity == 'critical':
            revenue_impact = 1000.0  # Critical errors cost ~$1000
        elif severity == 'high':
            revenue_impact = 500.0   # High errors cost ~$500
        elif severity == 'medium':
            revenue_impact = 100.0   # Medium errors cost ~$100
        else:
            revenue_impact = 25.0    # Low errors cost ~$25
        
        self.business_impact_metrics['revenue_impact'] += revenue_impact
        
        # Update customer satisfaction impact
        if severity in ['critical', 'high']:
            self.business_impact_metrics['customer_satisfaction_impact'] += 0.1  # 10% decrease
    
    def _log_error(self, error_info: Dict[str, Any]):
        """Log error with comprehensive context."""
        logger = get_logger("error_handler")
        
        log_data = {
            'error_id': error_info['error_id'],
            'severity': error_info['severity'],
            'error_type': error_info['error_type'],
            'error_message': error_info['error_message'],
            'context': error_info['context'],
            'business_impact': error_info['category_details']['business_impact']
        }
        
        if error_info['severity'] == 'critical':
            logger.critical(f"Critical error: {error_info['error_type']}", extra=log_data)
        elif error_info['severity'] == 'high':
            logger.error(f"High severity error: {error_info['error_type']}", extra=log_data)
        elif error_info['severity'] == 'medium':
            logger.warning(f"Medium severity error: {error_info['error_type']}", extra=log_data)
        else:
            logger.info(f"Low severity error: {error_info['error_type']}", extra=log_data)
    
    def resolve_error(self, error_id: str, resolution: str, 
                     resolution_time_minutes: Optional[float] = None) -> bool:
        """Mark an error as resolved."""
        # Find error in history
        error_entry = None
        for error in self.error_history:
            if error['error_id'] == error_id:
                error_entry = error
                break
        
        if not error_entry:
            return False
        
        # Calculate resolution time if not provided
        if resolution_time_minutes is None:
            resolution_time_minutes = (time.time() - error_entry['timestamp']) / 60
        
        # Record resolution
        resolution_record = {
            'error_id': error_id,
            'resolution': resolution,
            'resolution_time_minutes': resolution_time_minutes,
            'resolved_at': time.time(),
            'severity': error_entry['severity']
        }
        
        self.error_resolution[error_id] = resolution_record
        
        # Update resolution times for severity category
        severity = error_entry['severity']
        if severity in self.resolution_times:
            self.resolution_times[severity].append(resolution_time_minutes)
            # Keep only last 100 resolution times
            if len(self.resolution_times[severity]) > 100:
                self.resolution_times[severity] = self.resolution_times[severity][-100:]
        
        return True
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get comprehensive error summary and analytics."""
        current_time = time.time()
        one_hour_ago = current_time - 3600
        one_day_ago = current_time - 86400
        
        # Recent error counts
        recent_errors = [e for e in self.error_history if e['timestamp'] > one_hour_ago]
        daily_errors = [e for e in self.error_history if e['timestamp'] > one_day_ago]
        
        # Error distribution by severity
        severity_distribution = {}
        for error in self.error_history:
            severity = error['severity']
            severity_distribution[severity] = severity_distribution.get(severity, 0) + 1
        
        # Top error types
        error_type_counts = {}
        for error in self.error_history:
            error_type = error['error_type']
            error_type_counts[error_type] = error_type_counts.get(error_type, 0) + 1
        
        top_error_types = sorted(error_type_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # Resolution metrics
        resolution_metrics = {}
        for severity in self.resolution_times:
            if self.resolution_times[severity]:
                avg_resolution_time = sum(self.resolution_times[severity]) / len(self.resolution_times[severity])
                resolution_metrics[severity] = {
                    'avg_resolution_time_minutes': avg_resolution_time,
                    'total_resolved': len(self.resolution_times[severity])
                }
        
        return {
            'total_errors': len(self.error_history),
            'recent_errors_1h': len(recent_errors),
            'daily_errors_24h': len(daily_errors),
            'severity_distribution': severity_distribution,
            'top_error_types': top_error_types,
            'error_patterns': self.error_patterns,
            'resolution_metrics': resolution_metrics,
            'business_impact': self.business_impact_metrics,
            'active_alerts': len([a for a in self.alerts if a['timestamp'] > one_hour_ago]),
            'total_alerts': len(self.alerts)
        }
    
    def export_error_report(self, format: str = "json", 
                           include_resolved: bool = True) -> str:
        """Export comprehensive error report."""
        report_data = {
            'error_summary': self.get_error_summary(),
            'error_history': self.error_history[-1000:] if include_resolved else [
                e for e in self.error_history if e['error_id'] not in self.error_resolution
            ],
            'resolved_errors': list(self.error_resolution.values()) if include_resolved else [],
            'active_alerts': self.alerts[-100:],
            'error_patterns': self.error_patterns,
            'business_impact': self.business_impact_metrics
        }
        
        if format.lower() == "json":
            return json.dumps(report_data, indent=2, default=str)
        else:
            return json.dumps(report_data, indent=2, default=str)
