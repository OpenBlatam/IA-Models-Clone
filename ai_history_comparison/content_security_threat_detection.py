"""
Content Security & Threat Detection Engine - Advanced Security and Threat Management
=================================================================================

This module provides comprehensive content security capabilities including:
- Advanced threat detection and analysis
- Content security scanning and validation
- Malware and virus detection
- Phishing and social engineering detection
- Data leakage prevention
- Security incident response
- Threat intelligence integration
- Security analytics and reporting
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import uuid
import hashlib
import re
import base64
import requests
from collections import defaultdict, Counter
import redis
import sqlalchemy as sa
from sqlalchemy import create_engine, text
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import yara
import magic
import clamd
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
import jwt
import openai
import anthropic
import boto3
from google.cloud import securitycenter_v1
import virustotal_python
import shodan
import censys
import whois
import dns.resolver
import socket
import ssl
import hashlib
import hmac
import time
import threading
import queue

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ThreatType(Enum):
    """Threat type enumeration"""
    MALWARE = "malware"
    PHISHING = "phishing"
    SPAM = "spam"
    DATA_LEAKAGE = "data_leakage"
    SOCIAL_ENGINEERING = "social_engineering"
    SUSPICIOUS_CONTENT = "suspicious_content"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    INJECTION_ATTACK = "injection_attack"
    XSS = "xss"
    CSRF = "csrf"
    SQL_INJECTION = "sql_injection"
    BRUTE_FORCE = "brute_force"
    DDoS = "ddos"
    RANSOMWARE = "ransomware"
    TROJAN = "trojan"
    VIRUS = "virus"
    SPYWARE = "spyware"
    ADWARE = "adware"
    ROOTKIT = "rootkit"
    BACKDOOR = "backdoor"

class SecurityLevel(Enum):
    """Security level enumeration"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class IncidentStatus(Enum):
    """Incident status enumeration"""
    OPEN = "open"
    INVESTIGATING = "investigating"
    CONTAINED = "contained"
    RESOLVED = "resolved"
    CLOSED = "closed"

@dataclass
class SecurityThreat:
    """Security threat data structure"""
    threat_id: str
    content_id: str
    threat_type: ThreatType
    severity: SecurityLevel
    confidence_score: float
    description: str
    indicators: List[str] = field(default_factory=list)
    mitigation_strategies: List[str] = field(default_factory=list)
    detected_at: datetime = field(default_factory=datetime.utcnow)
    detected_by: str = "system"
    is_false_positive: bool = False
    resolved_at: Optional[datetime] = None

@dataclass
class SecurityIncident:
    """Security incident data structure"""
    incident_id: str
    title: str
    description: str
    threat_type: ThreatType
    severity: SecurityLevel
    status: IncidentStatus
    affected_content: List[str] = field(default_factory=list)
    affected_users: List[str] = field(default_factory=list)
    timeline: List[Dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    assigned_to: str = ""
    resolution_notes: str = ""

@dataclass
class SecurityScan:
    """Security scan data structure"""
    scan_id: str
    content_id: str
    scan_type: str
    results: Dict[str, Any] = field(default_factory=dict)
    threats_detected: List[str] = field(default_factory=list)
    scan_duration: float = 0.0
    scanned_at: datetime = field(default_factory=datetime.utcnow)
    scanner_version: str = ""

@dataclass
class ThreatIntelligence:
    """Threat intelligence data structure"""
    intelligence_id: str
    threat_type: ThreatType
    indicators: List[str] = field(default_factory=list)
    description: str = ""
    source: str = ""
    confidence: float = 0.0
    first_seen: datetime = field(default_factory=datetime.utcnow)
    last_seen: datetime = field(default_factory=datetime.utcnow)
    tags: List[str] = field(default_factory=list)

class ContentSecurityThreatDetection:
    """
    Advanced Content Security & Threat Detection Engine
    
    Provides comprehensive content security and threat detection capabilities
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the Content Security & Threat Detection Engine"""
        self.config = config
        self.security_threats = {}
        self.security_incidents = {}
        self.security_scans = {}
        self.threat_intelligence = {}
        self.redis_client = None
        self.database_engine = None
        
        # Initialize security components
        self._initialize_database()
        self._initialize_redis()
        self._initialize_security_tools()
        self._initialize_threat_intelligence()
        
        # Start background tasks
        self._start_background_tasks()
        
        logger.info("Content Security & Threat Detection Engine initialized successfully")
    
    def _initialize_database(self):
        """Initialize database connection"""
        try:
            if self.config.get("database_url"):
                self.database_engine = create_engine(self.config["database_url"])
                logger.info("Database connection established")
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
    
    def _initialize_redis(self):
        """Initialize Redis connection"""
        try:
            if self.config.get("redis_url"):
                self.redis_client = redis.Redis.from_url(self.config["redis_url"])
                logger.info("Redis connection established")
        except Exception as e:
            logger.error(f"Error initializing Redis: {e}")
    
    def _initialize_security_tools(self):
        """Initialize security tools and scanners"""
        try:
            # Initialize YARA rules engine
            try:
                self.yara_rules = yara.compile(filepath=self.config.get("yara_rules_path", "rules.yar"))
                logger.info("YARA rules engine initialized")
            except Exception as e:
                logger.warning(f"YARA rules not available: {e}")
                self.yara_rules = None
            
            # Initialize ClamAV antivirus
            try:
                self.clamav_client = clamd.ClamdUnixSocket()
                logger.info("ClamAV antivirus initialized")
            except Exception as e:
                logger.warning(f"ClamAV not available: {e}")
                self.clamav_client = None
            
            # Initialize file type detection
            self.magic_detector = magic.Magic(mime=True)
            
            # Initialize ML models for threat detection
            self._initialize_ml_models()
            
            logger.info("Security tools initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing security tools: {e}")
    
    def _initialize_ml_models(self):
        """Initialize ML models for threat detection"""
        try:
            # Initialize anomaly detection model
            self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
            
            # Initialize phishing detection model
            self.phishing_detector = RandomForestClassifier(n_estimators=100, random_state=42)
            
            # Initialize spam detection model
            self.spam_detector = RandomForestClassifier(n_estimators=100, random_state=42)
            
            # Initialize text vectorizer
            self.text_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
            
            logger.info("ML models initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing ML models: {e}")
    
    def _initialize_threat_intelligence(self):
        """Initialize threat intelligence sources"""
        try:
            # Initialize VirusTotal API
            if self.config.get("virustotal_api_key"):
                self.virustotal_client = virustotal_python.Virustotal(self.config["virustotal_api_key"])
                logger.info("VirusTotal API initialized")
            
            # Initialize Shodan API
            if self.config.get("shodan_api_key"):
                self.shodan_client = shodan.Shodan(self.config["shodan_api_key"])
                logger.info("Shodan API initialized")
            
            # Initialize Censys API
            if self.config.get("censys_api_id") and self.config.get("censys_api_secret"):
                self.censys_client = censys.Censys(
                    api_id=self.config["censys_api_id"],
                    api_secret=self.config["censys_api_secret"]
                )
                logger.info("Censys API initialized")
            
            # Initialize Google Cloud Security Command Center
            if self.config.get("gcp_credentials_path"):
                self.gcp_security_client = securitycenter_v1.SecurityCenterClient.from_service_account_file(
                    self.config["gcp_credentials_path"]
                )
                logger.info("Google Cloud Security Command Center initialized")
            
            logger.info("Threat intelligence sources initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing threat intelligence: {e}")
    
    def _start_background_tasks(self):
        """Start background tasks"""
        try:
            # Start threat monitoring task
            asyncio.create_task(self._monitor_threats_periodically())
            
            # Start threat intelligence update task
            asyncio.create_task(self._update_threat_intelligence_periodically())
            
            # Start incident response task
            asyncio.create_task(self._process_incidents_periodically())
            
            # Start security scanning task
            asyncio.create_task(self._scan_content_periodically())
            
            logger.info("Background tasks started")
            
        except Exception as e:
            logger.error(f"Error starting background tasks: {e}")
    
    async def scan_content_security(self, content_id: str, content: str, 
                                  file_path: str = None) -> SecurityScan:
        """Scan content for security threats"""
        try:
            scan_id = str(uuid.uuid4())
            start_time = time.time()
            
            # Initialize scan results
            scan_results = {
                "file_analysis": {},
                "content_analysis": {},
                "network_analysis": {},
                "behavioral_analysis": {},
                "threat_intelligence": {}
            }
            
            threats_detected = []
            
            # File analysis
            if file_path:
                file_results = await self._analyze_file_security(file_path)
                scan_results["file_analysis"] = file_results
                threats_detected.extend(file_results.get("threats", []))
            
            # Content analysis
            content_results = await self._analyze_content_security(content)
            scan_results["content_analysis"] = content_results
            threats_detected.extend(content_results.get("threats", []))
            
            # Network analysis
            network_results = await self._analyze_network_security(content)
            scan_results["network_analysis"] = network_results
            threats_detected.extend(network_results.get("threats", []))
            
            # Behavioral analysis
            behavioral_results = await self._analyze_behavioral_security(content)
            scan_results["behavioral_analysis"] = behavioral_results
            threats_detected.extend(behavioral_results.get("threats", []))
            
            # Threat intelligence check
            intelligence_results = await self._check_threat_intelligence(content)
            scan_results["threat_intelligence"] = intelligence_results
            threats_detected.extend(intelligence_results.get("threats", []))
            
            # Calculate scan duration
            scan_duration = time.time() - start_time
            
            # Create security scan
            security_scan = SecurityScan(
                scan_id=scan_id,
                content_id=content_id,
                scan_type="comprehensive",
                results=scan_results,
                threats_detected=threats_detected,
                scan_duration=scan_duration,
                scanner_version="1.0.0"
            )
            
            # Store scan results
            self.security_scans[scan_id] = security_scan
            
            # Create security threats if any detected
            for threat in threats_detected:
                await self._create_security_threat(content_id, threat)
            
            logger.info(f"Security scan completed for content {content_id}: {len(threats_detected)} threats detected")
            
            return security_scan
            
        except Exception as e:
            logger.error(f"Error scanning content security: {e}")
            raise
    
    async def _analyze_file_security(self, file_path: str) -> Dict[str, Any]:
        """Analyze file for security threats"""
        try:
            results = {
                "file_type": "",
                "file_size": 0,
                "checksum": "",
                "threats": [],
                "scan_results": {}
            }
            
            # Get file information
            import os
            results["file_size"] = os.path.getsize(file_path)
            results["file_type"] = self.magic_detector.from_file(file_path)
            
            # Calculate file checksum
            with open(file_path, 'rb') as f:
                file_content = f.read()
                results["checksum"] = hashlib.sha256(file_content).hexdigest()
            
            # YARA rules scanning
            if self.yara_rules:
                try:
                    yara_matches = self.yara_rules.match(file_path)
                    if yara_matches:
                        for match in yara_matches:
                            results["threats"].append({
                                "type": "yara_match",
                                "rule": match.rule,
                                "description": match.meta.get("description", ""),
                                "severity": match.meta.get("severity", "medium")
                            })
                except Exception as e:
                    logger.warning(f"YARA scanning failed: {e}")
            
            # ClamAV antivirus scanning
            if self.clamav_client:
                try:
                    clamav_result = self.clamav_client.scan(file_path)
                    if clamav_result:
                        for file_path_result, (status, virus_name) in clamav_result.items():
                            if status == "FOUND":
                                results["threats"].append({
                                    "type": "malware",
                                    "virus_name": virus_name,
                                    "severity": "high"
                                })
                except Exception as e:
                    logger.warning(f"ClamAV scanning failed: {e}")
            
            # File type validation
            if not self._is_safe_file_type(results["file_type"]):
                results["threats"].append({
                    "type": "suspicious_file_type",
                    "file_type": results["file_type"],
                    "severity": "medium"
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error analyzing file security: {e}")
            return {"threats": []}
    
    async def _analyze_content_security(self, content: str) -> Dict[str, Any]:
        """Analyze content for security threats"""
        try:
            results = {
                "threats": [],
                "analysis_results": {}
            }
            
            # Phishing detection
            phishing_result = await self._detect_phishing(content)
            if phishing_result["is_phishing"]:
                results["threats"].append({
                    "type": "phishing",
                    "confidence": phishing_result["confidence"],
                    "severity": "high"
                })
            
            # Spam detection
            spam_result = await self._detect_spam(content)
            if spam_result["is_spam"]:
                results["threats"].append({
                    "type": "spam",
                    "confidence": spam_result["confidence"],
                    "severity": "medium"
                })
            
            # Suspicious patterns detection
            suspicious_patterns = await self._detect_suspicious_patterns(content)
            results["threats"].extend(suspicious_patterns)
            
            # Data leakage detection
            data_leakage = await self._detect_data_leakage(content)
            if data_leakage["has_leakage"]:
                results["threats"].append({
                    "type": "data_leakage",
                    "leaked_data": data_leakage["leaked_data"],
                    "severity": "high"
                })
            
            # Social engineering detection
            social_engineering = await self._detect_social_engineering(content)
            if social_engineering["is_social_engineering"]:
                results["threats"].append({
                    "type": "social_engineering",
                    "confidence": social_engineering["confidence"],
                    "severity": "high"
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error analyzing content security: {e}")
            return {"threats": []}
    
    async def _analyze_network_security(self, content: str) -> Dict[str, Any]:
        """Analyze network-related security threats"""
        try:
            results = {
                "threats": [],
                "urls_analyzed": [],
                "domains_analyzed": []
            }
            
            # Extract URLs and domains
            urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', content)
            domains = re.findall(r'(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+[a-zA-Z]{2,}', content)
            
            # Analyze URLs
            for url in urls:
                url_analysis = await self._analyze_url_security(url)
                results["urls_analyzed"].append({
                    "url": url,
                    "analysis": url_analysis
                })
                
                if url_analysis["is_malicious"]:
                    results["threats"].append({
                        "type": "malicious_url",
                        "url": url,
                        "severity": "high"
                    })
            
            # Analyze domains
            for domain in domains:
                domain_analysis = await self._analyze_domain_security(domain)
                results["domains_analyzed"].append({
                    "domain": domain,
                    "analysis": domain_analysis
                })
                
                if domain_analysis["is_suspicious"]:
                    results["threats"].append({
                        "type": "suspicious_domain",
                        "domain": domain,
                        "severity": "medium"
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"Error analyzing network security: {e}")
            return {"threats": []}
    
    async def _analyze_behavioral_security(self, content: str) -> Dict[str, Any]:
        """Analyze behavioral security threats"""
        try:
            results = {
                "threats": [],
                "behavioral_indicators": []
            }
            
            # Detect injection attempts
            injection_patterns = [
                r'<script[^>]*>.*?</script>',  # XSS
                r'union\s+select',  # SQL injection
                r'drop\s+table',  # SQL injection
                r'javascript:',  # JavaScript injection
                r'vbscript:',  # VBScript injection
                r'onload\s*=',  # Event handler injection
                r'onerror\s*=',  # Event handler injection
            ]
            
            for pattern in injection_patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    results["threats"].append({
                        "type": "injection_attempt",
                        "pattern": pattern,
                        "severity": "high"
                    })
            
            # Detect suspicious commands
            suspicious_commands = [
                r'cmd\.exe',
                r'powershell',
                r'bash',
                r'sh\s+',
                r'wget\s+',
                r'curl\s+',
                r'nc\s+',
                r'telnet\s+'
            ]
            
            for command in suspicious_commands:
                if re.search(command, content, re.IGNORECASE):
                    results["threats"].append({
                        "type": "suspicious_command",
                        "command": command,
                        "severity": "medium"
                    })
            
            # Detect obfuscation attempts
            obfuscation_patterns = [
                r'\\x[0-9a-fA-F]{2}',  # Hex encoding
                r'%[0-9a-fA-F]{2}',  # URL encoding
                r'&#x[0-9a-fA-F]+;',  # HTML entity encoding
                r'base64',  # Base64 encoding
                r'eval\s*\(',  # Eval function
                r'String\.fromCharCode',  # Character code obfuscation
            ]
            
            for pattern in obfuscation_patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    results["threats"].append({
                        "type": "obfuscation_attempt",
                        "pattern": pattern,
                        "severity": "medium"
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"Error analyzing behavioral security: {e}")
            return {"threats": []}
    
    async def _check_threat_intelligence(self, content: str) -> Dict[str, Any]:
        """Check content against threat intelligence"""
        try:
            results = {
                "threats": [],
                "intelligence_matches": []
            }
            
            # Extract indicators from content
            indicators = self._extract_indicators(content)
            
            # Check against threat intelligence
            for indicator in indicators:
                intelligence_match = await self._check_indicator_intelligence(indicator)
                if intelligence_match["is_malicious"]:
                    results["threats"].append({
                        "type": "threat_intelligence_match",
                        "indicator": indicator,
                        "intelligence": intelligence_match,
                        "severity": "high"
                    })
                
                results["intelligence_matches"].append({
                    "indicator": indicator,
                    "match": intelligence_match
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error checking threat intelligence: {e}")
            return {"threats": []}
    
    def _extract_indicators(self, content: str) -> List[str]:
        """Extract threat indicators from content"""
        indicators = []
        
        # Extract IP addresses
        ip_pattern = r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'
        indicators.extend(re.findall(ip_pattern, content))
        
        # Extract domains
        domain_pattern = r'(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+[a-zA-Z]{2,}'
        indicators.extend(re.findall(domain_pattern, content))
        
        # Extract URLs
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        indicators.extend(re.findall(url_pattern, content))
        
        # Extract email addresses
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        indicators.extend(re.findall(email_pattern, content))
        
        # Extract file hashes (MD5, SHA1, SHA256)
        hash_patterns = [
            r'\b[a-fA-F0-9]{32}\b',  # MD5
            r'\b[a-fA-F0-9]{40}\b',  # SHA1
            r'\b[a-fA-F0-9]{64}\b'   # SHA256
        ]
        
        for pattern in hash_patterns:
            indicators.extend(re.findall(pattern, content))
        
        return list(set(indicators))  # Remove duplicates
    
    async def _check_indicator_intelligence(self, indicator: str) -> Dict[str, Any]:
        """Check indicator against threat intelligence"""
        try:
            result = {
                "is_malicious": False,
                "confidence": 0.0,
                "sources": [],
                "threat_types": []
            }
            
            # Check VirusTotal
            if hasattr(self, 'virustotal_client'):
                try:
                    vt_result = self.virustotal_client.file_report(indicator)
                    if vt_result.get("positives", 0) > 0:
                        result["is_malicious"] = True
                        result["confidence"] = vt_result["positives"] / vt_result["total"]
                        result["sources"].append("VirusTotal")
                        result["threat_types"].append("malware")
                except Exception as e:
                    logger.warning(f"VirusTotal check failed: {e}")
            
            # Check Shodan
            if hasattr(self, 'shodan_client') and self._is_ip_address(indicator):
                try:
                    shodan_result = self.shodan_client.host(indicator)
                    if shodan_result.get("vulns"):
                        result["is_malicious"] = True
                        result["confidence"] = 0.8
                        result["sources"].append("Shodan")
                        result["threat_types"].append("vulnerable_host")
                except Exception as e:
                    logger.warning(f"Shodan check failed: {e}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error checking indicator intelligence: {e}")
            return {"is_malicious": False, "confidence": 0.0, "sources": [], "threat_types": []}
    
    def _is_ip_address(self, indicator: str) -> bool:
        """Check if indicator is an IP address"""
        try:
            socket.inet_aton(indicator)
            return True
        except socket.error:
            return False
    
    async def _detect_phishing(self, content: str) -> Dict[str, Any]:
        """Detect phishing attempts in content"""
        try:
            # Phishing indicators
            phishing_indicators = [
                r'urgent\s+action\s+required',
                r'click\s+here\s+immediately',
                r'verify\s+your\s+account',
                r'suspended\s+account',
                r'limited\s+time\s+offer',
                r'congratulations\s+you\s+won',
                r'free\s+money',
                r'act\s+now',
                r'don\'t\s+miss\s+out',
                r'exclusive\s+offer'
            ]
            
            phishing_score = 0
            for indicator in phishing_indicators:
                if re.search(indicator, content, re.IGNORECASE):
                    phishing_score += 1
            
            # Check for suspicious URLs
            urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', content)
            suspicious_urls = 0
            
            for url in urls:
                if self._is_suspicious_url(url):
                    suspicious_urls += 1
            
            # Calculate confidence
            confidence = min(1.0, (phishing_score * 0.1) + (suspicious_urls * 0.2))
            
            return {
                "is_phishing": confidence > 0.3,
                "confidence": confidence,
                "indicators_found": phishing_score,
                "suspicious_urls": suspicious_urls
            }
            
        except Exception as e:
            logger.error(f"Error detecting phishing: {e}")
            return {"is_phishing": False, "confidence": 0.0}
    
    def _is_suspicious_url(self, url: str) -> bool:
        """Check if URL is suspicious"""
        try:
            # Check for URL shortening services
            shorteners = ['bit.ly', 'tinyurl.com', 'goo.gl', 't.co', 'ow.ly']
            for shortener in shorteners:
                if shortener in url:
                    return True
            
            # Check for suspicious domains
            suspicious_domains = ['free-money.com', 'win-prize.com', 'urgent-action.com']
            for domain in suspicious_domains:
                if domain in url:
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking suspicious URL: {e}")
            return False
    
    async def _detect_spam(self, content: str) -> Dict[str, Any]:
        """Detect spam content"""
        try:
            # Spam indicators
            spam_indicators = [
                r'buy\s+now',
                r'click\s+here',
                r'free\s+trial',
                r'no\s+obligation',
                r'limited\s+time',
                r'act\s+now',
                r'don\'t\s+wait',
                r'guaranteed\s+results',
                r'make\s+money\s+fast',
                r'work\s+from\s+home'
            ]
            
            spam_score = 0
            for indicator in spam_indicators:
                if re.search(indicator, content, re.IGNORECASE):
                    spam_score += 1
            
            # Check for excessive capitalization
            caps_ratio = sum(1 for c in content if c.isupper()) / len(content) if content else 0
            if caps_ratio > 0.3:
                spam_score += 2
            
            # Check for excessive punctuation
            punctuation_ratio = sum(1 for c in content if c in '!?') / len(content) if content else 0
            if punctuation_ratio > 0.1:
                spam_score += 1
            
            # Calculate confidence
            confidence = min(1.0, spam_score * 0.1)
            
            return {
                "is_spam": confidence > 0.3,
                "confidence": confidence,
                "indicators_found": spam_score
            }
            
        except Exception as e:
            logger.error(f"Error detecting spam: {e}")
            return {"is_spam": False, "confidence": 0.0}
    
    async def _detect_suspicious_patterns(self, content: str) -> List[Dict[str, Any]]:
        """Detect suspicious patterns in content"""
        try:
            threats = []
            
            # Detect potential SQL injection
            sql_patterns = [
                r'union\s+select',
                r'drop\s+table',
                r'delete\s+from',
                r'insert\s+into',
                r'update\s+set',
                r'exec\s*\(',
                r'xp_cmdshell'
            ]
            
            for pattern in sql_patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    threats.append({
                        "type": "sql_injection",
                        "pattern": pattern,
                        "severity": "high"
                    })
            
            # Detect potential XSS
            xss_patterns = [
                r'<script[^>]*>',
                r'javascript:',
                r'vbscript:',
                r'onload\s*=',
                r'onerror\s*=',
                r'onclick\s*='
            ]
            
            for pattern in xss_patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    threats.append({
                        "type": "xss",
                        "pattern": pattern,
                        "severity": "high"
                    })
            
            return threats
            
        except Exception as e:
            logger.error(f"Error detecting suspicious patterns: {e}")
            return []
    
    async def _detect_data_leakage(self, content: str) -> Dict[str, Any]:
        """Detect potential data leakage"""
        try:
            leaked_data = []
            
            # Detect credit card numbers
            cc_pattern = r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b'
            if re.search(cc_pattern, content):
                leaked_data.append("credit_card")
            
            # Detect SSN
            ssn_pattern = r'\b\d{3}-\d{2}-\d{4}\b'
            if re.search(ssn_pattern, content):
                leaked_data.append("ssn")
            
            # Detect email addresses
            email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            emails = re.findall(email_pattern, content)
            if len(emails) > 10:  # Suspicious number of emails
                leaked_data.append("email_list")
            
            # Detect phone numbers
            phone_pattern = r'\b\d{3}-\d{3}-\d{4}\b'
            phones = re.findall(phone_pattern, content)
            if len(phones) > 5:  # Suspicious number of phone numbers
                leaked_data.append("phone_list")
            
            return {
                "has_leakage": len(leaked_data) > 0,
                "leaked_data": leaked_data
            }
            
        except Exception as e:
            logger.error(f"Error detecting data leakage: {e}")
            return {"has_leakage": False, "leaked_data": []}
    
    async def _detect_social_engineering(self, content: str) -> Dict[str, Any]:
        """Detect social engineering attempts"""
        try:
            # Social engineering indicators
            se_indicators = [
                r'urgent\s+security\s+alert',
                r'your\s+account\s+has\s+been\s+compromised',
                r'verify\s+your\s+identity',
                r'confirm\s+your\s+password',
                r'update\s+your\s+information',
                r'security\s+breach',
                r'immediate\s+action\s+required',
                r'click\s+to\s+verify',
                r'confirm\s+your\s+email',
                r'validate\s+your\s+account'
            ]
            
            se_score = 0
            for indicator in se_indicators:
                if re.search(indicator, content, re.IGNORECASE):
                    se_score += 1
            
            # Check for authority impersonation
            authority_keywords = ['fbi', 'cia', 'irs', 'bank', 'paypal', 'amazon', 'microsoft', 'google']
            authority_mentions = sum(1 for keyword in authority_keywords if keyword.lower() in content.lower())
            
            if authority_mentions > 0:
                se_score += authority_mentions
            
            # Calculate confidence
            confidence = min(1.0, se_score * 0.15)
            
            return {
                "is_social_engineering": confidence > 0.3,
                "confidence": confidence,
                "indicators_found": se_score
            }
            
        except Exception as e:
            logger.error(f"Error detecting social engineering: {e}")
            return {"is_social_engineering": False, "confidence": 0.0}
    
    async def _analyze_url_security(self, url: str) -> Dict[str, Any]:
        """Analyze URL for security threats"""
        try:
            result = {
                "is_malicious": False,
                "threats": [],
                "analysis": {}
            }
            
            # Check URL structure
            if self._is_suspicious_url(url):
                result["is_malicious"] = True
                result["threats"].append("suspicious_url_structure")
            
            # Check domain reputation
            domain = self._extract_domain(url)
            if domain:
                domain_reputation = await self._check_domain_reputation(domain)
                if domain_reputation["is_malicious"]:
                    result["is_malicious"] = True
                    result["threats"].extend(domain_reputation["threats"])
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing URL security: {e}")
            return {"is_malicious": False, "threats": [], "analysis": {}}
    
    async def _analyze_domain_security(self, domain: str) -> Dict[str, Any]:
        """Analyze domain for security threats"""
        try:
            result = {
                "is_suspicious": False,
                "threats": [],
                "analysis": {}
            }
            
            # Check domain age
            try:
                domain_info = whois.whois(domain)
                if domain_info.creation_date:
                    domain_age = (datetime.now() - domain_info.creation_date).days
                    if domain_age < 30:  # Very new domain
                        result["is_suspicious"] = True
                        result["threats"].append("new_domain")
            except Exception as e:
                logger.warning(f"Domain age check failed: {e}")
            
            # Check for suspicious keywords
            suspicious_keywords = ['free', 'win', 'prize', 'money', 'urgent', 'click']
            if any(keyword in domain.lower() for keyword in suspicious_keywords):
                result["is_suspicious"] = True
                result["threats"].append("suspicious_keywords")
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing domain security: {e}")
            return {"is_suspicious": False, "threats": [], "analysis": {}}
    
    def _extract_domain(self, url: str) -> Optional[str]:
        """Extract domain from URL"""
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            return parsed.netloc
        except Exception:
            return None
    
    async def _check_domain_reputation(self, domain: str) -> Dict[str, Any]:
        """Check domain reputation"""
        try:
            result = {
                "is_malicious": False,
                "threats": []
            }
            
            # Check against known malicious domains
            malicious_domains = [
                'malware.com', 'phishing.com', 'spam.com',
                'virus.com', 'trojan.com', 'backdoor.com'
            ]
            
            if domain.lower() in malicious_domains:
                result["is_malicious"] = True
                result["threats"].append("known_malicious_domain")
            
            return result
            
        except Exception as e:
            logger.error(f"Error checking domain reputation: {e}")
            return {"is_malicious": False, "threats": []}
    
    def _is_safe_file_type(self, file_type: str) -> bool:
        """Check if file type is safe"""
        safe_types = [
            'text/plain',
            'text/html',
            'text/css',
            'text/javascript',
            'application/json',
            'application/xml',
            'image/jpeg',
            'image/png',
            'image/gif',
            'application/pdf'
        ]
        
        return file_type in safe_types
    
    async def _create_security_threat(self, content_id: str, threat_data: Dict[str, Any]) -> SecurityThreat:
        """Create security threat"""
        try:
            threat_id = str(uuid.uuid4())
            
            # Determine threat type and severity
            threat_type = ThreatType(threat_data.get("type", "suspicious_content"))
            severity = SecurityLevel(threat_data.get("severity", "medium"))
            
            threat = SecurityThreat(
                threat_id=threat_id,
                content_id=content_id,
                threat_type=threat_type,
                severity=severity,
                confidence_score=threat_data.get("confidence", 0.5),
                description=threat_data.get("description", f"Detected {threat_type.value}"),
                indicators=threat_data.get("indicators", []),
                mitigation_strategies=threat_data.get("mitigation_strategies", [])
            )
            
            # Store threat
            self.security_threats[threat_id] = threat
            
            # Create incident if high severity
            if severity in [SecurityLevel.HIGH, SecurityLevel.CRITICAL]:
                await self._create_security_incident(threat)
            
            logger.info(f"Security threat {threat_id} created for content {content_id}")
            
            return threat
            
        except Exception as e:
            logger.error(f"Error creating security threat: {e}")
            raise
    
    async def _create_security_incident(self, threat: SecurityThreat) -> SecurityIncident:
        """Create security incident"""
        try:
            incident_id = str(uuid.uuid4())
            
            incident = SecurityIncident(
                incident_id=incident_id,
                title=f"Security Threat: {threat.threat_type.value}",
                description=threat.description,
                threat_type=threat.threat_type,
                severity=threat.severity,
                status=IncidentStatus.OPEN,
                affected_content=[threat.content_id],
                timeline=[{
                    "timestamp": datetime.utcnow().isoformat(),
                    "event": "threat_detected",
                    "description": f"Threat {threat.threat_id} detected"
                }]
            )
            
            # Store incident
            self.security_incidents[incident_id] = incident
            
            logger.info(f"Security incident {incident_id} created for threat {threat.threat_id}")
            
            return incident
            
        except Exception as e:
            logger.error(f"Error creating security incident: {e}")
            raise
    
    async def _monitor_threats_periodically(self):
        """Monitor threats periodically"""
        while True:
            try:
                await asyncio.sleep(3600)  # Monitor every hour
                
                # In production, this would monitor for new threats
                logger.info("Threat monitoring completed")
                
            except Exception as e:
                logger.error(f"Error in threat monitoring: {e}")
                await asyncio.sleep(3600)
    
    async def _update_threat_intelligence_periodically(self):
        """Update threat intelligence periodically"""
        while True:
            try:
                await asyncio.sleep(86400)  # Update daily
                
                # In production, this would update threat intelligence feeds
                logger.info("Threat intelligence update completed")
                
            except Exception as e:
                logger.error(f"Error updating threat intelligence: {e}")
                await asyncio.sleep(86400)
    
    async def _process_incidents_periodically(self):
        """Process incidents periodically"""
        while True:
            try:
                await asyncio.sleep(1800)  # Process every 30 minutes
                
                # In production, this would process security incidents
                logger.info("Incident processing completed")
                
            except Exception as e:
                logger.error(f"Error processing incidents: {e}")
                await asyncio.sleep(1800)
    
    async def _scan_content_periodically(self):
        """Scan content periodically"""
        while True:
            try:
                await asyncio.sleep(7200)  # Scan every 2 hours
                
                # In production, this would scan content for threats
                logger.info("Content scanning completed")
                
            except Exception as e:
                logger.error(f"Error scanning content: {e}")
                await asyncio.sleep(7200)

# Example usage and testing
async def main():
    """Example usage of the Content Security & Threat Detection Engine"""
    try:
        # Initialize engine
        config = {
            "database_url": "postgresql://user:password@localhost/securitydb",
            "redis_url": "redis://localhost:6379",
            "virustotal_api_key": "your-virustotal-key",
            "shodan_api_key": "your-shodan-key"
        }
        
        engine = ContentSecurityThreatDetection(config)
        
        # Test content with potential threats
        test_content = """
        URGENT ACTION REQUIRED! Your account has been compromised.
        Click here immediately to verify your identity: http://suspicious-site.com/verify
        <script>alert('XSS attack')</script>
        Credit card: 1234-5678-9012-3456
        """
        
        # Scan content for security threats
        print("Scanning content for security threats...")
        security_scan = await engine.scan_content_security("content_001", test_content)
        
        print(f"Security scan completed: {security_scan.scan_id}")
        print(f"Threats detected: {len(security_scan.threats_detected)}")
        
        # Display scan results
        for threat_type, results in security_scan.results.items():
            if results.get("threats"):
                print(f"\n{threat_type.upper()} THREATS:")
                for threat in results["threats"]:
                    print(f"  - {threat['type']}: {threat.get('description', 'No description')}")
                    print(f"    Severity: {threat.get('severity', 'unknown')}")
        
        # Display security threats
        print(f"\nSECURITY THREATS CREATED: {len(engine.security_threats)}")
        for threat_id, threat in engine.security_threats.items():
            print(f"  - {threat.threat_type.value}: {threat.description}")
            print(f"    Severity: {threat.severity.value}")
            print(f"    Confidence: {threat.confidence_score:.2f}")
        
        # Display security incidents
        print(f"\nSECURITY INCIDENTS CREATED: {len(engine.security_incidents)}")
        for incident_id, incident in engine.security_incidents.items():
            print(f"  - {incident.title}")
            print(f"    Status: {incident.status.value}")
            print(f"    Severity: {incident.severity.value}")
        
        print("\nContent Security & Threat Detection Engine demo completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in main demo: {e}")

if __name__ == "__main__":
    asyncio.run(main())
























