#!/usr/bin/env python3
"""
🔒 HeyGen AI - Advanced Security System
======================================

This module provides comprehensive security features for the HeyGen AI system:
- Advanced threat detection and prevention
- Multi-layer security validation
- Real-time security monitoring
- Automated security response
- Compliance and audit features
"""

import asyncio
import hashlib
import hmac
import secrets
import time
import logging
import re
import json
import ipaddress
from typing import Dict, List, Optional, Any, Union, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import threading
import queue
from collections import defaultdict, deque
import socket
import ssl
import requests
from urllib.parse import urlparse
import jwt
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ThreatLevel(str, Enum):
    """Threat severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class SecurityEventType(str, Enum):
    """Security event types"""
    LOGIN_ATTEMPT = "login_attempt"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    THREAT_DETECTED = "threat_detected"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    INVALID_INPUT = "invalid_input"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    DATA_BREACH_ATTEMPT = "data_breach_attempt"

class SecurityAction(str, Enum):
    """Security response actions"""
    ALLOW = "allow"
    BLOCK = "block"
    CHALLENGE = "challenge"
    ALERT = "alert"
    QUARANTINE = "quarantine"

@dataclass
class SecurityEvent:
    """Security event record"""
    event_id: str
    event_type: SecurityEventType
    threat_level: ThreatLevel
    source_ip: str
    user_agent: str
    timestamp: datetime
    details: Dict[str, Any]
    action_taken: SecurityAction
    resolved: bool = False
    resolution_time: Optional[datetime] = None

@dataclass
class SecurityMetrics:
    """Security monitoring metrics"""
    total_events: int = 0
    blocked_requests: int = 0
    suspicious_activities: int = 0
    threat_detections: int = 0
    security_score: float = 100.0
    last_scan: datetime = field(default_factory=datetime.now)
    active_threats: int = 0

class ThreatDetector:
    """Advanced threat detection system"""
    
    def __init__(self):
        self.threat_patterns = self._load_threat_patterns()
        self.behavioral_patterns = self._load_behavioral_patterns()
        self.geo_blocklist = self._load_geo_blocklist()
        self.ip_reputation = {}
        self.user_behavior = defaultdict(list)
    
    def _load_threat_patterns(self) -> Dict[str, List[str]]:
        """Load comprehensive threat detection patterns"""
        return {
            'sql_injection': [
                r'(\b(union|select|insert|update|delete|drop|create|alter|exec|execute)\b)',
                r'(\b(or|and)\b\s+\d+\s*[=<>])',
                r'(\b(declare|cast|convert|waitfor|delay)\b)',
                r'(\b(load_file|into\s+outfile|into\s+dumpfile)\b)',
                r'(\b(version|user|database|schema)\b\s*\(\s*\))',
                r'(\b(concat|char|ascii|substring|length)\b)',
                r'(\b(if|case|when|then|else)\b)',
                r'(\b(like|regexp|rlike)\b)',
                r'(\b(limit|offset|order\s+by|group\s+by)\b)',
                r'(\b(having|where|from|join)\b)'
            ],
            'xss_attacks': [
                r'<script[^>]*>.*?</script>',
                r'javascript:',
                r'on\w+\s*=',
                r'<iframe[^>]*>',
                r'<object[^>]*>',
                r'<embed[^>]*>',
                r'<form[^>]*>',
                r'<input[^>]*>',
                r'<link[^>]*>',
                r'<meta[^>]*>',
                r'<style[^>]*>.*?</style>',
                r'expression\s*\(',
                r'url\s*\(',
                r'@import',
                r'behavior\s*:'
            ],
            'command_injection': [
                r'(\b(cmd|command|powershell|bash|sh|zsh|fish)\b)',
                r'(\b(system|eval|exec|execute|shell_exec|passthru)\b)',
                r'(\b(rm|del|format|fdisk|mkfs|dd|shutdown|reboot)\b)',
                r'(\b(net|netstat|ipconfig|ifconfig|ping|traceroute)\b)',
                r'(\b(wget|curl|nc|telnet|ssh|ftp|sftp)\b)',
                r'(\b(cat|type|more|less|head|tail|grep|find)\b)',
                r'(\b(ps|top|htop|kill|killall|pkill)\b)',
                r'(\b(chmod|chown|chgrp|umask)\b)',
                r'(\b(sudo|su|runas|doas)\b)',
                r'(\b(echo|print|printf|sprintf)\b)'
            ],
            'path_traversal': [
                r'\.\./',
                r'\.\.\\',
                r'%2e%2e%2f',
                r'%2e%2e%5c',
                r'%252e%252e%252f',
                r'%252e%252e%255c',
                r'\.\.%2f',
                r'\.\.%5c',
                r'\.\.%252f',
                r'\.\.%255c',
                r'\.\.%c0%af',
                r'\.\.%c1%9c',
                r'\.\.%c0%2f',
                r'\.\.%c1%af'
            ],
            'ldap_injection': [
                r'(\b(ldap|ldaps)\b)',
                r'(\b(cn|ou|dc|uid|mail|sn|givenName)\b)',
                r'(\b(filter|base|scope|attributes)\b)',
                r'(\b(and|or|not)\b)',
                r'(\b(equals|contains|startsWith|endsWith)\b)',
                r'(\b(>=|<=|>|<|~=)\b)',
                r'(\b(extensibleMatch|approximateMatch)\b)',
                r'(\b(substring|present|extensible)\b)'
            ],
            'xml_injection': [
                r'<!DOCTYPE\s+[^>]*>',
                r'<!ENTITY\s+[^>]*>',
                r'<![CDATA\[.*?\]\]>',
                r'<script[^>]*>.*?</script>',
                r'<iframe[^>]*>',
                r'<object[^>]*>',
                r'<embed[^>]*>',
                r'<link[^>]*>',
                r'<meta[^>]*>',
                r'<style[^>]*>.*?</style>'
            ],
            'ssrf_attacks': [
                r'file://',
                r'gopher://',
                r'ldap://',
                r'ldaps://',
                r'ftp://',
                r'ftps://',
                r'tftp://',
                r'sftp://',
                r'data://',
                r'javascript://',
                r'vbscript://',
                r'jar://',
                r'phar://',
                r'zip://',
                r'rar://',
                r'7z://',
                r'tar://',
                r'gz://',
                r'bz2://',
                r'xz://'
            ],
            'no_sql_injection': [
                r'(\b(mongo|mongodb|nosql|document)\b)',
                r'(\b(find|findOne|findAndModify|update|updateOne|updateMany)\b)',
                r'(\b(insert|insertOne|insertMany|remove|deleteOne|deleteMany)\b)',
                r'(\b(aggregate|distinct|count|limit|skip|sort)\b)',
                r'(\b(gt|gte|lt|lte|ne|in|nin|exists|regex)\b)',
                r'(\b(and|or|not|nor|all|elemMatch)\b)',
                r'(\b(size|type|where|mod|bitsAllSet|bitsAnySet)\b)',
                r'(\b(geoWithin|geoIntersects|geoNear|near|nearSphere)\b)'
            ]
        }
    
    def _load_behavioral_patterns(self) -> Dict[str, Any]:
        """Load behavioral analysis patterns"""
        return {
            'rapid_requests': {
                'threshold': 100,  # requests per minute
                'time_window': 60
            },
            'unusual_hours': {
                'start_hour': 6,
                'end_hour': 22
            },
            'suspicious_user_agents': [
                'sqlmap', 'nmap', 'nikto', 'dirb', 'gobuster',
                'burp', 'zap', 'w3af', 'acunetix', 'nessus',
                'scanner', 'bot', 'crawler', 'spider'
            ],
            'suspicious_paths': [
                '/admin', '/administrator', '/wp-admin', '/phpmyadmin',
                '/.env', '/config', '/backup', '/test', '/debug',
                '/api/v1/admin', '/api/v2/admin', '/api/admin'
            ]
        }
    
    def _load_geo_blocklist(self) -> Set[str]:
        """Load geographic blocklist (country codes)"""
        # This would typically load from a database or file
        return {'CN', 'RU', 'KP', 'IR'}  # Example blocklist
    
    def detect_threats(self, 
                      input_data: str, 
                      source_ip: str, 
                      user_agent: str,
                      request_path: str = "",
                      user_id: Optional[str] = None) -> Dict[str, Any]:
        """Comprehensive threat detection"""
        threats_detected = []
        threat_level = ThreatLevel.LOW
        
        # Pattern-based detection
        for category, patterns in self.threat_patterns.items():
            for pattern in patterns:
                if re.search(pattern, input_data, re.IGNORECASE):
                    severity = self._calculate_pattern_severity(category)
                    threats_detected.append({
                        'category': category,
                        'pattern': pattern,
                        'severity': severity,
                        'description': f"Detected {category} pattern"
                    })
        
        # Behavioral analysis
        behavioral_threats = self._analyze_behavior(source_ip, user_agent, request_path, user_id)
        threats_detected.extend(behavioral_threats)
        
        # Geographic analysis
        geo_threats = self._analyze_geographic(source_ip)
        threats_detected.extend(geo_threats)
        
        # IP reputation analysis
        reputation_threats = self._analyze_ip_reputation(source_ip)
        threats_detected.extend(reputation_threats)
        
        # Determine overall threat level
        if threats_detected:
            threat_level = max(threat['severity'] for threat in threats_detected)
        
        return {
            'threats_detected': threats_detected,
            'threat_level': threat_level,
            'is_safe': len(threats_detected) == 0,
            'risk_score': self._calculate_risk_score(threats_detected),
            'recommendations': self._get_security_recommendations(threats_detected)
        }
    
    def _calculate_pattern_severity(self, category: str) -> ThreatLevel:
        """Calculate threat severity based on category"""
        severity_map = {
            'sql_injection': ThreatLevel.CRITICAL,
            'command_injection': ThreatLevel.CRITICAL,
            'xss_attacks': ThreatLevel.HIGH,
            'path_traversal': ThreatLevel.HIGH,
            'ldap_injection': ThreatLevel.HIGH,
            'xml_injection': ThreatLevel.MEDIUM,
            'ssrf_attacks': ThreatLevel.HIGH,
            'no_sql_injection': ThreatLevel.HIGH
        }
        return severity_map.get(category, ThreatLevel.MEDIUM)
    
    def _analyze_behavior(self, source_ip: str, user_agent: str, request_path: str, user_id: Optional[str]) -> List[Dict[str, Any]]:
        """Analyze behavioral patterns"""
        threats = []
        current_time = datetime.now()
        
        # Track user behavior
        behavior_key = user_id or source_ip
        self.user_behavior[behavior_key].append({
            'timestamp': current_time,
            'user_agent': user_agent,
            'request_path': request_path
        })
        
        # Keep only recent behavior (last hour)
        cutoff_time = current_time - timedelta(hours=1)
        self.user_behavior[behavior_key] = [
            b for b in self.user_behavior[behavior_key] 
            if b['timestamp'] > cutoff_time
        ]
        
        # Check for rapid requests
        recent_requests = len(self.user_behavior[behavior_key])
        if recent_requests > self.behavioral_patterns['rapid_requests']['threshold']:
            threats.append({
                'category': 'behavioral',
                'pattern': 'rapid_requests',
                'severity': ThreatLevel.HIGH,
                'description': f"Rapid requests detected: {recent_requests} in last hour"
            })
        
        # Check for suspicious user agents
        if any(suspicious in user_agent.lower() for suspicious in self.behavioral_patterns['suspicious_user_agents']):
            threats.append({
                'category': 'behavioral',
                'pattern': 'suspicious_user_agent',
                'severity': ThreatLevel.MEDIUM,
                'description': f"Suspicious user agent detected: {user_agent}"
            })
        
        # Check for suspicious paths
        if any(suspicious in request_path.lower() for suspicious in self.behavioral_patterns['suspicious_paths']):
            threats.append({
                'category': 'behavioral',
                'pattern': 'suspicious_path',
                'severity': ThreatLevel.MEDIUM,
                'description': f"Suspicious path accessed: {request_path}"
            })
        
        return threats
    
    def _analyze_geographic(self, source_ip: str) -> List[Dict[str, Any]]:
        """Analyze geographic location"""
        threats = []
        
        try:
            # This would typically use a GeoIP database
            # For demo purposes, we'll simulate
            country_code = self._get_country_code(source_ip)
            
            if country_code in self.geo_blocklist:
                threats.append({
                    'category': 'geographic',
                    'pattern': 'blocked_country',
                    'severity': ThreatLevel.HIGH,
                    'description': f"Request from blocked country: {country_code}"
                })
        except Exception as e:
            logger.warning(f"Geographic analysis failed: {e}")
        
        return threats
    
    def _get_country_code(self, ip: str) -> str:
        """Get country code for IP address"""
        # This would typically use a GeoIP service
        # For demo purposes, return a random country
        return 'US'
    
    def _analyze_ip_reputation(self, source_ip: str) -> List[Dict[str, Any]]:
        """Analyze IP reputation"""
        threats = []
        
        # Check if IP is in reputation database
        if source_ip in self.ip_reputation:
            reputation = self.ip_reputation[source_ip]
            if reputation['score'] < 50:  # Low reputation score
                threats.append({
                    'category': 'reputation',
                    'pattern': 'low_reputation',
                    'severity': ThreatLevel.MEDIUM,
                    'description': f"Low reputation IP: {source_ip} (score: {reputation['score']})"
                })
        
        return threats
    
    def _calculate_risk_score(self, threats: List[Dict[str, Any]]) -> float:
        """Calculate overall risk score (0-100)"""
        if not threats:
            return 0.0
        
        severity_scores = {
            ThreatLevel.LOW: 1,
            ThreatLevel.MEDIUM: 3,
            ThreatLevel.HIGH: 7,
            ThreatLevel.CRITICAL: 10
        }
        
        total_score = sum(severity_scores.get(threat['severity'], 1) for threat in threats)
        max_possible = len(threats) * 10
        
        return min(100.0, (total_score / max_possible) * 100)
    
    def _get_security_recommendations(self, threats: List[Dict[str, Any]]) -> List[str]:
        """Get security recommendations based on detected threats"""
        recommendations = []
        
        categories = set(threat['category'] for threat in threats)
        
        if 'sql_injection' in categories:
            recommendations.append("Implement parameterized queries and input validation")
        
        if 'xss_attacks' in categories:
            recommendations.append("Sanitize HTML output and implement CSP headers")
        
        if 'command_injection' in categories:
            recommendations.append("Validate and sanitize command inputs")
        
        if 'path_traversal' in categories:
            recommendations.append("Validate file paths and use whitelist approach")
        
        if 'behavioral' in categories:
            recommendations.append("Implement rate limiting and behavioral analysis")
        
        if 'geographic' in categories:
            recommendations.append("Consider geographic restrictions")
        
        return recommendations

class SecurityMonitor:
    """Real-time security monitoring system"""
    
    def __init__(self):
        self.events = deque(maxlen=10000)  # Keep last 10k events
        self.metrics = SecurityMetrics()
        self.alert_queue = queue.Queue()
        self.monitoring_active = False
        self._lock = threading.RLock()
    
    def start_monitoring(self):
        """Start security monitoring"""
        self.monitoring_active = True
        logger.info("Security monitoring started")
    
    def stop_monitoring(self):
        """Stop security monitoring"""
        self.monitoring_active = False
        logger.info("Security monitoring stopped")
    
    def log_event(self, event: SecurityEvent):
        """Log security event"""
        with self._lock:
            self.events.append(event)
            self._update_metrics(event)
            
            # Check if alert is needed
            if self._should_alert(event):
                self.alert_queue.put(event)
    
    def _update_metrics(self, event: SecurityEvent):
        """Update security metrics"""
        self.metrics.total_events += 1
        
        if event.action_taken == SecurityAction.BLOCK:
            self.metrics.blocked_requests += 1
        
        if event.event_type == SecurityEventType.SUSPICIOUS_ACTIVITY:
            self.metrics.suspicious_activities += 1
        
        if event.event_type == SecurityEventType.THREAT_DETECTED:
            self.metrics.threat_detections += 1
        
        if event.threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]:
            self.metrics.active_threats += 1
        
        # Update security score
        self._update_security_score()
    
    def _should_alert(self, event: SecurityEvent) -> bool:
        """Determine if event should trigger alert"""
        return (
            event.threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL] or
            event.event_type == SecurityEventType.DATA_BREACH_ATTEMPT or
            event.action_taken == SecurityAction.QUARANTINE
        )
    
    def _update_security_score(self):
        """Update overall security score"""
        if self.metrics.total_events == 0:
            self.metrics.security_score = 100.0
            return
        
        # Calculate score based on threat ratio
        threat_ratio = self.metrics.threat_detections / self.metrics.total_events
        blocked_ratio = self.metrics.blocked_requests / self.metrics.total_events
        
        # Base score
        score = 100.0
        
        # Penalize for threats
        score -= threat_ratio * 50
        
        # Penalize for blocked requests (indicates attacks)
        score -= blocked_ratio * 30
        
        # Bonus for active monitoring
        if self.monitoring_active:
            score += 5
        
        self.metrics.security_score = max(0.0, min(100.0, score))
    
    def get_metrics(self) -> SecurityMetrics:
        """Get current security metrics"""
        with self._lock:
            return self.metrics
    
    def get_recent_events(self, limit: int = 100) -> List[SecurityEvent]:
        """Get recent security events"""
        with self._lock:
            return list(self.events)[-limit:]
    
    def get_threat_summary(self) -> Dict[str, Any]:
        """Get threat summary"""
        with self._lock:
            recent_events = list(self.events)[-1000:]  # Last 1000 events
            
            threat_counts = defaultdict(int)
            for event in recent_events:
                if event.event_type == SecurityEventType.THREAT_DETECTED:
                    threat_counts[event.threat_level.value] += 1
            
            return {
                'total_events': len(recent_events),
                'threat_counts': dict(threat_counts),
                'security_score': self.metrics.security_score,
                'active_threats': self.metrics.active_threats,
                'monitoring_active': self.monitoring_active
            }

class SecurityResponseSystem:
    """Automated security response system"""
    
    def __init__(self):
        self.response_rules = self._load_response_rules()
        self.blocked_ips = set()
        self.quarantined_ips = set()
        self.challenge_required = set()
        self.response_history = []
    
    def _load_response_rules(self) -> Dict[str, Dict[str, Any]]:
        """Load security response rules"""
        return {
            'sql_injection': {
                'action': SecurityAction.BLOCK,
                'duration': 3600,  # 1 hour
                'escalate': True
            },
            'command_injection': {
                'action': SecurityAction.BLOCK,
                'duration': 7200,  # 2 hours
                'escalate': True
            },
            'xss_attacks': {
                'action': SecurityAction.CHALLENGE,
                'duration': 1800,  # 30 minutes
                'escalate': False
            },
            'path_traversal': {
                'action': SecurityAction.BLOCK,
                'duration': 1800,  # 30 minutes
                'escalate': False
            },
            'behavioral': {
                'action': SecurityAction.CHALLENGE,
                'duration': 900,  # 15 minutes
                'escalate': False
            },
            'geographic': {
                'action': SecurityAction.BLOCK,
                'duration': 86400,  # 24 hours
                'escalate': True
            }
        }
    
    def determine_response(self, threat_analysis: Dict[str, Any], source_ip: str) -> Dict[str, Any]:
        """Determine appropriate security response"""
        threats = threat_analysis.get('threats_detected', [])
        threat_level = threat_analysis.get('threat_level', ThreatLevel.LOW)
        risk_score = threat_analysis.get('risk_score', 0.0)
        
        # Determine primary action
        primary_action = SecurityAction.ALLOW
        duration = 0
        escalate = False
        
        if threats:
            # Find highest priority threat
            highest_threat = max(threats, key=lambda t: self._get_threat_priority(t))
            category = highest_threat['category']
            
            if category in self.response_rules:
                rule = self.response_rules[category]
                primary_action = rule['action']
                duration = rule['duration']
                escalate = rule['escalate']
        
        # Override based on threat level
        if threat_level == ThreatLevel.CRITICAL:
            primary_action = SecurityAction.BLOCK
            duration = 86400  # 24 hours
            escalate = True
        elif threat_level == ThreatLevel.HIGH:
            primary_action = SecurityAction.BLOCK
            duration = 3600  # 1 hour
            escalate = True
        elif threat_level == ThreatLevel.MEDIUM:
            primary_action = SecurityAction.CHALLENGE
            duration = 1800  # 30 minutes
        elif threat_level == ThreatLevel.LOW:
            primary_action = SecurityAction.ALERT
            duration = 0
        
        # Apply response
        response = self._apply_response(source_ip, primary_action, duration)
        
        # Log response
        self.response_history.append({
            'timestamp': datetime.now(),
            'source_ip': source_ip,
            'action': primary_action,
            'duration': duration,
            'threat_level': threat_level,
            'risk_score': risk_score
        })
        
        return {
            'action': primary_action,
            'duration': duration,
            'escalate': escalate,
            'response_applied': response,
            'recommendations': threat_analysis.get('recommendations', [])
        }
    
    def _get_threat_priority(self, threat: Dict[str, Any]) -> int:
        """Get threat priority for sorting"""
        priority_map = {
            ThreatLevel.LOW: 1,
            ThreatLevel.MEDIUM: 2,
            ThreatLevel.HIGH: 3,
            ThreatLevel.CRITICAL: 4
        }
        return priority_map.get(threat['severity'], 1)
    
    def _apply_response(self, source_ip: str, action: SecurityAction, duration: int) -> bool:
        """Apply security response"""
        try:
            if action == SecurityAction.BLOCK:
                self.blocked_ips.add(source_ip)
                # Schedule unblock
                if duration > 0:
                    threading.Timer(duration, self._unblock_ip, args=[source_ip]).start()
            
            elif action == SecurityAction.QUARANTINE:
                self.quarantined_ips.add(source_ip)
                # Schedule unquarantine
                if duration > 0:
                    threading.Timer(duration, self._unquarantine_ip, args=[source_ip]).start()
            
            elif action == SecurityAction.CHALLENGE:
                self.challenge_required.add(source_ip)
                # Schedule challenge removal
                if duration > 0:
                    threading.Timer(duration, self._remove_challenge, args=[source_ip]).start()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to apply response: {e}")
            return False
    
    def _unblock_ip(self, source_ip: str):
        """Unblock IP address"""
        self.blocked_ips.discard(source_ip)
        logger.info(f"IP {source_ip} unblocked")
    
    def _unquarantine_ip(self, source_ip: str):
        """Unquarantine IP address"""
        self.quarantined_ips.discard(source_ip)
        logger.info(f"IP {source_ip} unquarantined")
    
    def _remove_challenge(self, source_ip: str):
        """Remove challenge requirement"""
        self.challenge_required.discard(source_ip)
        logger.info(f"Challenge removed for IP {source_ip}")
    
    def is_blocked(self, source_ip: str) -> bool:
        """Check if IP is blocked"""
        return source_ip in self.blocked_ips
    
    def is_quarantined(self, source_ip: str) -> bool:
        """Check if IP is quarantined"""
        return source_ip in self.quarantined_ips
    
    def requires_challenge(self, source_ip: str) -> bool:
        """Check if IP requires challenge"""
        return source_ip in self.challenge_required
    
    def get_response_summary(self) -> Dict[str, Any]:
        """Get response system summary"""
        return {
            'blocked_ips': len(self.blocked_ips),
            'quarantined_ips': len(self.quarantined_ips),
            'challenge_required': len(self.challenge_required),
            'total_responses': len(self.response_history),
            'recent_responses': self.response_history[-10:] if self.response_history else []
        }

class AdvancedSecuritySystem:
    """Main advanced security system"""
    
    def __init__(self):
        self.threat_detector = ThreatDetector()
        self.monitor = SecurityMonitor()
        self.response_system = SecurityResponseSystem()
        self.initialized = False
    
    async def initialize(self):
        """Initialize security system"""
        try:
            logger.info("Initializing Advanced Security System...")
            
            # Start monitoring
            self.monitor.start_monitoring()
            
            self.initialized = True
            logger.info("Advanced Security System initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize security system: {e}")
            raise
    
    def analyze_request(self, 
                       input_data: str,
                       source_ip: str,
                       user_agent: str,
                       request_path: str = "",
                       user_id: Optional[str] = None) -> Dict[str, Any]:
        """Analyze request for security threats"""
        if not self.initialized:
            return {'error': 'Security system not initialized'}
        
        # Check if IP is already blocked
        if self.response_system.is_blocked(source_ip):
            return {
                'action': SecurityAction.BLOCK,
                'reason': 'IP is blocked',
                'is_safe': False
            }
        
        # Check if IP is quarantined
        if self.response_system.is_quarantined(source_ip):
            return {
                'action': SecurityAction.QUARANTINE,
                'reason': 'IP is quarantined',
                'is_safe': False
            }
        
        # Check if challenge is required
        if self.response_system.requires_challenge(source_ip):
            return {
                'action': SecurityAction.CHALLENGE,
                'reason': 'Challenge required',
                'is_safe': False
            }
        
        # Perform threat detection
        threat_analysis = self.threat_detector.detect_threats(
            input_data, source_ip, user_agent, request_path, user_id
        )
        
        # Determine response
        response = self.response_system.determine_response(threat_analysis, source_ip)
        
        # Log security event
        event = SecurityEvent(
            event_id=secrets.token_urlsafe(16),
            event_type=SecurityEventType.THREAT_DETECTED if not threat_analysis['is_safe'] else SecurityEventType.LOGIN_ATTEMPT,
            threat_level=threat_analysis.get('threat_level', ThreatLevel.LOW),
            source_ip=source_ip,
            user_agent=user_agent,
            timestamp=datetime.now(),
            details=threat_analysis,
            action_taken=response['action']
        )
        
        self.monitor.log_event(event)
        
        return {
            'is_safe': threat_analysis['is_safe'],
            'threat_analysis': threat_analysis,
            'response': response,
            'event_id': event.event_id
        }
    
    def get_security_status(self) -> Dict[str, Any]:
        """Get current security status"""
        if not self.initialized:
            return {'status': 'not_initialized'}
        
        metrics = self.monitor.get_metrics()
        threat_summary = self.monitor.get_threat_summary()
        response_summary = self.response_system.get_response_summary()
        
        return {
            'status': 'operational',
            'metrics': metrics.__dict__,
            'threat_summary': threat_summary,
            'response_summary': response_summary,
            'monitoring_active': self.monitor.monitoring_active
        }
    
    async def shutdown(self):
        """Shutdown security system"""
        logger.info("Shutting down Advanced Security System...")
        
        # Stop monitoring
        self.monitor.stop_monitoring()
        
        self.initialized = False
        logger.info("Advanced Security System shutdown complete")

# Example usage and demonstration
async def main():
    """Demonstrate the advanced security system"""
    print("🔒 HeyGen AI - Advanced Security System Demo")
    print("=" * 50)
    
    # Initialize security system
    security_system = AdvancedSecuritySystem()
    
    try:
        # Initialize the system
        await security_system.initialize()
        
        # Test cases
        test_cases = [
            {
                'input': 'SELECT * FROM users WHERE id = 1 OR 1=1',
                'source_ip': '192.168.1.100',
                'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'request_path': '/api/users',
                'description': 'SQL Injection Attempt'
            },
            {
                'input': '<script>alert("XSS")</script>',
                'source_ip': '192.168.1.101',
                'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'request_path': '/api/comments',
                'description': 'XSS Attack Attempt'
            },
            {
                'input': 'Hello, this is a normal request',
                'source_ip': '192.168.1.102',
                'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'request_path': '/api/health',
                'description': 'Normal Request'
            },
            {
                'input': 'rm -rf /',
                'source_ip': '192.168.1.103',
                'user_agent': 'sqlmap/1.0',
                'request_path': '/admin',
                'description': 'Command Injection with Suspicious User Agent'
            }
        ]
        
        # Process test cases
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n🧪 Test Case {i}: {test_case['description']}")
            print(f"Input: {test_case['input']}")
            print(f"Source IP: {test_case['source_ip']}")
            print(f"User Agent: {test_case['user_agent']}")
            
            # Analyze request
            result = security_system.analyze_request(
                input_data=test_case['input'],
                source_ip=test_case['source_ip'],
                user_agent=test_case['user_agent'],
                request_path=test_case['request_path']
            )
            
            print(f"✅ Safe: {result['is_safe']}")
            print(f"🎯 Action: {result['response']['action']}")
            
            if not result['is_safe']:
                threats = result['threat_analysis']['threats_detected']
                print(f"⚠️  Threats detected: {len(threats)}")
                for threat in threats:
                    print(f"   - {threat['category']}: {threat['description']}")
        
        # Get security status
        print(f"\n📊 Security Status:")
        status = security_system.get_security_status()
        print(f"Security Score: {status['metrics']['security_score']:.2f}")
        print(f"Total Events: {status['metrics']['total_events']}")
        print(f"Blocked Requests: {status['metrics']['blocked_requests']}")
        print(f"Threat Detections: {status['metrics']['threat_detections']}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    
    finally:
        # Shutdown
        await security_system.shutdown()
        print("\n✅ Security system shutdown complete")

if __name__ == "__main__":
    asyncio.run(main())


