"""
Threat Detection for Instagram Captions API v10.0

Advanced threat detection and analysis capabilities.
"""

import re
import time
from typing import Dict, Any, List, Optional
from collections import defaultdict

class ThreatDetector:
    """Advanced threat detection system."""
    
    def __init__(self):
        self.threat_patterns = {
            'malware': [
                r'\.(exe|bat|cmd|com|pif|scr|vbs|js|jar|war|ear|apk|ipa)$',
                r'(virus|malware|trojan|worm|spyware|ransomware)',
                r'(cmd\.exe|powershell\.exe|bash|sh|zsh)',
                r'(eval\(|exec\(|system\(|os\.system)'
            ],
            'phishing': [
                r'(paypal|ebay|amazon|bank|credit|card|login|signin)',
                r'(verify|confirm|update|secure|account|password)',
                r'(http://|https://).*\.(tk|ml|ga|cf|gq|cc)',
                r'(bit\.ly|tinyurl|goo\.gl|t\.co|is\.gd)'
            ],
            'injection': [
                r'(\b(union|select|insert|update|delete|drop|create|alter)\b)',
                r'(\b(or|and)\b\s+\d+\s*[=<>])',
                r'(\b(exec|execute|sp_executesql)\b)',
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
            'xss': [
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
            'command_injection': [
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
                r'(\b(apt|yum|dnf|pacman|brew|snap|flatpak)\b)'
            ],
            'path_traversal': [
                r'\.\./',
                r'\.\.\\',
                r'%2e%2e%2f',
                r'%2e%2e%5c',
                r'\.\.%2f',
                r'\.\.%5c',
                r'\.\.%c0%af',
                r'\.\.%c1%9c',
                r'\.\.%c0%9v',
                r'\.\.%c0%af',
                r'\.\.%c1%9c',
                r'\.\.%c0%9v'
            ],
            'ssrf': [
                r'(http|https|ftp|file|gopher|dict|ldap)://',
                r'(localhost|127\.0\.0\.1|0\.0\.0\.0|::1)',
                r'(10\.|172\.(1[6-9]|2[0-9]|3[01])\.|192\.168\.)',
                r'(169\.254\.|224\.|240\.|255\.255\.255\.255)',
                r'(0\.0\.0\.0|255\.255\.255\.255)',
                r'(internal|intranet|local|private|test|dev|staging)'
            ]
        }
        
        self.threat_history = []
        self.threat_counts = defaultdict(int)
        self.risk_scores = defaultdict(int)
    
    def analyze_text(self, text: str, context: str = "") -> Dict[str, Any]:
        """Analyze text for potential threats."""
        threats_detected = []
        risk_score = 0
        severity = "low"
        
        for threat_type, patterns in self.threat_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    threat_info = {
                        'type': threat_type,
                        'pattern': pattern,
                        'matches': len(matches),
                        'severity': self._get_threat_severity(threat_type),
                        'description': self._get_threat_description(threat_type)
                    }
                    threats_detected.append(threat_info)
                    
                    # Calculate risk score
                    risk_score += len(matches) * self._get_threat_weight(threat_type)
        
        # Determine overall severity
        if risk_score >= 80:
            severity = "critical"
        elif risk_score >= 60:
            severity = "high"
        elif risk_score >= 40:
            severity = "medium"
        elif risk_score >= 20:
            severity = "low"
        else:
            severity = "safe"
        
        # Record threat for analysis
        if threats_detected:
            self._record_threat(text, threats_detected, risk_score, context)
        
        return {
            'threats_detected': threats_detected,
            'risk_score': min(100, risk_score),
            'severity': severity,
            'total_threats': len(threats_detected),
            'analysis_timestamp': time.time()
        }
    
    def _get_threat_severity(self, threat_type: str) -> str:
        """Get severity level for threat type."""
        severity_map = {
            'malware': 'critical',
            'injection': 'high',
            'xss': 'high',
            'command_injection': 'critical',
            'path_traversal': 'high',
            'ssrf': 'high',
            'phishing': 'medium'
        }
        return severity_map.get(threat_type, 'medium')
    
    def _get_threat_weight(self, threat_type: str) -> int:
        """Get weight for threat type in risk calculation."""
        weight_map = {
            'malware': 20,
            'injection': 15,
            'xss': 12,
            'command_injection': 25,
            'path_traversal': 15,
            'ssrf': 18,
            'phishing': 8
        }
        return weight_map.get(threat_type, 10)
    
    def _get_threat_description(self, threat_type: str) -> str:
        """Get description for threat type."""
        descriptions = {
            'malware': 'Potential malware or executable code detected',
            'injection': 'SQL injection or code injection attempt detected',
            'xss': 'Cross-site scripting (XSS) attack detected',
            'command_injection': 'Command injection or system command attempt detected',
            'path_traversal': 'Path traversal or directory traversal attempt detected',
            'ssrf': 'Server-side request forgery (SSRF) attempt detected',
            'phishing': 'Potential phishing or social engineering attempt detected'
        }
        return descriptions.get(threat_type, 'Unknown threat type detected')
    
    def _record_threat(self, text: str, threats: List[Dict], risk_score: int, context: str):
        """Record threat for analysis and monitoring."""
        threat_record = {
            'timestamp': time.time(),
            'text_sample': text[:100] + '...' if len(text) > 100 else text,
            'threats': threats,
            'risk_score': risk_score,
            'context': context
        }
        
        self.threat_history.append(threat_record)
        
        # Update threat counts
        for threat in threats:
            self.threat_counts[threat['type']] += 1
        
        # Keep only last 1000 threats
        if len(self.threat_history) > 1000:
            self.threat_history = self.threat_history[-1000:]
    
    def get_threat_summary(self) -> Dict[str, Any]:
        """Get summary of detected threats."""
        current_time = time.time()
        one_hour_ago = current_time - 3600
        one_day_ago = current_time - 86400
        
        recent_threats = [t for t in self.threat_history if t['timestamp'] > one_hour_ago]
        daily_threats = [t for t in self.threat_history if t['timestamp'] > one_day_ago]
        
        return {
            'total_threats': len(self.threat_history),
            'recent_threats_1h': len(recent_threats),
            'daily_threats_24h': len(daily_threats),
            'threat_type_distribution': dict(self.threat_counts),
            'average_risk_score': sum(t['risk_score'] for t in self.threat_history) / len(self.threat_history) if self.threat_history else 0,
            'highest_risk_score': max((t['risk_score'] for t in self.threat_history), default=0)
        }
    
    def clear_history(self):
        """Clear threat history."""
        self.threat_history.clear()
        self.threat_counts.clear()






