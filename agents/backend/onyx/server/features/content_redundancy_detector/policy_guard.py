"""
Policy Guardrails System
Implements content policies, rate limits, safety checks, and usage guardrails
"""

import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta

from pydantic import BaseModel, Field, validator


class PolicyViolationType(str, Enum):
    """Types of policy violations"""
    CONTENT = "content"
    RATE_LIMIT = "rate_limit"
    QUOTA = "quota"
    SAFETY = "safety"
    PII = "pii"
    TOXICITY = "toxicity"
    SPAM = "spam"
    SIZE = "size"


@dataclass
class PolicyViolation:
    """Represents a policy violation"""
    type: PolicyViolationType
    severity: str  # "low", "medium", "high", "critical"
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()


class ContentPolicy(BaseModel):
    """Content policy configuration"""
    max_content_length: int = Field(default=1000000, description="Max content length in characters")
    min_content_length: int = Field(default=10, description="Min content length in characters")
    blocked_patterns: List[str] = Field(default_factory=list, description="Regex patterns to block")
    blocked_domains: List[str] = Field(default_factory=list, description="Blocked domains")
    allowed_languages: List[str] = Field(default_factory=lambda: ["*"], description="Allowed language codes")
    require_safe_content: bool = Field(default=True, description="Require safe content")


class RateLimitPolicy(BaseModel):
    """Rate limit policy configuration"""
    requests_per_minute: int = Field(default=60, description="Max requests per minute")
    requests_per_hour: int = Field(default=1000, description="Max requests per hour")
    requests_per_day: int = Field(default=10000, description="Max requests per day")
    burst_size: int = Field(default=10, description="Burst allowance")
    cooldown_seconds: int = Field(default=60, description="Cooldown after violation")


class QuotaPolicy(BaseModel):
    """Quota policy configuration"""
    daily_quota: int = Field(default=1000, description="Daily quota")
    monthly_quota: int = Field(default=30000, description="Monthly quota")
    max_batch_size: int = Field(default=100, description="Max items in batch")
    max_concurrent_requests: int = Field(default=5, description="Max concurrent requests")


class SafetyPolicy(BaseModel):
    """Safety policy configuration"""
    check_toxicity: bool = Field(default=True, description="Check for toxic content")
    check_pii: bool = Field(default=True, description="Check for PII")
    check_spam: bool = Field(default=True, description="Check for spam")
    blocked_keywords: List[str] = Field(default_factory=list, description="Blocked keywords")
    allowed_file_types: List[str] = Field(default_factory=lambda: ["text", "html", "markdown"], description="Allowed file types")


class PolicyGuard:
    """
    Policy guardrails system for content validation and usage control
    
    Features:
    - Content validation (length, patterns, domains)
    - Rate limiting policies
    - Quota management
    - Safety checks (toxicity, PII, spam)
    - Usage tracking and enforcement
    """
    
    def __init__(
        self,
        content_policy: Optional[ContentPolicy] = None,
        rate_limit_policy: Optional[RateLimitPolicy] = None,
        quota_policy: Optional[QuotaPolicy] = None,
        safety_policy: Optional[SafetyPolicy] = None
    ):
        self.content_policy = content_policy or ContentPolicy()
        self.rate_limit_policy = rate_limit_policy or RateLimitPolicy()
        self.quota_policy = quota_policy or QuotaPolicy()
        self.safety_policy = safety_policy or SafetyPolicy()
        
        # Compile regex patterns for performance
        self._compiled_patterns = [
            re.compile(pattern, re.IGNORECASE) 
            for pattern in self.content_policy.blocked_patterns
        ]
    
    def validate_content(self, content: str, user_id: Optional[str] = None) -> Tuple[bool, List[PolicyViolation]]:
        """
        Validate content against policies
        
        Returns:
            Tuple of (is_valid, violations)
        """
        violations = []
        
        # Check content length
        content_length = len(content)
        if content_length > self.content_policy.max_content_length:
            violations.append(PolicyViolation(
                type=PolicyViolationType.SIZE,
                severity="high",
                message=f"Content exceeds maximum length ({self.content_policy.max_content_length} chars)",
                details={"content_length": content_length, "max_length": self.content_policy.max_content_length}
            ))
        
        if content_length < self.content_policy.min_content_length:
            violations.append(PolicyViolation(
                type=PolicyViolationType.SIZE,
                severity="low",
                message=f"Content below minimum length ({self.content_policy.min_content_length} chars)",
                details={"content_length": content_length, "min_length": self.content_policy.min_content_length}
            ))
        
        # Check blocked patterns
        for pattern in self._compiled_patterns:
            if pattern.search(content):
                violations.append(PolicyViolation(
                    type=PolicyViolationType.CONTENT,
                    severity="medium",
                    message="Content matches blocked pattern",
                    details={"pattern": pattern.pattern}
                ))
        
        # Check blocked keywords (safety)
        if self.safety_policy.blocked_keywords:
            content_lower = content.lower()
            found_keywords = [
                keyword for keyword in self.safety_policy.blocked_keywords
                if keyword.lower() in content_lower
            ]
            if found_keywords:
                violations.append(PolicyViolation(
                    type=PolicyViolationType.TOXICITY,
                    severity="high",
                    message="Content contains blocked keywords",
                    details={"keywords": found_keywords}
                ))
        
        # Check for PII (basic patterns)
        if self.safety_policy.check_pii:
            pii_violations = self._check_pii(content)
            violations.extend(pii_violations)
        
        return len(violations) == 0, violations
    
    def _check_pii(self, content: str) -> List[PolicyViolation]:
        """Check for Personally Identifiable Information"""
        violations = []
        
        # Email pattern
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        if re.search(email_pattern, content):
            violations.append(PolicyViolation(
                type=PolicyViolationType.PII,
                severity="medium",
                message="Potential email address detected",
                details={"type": "email"}
            ))
        
        # Credit card pattern (basic)
        cc_pattern = r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b'
        if re.search(cc_pattern, content):
            violations.append(PolicyViolation(
                type=PolicyViolationType.PII,
                severity="high",
                message="Potential credit card number detected",
                details={"type": "credit_card"}
            ))
        
        # SSN pattern (US)
        ssn_pattern = r'\b\d{3}-\d{2}-\d{4}\b'
        if re.search(ssn_pattern, content):
            violations.append(PolicyViolation(
                type=PolicyViolationType.PII,
                severity="critical",
                message="Potential SSN detected",
                details={"type": "ssn"}
            ))
        
        return violations
    
    def check_rate_limit(self, user_id: str, endpoint: str, current_count: int, 
                        window: str = "minute") -> Tuple[bool, Optional[PolicyViolation]]:
        """
        Check if request violates rate limit policy
        
        Args:
            user_id: User identifier
            endpoint: API endpoint
            current_count: Current request count in window
            window: Time window ("minute", "hour", "day")
        
        Returns:
            Tuple of (is_allowed, violation)
        """
        if window == "minute":
            limit = self.rate_limit_policy.requests_per_minute
        elif window == "hour":
            limit = self.rate_limit_policy.requests_per_hour
        elif window == "day":
            limit = self.rate_limit_policy.requests_per_day
        else:
            limit = self.rate_limit_policy.requests_per_minute
        
        if current_count >= limit:
            return False, PolicyViolation(
                type=PolicyViolationType.RATE_LIMIT,
                severity="medium",
                message=f"Rate limit exceeded for {window}",
                details={
                    "user_id": user_id,
                    "endpoint": endpoint,
                    "current_count": current_count,
                    "limit": limit,
                    "window": window
                }
            )
        
        return True, None
    
    def check_quota(self, user_id: str, current_usage: int, period: str = "day") -> Tuple[bool, Optional[PolicyViolation]]:
        """
        Check if request violates quota policy
        
        Args:
            user_id: User identifier
            current_usage: Current usage count
            period: Time period ("day", "month")
        
        Returns:
            Tuple of (is_allowed, violation)
        """
        if period == "day":
            quota = self.quota_policy.daily_quota
        elif period == "month":
            quota = self.quota_policy.monthly_quota
        else:
            quota = self.quota_policy.daily_quota
        
        if current_usage >= quota:
            return False, PolicyViolation(
                type=PolicyViolationType.QUOTA,
                severity="high",
                message=f"Quota exceeded for {period}",
                details={
                    "user_id": user_id,
                    "current_usage": current_usage,
                    "quota": quota,
                    "period": period
                }
            )
        
        return True, None
    
    def validate_batch(self, items: List[str], user_id: Optional[str] = None) -> Tuple[bool, List[PolicyViolation]]:
        """Validate batch of items"""
        violations = []
        
        # Check batch size
        if len(items) > self.quota_policy.max_batch_size:
            violations.append(PolicyViolation(
                type=PolicyViolationType.QUOTA,
                severity="medium",
                message=f"Batch size exceeds maximum ({self.quota_policy.max_batch_size} items)",
                details={
                    "batch_size": len(items),
                    "max_batch_size": self.quota_policy.max_batch_size
                }
            ))
        
        # Validate each item
        for idx, item in enumerate(items):
            is_valid, item_violations = self.validate_content(item, user_id)
            if not is_valid:
                for violation in item_violations:
                    violation.details = violation.details or {}
                    violation.details["item_index"] = idx
                    violations.append(violation)
        
        return len(violations) == 0, violations
    
    def get_policy_summary(self) -> Dict[str, Any]:
        """Get summary of all policies"""
        return {
            "content": {
                "max_length": self.content_policy.max_content_length,
                "min_length": self.content_policy.min_content_length,
                "blocked_patterns_count": len(self.content_policy.blocked_patterns),
                "blocked_domains_count": len(self.content_policy.blocked_domains)
            },
            "rate_limit": {
                "per_minute": self.rate_limit_policy.requests_per_minute,
                "per_hour": self.rate_limit_policy.requests_per_hour,
                "per_day": self.rate_limit_policy.requests_per_day,
                "burst_size": self.rate_limit_policy.burst_size
            },
            "quota": {
                "daily": self.quota_policy.daily_quota,
                "monthly": self.quota_policy.monthly_quota,
                "max_batch_size": self.quota_policy.max_batch_size,
                "max_concurrent": self.quota_policy.max_concurrent_requests
            },
            "safety": {
                "check_toxicity": self.safety_policy.check_toxicity,
                "check_pii": self.safety_policy.check_pii,
                "check_spam": self.safety_policy.check_spam,
                "blocked_keywords_count": len(self.safety_policy.blocked_keywords)
            }
        }


# Global policy guard instance (can be customized per application)
default_policy_guard = PolicyGuard()


