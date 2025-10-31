from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field
from pydantic import BaseModel, Field

from typing import Any, List, Dict, Optional
import logging
import asyncio
# Enums for scan types and status
class ScanType(str, Enum):
    PORT_SCAN = "port_scan"
    SERVICE_DETECTION = "service_detection"
    VULNERABILITY_SCAN = "vulnerability_scan"
    OS_DETECTION = "os_detection"
    BANNER_GRABBING = "banner_grabbing"
    DNS_ENUMERATION = "dns_enumeration"
    SUBDOMAIN_SCAN = "subdomain_scan"
    WEB_CRAWLING = "web_crawling"

class ScanStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

# Dataclass for CLI/core logic
@dataclass
class ScanRequestObj:
    targets: List[str]
    scan_type: ScanType = ScanType.PORT_SCAN
    max_concurrent: int = 5
    rate_limit_type: str = "adaptive"
    backoff_strategy: str = "exponential"
    max_retries: int = 3

@dataclass
class ScanResultObj:
    target: str
    success: bool
    response_time: float
    data: Dict[str, Any]
    error: Optional[str] = None
    retry_count: int = 0

@dataclass
class ScanResponseObj:
    scan_id: str
    status: ScanStatus
    results: List[ScanResultObj]
    stats: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.utcnow)

@dataclass
class ErrorObj:
    error: str
    details: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)

# Pydantic models for API
class ScanRequestModel(BaseModel):
    targets: List[str] = Field(..., min_items=1)
    scan_type: ScanType = Field(default=ScanType.PORT_SCAN)
    max_concurrent: int = Field(default=5, ge=1, le=20)
    rate_limit_type: str = Field(default="adaptive")
    backoff_strategy: str = Field(default="exponential")
    max_retries: int = Field(default=3, ge=0, le=10)

class ScanResultModel(BaseModel):
    target: str
    success: bool
    response_time: float
    data: Dict[str, Any]
    error: Optional[str] = None
    retry_count: int = 0

class ScanResponseModel(BaseModel):
    scan_id: str
    status: ScanStatus
    results: List[ScanResultModel]
    stats: Dict[str, Any]
    timestamp: datetime

class ErrorModel(BaseModel):
    error: str
    details: Optional[str] = None
    timestamp: datetime 