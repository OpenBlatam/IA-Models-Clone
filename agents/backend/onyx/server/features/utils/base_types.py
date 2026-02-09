from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

from __future__ import annotations
from typing import Any, Dict, List, Optional, Set, Type, TypeVar, Union
from datetime import datetime
from enum import Enum
import uuid
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Base Types - Onyx Integration
Type definitions and enums for model operations.
"""

# Type variables
T = TypeVar('T', bound='OnyxBaseModel')
M = TypeVar('M', bound='OnyxBaseModel')

# Basic types
JsonDict = Dict[str, Any]
JsonList = List[Any]
JsonValue = Union[str, int, float, bool, None, JsonDict, JsonList]

# Field types
FieldType = Union[str, int, float, bool, datetime, None]
FieldValue = Union[FieldType, List[FieldType], Dict[str, FieldType]]

# Model types
ModelId = Union[str, int]
ModelKey = str
ModelValue = Any
ModelData = Dict[str, Any]
ModelList = List["T"]
ModelDict = Dict[ModelKey, "T"]

# Index types
IndexField = str
IndexValue = Any
IndexKey = str
IndexData = Dict[IndexField, IndexValue]
IndexList = List[IndexData]
IndexDict = Dict[IndexKey, IndexData]

# Cache types
CacheKey = str
CacheValue = Any
CacheData = Dict[CacheKey, CacheValue]
CacheList = List[CacheData]
CacheDict = Dict[CacheKey, CacheData]

class CacheType(str, Enum):
    """Cache type constants."""
    MEMORY = "memory"
    REDIS = "redis"
    FILE = "file"
    DATABASE = "database"
    DISTRIBUTED = "distributed"
    LOCAL = "local"
    REMOTE = "remote"
    TEMPORARY = "temporary"
    PERMANENT = "permanent"

# Validation types
ValidationRule = Dict[str, Any]
ValidationRules = Dict[str, ValidationRule]
ValidationError = str
ValidationErrors = List[ValidationError]

# Event types
EventName = str
EventData = Dict[str, Any]
EventHandler = Any
EventHandlers = Dict[EventName, List[EventHandler]]

class EventType(str, Enum):
    """Event type constants."""
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    VALIDATE = "validate"
    PROCESS = "process"
    COMPLETE = "complete"
    FAIL = "fail"
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"
    DEBUG = "debug"

class EventStatus(str, Enum):
    """Event status constants."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"
    DEBUG = "debug"

# Status types
class ModelStatus(str, Enum):
    """Model status types."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    DELETED = "deleted"
    ARCHIVED = "archived"
    DRAFT = "draft"
    PUBLISHED = "published"
    PENDING = "pending"
    REJECTED = "rejected"
    APPROVED = "approved"

class StatusType(str, Enum):
    """Status type constants."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    DELETED = "deleted"
    ARCHIVED = "archived"
    DRAFT = "draft"
    PUBLISHED = "published"
    PENDING = "pending"
    REJECTED = "rejected"
    APPROVED = "approved"

class StatusCategory(str, Enum):
    """Status category constants."""
    USER = "user"
    PRODUCT = "product"
    ORDER = "order"
    CUSTOMER = "customer"
    INVENTORY = "inventory"
    PAYMENT = "payment"
    SHIPPING = "shipping"
    MARKETING = "marketing"
    ANALYTICS = "analytics"
    SYSTEM = "system"

class CategoryType(str, Enum):
    """Category type constants."""
    USER = "user"
    PRODUCT = "product"
    ORDER = "order"
    CUSTOMER = "customer"
    INVENTORY = "inventory"
    PAYMENT = "payment"
    SHIPPING = "shipping"
    MARKETING = "marketing"
    ANALYTICS = "analytics"
    SYSTEM = "system"

# Category types
class ModelCategory(str, Enum):
    """Model category types."""
    USER = "user"
    PRODUCT = "product"
    ORDER = "order"
    CUSTOMER = "customer"
    INVENTORY = "inventory"
    PAYMENT = "payment"
    SHIPPING = "shipping"
    MARKETING = "marketing"
    ANALYTICS = "analytics"
    SYSTEM = "system"

# Permission types
class ModelPermission(str, Enum):
    """Model permission types."""
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    ADMIN = "admin"
    OWNER = "owner"
    VIEWER = "viewer"
    EDITOR = "editor"
    MANAGER = "manager"

class PermissionType(str, Enum):
    """Permission type constants."""
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    ADMIN = "admin"
    OWNER = "owner"
    VIEWER = "viewer"
    EDITOR = "editor"
    MANAGER = "manager"

class PermissionStatus(str, Enum):
    """Permission status constants."""
    ACTIVE = "active"
    GRANTED = "granted"
    DENIED = "denied"
    PENDING = "pending"
    REVOKED = "revoked"

# Validation types
class ValidationType(str, Enum):
    """Validation type constants."""
    REQUIRED = "required"
    OPTIONAL = "optional"
    UNIQUE = "unique"
    EMAIL = "email"
    URL = "url"
    PHONE = "phone"
    DATE = "date"
    DATETIME = "datetime"
    NUMBER = "number"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    STRING = "string"
    ARRAY = "array"
    OBJECT = "object"

# Index types
class IndexType(str, Enum):
    """Index type constants."""
    PRIMARY = "primary"
    SECONDARY = "secondary"
    UNIQUE = "unique"
    COMPOUND = "compound"
    TEXT = "text"
    GEO = "geo"
    HASH = "hash"
    LIST = "list"
    SET = "set"
    SORTED_SET = "sorted_set"

class IndexStatus(str, Enum):
    """Index status constants."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    DELETED = "deleted"
    ARCHIVED = "archived"
    DRAFT = "draft"
    PUBLISHED = "published"
    PENDING = "pending"
    REJECTED = "rejected"
    APPROVED = "approved"

# Cache configuration
CACHE_TTL = 2  # seconds
CACHE_SIZE = 1000
VALIDATION_TIMEOUT = 2.0  # seconds 