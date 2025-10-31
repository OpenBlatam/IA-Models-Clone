from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

from typing import Any, Dict, List, Optional, Set, Type, TypeVar, Union
from datetime import datetime
import logging
import os
from pathlib import Path
from .model_types import (
from .model_config import ModelConfig
from .model_helpers import (
from .model_mixins import (
from .model_decorators import (
from .model_exceptions import (
from datetime import datetime
from typing import List, Optional
import logging
from typing import Any, List, Dict, Optional
import asyncio
"""
Model Examples - Onyx Integration
Example usage of model operations.
"""
    JsonDict, JsonList, JsonValue, FieldType, FieldValue,
    ModelId, ModelKey, ModelValue, ModelData, ModelList, ModelDict,
    IndexField, IndexValue, IndexKey, IndexData, IndexList, IndexDict,
    CacheKey, CacheValue, CacheData, CacheList, CacheDict,
    ValidationRule, ValidationRules, ValidationError, ValidationErrors,
    EventName, EventData, EventHandler, EventHandlers,
    ModelStatus, ModelCategory, ModelPermission,
    OnyxBaseModel, ModelField, ModelSchema, ModelRegistry,
    ModelCache, ModelIndex, ModelEvent, ModelValidation, ModelFactory
)
    validate_email, validate_url, validate_phone, validate_date, validate_datetime,
    validate_field_type, validate_field_value, validate_model_fields,
    create_model_index, create_model_cache, create_model_event,
    serialize_model, deserialize_model,
    get_model_indexes, get_model_cache, get_model_events,
    update_model_timestamps, update_model_status, update_model_version, update_model_metadata
)
    TimestampMixin, SoftDeleteMixin, VersionMixin, AuditMixin,
    ValidationMixin, CacheMixin, SerializationMixin, IndexingMixin, LoggingMixin
)
    register_model, cache_model, validate_model, track_changes,
    require_active, log_operations, enforce_version, validate_schema
)
    OnyxModelError, ValidationError, IndexingError, CacheError,
    SerializationError, VersionError, AuditError, SoftDeleteError,
    TimestampError, RegistryError, FactoryError
)

T = TypeVar('T', bound=OnyxBaseModel)

# Example 1: Basic Model
@register_model
class UserModel(OnyxBaseModel):
    name: str
    email: str
    age: Optional[int] = None
    
    # Define schema
    schema = ModelSchema(
        name="user",
        fields={
            "name": ModelField(
                name="name",
                type="string",
                required=True,
                description="User's full name"
            ),
            "email": ModelField(
                name="email",
                type="string",
                required=True,
                unique=True,
                description="User's email address"
            ),
            "age": ModelField(
                name="age",
                type="integer",
                required=False,
                description="User's age"
            )
        },
        indexes=["email"],
        cache=["id", "email"],
        validation={
            "email": {
                "type": "string",
                "format": "email",
                "required": True
            },
            "age": {
                "type": "integer",
                "minimum": 0,
                "maximum": 150
            }
        }
    )

# Example 2: Model with Mixins
@register_model
class ProductModel(
    OnyxBaseModel,
    TimestampMixin,
    SoftDeleteMixin,
    VersionMixin,
    AuditMixin,
    ValidationMixin,
    CacheMixin,
    SerializationMixin,
    IndexingMixin,
    LoggingMixin
):
    name: str
    sku: str
    price: float
    description: Optional[str] = None
    category: str
    tags: List[str] = []
    stock: int = 0
    
    # Define schema
    schema = ModelSchema(
        name="product",
        fields={
            "name": ModelField(
                name="name",
                type="string",
                required=True,
                description="Product name"
            ),
            "sku": ModelField(
                name="sku",
                type="string",
                required=True,
                unique=True,
                description="Product SKU"
            ),
            "price": ModelField(
                name="price",
                type="float",
                required=True,
                description="Product price"
            ),
            "description": ModelField(
                name="description",
                type="string",
                required=False,
                description="Product description"
            ),
            "category": ModelField(
                name="category",
                type="string",
                required=True,
                description="Product category"
            ),
            "tags": ModelField(
                name="tags",
                type="array",
                required=False,
                description="Product tags"
            ),
            "stock": ModelField(
                name="stock",
                type="integer",
                required=False,
                description="Product stock"
            )
        },
        indexes=["sku", "category"],
        cache=["id", "sku"],
        validation={
            "sku": {
                "type": "string",
                "required": True,
                "pattern": r"^[A-Z0-9-]+$"
            },
            "price": {
                "type": "float",
                "required": True,
                "minimum": 0
            },
            "stock": {
                "type": "integer",
                "required": False,
                "minimum": 0
            }
        }
    )

# Example 3: Model with Decorators
@register_model
class OrderModel(OnyxBaseModel):
    customer_id: str
    items: List[Dict[str, Any]]
    total: float
    status: str
    shipping_address: Dict[str, Any]
    payment_info: Dict[str, Any]
    
    # Define schema
    schema = ModelSchema(
        name="order",
        fields={
            "customer_id": ModelField(
                name="customer_id",
                type="string",
                required=True,
                description="Customer ID"
            ),
            "items": ModelField(
                name="items",
                type="array",
                required=True,
                description="Order items"
            ),
            "total": ModelField(
                name="total",
                type="float",
                required=True,
                description="Order total"
            ),
            "status": ModelField(
                name="status",
                type="string",
                required=True,
                description="Order status"
            ),
            "shipping_address": ModelField(
                name="shipping_address",
                type="object",
                required=True,
                description="Shipping address"
            ),
            "payment_info": ModelField(
                name="payment_info",
                type="object",
                required=True,
                description="Payment information"
            )
        },
        indexes=["customer_id", "status"],
        cache=["id", "customer_id"],
        validation={
            "total": {
                "type": "float",
                "required": True,
                "minimum": 0
            },
            "status": {
                "type": "string",
                "required": True,
                "enum": ["pending", "processing", "shipped", "delivered", "cancelled"]
            }
        }
    )
    
    @validate_model(validate_types=True, validate_custom=True)
    @cache_model(key_field="customer_id")
    @track_changes
    @require_active
    @log_operations(logging.getLogger(__name__))
    @enforce_version("1.0.0")
    @validate_schema(schema.validation)
    def update_status(self, status: str) -> None:
        self.status = status
        self.updated_at = datetime.utcnow()

# Example 4: Model with Events
@register_model
class EventModel(OnyxBaseModel):
    name: str
    description: str
    start_date: datetime
    end_date: datetime
    location: Dict[str, Any]
    attendees: List[str]
    status: str
    
    # Define schema
    schema = ModelSchema(
        name="event",
        fields={
            "name": ModelField(
                name="name",
                type="string",
                required=True,
                description="Event name"
            ),
            "description": ModelField(
                name="description",
                type="string",
                required=True,
                description="Event description"
            ),
            "start_date": ModelField(
                name="start_date",
                type="datetime",
                required=True,
                description="Event start date"
            ),
            "end_date": ModelField(
                name="end_date",
                type="datetime",
                required=True,
                description="Event end date"
            ),
            "location": ModelField(
                name="location",
                type="object",
                required=True,
                description="Event location"
            ),
            "attendees": ModelField(
                name="attendees",
                type="array",
                required=False,
                description="Event attendees"
            ),
            "status": ModelField(
                name="status",
                type="string",
                required=True,
                description="Event status"
            )
        },
        indexes=["name", "start_date", "status"],
        cache=["id", "name"],
        validation={
            "start_date": {
                "type": "datetime",
                "required": True
            },
            "end_date": {
                "type": "datetime",
                "required": True
            },
            "status": {
                "type": "string",
                "required": True,
                "enum": ["scheduled", "in_progress", "completed", "cancelled"]
            }
        },
        events={
            "created": ["send_notification", "update_calendar"],
            "updated": ["send_notification", "update_calendar"],
            "cancelled": ["send_notification", "update_calendar", "process_refunds"]
        }
    )

# Example 5: Model with Custom Validation
@register_model
class CustomModel(OnyxBaseModel):
    field1: str
    field2: int
    field3: List[str]
    field4: Dict[str, Any]
    
    # Define schema
    schema = ModelSchema(
        name="custom",
        fields={
            "field1": ModelField(
                name="field1",
                type="string",
                required=True,
                description="Custom field 1"
            ),
            "field2": ModelField(
                name="field2",
                type="integer",
                required=True,
                description="Custom field 2"
            ),
            "field3": ModelField(
                name="field3",
                type="array",
                required=True,
                description="Custom field 3"
            ),
            "field4": ModelField(
                name="field4",
                type="object",
                required=True,
                description="Custom field 4"
            )
        },
        validation={
            "field1": {
                "type": "string",
                "required": True,
                "pattern": r"^[A-Za-z]+$"
            },
            "field2": {
                "type": "integer",
                "required": True,
                "minimum": 0,
                "maximum": 100
            },
            "field3": {
                "type": "array",
                "required": True,
                "min_items": 1,
                "max_items": 10,
                "items": {
                    "type": "string",
                    "pattern": r"^[A-Za-z0-9]+$"
                }
            },
            "field4": {
                "type": "object",
                "required": True,
                "properties": {
                    "key1": {
                        "type": "string",
                        "required": True
                    },
                    "key2": {
                        "type": "integer",
                        "required": True
                    }
                }
            }
        }
    )

# Example usage:
"""

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Example 1: Basic Model
user = UserModel(
    name="John",
    email="john@example.com",
    age=30
)

# Validate
if user.is_valid():
    print("User is valid")
else:
    print("Validation errors:", user.validate())

# Example 2: Model with Mixins
product = ProductModel(
    name="Product 1",
    sku="PROD-001",
    price=99.99,
    description="Product description",
    category="Electronics",
    tags=["new", "featured"],
    stock=100
)

# Use mixin methods
product.set_audit_fields("user123")
product.cache("sku")
product.log_info("Product created successfully")

# Example 3: Model with Decorators
order = OrderModel(
    customer_id="CUST-001",
    items=[
        {"product_id": "PROD-001", "quantity": 2, "price": 99.99}
    ],
    total=199.98,
    status="pending",
    shipping_address={
        "street": "123 Main St",
        "city": "New York",
        "state": "NY",
        "zip": "10001"
    },
    payment_info={
        "method": "credit_card",
        "card_number": "****-****-****-1234"
    }
)

# Use decorated method
order.update_status("processing")

# Example 4: Model with Events
event = EventModel(
    name="Conference 2024",
    description="Annual conference",
    start_date=datetime(2024, 1, 1, 9, 0),
    end_date=datetime(2024, 1, 1, 17, 0),
    location={
        "name": "Convention Center",
        "address": "456 Event St",
        "city": "New York",
        "state": "NY",
        "zip": "10001"
    },
    attendees=["user1", "user2", "user3"],
    status="scheduled"
)

# Get events
created_events = event.get_events("created")
print("Created events:", created_events)

# Example 5: Model with Custom Validation
custom = CustomModel(
    field1="abc",
    field2=50,
    field3=["item1", "item2", "item3"],
    field4={
        "key1": "value1",
        "key2": 123
    }
)

# Validate
if custom.is_valid():
    print("Custom model is valid")
else:
    print("Validation errors:", custom.validate())
""" 