from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from enum import Enum

from typing import Any, List, Dict, Optional
import logging
import asyncio
class ProductStatus(str, Enum):
    DRAFT = "draft"
    ACTIVE = "active"
    INACTIVE = "inactive"
    DISCONTINUED = "discontinued"
    OUT_OF_STOCK = "out_of_stock"

class ProductType(str, Enum):
    PHYSICAL = "physical"
    DIGITAL = "digital"
    SERVICE = "service"
    SUBSCRIPTION = "subscription"
    BUNDLE = "bundle"

class PriceType(str, Enum):
    FIXED = "fixed"
    VARIABLE = "variable"
    TIERED = "tiered"
    SUBSCRIPTION = "subscription"

class InventoryTracking(str, Enum):
    TRACK = "track"
    NO_TRACK = "no_track"
    BACKORDER = "backorder" 