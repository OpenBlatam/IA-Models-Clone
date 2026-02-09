from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from typing import Any, Dict, List, Optional, Union, TypeVar, Generic
from datetime import datetime
import json
import hashlib
from pydantic import BaseModel
from .redis_utils import RedisUtils
from .redis_config import get_config
import logging
from pydantic import BaseModel
from datetime import datetime
from typing import Any, List, Dict, Optional
import asyncio
"""
Redis Indexer - Onyx Integration
Indexing system for Onyx models using Redis.
"""

logger = logging.getLogger(__name__)

T = TypeVar('T', bound=BaseModel)

class RedisIndexer:
    """Indexer for Onyx models using Redis."""
    
    def __init__(self, config: Any = None):
        """Initialize Redis indexer."""
        self.config = config or {}
        self.redis_utils = RedisUtils(get_config())
        # Support both dict and Pydantic model
        if isinstance(self.config, dict):
            self.index_prefix = self.config.get("index_prefix", "index")
            self.index_ttl = self.config.get("index_ttl", 86400)
        else:
            self.index_prefix = getattr(self.config, "index_prefix", "index")
            self.index_ttl = getattr(self.config, "index_ttl", 86400)
    
    def _generate_index_key(self, model_name: str, field: str, value: Any) -> str:
        """Generate a Redis key for an index."""
        # Create a hash of the value to ensure valid key
        value_hash = hashlib.md5(str(value).encode()).hexdigest()
        return f"{self.index_prefix}:{model_name}:{field}:{value_hash}"
    
    def _generate_model_key(self, model_name: str, model_id: str) -> str:
        """Generate a Redis key for a model."""
        return f"{self.index_prefix}:{model_name}:{model_id}"
    
    def index_model(self, model: T, model_name: str, index_fields: List[str]) -> None:
        """Index a model in Redis."""
        try:
            # Get model data
            model_data = model.model_dump()
            model_id = str(model_data.get("id", model_data.get("_id")))
            
            # Store model data
            self.redis_utils.cache_data(
                data=model_data,
                prefix=self.index_prefix,
                identifier=f"{model_name}:{model_id}",
                expire=self.index_ttl
            )
            
            # Create indexes for specified fields
            for field in index_fields:
                if field in model_data:
                    value = model_data[field]
                    if value is not None:
                        # Create index entry
                        index_key = self._generate_index_key(model_name, field, value)
                        self.redis_utils.cache_data(
                            data=model_id,
                            prefix=self.index_prefix,
                            identifier=index_key,
                            expire=self.index_ttl
                        )
            
            logger.debug(f"Indexed model {model_name}:{model_id}")
        except Exception as e:
            logger.error(f"Error indexing model: {e}")
            raise
    
    def index_batch(self, models: List[T], model_name: str, index_fields: List[str]) -> None:
        """Index multiple models in Redis."""
        try:
            # Prepare batch data
            model_data = {}
            index_data = {}
            
            for model in models:
                # Get model data
                data = model.model_dump()
                model_id = str(data.get("id", data.get("_id")))
                
                # Store model data
                model_data[f"{model_name}:{model_id}"] = data
                
                # Create indexes for specified fields
                for field in index_fields:
                    if field in data:
                        value = data[field]
                        if value is not None:
                            # Create index entry
                            index_key = self._generate_index_key(model_name, field, value)
                            index_data[index_key] = model_id
            
            # Store model data in batch
            self.redis_utils.cache_batch(
                data_dict=model_data,
                prefix=self.index_prefix,
                expire=self.index_ttl
            )
            
            # Store index data in batch
            self.redis_utils.cache_batch(
                data_dict=index_data,
                prefix=self.index_prefix,
                expire=self.index_ttl
            )
            
            logger.debug(f"Indexed {len(models)} models of type {model_name}")
        except Exception as e:
            logger.error(f"Error indexing batch: {e}")
            raise
    
    def find_by_index(self, model_name: str, field: str, value: Any, 
                     model_class: type[T]) -> Optional[T]:
        """Find a model by indexed field value."""
        try:
            # Get index key
            index_key = self._generate_index_key(model_name, field, value)
            
            # Get model ID from index
            model_id = self.redis_utils.get_cached_data(
                prefix=self.index_prefix,
                identifier=index_key
            )
            
            if model_id is None:
                return None
            
            # Get model data
            model_data = self.redis_utils.get_cached_data(
                prefix=self.index_prefix,
                identifier=f"{model_name}:{model_id}",
                model_class=model_class
            )
            
            return model_data
        except Exception as e:
            logger.error(f"Error finding by index: {e}")
            return None
    
    def find_batch_by_index(self, model_name: str, field: str, values: List[Any], 
                           model_class: type[T]) -> Dict[Any, Optional[T]]:
        """Find multiple models by indexed field values."""
        try:
            # Generate index keys
            index_keys = [
                self._generate_index_key(model_name, field, value)
                for value in values
            ]
            
            # Get model IDs from indexes
            model_ids = self.redis_utils.get_cached_batch(
                prefix=self.index_prefix,
                identifiers=index_keys
            )
            
            # Get model data for found IDs
            model_keys = [
                f"{model_name}:{model_id}"
                for model_id in model_ids.values()
                if model_id is not None
            ]
            
            models = self.redis_utils.get_cached_batch(
                prefix=self.index_prefix,
                identifiers=model_keys,
                model_class=model_class
            )
            
            # Map values to models
            return {
                value: models.get(f"{model_name}:{model_ids.get(self._generate_index_key(model_name, field, value))}")
                for value in values
            }
        except Exception as e:
            logger.error(f"Error finding batch by index: {e}")
            return {value: None for value in values}
    
    def remove_index(self, model_name: str, field: str, value: Any) -> None:
        """Remove an index entry."""
        try:
            # Get index key
            index_key = self._generate_index_key(model_name, field, value)
            
            # Remove index
            self.redis_utils.delete_key(
                prefix=self.index_prefix,
                identifier=index_key
            )
            
            logger.debug(f"Removed index {model_name}:{field}:{value}")
        except Exception as e:
            logger.error(f"Error removing index: {e}")
            raise
    
    def remove_model(self, model_name: str, model_id: str) -> None:
        """Remove a model and its indexes."""
        try:
            # Get model data
            model_data = self.redis_utils.get_cached_data(
                prefix=self.index_prefix,
                identifier=f"{model_name}:{model_id}"
            )
            
            if model_data is None:
                return
            
            # Remove model
            self.redis_utils.delete_key(
                prefix=self.index_prefix,
                identifier=f"{model_name}:{model_id}"
            )
            
            # Remove indexes
            for field, value in model_data.items():
                if value is not None:
                    index_key = self._generate_index_key(model_name, field, value)
                    self.redis_utils.delete_key(
                        prefix=self.index_prefix,
                        identifier=index_key
                    )
            
            logger.debug(f"Removed model {model_name}:{model_id}")
        except Exception as e:
            logger.error(f"Error removing model: {e}")
            raise
    
    def update_index(self, model: T, model_name: str, index_fields: List[str]) -> None:
        """Update a model's indexes."""
        try:
            # Get model data
            model_data = model.model_dump()
            model_id = str(model_data.get("id", model_data.get("_id")))
            
            # Get old model data
            old_data = self.redis_utils.get_cached_data(
                prefix=self.index_prefix,
                identifier=f"{model_name}:{model_id}"
            )
            
            if old_data is not None:
                # Remove old indexes
                for field in index_fields:
                    if field in old_data and old_data[field] is not None:
                        index_key = self._generate_index_key(model_name, field, old_data[field])
                        self.redis_utils.delete_key(
                            prefix=self.index_prefix,
                            identifier=index_key
                        )
            
            # Index updated model
            self.index_model(model, model_name, index_fields)
            
            logger.debug(f"Updated indexes for {model_name}:{model_id}")
        except Exception as e:
            logger.error(f"Error updating index: {e}")
            raise
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about indexes."""
        try:
            # Get all index keys
            index_keys = self.redis_utils.scan_keys(
                prefix=self.index_prefix,
                pattern="*"
            )
            
            # Count indexes by model and field
            stats = {}
            for key in index_keys:
                parts = key.split(":")
                if len(parts) >= 4:
                    model_name = parts[1]
                    field = parts[2]
                    
                    if model_name not in stats:
                        stats[model_name] = {}
                    
                    if field not in stats[model_name]:
                        stats[model_name][field] = 0
                    
                    stats[model_name][field] += 1
            
            return stats
        except Exception as e:
            logger.error(f"Error getting index stats: {e}")
            return {}

# Global Redis indexer instance
redis_indexer = RedisIndexer()

# Example usage:
"""

# Define a model
class UserModel(BaseModel):
    id: str
    name: str
    email: str
    created_at: datetime = datetime.utcnow()

# Create a user
user = UserModel(
    id="1",
    name="John Doe",
    email="john@example.com"
)

# Index the user
redis_indexer.index_model(
    model=user,
    model_name="user",
    index_fields=["id", "email"]
)

# Find user by email
found_user = redis_indexer.find_by_index(
    model_name="user",
    field="email",
    value="john@example.com",
    model_class=UserModel
)

# Update user
user.name = "John Updated"
redis_indexer.update_index(
    model=user,
    model_name="user",
    index_fields=["id", "email"]
)

# Remove user
redis_indexer.remove_model(
    model_name="user",
    model_id="1"
)

# Get index stats
stats = redis_indexer.get_index_stats()
""" 