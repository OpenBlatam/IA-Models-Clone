from typing import Optional
import logging

from ...domain.entities import User
from ...domain.interfaces import IUserRepository
from ..adapters import IDatabaseAdapter
from ..mappers import UserMapper
from ....utils.retry import retry, RetryConfig

logger = logging.getLogger(__name__)

# Retry config for critical database operations
DB_RETRY_CONFIG = RetryConfig(
    max_attempts=3,
    initial_delay=0.5,
    max_delay=5.0,
    exponential_base=2.0,
    jitter=True
)


class UserRepository(IUserRepository):
    
    def __init__(self, database: IDatabaseAdapter):
        self.database = database
        self.table_name = "users"
    
    @retry(config=DB_RETRY_CONFIG)
    async def create(self, user: User) -> User:
        """Create user with retry logic for resilience"""
        data = UserMapper.to_dict(user)
        await self.database.insert(self.table_name, data)
        logger.debug(f"Created user {user.id}")
        return user
    
    @retry(config=DB_RETRY_CONFIG)
    async def get_by_id(self, user_id: str) -> Optional[User]:
        """Get user by ID with retry logic"""
        data = await self.database.get(self.table_name, {"id": user_id})
        if not data:
            return None
        
        return UserMapper.to_entity(data)
    
    async def get_by_email(self, email: str) -> Optional[User]:
        """Get user by email (read operation, less critical for retry)"""
        results = await self.database.query(
            self.table_name,
            filter_conditions={"email": email},
            limit=1
        )
        
        if not results:
            return None
        
        return UserMapper.to_entity(results[0])
    
    @retry(config=DB_RETRY_CONFIG)
    async def update(self, user: User) -> User:
        """Update user with retry logic for resilience"""
        data = UserMapper.to_update_dict(user)
        await self.database.update(self.table_name, {"id": user.id}, data)
        logger.debug(f"Updated user {user.id}")
        return user

