"""
BUL System - Practical Services
Real, practical services for the BUL system
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import uuid
import hashlib
import secrets
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc, func
import openai
import redis
import json

from database.practical_models import (
    User, Document, DocumentVersion, APIKey, UsageStats, Template,
    SystemLog, RateLimit, AIConfig, AIUsageStats, Workflow, WorkflowExecution
)

logger = logging.getLogger(__name__)

class UserService:
    """Real user service"""
    
    def __init__(self, db: Session):
        self.db = db
    
    async def create_user(self, email: str, username: str, password: str, full_name: str = None) -> User:
        """Create a new user"""
        try:
            # Check if user already exists
            existing_user = self.db.query(User).filter(
                or_(User.email == email, User.username == username)
            ).first()
            
            if existing_user:
                raise ValueError("User with this email or username already exists")
            
            # Hash password
            hashed_password = self._hash_password(password)
            
            # Create user
            user = User(
                email=email,
                username=username,
                hashed_password=hashed_password,
                full_name=full_name
            )
            
            self.db.add(user)
            self.db.commit()
            self.db.refresh(user)
            
            # Log user creation
            await self._log_system_event("user_created", f"User {username} created", user.id)
            
            logger.info(f"User created: {username}")
            return user
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error creating user: {e}")
            raise
    
    async def authenticate_user(self, email: str, password: str) -> Optional[User]:
        """Authenticate user with email and password"""
        try:
            user = self.db.query(User).filter(User.email == email).first()
            
            if not user or not user.is_active:
                return None
            
            if not self._verify_password(password, user.hashed_password):
                return None
            
            # Update last login
            user.last_login = datetime.utcnow()
            self.db.commit()
            
            logger.info(f"User authenticated: {email}")
            return user
            
        except Exception as e:
            logger.error(f"Error authenticating user: {e}")
            return None
    
    async def get_user_by_id(self, user_id: str) -> Optional[User]:
        """Get user by ID"""
        return self.db.query(User).filter(User.id == user_id).first()
    
    async def update_user(self, user_id: str, **kwargs) -> Optional[User]:
        """Update user information"""
        try:
            user = await self.get_user_by_id(user_id)
            if not user:
                return None
            
            for key, value in kwargs.items():
                if hasattr(user, key) and key != 'id':
                    setattr(user, key, value)
            
            user.updated_at = datetime.utcnow()
            self.db.commit()
            
            logger.info(f"User updated: {user_id}")
            return user
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error updating user: {e}")
            raise
    
    def _hash_password(self, password: str) -> str:
        """Hash password using bcrypt"""
        import bcrypt
        salt = bcrypt.gensalt()
        return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')
    
    def _verify_password(self, password: str, hashed_password: str) -> bool:
        """Verify password"""
        import bcrypt
        return bcrypt.checkpw(password.encode('utf-8'), hashed_password.encode('utf-8'))
    
    async def _log_system_event(self, level: str, message: str, user_id: str = None):
        """Log system event"""
        log_entry = SystemLog(
            level=level,
            message=message,
            user_id=user_id
        )
        self.db.add(log_entry)
        self.db.commit()

class DocumentService:
    """Real document service"""
    
    def __init__(self, db: Session):
        self.db = db
    
    async def create_document(self, user_id: str, title: str, content: str, 
                            template_type: str, language: str = "es", 
                            format: str = "pdf", metadata: Dict = None) -> Document:
        """Create a new document"""
        try:
            document = Document(
                user_id=user_id,
                title=title,
                content=content,
                template_type=template_type,
                language=language,
                format=format,
                metadata=metadata or {}
            )
            
            self.db.add(document)
            self.db.commit()
            self.db.refresh(document)
            
            # Create initial version
            await self._create_document_version(document.id, content, user_id)
            
            # Update usage stats
            await self._update_usage_stats(user_id, documents_created=1)
            
            logger.info(f"Document created: {document.id}")
            return document
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error creating document: {e}")
            raise
    
    async def get_document(self, document_id: str, user_id: str) -> Optional[Document]:
        """Get document by ID"""
        return self.db.query(Document).filter(
            and_(Document.id == document_id, Document.user_id == user_id)
        ).first()
    
    async def list_user_documents(self, user_id: str, limit: int = 10, offset: int = 0) -> List[Document]:
        """List user documents with pagination"""
        return self.db.query(Document).filter(
            Document.user_id == user_id
        ).order_by(desc(Document.created_at)).offset(offset).limit(limit).all()
    
    async def update_document(self, document_id: str, user_id: str, **kwargs) -> Optional[Document]:
        """Update document"""
        try:
            document = await self.get_document(document_id, user_id)
            if not document:
                return None
            
            # Create new version if content changed
            if 'content' in kwargs and kwargs['content'] != document.content:
                await self._create_document_version(document_id, kwargs['content'], user_id)
            
            for key, value in kwargs.items():
                if hasattr(document, key) and key != 'id':
                    setattr(document, key, value)
            
            document.updated_at = datetime.utcnow()
            self.db.commit()
            
            logger.info(f"Document updated: {document_id}")
            return document
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error updating document: {e}")
            raise
    
    async def delete_document(self, document_id: str, user_id: str) -> bool:
        """Delete document"""
        try:
            document = await self.get_document(document_id, user_id)
            if not document:
                return False
            
            self.db.delete(document)
            self.db.commit()
            
            logger.info(f"Document deleted: {document_id}")
            return True
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error deleting document: {e}")
            raise
    
    async def _create_document_version(self, document_id: str, content: str, user_id: str):
        """Create document version"""
        # Get next version number
        last_version = self.db.query(DocumentVersion).filter(
            DocumentVersion.document_id == document_id
        ).order_by(desc(DocumentVersion.version_number)).first()
        
        version_number = (last_version.version_number + 1) if last_version else 1
        
        version = DocumentVersion(
            document_id=document_id,
            version_number=version_number,
            content=content,
            created_by=user_id
        )
        
        self.db.add(version)
        self.db.commit()
    
    async def _update_usage_stats(self, user_id: str, **kwargs):
        """Update usage statistics"""
        today = datetime.utcnow().date()
        stats = self.db.query(UsageStats).filter(
            and_(UsageStats.user_id == user_id, func.date(UsageStats.date) == today)
        ).first()
        
        if not stats:
            stats = UsageStats(user_id=user_id)
            self.db.add(stats)
        
        for key, value in kwargs.items():
            if hasattr(stats, key):
                setattr(stats, key, getattr(stats, key) + value)
        
        self.db.commit()

class APIService:
    """Real API service"""
    
    def __init__(self, db: Session, redis_client: redis.Redis):
        self.db = db
        self.redis = redis_client
    
    async def create_api_key(self, user_id: str, key_name: str, permissions: List[str]) -> APIKey:
        """Create API key"""
        try:
            # Generate API key
            api_key = self._generate_api_key()
            key_hash = self._hash_api_key(api_key)
            
            api_key_obj = APIKey(
                user_id=user_id,
                key_name=key_name,
                key_hash=key_hash,
                permissions=permissions
            )
            
            self.db.add(api_key_obj)
            self.db.commit()
            self.db.refresh(api_key_obj)
            
            # Store in Redis for fast lookup
            await self._store_api_key_in_redis(api_key, api_key_obj)
            
            logger.info(f"API key created: {key_name}")
            return api_key_obj, api_key
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error creating API key: {e}")
            raise
    
    async def validate_api_key(self, api_key: str) -> Optional[Tuple[User, List[str]]]:
        """Validate API key"""
        try:
            # Check Redis first
            cached_data = await self._get_api_key_from_redis(api_key)
            if cached_data:
                user_id, permissions = cached_data
                user = self.db.query(User).filter(User.id == user_id).first()
                if user and user.is_active:
                    return user, permissions
            
            # Check database
            key_hash = self._hash_api_key(api_key)
            api_key_obj = self.db.query(APIKey).filter(
                and_(APIKey.key_hash == key_hash, APIKey.is_active == True)
            ).first()
            
            if not api_key_obj:
                return None
            
            user = self.db.query(User).filter(User.id == api_key_obj.user_id).first()
            if not user or not user.is_active:
                return None
            
            # Update last used
            api_key_obj.last_used = datetime.utcnow()
            self.db.commit()
            
            # Cache in Redis
            await self._store_api_key_in_redis(api_key, api_key_obj)
            
            return user, api_key_obj.permissions
            
        except Exception as e:
            logger.error(f"Error validating API key: {e}")
            return None
    
    async def check_rate_limit(self, user_id: str, endpoint: str, limit: int = 100, window: int = 3600) -> bool:
        """Check rate limit"""
        try:
            window_start = datetime.utcnow() - timedelta(seconds=window)
            
            # Check current rate limit
            current_limit = self.db.query(RateLimit).filter(
                and_(
                    RateLimit.user_id == user_id,
                    RateLimit.endpoint == endpoint,
                    RateLimit.window_start >= window_start
                )
            ).first()
            
            if current_limit and current_limit.requests_count >= limit:
                return False
            
            # Update or create rate limit
            if current_limit:
                current_limit.requests_count += 1
            else:
                new_limit = RateLimit(
                    user_id=user_id,
                    endpoint=endpoint,
                    requests_count=1,
                    window_start=datetime.utcnow(),
                    window_end=datetime.utcnow() + timedelta(seconds=window)
                )
                self.db.add(new_limit)
            
            self.db.commit()
            return True
            
        except Exception as e:
            logger.error(f"Error checking rate limit: {e}")
            return True  # Allow on error
    
    def _generate_api_key(self) -> str:
        """Generate API key"""
        return f"bul_{secrets.token_urlsafe(32)}"
    
    def _hash_api_key(self, api_key: str) -> str:
        """Hash API key"""
        return hashlib.sha256(api_key.encode()).hexdigest()
    
    async def _store_api_key_in_redis(self, api_key: str, api_key_obj: APIKey):
        """Store API key in Redis"""
        try:
            data = {
                "user_id": api_key_obj.user_id,
                "permissions": api_key_obj.permissions
            }
            self.redis.setex(f"api_key:{api_key}", 3600, json.dumps(data))
        except Exception as e:
            logger.error(f"Error storing API key in Redis: {e}")
    
    async def _get_api_key_from_redis(self, api_key: str) -> Optional[Tuple[str, List[str]]]:
        """Get API key from Redis"""
        try:
            data = self.redis.get(f"api_key:{api_key}")
            if data:
                parsed_data = json.loads(data)
                return parsed_data["user_id"], parsed_data["permissions"]
            return None
        except Exception as e:
            logger.error(f"Error getting API key from Redis: {e}")
            return None

class AIService:
    """Real AI service"""
    
    def __init__(self, db: Session):
        self.db = db
    
    async def get_active_ai_config(self) -> Optional[AIConfig]:
        """Get active AI configuration"""
        return self.db.query(AIConfig).filter(AIConfig.is_active == True).first()
    
    async def generate_content(self, prompt: str, user_id: str) -> str:
        """Generate content using AI"""
        try:
            ai_config = await self.get_active_ai_config()
            if not ai_config:
                raise ValueError("No active AI configuration found")
            
            # Configure OpenAI
            openai.api_key = ai_config.api_key
            
            # Generate content
            response = await openai.ChatCompletion.acreate(
                model=ai_config.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=ai_config.max_tokens,
                temperature=ai_config.temperature
            )
            
            content = response.choices[0].message.content
            
            # Update usage stats
            await self._update_ai_usage_stats(ai_config.id, len(prompt), len(content))
            
            logger.info(f"Content generated for user: {user_id}")
            return content
            
        except Exception as e:
            logger.error(f"Error generating content: {e}")
            raise
    
    async def _update_ai_usage_stats(self, ai_config_id: str, input_tokens: int, output_tokens: int):
        """Update AI usage statistics"""
        today = datetime.utcnow().date()
        stats = self.db.query(AIUsageStats).filter(
            and_(AIUsageStats.ai_config_id == ai_config_id, func.date(AIUsageStats.date) == today)
        ).first()
        
        if not stats:
            stats = AIUsageStats(ai_config_id=ai_config_id)
            self.db.add(stats)
        
        stats.requests_count += 1
        stats.tokens_used += input_tokens + output_tokens
        stats.cost += (input_tokens + output_tokens) * 0.0001  # Example pricing
        
        self.db.commit()

class WorkflowService:
    """Real workflow service"""
    
    def __init__(self, db: Session):
        self.db = db
    
    async def create_workflow(self, name: str, description: str, steps: List[Dict], user_id: str) -> Workflow:
        """Create workflow"""
        try:
            workflow = Workflow(
                name=name,
                description=description,
                steps=steps,
                created_by=user_id
            )
            
            self.db.add(workflow)
            self.db.commit()
            self.db.refresh(workflow)
            
            logger.info(f"Workflow created: {workflow.id}")
            return workflow
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error creating workflow: {e}")
            raise
    
    async def execute_workflow(self, workflow_id: str, user_id: str, input_data: Dict) -> WorkflowExecution:
        """Execute workflow"""
        try:
            workflow = self.db.query(Workflow).filter(Workflow.id == workflow_id).first()
            if not workflow:
                raise ValueError("Workflow not found")
            
            execution = WorkflowExecution(
                workflow_id=workflow_id,
                user_id=user_id,
                input_data=input_data,
                status="running",
                started_at=datetime.utcnow()
            )
            
            self.db.add(execution)
            self.db.commit()
            self.db.refresh(execution)
            
            # Execute workflow steps
            await self._execute_workflow_steps(execution, workflow.steps)
            
            logger.info(f"Workflow executed: {execution.id}")
            return execution
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error executing workflow: {e}")
            raise
    
    async def _execute_workflow_steps(self, execution: WorkflowExecution, steps: List[Dict]):
        """Execute workflow steps"""
        try:
            output_data = {}
            
            for step in steps:
                step_type = step.get("type")
                
                if step_type == "ai_generate":
                    ai_service = AIService(self.db)
                    content = await ai_service.generate_content(step["prompt"], execution.user_id)
                    output_data[step["output_key"]] = content
                
                elif step_type == "transform":
                    # Simple transformation logic
                    input_value = output_data.get(step["input_key"])
                    if input_value:
                        output_data[step["output_key"]] = step["transformation"](input_value)
                
                # Add more step types as needed
            
            execution.output_data = output_data
            execution.status = "completed"
            execution.completed_at = datetime.utcnow()
            
            self.db.commit()
            
        except Exception as e:
            execution.status = "failed"
            execution.error_message = str(e)
            execution.completed_at = datetime.utcnow()
            self.db.commit()
            raise













