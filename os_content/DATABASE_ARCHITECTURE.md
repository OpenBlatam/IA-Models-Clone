# Database Architecture - OS Content UGC Video Generator

## 🗄️ **Database Layer Implementation**

### **Overview**
The database layer provides persistent storage for video requests, processing tasks, uploaded files, and system metrics using SQLAlchemy with async support.

## 📊 **Database Models**

### **1. User Model**
```python
class User(Base):
    __tablename__ = "users"
    
    id = Column(String(36), primary_key=True)
    username = Column(String(100), unique=True, nullable=False)
    email = Column(String(254), unique=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = Column(Boolean, default=True)
```

### **2. VideoRequest Model**
```python
class VideoRequest(Base):
    __tablename__ = "video_requests"
    
    id = Column(String(36), primary_key=True)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False)
    title = Column(String(200), nullable=False)
    text_prompt = Column(Text, nullable=False)
    description = Column(Text)
    ugc_type = Column(String(50), default="ugc_video_ad")
    language = Column(String(10), default="es")
    duration_per_image = Column(Float, default=3.0)
    resolution_width = Column(Integer, default=1080)
    resolution_height = Column(Integer, default=1920)
    status = Column(String(20), default="queued")
    progress = Column(Float, default=0.0)
    estimated_duration = Column(Float)
    video_url = Column(String(500))
    local_path = Column(String(500))
    cdn_url = Column(String(500))
    nlp_analysis = Column(JSON)
    metadata = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    error_message = Column(Text)
```

### **3. ProcessingTask Model**
```python
class ProcessingTask(Base):
    __tablename__ = "processing_tasks"
    
    id = Column(String(36), primary_key=True)
    video_request_id = Column(String(36), ForeignKey("video_requests.id"), nullable=False)
    task_type = Column(String(50), nullable=False)
    priority = Column(String(20), default="normal")
    status = Column(String(20), default="pending")
    progress = Column(Float, default=0.0)
    retries = Column(Integer, default=0)
    max_retries = Column(Integer, default=3)
    timeout = Column(Integer)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    error_message = Column(Text)
    result_data = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
```

### **4. UploadedFile Model**
```python
class UploadedFile(Base):
    __tablename__ = "uploaded_files"
    
    id = Column(String(36), primary_key=True)
    video_request_id = Column(String(36), ForeignKey("video_requests.id"), nullable=False)
    filename = Column(String(255), nullable=False)
    original_filename = Column(String(255), nullable=False)
    file_path = Column(String(500), nullable=False)
    file_size = Column(Integer, nullable=False)
    content_type = Column(String(100), nullable=False)
    file_type = Column(String(20), nullable=False)
    checksum = Column(String(64))
    uploaded_at = Column(DateTime, default=datetime.utcnow)
    is_processed = Column(Boolean, default=False)
```

### **5. SystemMetrics Model**
```python
class SystemMetrics(Base):
    __tablename__ = "system_metrics"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    cpu_usage = Column(Float)
    memory_usage = Column(Float)
    disk_usage = Column(Float)
    network_io = Column(JSON)
    cache_hit_rate = Column(Float)
    request_count = Column(Integer)
    error_count = Column(Integer)
    average_response_time = Column(Float)
    active_connections = Column(Integer)
    queue_size = Column(Integer)
    metrics_data = Column(JSON)
```

## 🔗 **Database Relationships**

```
User (1) ──── (N) VideoRequest
VideoRequest (1) ──── (N) ProcessingTask
VideoRequest (1) ──── (N) UploadedFile
```

## 🏗️ **Repository Pattern**

### **BaseRepository**
```python
class BaseRepository:
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def commit(self):
        await self.session.commit()
    
    async def rollback(self):
        await self.session.rollback()
```

### **VideoRepository**
```python
class VideoRepository(BaseRepository):
    async def create_video_request(self, **kwargs) -> VideoRequest:
        # Create new video request
    
    async def get_video_request_by_id(self, request_id: str) -> Optional[VideoRequest]:
        # Get video request with relationships
    
    async def update_video_request(self, request_id: str, **kwargs) -> Optional[VideoRequest]:
        # Update video request
    
    async def get_video_requests_by_status(self, status: str, limit: int = 100) -> List[VideoRequest]:
        # Get requests by status
```

### **TaskRepository**
```python
class TaskRepository(BaseRepository):
    async def create_task(self, **kwargs) -> ProcessingTask:
        # Create new processing task
    
    async def get_pending_tasks(self, limit: int = 50) -> List[ProcessingTask]:
        # Get pending tasks ordered by priority
    
    async def mark_task_completed(self, task_id: str, result_data: Dict[str, Any] = None) -> bool:
        # Mark task as completed
    
    async def mark_task_failed(self, task_id: str, error_message: str) -> bool:
        # Mark task as failed
```

### **FileRepository**
```python
class FileRepository(BaseRepository):
    async def create_uploaded_file(self, **kwargs) -> UploadedFile:
        # Create uploaded file record
    
    async def get_files_by_type(self, video_request_id: str, file_type: str) -> List[UploadedFile]:
        # Get files by type
    
    async def mark_file_processed(self, file_id: str) -> bool:
        # Mark file as processed
```

## 🔌 **Database Connection Management**

### **DatabaseConnection Class**
```python
class DatabaseConnection:
    def __init__(self):
        self.engine = None
        self.session_factory = None
    
    async def initialize(self):
        # Initialize async engine with connection pooling
    
    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        # Get database session with automatic cleanup
    
    async def create_tables(self):
        # Create all database tables
    
    async def health_check(self) -> bool:
        # Check database health
```

### **Connection Pooling**
- **Pool Size**: 20 connections
- **Max Overflow**: 30 connections
- **Pool Recycle**: 3600 seconds
- **Pool Timeout**: 30 seconds
- **Pool Pre-ping**: Enabled

## 🗃️ **Supported Databases**

### **1. SQLite (Default)**
```python
# Development database
DATABASE_URL = "sqlite+aiosqlite:///./os_content.db"
```

### **2. PostgreSQL**
```python
# Production database
DATABASE_URL = "postgresql+asyncpg://user:password@host:port/database"
```

### **3. MySQL**
```python
# Alternative production database
DATABASE_URL = "mysql+aiomysql://user:password@host:port/database"
```

## 📈 **Database Operations**

### **Video Request Lifecycle**
1. **Create Request**: Save to database with status "queued"
2. **Start Processing**: Update status to "processing"
3. **Update Progress**: Track processing progress
4. **Complete**: Update status to "completed" with video URL
5. **Error Handling**: Update status to "failed" with error message

### **Task Management**
1. **Create Task**: Associate with video request
2. **Queue Task**: Set status to "pending"
3. **Execute Task**: Update status to "processing"
4. **Complete Task**: Mark as "completed" with results
5. **Retry Logic**: Handle failed tasks with retry count

### **File Management**
1. **Upload File**: Save file metadata to database
2. **Validate File**: Check file integrity with checksum
3. **Process File**: Mark as processed after video creation
4. **Cleanup**: Remove old files based on retention policy

## 🔒 **Data Integrity**

### **Foreign Key Constraints**
- VideoRequest.user_id → User.id
- ProcessingTask.video_request_id → VideoRequest.id
- UploadedFile.video_request_id → VideoRequest.id

### **Indexes**
- Primary keys on all tables
- Index on cache_entries.key for fast lookups
- Index on video_requests.status for status queries
- Index on processing_tasks.status for task management

### **Data Validation**
- Required fields validation
- File size limits
- Content type validation
- Checksum verification

## 📊 **Performance Optimizations**

### **Connection Pooling**
- Reuse database connections
- Automatic connection health checks
- Configurable pool sizes

### **Async Operations**
- Non-blocking database operations
- Concurrent query execution
- Efficient session management

### **Query Optimization**
- Eager loading of relationships
- Pagination for large result sets
- Indexed queries for common operations

### **Caching Integration**
- Database results cached in memory
- Cache invalidation on updates
- Multi-level caching strategy

## 🔍 **Monitoring and Metrics**

### **Database Metrics**
- Connection pool statistics
- Query performance metrics
- Error rates and response times
- Storage usage and growth

### **Health Checks**
- Database connectivity
- Connection pool health
- Query response times
- Error rate monitoring

## 🚀 **Deployment Considerations**

### **Development**
- SQLite for local development
- Automatic table creation
- Debug logging enabled

### **Production**
- PostgreSQL or MySQL
- Connection pooling
- Read replicas for scaling
- Backup and recovery strategies

### **Migration Strategy**
- Alembic for schema migrations
- Version-controlled migrations
- Rollback capabilities
- Data migration scripts

## 📋 **Usage Examples**

### **Creating a Video Request**
```python
async for session in get_db_session():
    video_repo = VideoRepository(session)
    video_request = await video_repo.create_video_request(
        user_id=user_id,
        title=title,
        text_prompt=text_prompt,
        language=language
    )
```

### **Getting Video Status**
```python
async for session in get_db_session():
    video_repo = VideoRepository(session)
    video_request = await video_repo.get_video_request_by_id(request_id)
    if video_request:
        status = video_request.status
        progress = video_request.progress
```

### **Updating Progress**
```python
async for session in get_db_session():
    video_repo = VideoRepository(session)
    await video_repo.update_video_request(
        request_id,
        status="processing",
        progress=0.5
    )
```

## 🎯 **Benefits**

### **Data Persistence**
- ✅ Persistent storage of all operations
- ✅ Audit trail and history tracking
- ✅ Data recovery capabilities

### **Scalability**
- ✅ Connection pooling for high concurrency
- ✅ Efficient query optimization
- ✅ Support for multiple database backends

### **Reliability**
- ✅ Transaction management
- ✅ Error handling and rollback
- ✅ Data integrity constraints

### **Monitoring**
- ✅ Performance metrics collection
- ✅ Health monitoring
- ✅ Query analysis and optimization

The database layer provides a robust foundation for data persistence, enabling reliable video processing workflows with full audit trails and monitoring capabilities. 