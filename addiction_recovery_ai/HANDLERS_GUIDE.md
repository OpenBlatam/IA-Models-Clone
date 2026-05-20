# Handlers Guide - Addiction Recovery AI

## ✅ Handlers Structure

### Handlers Components

```
handlers/
├── event_handlers.py  # ✅ Event handlers
└── task_handlers.py   # ✅ Task handlers
```

## 📦 Handler Components

### `handlers/event_handlers.py` - Event Handlers
- **Status**: ✅ Active
- **Purpose**: Event handling infrastructure
- **Features**: Event processing, event routing, event handlers

**Usage:**
```python
from handlers.event_handlers import EventHandler

handler = EventHandler()

@handler.on("user_created")
async def handle_user_created(event):
    # Handle event
    pass
```

### `handlers/task_handlers.py` - Task Handlers
- **Status**: ✅ Active
- **Purpose**: Background task handling
- **Features**: Task processing, task queues, async tasks

**Usage:**
```python
from handlers.task_handlers import TaskHandler

handler = TaskHandler()

@handler.task("process_assessment")
async def process_assessment(data):
    # Process task
    pass
```

## 📝 Usage Examples

### Event Handlers
```python
from handlers.event_handlers import EventHandler

event_handler = EventHandler()

# Register handler
@event_handler.on("assessment_created")
async def handle_assessment(event):
    # Process assessment creation
    pass

# Emit event
await event_handler.emit("assessment_created", {"user_id": "123"})
```

### Task Handlers
```python
from handlers.task_handlers import TaskHandler

task_handler = TaskHandler()

# Register task
@task_handler.task("generate_report")
async def generate_report(user_id):
    # Generate report
    pass

# Queue task
await task_handler.queue("generate_report", {"user_id": "123"})
```

## 📚 Additional Resources

- See `INFRASTRUCTURE_GUIDE.md` for infrastructure
- See `MICROSERVICES_GUIDE.md` for microservices
- See `AWS_GUIDE.md` for AWS components






