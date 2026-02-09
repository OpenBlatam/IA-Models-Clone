# Casos de Uso Reales - FileStorage

## Ejemplos Prácticos de Integración

### Caso 1: Sistema de Gestión de Usuarios

```python
from utils.file_storage import FileStorage
from datetime import datetime

class UserManager:
    def __init__(self, storage_path: str = "data/users.json"):
        self.storage = FileStorage(storage_path)
    
    def create_user(self, username: str, email: str, role: str = "user") -> str:
        """Create a new user"""
        user_id = f"user_{int(datetime.now().timestamp())}"
        user = {
            "id": user_id,
            "username": username,
            "email": email,
            "role": role,
            "created_at": datetime.now().isoformat(),
            "active": True
        }
        self.storage.add(user)
        return user_id
    
    def update_user_role(self, user_id: str, new_role: str) -> bool:
        """Update user role"""
        return self.storage.update(user_id, {
            "role": new_role,
            "updated_at": datetime.now().isoformat()
        })
    
    def deactivate_user(self, user_id: str) -> bool:
        """Deactivate a user"""
        return self.storage.update(user_id, {
            "active": False,
            "deactivated_at": datetime.now().isoformat()
        })
    
    def get_active_users(self) -> list:
        """Get all active users"""
        all_users = self.storage.read()
        return [u for u in all_users if u.get("active", False)]
    
    def delete_user(self, user_id: str) -> bool:
        """Permanently delete a user"""
        return self.storage.delete(user_id)


# Usage
user_manager = UserManager()
user_id = user_manager.create_user("alice", "alice@example.com", "admin")
user_manager.update_user_role(user_id, "super_admin")
active_users = user_manager.get_active_users()
```

### Caso 2: Sistema de Inventario

```python
from utils.file_storage import FileStorage
from typing import Optional

class InventoryManager:
    def __init__(self, storage_path: str = "data/inventory.json"):
        self.storage = FileStorage(storage_path)
    
    def add_product(self, name: str, price: float, stock: int) -> str:
        """Add a new product"""
        product_id = f"prod_{len(self.storage.read()) + 1}"
        product = {
            "id": product_id,
            "name": name,
            "price": price,
            "stock": stock,
            "low_stock_threshold": 10
        }
        self.storage.add(product)
        return product_id
    
    def update_stock(self, product_id: str, quantity: int) -> bool:
        """Update product stock"""
        product = self.storage.get(product_id)
        if not product:
            return False
        
        new_stock = product.get("stock", 0) + quantity
        if new_stock < 0:
            return False
        
        return self.storage.update(product_id, {"stock": new_stock})
    
    def sell_product(self, product_id: str, quantity: int) -> bool:
        """Sell a product (decrease stock)"""
        return self.update_stock(product_id, -quantity)
    
    def get_low_stock_products(self) -> list:
        """Get products with low stock"""
        all_products = self.storage.read()
        return [
            p for p in all_products
            if p.get("stock", 0) <= p.get("low_stock_threshold", 10)
        ]
    
    def get_product_price(self, product_id: str) -> Optional[float]:
        """Get product price"""
        product = self.storage.get(product_id)
        return product.get("price") if product else None


# Usage
inventory = InventoryManager()
prod_id = inventory.add_product("Laptop", 999.99, 50)
inventory.sell_product(prod_id, 5)
low_stock = inventory.get_low_stock_products()
```

### Caso 3: Sistema de Tareas (To-Do)

```python
from utils.file_storage import FileStorage
from datetime import datetime
from enum import Enum

class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"

class TaskManager:
    def __init__(self, storage_path: str = "data/tasks.json"):
        self.storage = FileStorage(storage_path)
    
    def create_task(self, title: str, description: str = "", priority: int = 1) -> str:
        """Create a new task"""
        task_id = f"task_{int(datetime.now().timestamp())}"
        task = {
            "id": task_id,
            "title": title,
            "description": description,
            "status": TaskStatus.PENDING.value,
            "priority": priority,
            "created_at": datetime.now().isoformat(),
            "completed_at": None
        }
        self.storage.add(task)
        return task_id
    
    def update_task_status(self, task_id: str, status: TaskStatus) -> bool:
        """Update task status"""
        updates = {
            "status": status.value,
            "updated_at": datetime.now().isoformat()
        }
        
        if status == TaskStatus.COMPLETED:
            updates["completed_at"] = datetime.now().isoformat()
        
        return self.storage.update(task_id, updates)
    
    def get_tasks_by_status(self, status: TaskStatus) -> list:
        """Get all tasks with a specific status"""
        all_tasks = self.storage.read()
        return [t for t in all_tasks if t.get("status") == status.value]
    
    def get_high_priority_tasks(self) -> list:
        """Get high priority tasks (priority >= 3)"""
        all_tasks = self.storage.read()
        return [t for t in all_tasks if t.get("priority", 0) >= 3]
    
    def complete_task(self, task_id: str) -> bool:
        """Mark a task as completed"""
        return self.update_task_status(task_id, TaskStatus.COMPLETED)


# Usage
task_manager = TaskManager()
task_id = task_manager.create_task("Refactor code", "Improve code structure", priority=3)
task_manager.update_task_status(task_id, TaskStatus.IN_PROGRESS)
pending_tasks = task_manager.get_tasks_by_status(TaskStatus.PENDING)
task_manager.complete_task(task_id)
```

### Caso 4: Sistema de Configuración

```python
from utils.file_storage import FileStorage
from typing import Any, Dict

class ConfigManager:
    def __init__(self, config_path: str = "data/config.json"):
        self.storage = FileStorage(config_path)
        self._initialize_default_config()
    
    def _initialize_default_config(self):
        """Initialize with default configuration if empty"""
        if len(self.storage.read()) == 0:
            default_config = {
                "id": "main_config",
                "app_name": "MyApp",
                "version": "1.0.0",
                "debug": False,
                "max_connections": 100,
                "timeout": 30
            }
            self.storage.write([default_config])
    
    def get_config(self) -> Dict[str, Any]:
        """Get current configuration"""
        configs = self.storage.read()
        return configs[0] if configs else {}
    
    def update_setting(self, key: str, value: Any) -> bool:
        """Update a specific setting"""
        config = self.get_config()
        if not config:
            return False
        
        updates = {key: value}
        return self.storage.update("main_config", updates)
    
    def get_setting(self, key: str, default: Any = None) -> Any:
        """Get a specific setting value"""
        config = self.get_config()
        return config.get(key, default)
    
    def reset_to_defaults(self):
        """Reset configuration to defaults"""
        self._initialize_default_config()


# Usage
config = ConfigManager()
config.update_setting("debug", True)
debug_mode = config.get_setting("debug", False)
max_conn = config.get_setting("max_connections", 50)
```

### Caso 5: Sistema de Logs de Actividad

```python
from utils.file_storage import FileStorage
from datetime import datetime
from typing import List, Dict

class ActivityLogger:
    def __init__(self, log_path: str = "data/activity_log.json"):
        self.storage = FileStorage(log_path)
    
    def log_activity(self, user_id: str, action: str, details: Dict = None) -> str:
        """Log a user activity"""
        log_id = f"log_{int(datetime.now().timestamp() * 1000)}"
        log_entry = {
            "id": log_id,
            "user_id": user_id,
            "action": action,
            "details": details or {},
            "timestamp": datetime.now().isoformat()
        }
        self.storage.add(log_entry)
        return log_id
    
    def get_user_activities(self, user_id: str) -> List[Dict]:
        """Get all activities for a user"""
        all_logs = self.storage.read()
        return [log for log in all_logs if log.get("user_id") == user_id]
    
    def get_recent_activities(self, limit: int = 10) -> List[Dict]:
        """Get recent activities"""
        all_logs = self.storage.read()
        sorted_logs = sorted(
            all_logs,
            key=lambda x: x.get("timestamp", ""),
            reverse=True
        )
        return sorted_logs[:limit]
    
    def get_activities_by_action(self, action: str) -> List[Dict]:
        """Get all activities of a specific action type"""
        all_logs = self.storage.read()
        return [log for log in all_logs if log.get("action") == action]


# Usage
logger = ActivityLogger()
logger.log_activity("user123", "login", {"ip": "192.168.1.1"})
logger.log_activity("user123", "view_page", {"page": "/dashboard"})
user_logs = logger.get_user_activities("user123")
recent = logger.get_recent_activities(5)
```

### Caso 6: Sistema de Reservas

```python
from utils.file_storage import FileStorage
from datetime import datetime, timedelta
from typing import List, Optional

class BookingSystem:
    def __init__(self, storage_path: str = "data/bookings.json"):
        self.storage = FileStorage(storage_path)
    
    def create_booking(self, user_id: str, resource_id: str, 
                      start_time: datetime, duration_hours: int) -> str:
        """Create a new booking"""
        booking_id = f"booking_{int(datetime.now().timestamp())}"
        end_time = start_time + timedelta(hours=duration_hours)
        
        booking = {
            "id": booking_id,
            "user_id": user_id,
            "resource_id": resource_id,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_hours": duration_hours,
            "status": "confirmed",
            "created_at": datetime.now().isoformat()
        }
        self.storage.add(booking)
        return booking_id
    
    def cancel_booking(self, booking_id: str) -> bool:
        """Cancel a booking"""
        return self.storage.update(booking_id, {
            "status": "cancelled",
            "cancelled_at": datetime.now().isoformat()
        })
    
    def get_user_bookings(self, user_id: str) -> List[Dict]:
        """Get all bookings for a user"""
        all_bookings = self.storage.read()
        return [b for b in all_bookings if b.get("user_id") == user_id]
    
    def get_upcoming_bookings(self) -> List[Dict]:
        """Get all upcoming confirmed bookings"""
        all_bookings = self.storage.read()
        now = datetime.now()
        
        upcoming = []
        for booking in all_bookings:
            if booking.get("status") != "confirmed":
                continue
            
            start_time = datetime.fromisoformat(booking.get("start_time", ""))
            if start_time > now:
                upcoming.append(booking)
        
        return sorted(upcoming, key=lambda x: x.get("start_time", ""))


# Usage
booking_system = BookingSystem()
booking_id = booking_system.create_booking(
    "user123",
    "room_101",
    datetime.now() + timedelta(days=1),
    2
)
user_bookings = booking_system.get_user_bookings("user123")
upcoming = booking_system.get_upcoming_bookings()
```

## Patrones Comunes

### Patrón 1: Búsqueda y Filtrado

```python
def find_by_criteria(storage: FileStorage, **criteria) -> List[Dict]:
    """Generic find by criteria"""
    all_records = storage.read()
    matches = []
    
    for record in all_records:
        if all(record.get(key) == value for key, value in criteria.items()):
            matches.append(record)
    
    return matches
```

### Patrón 2: Actualización Condicional

```python
def update_if_condition(storage: FileStorage, record_id: str, 
                        updates: Dict, condition: callable) -> bool:
    """Update only if condition is met"""
    record = storage.get(record_id)
    if not record:
        return False
    
    if condition(record):
        return storage.update(record_id, updates)
    return False
```

### Patrón 3: Batch Operations

```python
def batch_update(storage: FileStorage, updates: List[Dict]) -> Dict[str, bool]:
    """Update multiple records at once"""
    results = {}
    records = storage.read()
    
    for update_item in updates:
        record_id = update_item.get("id")
        update_data = update_item.get("data", {})
        
        found = False
        for i, record in enumerate(records):
            if record.get("id") == record_id:
                records[i].update(update_data)
                found = True
                results[record_id] = True
                break
        
        if not found:
            results[record_id] = False
    
    if any(results.values()):
        storage.write(records)
    
    return results
```

## Mejores Prácticas Aplicadas

1. ✅ **Separación de Responsabilidades**: Cada clase maneja un dominio específico
2. ✅ **Validación de Datos**: Validación antes de operaciones críticas
3. ✅ **Manejo de Errores**: Try-except apropiado en operaciones sensibles
4. ✅ **Type Hints**: Tipado completo para mejor IDE support
5. ✅ **Documentación**: Docstrings claros en todos los métodos
6. ✅ **Reutilización**: FileStorage como componente base reutilizable


