"""
Migraciones de base de datos para Robot Movement AI v2.0
"""

from .migration_manager import Migration


def create_initial_schema(db):
    """Crear esquema inicial de la base de datos"""
    cursor = db.cursor()
    
    # Tabla de robots
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS robots (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            brand TEXT NOT NULL,
            model TEXT,
            status TEXT NOT NULL DEFAULT 'disconnected',
            position_x REAL DEFAULT 0.0,
            position_y REAL DEFAULT 0.0,
            position_z REAL DEFAULT 0.0,
            orientation_x REAL DEFAULT 0.0,
            orientation_y REAL DEFAULT 0.0,
            orientation_z REAL DEFAULT 0.0,
            orientation_w REAL DEFAULT 1.0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Tabla de movimientos
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS robot_movements (
            id TEXT PRIMARY KEY,
            robot_id TEXT NOT NULL,
            start_position_x REAL NOT NULL,
            start_position_y REAL NOT NULL,
            start_position_z REAL NOT NULL,
            end_position_x REAL NOT NULL,
            end_position_y REAL NOT NULL,
            end_position_z REAL NOT NULL,
            duration_seconds REAL,
            status TEXT NOT NULL DEFAULT 'pending',
            error_message TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (robot_id) REFERENCES robots(id)
        )
    """)
    
    db.commit()


def create_indexes(db):
    """Crear índices para optimizar queries"""
    cursor = db.cursor()
    
    # Índices para robots
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_robots_status ON robots(status)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_robots_brand ON robots(brand)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_robots_created_at ON robots(created_at)")
    
    # Índices para movimientos
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_movements_robot_id ON robot_movements(robot_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_movements_status ON robot_movements(status)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_movements_created_at ON robot_movements(created_at)")
    
    db.commit()


def rollback_initial_schema(db):
    """Revertir esquema inicial"""
    cursor = db.cursor()
    cursor.execute("DROP TABLE IF EXISTS robot_movements")
    cursor.execute("DROP TABLE IF EXISTS robots")
    db.commit()


# Migraciones registradas
MIGRATIONS = [
    Migration(
        version="001",
        name="create_initial_schema",
        description="Create initial database schema",
        up=create_initial_schema,
        down=rollback_initial_schema
    ),
    Migration(
        version="002",
        name="create_indexes",
        description="Create database indexes for performance",
        up=create_indexes,
        down=lambda db: None  # Los índices se pueden eliminar manualmente si es necesario
    ),
]




