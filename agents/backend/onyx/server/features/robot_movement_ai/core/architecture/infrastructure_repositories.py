"""
Infrastructure Repositories
===========================

Implementaciones concretas de repositorios para persistencia.
Soporta múltiples backends: In-Memory, SQL, y con cache.
"""

import logging
import json
from typing import Optional, List, Dict, Any
from datetime import datetime
from abc import ABC, abstractmethod

from .application_layer import IRobotRepository, IMovementRepository
from .domain_improved import Robot, RobotMovement
from .domain_improved import (
    Robot,
    RobotMovement,
    Position,
    Orientation,
    MovementStatus
)
from .error_handling import InfrastructureError, ErrorCode, ErrorContext

logger = logging.getLogger(__name__)


# ============================================================================
# Base Repository (Abstract)
# ============================================================================

class BaseRepository(ABC):
    """Clase base para repositorios."""
    
    def __init__(self):
        """Inicializar repositorio base."""
        self._initialized = False
    
    async def initialize(self):
        """Inicializar repositorio."""
        if not self._initialized:
            await self._initialize_impl()
            self._initialized = True
    
    @abstractmethod
    async def _initialize_impl(self):
        """Implementación de inicialización."""
        pass


# ============================================================================
# In-Memory Repositories (Development/Testing)
# ============================================================================

class InMemoryRobotRepository(BaseRepository, IRobotRepository):
    """
    Repositorio en memoria para robots.
    
    Útil para desarrollo, testing y demos.
    """
    
    def __init__(self):
        """Inicializar repositorio."""
        super().__init__()
        self._robots: Dict[str, Robot] = {}
        self._robots_by_brand: Dict[str, List[str]] = {}
    
    async def _initialize_impl(self):
        """Inicializar repositorio en memoria."""
        logger.info("Inicializando repositorio en memoria para robots")
    
    async def find_by_id(self, robot_id: str) -> Optional[Robot]:
        """
        Encontrar robot por ID.
        
        Args:
            robot_id: ID del robot
            
        Returns:
            Robot o None si no existe
        """
        await self.initialize()
        return self._robots.get(robot_id)
    
    async def save(self, robot: Robot) -> None:
        """
        Guardar robot.
        
        Args:
            robot: Robot a guardar
        """
        await self.initialize()
        
        # Guardar robot
        self._robots[robot.id] = robot
        
        # Indexar por marca
        brand = robot.brand
        if brand not in self._robots_by_brand:
            self._robots_by_brand[brand] = []
        if robot.id not in self._robots_by_brand[brand]:
            self._robots_by_brand[brand].append(robot.id)
        
        logger.debug(f"Robot guardado: {robot.id} ({robot.brand})")
    
    async def find_all(self) -> List[Robot]:
        """
        Encontrar todos los robots.
        
        Returns:
            Lista de todos los robots
        """
        await self.initialize()
        return list(self._robots.values())
    
    async def find_by_brand(self, brand: str) -> List[Robot]:
        """
        Encontrar robots por marca.
        
        Args:
            brand: Marca del robot
            
        Returns:
            Lista de robots de la marca
        """
        await self.initialize()
        robot_ids = self._robots_by_brand.get(brand, [])
        return [self._robots[rid] for rid in robot_ids if rid in self._robots]
    
    async def delete(self, robot_id: str) -> bool:
        """
        Eliminar robot.
        
        Args:
            robot_id: ID del robot
            
        Returns:
            True si se eliminó, False si no existe
        """
        await self.initialize()
        robot = self._robots.pop(robot_id, None)
        if robot:
            # Remover del índice
            brand = robot.brand
            if brand in self._robots_by_brand:
                self._robots_by_brand[brand] = [
                    rid for rid in self._robots_by_brand[brand] if rid != robot_id
                ]
            logger.debug(f"Robot eliminado: {robot_id}")
            return True
        return False
    
    async def count(self) -> int:
        """Contar robots."""
        await self.initialize()
        return len(self._robots)
    
    def clear(self):
        """Limpiar todos los robots (solo para testing)."""
        self._robots.clear()
        self._robots_by_brand.clear()


class InMemoryMovementRepository(BaseRepository, IMovementRepository):
    """
    Repositorio en memoria para movimientos.
    
    Útil para desarrollo, testing y demos.
    """
    
    def __init__(self):
        """Inicializar repositorio."""
        super().__init__()
        self._movements: Dict[str, RobotMovement] = {}
        self._movements_by_robot: Dict[str, List[str]] = {}
        self._movements_by_status: Dict[MovementStatus, List[str]] = {
            status: [] for status in MovementStatus
        }
    
    async def _initialize_impl(self):
        """Inicializar repositorio en memoria."""
        logger.info("Inicializando repositorio en memoria para movimientos")
    
    async def find_by_id(self, movement_id: str) -> Optional[RobotMovement]:
        """
        Encontrar movimiento por ID.
        
        Args:
            movement_id: ID del movimiento
            
        Returns:
            Movimiento o None si no existe
        """
        await self.initialize()
        return self._movements.get(movement_id)
    
    async def save(self, movement: RobotMovement) -> None:
        """
        Guardar movimiento.
        
        Args:
            movement: Movimiento a guardar
        """
        await self.initialize()
        
        # Obtener estado anterior para actualizar índices
        old_movement = self._movements.get(movement.id)
        old_status = old_movement.status if old_movement else None
        
        # Guardar movimiento
        self._movements[movement.id] = movement
        
        # Indexar por robot
        robot_id = movement.robot_id
        if robot_id not in self._movements_by_robot:
            self._movements_by_robot[robot_id] = []
        if movement.id not in self._movements_by_robot[robot_id]:
            self._movements_by_robot[robot_id].append(movement.id)
        
        # Indexar por estado
        current_status = movement.status
        
        # Remover del estado anterior si cambió
        if old_status and old_status != current_status:
            if old_status in self._movements_by_status:
                self._movements_by_status[old_status] = [
                    mid for mid in self._movements_by_status[old_status] if mid != movement.id
                ]
        
        # Agregar al nuevo estado
        if movement.id not in self._movements_by_status[current_status]:
            self._movements_by_status[current_status].append(movement.id)
        
        logger.debug(f"Movimiento guardado: {movement.id} (robot: {robot_id}, status: {current_status.value})")
    
    async def find_by_robot_id(
        self,
        robot_id: str,
        limit: int = 100
    ) -> List[RobotMovement]:
        """
        Encontrar movimientos por robot ID.
        
        Args:
            robot_id: ID del robot
            limit: Límite de resultados
            
        Returns:
            Lista de movimientos
        """
        await self.initialize()
        movement_ids = self._movements_by_robot.get(robot_id, [])
        movements = [self._movements[mid] for mid in movement_ids if mid in self._movements]
        # Ordenar por fecha de creación (más recientes primero)
        movements.sort(key=lambda m: m.created_at, reverse=True)
        return movements[:limit]
    
    async def find_by_status(
        self,
        status: MovementStatus,
        limit: int = 100
    ) -> List[RobotMovement]:
        """
        Encontrar movimientos por estado.
        
        Args:
            status: Estado del movimiento
            limit: Límite de resultados
            
        Returns:
            Lista de movimientos
        """
        await self.initialize()
        movement_ids = self._movements_by_status.get(status, [])
        movements = [self._movements[mid] for mid in movement_ids if mid in self._movements]
        movements.sort(key=lambda m: m.created_at, reverse=True)
        return movements[:limit]
    
    async def delete(self, movement_id: str) -> bool:
        """
        Eliminar movimiento.
        
        Args:
            movement_id: ID del movimiento
            
        Returns:
            True si se eliminó, False si no existe
        """
        await self.initialize()
        movement = self._movements.pop(movement_id, None)
        if movement:
            # Remover de índices
            robot_id = movement.robot_id
            if robot_id in self._movements_by_robot:
                self._movements_by_robot[robot_id] = [
                    mid for mid in self._movements_by_robot[robot_id] if mid != movement_id
                ]
            
            status = movement.status
            if status in self._movements_by_status:
                self._movements_by_status[status] = [
                    mid for mid in self._movements_by_status[status] if mid != movement_id
                ]
            
            logger.debug(f"Movimiento eliminado: {movement_id}")
            return True
        return False
    
    async def count(self) -> int:
        """Contar movimientos."""
        await self.initialize()
        return len(self._movements)
    
    def clear(self):
        """Limpiar todos los movimientos (solo para testing)."""
        self._movements.clear()
        self._movements_by_robot.clear()
        self._movements_by_status = {status: [] for status in MovementStatus}


# ============================================================================
# SQL Repositories (Production)
# ============================================================================

class SQLRobotRepository(BaseRepository, IRobotRepository):
    """
    Repositorio SQL para robots.
    
    Implementación con base de datos SQL (SQLite, PostgreSQL, MySQL, etc.).
    """
    
    def __init__(self, db_connection: Any):
        """
        Inicializar repositorio SQL.
        
        Args:
            db_connection: Conexión a base de datos (sqlite3, asyncpg, etc.)
        """
        super().__init__()
        self.db = db_connection
        self._table_name = "robots"
    
    async def _initialize_impl(self):
        """Inicializar tablas."""
        logger.info("Inicializando repositorio SQL para robots")
        await self._create_table()
    
    async def _create_table(self):
        """Crear tabla de robots si no existe."""
        # SQL genérico (ajustar según motor de BD)
        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS {self._table_name} (
            id TEXT PRIMARY KEY,
            brand TEXT NOT NULL,
            model TEXT NOT NULL,
            current_position_x REAL,
            current_position_y REAL,
            current_position_z REAL,
            current_orientation_qx REAL,
            current_orientation_qy REAL,
            current_orientation_qz REAL,
            current_orientation_qw REAL,
            is_connected BOOLEAN DEFAULT FALSE,
            active_movements TEXT DEFAULT '[]',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
        
        try:
            if hasattr(self.db, 'execute'):
                # SQLite sync
                self.db.execute(create_table_sql)
                self.db.commit()
            elif hasattr(self.db, 'fetch'):
                # asyncpg
                await self.db.execute(create_table_sql)
            else:
                logger.warning("Tipo de conexión de BD no reconocido")
        except Exception as e:
            logger.error(f"Error creando tabla: {e}")
            raise InfrastructureError(
                f"Error creando tabla de robots: {e}",
                ErrorCode.INFRASTRUCTURE_DATABASE_ERROR
            )
    
    async def find_by_id(self, robot_id: str) -> Optional[Robot]:
        """
        Encontrar robot por ID.
        
        Args:
            robot_id: ID del robot
            
        Returns:
            Robot o None si no existe
        """
        await self.initialize()
        
        query = f"SELECT * FROM {self._table_name} WHERE id = ?"
        
        try:
            if hasattr(self.db, 'execute'):
                # SQLite sync
                cursor = self.db.execute(query, (robot_id,))
                row = cursor.fetchone()
            elif hasattr(self.db, 'fetchrow'):
                # asyncpg
                row = await self.db.fetchrow(query, robot_id)
            else:
                raise InfrastructureError(
                    "Tipo de conexión de BD no soportado",
                    ErrorCode.INFRASTRUCTURE_DATABASE_ERROR
                )
            
            if not row:
                return None
            
            return self._row_to_robot(row)
        
        except Exception as e:
            logger.error(f"Error buscando robot {robot_id}: {e}")
            raise InfrastructureError(
                f"Error buscando robot: {e}",
                ErrorCode.INFRASTRUCTURE_DATABASE_ERROR
            )
    
    async def save(self, robot: Robot) -> None:
        """
        Guardar robot.
        
        Args:
            robot: Robot a guardar
        """
        await self.initialize()
        
        # Convertir a dict para insertar
        data = {
            'id': robot.id,
            'brand': robot.brand,
            'model': robot.model,
            'current_position_x': robot.current_position.x if robot.current_position else None,
            'current_position_y': robot.current_position.y if robot.current_position else None,
            'current_position_z': robot.current_position.z if robot.current_position else None,
            'current_orientation_qx': robot.current_orientation.qx if robot.current_orientation else None,
            'current_orientation_qy': robot.current_orientation.qy if robot.current_orientation else None,
            'current_orientation_qz': robot.current_orientation.qz if robot.current_orientation else None,
            'current_orientation_qw': robot.current_orientation.qw if robot.current_orientation else None,
            'is_connected': robot.is_connected,
            'active_movements': json.dumps(list(robot._active_movements)),
            'updated_at': datetime.now()
        }
        
        # UPSERT
        insert_sql = f"""
        INSERT OR REPLACE INTO {self._table_name} 
        (id, brand, model, current_position_x, current_position_y, current_position_z,
         current_orientation_qx, current_orientation_qy, current_orientation_qz, current_orientation_qw,
         is_connected, active_movements, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 
                COALESCE((SELECT created_at FROM {self._table_name} WHERE id = ?), CURRENT_TIMESTAMP), ?)
        """
        
        try:
            if hasattr(self.db, 'execute'):
                # SQLite sync
                self.db.execute(insert_sql, (
                    data['id'], data['brand'], data['model'],
                    data['current_position_x'], data['current_position_y'], data['current_position_z'],
                    data['current_orientation_qx'], data['current_orientation_qy'],
                    data['current_orientation_qz'], data['current_orientation_qw'],
                    data['is_connected'], data['active_movements'],
                    data['id'], data['updated_at']
                ))
                self.db.commit()
            elif hasattr(self.db, 'execute'):
                # asyncpg
                await self.db.execute(insert_sql, *[
                    data['id'], data['brand'], data['model'],
                    data['current_position_x'], data['current_position_y'], data['current_position_z'],
                    data['current_orientation_qx'], data['current_orientation_qy'],
                    data['current_orientation_qz'], data['current_orientation_qw'],
                    data['is_connected'], data['active_movements'],
                    data['id'], data['updated_at']
                ])
            
            logger.debug(f"Robot guardado en BD: {robot.id}")
        
        except Exception as e:
            logger.error(f"Error guardando robot {robot.id}: {e}")
            raise InfrastructureError(
                f"Error guardando robot: {e}",
                ErrorCode.INFRASTRUCTURE_DATABASE_ERROR
            )
    
    async def find_all(self) -> List[Robot]:
        """
        Encontrar todos los robots.
        
        Returns:
            Lista de todos los robots
        """
        await self.initialize()
        
        query = f"SELECT * FROM {self._table_name}"
        
        try:
            if hasattr(self.db, 'execute'):
                # SQLite sync
                cursor = self.db.execute(query)
                rows = cursor.fetchall()
            elif hasattr(self.db, 'fetch'):
                # asyncpg
                rows = await self.db.fetch(query)
            else:
                raise InfrastructureError(
                    "Tipo de conexión de BD no soportado",
                    ErrorCode.INFRASTRUCTURE_DATABASE_ERROR
                )
            
            return [self._row_to_robot(row) for row in rows]
        
        except Exception as e:
            logger.error(f"Error buscando todos los robots: {e}")
            raise InfrastructureError(
                f"Error buscando robots: {e}",
                ErrorCode.INFRASTRUCTURE_DATABASE_ERROR
            )
    
    def _row_to_robot(self, row: Any) -> Robot:
        """
        Convertir fila de BD a entidad Robot.
        
        Args:
            row: Fila de base de datos
            
        Returns:
            Entidad Robot
        """
        # Obtener valores según tipo de BD
        if isinstance(row, dict):
            # asyncpg
            id_val = row['id']
            brand = row['brand']
            model = row['model']
            pos_x = row.get('current_position_x')
            pos_y = row.get('current_position_y')
            pos_z = row.get('current_position_z')
            ori_qx = row.get('current_orientation_qx')
            ori_qy = row.get('current_orientation_qy')
            ori_qz = row.get('current_orientation_qz')
            ori_qw = row.get('current_orientation_qw')
            is_connected = row.get('is_connected', False)
            active_movements = json.loads(row.get('active_movements', '[]'))
        else:
            # SQLite tuple
            id_val = row[0]
            brand = row[1]
            model = row[2]
            pos_x = row[3]
            pos_y = row[4]
            pos_z = row[5]
            ori_qx = row[6]
            ori_qy = row[7]
            ori_qz = row[8]
            ori_qw = row[9]
            is_connected = bool(row[10]) if row[10] is not None else False
            active_movements = json.loads(row[11] if row[11] else '[]')
        
        # Crear value objects
        current_position = None
        if pos_x is not None and pos_y is not None and pos_z is not None:
            current_position = Position(x=pos_x, y=pos_y, z=pos_z)
        
        current_orientation = None
        if all([ori_qx, ori_qy, ori_qz, ori_qw]):
            current_orientation = Orientation(qx=ori_qx, qy=ori_qy, qz=ori_qz, qw=ori_qw)
        
        # Crear entidad
        robot = Robot(
            robot_id=id_val,
            brand=brand,
            model=model,
            current_position=current_position,
            current_orientation=current_orientation
        )
        
        # Restaurar estado
        if is_connected:
            robot._is_connected = True
        robot._active_movements = set(active_movements)
        
        return robot


class SQLMovementRepository(BaseRepository, IMovementRepository):
    """
    Repositorio SQL para movimientos.
    
    Implementación con base de datos SQL.
    """
    
    def __init__(self, db_connection: Any):
        """
        Inicializar repositorio SQL.
        
        Args:
            db_connection: Conexión a base de datos
        """
        super().__init__()
        self.db = db_connection
        self._table_name = "robot_movements"
    
    async def _initialize_impl(self):
        """Inicializar tablas."""
        logger.info("Inicializando repositorio SQL para movimientos")
        await self._create_table()
    
    async def _create_table(self):
        """Crear tabla de movimientos si no existe."""
        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS {self._table_name} (
            id TEXT PRIMARY KEY,
            robot_id TEXT NOT NULL,
            target_position_x REAL NOT NULL,
            target_position_y REAL NOT NULL,
            target_position_z REAL NOT NULL,
            target_orientation_qx REAL,
            target_orientation_qy REAL,
            target_orientation_qz REAL,
            target_orientation_qw REAL,
            current_position_x REAL,
            current_position_y REAL,
            current_position_z REAL,
            status TEXT NOT NULL,
            trajectory TEXT DEFAULT '[]',
            started_at TIMESTAMP,
            completed_at TIMESTAMP,
            error_message TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (robot_id) REFERENCES robots(id)
        )
        """
        
        try:
            if hasattr(self.db, 'execute'):
                self.db.execute(create_table_sql)
                self.db.commit()
            elif hasattr(self.db, 'fetch'):
                await self.db.execute(create_table_sql)
        except Exception as e:
            logger.error(f"Error creando tabla de movimientos: {e}")
            raise InfrastructureError(
                f"Error creando tabla de movimientos: {e}",
                ErrorCode.INFRASTRUCTURE_DATABASE_ERROR
            )
    
    async def find_by_id(self, movement_id: str) -> Optional[RobotMovement]:
        """Encontrar movimiento por ID."""
        await self.initialize()
        
        query = f"SELECT * FROM {self._table_name} WHERE id = ?"
        
        try:
            if hasattr(self.db, 'execute'):
                cursor = self.db.execute(query, (movement_id,))
                row = cursor.fetchone()
            elif hasattr(self.db, 'fetchrow'):
                row = await self.db.fetchrow(query, movement_id)
            else:
                raise InfrastructureError(
                    "Tipo de conexión de BD no soportado",
                    ErrorCode.INFRASTRUCTURE_DATABASE_ERROR
                )
            
            if not row:
                return None
            
            return self._row_to_movement(row)
        
        except Exception as e:
            logger.error(f"Error buscando movimiento {movement_id}: {e}")
            raise InfrastructureError(
                f"Error buscando movimiento: {e}",
                ErrorCode.INFRASTRUCTURE_DATABASE_ERROR
            )
    
    async def save(self, movement: RobotMovement) -> None:
        """Guardar movimiento."""
        await self.initialize()
        
        # Serializar trayectoria
        trajectory_json = json.dumps([p.to_dict() for p in movement.trajectory])
        
        data = {
            'id': movement.id,
            'robot_id': movement.robot_id,
            'target_position_x': movement.target_position.x,
            'target_position_y': movement.target_position.y,
            'target_position_z': movement.target_position.z,
            'target_orientation_qx': movement.target_orientation.qx if movement.target_orientation else None,
            'target_orientation_qy': movement.target_orientation.qy if movement.target_orientation else None,
            'target_orientation_qz': movement.target_orientation.qz if movement.target_orientation else None,
            'target_orientation_qw': movement.target_orientation.qw if movement.target_orientation else None,
            'current_position_x': movement.current_position.x if movement.current_position else None,
            'current_position_y': movement.current_position.y if movement.current_position else None,
            'current_position_z': movement.current_position.z if movement.current_position else None,
            'status': movement.status.value,
            'trajectory': trajectory_json,
            'started_at': movement._started_at,
            'completed_at': movement._completed_at,
            'error_message': movement._error_message,
            'updated_at': datetime.now()
        }
        
        insert_sql = f"""
        INSERT OR REPLACE INTO {self._table_name}
        (id, robot_id, target_position_x, target_position_y, target_position_z,
         target_orientation_qx, target_orientation_qy, target_orientation_qz, target_orientation_qw,
         current_position_x, current_position_y, current_position_z,
         status, trajectory, started_at, completed_at, error_message, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                COALESCE((SELECT created_at FROM {self._table_name} WHERE id = ?), CURRENT_TIMESTAMP), ?)
        """
        
        try:
            if hasattr(self.db, 'execute'):
                self.db.execute(insert_sql, (
                    data['id'], data['robot_id'],
                    data['target_position_x'], data['target_position_y'], data['target_position_z'],
                    data['target_orientation_qx'], data['target_orientation_qy'],
                    data['target_orientation_qz'], data['target_orientation_qw'],
                    data['current_position_x'], data['current_position_y'], data['current_position_z'],
                    data['status'], data['trajectory'],
                    data['started_at'], data['completed_at'], data['error_message'],
                    data['id'], data['updated_at']
                ))
                self.db.commit()
            elif hasattr(self.db, 'execute'):
                await self.db.execute(insert_sql, *[
                    data['id'], data['robot_id'],
                    data['target_position_x'], data['target_position_y'], data['target_position_z'],
                    data['target_orientation_qx'], data['target_orientation_qy'],
                    data['target_orientation_qz'], data['target_orientation_qw'],
                    data['current_position_x'], data['current_position_y'], data['current_position_z'],
                    data['status'], data['trajectory'],
                    data['started_at'], data['completed_at'], data['error_message'],
                    data['id'], data['updated_at']
                ])
            
            logger.debug(f"Movimiento guardado en BD: {movement.id}")
        
        except Exception as e:
            logger.error(f"Error guardando movimiento {movement.id}: {e}")
            raise InfrastructureError(
                f"Error guardando movimiento: {e}",
                ErrorCode.INFRASTRUCTURE_DATABASE_ERROR
            )
    
    async def find_by_robot_id(
        self,
        robot_id: str,
        limit: int = 100
    ) -> List[RobotMovement]:
        """Encontrar movimientos por robot ID."""
        await self.initialize()
        
        query = f"""
        SELECT * FROM {self._table_name}
        WHERE robot_id = ?
        ORDER BY created_at DESC
        LIMIT ?
        """
        
        try:
            if hasattr(self.db, 'execute'):
                cursor = self.db.execute(query, (robot_id, limit))
                rows = cursor.fetchall()
            elif hasattr(self.db, 'fetch'):
                rows = await self.db.fetch(query, robot_id, limit)
            else:
                raise InfrastructureError(
                    "Tipo de conexión de BD no soportado",
                    ErrorCode.INFRASTRUCTURE_DATABASE_ERROR
                )
            
            return [self._row_to_movement(row) for row in rows]
        
        except Exception as e:
            logger.error(f"Error buscando movimientos del robot {robot_id}: {e}")
            raise InfrastructureError(
                f"Error buscando movimientos: {e}",
                ErrorCode.INFRASTRUCTURE_DATABASE_ERROR
            )
    
    def _row_to_movement(self, row: Any) -> RobotMovement:
        """Convertir fila de BD a entidad RobotMovement."""
        # Similar a _row_to_robot pero para movimientos
        # Implementación simplificada - ajustar según estructura de BD
        if isinstance(row, dict):
            id_val = row['id']
            robot_id = row['robot_id']
            target_x = row['target_position_x']
            target_y = row['target_position_y']
            target_z = row['target_position_z']
            status_str = row['status']
        else:
            id_val = row[0]
            robot_id = row[1]
            target_x = row[2]
            target_y = row[3]
            target_z = row[4]
            status_str = row[12]
        
        target_position = Position(x=target_x, y=target_y, z=target_z)
        target_orientation = None
        
        movement = RobotMovement(
            robot_id=robot_id,
            target_position=target_position,
            target_orientation=target_orientation,
            movement_id=id_val
        )
        
        # Restaurar estado
        movement._status = MovementStatus(status_str)
        # Restaurar otros campos según necesidad
        
        return movement

