"""
Sistema de backup y restore para Robot Movement AI v2.0
Backup automático con versionado y restore selectivo
"""

from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
import gzip
from pathlib import Path


class BackupType(str, Enum):
    """Tipos de backup"""
    FULL = "full"
    INCREMENTAL = "incremental"
    DIFFERENTIAL = "differential"


@dataclass
class BackupMetadata:
    """Metadata de un backup"""
    id: str
    timestamp: datetime
    backup_type: BackupType
    size_bytes: int
    entities_count: int
    movements_count: int
    checksum: str
    description: Optional[str] = None
    tags: List[str] = field(default_factory=list)


class BackupManager:
    """Gestor de backups"""
    
    def __init__(self, backup_dir: str = "backups"):
        """
        Inicializar gestor de backups
        
        Args:
            backup_dir: Directorio para almacenar backups
        """
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.backup_dir / "metadata.json"
        self.metadata: List[BackupMetadata] = []
        self._load_metadata()
    
    def _load_metadata(self):
        """Cargar metadata de backups"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    data = json.load(f)
                    self.metadata = [
                        BackupMetadata(
                            id=m['id'],
                            timestamp=datetime.fromisoformat(m['timestamp']),
                            backup_type=BackupType(m['backup_type']),
                            size_bytes=m['size_bytes'],
                            entities_count=m['entities_count'],
                            movements_count=m['movements_count'],
                            checksum=m['checksum'],
                            description=m.get('description'),
                            tags=m.get('tags', [])
                        )
                        for m in data
                    ]
            except Exception:
                self.metadata = []
    
    def _save_metadata(self):
        """Guardar metadata de backups"""
        data = [
            {
                'id': m.id,
                'timestamp': m.timestamp.isoformat(),
                'backup_type': m.backup_type.value,
                'size_bytes': m.size_bytes,
                'entities_count': m.entities_count,
                'movements_count': m.movements_count,
                'checksum': m.checksum,
                'description': m.description,
                'tags': m.tags
            }
            for m in self.metadata
        ]
        
        with open(self.metadata_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    async def create_backup(
        self,
        backup_type: BackupType = BackupType.FULL,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        compress: bool = True
    ) -> BackupMetadata:
        """
        Crear backup
        
        Args:
            backup_type: Tipo de backup
            description: Descripción del backup
            tags: Tags para categorizar
            compress: Si comprimir el backup
            
        Returns:
            Metadata del backup creado
        """
        import uuid
        import hashlib
        
        from core.architecture.di_setup import resolve_service
        from core.architecture.infrastructure_repositories import IRobotRepository, IMovementRepository
        
        backup_id = str(uuid.uuid4())
        timestamp = datetime.now()
        backup_filename = f"backup_{backup_id}_{timestamp.strftime('%Y%m%d_%H%M%S')}"
        
        if compress:
            backup_filename += ".gz"
        
        backup_path = self.backup_dir / backup_filename
        
        # Obtener repositorios
        robot_repo = resolve_service(IRobotRepository)
        movement_repo = resolve_service(IMovementRepository)
        
        # Exportar datos
        robots = await robot_repo.find_all()
        movements = await movement_repo.find_all()
        
        backup_data = {
            'backup_id': backup_id,
            'timestamp': timestamp.isoformat(),
            'backup_type': backup_type.value,
            'robots': [self._serialize_robot(r) for r in robots],
            'movements': [self._serialize_movement(m) for m in movements]
        }
        
        # Guardar backup
        json_data = json.dumps(backup_data, indent=2, default=str)
        
        if compress:
            with gzip.open(backup_path, 'wt') as f:
                f.write(json_data)
        else:
            with open(backup_path, 'w') as f:
                f.write(json_data)
        
        # Calcular checksum
        checksum = hashlib.md5(json_data.encode()).hexdigest()
        
        # Crear metadata
        metadata = BackupMetadata(
            id=backup_id,
            timestamp=timestamp,
            backup_type=backup_type,
            size_bytes=backup_path.stat().st_size,
            entities_count=len(robots),
            movements_count=len(movements),
            checksum=checksum,
            description=description,
            tags=tags or []
        )
        
        self.metadata.append(metadata)
        self._save_metadata()
        
        return metadata
    
    async def restore_backup(
        self,
        backup_id: str,
        restore_robots: bool = True,
        restore_movements: bool = True
    ) -> Dict[str, Any]:
        """
        Restaurar backup
        
        Args:
            backup_id: ID del backup a restaurar
            restore_robots: Si restaurar robots
            restore_movements: Si restaurar movements
            
        Returns:
            Resultado de la restauración
        """
        # Encontrar backup
        metadata = next((m for m in self.metadata if m.id == backup_id), None)
        if not metadata:
            raise ValueError(f"Backup {backup_id} no encontrado")
        
        # Encontrar archivo
        backup_files = list(self.backup_dir.glob(f"backup_{backup_id}_*"))
        if not backup_files:
            raise ValueError(f"Archivo de backup {backup_id} no encontrado")
        
        backup_path = backup_files[0]
        
        # Leer backup
        if backup_path.suffix == '.gz':
            with gzip.open(backup_path, 'rt') as f:
                backup_data = json.load(f)
        else:
            with open(backup_path, 'r') as f:
                backup_data = json.load(f)
        
        from core.architecture.di_setup import resolve_service
        from core.architecture.infrastructure_repositories import IRobotRepository, IMovementRepository
        
        robot_repo = resolve_service(IRobotRepository)
        movement_repo = resolve_service(IMovementRepository)
        
        restored_robots = 0
        restored_movements = 0
        
        # Restaurar robots
        if restore_robots:
            for robot_data in backup_data.get('robots', []):
                robot = self._deserialize_robot(robot_data)
                await robot_repo.save(robot)
                restored_robots += 1
        
        # Restaurar movements
        if restore_movements:
            for movement_data in backup_data.get('movements', []):
                movement = self._deserialize_movement(movement_data)
                await movement_repo.save(movement)
                restored_movements += 1
        
        return {
            'backup_id': backup_id,
            'restored_robots': restored_robots,
            'restored_movements': restored_movements,
            'timestamp': datetime.now().isoformat()
        }
    
    def list_backups(self, tags: Optional[List[str]] = None) -> List[BackupMetadata]:
        """Listar backups disponibles"""
        if tags:
            return [
                m for m in self.metadata
                if any(tag in m.tags for tag in tags)
            ]
        return self.metadata
    
    def delete_backup(self, backup_id: str):
        """Eliminar backup"""
        # Eliminar metadata
        self.metadata = [m for m in self.metadata if m.id != backup_id]
        self._save_metadata()
        
        # Eliminar archivo
        backup_files = list(self.backup_dir.glob(f"backup_{backup_id}_*"))
        for backup_file in backup_files:
            backup_file.unlink()
    
    def _serialize_robot(self, robot) -> Dict[str, Any]:
        """Serializar robot para backup"""
        return {
            'id': robot.id,
            'name': robot.name,
            'status': robot.status.value,
            'position': {
                'x': robot.position.x,
                'y': robot.position.y,
                'z': robot.position.z
            },
            'orientation': {
                'x': robot.orientation.x,
                'y': robot.orientation.y,
                'z': robot.orientation.z,
                'w': robot.orientation.w
            }
        }
    
    def _deserialize_robot(self, data: Dict[str, Any]):
        """Deserializar robot desde backup"""
        from core.architecture.domain_improved import Robot, Position, Orientation, MovementStatus
        
        return Robot(
            id=data['id'],
            name=data['name'],
            status=MovementStatus(data['status']),
            position=Position(
                x=data['position']['x'],
                y=data['position']['y'],
                z=data['position']['z']
            ),
            orientation=Orientation(
                x=data['orientation']['x'],
                y=data['orientation']['y'],
                z=data['orientation']['z'],
                w=data['orientation']['w']
            )
        )
    
    def _serialize_movement(self, movement) -> Dict[str, Any]:
        """Serializar movement para backup"""
        return {
            'id': movement.id,
            'robot_id': movement.robot_id,
            'start_position': {
                'x': movement.start_position.x,
                'y': movement.start_position.y,
                'z': movement.start_position.z
            },
            'end_position': {
                'x': movement.end_position.x,
                'y': movement.end_position.y,
                'z': movement.end_position.z
            },
            'status': movement.status.value,
            'duration': movement.duration,
            'timestamp': movement.timestamp.isoformat() if hasattr(movement.timestamp, 'isoformat') else str(movement.timestamp)
        }
    
    def _deserialize_movement(self, data: Dict[str, Any]):
        """Deserializar movement desde backup"""
        from core.architecture.domain_improved import RobotMovement, Position, MovementStatus
        from datetime import datetime
        
        return RobotMovement(
            id=data['id'],
            robot_id=data['robot_id'],
            start_position=Position(
                x=data['start_position']['x'],
                y=data['start_position']['y'],
                z=data['start_position']['z']
            ),
            end_position=Position(
                x=data['end_position']['x'],
                y=data['end_position']['y'],
                z=data['end_position']['z']
            ),
            status=MovementStatus(data['status']),
            duration=data.get('duration'),
            timestamp=datetime.fromisoformat(data['timestamp']) if isinstance(data['timestamp'], str) else data['timestamp']
        )

