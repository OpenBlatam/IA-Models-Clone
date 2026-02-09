"""
Sistema de exportación/importación de datos para Robot Movement AI v2.0
Exportación a múltiples formatos (JSON, CSV, Excel, Parquet)
"""

from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum
import json
import csv
from io import StringIO, BytesIO

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    PARQUET_AVAILABLE = True
except ImportError:
    PARQUET_AVAILABLE = False


class ExportFormat(str, Enum):
    """Formatos de exportación disponibles"""
    JSON = "json"
    CSV = "csv"
    EXCEL = "excel"
    PARQUET = "parquet"


class DataExporter:
    """Exportador de datos"""
    
    def __init__(self):
        """Inicializar exportador"""
        pass
    
    async def export_robots(
        self,
        format: ExportFormat = ExportFormat.JSON,
        include_movements: bool = False,
        filters: Optional[Dict[str, Any]] = None
    ) -> bytes:
        """
        Exportar robots
        
        Args:
            format: Formato de exportación
            include_movements: Si incluir movements relacionados
            filters: Filtros opcionales
            
        Returns:
            Datos exportados como bytes
        """
        from core.architecture.di_setup import resolve_service
        from core.architecture.infrastructure_repositories import IRobotRepository, IMovementRepository
        
        robot_repo = resolve_service(IRobotRepository)
        robots = await robot_repo.find_all()
        
        # Aplicar filtros
        if filters:
            robots = self._apply_filters(robots, filters)
        
        # Preparar datos
        data = []
        for robot in robots:
            robot_data = {
                'id': robot.id,
                'name': robot.name,
                'status': robot.status.value,
                'position_x': robot.position.x,
                'position_y': robot.position.y,
                'position_z': robot.position.z,
                'orientation_x': robot.orientation.x,
                'orientation_y': robot.orientation.y,
                'orientation_z': robot.orientation.z,
                'orientation_w': robot.orientation.w
            }
            
            if include_movements:
                movement_repo = resolve_service(IMovementRepository)
                movements = await movement_repo.find_by_robot_id(robot.id)
                robot_data['movements'] = [
                    {
                        'id': m.id,
                        'start_x': m.start_position.x,
                        'start_y': m.start_position.y,
                        'start_z': m.start_position.z,
                        'end_x': m.end_position.x,
                        'end_y': m.end_position.y,
                        'end_z': m.end_position.z,
                        'status': m.status.value,
                        'duration': m.duration,
                        'timestamp': m.timestamp.isoformat() if hasattr(m.timestamp, 'isoformat') else str(m.timestamp)
                    }
                    for m in movements
                ]
            
            data.append(robot_data)
        
        # Exportar según formato
        if format == ExportFormat.JSON:
            return json.dumps(data, indent=2, default=str).encode('utf-8')
        elif format == ExportFormat.CSV:
            return self._export_to_csv(data, include_movements)
        elif format == ExportFormat.EXCEL:
            return self._export_to_excel(data, include_movements)
        elif format == ExportFormat.PARQUET:
            return self._export_to_parquet(data, include_movements)
        else:
            raise ValueError(f"Formato no soportado: {format}")
    
    async def export_movements(
        self,
        format: ExportFormat = ExportFormat.JSON,
        robot_id: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> bytes:
        """
        Exportar movements
        
        Args:
            format: Formato de exportación
            robot_id: Filtrar por robot_id
            start_date: Fecha de inicio
            end_date: Fecha de fin
            
        Returns:
            Datos exportados como bytes
        """
        from core.architecture.di_setup import resolve_service
        from core.architecture.infrastructure_repositories import IMovementRepository
        
        movement_repo = resolve_service(IMovementRepository)
        
        if robot_id:
            movements = await movement_repo.find_by_robot_id(robot_id)
        else:
            movements = await movement_repo.find_all()
        
        # Filtrar por fecha
        if start_date:
            movements = [m for m in movements if m.timestamp >= start_date]
        if end_date:
            movements = [m for m in movements if m.timestamp <= end_date]
        
        # Preparar datos
        data = [
            {
                'id': m.id,
                'robot_id': m.robot_id,
                'start_x': m.start_position.x,
                'start_y': m.start_position.y,
                'start_z': m.start_position.z,
                'end_x': m.end_position.x,
                'end_y': m.end_position.y,
                'end_z': m.end_position.z,
                'status': m.status.value,
                'duration': m.duration,
                'timestamp': m.timestamp.isoformat() if hasattr(m.timestamp, 'isoformat') else str(m.timestamp)
            }
            for m in movements
        ]
        
        # Exportar según formato
        if format == ExportFormat.JSON:
            return json.dumps(data, indent=2, default=str).encode('utf-8')
        elif format == ExportFormat.CSV:
            return self._export_to_csv(data, False)
        elif format == ExportFormat.EXCEL:
            return self._export_to_excel(data, False)
        elif format == ExportFormat.PARQUET:
            return self._export_to_parquet(data, False)
        else:
            raise ValueError(f"Formato no soportado: {format}")
    
    def _apply_filters(self, robots: List, filters: Dict[str, Any]) -> List:
        """Aplicar filtros a robots"""
        filtered = robots
        
        if 'status' in filters:
            filtered = [r for r in filtered if r.status.value == filters['status']]
        
        if 'name' in filters:
            filtered = [r for r in filtered if filters['name'].lower() in r.name.lower()]
        
        return filtered
    
    def _export_to_csv(self, data: List[Dict], include_movements: bool) -> bytes:
        """Exportar a CSV"""
        if not data:
            return b""
        
        output = StringIO()
        
        if include_movements:
            # CSV anidado (no ideal, pero funcional)
            writer = csv.DictWriter(output, fieldnames=data[0].keys())
            writer.writeheader()
            for row in data:
                # Convertir movements a string JSON
                if 'movements' in row:
                    row['movements'] = json.dumps(row['movements'])
                writer.writerow(row)
        else:
            writer = csv.DictWriter(output, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)
        
        return output.getvalue().encode('utf-8')
    
    def _export_to_excel(self, data: List[Dict], include_movements: bool) -> bytes:
        """Exportar a Excel"""
        if not PANDAS_AVAILABLE:
            raise ImportError("pandas requerido para exportación Excel")
        
        if not data:
            return b""
        
        df = pd.DataFrame(data)
        
        if include_movements and 'movements' in df.columns:
            # Expandir movements a columnas separadas
            df['movements'] = df['movements'].apply(lambda x: json.dumps(x) if x else '[]')
        
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Robots')
        
        return output.getvalue()
    
    def _export_to_parquet(self, data: List[Dict], include_movements: bool) -> bytes:
        """Exportar a Parquet"""
        if not PARQUET_AVAILABLE:
            raise ImportError("pyarrow requerido para exportación Parquet")
        
        if not data:
            return b""
        
        if include_movements and 'movements' in data[0]:
            # Convertir movements a string para Parquet
            for row in data:
                if 'movements' in row:
                    row['movements'] = json.dumps(row['movements'])
        
        df = pd.DataFrame(data)
        table = pa.Table.from_pandas(df)
        
        output = BytesIO()
        pq.write_table(table, output)
        
        return output.getvalue()
    
    async def import_robots(self, data: bytes, format: ExportFormat = ExportFormat.JSON) -> Dict[str, Any]:
        """
        Importar robots
        
        Args:
            data: Datos a importar
            format: Formato de los datos
            
        Returns:
            Resultado de la importación
        """
        from core.architecture.di_setup import resolve_service
        from core.architecture.infrastructure_repositories import IRobotRepository
        from core.architecture.domain_improved import Robot, Position, Orientation, MovementStatus
        
        robot_repo = resolve_service(IRobotRepository)
        
        # Parsear datos según formato
        if format == ExportFormat.JSON:
            robots_data = json.loads(data.decode('utf-8'))
        elif format == ExportFormat.CSV:
            robots_data = self._parse_csv(data)
        elif format == ExportFormat.EXCEL:
            robots_data = self._parse_excel(data)
        else:
            raise ValueError(f"Formato de importación no soportado: {format}")
        
        imported = 0
        errors = []
        
        for robot_data in robots_data:
            try:
                robot = Robot(
                    id=robot_data['id'],
                    name=robot_data['name'],
                    status=MovementStatus(robot_data['status']),
                    position=Position(
                        x=robot_data['position_x'],
                        y=robot_data['position_y'],
                        z=robot_data['position_z']
                    ),
                    orientation=Orientation(
                        x=robot_data['orientation_x'],
                        y=robot_data['orientation_y'],
                        z=robot_data['orientation_z'],
                        w=robot_data['orientation_w']
                    )
                )
                await robot_repo.save(robot)
                imported += 1
            except Exception as e:
                errors.append({'robot_id': robot_data.get('id'), 'error': str(e)})
        
        return {
            'imported': imported,
            'errors': errors,
            'total': len(robots_data)
        }
    
    def _parse_csv(self, data: bytes) -> List[Dict]:
        """Parsear CSV"""
        input_str = StringIO(data.decode('utf-8'))
        reader = csv.DictReader(input_str)
        return list(reader)
    
    def _parse_excel(self, data: bytes) -> List[Dict]:
        """Parsear Excel"""
        if not PANDAS_AVAILABLE:
            raise ImportError("pandas requerido para importación Excel")
        
        df = pd.read_excel(BytesIO(data))
        return df.to_dict('records')



