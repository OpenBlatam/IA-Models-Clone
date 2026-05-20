"""
Database Service
================

Servicio de persistencia de datos usando SQLite.
"""

import sqlite3
import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path
from contextlib import contextmanager
import threading

logger = logging.getLogger(__name__)


class DatabaseService:
    """Servicio de base de datos para persistencia de datos."""
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Inicializar servicio de base de datos.
        
        Args:
            db_path: Ruta a la base de datos SQLite
        """
        self.db_path = Path(db_path or "artist_manager.db")
        self._local = threading.local()
        self._init_database()
        self._logger = logger
    
    def _get_connection(self) -> sqlite3.Connection:
        """Obtener conexión a la base de datos (thread-safe)."""
        if not hasattr(self._local, 'connection'):
            self._local.connection = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False
            )
            self._local.connection.row_factory = sqlite3.Row
        return self._local.connection
    
    @contextmanager
    def get_cursor(self):
        """Context manager para cursor de base de datos."""
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            yield cursor
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            cursor.close()
    
    def _init_database(self):
        """Inicializar tablas de la base de datos."""
        with self.get_cursor() as cursor:
            # Tabla de eventos
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    id TEXT PRIMARY KEY,
                    artist_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    description TEXT,
                    event_type TEXT NOT NULL,
                    start_time TEXT NOT NULL,
                    end_time TEXT NOT NULL,
                    location TEXT,
                    attendees TEXT,
                    reminders TEXT,
                    protocol_requirements TEXT,
                    wardrobe_requirements TEXT,
                    notes TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            
            # Tabla de rutinas
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS routines (
                    id TEXT PRIMARY KEY,
                    artist_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    description TEXT,
                    routine_type TEXT NOT NULL,
                    scheduled_time TEXT NOT NULL,
                    duration_minutes INTEGER NOT NULL,
                    priority INTEGER DEFAULT 5,
                    days_of_week TEXT,
                    is_required INTEGER DEFAULT 1,
                    reminders TEXT,
                    notes TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            
            # Tabla de completaciones de rutinas
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS routine_completions (
                    id TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL,
                    artist_id TEXT NOT NULL,
                    completed_at TEXT NOT NULL,
                    status TEXT NOT NULL,
                    notes TEXT,
                    actual_duration_minutes INTEGER,
                    FOREIGN KEY (task_id) REFERENCES routines(id)
                )
            """)
            
            # Tabla de protocolos
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS protocols (
                    id TEXT PRIMARY KEY,
                    artist_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    description TEXT,
                    category TEXT NOT NULL,
                    priority TEXT NOT NULL,
                    rules TEXT,
                    do_s TEXT,
                    dont_s TEXT,
                    context TEXT,
                    applicable_events TEXT,
                    notes TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            
            # Tabla de cumplimiento de protocolos
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS protocol_compliance (
                    id TEXT PRIMARY KEY,
                    protocol_id TEXT NOT NULL,
                    artist_id TEXT NOT NULL,
                    event_id TEXT,
                    checked_at TEXT NOT NULL,
                    is_compliant INTEGER NOT NULL,
                    notes TEXT,
                    violations TEXT,
                    FOREIGN KEY (protocol_id) REFERENCES protocols(id)
                )
            """)
            
            # Tabla de items de vestimenta
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS wardrobe_items (
                    id TEXT PRIMARY KEY,
                    artist_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    category TEXT NOT NULL,
                    color TEXT NOT NULL,
                    brand TEXT,
                    size TEXT,
                    season TEXT DEFAULT 'all_season',
                    dress_codes TEXT,
                    notes TEXT,
                    image_url TEXT,
                    last_worn TEXT,
                    created_at TEXT NOT NULL
                )
            """)
            
            # Tabla de outfits
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS outfits (
                    id TEXT PRIMARY KEY,
                    artist_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    items TEXT NOT NULL,
                    dress_code TEXT NOT NULL,
                    occasion TEXT NOT NULL,
                    season TEXT DEFAULT 'all_season',
                    notes TEXT,
                    image_url TEXT,
                    last_worn TEXT,
                    created_at TEXT NOT NULL
                )
            """)
            
            # Tabla de recomendaciones de vestimenta
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS wardrobe_recommendations (
                    id TEXT PRIMARY KEY,
                    artist_id TEXT NOT NULL,
                    event_id TEXT,
                    occasion TEXT NOT NULL,
                    dress_code TEXT NOT NULL,
                    recommended_outfit_id TEXT,
                    recommended_items TEXT,
                    reasoning TEXT,
                    weather_considerations TEXT,
                    created_at TEXT NOT NULL
                )
            """)
            
            # Índices para mejor rendimiento
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_artist ON events(artist_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_start ON events(start_time)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_routines_artist ON routines(artist_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_protocols_artist ON protocols(artist_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_wardrobe_artist ON wardrobe_items(artist_id)")
            
            self._logger.info("Database initialized successfully")
    
    def save_event(self, artist_id: str, event_data: Dict[str, Any]) -> bool:
        """Guardar evento en base de datos."""
        try:
            with self.get_cursor() as cursor:
                cursor.execute("""
                    INSERT OR REPLACE INTO events (
                        id, artist_id, title, description, event_type,
                        start_time, end_time, location, attendees, reminders,
                        protocol_requirements, wardrobe_requirements, notes,
                        created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    event_data['id'],
                    artist_id,
                    event_data['title'],
                    event_data.get('description'),
                    event_data['event_type'],
                    event_data['start_time'],
                    event_data['end_time'],
                    event_data.get('location'),
                    json.dumps(event_data.get('attendees', [])),
                    json.dumps(event_data.get('reminders', [])),
                    json.dumps(event_data.get('protocol_requirements', [])),
                    event_data.get('wardrobe_requirements'),
                    event_data.get('notes'),
                    datetime.now().isoformat(),
                    datetime.now().isoformat()
                ))
            return True
        except Exception as e:
            self._logger.error(f"Error saving event: {str(e)}")
            return False
    
    def load_events(self, artist_id: str) -> List[Dict[str, Any]]:
        """Cargar eventos de un artista."""
        try:
            with self.get_cursor() as cursor:
                cursor.execute("SELECT * FROM events WHERE artist_id = ?", (artist_id,))
                rows = cursor.fetchall()
                
                events = []
                for row in rows:
                    event = dict(row)
                    event['attendees'] = json.loads(event.get('attendees') or '[]')
                    event['reminders'] = json.loads(event.get('reminders') or '[]')
                    event['protocol_requirements'] = json.loads(event.get('protocol_requirements') or '[]')
                    events.append(event)
                
                return events
        except Exception as e:
            self._logger.error(f"Error loading events: {str(e)}")
            return []
    
    def save_routine(self, artist_id: str, routine_data: Dict[str, Any]) -> bool:
        """Guardar rutina en base de datos."""
        try:
            with self.get_cursor() as cursor:
                cursor.execute("""
                    INSERT OR REPLACE INTO routines (
                        id, artist_id, title, description, routine_type,
                        scheduled_time, duration_minutes, priority, days_of_week,
                        is_required, reminders, notes, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    routine_data['id'],
                    artist_id,
                    routine_data['title'],
                    routine_data.get('description'),
                    routine_data['routine_type'],
                    routine_data['scheduled_time'],
                    routine_data['duration_minutes'],
                    routine_data.get('priority', 5),
                    json.dumps(routine_data.get('days_of_week', [])),
                    int(routine_data.get('is_required', True)),
                    json.dumps(routine_data.get('reminders', [])),
                    routine_data.get('notes'),
                    datetime.now().isoformat(),
                    datetime.now().isoformat()
                ))
            return True
        except Exception as e:
            self._logger.error(f"Error saving routine: {str(e)}")
            return False
    
    def load_routines(self, artist_id: str) -> List[Dict[str, Any]]:
        """Cargar rutinas de un artista."""
        try:
            with self.get_cursor() as cursor:
                cursor.execute("SELECT * FROM routines WHERE artist_id = ?", (artist_id,))
                rows = cursor.fetchall()
                
                routines = []
                for row in rows:
                    routine = dict(row)
                    routine['days_of_week'] = json.loads(routine.get('days_of_week') or '[]')
                    routine['reminders'] = json.loads(routine.get('reminders') or '[]')
                    routine['is_required'] = bool(routine.get('is_required', 1))
                    routines.append(routine)
                
                return routines
        except Exception as e:
            self._logger.error(f"Error loading routines: {str(e)}")
            return []
    
    def save_routine_completion(self, artist_id: str, completion_data: Dict[str, Any]) -> bool:
        """Guardar completación de rutina."""
        try:
            import uuid
            with self.get_cursor() as cursor:
                cursor.execute("""
                    INSERT INTO routine_completions (
                        id, task_id, artist_id, completed_at, status,
                        notes, actual_duration_minutes
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    str(uuid.uuid4()),
                    completion_data['task_id'],
                    artist_id,
                    completion_data['completed_at'],
                    completion_data['status'],
                    completion_data.get('notes'),
                    completion_data.get('actual_duration_minutes')
                ))
            return True
        except Exception as e:
            self._logger.error(f"Error saving routine completion: {str(e)}")
            return False
    
    def save_protocol(self, artist_id: str, protocol_data: Dict[str, Any]) -> bool:
        """Guardar protocolo en base de datos."""
        try:
            with self.get_cursor() as cursor:
                cursor.execute("""
                    INSERT OR REPLACE INTO protocols (
                        id, artist_id, title, description, category,
                        priority, rules, do_s, dont_s, context,
                        applicable_events, notes, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    protocol_data['id'],
                    artist_id,
                    protocol_data['title'],
                    protocol_data.get('description'),
                    protocol_data['category'],
                    protocol_data['priority'],
                    json.dumps(protocol_data.get('rules', [])),
                    json.dumps(protocol_data.get('do_s', [])),
                    json.dumps(protocol_data.get('dont_s', [])),
                    protocol_data.get('context'),
                    json.dumps(protocol_data.get('applicable_events', [])),
                    protocol_data.get('notes'),
                    datetime.now().isoformat(),
                    datetime.now().isoformat()
                ))
            return True
        except Exception as e:
            self._logger.error(f"Error saving protocol: {str(e)}")
            return False
    
    def load_protocols(self, artist_id: str) -> List[Dict[str, Any]]:
        """Cargar protocolos de un artista."""
        try:
            with self.get_cursor() as cursor:
                cursor.execute("SELECT * FROM protocols WHERE artist_id = ?", (artist_id,))
                rows = cursor.fetchall()
                
                protocols = []
                for row in rows:
                    protocol = dict(row)
                    protocol['rules'] = json.loads(protocol.get('rules') or '[]')
                    protocol['do_s'] = json.loads(protocol.get('do_s') or '[]')
                    protocol['dont_s'] = json.loads(protocol.get('dont_s') or '[]')
                    protocol['applicable_events'] = json.loads(protocol.get('applicable_events') or '[]')
                    protocols.append(protocol)
                
                return protocols
        except Exception as e:
            self._logger.error(f"Error loading protocols: {str(e)}")
            return []
    
    def save_wardrobe_item(self, artist_id: str, item_data: Dict[str, Any]) -> bool:
        """Guardar item de vestimenta en base de datos."""
        try:
            with self.get_cursor() as cursor:
                cursor.execute("""
                    INSERT OR REPLACE INTO wardrobe_items (
                        id, artist_id, name, category, color, brand, size,
                        season, dress_codes, notes, image_url, last_worn, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    item_data['id'],
                    artist_id,
                    item_data['name'],
                    item_data['category'],
                    item_data['color'],
                    item_data.get('brand'),
                    item_data.get('size'),
                    item_data.get('season', 'all_season'),
                    json.dumps(item_data.get('dress_codes', [])),
                    item_data.get('notes'),
                    item_data.get('image_url'),
                    item_data.get('last_worn'),
                    datetime.now().isoformat()
                ))
            return True
        except Exception as e:
            self._logger.error(f"Error saving wardrobe item: {str(e)}")
            return False
    
    def load_wardrobe_items(self, artist_id: str) -> List[Dict[str, Any]]:
        """Cargar items de vestimenta de un artista."""
        try:
            with self.get_cursor() as cursor:
                cursor.execute("SELECT * FROM wardrobe_items WHERE artist_id = ?", (artist_id,))
                rows = cursor.fetchall()
                
                items = []
                for row in rows:
                    item = dict(row)
                    item['dress_codes'] = json.loads(item.get('dress_codes') or '[]')
                    items.append(item)
                
                return items
        except Exception as e:
            self._logger.error(f"Error loading wardrobe items: {str(e)}")
            return []
    
    def close(self):
        """Cerrar conexiones."""
        if hasattr(self._local, 'connection'):
            self._local.connection.close()
            delattr(self._local, 'connection')

