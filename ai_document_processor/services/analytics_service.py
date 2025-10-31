"""
Servicio de Analytics y Métricas
===============================

Servicio para análisis de uso, métricas de rendimiento y estadísticas del sistema.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import json
import os
from pathlib import Path
import sqlite3
from contextlib import asynccontextmanager

from models.document_models import DocumentAnalysis, ProfessionalFormat, DocumentProcessingResponse

logger = logging.getLogger(__name__)

@dataclass
class ProcessingMetrics:
    """Métricas de procesamiento"""
    total_documents: int
    successful_documents: int
    failed_documents: int
    average_processing_time: float
    total_processing_time: float
    success_rate: float

@dataclass
class FormatUsageStats:
    """Estadísticas de uso por formato"""
    format_name: str
    usage_count: int
    success_rate: float
    average_time: float

@dataclass
class LanguageStats:
    """Estadísticas por idioma"""
    language: str
    document_count: int
    average_confidence: float
    most_common_area: str

@dataclass
class SystemPerformance:
    """Rendimiento del sistema"""
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    active_connections: int
    queue_size: int
    response_time_avg: float

@dataclass
class UserActivity:
    """Actividad del usuario"""
    timestamp: datetime
    endpoint: str
    method: str
    response_time: float
    status_code: int
    user_agent: str
    ip_address: str

class AnalyticsService:
    """Servicio de analytics y métricas"""
    
    def __init__(self, db_path: str = "analytics.db"):
        self.db_path = db_path
        self.db_connection = None
        self.metrics_cache = {}
        self.cache_ttl = 300  # 5 minutos
        
    async def initialize(self):
        """Inicializa el servicio de analytics"""
        logger.info("Inicializando servicio de analytics...")
        
        # Crear directorio de datos si no existe
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)
        
        # Inicializar base de datos
        await self._init_database()
        
        # Inicializar métricas
        await self._load_initial_metrics()
        
        logger.info("Servicio de analytics inicializado")
    
    async def _init_database(self):
        """Inicializa la base de datos de analytics"""
        try:
            self.db_connection = sqlite3.connect(self.db_path, check_same_thread=False)
            cursor = self.db_connection.cursor()
            
            # Tabla de procesamiento de documentos
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS document_processing (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    filename TEXT,
                    file_size INTEGER,
                    file_type TEXT,
                    target_format TEXT,
                    language TEXT,
                    processing_time REAL,
                    success BOOLEAN,
                    error_message TEXT,
                    classification_area TEXT,
                    classification_category TEXT,
                    classification_confidence REAL,
                    word_count INTEGER,
                    character_count INTEGER
                )
            """)
            
            # Tabla de actividad de usuarios
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_activity (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    endpoint TEXT,
                    method TEXT,
                    response_time REAL,
                    status_code INTEGER,
                    user_agent TEXT,
                    ip_address TEXT,
                    request_size INTEGER,
                    response_size INTEGER
                )
            """)
            
            # Tabla de rendimiento del sistema
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS system_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    cpu_usage REAL,
                    memory_usage REAL,
                    disk_usage REAL,
                    active_connections INTEGER,
                    queue_size INTEGER,
                    response_time_avg REAL
                )
            """)
            
            # Tabla de análisis avanzado
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS advanced_analysis (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    document_id INTEGER,
                    sentiment_positive REAL,
                    sentiment_negative REAL,
                    sentiment_neutral REAL,
                    sentiment_overall TEXT,
                    entities_count INTEGER,
                    topics_count INTEGER,
                    readability_score REAL,
                    complexity_level TEXT,
                    FOREIGN KEY (document_id) REFERENCES document_processing (id)
                )
            """)
            
            # Tabla de traducciones
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS translations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    document_id INTEGER,
                    source_language TEXT,
                    target_language TEXT,
                    translation_confidence REAL,
                    translation_method TEXT,
                    quality_score REAL,
                    FOREIGN KEY (document_id) REFERENCES document_processing (id)
                )
            """)
            
            self.db_connection.commit()
            logger.info("Base de datos de analytics inicializada")
            
        except Exception as e:
            logger.error(f"Error inicializando base de datos: {e}")
            raise
    
    async def _load_initial_metrics(self):
        """Carga métricas iniciales"""
        try:
            # Cargar métricas desde la base de datos
            await self._refresh_metrics_cache()
        except Exception as e:
            logger.warning(f"Error cargando métricas iniciales: {e}")
    
    async def _refresh_metrics_cache(self):
        """Actualiza el cache de métricas"""
        try:
            cursor = self.db_connection.cursor()
            
            # Métricas de procesamiento
            cursor.execute("""
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful,
                    SUM(CASE WHEN success = 0 THEN 1 ELSE 0 END) as failed,
                    AVG(processing_time) as avg_time,
                    SUM(processing_time) as total_time
                FROM document_processing
            """)
            
            result = cursor.fetchone()
            if result:
                total, successful, failed, avg_time, total_time = result
                success_rate = (successful / total * 100) if total > 0 else 0
                
                self.metrics_cache['processing'] = ProcessingMetrics(
                    total_documents=total or 0,
                    successful_documents=successful or 0,
                    failed_documents=failed or 0,
                    average_processing_time=avg_time or 0.0,
                    total_processing_time=total_time or 0.0,
                    success_rate=success_rate
                )
            
            # Estadísticas por formato
            cursor.execute("""
                SELECT 
                    target_format,
                    COUNT(*) as count,
                    AVG(CASE WHEN success = 1 THEN 1.0 ELSE 0.0 END) * 100 as success_rate,
                    AVG(processing_time) as avg_time
                FROM document_processing
                GROUP BY target_format
            """)
            
            format_stats = []
            for row in cursor.fetchall():
                format_name, count, success_rate, avg_time = row
                format_stats.append(FormatUsageStats(
                    format_name=format_name,
                    usage_count=count,
                    success_rate=success_rate or 0.0,
                    average_time=avg_time or 0.0
                ))
            
            self.metrics_cache['format_usage'] = format_stats
            
            # Estadísticas por idioma
            cursor.execute("""
                SELECT 
                    language,
                    COUNT(*) as count,
                    AVG(classification_confidence) as avg_confidence,
                    classification_area
                FROM document_processing
                WHERE success = 1
                GROUP BY language, classification_area
                ORDER BY count DESC
            """)
            
            language_stats = {}
            for row in cursor.fetchall():
                language, count, avg_confidence, area = row
                if language not in language_stats:
                    language_stats[language] = LanguageStats(
                        language=language,
                        document_count=0,
                        average_confidence=0.0,
                        most_common_area=""
                    )
                
                language_stats[language].document_count += count
                language_stats[language].average_confidence = avg_confidence or 0.0
                language_stats[language].most_common_area = area or ""
            
            self.metrics_cache['language_stats'] = list(language_stats.values())
            
        except Exception as e:
            logger.error(f"Error actualizando cache de métricas: {e}")
    
    async def log_document_processing(
        self, 
        filename: str,
        file_size: int,
        file_type: str,
        target_format: str,
        language: str,
        processing_time: float,
        success: bool,
        error_message: Optional[str] = None,
        analysis: Optional[DocumentAnalysis] = None
    ) -> int:
        """Registra el procesamiento de un documento"""
        try:
            cursor = self.db_connection.cursor()
            
            # Insertar registro de procesamiento
            cursor.execute("""
                INSERT INTO document_processing (
                    filename, file_size, file_type, target_format, language,
                    processing_time, success, error_message, classification_area,
                    classification_category, classification_confidence, word_count, character_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                filename, file_size, file_type, target_format, language,
                processing_time, success, error_message,
                analysis.area.value if analysis else None,
                analysis.category.value if analysis else None,
                analysis.confidence if analysis else None,
                analysis.word_count if analysis else None,
                len(analysis.summary) if analysis and analysis.summary else None
            ))
            
            document_id = cursor.lastrowid
            self.db_connection.commit()
            
            # Actualizar cache
            await self._refresh_metrics_cache()
            
            logger.info(f"Registrado procesamiento de documento: {filename}")
            return document_id
            
        except Exception as e:
            logger.error(f"Error registrando procesamiento: {e}")
            return -1
    
    async def log_user_activity(
        self,
        endpoint: str,
        method: str,
        response_time: float,
        status_code: int,
        user_agent: str,
        ip_address: str,
        request_size: int = 0,
        response_size: int = 0
    ):
        """Registra actividad del usuario"""
        try:
            cursor = self.db_connection.cursor()
            
            cursor.execute("""
                INSERT INTO user_activity (
                    endpoint, method, response_time, status_code,
                    user_agent, ip_address, request_size, response_size
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                endpoint, method, response_time, status_code,
                user_agent, ip_address, request_size, response_size
            ))
            
            self.db_connection.commit()
            
        except Exception as e:
            logger.error(f"Error registrando actividad de usuario: {e}")
    
    async def log_system_performance(
        self,
        cpu_usage: float,
        memory_usage: float,
        disk_usage: float,
        active_connections: int,
        queue_size: int,
        response_time_avg: float
    ):
        """Registra rendimiento del sistema"""
        try:
            cursor = self.db_connection.cursor()
            
            cursor.execute("""
                INSERT INTO system_performance (
                    cpu_usage, memory_usage, disk_usage, active_connections,
                    queue_size, response_time_avg
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (
                cpu_usage, memory_usage, disk_usage, active_connections,
                queue_size, response_time_avg
            ))
            
            self.db_connection.commit()
            
        except Exception as e:
            logger.error(f"Error registrando rendimiento del sistema: {e}")
    
    async def log_advanced_analysis(
        self,
        document_id: int,
        sentiment_positive: float,
        sentiment_negative: float,
        sentiment_neutral: float,
        sentiment_overall: str,
        entities_count: int,
        topics_count: int,
        readability_score: float,
        complexity_level: str
    ):
        """Registra análisis avanzado"""
        try:
            cursor = self.db_connection.cursor()
            
            cursor.execute("""
                INSERT INTO advanced_analysis (
                    document_id, sentiment_positive, sentiment_negative, sentiment_neutral,
                    sentiment_overall, entities_count, topics_count, readability_score, complexity_level
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                document_id, sentiment_positive, sentiment_negative, sentiment_neutral,
                sentiment_overall, entities_count, topics_count, readability_score, complexity_level
            ))
            
            self.db_connection.commit()
            
        except Exception as e:
            logger.error(f"Error registrando análisis avanzado: {e}")
    
    async def log_translation(
        self,
        document_id: int,
        source_language: str,
        target_language: str,
        translation_confidence: float,
        translation_method: str,
        quality_score: float
    ):
        """Registra traducción"""
        try:
            cursor = self.db_connection.cursor()
            
            cursor.execute("""
                INSERT INTO translations (
                    document_id, source_language, target_language,
                    translation_confidence, translation_method, quality_score
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (
                document_id, source_language, target_language,
                translation_confidence, translation_method, quality_score
            ))
            
            self.db_connection.commit()
            
        except Exception as e:
            logger.error(f"Error registrando traducción: {e}")
    
    async def get_processing_metrics(self, days: int = 30) -> ProcessingMetrics:
        """Obtiene métricas de procesamiento"""
        try:
            # Verificar cache
            cache_key = f"processing_{days}"
            if cache_key in self.metrics_cache:
                return self.metrics_cache[cache_key]
            
            cursor = self.db_connection.cursor()
            
            # Calcular fecha límite
            limit_date = datetime.now() - timedelta(days=days)
            
            cursor.execute("""
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful,
                    SUM(CASE WHEN success = 0 THEN 1 ELSE 0 END) as failed,
                    AVG(processing_time) as avg_time,
                    SUM(processing_time) as total_time
                FROM document_processing
                WHERE timestamp >= ?
            """, (limit_date,))
            
            result = cursor.fetchone()
            if result:
                total, successful, failed, avg_time, total_time = result
                success_rate = (successful / total * 100) if total > 0 else 0
                
                metrics = ProcessingMetrics(
                    total_documents=total or 0,
                    successful_documents=successful or 0,
                    failed_documents=failed or 0,
                    average_processing_time=avg_time or 0.0,
                    total_processing_time=total_time or 0.0,
                    success_rate=success_rate
                )
                
                # Cachear resultado
                self.metrics_cache[cache_key] = metrics
                return metrics
            
            return ProcessingMetrics(0, 0, 0, 0.0, 0.0, 0.0)
            
        except Exception as e:
            logger.error(f"Error obteniendo métricas de procesamiento: {e}")
            return ProcessingMetrics(0, 0, 0, 0.0, 0.0, 0.0)
    
    async def get_format_usage_stats(self, days: int = 30) -> List[FormatUsageStats]:
        """Obtiene estadísticas de uso por formato"""
        try:
            cache_key = f"format_usage_{days}"
            if cache_key in self.metrics_cache:
                return self.metrics_cache[cache_key]
            
            cursor = self.db_connection.cursor()
            limit_date = datetime.now() - timedelta(days=days)
            
            cursor.execute("""
                SELECT 
                    target_format,
                    COUNT(*) as count,
                    AVG(CASE WHEN success = 1 THEN 1.0 ELSE 0.0 END) * 100 as success_rate,
                    AVG(processing_time) as avg_time
                FROM document_processing
                WHERE timestamp >= ?
                GROUP BY target_format
                ORDER BY count DESC
            """, (limit_date,))
            
            stats = []
            for row in cursor.fetchall():
                format_name, count, success_rate, avg_time = row
                stats.append(FormatUsageStats(
                    format_name=format_name,
                    usage_count=count,
                    success_rate=success_rate or 0.0,
                    average_time=avg_time or 0.0
                ))
            
            self.metrics_cache[cache_key] = stats
            return stats
            
        except Exception as e:
            logger.error(f"Error obteniendo estadísticas de formato: {e}")
            return []
    
    async def get_language_stats(self, days: int = 30) -> List[LanguageStats]:
        """Obtiene estadísticas por idioma"""
        try:
            cache_key = f"language_stats_{days}"
            if cache_key in self.metrics_cache:
                return self.metrics_cache[cache_key]
            
            cursor = self.db_connection.cursor()
            limit_date = datetime.now() - timedelta(days=days)
            
            cursor.execute("""
                SELECT 
                    language,
                    COUNT(*) as count,
                    AVG(classification_confidence) as avg_confidence,
                    classification_area
                FROM document_processing
                WHERE timestamp >= ? AND success = 1
                GROUP BY language
                ORDER BY count DESC
            """, (limit_date,))
            
            language_stats = {}
            for row in cursor.fetchall():
                language, count, avg_confidence, area = row
                if language not in language_stats:
                    language_stats[language] = LanguageStats(
                        language=language,
                        document_count=0,
                        average_confidence=0.0,
                        most_common_area=""
                    )
                
                language_stats[language].document_count += count
                language_stats[language].average_confidence = avg_confidence or 0.0
                language_stats[language].most_common_area = area or ""
            
            stats = list(language_stats.values())
            self.metrics_cache[cache_key] = stats
            return stats
            
        except Exception as e:
            logger.error(f"Error obteniendo estadísticas de idioma: {e}")
            return []
    
    async def get_system_performance(self, hours: int = 24) -> List[SystemPerformance]:
        """Obtiene rendimiento del sistema"""
        try:
            cursor = self.db_connection.cursor()
            limit_time = datetime.now() - timedelta(hours=hours)
            
            cursor.execute("""
                SELECT 
                    cpu_usage, memory_usage, disk_usage, active_connections,
                    queue_size, response_time_avg, timestamp
                FROM system_performance
                WHERE timestamp >= ?
                ORDER BY timestamp DESC
                LIMIT 100
            """, (limit_time,))
            
            performance_data = []
            for row in cursor.fetchall():
                cpu, memory, disk, connections, queue, response_time, timestamp = row
                performance_data.append(SystemPerformance(
                    cpu_usage=cpu or 0.0,
                    memory_usage=memory or 0.0,
                    disk_usage=disk or 0.0,
                    active_connections=connections or 0,
                    queue_size=queue or 0,
                    response_time_avg=response_time or 0.0
                ))
            
            return performance_data
            
        except Exception as e:
            logger.error(f"Error obteniendo rendimiento del sistema: {e}")
            return []
    
    async def get_user_activity_summary(self, days: int = 7) -> Dict[str, Any]:
        """Obtiene resumen de actividad de usuarios"""
        try:
            cursor = self.db_connection.cursor()
            limit_date = datetime.now() - timedelta(days=days)
            
            # Actividad por endpoint
            cursor.execute("""
                SELECT 
                    endpoint,
                    COUNT(*) as requests,
                    AVG(response_time) as avg_response_time,
                    SUM(CASE WHEN status_code >= 400 THEN 1 ELSE 0 END) as errors
                FROM user_activity
                WHERE timestamp >= ?
                GROUP BY endpoint
                ORDER BY requests DESC
            """, (limit_date,))
            
            endpoint_stats = []
            for row in cursor.fetchall():
                endpoint, requests, avg_time, errors = row
                endpoint_stats.append({
                    "endpoint": endpoint,
                    "requests": requests,
                    "avg_response_time": avg_time or 0.0,
                    "errors": errors,
                    "error_rate": (errors / requests * 100) if requests > 0 else 0
                })
            
            # Actividad por día
            cursor.execute("""
                SELECT 
                    DATE(timestamp) as date,
                    COUNT(*) as requests,
                    AVG(response_time) as avg_response_time
                FROM user_activity
                WHERE timestamp >= ?
                GROUP BY DATE(timestamp)
                ORDER BY date DESC
            """, (limit_date,))
            
            daily_stats = []
            for row in cursor.fetchall():
                date, requests, avg_time = row
                daily_stats.append({
                    "date": date,
                    "requests": requests,
                    "avg_response_time": avg_time or 0.0
                })
            
            return {
                "endpoint_stats": endpoint_stats,
                "daily_stats": daily_stats,
                "period_days": days
            }
            
        except Exception as e:
            logger.error(f"Error obteniendo resumen de actividad: {e}")
            return {"endpoint_stats": [], "daily_stats": [], "period_days": days}
    
    async def get_analytics_dashboard_data(self) -> Dict[str, Any]:
        """Obtiene datos completos para dashboard de analytics"""
        try:
            # Obtener métricas en paralelo
            processing_metrics, format_stats, language_stats, user_activity = await asyncio.gather(
                self.get_processing_metrics(30),
                self.get_format_usage_stats(30),
                self.get_language_stats(30),
                self.get_user_activity_summary(7)
            )
            
            return {
                "processing_metrics": asdict(processing_metrics),
                "format_usage": [asdict(stat) for stat in format_stats],
                "language_stats": [asdict(stat) for stat in language_stats],
                "user_activity": user_activity,
                "timestamp": datetime.now().isoformat(),
                "cache_status": {
                    "cached_items": len(self.metrics_cache),
                    "cache_ttl": self.cache_ttl
                }
            }
            
        except Exception as e:
            logger.error(f"Error obteniendo datos del dashboard: {e}")
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def cleanup_old_data(self, days_to_keep: int = 90):
        """Limpia datos antiguos de la base de datos"""
        try:
            cursor = self.db_connection.cursor()
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            
            # Limpiar datos antiguos
            tables = ['document_processing', 'user_activity', 'system_performance', 'advanced_analysis', 'translations']
            
            for table in tables:
                cursor.execute(f"DELETE FROM {table} WHERE timestamp < ?", (cutoff_date,))
                deleted = cursor.rowcount
                logger.info(f"Eliminados {deleted} registros antiguos de {table}")
            
            self.db_connection.commit()
            
            # Vacuum para optimizar base de datos
            cursor.execute("VACUUM")
            
            logger.info(f"Limpieza de datos completada. Manteniendo {days_to_keep} días de datos")
            
        except Exception as e:
            logger.error(f"Error limpiando datos antiguos: {e}")
    
    async def export_analytics_data(self, output_file: str, days: int = 30) -> bool:
        """Exporta datos de analytics a archivo JSON"""
        try:
            dashboard_data = await self.get_analytics_dashboard_data()
            
            # Agregar datos adicionales
            export_data = {
                "export_info": {
                    "export_date": datetime.now().isoformat(),
                    "period_days": days,
                    "version": "2.0.0"
                },
                "analytics_data": dashboard_data
            }
            
            # Guardar archivo
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"Datos de analytics exportados a: {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error exportando datos de analytics: {e}")
            return False
    
    async def close(self):
        """Cierra el servicio de analytics"""
        try:
            if self.db_connection:
                self.db_connection.close()
            logger.info("Servicio de analytics cerrado")
        except Exception as e:
            logger.error(f"Error cerrando servicio de analytics: {e}")


