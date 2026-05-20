"""
Database Manager - Gestor de Base de Datos
=========================================

Gestor de base de datos real y funcional para el sistema ultra refactorizado.
"""

import asyncio
import json
from typing import List, Dict, Any, Optional, Union
from datetime import datetime, timedelta
import sqlite3
import aiosqlite
from contextlib import asynccontextmanager
import logging

from ..core.models import (
    HistoryEntry, ComparisonResult, QualityReport, AnalysisJob, 
    TrendAnalysis, SystemMetrics, ModelType, AnalysisStatus
)

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Gestor de base de datos real y funcional."""
    
    def __init__(self, db_path: str = "ai_history_comparison.db"):
        self.db_path = db_path
        self.connection = None
    
    async def initialize(self):
        """Inicializar la base de datos."""
        try:
            self.connection = await aiosqlite.connect(self.db_path)
            await self._create_tables()
            logger.info("Database initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            raise
    
    async def close(self):
        """Cerrar conexión a la base de datos."""
        if self.connection:
            await self.connection.close()
            logger.info("Database connection closed")
    
    async def health_check(self) -> bool:
        """Verificar salud de la base de datos."""
        try:
            if not self.connection:
                return False
            
            async with self.connection.execute("SELECT 1") as cursor:
                await cursor.fetchone()
            return True
        except Exception:
            return False
    
    async def _create_tables(self):
        """Crear tablas de la base de datos."""
        tables = [
            """
            CREATE TABLE IF NOT EXISTS history_entries (
                id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                model_type TEXT NOT NULL,
                model_version TEXT NOT NULL,
                prompt TEXT NOT NULL,
                response TEXT NOT NULL,
                metadata TEXT,
                response_time_ms INTEGER,
                token_count INTEGER,
                cost_usd REAL,
                coherence_score REAL,
                relevance_score REAL,
                creativity_score REAL,
                accuracy_score REAL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS comparison_results (
                id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                entry_1_id TEXT NOT NULL,
                entry_2_id TEXT NOT NULL,
                semantic_similarity REAL NOT NULL,
                lexical_similarity REAL NOT NULL,
                structural_similarity REAL NOT NULL,
                overall_similarity REAL NOT NULL,
                differences TEXT,
                improvements TEXT,
                analysis_details TEXT,
                FOREIGN KEY (entry_1_id) REFERENCES history_entries (id),
                FOREIGN KEY (entry_2_id) REFERENCES history_entries (id)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS quality_reports (
                id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                entry_id TEXT NOT NULL,
                overall_quality REAL NOT NULL,
                coherence REAL NOT NULL,
                relevance REAL NOT NULL,
                creativity REAL NOT NULL,
                accuracy REAL NOT NULL,
                clarity REAL NOT NULL,
                recommendations TEXT,
                strengths TEXT,
                weaknesses TEXT,
                analysis_method TEXT,
                confidence_score REAL NOT NULL,
                FOREIGN KEY (entry_id) REFERENCES history_entries (id)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS analysis_jobs (
                id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                job_type TEXT NOT NULL,
                status TEXT NOT NULL,
                parameters TEXT,
                target_entries TEXT,
                results TEXT,
                error_message TEXT,
                created_by TEXT,
                priority INTEGER DEFAULT 1
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS trend_analyses (
                id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                start_date TEXT NOT NULL,
                end_date TEXT NOT NULL,
                model_performance_trends TEXT,
                quality_trends TEXT,
                usage_trends TEXT,
                key_insights TEXT,
                recommendations TEXT,
                total_entries_analyzed INTEGER DEFAULT 0,
                confidence_level REAL NOT NULL
            )
            """
        ]
        
        for table_sql in tables:
            await self.connection.execute(table_sql)
        
        await self.connection.commit()
        
        # Crear índices para mejorar rendimiento
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_history_timestamp ON history_entries(timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_history_model_type ON history_entries(model_type)",
            "CREATE INDEX IF NOT EXISTS idx_comparison_entry_1 ON comparison_results(entry_1_id)",
            "CREATE INDEX IF NOT EXISTS idx_comparison_entry_2 ON comparison_results(entry_2_id)",
            "CREATE INDEX IF NOT EXISTS idx_quality_entry_id ON quality_reports(entry_id)",
            "CREATE INDEX IF NOT EXISTS idx_jobs_status ON analysis_jobs(status)",
            "CREATE INDEX IF NOT EXISTS idx_jobs_type ON analysis_jobs(job_type)"
        ]
        
        for index_sql in indexes:
            await self.connection.execute(index_sql)
        
        await self.connection.commit()
    
    # Métodos para HistoryEntry
    async def create_history_entry(self, entry: HistoryEntry) -> HistoryEntry:
        """Crear una entrada de historial."""
        try:
            await self.connection.execute(
                """
                INSERT INTO history_entries (
                    id, timestamp, model_type, model_version, prompt, response,
                    metadata, response_time_ms, token_count, cost_usd,
                    coherence_score, relevance_score, creativity_score, accuracy_score
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    entry.id,
                    entry.timestamp.isoformat(),
                    entry.model_type.value,
                    entry.model_version,
                    entry.prompt,
                    entry.response,
                    json.dumps(entry.metadata),
                    entry.response_time_ms,
                    entry.token_count,
                    entry.cost_usd,
                    entry.coherence_score,
                    entry.relevance_score,
                    entry.creativity_score,
                    entry.accuracy_score
                )
            )
            await self.connection.commit()
            return entry
        except Exception as e:
            logger.error(f"Error creating history entry: {e}")
            raise
    
    async def get_history_entry(self, entry_id: str) -> Optional[HistoryEntry]:
        """Obtener una entrada de historial por ID."""
        try:
            async with self.connection.execute(
                "SELECT * FROM history_entries WHERE id = ?",
                (entry_id,)
            ) as cursor:
                row = await cursor.fetchone()
                if row:
                    return self._row_to_history_entry(row)
                return None
        except Exception as e:
            logger.error(f"Error getting history entry {entry_id}: {e}")
            raise
    
    async def get_history_entries(
        self,
        skip: int = 0,
        limit: int = 100,
        model_type: Optional[ModelType] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[HistoryEntry]:
        """Obtener entradas de historial con filtros."""
        try:
            query = "SELECT * FROM history_entries WHERE 1=1"
            params = []
            
            if model_type:
                query += " AND model_type = ?"
                params.append(model_type.value)
            
            if start_date:
                query += " AND timestamp >= ?"
                params.append(start_date.isoformat())
            
            if end_date:
                query += " AND timestamp <= ?"
                params.append(end_date.isoformat())
            
            query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
            params.extend([limit, skip])
            
            async with self.connection.execute(query, params) as cursor:
                rows = await cursor.fetchall()
                return [self._row_to_history_entry(row) for row in rows]
        except Exception as e:
            logger.error(f"Error getting history entries: {e}")
            raise
    
    async def delete_history_entry(self, entry_id: str) -> bool:
        """Eliminar una entrada de historial."""
        try:
            cursor = await self.connection.execute(
                "DELETE FROM history_entries WHERE id = ?",
                (entry_id,)
            )
            await self.connection.commit()
            return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"Error deleting history entry {entry_id}: {e}")
            raise
    
    def _row_to_history_entry(self, row) -> HistoryEntry:
        """Convertir fila de base de datos a HistoryEntry."""
        return HistoryEntry(
            id=row[0],
            timestamp=datetime.fromisoformat(row[1]),
            model_type=ModelType(row[2]),
            model_version=row[3],
            prompt=row[4],
            response=row[5],
            metadata=json.loads(row[6]) if row[6] else {},
            response_time_ms=row[7],
            token_count=row[8],
            cost_usd=row[9],
            coherence_score=row[10],
            relevance_score=row[11],
            creativity_score=row[12],
            accuracy_score=row[13]
        )
    
    # Métodos para ComparisonResult
    async def create_comparison(self, comparison: ComparisonResult) -> ComparisonResult:
        """Crear una comparación."""
        try:
            await self.connection.execute(
                """
                INSERT INTO comparison_results (
                    id, timestamp, entry_1_id, entry_2_id,
                    semantic_similarity, lexical_similarity, structural_similarity, overall_similarity,
                    differences, improvements, analysis_details
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    comparison.id,
                    comparison.timestamp.isoformat(),
                    comparison.entry_1_id,
                    comparison.entry_2_id,
                    comparison.semantic_similarity,
                    comparison.lexical_similarity,
                    comparison.structural_similarity,
                    comparison.overall_similarity,
                    json.dumps(comparison.differences),
                    json.dumps(comparison.improvements),
                    json.dumps(comparison.analysis_details)
                )
            )
            await self.connection.commit()
            return comparison
        except Exception as e:
            logger.error(f"Error creating comparison: {e}")
            raise
    
    async def get_comparison(self, comparison_id: str) -> Optional[ComparisonResult]:
        """Obtener una comparación por ID."""
        try:
            async with self.connection.execute(
                "SELECT * FROM comparison_results WHERE id = ?",
                (comparison_id,)
            ) as cursor:
                row = await cursor.fetchone()
                if row:
                    return self._row_to_comparison_result(row)
                return None
        except Exception as e:
            logger.error(f"Error getting comparison {comparison_id}: {e}")
            raise
    
    async def get_comparisons(
        self,
        skip: int = 0,
        limit: int = 100,
        entry_1_id: Optional[str] = None,
        entry_2_id: Optional[str] = None
    ) -> List[ComparisonResult]:
        """Obtener comparaciones con filtros."""
        try:
            query = "SELECT * FROM comparison_results WHERE 1=1"
            params = []
            
            if entry_1_id:
                query += " AND entry_1_id = ?"
                params.append(entry_1_id)
            
            if entry_2_id:
                query += " AND entry_2_id = ?"
                params.append(entry_2_id)
            
            query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
            params.extend([limit, skip])
            
            async with self.connection.execute(query, params) as cursor:
                rows = await cursor.fetchall()
                return [self._row_to_comparison_result(row) for row in rows]
        except Exception as e:
            logger.error(f"Error getting comparisons: {e}")
            raise
    
    def _row_to_comparison_result(self, row) -> ComparisonResult:
        """Convertir fila de base de datos a ComparisonResult."""
        return ComparisonResult(
            id=row[0],
            timestamp=datetime.fromisoformat(row[1]),
            entry_1_id=row[2],
            entry_2_id=row[3],
            semantic_similarity=row[4],
            lexical_similarity=row[5],
            structural_similarity=row[6],
            overall_similarity=row[7],
            differences=json.loads(row[8]) if row[8] else [],
            improvements=json.loads(row[9]) if row[9] else [],
            analysis_details=json.loads(row[10]) if row[10] else {}
        )
    
    # Métodos para QualityReport
    async def create_quality_report(self, report: QualityReport) -> QualityReport:
        """Crear un reporte de calidad."""
        try:
            await self.connection.execute(
                """
                INSERT INTO quality_reports (
                    id, timestamp, entry_id, overall_quality,
                    coherence, relevance, creativity, accuracy, clarity,
                    recommendations, strengths, weaknesses,
                    analysis_method, confidence_score
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    report.id,
                    report.timestamp.isoformat(),
                    report.entry_id,
                    report.overall_quality,
                    report.coherence,
                    report.relevance,
                    report.creativity,
                    report.accuracy,
                    report.clarity,
                    json.dumps(report.recommendations),
                    json.dumps(report.strengths),
                    json.dumps(report.weaknesses),
                    report.analysis_method,
                    report.confidence_score
                )
            )
            await self.connection.commit()
            return report
        except Exception as e:
            logger.error(f"Error creating quality report: {e}")
            raise
    
    async def get_quality_report(self, report_id: str) -> Optional[QualityReport]:
        """Obtener un reporte de calidad por ID."""
        try:
            async with self.connection.execute(
                "SELECT * FROM quality_reports WHERE id = ?",
                (report_id,)
            ) as cursor:
                row = await cursor.fetchone()
                if row:
                    return self._row_to_quality_report(row)
                return None
        except Exception as e:
            logger.error(f"Error getting quality report {report_id}: {e}")
            raise
    
    async def get_quality_reports(
        self,
        skip: int = 0,
        limit: int = 100,
        entry_id: Optional[str] = None,
        min_quality: Optional[float] = None
    ) -> List[QualityReport]:
        """Obtener reportes de calidad con filtros."""
        try:
            query = "SELECT * FROM quality_reports WHERE 1=1"
            params = []
            
            if entry_id:
                query += " AND entry_id = ?"
                params.append(entry_id)
            
            if min_quality is not None:
                query += " AND overall_quality >= ?"
                params.append(min_quality)
            
            query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
            params.extend([limit, skip])
            
            async with self.connection.execute(query, params) as cursor:
                rows = await cursor.fetchall()
                return [self._row_to_quality_report(row) for row in rows]
        except Exception as e:
            logger.error(f"Error getting quality reports: {e}")
            raise
    
    def _row_to_quality_report(self, row) -> QualityReport:
        """Convertir fila de base de datos a QualityReport."""
        return QualityReport(
            id=row[0],
            timestamp=datetime.fromisoformat(row[1]),
            entry_id=row[2],
            overall_quality=row[3],
            coherence=row[4],
            relevance=row[5],
            creativity=row[6],
            accuracy=row[7],
            clarity=row[8],
            recommendations=json.loads(row[9]) if row[9] else [],
            strengths=json.loads(row[10]) if row[10] else [],
            weaknesses=json.loads(row[11]) if row[11] else [],
            analysis_method=row[12],
            confidence_score=row[13]
        )
    
    # Métodos para AnalysisJob
    async def create_analysis_job(self, job: AnalysisJob) -> AnalysisJob:
        """Crear un trabajo de análisis."""
        try:
            await self.connection.execute(
                """
                INSERT INTO analysis_jobs (
                    id, timestamp, job_type, status, parameters,
                    target_entries, results, error_message, created_by, priority
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job.id,
                    job.timestamp.isoformat(),
                    job.job_type,
                    job.status.value,
                    json.dumps(job.parameters),
                    json.dumps(job.target_entries),
                    json.dumps(job.results) if job.results else None,
                    job.error_message,
                    job.created_by,
                    job.priority
                )
            )
            await self.connection.commit()
            return job
        except Exception as e:
            logger.error(f"Error creating analysis job: {e}")
            raise
    
    async def get_analysis_job(self, job_id: str) -> Optional[AnalysisJob]:
        """Obtener un trabajo de análisis por ID."""
        try:
            async with self.connection.execute(
                "SELECT * FROM analysis_jobs WHERE id = ?",
                (job_id,)
            ) as cursor:
                row = await cursor.fetchone()
                if row:
                    return self._row_to_analysis_job(row)
                return None
        except Exception as e:
            logger.error(f"Error getting analysis job {job_id}: {e}")
            raise
    
    async def get_analysis_jobs(
        self,
        skip: int = 0,
        limit: int = 100,
        status: Optional[AnalysisStatus] = None,
        job_type: Optional[str] = None
    ) -> List[AnalysisJob]:
        """Obtener trabajos de análisis con filtros."""
        try:
            query = "SELECT * FROM analysis_jobs WHERE 1=1"
            params = []
            
            if status:
                query += " AND status = ?"
                params.append(status.value)
            
            if job_type:
                query += " AND job_type = ?"
                params.append(job_type)
            
            query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
            params.extend([limit, skip])
            
            async with self.connection.execute(query, params) as cursor:
                rows = await cursor.fetchall()
                return [self._row_to_analysis_job(row) for row in rows]
        except Exception as e:
            logger.error(f"Error getting analysis jobs: {e}")
            raise
    
    async def update_job_status(
        self, 
        job_id: str, 
        status: AnalysisStatus, 
        error_message: Optional[str] = None
    ):
        """Actualizar estado de un trabajo."""
        try:
            if error_message:
                await self.connection.execute(
                    "UPDATE analysis_jobs SET status = ?, error_message = ? WHERE id = ?",
                    (status.value, error_message, job_id)
                )
            else:
                await self.connection.execute(
                    "UPDATE analysis_jobs SET status = ? WHERE id = ?",
                    (status.value, job_id)
                )
            await self.connection.commit()
        except Exception as e:
            logger.error(f"Error updating job status {job_id}: {e}")
            raise
    
    def _row_to_analysis_job(self, row) -> AnalysisJob:
        """Convertir fila de base de datos a AnalysisJob."""
        return AnalysisJob(
            id=row[0],
            timestamp=datetime.fromisoformat(row[1]),
            job_type=row[2],
            status=AnalysisStatus(row[3]),
            parameters=json.loads(row[4]) if row[4] else {},
            target_entries=json.loads(row[5]) if row[5] else [],
            results=json.loads(row[6]) if row[6] else None,
            error_message=row[7],
            created_by=row[8],
            priority=row[9]
        )
    
    # Métodos para métricas del sistema
    async def get_system_metrics(self) -> SystemMetrics:
        """Obtener métricas del sistema."""
        try:
            # Contar entradas
            async with self.connection.execute("SELECT COUNT(*) FROM history_entries") as cursor:
                total_entries = (await cursor.fetchone())[0]
            
            # Contar comparaciones
            async with self.connection.execute("SELECT COUNT(*) FROM comparison_results") as cursor:
                total_comparisons = (await cursor.fetchone())[0]
            
            # Contar reportes de calidad
            async with self.connection.execute("SELECT COUNT(*) FROM quality_reports") as cursor:
                total_quality_reports = (await cursor.fetchone())[0]
            
            # Contar trabajos activos
            async with self.connection.execute(
                "SELECT COUNT(*) FROM analysis_jobs WHERE status IN ('pending', 'processing')"
            ) as cursor:
                active_jobs = (await cursor.fetchone())[0]
            
            # Calcular métricas promedio
            async with self.connection.execute(
                """
                SELECT 
                    AVG(COALESCE(coherence_score, 0)) as avg_coherence,
                    AVG(COALESCE(relevance_score, 0)) as avg_relevance,
                    AVG(COALESCE(creativity_score, 0)) as avg_creativity,
                    AVG(COALESCE(accuracy_score, 0)) as avg_accuracy,
                    AVG(COALESCE(response_time_ms, 0)) as avg_response_time,
                    SUM(COALESCE(token_count, 0)) as total_tokens,
                    SUM(COALESCE(cost_usd, 0)) as total_cost
                FROM history_entries
                """
            ) as cursor:
                row = await cursor.fetchone()
                avg_coherence = row[0] or 0.0
                avg_relevance = row[1] or 0.0
                avg_creativity = row[2] or 0.0
                avg_accuracy = row[3] or 0.0
                avg_response_time = row[4] or 0.0
                total_tokens = row[5] or 0
                total_cost = row[6] or 0.0
            
            # Calcular calidad general promedio
            overall_quality = (avg_coherence + avg_relevance + avg_creativity + avg_accuracy) / 4
            
            # Estadísticas por modelo
            model_stats = {}
            async with self.connection.execute(
                """
                SELECT 
                    model_type,
                    COUNT(*) as count,
                    AVG(COALESCE(coherence_score, 0)) as avg_coherence,
                    AVG(COALESCE(response_time_ms, 0)) as avg_response_time
                FROM history_entries
                GROUP BY model_type
                """
            ) as cursor:
                rows = await cursor.fetchall()
                for row in rows:
                    model_stats[row[0]] = {
                        "count": row[1],
                        "avg_coherence": row[2] or 0.0,
                        "avg_response_time": row[3] or 0.0
                    }
            
            return SystemMetrics(
                total_entries=total_entries,
                total_comparisons=total_comparisons,
                total_quality_reports=total_quality_reports,
                active_jobs=active_jobs,
                average_quality_score=overall_quality,
                average_coherence_score=avg_coherence,
                average_relevance_score=avg_relevance,
                average_creativity_score=avg_creativity,
                average_accuracy_score=avg_accuracy,
                average_response_time_ms=avg_response_time,
                total_tokens_processed=total_tokens,
                total_cost_usd=total_cost,
                model_usage_stats=model_stats
            )
        except Exception as e:
            logger.error(f"Error getting system metrics: {e}")
            raise
    
    async def get_trend_analysis(
        self,
        start_date: datetime,
        end_date: datetime,
        model_type: Optional[ModelType] = None
    ) -> List[TrendAnalysis]:
        """Obtener análisis de tendencias."""
        try:
            # Implementar análisis de tendencias básico
            # Por ahora, retornar una lista vacía
            return []
        except Exception as e:
            logger.error(f"Error getting trend analysis: {e}")
            raise




