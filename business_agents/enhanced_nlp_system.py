"""
Enhanced NLP System
===================

Sistema NLP mejorado con optimizaciones avanzadas, caché inteligente,
métricas en tiempo real, análisis de tendencias y procesamiento asíncrono.
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime
from dataclasses import dataclass
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import enhanced components
from .nlp_cache import nlp_cache, IntelligentNLPCache
from .nlp_metrics import nlp_monitoring, NLPMonitoringSystem
from .nlp_trends import nlp_trend_analyzer, NLPTrendAnalyzer
from .advanced_nlp_system import advanced_nlp_system, AdvancedNLPSystem
from .nlp_config import nlp_config

logger = logging.getLogger(__name__)

@dataclass
class EnhancedAnalysisResult:
    """Resultado de análisis mejorado."""
    text: str
    language: str
    analysis: Dict[str, Any]
    processing_time: float
    cache_hit: bool = False
    quality_score: float = 0.0
    confidence: float = 0.0
    recommendations: List[str] = None
    trends: Dict[str, Any] = None
    anomalies: List[Dict[str, Any]] = None
    predictions: List[Dict[str, Any]] = None
    timestamp: datetime = None

class EnhancedNLPSystem:
    """Sistema NLP mejorado con optimizaciones avanzadas."""
    
    def __init__(self):
        """Initialize enhanced NLP system."""
        self.base_system = advanced_nlp_system
        self.cache = nlp_cache
        self.monitoring = nlp_monitoring
        self.trend_analyzer = nlp_trend_analyzer
        
        # Performance optimization
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.batch_size = nlp_config.batch_size
        self.max_concurrent = nlp_config.max_concurrent_requests
        
        # Quality assurance
        self.quality_threshold = 0.7
        self.confidence_threshold = 0.8
        
        # System state
        self.is_initialized = False
        self.initialization_time = None
        
        # Background tasks
        self._background_tasks = []
        self._running = False
    
    async def initialize(self):
        """Initialize enhanced NLP system."""
        if self.is_initialized:
            return
        
        start_time = time.time()
        
        try:
            logger.info("Initializing Enhanced NLP System...")
            
            # Initialize base system
            await self.base_system.initialize()
            
            # Initialize cache
            await self.cache.start()
            
            # Initialize monitoring
            await self.monitoring.start_monitoring()
            
            # Start background tasks
            await self._start_background_tasks()
            
            self.is_initialized = True
            self.initialization_time = time.time() - start_time
            
            # Record initialization metrics
            await self.monitoring.record_request(
                task="system_initialization",
                processing_time=self.initialization_time,
                success=True
            )
            
            logger.info(f"Enhanced NLP System initialized in {self.initialization_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Failed to initialize Enhanced NLP System: {e}")
            await self.monitoring.record_request(
                task="system_initialization",
                processing_time=time.time() - start_time,
                success=False,
                error_type="initialization_error"
            )
            raise
    
    async def _start_background_tasks(self):
        """Start background optimization tasks."""
        self._running = True
        
        # Cache optimization task
        cache_task = asyncio.create_task(self._cache_optimization_loop())
        self._background_tasks.append(cache_task)
        
        # Metrics collection task
        metrics_task = asyncio.create_task(self._metrics_collection_loop())
        self._background_tasks.append(metrics_task)
        
        # Trend analysis task
        trends_task = asyncio.create_task(self._trend_analysis_loop())
        self._background_tasks.append(trends_task)
        
        logger.info("Background tasks started")
    
    async def _cache_optimization_loop(self):
        """Background cache optimization."""
        while self._running:
            try:
                await asyncio.sleep(300)  # Every 5 minutes
                await self.cache.optimize()
                
                # Record cache metrics
                cache_stats = self.cache.get_stats()
                await self.monitoring.record_quality_metrics(
                    task="cache_optimization",
                    accuracy=cache_stats['hit_rate']
                )
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cache optimization error: {e}")
    
    async def _metrics_collection_loop(self):
        """Background metrics collection."""
        while self._running:
            try:
                await asyncio.sleep(60)  # Every minute
                
                # Collect system metrics
                system_metrics = self.monitoring.get_system_metrics()
                
                # Record trend data
                for metric_name, value in system_metrics.items():
                    if isinstance(value, (int, float)):
                        await self.trend_analyzer.record_metric_value(
                            metric_name=metric_name,
                            value=float(value),
                            metadata={'source': 'system_monitoring'}
                        )
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Metrics collection error: {e}")
    
    async def _trend_analysis_loop(self):
        """Background trend analysis."""
        while self._running:
            try:
                await asyncio.sleep(1800)  # Every 30 minutes
                
                # Analyze trends
                trends = await self.trend_analyzer.analyze_trends(hours=24)
                
                # Detect anomalies
                anomalies = await self.trend_analyzer.detect_anomalies(hours=24)
                
                # Generate predictions
                predictions = await self.trend_analyzer.generate_predictions(hours=24)
                
                # Log insights
                insights = self.trend_analyzer.get_insights(hours=24)
                if insights:
                    logger.info(f"Trend insights: {insights}")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Trend analysis error: {e}")
    
    async def analyze_text_enhanced(
        self,
        text: str,
        language: str = "en",
        use_cache: bool = True,
        quality_check: bool = True,
        include_trends: bool = False
    ) -> EnhancedAnalysisResult:
        """Perform enhanced text analysis with optimizations."""
        if not self.is_initialized:
            await self.initialize()
        
        start_time = time.time()
        cache_hit = False
        
        try:
            # Check cache first
            if use_cache:
                cached_result = await self.cache.get(
                    text=text,
                    task="enhanced_analysis",
                    language=language
                )
                if cached_result:
                    cache_hit = True
                    logger.debug("Cache hit for enhanced analysis")
                    return EnhancedAnalysisResult(
                        text=text,
                        language=language,
                        analysis=cached_result,
                        processing_time=time.time() - start_time,
                        cache_hit=True,
                        timestamp=datetime.now()
                    )
            
            # Perform analysis
            analysis_result = await self.base_system.analyze_text_advanced(text, language)
            
            # Quality assessment
            quality_score = 0.0
            confidence = 0.0
            if quality_check:
                quality_score = await self._assess_quality(analysis_result)
                confidence = await self._calculate_confidence(analysis_result)
            
            # Generate recommendations
            recommendations = await self._generate_recommendations(analysis_result, quality_score)
            
            # Trend analysis (if requested)
            trends = None
            anomalies = None
            predictions = None
            if include_trends:
                trends = await self._analyze_trends(analysis_result)
                anomalies = await self._detect_anomalies(analysis_result)
                predictions = await self._generate_predictions(analysis_result)
            
            # Create enhanced result
            result = EnhancedAnalysisResult(
                text=text,
                language=language,
                analysis=analysis_result,
                processing_time=time.time() - start_time,
                cache_hit=cache_hit,
                quality_score=quality_score,
                confidence=confidence,
                recommendations=recommendations,
                trends=trends,
                anomalies=anomalies,
                predictions=predictions,
                timestamp=datetime.now()
            )
            
            # Cache result
            if use_cache and not cache_hit:
                await self.cache.set(
                    text=text,
                    task="enhanced_analysis",
                    value=analysis_result,
                    language=language,
                    ttl=7200  # 2 hours
                )
            
            # Record metrics
            await self.monitoring.record_request(
                task="enhanced_analysis",
                processing_time=result.processing_time,
                success=True,
                language=language,
                text_length=len(text)
            )
            
            # Record quality metrics
            if quality_score > 0:
                await self.monitoring.record_quality_metrics(
                    task="enhanced_analysis",
                    accuracy=quality_score,
                    precision=confidence
                )
            
            # Record trend data
            await self.trend_analyzer.record_metric_value(
                metric_name="processing_time",
                value=result.processing_time,
                metadata={'task': 'enhanced_analysis', 'language': language}
            )
            
            if quality_score > 0:
                await self.trend_analyzer.record_metric_value(
                    metric_name="quality_score",
                    value=quality_score,
                    metadata={'task': 'enhanced_analysis', 'language': language}
                )
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            
            # Record error metrics
            await self.monitoring.record_request(
                task="enhanced_analysis",
                processing_time=processing_time,
                success=False,
                error_type=type(e).__name__,
                language=language,
                text_length=len(text)
            )
            
            logger.error(f"Enhanced analysis failed: {e}")
            raise
    
    async def batch_analyze_enhanced(
        self,
        texts: List[str],
        language: str = "en",
        use_cache: bool = True,
        quality_check: bool = True
    ) -> List[EnhancedAnalysisResult]:
        """Perform enhanced batch analysis."""
        if not self.is_initialized:
            await self.initialize()
        
        start_time = time.time()
        results = []
        
        try:
            # Process in batches
            batch_size = min(self.batch_size, len(texts))
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                
                # Process batch concurrently
                tasks = [
                    self.analyze_text_enhanced(
                        text=text,
                        language=language,
                        use_cache=use_cache,
                        quality_check=quality_check,
                        include_trends=False
                    )
                    for text in batch
                ]
                
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Handle results
                for j, result in enumerate(batch_results):
                    if isinstance(result, Exception):
                        logger.error(f"Batch analysis error for text {i + j}: {result}")
                        # Create error result
                        error_result = EnhancedAnalysisResult(
                            text=batch[j],
                            language=language,
                            analysis={},
                            processing_time=0,
                            quality_score=0,
                            confidence=0,
                            recommendations=["Analysis failed"],
                            timestamp=datetime.now()
                        )
                        results.append(error_result)
                    else:
                        results.append(result)
            
            # Record batch metrics
            total_time = time.time() - start_time
            await self.monitoring.record_request(
                task="batch_analysis",
                processing_time=total_time,
                success=True,
                language=language,
                text_length=sum(len(text) for text in texts)
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Batch analysis failed: {e}")
            raise
    
    async def _assess_quality(self, analysis: Dict[str, Any]) -> float:
        """Assess quality of analysis results."""
        try:
            quality_score = 0.0
            factors = 0
            
            # Sentiment quality
            sentiment = analysis.get('sentiment', {})
            if sentiment.get('ensemble'):
                quality_score += 0.2
                factors += 1
            
            # Entity quality
            entities = analysis.get('entities', [])
            if len(entities) > 0:
                quality_score += 0.2
                factors += 1
            
            # Keyword quality
            keywords = analysis.get('keywords', [])
            if len(keywords) > 0:
                quality_score += 0.2
                factors += 1
            
            # Readability quality
            readability = analysis.get('readability', {})
            if readability.get('average_score', 0) > 0:
                quality_score += 0.2
                factors += 1
            
            # Statistics quality
            stats = analysis.get('statistics', {})
            if stats.get('word_count', 0) > 0:
                quality_score += 0.2
                factors += 1
            
            return quality_score / factors if factors > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Quality assessment failed: {e}")
            return 0.0
    
    async def _calculate_confidence(self, analysis: Dict[str, Any]) -> float:
        """Calculate confidence score for analysis."""
        try:
            confidence = 0.0
            factors = 0
            
            # Sentiment confidence
            sentiment = analysis.get('sentiment', {})
            if sentiment.get('ensemble', {}).get('confidence'):
                confidence += sentiment['ensemble']['confidence']
                factors += 1
            
            # Entity confidence
            entities = analysis.get('entities', [])
            if entities:
                entity_confidences = [e.get('confidence', 0) for e in entities if 'confidence' in e]
                if entity_confidences:
                    confidence += sum(entity_confidences) / len(entity_confidences)
                    factors += 1
            
            # Overall confidence
            return confidence / factors if factors > 0 else 0.5
            
        except Exception as e:
            logger.error(f"Confidence calculation failed: {e}")
            return 0.5
    
    async def _generate_recommendations(
        self,
        analysis: Dict[str, Any],
        quality_score: float
    ) -> List[str]:
        """Generate recommendations based on analysis."""
        recommendations = []
        
        try:
            # Quality-based recommendations
            if quality_score < self.quality_threshold:
                recommendations.append("Consider improving text quality for better analysis results")
            
            # Sentiment recommendations
            sentiment = analysis.get('sentiment', {})
            if sentiment.get('ensemble', {}).get('sentiment') == 'negative':
                recommendations.append("Consider revising content to improve sentiment")
            
            # Readability recommendations
            readability = analysis.get('readability', {})
            if readability.get('average_score', 0) < 50:
                recommendations.append("Simplify language to improve readability")
            
            # Keyword recommendations
            keywords = analysis.get('keywords', [])
            if len(keywords) < 5:
                recommendations.append("Add more relevant keywords to improve SEO")
            
            # Entity recommendations
            entities = analysis.get('entities', [])
            if not any(e.get('label') == 'ORG' for e in entities):
                recommendations.append("Consider mentioning relevant organizations")
            
            return recommendations[:5]  # Limit to top 5
            
        except Exception as e:
            logger.error(f"Recommendation generation failed: {e}")
            return ["Analysis completed successfully"]
    
    async def _analyze_trends(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze trends for the analysis."""
        try:
            # Get trend analysis for relevant metrics
            trends = await self.trend_analyzer.analyze_trends(hours=24)
            
            return {
                'processing_trends': trends.get('processing_time'),
                'quality_trends': trends.get('quality_score'),
                'sentiment_trends': trends.get('sentiment_accuracy')
            }
            
        except Exception as e:
            logger.error(f"Trend analysis failed: {e}")
            return {}
    
    async def _detect_anomalies(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect anomalies in the analysis."""
        try:
            anomalies = await self.trend_analyzer.detect_anomalies(hours=24)
            
            return [
                {
                    'metric': metric,
                    'anomalies': [
                        {
                            'type': anomaly.anomaly_type.value,
                            'severity': anomaly.severity,
                            'description': anomaly.description,
                            'timestamp': anomaly.timestamp.isoformat()
                        }
                        for anomaly in metric_anomalies
                    ]
                }
                for metric, metric_anomalies in anomalies.items()
            ]
            
        except Exception as e:
            logger.error(f"Anomaly detection failed: {e}")
            return []
    
    async def _generate_predictions(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate predictions based on analysis."""
        try:
            predictions = await self.trend_analyzer.generate_predictions(hours=24)
            
            return [
                {
                    'metric': metric,
                    'predictions': [
                        {
                            'timestamp': pred.timestamp.isoformat(),
                            'predicted_value': pred.predicted_value,
                            'confidence': pred.confidence,
                            'model': pred.model_used
                        }
                        for pred in metric_predictions
                    ]
                }
                for metric, metric_predictions in predictions.items()
            ]
            
        except Exception as e:
            logger.error(f"Prediction generation failed: {e}")
            return []
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        try:
            # Base system health
            base_health = await self.base_system.get_system_health()
            
            # Cache status
            cache_stats = self.cache.get_stats()
            cache_memory = self.cache.get_memory_usage()
            
            # Monitoring status
            performance_metrics = self.monitoring.get_performance_metrics()
            system_metrics = self.monitoring.get_system_metrics()
            health_status = self.monitoring.get_health_status()
            
            # Trend analysis
            trend_summary = self.trend_analyzer.get_trend_summary()
            anomaly_summary = self.trend_analyzer.get_anomaly_summary()
            prediction_summary = self.trend_analyzer.get_prediction_summary()
            
            return {
                'system': {
                    'initialized': self.is_initialized,
                    'initialization_time': self.initialization_time,
                    'base_system_health': base_health
                },
                'cache': {
                    'stats': cache_stats,
                    'memory_usage': cache_memory
                },
                'monitoring': {
                    'performance': performance_metrics,
                    'system': system_metrics,
                    'health': health_status
                },
                'trends': {
                    'summary': trend_summary,
                    'anomalies': anomaly_summary,
                    'predictions': prediction_summary
                },
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get system status: {e}")
            return {'error': str(e), 'timestamp': datetime.now().isoformat()}
    
    async def shutdown(self):
        """Shutdown enhanced NLP system."""
        try:
            logger.info("Shutting down Enhanced NLP System...")
            
            # Stop background tasks
            self._running = False
            for task in self._background_tasks:
                task.cancel()
            
            # Stop components
            await self.cache.stop()
            await self.monitoring.stop_monitoring()
            
            # Shutdown executor
            self.executor.shutdown(wait=True)
            
            logger.info("Enhanced NLP System shutdown complete")
            
        except Exception as e:
            logger.error(f"Shutdown error: {e}")

# Global enhanced NLP system instance
enhanced_nlp_system = EnhancedNLPSystem()












