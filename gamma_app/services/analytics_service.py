"""
Gamma App - Analytics Service
Advanced analytics and performance tracking for content generation
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter
import statistics
import redis
from sqlalchemy import create_engine, Column, String, DateTime, JSON, Integer, Float, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

logger = logging.getLogger(__name__)

Base = declarative_base()

class ContentAnalyticsDB(Base):
    """Database model for content analytics"""
    __tablename__ = "content_analytics"
    
    id = Column(String, primary_key=True)
    content_id = Column(String, nullable=False)
    user_id = Column(String, nullable=False)
    content_type = Column(String, nullable=False)
    action = Column(String, nullable=False)  # created, viewed, exported, shared
    timestamp = Column(DateTime, default=datetime.utcnow)
    metadata = Column(JSON, default=dict)

class UserAnalyticsDB(Base):
    """Database model for user analytics"""
    __tablename__ = "user_analytics"
    
    id = Column(String, primary_key=True)
    user_id = Column(String, nullable=False)
    session_id = Column(String, nullable=False)
    action = Column(String, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    duration = Column(Float, default=0.0)
    metadata = Column(JSON, default=dict)

class PerformanceMetricsDB(Base):
    """Database model for performance metrics"""
    __tablename__ = "performance_metrics"
    
    id = Column(String, primary_key=True)
    metric_name = Column(String, nullable=False)
    metric_value = Column(Float, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    metadata = Column(JSON, default=dict)

@dataclass
class ContentMetrics:
    """Content performance metrics"""
    content_id: str
    views: int
    exports: int
    shares: int
    engagement_score: float
    quality_score: float
    creation_time: datetime
    last_activity: datetime
    user_feedback: List[Dict[str, Any]]

@dataclass
class UserMetrics:
    """User activity metrics"""
    user_id: str
    total_content: int
    total_exports: int
    active_sessions: int
    collaboration_time: float
    last_activity: datetime
    content_types: Dict[str, int]
    export_formats: Dict[str, int]

@dataclass
class SystemMetrics:
    """System performance metrics"""
    total_users: int
    active_users: int
    total_content: int
    total_exports: int
    average_response_time: float
    error_rate: float
    uptime: float

class AnalyticsService:
    """
    Advanced analytics service for content generation platform
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize analytics service"""
        self.config = config or {}
        self.metrics_cache = {}
        self.real_time_metrics = defaultdict(list)
        
        # Initialize Redis for real-time analytics
        try:
            self.redis_client = redis.Redis(
                host=self.config.get('redis_host', 'localhost'),
                port=self.config.get('redis_port', 6379),
                db=self.config.get('redis_db', 1),
                decode_responses=True
            )
            self.redis_client.ping()
            logger.info("Redis connection established for analytics")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}")
            self.redis_client = None
        
        # Initialize database
        self._init_database()
        
        logger.info("Analytics Service initialized successfully")

    def _init_database(self):
        """Initialize database connection"""
        try:
            database_url = self.config.get('database_url', 'sqlite:///analytics.db')
            self.engine = create_engine(database_url)
            Base.metadata.create_all(self.engine)
            SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
            self.db_session = SessionLocal()
            logger.info("Analytics database connection established")
        except Exception as e:
            logger.error(f"Analytics database connection failed: {e}")
            self.db_session = None

    async def track_content_creation(self, user_id: str, content_id: str, 
                                   content_type: str, processing_time: float,
                                   quality_score: float, metadata: Dict[str, Any] = None):
        """Track content creation event"""
        try:
            event = {
                'content_id': content_id,
                'user_id': user_id,
                'content_type': content_type,
                'action': 'created',
                'timestamp': datetime.now(),
                'metadata': {
                    'processing_time': processing_time,
                    'quality_score': quality_score,
                    **(metadata or {})
                }
            }
            
            # Save to database
            await self._save_content_analytics(event)
            
            # Update real-time metrics
            await self._update_real_time_metrics('content_created', event)
            
            # Update Redis cache
            if self.redis_client:
                await self._update_redis_metrics('content_created', event)
            
            logger.info(f"Tracked content creation: {content_id}")
            
        except Exception as e:
            logger.error(f"Error tracking content creation: {e}")

    async def track_content_view(self, user_id: str, content_id: str, 
                               view_duration: float = 0.0, metadata: Dict[str, Any] = None):
        """Track content view event"""
        try:
            event = {
                'content_id': content_id,
                'user_id': user_id,
                'action': 'viewed',
                'timestamp': datetime.now(),
                'metadata': {
                    'view_duration': view_duration,
                    **(metadata or {})
                }
            }
            
            await self._save_content_analytics(event)
            await self._update_real_time_metrics('content_viewed', event)
            
            if self.redis_client:
                await self._update_redis_metrics('content_viewed', event)
            
        except Exception as e:
            logger.error(f"Error tracking content view: {e}")

    async def track_content_export(self, user_id: str, content_id: str, 
                                 export_format: str, file_size: int,
                                 metadata: Dict[str, Any] = None):
        """Track content export event"""
        try:
            event = {
                'content_id': content_id,
                'user_id': user_id,
                'action': 'exported',
                'timestamp': datetime.now(),
                'metadata': {
                    'export_format': export_format,
                    'file_size': file_size,
                    **(metadata or {})
                }
            }
            
            await self._save_content_analytics(event)
            await self._update_real_time_metrics('content_exported', event)
            
            if self.redis_client:
                await self._update_redis_metrics('content_exported', event)
            
        except Exception as e:
            logger.error(f"Error tracking content export: {e}")

    async def track_collaboration_session(self, user_id: str, session_id: str, 
                                        action: str, duration: float = 0.0,
                                        metadata: Dict[str, Any] = None):
        """Track collaboration session event"""
        try:
            event = {
                'user_id': user_id,
                'session_id': session_id,
                'action': action,
                'timestamp': datetime.now(),
                'duration': duration,
                'metadata': metadata or {}
            }
            
            await self._save_user_analytics(event)
            await self._update_real_time_metrics('collaboration', event)
            
        except Exception as e:
            logger.error(f"Error tracking collaboration session: {e}")

    async def track_performance_metric(self, metric_name: str, metric_value: float,
                                     metadata: Dict[str, Any] = None):
        """Track system performance metric"""
        try:
            event = {
                'metric_name': metric_name,
                'metric_value': metric_value,
                'timestamp': datetime.now(),
                'metadata': metadata or {}
            }
            
            await self._save_performance_metrics(event)
            
            # Update real-time metrics
            self.real_time_metrics[metric_name].append({
                'value': metric_value,
                'timestamp': datetime.now()
            })
            
            # Keep only last 1000 entries
            if len(self.real_time_metrics[metric_name]) > 1000:
                self.real_time_metrics[metric_name] = self.real_time_metrics[metric_name][-1000:]
            
        except Exception as e:
            logger.error(f"Error tracking performance metric: {e}")

    async def get_dashboard_data(self, user_id: str, time_period: str = "7d") -> Dict[str, Any]:
        """Get dashboard analytics data"""
        try:
            time_delta = self._parse_time_period(time_period)
            start_time = datetime.now() - time_delta
            
            # Get user metrics
            user_metrics = await self._get_user_metrics(user_id, start_time)
            
            # Get content metrics
            content_metrics = await self._get_content_metrics(user_id, start_time)
            
            # Get recent activity
            recent_activity = await self._get_recent_activity(user_id, start_time)
            
            # Get performance metrics
            performance_metrics = await self._get_performance_metrics(start_time)
            
            return {
                'user_metrics': asdict(user_metrics),
                'content_metrics': content_metrics,
                'recent_activity': recent_activity,
                'performance_metrics': performance_metrics,
                'time_period': time_period,
                'generated_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting dashboard data: {e}")
            return {}

    async def get_content_performance(self, user_id: str, content_id: Optional[str] = None,
                                    time_period: str = "30d") -> Dict[str, Any]:
        """Get content performance analytics"""
        try:
            time_delta = self._parse_time_period(time_period)
            start_time = datetime.now() - time_delta
            
            if content_id:
                # Get specific content performance
                metrics = await self._get_content_metrics_by_id(content_id, start_time)
            else:
                # Get all user content performance
                metrics = await self._get_user_content_performance(user_id, start_time)
            
            return {
                'content_performance': metrics,
                'time_period': time_period,
                'generated_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting content performance: {e}")
            return {}

    async def get_collaboration_stats(self, user_id: str, time_period: str = "30d") -> Dict[str, Any]:
        """Get collaboration statistics"""
        try:
            time_delta = self._parse_time_period(time_period)
            start_time = datetime.now() - time_delta
            
            # Get collaboration metrics
            stats = await self._get_collaboration_metrics(user_id, start_time)
            
            return {
                'collaboration_stats': stats,
                'time_period': time_period,
                'generated_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting collaboration stats: {e}")
            return {}

    async def get_system_metrics(self) -> SystemMetrics:
        """Get system-wide metrics"""
        try:
            # Get total users
            total_users = await self._get_total_users()
            active_users = await self._get_active_users()
            
            # Get content metrics
            total_content = await self._get_total_content()
            total_exports = await self._get_total_exports()
            
            # Get performance metrics
            avg_response_time = await self._get_average_response_time()
            error_rate = await self._get_error_rate()
            uptime = await self._get_system_uptime()
            
            return SystemMetrics(
                total_users=total_users,
                active_users=active_users,
                total_content=total_content,
                total_exports=total_exports,
                average_response_time=avg_response_time,
                error_rate=error_rate,
                uptime=uptime
            )
            
        except Exception as e:
            logger.error(f"Error getting system metrics: {e}")
            return SystemMetrics(0, 0, 0, 0, 0.0, 0.0, 0.0)

    async def get_trend_analysis(self, metric_name: str, time_period: str = "30d") -> Dict[str, Any]:
        """Get trend analysis for a specific metric"""
        try:
            time_delta = self._parse_time_period(time_period)
            start_time = datetime.now() - time_delta
            
            # Get historical data
            historical_data = await self._get_historical_metrics(metric_name, start_time)
            
            # Calculate trends
            trend_analysis = self._calculate_trends(historical_data)
            
            return {
                'metric_name': metric_name,
                'trend_analysis': trend_analysis,
                'historical_data': historical_data,
                'time_period': time_period,
                'generated_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting trend analysis: {e}")
            return {}

    async def _save_content_analytics(self, event: Dict[str, Any]):
        """Save content analytics to database"""
        try:
            if self.db_session:
                analytics = ContentAnalyticsDB(
                    id=f"{event['content_id']}_{event['action']}_{int(event['timestamp'].timestamp())}",
                    content_id=event['content_id'],
                    user_id=event['user_id'],
                    content_type=event.get('content_type', ''),
                    action=event['action'],
                    timestamp=event['timestamp'],
                    metadata=event.get('metadata', {})
                )
                self.db_session.add(analytics)
                self.db_session.commit()
        except Exception as e:
            logger.error(f"Error saving content analytics: {e}")

    async def _save_user_analytics(self, event: Dict[str, Any]):
        """Save user analytics to database"""
        try:
            if self.db_session:
                analytics = UserAnalyticsDB(
                    id=f"{event['user_id']}_{event['action']}_{int(event['timestamp'].timestamp())}",
                    user_id=event['user_id'],
                    session_id=event['session_id'],
                    action=event['action'],
                    timestamp=event['timestamp'],
                    duration=event.get('duration', 0.0),
                    metadata=event.get('metadata', {})
                )
                self.db_session.add(analytics)
                self.db_session.commit()
        except Exception as e:
            logger.error(f"Error saving user analytics: {e}")

    async def _save_performance_metrics(self, event: Dict[str, Any]):
        """Save performance metrics to database"""
        try:
            if self.db_session:
                metric = PerformanceMetricsDB(
                    id=f"{event['metric_name']}_{int(event['timestamp'].timestamp())}",
                    metric_name=event['metric_name'],
                    metric_value=event['metric_value'],
                    timestamp=event['timestamp'],
                    metadata=event.get('metadata', {})
                )
                self.db_session.add(metric)
                self.db_session.commit()
        except Exception as e:
            logger.error(f"Error saving performance metrics: {e}")

    async def _update_real_time_metrics(self, metric_type: str, event: Dict[str, Any]):
        """Update real-time metrics cache"""
        try:
            self.real_time_metrics[metric_type].append({
                'event': event,
                'timestamp': datetime.now()
            })
            
            # Keep only last 1000 entries
            if len(self.real_time_metrics[metric_type]) > 1000:
                self.real_time_metrics[metric_type] = self.real_time_metrics[metric_type][-1000:]
                
        except Exception as e:
            logger.error(f"Error updating real-time metrics: {e}")

    async def _update_redis_metrics(self, metric_type: str, event: Dict[str, Any]):
        """Update Redis metrics cache"""
        try:
            if self.redis_client:
                key = f"analytics:{metric_type}:{datetime.now().strftime('%Y%m%d%H')}"
                self.redis_client.lpush(key, json.dumps(event, default=str))
                self.redis_client.expire(key, 86400)  # 24 hours
        except Exception as e:
            logger.error(f"Error updating Redis metrics: {e}")

    def _parse_time_period(self, time_period: str) -> timedelta:
        """Parse time period string to timedelta"""
        period_map = {
            '1d': timedelta(days=1),
            '7d': timedelta(days=7),
            '30d': timedelta(days=30),
            '90d': timedelta(days=90),
            '1y': timedelta(days=365)
        }
        return period_map.get(time_period, timedelta(days=7))

    async def _get_user_metrics(self, user_id: str, start_time: datetime) -> UserMetrics:
        """Get user metrics"""
        try:
            if self.db_session:
                # Get total content created
                total_content = self.db_session.query(ContentAnalyticsDB).filter(
                    ContentAnalyticsDB.user_id == user_id,
                    ContentAnalyticsDB.action == 'created',
                    ContentAnalyticsDB.timestamp >= start_time
                ).count()
                
                # Get total exports
                total_exports = self.db_session.query(ContentAnalyticsDB).filter(
                    ContentAnalyticsDB.user_id == user_id,
                    ContentAnalyticsDB.action == 'exported',
                    ContentAnalyticsDB.timestamp >= start_time
                ).count()
                
                # Get active sessions
                active_sessions = self.db_session.query(UserAnalyticsDB).filter(
                    UserAnalyticsDB.user_id == user_id,
                    UserAnalyticsDB.action == 'session_start',
                    UserAnalyticsDB.timestamp >= start_time
                ).count()
                
                # Get collaboration time
                collaboration_time = self.db_session.query(UserAnalyticsDB).filter(
                    UserAnalyticsDB.user_id == user_id,
                    UserAnalyticsDB.timestamp >= start_time
                ).with_entities(UserAnalyticsDB.duration).all()
                
                total_collaboration_time = sum(row[0] for row in collaboration_time if row[0])
                
                # Get content types
                content_types = self.db_session.query(ContentAnalyticsDB).filter(
                    ContentAnalyticsDB.user_id == user_id,
                    ContentAnalyticsDB.action == 'created',
                    ContentAnalyticsDB.timestamp >= start_time
                ).with_entities(ContentAnalyticsDB.content_type).all()
                
                content_type_counts = Counter(row[0] for row in content_types)
                
                # Get export formats
                export_formats = self.db_session.query(ContentAnalyticsDB).filter(
                    ContentAnalyticsDB.user_id == user_id,
                    ContentAnalyticsDB.action == 'exported',
                    ContentAnalyticsDB.timestamp >= start_time
                ).with_entities(ContentAnalyticsDB.metadata).all()
                
                export_format_counts = Counter()
                for row in export_formats:
                    if row[0] and 'export_format' in row[0]:
                        export_format_counts[row[0]['export_format']] += 1
                
                return UserMetrics(
                    user_id=user_id,
                    total_content=total_content,
                    total_exports=total_exports,
                    active_sessions=active_sessions,
                    collaboration_time=total_collaboration_time,
                    last_activity=datetime.now(),
                    content_types=dict(content_type_counts),
                    export_formats=dict(export_format_counts)
                )
            
            return UserMetrics(user_id, 0, 0, 0, 0.0, datetime.now(), {}, {})
            
        except Exception as e:
            logger.error(f"Error getting user metrics: {e}")
            return UserMetrics(user_id, 0, 0, 0, 0.0, datetime.now(), {}, {})

    async def _get_content_metrics(self, user_id: str, start_time: datetime) -> List[Dict[str, Any]]:
        """Get content metrics for user"""
        try:
            if self.db_session:
                # Get content analytics
                analytics = self.db_session.query(ContentAnalyticsDB).filter(
                    ContentAnalyticsDB.user_id == user_id,
                    ContentAnalyticsDB.timestamp >= start_time
                ).all()
                
                # Group by content ID
                content_metrics = defaultdict(lambda: {
                    'views': 0, 'exports': 0, 'shares': 0,
                    'creation_time': None, 'last_activity': None
                })
                
                for analytic in analytics:
                    content_id = analytic.content_id
                    action = analytic.action
                    
                    if action == 'created':
                        content_metrics[content_id]['creation_time'] = analytic.timestamp
                    elif action == 'viewed':
                        content_metrics[content_id]['views'] += 1
                    elif action == 'exported':
                        content_metrics[content_id]['exports'] += 1
                    elif action == 'shared':
                        content_metrics[content_id]['shares'] += 1
                    
                    content_metrics[content_id]['last_activity'] = analytic.timestamp
                
                # Convert to list
                return [
                    {
                        'content_id': content_id,
                        **metrics
                    }
                    for content_id, metrics in content_metrics.items()
                ]
            
            return []
            
        except Exception as e:
            logger.error(f"Error getting content metrics: {e}")
            return []

    async def _get_recent_activity(self, user_id: str, start_time: datetime) -> List[Dict[str, Any]]:
        """Get recent activity for user"""
        try:
            if self.db_session:
                activities = self.db_session.query(ContentAnalyticsDB).filter(
                    ContentAnalyticsDB.user_id == user_id,
                    ContentAnalyticsDB.timestamp >= start_time
                ).order_by(ContentAnalyticsDB.timestamp.desc()).limit(20).all()
                
                return [
                    {
                        'content_id': activity.content_id,
                        'action': activity.action,
                        'timestamp': activity.timestamp.isoformat(),
                        'metadata': activity.metadata
                    }
                    for activity in activities
                ]
            
            return []
            
        except Exception as e:
            logger.error(f"Error getting recent activity: {e}")
            return []

    async def _get_performance_metrics(self, start_time: datetime) -> Dict[str, Any]:
        """Get performance metrics"""
        try:
            if self.db_session:
                metrics = self.db_session.query(PerformanceMetricsDB).filter(
                    PerformanceMetricsDB.timestamp >= start_time
                ).all()
                
                # Group by metric name
                metric_groups = defaultdict(list)
                for metric in metrics:
                    metric_groups[metric.metric_name].append(metric.metric_value)
                
                # Calculate statistics
                performance_data = {}
                for metric_name, values in metric_groups.items():
                    if values:
                        performance_data[metric_name] = {
                            'average': statistics.mean(values),
                            'min': min(values),
                            'max': max(values),
                            'count': len(values)
                        }
                
                return performance_data
            
            return {}
            
        except Exception as e:
            logger.error(f"Error getting performance metrics: {e}")
            return {}

    async def _get_content_metrics_by_id(self, content_id: str, start_time: datetime) -> Dict[str, Any]:
        """Get metrics for specific content"""
        try:
            if self.db_session:
                analytics = self.db_session.query(ContentAnalyticsDB).filter(
                    ContentAnalyticsDB.content_id == content_id,
                    ContentAnalyticsDB.timestamp >= start_time
                ).all()
                
                metrics = {
                    'content_id': content_id,
                    'views': 0,
                    'exports': 0,
                    'shares': 0,
                    'creation_time': None,
                    'last_activity': None,
                    'quality_score': 0.0,
                    'processing_time': 0.0
                }
                
                for analytic in analytics:
                    action = analytic.action
                    
                    if action == 'created':
                        metrics['creation_time'] = analytic.timestamp
                        if analytic.metadata:
                            metrics['quality_score'] = analytic.metadata.get('quality_score', 0.0)
                            metrics['processing_time'] = analytic.metadata.get('processing_time', 0.0)
                    elif action == 'viewed':
                        metrics['views'] += 1
                    elif action == 'exported':
                        metrics['exports'] += 1
                    elif action == 'shared':
                        metrics['shares'] += 1
                    
                    metrics['last_activity'] = analytic.timestamp
                
                # Calculate engagement score
                metrics['engagement_score'] = self._calculate_engagement_score(metrics)
                
                return metrics
            
            return {}
            
        except Exception as e:
            logger.error(f"Error getting content metrics by ID: {e}")
            return {}

    async def _get_user_content_performance(self, user_id: str, start_time: datetime) -> List[Dict[str, Any]]:
        """Get performance for all user content"""
        try:
            if self.db_session:
                # Get all content created by user
                content_ids = self.db_session.query(ContentAnalyticsDB).filter(
                    ContentAnalyticsDB.user_id == user_id,
                    ContentAnalyticsDB.action == 'created',
                    ContentAnalyticsDB.timestamp >= start_time
                ).with_entities(ContentAnalyticsDB.content_id).distinct().all()
                
                # Get metrics for each content
                content_performance = []
                for (content_id,) in content_ids:
                    metrics = await self._get_content_metrics_by_id(content_id, start_time)
                    if metrics:
                        content_performance.append(metrics)
                
                return content_performance
            
            return []
            
        except Exception as e:
            logger.error(f"Error getting user content performance: {e}")
            return []

    async def _get_collaboration_metrics(self, user_id: str, start_time: datetime) -> Dict[str, Any]:
        """Get collaboration metrics"""
        try:
            if self.db_session:
                # Get collaboration sessions
                sessions = self.db_session.query(UserAnalyticsDB).filter(
                    UserAnalyticsDB.user_id == user_id,
                    UserAnalyticsDB.timestamp >= start_time
                ).all()
                
                total_sessions = len(set(session.session_id for session in sessions))
                active_sessions = len([s for s in sessions if s.action == 'session_active'])
                total_participants = len(set(session.session_id for session in sessions))
                
                # Calculate average session duration
                session_durations = [s.duration for s in sessions if s.duration > 0]
                avg_duration = statistics.mean(session_durations) if session_durations else 0.0
                
                # Get most active users (would need additional data)
                most_active_users = []
                
                # Get session types
                session_types = Counter(s.action for s in sessions)
                
                return {
                    'total_sessions': total_sessions,
                    'active_sessions': active_sessions,
                    'total_participants': total_participants,
                    'average_session_duration': avg_duration,
                    'most_active_users': most_active_users,
                    'session_types': dict(session_types)
                }
            
            return {}
            
        except Exception as e:
            logger.error(f"Error getting collaboration metrics: {e}")
            return {}

    def _calculate_engagement_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate engagement score for content"""
        try:
            views = metrics.get('views', 0)
            exports = metrics.get('exports', 0)
            shares = metrics.get('shares', 0)
            
            # Simple engagement calculation
            # Views = 1 point, Exports = 5 points, Shares = 10 points
            engagement_score = views + (exports * 5) + (shares * 10)
            
            # Normalize to 0-100 scale
            return min(100.0, engagement_score)
            
        except Exception as e:
            logger.error(f"Error calculating engagement score: {e}")
            return 0.0

    def _calculate_trends(self, historical_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate trends from historical data"""
        try:
            if not historical_data:
                return {'trend': 'no_data', 'change_percentage': 0.0}
            
            # Extract values and timestamps
            values = [point['value'] for point in historical_data]
            timestamps = [point['timestamp'] for point in historical_data]
            
            if len(values) < 2:
                return {'trend': 'insufficient_data', 'change_percentage': 0.0}
            
            # Calculate trend direction
            first_half = values[:len(values)//2]
            second_half = values[len(values)//2:]
            
            first_avg = statistics.mean(first_half)
            second_avg = statistics.mean(second_half)
            
            if second_avg > first_avg * 1.1:
                trend = 'increasing'
            elif second_avg < first_avg * 0.9:
                trend = 'decreasing'
            else:
                trend = 'stable'
            
            # Calculate change percentage
            change_percentage = ((second_avg - first_avg) / first_avg * 100) if first_avg > 0 else 0.0
            
            return {
                'trend': trend,
                'change_percentage': change_percentage,
                'first_period_average': first_avg,
                'second_period_average': second_avg
            }
            
        except Exception as e:
            logger.error(f"Error calculating trends: {e}")
            return {'trend': 'error', 'change_percentage': 0.0}

    async def _get_total_users(self) -> int:
        """Get total number of users"""
        try:
            if self.db_session:
                return self.db_session.query(ContentAnalyticsDB).with_entities(
                    ContentAnalyticsDB.user_id
                ).distinct().count()
            return 0
        except Exception as e:
            logger.error(f"Error getting total users: {e}")
            return 0

    async def _get_active_users(self) -> int:
        """Get number of active users (last 24 hours)"""
        try:
            if self.db_session:
                cutoff_time = datetime.now() - timedelta(hours=24)
                return self.db_session.query(ContentAnalyticsDB).filter(
                    ContentAnalyticsDB.timestamp >= cutoff_time
                ).with_entities(ContentAnalyticsDB.user_id).distinct().count()
            return 0
        except Exception as e:
            logger.error(f"Error getting active users: {e}")
            return 0

    async def _get_total_content(self) -> int:
        """Get total number of content created"""
        try:
            if self.db_session:
                return self.db_session.query(ContentAnalyticsDB).filter(
                    ContentAnalyticsDB.action == 'created'
                ).count()
            return 0
        except Exception as e:
            logger.error(f"Error getting total content: {e}")
            return 0

    async def _get_total_exports(self) -> int:
        """Get total number of exports"""
        try:
            if self.db_session:
                return self.db_session.query(ContentAnalyticsDB).filter(
                    ContentAnalyticsDB.action == 'exported'
                ).count()
            return 0
        except Exception as e:
            logger.error(f"Error getting total exports: {e}")
            return 0

    async def _get_average_response_time(self) -> float:
        """Get average response time"""
        try:
            if self.db_session:
                response_times = self.db_session.query(PerformanceMetricsDB).filter(
                    PerformanceMetricsDB.metric_name == 'response_time'
                ).with_entities(PerformanceMetricsDB.metric_value).all()
                
                if response_times:
                    return statistics.mean(row[0] for row in response_times)
            return 0.0
        except Exception as e:
            logger.error(f"Error getting average response time: {e}")
            return 0.0

    async def _get_error_rate(self) -> float:
        """Get system error rate"""
        try:
            if self.db_session:
                total_requests = self.db_session.query(PerformanceMetricsDB).filter(
                    PerformanceMetricsDB.metric_name == 'total_requests'
                ).with_entities(PerformanceMetricsDB.metric_value).first()
                
                error_requests = self.db_session.query(PerformanceMetricsDB).filter(
                    PerformanceMetricsDB.metric_name == 'error_requests'
                ).with_entities(PerformanceMetricsDB.metric_value).first()
                
                if total_requests and error_requests and total_requests[0] > 0:
                    return (error_requests[0] / total_requests[0]) * 100
            return 0.0
        except Exception as e:
            logger.error(f"Error getting error rate: {e}")
            return 0.0

    async def _get_system_uptime(self) -> float:
        """Get system uptime percentage"""
        try:
            # This would be calculated based on system monitoring
            # For now, return a placeholder
            return 99.9
        except Exception as e:
            logger.error(f"Error getting system uptime: {e}")
            return 0.0

    async def _get_historical_metrics(self, metric_name: str, start_time: datetime) -> List[Dict[str, Any]]:
        """Get historical metrics data"""
        try:
            if self.db_session:
                metrics = self.db_session.query(PerformanceMetricsDB).filter(
                    PerformanceMetricsDB.metric_name == metric_name,
                    PerformanceMetricsDB.timestamp >= start_time
                ).order_by(PerformanceMetricsDB.timestamp).all()
                
                return [
                    {
                        'value': metric.metric_value,
                        'timestamp': metric.timestamp.isoformat()
                    }
                    for metric in metrics
                ]
            return []
        except Exception as e:
            logger.error(f"Error getting historical metrics: {e}")
            return []

    def get_real_time_metrics(self) -> Dict[str, Any]:
        """Get real-time metrics from cache"""
        try:
            return {
                metric_type: len(events)
                for metric_type, events in self.real_time_metrics.items()
            }
        except Exception as e:
            logger.error(f"Error getting real-time metrics: {e}")
            return {}

    async def cleanup_old_analytics(self, days_to_keep: int = 90):
        """Clean up old analytics data"""
        try:
            if self.db_session:
                cutoff_time = datetime.now() - timedelta(days=days_to_keep)
                
                # Clean up content analytics
                old_content_analytics = self.db_session.query(ContentAnalyticsDB).filter(
                    ContentAnalyticsDB.timestamp < cutoff_time
                ).delete()
                
                # Clean up user analytics
                old_user_analytics = self.db_session.query(UserAnalyticsDB).filter(
                    UserAnalyticsDB.timestamp < cutoff_time
                ).delete()
                
                # Clean up performance metrics
                old_performance_metrics = self.db_session.query(PerformanceMetricsDB).filter(
                    PerformanceMetricsDB.timestamp < cutoff_time
                ).delete()
                
                self.db_session.commit()
                
                logger.info(f"Cleaned up {old_content_analytics} content analytics, "
                          f"{old_user_analytics} user analytics, "
                          f"{old_performance_metrics} performance metrics")
                
        except Exception as e:
            logger.error(f"Error cleaning up old analytics: {e}")



























