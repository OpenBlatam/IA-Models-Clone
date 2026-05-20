"""
Analytics Service
================

Advanced analytics service for the Bulk TruthGPT system.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import json
import numpy as np
from dataclasses import dataclass
from collections import defaultdict, Counter

logger = logging.getLogger(__name__)

@dataclass
class AnalyticsEvent:
    """Analytics event data structure."""
    event_type: str
    event_data: Dict[str, Any]
    timestamp: datetime
    user_id: Optional[str] = None
    session_id: Optional[str] = None

class AnalyticsService:
    """
    Advanced analytics service.
    
    Features:
    - Event tracking
    - Performance metrics
    - User behavior analysis
    - Content quality analytics
    - System performance tracking
    - Real-time dashboards
    """
    
    def __init__(self):
        self.events = []
        self.metrics = defaultdict(list)
        self.user_sessions = {}
        self.performance_data = {}
        
    async def initialize(self):
        """Initialize analytics service."""
        logger.info("Initializing Analytics Service...")
        
        try:
            # Start background tasks
            asyncio.create_task(self._process_events())
            asyncio.create_task(self._aggregate_metrics())
            asyncio.create_task(self._cleanup_old_data())
            
            logger.info("Analytics Service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Analytics Service: {str(e)}")
            raise
    
    async def track_event(
        self,
        event_type: str,
        event_data: Dict[str, Any],
        user_id: Optional[str] = None,
        session_id: Optional[str] = None
    ):
        """Track analytics event."""
        try:
            event = AnalyticsEvent(
                event_type=event_type,
                event_data=event_data,
                timestamp=datetime.utcnow(),
                user_id=user_id,
                session_id=session_id
            )
            
            self.events.append(event)
            
            # Update session data
            if session_id:
                if session_id not in self.user_sessions:
                    self.user_sessions[session_id] = {
                        'user_id': user_id,
                        'start_time': event.timestamp,
                        'last_activity': event.timestamp,
                        'events': []
                    }
                
                self.user_sessions[session_id]['last_activity'] = event.timestamp
                self.user_sessions[session_id]['events'].append(event)
            
            logger.debug(f"Tracked event: {event_type}")
            
        except Exception as e:
            logger.error(f"Failed to track event: {str(e)}")
    
    async def track_generation_event(
        self,
        task_id: str,
        event_type: str,
        data: Dict[str, Any]
    ):
        """Track document generation events."""
        try:
            event_data = {
                'task_id': task_id,
                'event_type': event_type,
                **data
            }
            
            await self.track_event(
                event_type=f"generation_{event_type}",
                event_data=event_data
            )
            
        except Exception as e:
            logger.error(f"Failed to track generation event: {str(e)}")
    
    async def track_quality_metrics(
        self,
        task_id: str,
        document_id: str,
        quality_score: float,
        metrics: Dict[str, Any]
    ):
        """Track content quality metrics."""
        try:
            await self.track_event(
                event_type="quality_metrics",
                event_data={
                    'task_id': task_id,
                    'document_id': document_id,
                    'quality_score': quality_score,
                    'metrics': metrics,
                    'timestamp': datetime.utcnow().isoformat()
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to track quality metrics: {str(e)}")
    
    async def track_performance_metrics(
        self,
        metric_name: str,
        value: float,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Track performance metrics."""
        try:
            self.metrics[metric_name].append({
                'value': value,
                'timestamp': datetime.utcnow(),
                'metadata': metadata or {}
            })
            
        except Exception as e:
            logger.error(f"Failed to track performance metrics: {str(e)}")
    
    async def _process_events(self):
        """Process analytics events."""
        while True:
            try:
                await asyncio.sleep(10)  # Process every 10 seconds
                
                # Process recent events
                recent_events = [
                    e for e in self.events
                    if e.timestamp > datetime.utcnow() - timedelta(minutes=1)
                ]
                
                # Group events by type
                event_groups = defaultdict(list)
                for event in recent_events:
                    event_groups[event.event_type].append(event)
                
                # Process each event type
                for event_type, events in event_groups.items():
                    await self._process_event_group(event_type, events)
                
            except Exception as e:
                logger.error(f"Error processing events: {str(e)}")
    
    async def _process_event_group(self, event_type: str, events: List[AnalyticsEvent]):
        """Process a group of events."""
        try:
            if event_type.startswith('generation_'):
                await self._process_generation_events(events)
            elif event_type == 'quality_metrics':
                await self._process_quality_events(events)
            elif event_type == 'user_interaction':
                await self._process_user_events(events)
            
        except Exception as e:
            logger.error(f"Failed to process event group {event_type}: {str(e)}")
    
    async def _process_generation_events(self, events: List[AnalyticsEvent]):
        """Process generation events."""
        try:
            # Calculate generation statistics
            total_events = len(events)
            successful_events = len([e for e in events if e.event_data.get('success', False)])
            
            if total_events > 0:
                success_rate = successful_events / total_events
                await self.track_performance_metrics('generation_success_rate', success_rate)
            
        except Exception as e:
            logger.error(f"Failed to process generation events: {str(e)}")
    
    async def _process_quality_events(self, events: List[AnalyticsEvent]):
        """Process quality events."""
        try:
            if not events:
                return
            
            # Calculate quality statistics
            quality_scores = [e.event_data.get('quality_score', 0) for e in events]
            
            if quality_scores:
                avg_quality = np.mean(quality_scores)
                min_quality = np.min(quality_scores)
                max_quality = np.max(quality_scores)
                
                await self.track_performance_metrics('average_quality_score', avg_quality)
                await self.track_performance_metrics('min_quality_score', min_quality)
                await self.track_performance_metrics('max_quality_score', max_quality)
            
        except Exception as e:
            logger.error(f"Failed to process quality events: {str(e)}")
    
    async def _process_user_events(self, events: List[AnalyticsEvent]):
        """Process user events."""
        try:
            # Analyze user behavior
            user_actions = [e.event_data.get('action', '') for e in events]
            action_counts = Counter(user_actions)
            
            # Track most common actions
            for action, count in action_counts.most_common(5):
                await self.track_performance_metrics(f'user_action_{action}', count)
            
        except Exception as e:
            logger.error(f"Failed to process user events: {str(e)}")
    
    async def _aggregate_metrics(self):
        """Aggregate metrics for reporting."""
        while True:
            try:
                await asyncio.sleep(300)  # Aggregate every 5 minutes
                
                # Aggregate performance metrics
                for metric_name, values in self.metrics.items():
                    if values:
                        recent_values = [
                            v for v in values
                            if v['timestamp'] > datetime.utcnow() - timedelta(hours=1)
                        ]
                        
                        if recent_values:
                            metric_values = [v['value'] for v in recent_values]
                            
                            aggregated = {
                                'metric_name': metric_name,
                                'count': len(metric_values),
                                'average': np.mean(metric_values),
                                'min': np.min(metric_values),
                                'max': np.max(metric_values),
                                'std': np.std(metric_values),
                                'timestamp': datetime.utcnow()
                            }
                            
                            # Store aggregated data
                            if metric_name not in self.performance_data:
                                self.performance_data[metric_name] = []
                            
                            self.performance_data[metric_name].append(aggregated)
                            
                            # Keep only recent aggregated data
                            cutoff_time = datetime.utcnow() - timedelta(days=7)
                            self.performance_data[metric_name] = [
                                d for d in self.performance_data[metric_name]
                                if d['timestamp'] > cutoff_time
                            ]
                
            except Exception as e:
                logger.error(f"Error aggregating metrics: {str(e)}")
    
    async def _cleanup_old_data(self):
        """Cleanup old analytics data."""
        while True:
            try:
                await asyncio.sleep(3600)  # Cleanup every hour
                
                cutoff_time = datetime.utcnow() - timedelta(days=30)
                
                # Cleanup events
                self.events = [e for e in self.events if e.timestamp > cutoff_time]
                
                # Cleanup metrics
                for metric_name in self.metrics:
                    self.metrics[metric_name] = [
                        v for v in self.metrics[metric_name]
                        if v['timestamp'] > cutoff_time
                    ]
                
                # Cleanup sessions
                self.user_sessions = {
                    sid: session for sid, session in self.user_sessions.items()
                    if session['last_activity'] > cutoff_time
                }
                
            except Exception as e:
                logger.error(f"Error cleaning up old data: {str(e)}")
    
    async def get_generation_analytics(
        self, 
        time_range: str = "24h"
    ) -> Dict[str, Any]:
        """Get generation analytics."""
        try:
            # Calculate time range
            if time_range == "1h":
                cutoff_time = datetime.utcnow() - timedelta(hours=1)
            elif time_range == "24h":
                cutoff_time = datetime.utcnow() - timedelta(days=1)
            elif time_range == "7d":
                cutoff_time = datetime.utcnow() - timedelta(days=7)
            elif time_range == "30d":
                cutoff_time = datetime.utcnow() - timedelta(days=30)
            else:
                cutoff_time = datetime.utcnow() - timedelta(days=1)
            
            # Filter events
            recent_events = [
                e for e in self.events
                if e.timestamp > cutoff_time and e.event_type.startswith('generation_')
            ]
            
            # Calculate statistics
            total_events = len(recent_events)
            successful_events = len([
                e for e in recent_events
                if e.event_data.get('success', False)
            ])
            
            # Group by task
            tasks = defaultdict(list)
            for event in recent_events:
                task_id = event.event_data.get('task_id')
                if task_id:
                    tasks[task_id].append(event)
            
            # Calculate per-task statistics
            task_stats = []
            for task_id, task_events in tasks.items():
                task_successful = len([
                    e for e in task_events
                    if e.event_data.get('success', False)
                ])
                
                task_stats.append({
                    'task_id': task_id,
                    'total_events': len(task_events),
                    'successful_events': task_successful,
                    'success_rate': task_successful / len(task_events) if task_events else 0
                })
            
            return {
                'time_range': time_range,
                'total_events': total_events,
                'successful_events': successful_events,
                'success_rate': successful_events / total_events if total_events > 0 else 0,
                'unique_tasks': len(tasks),
                'task_statistics': task_stats,
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get generation analytics: {str(e)}")
            return {}
    
    async def get_quality_analytics(
        self, 
        time_range: str = "24h"
    ) -> Dict[str, Any]:
        """Get quality analytics."""
        try:
            # Calculate time range
            if time_range == "1h":
                cutoff_time = datetime.utcnow() - timedelta(hours=1)
            elif time_range == "24h":
                cutoff_time = datetime.utcnow() - timedelta(days=1)
            elif time_range == "7d":
                cutoff_time = datetime.utcnow() - timedelta(days=7)
            elif time_range == "30d":
                cutoff_time = datetime.utcnow() - timedelta(days=30)
            else:
                cutoff_time = datetime.utcnow() - timedelta(days=1)
            
            # Filter quality events
            quality_events = [
                e for e in self.events
                if e.timestamp > cutoff_time and e.event_type == 'quality_metrics'
            ]
            
            if not quality_events:
                return {
                    'time_range': time_range,
                    'total_documents': 0,
                    'average_quality': 0.0,
                    'quality_distribution': {},
                    'timestamp': datetime.utcnow().isoformat()
                }
            
            # Extract quality scores
            quality_scores = [
                e.event_data.get('quality_score', 0)
                for e in quality_events
            ]
            
            # Calculate statistics
            avg_quality = np.mean(quality_scores)
            min_quality = np.min(quality_scores)
            max_quality = np.max(quality_scores)
            std_quality = np.std(quality_scores)
            
            # Quality distribution
            quality_ranges = {
                'excellent (0.9-1.0)': len([s for s in quality_scores if 0.9 <= s <= 1.0]),
                'good (0.7-0.9)': len([s for s in quality_scores if 0.7 <= s < 0.9]),
                'fair (0.5-0.7)': len([s for s in quality_scores if 0.5 <= s < 0.7]),
                'poor (0.0-0.5)': len([s for s in quality_scores if 0.0 <= s < 0.5])
            }
            
            return {
                'time_range': time_range,
                'total_documents': len(quality_events),
                'average_quality': avg_quality,
                'min_quality': min_quality,
                'max_quality': max_quality,
                'std_quality': std_quality,
                'quality_distribution': quality_ranges,
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get quality analytics: {str(e)}")
            return {}
    
    async def get_performance_analytics(
        self, 
        metric_name: Optional[str] = None,
        time_range: str = "24h"
    ) -> Dict[str, Any]:
        """Get performance analytics."""
        try:
            # Calculate time range
            if time_range == "1h":
                cutoff_time = datetime.utcnow() - timedelta(hours=1)
            elif time_range == "24h":
                cutoff_time = datetime.utcnow() - timedelta(days=1)
            elif time_range == "7d":
                cutoff_time = datetime.utcnow() - timedelta(days=7)
            elif time_range == "30d":
                cutoff_time = datetime.utcnow() - timedelta(days=30)
            else:
                cutoff_time = datetime.utcnow() - timedelta(days=1)
            
            if metric_name:
                # Get specific metric
                if metric_name in self.performance_data:
                    metric_data = [
                        d for d in self.performance_data[metric_name]
                        if d['timestamp'] > cutoff_time
                    ]
                    
                    if metric_data:
                        values = [d['average'] for d in metric_data]
                        return {
                            'metric_name': metric_name,
                            'time_range': time_range,
                            'data_points': len(metric_data),
                            'average': np.mean(values),
                            'min': np.min(values),
                            'max': np.max(values),
                            'trend': self._calculate_trend(values),
                            'timestamp': datetime.utcnow().isoformat()
                        }
                else:
                    return {'error': f'Metric {metric_name} not found'}
            else:
                # Get all metrics summary
                all_metrics = {}
                for metric_name, metric_data in self.performance_data.items():
                    recent_data = [
                        d for d in metric_data
                        if d['timestamp'] > cutoff_time
                    ]
                    
                    if recent_data:
                        values = [d['average'] for d in recent_data]
                        all_metrics[metric_name] = {
                            'data_points': len(recent_data),
                            'average': np.mean(values),
                            'min': np.min(values),
                            'max': np.max(values)
                        }
                
                return {
                    'time_range': time_range,
                    'metrics': all_metrics,
                    'timestamp': datetime.utcnow().isoformat()
                }
            
        except Exception as e:
            logger.error(f"Failed to get performance analytics: {str(e)}")
            return {}
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction."""
        try:
            if len(values) < 2:
                return "insufficient_data"
            
            # Simple linear trend
            x = np.arange(len(values))
            y = np.array(values)
            
            # Calculate slope
            slope = np.polyfit(x, y, 1)[0]
            
            if slope > 0.1:
                return "increasing"
            elif slope < -0.1:
                return "decreasing"
            else:
                return "stable"
                
        except Exception as e:
            logger.error(f"Failed to calculate trend: {str(e)}")
            return "unknown"
    
    async def get_user_analytics(
        self, 
        time_range: str = "24h"
    ) -> Dict[str, Any]:
        """Get user analytics."""
        try:
            # Calculate time range
            if time_range == "1h":
                cutoff_time = datetime.utcnow() - timedelta(hours=1)
            elif time_range == "24h":
                cutoff_time = datetime.utcnow() - timedelta(days=1)
            elif time_range == "7d":
                cutoff_time = datetime.utcnow() - timedelta(days=7)
            elif time_range == "30d":
                cutoff_time = datetime.utcnow() - timedelta(days=30)
            else:
                cutoff_time = datetime.utcnow() - timedelta(days=1)
            
            # Filter recent sessions
            recent_sessions = {
                sid: session for sid, session in self.user_sessions.items()
                if session['last_activity'] > cutoff_time
            }
            
            # Calculate session statistics
            total_sessions = len(recent_sessions)
            unique_users = len(set(session['user_id'] for session in recent_sessions.values() if session['user_id']))
            
            # Session duration analysis
            session_durations = []
            for session in recent_sessions.values():
                duration = (session['last_activity'] - session['start_time']).total_seconds()
                session_durations.append(duration)
            
            avg_session_duration = np.mean(session_durations) if session_durations else 0
            
            # Event analysis
            all_events = []
            for session in recent_sessions.values():
                all_events.extend(session['events'])
            
            event_types = Counter(e.event_type for e in all_events)
            
            return {
                'time_range': time_range,
                'total_sessions': total_sessions,
                'unique_users': unique_users,
                'average_session_duration': avg_session_duration,
                'total_events': len(all_events),
                'event_types': dict(event_types.most_common(10)),
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get user analytics: {str(e)}")
            return {}
    
    async def cleanup(self):
        """Cleanup analytics service."""
        try:
            logger.info("Analytics Service cleaned up successfully")
            
        except Exception as e:
            logger.error(f"Failed to cleanup Analytics Service: {str(e)}")











