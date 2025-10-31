import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
import json
import time
import logging
from dataclasses import dataclass, field
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import threading
from collections import deque, defaultdict
import pickle
import hashlib
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


@dataclass
class ABTestConfig:
    """Configuration for A/B testing"""
    test_name: str
    variants: List[str]
    traffic_split: List[float] = None
    duration_days: int = 30
    confidence_level: float = 0.95
    minimum_sample_size: int = 100
    primary_metric: str = "engagement_score"
    secondary_metrics: List[str] = field(default_factory=lambda: ["viral_potential", "content_quality"])
    
    def __post_init__(self):
        if self.traffic_split is None:
            self.traffic_split = [1.0 / len(self.variants)] * len(self.variants)


@dataclass
class RealTimeConfig:
    """Configuration for real-time optimization"""
    update_interval_seconds: int = 300  # 5 minutes
    batch_size: int = 32
    learning_rate: float = 1e-5
    memory_size: int = 10000
    performance_threshold: float = 0.7
    adaptation_rate: float = 0.1


class ABTestManager:
    """A/B Testing Manager for Facebook Content Optimization"""
    
    def __init__(self, config: ABTestConfig):
        self.config = config
        self.results = defaultdict(lambda: {
            'samples': [],
            'metrics': defaultdict(list),
            'start_time': None,
            'end_time': None,
            'status': 'running'
        })
        self.current_variant = None
        self.test_id = self._generate_test_id()
        
        # Initialize logging
        self.logger = logging.getLogger(f"ABTest_{self.test_id}")
        self.logger.setLevel(logging.INFO)
        
        # Create results directory
        self.results_dir = Path(f"ab_test_results/{self.test_id}")
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def _generate_test_id(self) -> str:
        """Generate unique test ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{self.config.test_name}_{timestamp}"
    
    def assign_variant(self, user_id: str) -> str:
        """Assign user to a variant based on traffic split"""
        if self.current_variant is None:
            # Use hash-based assignment for consistency
            hash_value = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
            cumulative_prob = 0
            for i, prob in enumerate(self.config.traffic_split):
                cumulative_prob += prob
                if hash_value / (2**128) <= cumulative_prob:
                    self.current_variant = self.config.variants[i]
                    break
        
        return self.current_variant
    
    def record_result(self, variant: str, metrics: Dict[str, float], user_id: str = None):
        """Record test result for a variant"""
        if variant not in self.config.variants:
            self.logger.warning(f"Unknown variant: {variant}")
            return
        
        # Initialize start time if first result
        if self.results[variant]['start_time'] is None:
            self.results[variant]['start_time'] = datetime.now()
        
        # Record metrics
        self.results[variant]['samples'].append({
            'user_id': user_id,
            'timestamp': datetime.now(),
            'metrics': metrics
        })
        
        # Update aggregated metrics
        for metric_name, value in metrics.items():
            self.results[variant]['metrics'][metric_name].append(value)
        
        # Check if test should end
        self._check_test_completion()
    
    def _check_test_completion(self):
        """Check if A/B test should be completed"""
        total_samples = sum(len(result['samples']) for result in self.results.values())
        
        # Check minimum sample size
        if total_samples < self.config.minimum_sample_size:
            return
        
        # Check duration
        for variant, result in self.results.items():
            if result['start_time'] is None:
                continue
            
            duration = datetime.now() - result['start_time']
            if duration.days >= self.config.duration_days:
                self._complete_test()
                return
    
    def _complete_test(self):
        """Complete the A/B test and analyze results"""
        self.logger.info("Completing A/B test...")
        
        for variant in self.results:
            self.results[variant]['end_time'] = datetime.now()
            self.results[variant]['status'] = 'completed'
        
        # Perform statistical analysis
        analysis = self._perform_statistical_analysis()
        
        # Save results
        self._save_results(analysis)
        
        # Log winner
        if analysis['winner']:
            self.logger.info(f"Test winner: {analysis['winner']}")
            self.logger.info(f"Improvement: {analysis['improvement']:.2%}")
    
    def _perform_statistical_analysis(self) -> Dict[str, Any]:
        """Perform statistical analysis of A/B test results"""
        analysis = {
            'winner': None,
            'improvement': 0.0,
            'confidence': 0.0,
            'p_value': 1.0,
            'effect_size': 0.0
        }
        
        # Get primary metric data
        primary_metric = self.config.primary_metric
        variant_data = {}
        
        for variant in self.config.variants:
            if primary_metric in self.results[variant]['metrics']:
                variant_data[variant] = np.array(self.results[variant]['metrics'][primary_metric])
        
        if len(variant_data) < 2:
            return analysis
        
        # Perform t-test between variants
        variants = list(variant_data.keys())
        best_variant = None
        best_p_value = 1.0
        
        for i, variant1 in enumerate(variants):
            for variant2 in variants[i+1:]:
                # Perform t-test
                t_stat, p_value = stats.ttest_ind(variant_data[variant1], variant_data[variant2])
                
                # Calculate effect size (Cohen's d)
                pooled_std = np.sqrt(((len(variant_data[variant1]) - 1) * np.var(variant_data[variant1], ddof=1) +
                                    (len(variant_data[variant2]) - 1) * np.var(variant_data[variant2], ddof=1)) /
                                   (len(variant_data[variant1]) + len(variant_data[variant2]) - 2))
                
                effect_size = (np.mean(variant_data[variant1]) - np.mean(variant_data[variant2])) / pooled_std
                
                # Determine winner
                if p_value < best_p_value and p_value < (1 - self.config.confidence_level):
                    if np.mean(variant_data[variant1]) > np.mean(variant_data[variant2]):
                        best_variant = variant1
                        best_p_value = p_value
                        analysis['effect_size'] = abs(effect_size)
                    else:
                        best_variant = variant2
                        best_p_value = p_value
                        analysis['effect_size'] = abs(effect_size)
        
        if best_variant:
            analysis['winner'] = best_variant
            analysis['p_value'] = best_p_value
            analysis['confidence'] = 1 - best_p_value
            
            # Calculate improvement
            baseline_variant = [v for v in variants if v != best_variant][0]
            improvement = ((np.mean(variant_data[best_variant]) - np.mean(variant_data[baseline_variant])) /
                          np.mean(variant_data[baseline_variant]))
            analysis['improvement'] = improvement
        
        return analysis
    
    def _save_results(self, analysis: Dict[str, Any]):
        """Save A/B test results"""
        results_data = {
            'test_id': self.test_id,
            'config': vars(self.config),
            'results': dict(self.results),
            'analysis': analysis,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save as JSON
        with open(self.results_dir / 'results.json', 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        
        # Save as pickle for programmatic access
        with open(self.results_dir / 'results.pkl', 'wb') as f:
            pickle.dump(results_data, f)
        
        # Generate visualization
        self._generate_visualization()
    
    def _generate_visualization(self):
        """Generate visualization of A/B test results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'A/B Test Results: {self.config.test_name}', fontsize=16)
        
        # 1. Primary metric comparison
        ax1 = axes[0, 0]
        primary_metric = self.config.primary_metric
        variant_data = []
        variant_names = []
        
        for variant in self.config.variants:
            if primary_metric in self.results[variant]['metrics']:
                variant_data.append(self.results[variant]['metrics'][primary_metric])
                variant_names.append(variant)
        
        if variant_data:
            ax1.boxplot(variant_data, labels=variant_names)
            ax1.set_title(f'{primary_metric} Distribution')
            ax1.set_ylabel(primary_metric)
        
        # 2. Sample size over time
        ax2 = axes[0, 1]
        for variant in self.config.variants:
            if self.results[variant]['samples']:
                timestamps = [sample['timestamp'] for sample in self.results[variant]['samples']]
                sample_counts = list(range(1, len(timestamps) + 1))
                ax2.plot(timestamps, sample_counts, label=variant, marker='o')
        
        ax2.set_title('Sample Size Over Time')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Sample Count')
        ax2.legend()
        ax2.tick_params(axis='x', rotation=45)
        
        # 3. Secondary metrics comparison
        ax3 = axes[1, 0]
        if self.config.secondary_metrics:
            metric = self.config.secondary_metrics[0]
            variant_data = []
            variant_names = []
            
            for variant in self.config.variants:
                if metric in self.results[variant]['metrics']:
                    variant_data.append(self.results[variant]['metrics'][metric])
                    variant_names.append(variant)
            
            if variant_data:
                ax3.boxplot(variant_data, labels=variant_names)
                ax3.set_title(f'{metric} Distribution')
                ax3.set_ylabel(metric)
        
        # 4. Confidence intervals
        ax4 = axes[1, 1]
        if len(self.config.variants) >= 2:
            variants = list(self.results.keys())
            means = []
            stds = []
            names = []
            
            for variant in variants:
                if primary_metric in self.results[variant]['metrics']:
                    data = self.results[variant]['metrics'][primary_metric]
                    means.append(np.mean(data))
                    stds.append(np.std(data) / np.sqrt(len(data)))
                    names.append(variant)
            
            if means:
                x_pos = np.arange(len(names))
                ax4.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7)
                ax4.set_title('Confidence Intervals')
                ax4.set_ylabel(primary_metric)
                ax4.set_xticks(x_pos)
                ax4.set_xticklabels(names)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'visualization.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def get_test_status(self) -> Dict[str, Any]:
        """Get current test status"""
        status = {
            'test_id': self.test_id,
            'status': 'running',
            'total_samples': sum(len(result['samples']) for result in self.results.values()),
            'variants': {}
        }
        
        for variant in self.config.variants:
            if variant in self.results:
                result = self.results[variant]
                status['variants'][variant] = {
                    'samples': len(result['samples']),
                    'status': result['status'],
                    'start_time': result['start_time'],
                    'end_time': result['end_time']
                }
        
        return status


class RealTimeOptimizer:
    """Real-time optimization system for Facebook content"""
    
    def __init__(self, config: RealTimeConfig, model: nn.Module):
        self.config = config
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        self.criterion = nn.MSELoss()
        
        # Performance memory
        self.performance_memory = deque(maxlen=config.memory_size)
        self.adaptation_history = []
        
        # Threading for real-time updates
        self.lock = threading.Lock()
        self.running = False
        self.update_thread = None
        
        # Performance monitoring
        self.performance_metrics = {
            'avg_engagement': 0.0,
            'avg_viral_potential': 0.0,
            'adaptation_count': 0,
            'last_update': datetime.now()
        }
        
        self.logger = logging.getLogger("RealTimeOptimizer")
        self.logger.setLevel(logging.INFO)
    
    def start(self):
        """Start real-time optimization"""
        if self.running:
            self.logger.warning("Real-time optimizer is already running")
            return
        
        self.running = True
        self.update_thread = threading.Thread(target=self._update_loop)
        self.update_thread.daemon = True
        self.update_thread.start()
        
        self.logger.info("Real-time optimizer started")
    
    def stop(self):
        """Stop real-time optimization"""
        self.running = False
        if self.update_thread:
            self.update_thread.join()
        
        self.logger.info("Real-time optimizer stopped")
    
    def _update_loop(self):
        """Main update loop for real-time optimization"""
        while self.running:
            try:
                # Check if we have enough data for adaptation
                if len(self.performance_memory) >= self.config.batch_size:
                    self._adapt_model()
                
                # Update performance metrics
                self._update_performance_metrics()
                
                # Sleep for update interval
                time.sleep(self.config.update_interval_seconds)
                
            except Exception as e:
                self.logger.error(f"Error in update loop: {e}")
                time.sleep(60)  # Wait before retrying
    
    def record_performance(self, content_id: str, metrics: Dict[str, float], features: torch.Tensor):
        """Record performance for content optimization"""
        with self.lock:
            performance_data = {
                'content_id': content_id,
                'metrics': metrics,
                'features': features.clone().detach(),
                'timestamp': datetime.now()
            }
            
            self.performance_memory.append(performance_data)
    
    def _adapt_model(self):
        """Adapt model based on recent performance"""
        with self.lock:
            # Get recent performance data
            recent_data = list(self.performance_memory)[-self.config.batch_size:]
            
            # Prepare training data
            features = torch.stack([data['features'] for data in recent_data])
            targets = torch.tensor([data['metrics']['engagement_score'] for data in recent_data])
            
            # Check if performance is below threshold
            avg_performance = targets.mean().item()
            if avg_performance < self.config.performance_threshold:
                # Perform adaptation
                self.optimizer.zero_grad()
                
                # Forward pass
                predictions = self.model(features)
                loss = self.criterion(predictions, targets.unsqueeze(1))
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                # Record adaptation
                adaptation_record = {
                    'timestamp': datetime.now(),
                    'avg_performance': avg_performance,
                    'loss': loss.item(),
                    'sample_size': len(recent_data)
                }
                
                self.adaptation_history.append(adaptation_record)
                self.performance_metrics['adaptation_count'] += 1
                
                self.logger.info(f"Model adapted - Avg performance: {avg_performance:.3f}, Loss: {loss.item():.3f}")
    
    def _update_performance_metrics(self):
        """Update performance metrics"""
        with self.lock:
            if self.performance_memory:
                recent_data = list(self.performance_memory)[-100:]  # Last 100 samples
                
                engagement_scores = [data['metrics']['engagement_score'] for data in recent_data]
                viral_potentials = [data['metrics'].get('viral_potential', 0.0) for data in recent_data]
                
                self.performance_metrics['avg_engagement'] = np.mean(engagement_scores)
                self.performance_metrics['avg_viral_potential'] = np.mean(viral_potentials)
                self.performance_metrics['last_update'] = datetime.now()
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        with self.lock:
            summary = {
                'performance_metrics': self.performance_metrics.copy(),
                'memory_size': len(self.performance_memory),
                'adaptation_count': len(self.adaptation_history),
                'running': self.running
            }
            
            # Add recent adaptations
            if self.adaptation_history:
                recent_adaptations = self.adaptation_history[-5:]  # Last 5 adaptations
                summary['recent_adaptations'] = recent_adaptations
            
            return summary


class PerformanceMonitor:
    """Performance monitoring system for Facebook content optimization"""
    
    def __init__(self, save_dir: str = "performance_monitoring"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Performance data storage
        self.performance_data = defaultdict(list)
        self.alerts = []
        
        # Monitoring thresholds
        self.thresholds = {
            'engagement_score': {'min': 0.3, 'max': 1.0},
            'viral_potential': {'min': 0.1, 'max': 1.0},
            'content_quality': {'min': 0.5, 'max': 1.0}
        }
        
        self.logger = logging.getLogger("PerformanceMonitor")
        self.logger.setLevel(logging.INFO)
    
    def record_metrics(self, content_id: str, metrics: Dict[str, float], metadata: Dict[str, Any] = None):
        """Record performance metrics"""
        timestamp = datetime.now()
        
        record = {
            'content_id': content_id,
            'timestamp': timestamp,
            'metrics': metrics,
            'metadata': metadata or {}
        }
        
        # Store in memory
        self.performance_data[content_id].append(record)
        
        # Check for alerts
        self._check_alerts(content_id, metrics, timestamp)
        
        # Save to file periodically
        if len(self.performance_data) % 100 == 0:
            self._save_performance_data()
    
    def _check_alerts(self, content_id: str, metrics: Dict[str, float], timestamp: datetime):
        """Check for performance alerts"""
        for metric_name, value in metrics.items():
            if metric_name in self.thresholds:
                threshold = self.thresholds[metric_name]
                
                if value < threshold['min'] or value > threshold['max']:
                    alert = {
                        'content_id': content_id,
                        'metric': metric_name,
                        'value': value,
                        'threshold': threshold,
                        'timestamp': timestamp,
                        'severity': 'high' if abs(value - threshold['min']) > 0.3 else 'medium'
                    }
                    
                    self.alerts.append(alert)
                    self.logger.warning(f"Performance alert: {alert}")
    
    def _save_performance_data(self):
        """Save performance data to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save performance data
        performance_file = self.save_dir / f"performance_data_{timestamp}.json"
        with open(performance_file, 'w') as f:
            json.dump(dict(self.performance_data), f, indent=2, default=str)
        
        # Save alerts
        alerts_file = self.save_dir / f"alerts_{timestamp}.json"
        with open(alerts_file, 'w') as f:
            json.dump(self.alerts, f, indent=2, default=str)
    
    def generate_performance_report(self, start_date: datetime = None, end_date: datetime = None) -> Dict[str, Any]:
        """Generate performance report"""
        if start_date is None:
            start_date = datetime.now() - timedelta(days=30)
        if end_date is None:
            end_date = datetime.now()
        
        # Filter data by date range
        filtered_data = []
        for content_id, records in self.performance_data.items():
            for record in records:
                if start_date <= record['timestamp'] <= end_date:
                    filtered_data.append(record)
        
        if not filtered_data:
            return {'error': 'No data in specified date range'}
        
        # Calculate statistics
        all_metrics = defaultdict(list)
        for record in filtered_data:
            for metric_name, value in record['metrics'].items():
                all_metrics[metric_name].append(value)
        
        report = {
            'period': {
                'start_date': start_date.isoformat(),
                'end_date': end_date.isoformat()
            },
            'summary': {
                'total_content': len(filtered_data),
                'unique_content': len(set(record['content_id'] for record in filtered_data))
            },
            'metrics': {}
        }
        
        for metric_name, values in all_metrics.items():
            report['metrics'][metric_name] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'median': np.median(values),
                'count': len(values)
            }
        
        # Add alerts summary
        period_alerts = [alert for alert in self.alerts if start_date <= alert['timestamp'] <= end_date]
        report['alerts'] = {
            'total': len(period_alerts),
            'high_severity': len([a for a in period_alerts if a['severity'] == 'high']),
            'medium_severity': len([a for a in period_alerts if a['severity'] == 'medium'])
        }
        
        return report
    
    def plot_performance_trends(self, metric_name: str = 'engagement_score', days: int = 30):
        """Plot performance trends"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Filter data
        filtered_data = []
        for records in self.performance_data.values():
            for record in records:
                if start_date <= record['timestamp'] <= end_date:
                    if metric_name in record['metrics']:
                        filtered_data.append({
                            'timestamp': record['timestamp'],
                            'value': record['metrics'][metric_name]
                        })
        
        if not filtered_data:
            print(f"No data found for {metric_name} in the last {days} days")
            return
        
        # Create DataFrame
        df = pd.DataFrame(filtered_data)
        df['date'] = pd.to_datetime(df['timestamp']).dt.date
        
        # Group by date and calculate daily statistics
        daily_stats = df.groupby('date').agg({
            'value': ['mean', 'std', 'count']
        }).reset_index()
        
        # Plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Daily mean with confidence intervals
        ax1.plot(daily_stats['date'], daily_stats[('value', 'mean')], 'b-', label='Daily Mean')
        ax1.fill_between(
            daily_stats['date'],
            daily_stats[('value', 'mean')] - daily_stats[('value', 'std')],
            daily_stats[('value', 'mean')] + daily_stats[('value', 'std')],
            alpha=0.3, label='±1 Std Dev'
        )
        ax1.set_title(f'{metric_name} Daily Trends')
        ax1.set_ylabel(metric_name)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Sample count
        ax2.bar(daily_stats['date'], daily_stats[('value', 'count')], alpha=0.7)
        ax2.set_title('Daily Sample Count')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Sample Count')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / f'{metric_name}_trends.png', dpi=300, bbox_inches='tight')
        plt.show()


# Example usage and integration
def create_optimization_system():
    """Create and configure the complete optimization system"""
    
    # A/B Test Configuration
    ab_config = ABTestConfig(
        test_name="content_optimization_v1",
        variants=["baseline", "optimized", "aggressive"],
        traffic_split=[0.33, 0.33, 0.34],
        duration_days=14,
        primary_metric="engagement_score"
    )
    
    # Real-time Configuration
    rt_config = RealTimeConfig(
        update_interval_seconds=300,
        batch_size=64,
        learning_rate=1e-5,
        performance_threshold=0.6
    )
    
    # Create components
    ab_manager = ABTestManager(ab_config)
    performance_monitor = PerformanceMonitor()
    
    # Note: Real-time optimizer requires a model instance
    # rt_optimizer = RealTimeOptimizer(rt_config, model)
    
    return {
        'ab_manager': ab_manager,
        'performance_monitor': performance_monitor,
        # 'rt_optimizer': rt_optimizer
    }


if __name__ == "__main__":
    # Example usage
    system = create_optimization_system()
    
    # Simulate A/B test
    ab_manager = system['ab_manager']
    performance_monitor = system['performance_monitor']
    
    # Simulate content performance
    for i in range(1000):
        user_id = f"user_{i}"
        variant = ab_manager.assign_variant(user_id)
        
        # Simulate metrics
        if variant == "baseline":
            engagement = np.random.normal(0.5, 0.1)
        elif variant == "optimized":
            engagement = np.random.normal(0.6, 0.1)
        else:  # aggressive
            engagement = np.random.normal(0.7, 0.15)
        
        metrics = {
            'engagement_score': max(0, min(1, engagement)),
            'viral_potential': np.random.uniform(0, 1),
            'content_quality': np.random.uniform(0.3, 1.0)
        }
        
        # Record results
        ab_manager.record_result(variant, metrics, user_id)
        performance_monitor.record_metrics(f"content_{i}", metrics, {'variant': variant})
    
    # Generate reports
    print("A/B Test Status:", ab_manager.get_test_status())
    
    report = performance_monitor.generate_performance_report()
    print("Performance Report:", json.dumps(report, indent=2))
    
    # Generate visualizations
    performance_monitor.plot_performance_trends('engagement_score', days=7)


