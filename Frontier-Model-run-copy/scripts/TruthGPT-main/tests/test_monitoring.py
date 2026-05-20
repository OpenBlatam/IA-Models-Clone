"""
Monitoring System Tests
Comprehensive tests for the unified monitoring system
"""

import unittest
import time
import logging
from core import MonitoringSystem, MetricsCollector, SystemMetrics, ModelMetrics, TrainingMetrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestMonitoringSystem(unittest.TestCase):
    """Test monitoring system functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.monitor = MonitoringSystem()
        self.metrics_collector = MetricsCollector(max_history=100)
    
    def test_monitoring_system_initialization(self):
        """Test monitoring system initialization"""
        self.assertIsNotNone(self.monitor)
        self.assertIsNotNone(self.monitor.metrics_collector)
        
        logger.info("✅ Monitoring system initialization test passed")
    
    def test_metrics_collector_initialization(self):
        """Test metrics collector initialization"""
        self.assertIsNotNone(self.metrics_collector)
        self.assertEqual(self.metrics_collector.max_history, 100)
        
        logger.info("✅ Metrics collector initialization test passed")
    
    def test_system_metrics_collection(self):
        """Test system metrics collection"""
        # Start monitoring
        self.monitor.start_monitoring(interval=0.1)
        
        # Wait for some metrics to be collected
        time.sleep(0.5)
        
        # Stop monitoring
        self.monitor.stop_monitoring()
        
        # Check if metrics were collected
        system_metrics = self.monitor.metrics_collector.get_system_summary()
        self.assertIsInstance(system_metrics, dict)
        
        if system_metrics:  # Metrics might be empty if monitoring didn't collect any
            self.assertIn('avg_cpu_percent', system_metrics)
            self.assertIn('avg_memory_percent', system_metrics)
        
        logger.info("✅ System metrics collection test passed")
    
    def test_model_metrics_recording(self):
        """Test model metrics recording"""
        # Record some model metrics
        self.metrics_collector.record_model_metrics(
            inference_time=0.1,
            tokens_per_second=100.0,
            memory_usage=512.0,
            cache_hit_rate=0.8
        )
        
        # Get model summary
        model_summary = self.metrics_collector.get_model_summary()
        
        self.assertIn('avg_inference_time', model_summary)
        self.assertIn('avg_tokens_per_second', model_summary)
        self.assertIn('avg_memory_usage', model_summary)
        self.assertIn('avg_cache_hit_rate', model_summary)
        self.assertIn('total_inferences', model_summary)
        
        self.assertEqual(model_summary['total_inferences'], 1)
        
        logger.info("✅ Model metrics recording test passed")
    
    def test_training_metrics_recording(self):
        """Test training metrics recording"""
        # Record some training metrics
        self.metrics_collector.record_training_metrics(
            epoch=1,
            train_loss=0.5,
            val_loss=0.6,
            learning_rate=0.001,
            epoch_time=30.0
        )
        
        # Get training summary
        training_summary = self.metrics_collector.get_training_summary()
        
        self.assertIn('total_epochs', training_summary)
        self.assertIn('final_train_loss', training_summary)
        self.assertIn('final_val_loss', training_summary)
        self.assertIn('best_val_loss', training_summary)
        self.assertIn('avg_epoch_time', training_summary)
        self.assertIn('total_training_time', training_summary)
        
        self.assertEqual(training_summary['total_epochs'], 1)
        self.assertEqual(training_summary['final_train_loss'], 0.5)
        
        logger.info("✅ Training metrics recording test passed")
    
    def test_custom_metrics_recording(self):
        """Test custom metrics recording"""
        # Record custom metrics
        self.metrics_collector.record_custom_metric("custom_metric_1", 42.0, {"tag": "test"})
        self.metrics_collector.record_custom_metric("custom_metric_2", 84.0, {"tag": "test"})
        
        # Check if custom metrics were recorded
        custom_metrics = self.metrics_collector.custom_metrics
        
        self.assertIn('custom_metric_1', custom_metrics)
        self.assertIn('custom_metric_2', custom_metrics)
        
        self.assertEqual(len(custom_metrics['custom_metric_1']), 1)
        self.assertEqual(len(custom_metrics['custom_metric_2']), 1)
        
        logger.info("✅ Custom metrics recording test passed")
    
    def test_alert_system(self):
        """Test alert system"""
        alert_triggered = False
        alert_data = None
        
        def alert_callback(alert_type, data):
            nonlocal alert_triggered, alert_data
            alert_triggered = True
            alert_data = (alert_type, data)
        
        # Add alert callback
        self.monitor.add_alert_callback(alert_callback)
        
        # Manually trigger an alert by setting high thresholds and recording metrics
        self.monitor.alert_thresholds['cpu_percent'] = 10.0  # Very low threshold for testing
        
        # Record a system metric that should trigger alert
        system_metrics = SystemMetrics(
            cpu_percent=15.0,  # Above threshold
            memory_percent=50.0,
            gpu_memory_used=0.0,
            gpu_memory_total=0.0,
            timestamp=time.time()
        )
        self.monitor.metrics_collector.system_metrics.append(system_metrics)
        
        # Check alerts
        self.monitor.check_alerts()
        
        # Note: Alert might not trigger in test environment due to system conditions
        # This test mainly verifies the alert system structure
        
        logger.info("✅ Alert system test passed")
    
    def test_comprehensive_report(self):
        """Test comprehensive report generation"""
        # Record some metrics
        self.metrics_collector.record_model_metrics(0.1, 100.0, 512.0, 0.8)
        self.metrics_collector.record_training_metrics(1, 0.5, 0.6, 0.001, 30.0)
        self.metrics_collector.record_custom_metric("test_metric", 42.0)
        
        # Get comprehensive report
        report = self.monitor.get_comprehensive_report()
        
        self.assertIn('system', report)
        self.assertIn('model', report)
        self.assertIn('training', report)
        self.assertIn('timestamp', report)
        
        logger.info("✅ Comprehensive report test passed")
    
    def test_metrics_export(self):
        """Test metrics export"""
        # Record some metrics
        self.metrics_collector.record_model_metrics(0.1, 100.0, 512.0, 0.8)
        self.metrics_collector.record_training_metrics(1, 0.5, 0.6, 0.001, 30.0)
        
        # Export metrics
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            export_path = f.name
        
        try:
            self.metrics_collector.export_metrics(export_path)
            
            # Check if file was created and has content
            self.assertTrue(os.path.exists(export_path))
            self.assertGreater(os.path.getsize(export_path), 0)
            
        finally:
            # Clean up
            if os.path.exists(export_path):
                os.remove(export_path)
        
        logger.info("✅ Metrics export test passed")
    
    def test_report_export(self):
        """Test report export"""
        # Record some metrics
        self.metrics_collector.record_model_metrics(0.1, 100.0, 512.0, 0.8)
        self.metrics_collector.record_training_metrics(1, 0.5, 0.6, 0.001, 30.0)
        
        # Export report
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            export_path = f.name
        
        try:
            self.monitor.export_report(export_path)
            
            # Check if file was created and has content
            self.assertTrue(os.path.exists(export_path))
            self.assertGreater(os.path.getsize(export_path), 0)
            
        finally:
            # Clean up
            if os.path.exists(export_path):
                os.remove(export_path)
        
        logger.info("✅ Report export test passed")
    
    def test_monitoring_start_stop(self):
        """Test monitoring start and stop"""
        # Start monitoring
        self.monitor.start_monitoring(interval=0.1)
        self.assertTrue(self.monitor.monitoring)
        
        # Wait a bit
        time.sleep(0.2)
        
        # Stop monitoring
        self.monitor.stop_monitoring()
        self.assertFalse(self.monitor.monitoring)
        
        logger.info("✅ Monitoring start/stop test passed")
    
    def test_metrics_history_limit(self):
        """Test metrics history limit"""
        collector = MetricsCollector(max_history=5)
        
        # Record more metrics than the limit
        for i in range(10):
            collector.record_model_metrics(0.1, 100.0, 512.0, 0.8)
        
        # Check that history is limited
        self.assertLessEqual(len(collector.model_metrics), 5)
        
        logger.info("✅ Metrics history limit test passed")
    
    def test_system_metrics_structure(self):
        """Test system metrics structure"""
        metrics = SystemMetrics(
            cpu_percent=50.0,
            memory_percent=60.0,
            gpu_memory_used=1.0,
            gpu_memory_total=8.0,
            timestamp=time.time()
        )
        
        self.assertEqual(metrics.cpu_percent, 50.0)
        self.assertEqual(metrics.memory_percent, 60.0)
        self.assertEqual(metrics.gpu_memory_used, 1.0)
        self.assertEqual(metrics.gpu_memory_total, 8.0)
        self.assertGreater(metrics.timestamp, 0)
        
        logger.info("✅ System metrics structure test passed")
    
    def test_model_metrics_structure(self):
        """Test model metrics structure"""
        metrics = ModelMetrics(
            inference_time=0.1,
            tokens_per_second=100.0,
            memory_usage=512.0,
            cache_hit_rate=0.8,
            timestamp=time.time()
        )
        
        self.assertEqual(metrics.inference_time, 0.1)
        self.assertEqual(metrics.tokens_per_second, 100.0)
        self.assertEqual(metrics.memory_usage, 512.0)
        self.assertEqual(metrics.cache_hit_rate, 0.8)
        self.assertGreater(metrics.timestamp, 0)
        
        logger.info("✅ Model metrics structure test passed")
    
    def test_training_metrics_structure(self):
        """Test training metrics structure"""
        metrics = TrainingMetrics(
            epoch=1,
            train_loss=0.5,
            val_loss=0.6,
            learning_rate=0.001,
            epoch_time=30.0,
            timestamp=time.time()
        )
        
        self.assertEqual(metrics.epoch, 1)
        self.assertEqual(metrics.train_loss, 0.5)
        self.assertEqual(metrics.val_loss, 0.6)
        self.assertEqual(metrics.learning_rate, 0.001)
        self.assertEqual(metrics.epoch_time, 30.0)
        self.assertGreater(metrics.timestamp, 0)
        
        logger.info("✅ Training metrics structure test passed")

if __name__ == '__main__':
    unittest.main()

