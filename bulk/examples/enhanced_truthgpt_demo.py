"""
Enhanced TruthGPT Bulk Document Generation Demo
==============================================

Demo mejorado que muestra todas las características avanzadas del sistema TruthGPT:
- Caching inteligente
- Optimización de prompts
- Balanceo de carga de modelos
- Monitoreo avanzado
- Métricas de calidad
- Recuperación de errores mejorada
"""

import asyncio
import logging
import json
from datetime import datetime
from pathlib import Path
import sys
import os

# Add the parent directory to the path to import our modules
sys.path.append(str(Path(__file__).parent.parent))

from core.enhanced_truthgpt_processor import EnhancedTruthGPTProcessor, get_global_enhanced_processor
from config.truthgpt_config import TruthGPTConfig
from monitoring.advanced_monitoring import AdvancedMonitoringSystem

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EnhancedTruthGPTDemo:
    """Enhanced demo class for TruthGPT bulk document generation."""
    
    def __init__(self):
        self.processor = get_global_enhanced_processor()
        self.monitoring = AdvancedMonitoringSystem(self.processor)
        self.setup_enhanced_callbacks()
    
    def setup_enhanced_callbacks(self):
        """Setup enhanced callbacks for monitoring document generation."""
        
        async def on_document_generated(task):
            """Enhanced callback when a document is generated."""
            logger.info(f"✅ Enhanced Document Generated: {task.document_type} for {task.business_area}")
            logger.info(f"   Task ID: {task.id}")
            logger.info(f"   Quality Score: {task.quality_score:.2f}")
            logger.info(f"   Processing Time: {task.processing_time:.2f}s")
            logger.info(f"   Model Used: {task.model_used}")
            logger.info(f"   Cache Hit: {task.cache_hit}")
            logger.info(f"   Optimization Applied: {task.optimization_applied}")
            
            # Save enhanced document to file
            await self.save_enhanced_document_to_file(task)
        
        async def on_quality_assessed(task, quality_score):
            """Callback when quality is assessed."""
            if quality_score >= 0.9:
                logger.info(f"🌟 Excellent Quality: {quality_score:.2f}")
            elif quality_score >= 0.8:
                logger.info(f"👍 Good Quality: {quality_score:.2f}")
            elif quality_score >= 0.7:
                logger.info(f"✅ Acceptable Quality: {quality_score:.2f}")
            else:
                logger.warning(f"⚠️ Low Quality: {quality_score:.2f}")
        
        async def on_alert(alert):
            """Callback for monitoring alerts."""
            logger.warning(f"🚨 ALERT [{alert.severity.upper()}]: {alert.type} - {alert.message}")
        
        async def on_metric_update(system_metrics, processing_metrics):
            """Callback for metric updates."""
            logger.debug(f"📊 Metrics: CPU {system_metrics.cpu_percent:.1f}%, "
                        f"Memory {system_metrics.memory_percent:.1f}%, "
                        f"Throughput {processing_metrics.throughput_per_minute:.1f} docs/min")
        
        async def on_prediction(prediction):
            """Callback for performance predictions."""
            logger.info(f"🔮 Prediction: Throughput={prediction.predicted_throughput:.1f}, "
                       f"Quality={prediction.predicted_quality_score:.2f}")
            if prediction.recommendations:
                logger.info(f"💡 Recommendations: {', '.join(prediction.recommendations)}")
        
        # Set enhanced callbacks
        self.processor.set_enhanced_callbacks(
            document_callback=on_document_generated,
            quality_callback=on_quality_assessed,
            error_callback=None
        )
        
        # Set monitoring callbacks
        self.monitoring.set_callbacks(
            alert_callback=on_alert,
            metric_callback=on_metric_update,
            prediction_callback=on_prediction
        )
    
    async def save_enhanced_document_to_file(self, task):
        """Save enhanced generated document to file with metadata."""
        try:
            # Create enhanced output directory
            output_dir = Path("enhanced_generated_documents")
            output_dir.mkdir(exist_ok=True)
            
            # Create filename with enhanced metadata
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{task.document_type}_{task.business_area}_{task.quality_score:.2f}_{timestamp}.md"
            filepath = output_dir / filename
            
            # Write enhanced content to file
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"# {task.document_type.replace('_', ' ').title()}\n")
                f.write(f"**Business Area:** {task.business_area}\n")
                f.write(f"**Generated:** {datetime.now().isoformat()}\n")
                f.write(f"**Task ID:** {task.id}\n")
                f.write(f"**Quality Score:** {task.quality_score:.2f}\n")
                f.write(f"**Processing Time:** {task.processing_time:.2f}s\n")
                f.write(f"**Model Used:** {task.model_used}\n")
                f.write(f"**Cache Hit:** {task.cache_hit}\n")
                f.write(f"**Optimization Applied:** {task.optimization_applied}\n")
                f.write(f"**Variations Generated:** {task.variations_generated}\n")
                f.write(f"**Cross References:** {', '.join(task.cross_references) if task.cross_references else 'None'}\n")
                f.write("\n---\n\n")
                f.write(task.content)
            
            logger.info(f"💾 Enhanced document saved: {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save enhanced document: {e}")
    
    async def demo_enhanced_basic_generation(self):
        """Demo enhanced basic bulk document generation."""
        logger.info("🚀 Starting Enhanced Basic Bulk Generation Demo")
        
        # Submit an enhanced bulk request
        request_id = await self.processor.submit_enhanced_bulk_request(
            query="Estrategias de transformación digital para pequeñas empresas",
            document_types=["business_plan", "marketing_strategy", "operational_manual"],
            business_areas=["strategy", "marketing", "operations"],
            max_documents=9,  # 3 types × 3 areas
            continuous_mode=True,
            priority=1,
            enable_caching=True,
            enable_optimization=True,
            quality_threshold=0.85,
            enable_variations=True,
            max_variations=2,
            enable_cross_referencing=True,
            target_audience="empresarios y directivos",
            language="es",
            tone="professional"
        )
        
        logger.info(f"📝 Enhanced request submitted: {request_id}")
        
        # Monitor progress with enhanced metrics
        await self.monitor_enhanced_request_progress(request_id)
        
        return request_id
    
    async def demo_enhanced_continuous_generation(self):
        """Demo enhanced continuous document generation."""
        logger.info("🔄 Starting Enhanced Continuous Generation Demo")
        
        # Submit a large enhanced bulk request
        request_id = await self.processor.submit_enhanced_bulk_request(
            query="Implementación de inteligencia artificial en empresas modernas",
            document_types=[
                "business_plan", "marketing_strategy", "technical_documentation",
                "financial_analysis", "hr_policy", "operational_manual",
                "implementation_plan", "risk_assessment", "quality_manual"
            ],
            business_areas=[
                "strategy", "marketing", "technical", "finance", "hr", "operations",
                "innovation", "quality_assurance", "risk_management"
            ],
            max_documents=50,  # Large number for continuous generation
            continuous_mode=True,
            priority=1,
            enable_caching=True,
            enable_optimization=True,
            quality_threshold=0.9,  # High quality threshold
            enable_variations=True,
            max_variations=5,
            enable_cross_referencing=True,
            enable_evolution=True,
            target_audience="directivos y equipos técnicos",
            language="es",
            tone="authoritative"
        )
        
        logger.info(f"📝 Large enhanced request submitted: {request_id}")
        
        # Monitor progress with periodic updates
        await self.monitor_enhanced_request_progress(request_id, update_interval=15)
        
        return request_id
    
    async def demo_enhanced_multiple_requests(self):
        """Demo multiple enhanced concurrent requests."""
        logger.info("🎯 Starting Enhanced Multiple Requests Demo")
        
        # Submit multiple enhanced requests with different configurations
        requests = []
        
        # Request 1: Marketing focus with high quality
        request1 = await self.processor.submit_enhanced_bulk_request(
            query="Estrategias de marketing digital y redes sociales",
            document_types=["marketing_strategy", "content_strategy", "sales_presentation"],
            business_areas=["marketing", "content", "sales"],
            max_documents=15,
            continuous_mode=True,
            priority=1,
            enable_caching=True,
            enable_optimization=True,
            quality_threshold=0.9,
            enable_variations=True,
            max_variations=3,
            target_audience="equipos de marketing",
            language="es",
            tone="creative"
        )
        requests.append(request1)
        
        # Request 2: Technical focus with optimization
        request2 = await self.processor.submit_enhanced_bulk_request(
            query="Arquitectura de microservicios y cloud computing",
            document_types=["technical_documentation", "implementation_plan", "best_practices_guide"],
            business_areas=["technical", "operations", "innovation"],
            max_documents=12,
            continuous_mode=True,
            priority=2,
            enable_caching=True,
            enable_optimization=True,
            quality_threshold=0.85,
            enable_variations=True,
            max_variations=2,
            target_audience="desarrolladores y arquitectos",
            language="es",
            tone="technical"
        )
        requests.append(request2)
        
        # Request 3: Business focus with cross-referencing
        request3 = await self.processor.submit_enhanced_bulk_request(
            query="Planificación financiera y gestión de riesgos empresariales",
            document_types=["financial_analysis", "business_plan", "risk_assessment"],
            business_areas=["finance", "strategy", "risk_management"],
            max_documents=9,
            continuous_mode=True,
            priority=1,
            enable_caching=True,
            enable_optimization=True,
            quality_threshold=0.9,
            enable_variations=True,
            max_variations=2,
            enable_cross_referencing=True,
            target_audience="directivos financieros",
            language="es",
            tone="formal"
        )
        requests.append(request3)
        
        logger.info(f"📝 Multiple enhanced requests submitted: {requests}")
        
        # Monitor all requests
        await self.monitor_enhanced_multiple_requests(requests)
        
        return requests
    
    async def monitor_enhanced_request_progress(self, request_id: str, update_interval: int = 10):
        """Monitor the enhanced progress of a single request."""
        logger.info(f"👀 Monitoring enhanced request: {request_id}")
        
        while True:
            try:
                status = await self.processor.get_enhanced_request_status(request_id)
                if not status:
                    logger.warning(f"Request {request_id} not found")
                    break
                
                logger.info(f"📊 Enhanced Progress: {status['documents_generated']}/{status['max_documents']} "
                          f"({status['progress_percentage']:.1f}%)")
                logger.info(f"   Quality Score: {status['average_quality_score']:.2f}")
                logger.info(f"   Cache Hit Rate: {status['cache_hit_rate']:.1%}")
                logger.info(f"   Processing Efficiency: {status['processing_efficiency']:.1%}")
                logger.info(f"   Active Tasks: {status['active_tasks']}, Queued: {status['queued_tasks']}")
                
                # Check if completed
                if status['documents_generated'] >= status['max_documents']:
                    logger.info(f"✅ Enhanced request {request_id} completed!")
                    break
                
                await asyncio.sleep(update_interval)
                
            except Exception as e:
                logger.error(f"Error monitoring enhanced request: {e}")
                break
    
    async def monitor_enhanced_multiple_requests(self, request_ids: List[str]):
        """Monitor multiple enhanced requests concurrently."""
        logger.info(f"👀 Monitoring {len(request_ids)} enhanced requests")
        
        tasks = []
        for request_id in request_ids:
            task = asyncio.create_task(self.monitor_enhanced_request_progress(request_id, 15))
            tasks.append(task)
        
        # Wait for all monitoring tasks to complete
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def get_enhanced_final_results(self, request_id: str):
        """Get enhanced final results for a request."""
        try:
            # Get enhanced status
            status = await self.processor.get_enhanced_request_status(request_id)
            
            logger.info(f"📋 Enhanced Final Results for {request_id}:")
            logger.info(f"   Total Documents: {status['documents_generated']}")
            logger.info(f"   Average Quality Score: {status['average_quality_score']:.2f}")
            logger.info(f"   Cache Hit Rate: {status['cache_hit_rate']:.1%}")
            logger.info(f"   Processing Efficiency: {status['processing_efficiency']:.1%}")
            logger.info(f"   Optimization Enabled: {status['optimization_enabled']}")
            logger.info(f"   Variations Enabled: {status['variations_enabled']}")
            logger.info(f"   Target Audience: {status['target_audience']}")
            logger.info(f"   Language: {status['language']}")
            logger.info(f"   Tone: {status['tone']}")
            
            # Get documents with quality analysis
            documents = []
            quality_scores = []
            
            for task in self.processor.completed_tasks.values():
                if task.request_id == request_id and task.status == "completed":
                    documents.append(task)
                    quality_scores.append(task.quality_score)
            
            # Quality analysis
            if quality_scores:
                logger.info("   Quality Analysis:")
                logger.info(f"     Average: {statistics.mean(quality_scores):.2f}")
                logger.info(f"     Min: {min(quality_scores):.2f}")
                logger.info(f"     Max: {max(quality_scores):.2f}")
                
                excellent = len([s for s in quality_scores if s >= 0.9])
                good = len([s for s in quality_scores if 0.8 <= s < 0.9])
                acceptable = len([s for s in quality_scores if 0.7 <= s < 0.8])
                poor = len([s for s in quality_scores if s < 0.7])
                
                logger.info(f"     Excellent (≥0.9): {excellent}")
                logger.info(f"     Good (0.8-0.9): {good}")
                logger.info(f"     Acceptable (0.7-0.8): {acceptable}")
                logger.info(f"     Poor (<0.7): {poor}")
            
            # Model usage analysis
            model_usage = {}
            cache_hits = 0
            optimization_applied = 0
            
            for task in documents:
                if task.model_used:
                    model_usage[task.model_used] = model_usage.get(task.model_used, 0) + 1
                if task.cache_hit:
                    cache_hits += 1
                if task.optimization_applied:
                    optimization_applied += 1
            
            logger.info("   Model Usage:")
            for model, count in model_usage.items():
                logger.info(f"     {model}: {count} documents")
            
            logger.info(f"   Cache Hits: {cache_hits}/{len(documents)} ({cache_hits/len(documents)*100:.1f}%)")
            logger.info(f"   Optimization Applied: {optimization_applied}/{len(documents)} ({optimization_applied/len(documents)*100:.1f}%)")
            
            return documents
            
        except Exception as e:
            logger.error(f"Error getting enhanced final results: {e}")
            return []
    
    async def demo_monitoring_dashboard(self):
        """Demo the monitoring dashboard."""
        logger.info("📊 Starting Monitoring Dashboard Demo")
        
        # Start monitoring
        asyncio.create_task(self.monitoring.start_monitoring())
        
        # Wait a bit for metrics to be collected
        await asyncio.sleep(30)
        
        # Get dashboard data
        dashboard_data = self.monitoring.get_monitoring_dashboard_data()
        
        logger.info("📈 Monitoring Dashboard Data:")
        logger.info(f"   System Health: {dashboard_data.get('system_metrics', {}).get('cpu_percent', 'N/A')}% CPU")
        logger.info(f"   Memory Usage: {dashboard_data.get('system_metrics', {}).get('memory_percent', 'N/A')}%")
        logger.info(f"   Active Alerts: {dashboard_data.get('alerts', {}).get('summary', {}).get('total', 0)}")
        logger.info(f"   Performance Trends: {dashboard_data.get('performance_trends', {})}")
        
        # Get historical metrics
        historical_data = self.monitoring.get_historical_metrics(1)  # Last hour
        logger.info(f"   Historical Summary: {historical_data.get('summary', {})}")
    
    async def run_comprehensive_enhanced_demo(self):
        """Run a comprehensive demo of all enhanced features."""
        logger.info("🎬 Starting Comprehensive Enhanced TruthGPT Demo")
        logger.info("=" * 70)
        
        try:
            # Start the enhanced processor
            logger.info("🚀 Starting enhanced continuous processing...")
            asyncio.create_task(self.processor.start_enhanced_processing())
            await asyncio.sleep(3)  # Give it time to start
            
            # Demo 1: Enhanced basic bulk generation
            logger.info("\n📝 Demo 1: Enhanced Basic Bulk Generation")
            logger.info("-" * 50)
            request1 = await self.demo_enhanced_basic_generation()
            await self.get_enhanced_final_results(request1)
            
            await asyncio.sleep(5)  # Brief pause
            
            # Demo 2: Enhanced multiple concurrent requests
            logger.info("\n📝 Demo 2: Enhanced Multiple Concurrent Requests")
            logger.info("-" * 50)
            requests = await self.demo_enhanced_multiple_requests()
            
            # Get results for all requests
            for request_id in requests:
                await self.get_enhanced_final_results(request_id)
            
            await asyncio.sleep(5)  # Brief pause
            
            # Demo 3: Enhanced large continuous generation
            logger.info("\n📝 Demo 3: Enhanced Large Continuous Generation")
            logger.info("-" * 50)
            request3 = await self.demo_enhanced_continuous_generation()
            await self.get_enhanced_final_results(request3)
            
            # Demo 4: Monitoring dashboard
            logger.info("\n📊 Demo 4: Enhanced Monitoring Dashboard")
            logger.info("-" * 50)
            await self.demo_monitoring_dashboard()
            
            # Final enhanced statistics
            logger.info("\n📊 Final Enhanced Processing Statistics")
            logger.info("-" * 50)
            stats = self.processor.get_enhanced_processing_stats()
            for key, value in stats.items():
                logger.info(f"   {key}: {value}")
            
            logger.info("\n🎉 Enhanced demo completed successfully!")
            
        except Exception as e:
            logger.error(f"Enhanced demo failed: {e}")
        finally:
            # Stop processing
            self.processor.stop_processing()
            self.monitoring.stop_monitoring()
            logger.info("🛑 Enhanced processing and monitoring stopped")

async def main():
    """Main function to run the enhanced demo."""
    demo = EnhancedTruthGPTDemo()
    await demo.run_comprehensive_enhanced_demo()

if __name__ == "__main__":
    # Check if OpenRouter API key is configured
    if not os.getenv("OPENROUTER_API_KEY"):
        logger.error("❌ OPENROUTER_API_KEY environment variable not set!")
        logger.info("Please set your OpenRouter API key:")
        logger.info("export OPENROUTER_API_KEY='your-api-key-here'")
        sys.exit(1)
    
    # Run the enhanced demo
    asyncio.run(main())



























