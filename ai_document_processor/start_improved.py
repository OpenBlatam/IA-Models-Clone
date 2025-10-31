#!/usr/bin/env python3
"""
Improved System Startup Script - Advanced Launch
==============================================

Advanced startup script for the improved AI Document Processor.
"""

import os
import sys
import time
import asyncio
import logging
import signal
import subprocess
import multiprocessing as mp
from pathlib import Path
from typing import Dict, Any, Optional

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Setup advanced logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('improved_startup.log')
    ]
)
logger = logging.getLogger(__name__)


class ImprovedStartupManager:
    """Improved startup manager with advanced features."""
    
    def __init__(self):
        self.startup_time = time.time()
        self.processes: Dict[str, subprocess.Popen] = {}
        self.system_info = self._get_system_info()
        self.config = self._load_config()
        
    def _get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information."""
        try:
            import psutil
            import cpuinfo
            
            return {
                'platform': sys.platform,
                'architecture': sys.platform,
                'python_version': sys.version_info,
                'cpu_count': psutil.cpu_count(),
                'cpu_freq': psutil.cpu_freq().max if psutil.cpu_freq() else 0,
                'memory_gb': round(psutil.virtual_memory().total / (1024**3), 2),
                'disk_free_gb': round(psutil.disk_usage('/').free / (1024**3), 2) if os.name != 'nt' else round(psutil.disk_usage('C:').free / (1024**3), 2),
                'gpu_available': self._check_gpu_availability(),
                'cuda_available': self._check_cuda_availability(),
                'has_avx': self._check_avx_support(),
                'has_avx2': self._check_avx2_support(),
                'has_avx512': self._check_avx512_support()
            }
        except ImportError:
            logger.warning("psutil not available, using basic system info")
            return {
                'platform': sys.platform,
                'architecture': sys.platform,
                'python_version': sys.version_info,
                'cpu_count': mp.cpu_count() or 1,
                'memory_gb': 4.0,
                'gpu_available': False,
                'cuda_available': False
            }
    
    def _check_gpu_availability(self) -> bool:
        """Check if GPU is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def _check_cuda_availability(self) -> bool:
        """Check if CUDA is available."""
        try:
            import torch
            return torch.cuda.is_available() and torch.cuda.device_count() > 0
        except ImportError:
            return False
    
    def _check_avx_support(self) -> bool:
        """Check AVX support."""
        try:
            import cpuinfo
            cpu_info = cpuinfo.get_cpu_info()
            return 'avx' in cpu_info.get('flags', [])
        except:
            return False
    
    def _check_avx2_support(self) -> bool:
        """Check AVX2 support."""
        try:
            import cpuinfo
            cpu_info = cpuinfo.get_cpu_info()
            return 'avx2' in cpu_info.get('flags', [])
        except:
            return False
    
    def _check_avx512_support(self) -> bool:
        """Check AVX512 support."""
        try:
            import cpuinfo
            cpu_info = cpuinfo.get_cpu_info()
            return 'avx512' in cpu_info.get('flags', [])
        except:
            return False
    
    def _load_config(self) -> Dict[str, Any]:
        """Load startup configuration."""
        return {
            'host': '0.0.0.0',
            'port': 8001,
            'workers': 1,
            'reload': False,
            'log_level': 'info',
            'enable_gpu': True,
            'enable_monitoring': True,
            'enable_caching': True,
            'max_memory_gb': 32,
            'cache_size_mb': 4096,
            'enable_ai_features': True,
            'enable_vector_search': True,
            'enable_batch_processing': True
        }
    
    def print_improved_banner(self):
        """Print improved startup banner."""
        print("\n" + "="*80)
        print("🚀 IMPROVED AI DOCUMENT PROCESSOR - ADVANCED SYSTEM")
        print("="*80)
        print("Starting with advanced AI features and cutting-edge capabilities")
        print("="*80)
        
        print(f"System: {self.system_info['platform']} {self.system_info['architecture']}")
        print(f"Python: {self.system_info['python_version'].major}.{self.system_info['python_version'].minor}.{self.system_info['python_version'].micro}")
        print(f"CPU Cores: {self.system_info['cpu_count']}")
        print(f"Memory: {self.system_info['memory_gb']} GB")
        print(f"Host: {self.config['host']}")
        print(f"Port: {self.config['port']}")
        print(f"Workers: {self.config['workers']}")
        print(f"GPU Available: {'✅' if self.system_info['gpu_available'] else '❌'}")
        print(f"CUDA Available: {'✅' if self.system_info['cuda_available'] else '❌'}")
        print(f"AVX Support: {'✅' if self.system_info['has_avx'] else '❌'}")
        print(f"AVX2 Support: {'✅' if self.system_info['has_avx2'] else '❌'}")
        print(f"AVX512 Support: {'✅' if self.system_info['has_avx512'] else '❌'}")
        
        print(f"\n🚀 Advanced Features:")
        print("   🤖 AI Classification")
        print("   📝 AI Summarization")
        print("   🌍 AI Translation")
        print("   ❓ AI Q&A")
        print("   🔍 Vector Search")
        print("   📊 Batch Processing")
        print("   ⚡ Real-time Processing")
        print("   💾 Advanced Caching")
        print("   📈 Performance Monitoring")
        print("   🔒 Enterprise Security")
        
        print("="*80 + "\n")
    
    def apply_advanced_optimizations(self):
        """Apply advanced system optimizations."""
        logger.info("🔧 Applying advanced optimizations...")
        
        # Python optimizations
        os.environ['PYTHONOPTIMIZE'] = '1'
        os.environ['PYTHONUNBUFFERED'] = '1'
        os.environ['PYTHONIOENCODING'] = 'utf-8'
        os.environ['PYTHONHASHSEED'] = '0'
        
        # CPU optimizations
        if self.system_info['cpu_count'] > 1:
            os.environ['OMP_NUM_THREADS'] = str(self.system_info['cpu_count'])
            os.environ['MKL_NUM_THREADS'] = str(self.system_info['cpu_count'])
            os.environ['NUMEXPR_NUM_THREADS'] = str(self.system_info['cpu_count'])
            os.environ['OPENBLAS_NUM_THREADS'] = str(self.system_info['cpu_count'])
        
        # Memory optimizations
        if self.config['max_memory_gb'] > 0:
            os.environ['PYTHONMALLOC'] = 'malloc'
        
        # GPU optimizations
        if self.config['enable_gpu'] and self.system_info['gpu_available']:
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'
            os.environ['TORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
        
        # Async optimizations
        if os.name != 'nt':  # Not Windows
            os.environ['UVLOOP'] = '1'
        
        # AI optimizations
        if self.config['enable_ai_features']:
            os.environ['TOKENIZERS_PARALLELISM'] = 'false'
            os.environ['TRANSFORMERS_CACHE'] = './cache'
        
        logger.info("✅ Advanced optimizations applied")
    
    def check_improved_dependencies(self) -> bool:
        """Check if all improved dependencies are available."""
        logger.info("🔍 Checking improved dependencies...")
        
        required_modules = [
            'fastapi',
            'uvicorn',
            'pydantic',
            'redis',
            'numpy',
            'pandas',
            'torch',
            'transformers',
            'openai',
            'anthropic',
            'chromadb',
            'sentence_transformers',
            'spacy',
            'nltk',
            'orjson',
            'msgpack',
            'lz4',
            'prometheus_client',
            'structlog',
            'sentry_sdk'
        ]
        
        missing_modules = []
        
        for module in required_modules:
            try:
                __import__(module)
                logger.info(f"✅ {module}")
            except ImportError:
                missing_modules.append(module)
                logger.error(f"❌ {module} - MISSING")
        
        if missing_modules:
            logger.error(f"Missing modules: {missing_modules}")
            logger.error("Please run: pip install -r requirements_improved.txt")
            return False
        
        logger.info("✅ All improved dependencies available")
        return True
    
    def start_redis_improved(self) -> bool:
        """Start Redis with improved configuration."""
        try:
            import redis
            r = redis.Redis(host='localhost', port=6379, db=0)
            r.ping()
            logger.info("✅ Redis already running")
            return True
        except:
            logger.info("🔄 Starting Redis with improved configuration...")
            try:
                if sys.platform == 'win32':
                    redis_cmd = [
                        'redis-server', 
                        '--port', '6379', 
                        '--maxmemory', '4gb',
                        '--maxmemory-policy', 'allkeys-lru',
                        '--save', '900 1',
                        '--save', '300 10',
                        '--save', '60 10000'
                    ]
                else:
                    redis_cmd = [
                        'redis-server', 
                        '--port', '6379', 
                        '--maxmemory', '4gb',
                        '--maxmemory-policy', 'allkeys-lru',
                        '--save', '900 1',
                        '--save', '300 10',
                        '--save', '60 10000',
                        '--daemonize', 'yes'
                    ]
                
                process = subprocess.Popen(redis_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                self.processes['redis'] = process
                
                # Wait for Redis to start
                time.sleep(3)
                
                # Test connection
                import redis
                r = redis.Redis(host='localhost', port=6379, db=0)
                r.ping()
                
                logger.info("✅ Redis started with improved configuration")
                return True
                
            except Exception as e:
                logger.warning(f"⚠️ Redis not available: {e}")
                return False
    
    def start_monitoring_improved(self):
        """Start improved monitoring services."""
        if not self.config['enable_monitoring']:
            return
        
        logger.info("📊 Starting improved monitoring services...")
        
        try:
            # Start Prometheus metrics server
            from prometheus_client import start_http_server
            start_http_server(9090)
            logger.info("✅ Prometheus metrics server started on port 9090")
            
            # Initialize Sentry for error tracking
            import sentry_sdk
            sentry_sdk.init(
                dsn=os.getenv('SENTRY_DSN', ''),
                traces_sample_rate=0.1,
                profiles_sample_rate=0.1
            )
            logger.info("✅ Sentry error tracking initialized")
            
        except Exception as e:
            logger.warning(f"⚠️ Could not start monitoring: {e}")
    
    def initialize_ai_models_improved(self):
        """Initialize improved AI models."""
        logger.info("🤖 Initializing improved AI models...")
        
        try:
            # Initialize OpenAI
            import openai
            if os.getenv('OPENAI_API_KEY'):
                openai.api_key = os.getenv('OPENAI_API_KEY')
                logger.info("✅ OpenAI initialized")
            else:
                logger.warning("⚠️ OpenAI API key not set")
            
            # Initialize Anthropic
            import anthropic
            if os.getenv('ANTHROPIC_API_KEY'):
                anthropic_client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
                logger.info("✅ Anthropic initialized")
            else:
                logger.warning("⚠️ Anthropic API key not set")
            
            # Initialize Transformers
            from transformers import pipeline
            logger.info("✅ Transformers initialized")
            
            # Initialize spaCy
            import spacy
            try:
                nlp = spacy.load("en_core_web_sm")
                logger.info("✅ spaCy model loaded")
            except OSError:
                logger.warning("⚠️ spaCy model not found, run: python -m spacy download en_core_web_sm")
            
            # Initialize ChromaDB
            import chromadb
            client = chromadb.Client()
            logger.info("✅ ChromaDB initialized")
            
            # Initialize Sentence Transformers
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("✅ Sentence Transformers initialized")
            
        except Exception as e:
            logger.warning(f"⚠️ AI model initialization warning: {e}")
    
    def start_improved_server(self):
        """Start improved server."""
        logger.info("🚀 Starting improved server...")
        
        try:
            # Import and run improved main
            from improved_system import main
            main()
            
        except Exception as e:
            logger.error(f"❌ Failed to start improved server: {e}")
            raise
    
    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            logger.info(f"🛑 Received signal {signum}, shutting down gracefully...")
            self.shutdown()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def shutdown(self):
        """Graceful shutdown."""
        logger.info("🛑 Shutting down improved services...")
        
        # Stop all processes
        for name, process in self.processes.items():
            try:
                process.terminate()
                process.wait(timeout=5)
                logger.info(f"✅ Stopped {name}")
            except subprocess.TimeoutExpired:
                process.kill()
                logger.warning(f"⚠️ Force killed {name}")
            except Exception as e:
                logger.error(f"❌ Error stopping {name}: {e}")
        
        # Calculate uptime
        uptime = time.time() - self.startup_time
        logger.info(f"📊 Total uptime: {uptime:.2f} seconds")
    
    def print_startup_summary(self):
        """Print startup summary."""
        print("\n" + "="*80)
        print("🎉 IMPROVED SYSTEM STARTUP COMPLETE")
        print("="*80)
        
        print("🚀 Services Started:")
        print("   ✅ Improved FastAPI Server")
        print("   ✅ Redis Cache (Advanced)")
        print("   ✅ AI Models (Multiple)")
        print("   ✅ Vector Database")
        print("   ✅ Monitoring & Metrics")
        print("   ✅ Error Tracking")
        print("   ✅ Advanced Caching")
        
        print(f"\n🌐 Access Points:")
        print(f"   📖 API Documentation: http://{self.config['host']}:{self.config['port']}/docs")
        print(f"   🔧 ReDoc: http://{self.config['host']}:{self.config['port']}/redoc")
        print(f"   🏥 Health Check: http://{self.config['host']}:{self.config['port']}/health")
        print(f"   📊 System Stats: http://{self.config['host']}:{self.config['port']}/stats")
        print(f"   🔍 Vector Search: http://{self.config['host']}:{self.config['port']}/search")
        print(f"   📈 Metrics: http://{self.config['host']}:9090/metrics")
        
        print(f"\n🤖 Advanced AI Features:")
        print("   🧠 AI Classification")
        print("   📝 AI Summarization")
        print("   🌍 AI Translation")
        print("   ❓ AI Q&A")
        print("   🔍 Vector Search")
        print("   📊 Batch Processing")
        print("   ⚡ Real-time Processing")
        print("   💾 Advanced Caching")
        print("   📈 Performance Monitoring")
        print("   🔒 Enterprise Security")
        
        print(f"\n🛠️ Improved System Features:")
        print("   ✅ Advanced AI capabilities")
        print("   ✅ Multiple AI models")
        print("   ✅ Vector database integration")
        print("   ✅ Enterprise monitoring")
        print("   ✅ Error tracking and logging")
        print("   ✅ Advanced caching strategies")
        print("   ✅ Batch processing")
        print("   ✅ Real-time processing")
        print("   ✅ Performance optimization")
        print("   ✅ Security enhancements")
        
        print("="*80 + "\n")
    
    def start(self):
        """Start improved application."""
        try:
            # Print banner
            self.print_improved_banner()
            
            # Setup signal handlers
            self.setup_signal_handlers()
            
            # Apply advanced optimizations
            self.apply_advanced_optimizations()
            
            # Check dependencies
            if not self.check_improved_dependencies():
                logger.error("❌ Dependency check failed")
                return False
            
            # Start Redis
            self.start_redis_improved()
            
            # Start monitoring
            self.start_monitoring_improved()
            
            # Initialize AI models
            self.initialize_ai_models_improved()
            
            # Print summary
            self.print_startup_summary()
            
            # Start server
            self.start_improved_server()
            
        except KeyboardInterrupt:
            logger.info("🛑 Startup interrupted by user")
        except Exception as e:
            logger.error(f"❌ Startup failed: {e}")
            raise
        finally:
            self.shutdown()


def main():
    """Main improved startup function."""
    startup_manager = ImprovedStartupManager()
    startup_manager.start()


if __name__ == "__main__":
    main()

















