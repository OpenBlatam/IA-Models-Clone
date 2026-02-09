#!/usr/bin/env python3
"""
Ultimate TruthGPT Demo Script
============================

Script de demostración para la aplicación definitiva de TruthGPT con todas
las características ultra avanzadas integradas.

Características demostradas:
- Generación masiva de documentos
- Análisis completo de contenido
- Analytics avanzado
- Clustering inteligente
- Análisis de sentimientos
- Métricas de contenido
- Streaming de resultados
"""

import asyncio
import aiohttp
import json
import time
from datetime import datetime
from typing import Dict, List, Any
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

class UltimateTruthGPTDemo:
    """Demo class for Ultimate TruthGPT application."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def check_health(self) -> bool:
        """Check if the application is healthy."""
        try:
            async with self.session.get(f"{self.base_url}/health") as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ Health Check: {data['status']}")
                    print(f"   Components: {data['statistics']['healthy_components']}/{data['statistics']['total_components']}")
                    return True
                else:
                    print(f"❌ Health Check Failed: HTTP {response.status}")
                    return False
        except Exception as e:
            print(f"❌ Health Check Error: {e}")
            return False
    
    async def get_system_info(self) -> Dict[str, Any]:
        """Get system information."""
        try:
            async with self.session.get(f"{self.base_url}/system/info") as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"🚀 System: {data['application']}")
                    print(f"   Version: {data['version']}")
                    print(f"   Features: {len(data['features'])}")
                    return data
                else:
                    print(f"❌ System Info Failed: HTTP {response.status}")
                    return {}
        except Exception as e:
            print(f"❌ System Info Error: {e}")
            return {}
    
    async def demo_bulk_generation(self, query: str, document_count: int = 10) -> Dict[str, Any]:
        """Demo bulk document generation."""
        print(f"\n📝 Demo: Bulk Generation")
        print(f"   Query: {query}")
        print(f"   Documents: {document_count}")
        
        try:
            start_time = time.time()
            
            payload = {
                "query": query,
                "document_count": document_count,
                "business_area": "technology",
                "enable_workflow": True,
                "enable_redundancy_detection": True,
                "enable_analytics": True,
                "enable_ai_history": True,
                "enable_prompt_evolution": True,
                "enable_clustering": True,
                "enable_sentiment_analysis": True,
                "enable_content_metrics": True,
                "streaming": False,
                "metadata": {
                    "demo": True,
                    "timestamp": datetime.now().isoformat()
                }
            }
            
            async with self.session.post(
                f"{self.base_url}/generate/bulk",
                json=payload
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    end_time = time.time()
                    
                    print(f"✅ Generation Completed")
                    print(f"   Request ID: {data['request_id']}")
                    print(f"   Total Documents: {data['total_documents']}")
                    print(f"   Processing Time: {data['processing_time']:.2f}s")
                    print(f"   Average per Document: {data['processing_time']/data['total_documents']:.2f}s")
                    
                    # Show sample documents
                    if data['generated_documents']:
                        print(f"\n📄 Sample Documents:")
                        for i, doc in enumerate(data['generated_documents'][:3]):
                            print(f"   Document {i+1}: {doc['document_id']}")
                            print(f"     Quality: {doc['quality_score']:.2f}")
                            print(f"     Performance: {doc['performance_score']:.2f}")
                            print(f"     Sentiment: {doc['sentiment_score']:.2f}")
                            print(f"     Content Preview: {doc['content'][:100]}...")
                    
                    return data
                else:
                    error_text = await response.text()
                    print(f"❌ Generation Failed: HTTP {response.status}")
                    print(f"   Error: {error_text}")
                    return {}
                    
        except Exception as e:
            print(f"❌ Generation Error: {e}")
            return {}
    
    async def demo_streaming_generation(self, query: str, document_count: int = 5) -> None:
        """Demo streaming document generation."""
        print(f"\n🌊 Demo: Streaming Generation")
        print(f"   Query: {query}")
        print(f"   Documents: {document_count}")
        
        try:
            payload = {
                "query": query,
                "document_count": document_count,
                "business_area": "business",
                "enable_workflow": True,
                "enable_analytics": True,
                "enable_clustering": True,
                "enable_sentiment_analysis": True,
                "enable_content_metrics": True
            }
            
            async with self.session.post(
                f"{self.base_url}/generate/bulk/stream",
                json=payload
            ) as response:
                if response.status == 200:
                    print(f"✅ Streaming Started")
                    
                    async for line in response.content:
                        if line:
                            line_str = line.decode('utf-8').strip()
                            if line_str.startswith('data: '):
                                data_str = line_str[6:]  # Remove 'data: ' prefix
                                try:
                                    data = json.loads(data_str)
                                    if data['type'] == 'start':
                                        print(f"   Request ID: {data['request_id']}")
                                    elif data['type'] == 'document':
                                        doc = data
                                        print(f"   📄 {doc['document_id']}: Quality={doc['analysis'].get('quality_score', 0):.2f}, Sentiment={doc['analysis'].get('sentiment_score', 0):.2f}")
                                    elif data['type'] == 'complete':
                                        print(f"   ✅ Streaming Completed")
                                        break
                                except json.JSONDecodeError:
                                    continue
                else:
                    error_text = await response.text()
                    print(f"❌ Streaming Failed: HTTP {response.status}")
                    print(f"   Error: {error_text}")
                    
        except Exception as e:
            print(f"❌ Streaming Error: {e}")
    
    async def demo_analytics(self) -> None:
        """Demo analytics features."""
        print(f"\n📊 Demo: Advanced Analytics")
        
        try:
            # Get analytics report
            async with self.session.get(f"{self.base_url}/analytics/report") as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ Analytics Report Retrieved")
                    print(f"   Total Analyses: {data.get('total_analyses', 0)}")
                    print(f"   Average Quality: {data.get('average_quality', 0):.2f}")
                    print(f"   Average Performance: {data.get('average_performance', 0):.2f}")
                    
                    if 'predictions' in data:
                        predictions = data['predictions']
                        print(f"   Predictions Available: {len(predictions)}")
                else:
                    print(f"❌ Analytics Report Failed: HTTP {response.status}")
            
            # Get predictions
            async with self.session.get(f"{self.base_url}/analytics/predictions") as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ Predictions Retrieved")
                    print(f"   Quality Predictions: {len(data.get('quality_predictions', []))}")
                    print(f"   Performance Predictions: {len(data.get('performance_predictions', []))}")
                else:
                    print(f"❌ Predictions Failed: HTTP {response.status}")
                    
        except Exception as e:
            print(f"❌ Analytics Error: {e}")
    
    async def demo_ai_history(self) -> None:
        """Demo AI history analysis."""
        print(f"\n🤖 Demo: AI History Analysis")
        
        try:
            async with self.session.get(f"{self.base_url}/ai-history/analysis") as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ AI History Analysis Retrieved")
                    print(f"   Models Analyzed: {data.get('total_models', 0)}")
                    print(f"   Evolution Trends: {len(data.get('evolution_trends', []))}")
                    print(f"   Performance Comparison: {len(data.get('performance_comparison', []))}")
                else:
                    print(f"❌ AI History Analysis Failed: HTTP {response.status}")
                    
        except Exception as e:
            print(f"❌ AI History Error: {e}")
    
    async def demo_prompt_evolution(self) -> None:
        """Demo prompt evolution."""
        print(f"\n🧬 Demo: Prompt Evolution")
        
        try:
            async with self.session.get(f"{self.base_url}/prompt-evolution/analysis") as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ Prompt Evolution Analysis Retrieved")
                    print(f"   Prompts Analyzed: {data.get('total_prompts', 0)}")
                    print(f"   Evolution Cycles: {data.get('evolution_cycles', 0)}")
                    print(f"   Optimization Strategies: {len(data.get('optimization_strategies', []))}")
                else:
                    print(f"❌ Prompt Evolution Analysis Failed: HTTP {response.status}")
                    
        except Exception as e:
            print(f"❌ Prompt Evolution Error: {e}")
    
    async def demo_clustering(self) -> None:
        """Demo clustering analysis."""
        print(f"\n🔗 Demo: Smart Clustering")
        
        try:
            async with self.session.get(f"{self.base_url}/clustering/analysis") as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ Clustering Analysis Retrieved")
                    print(f"   Clusters Found: {data.get('total_clusters', 0)}")
                    print(f"   Similarity Analysis: {len(data.get('similarity_analysis', []))}")
                    print(f"   Clustering Methods: {len(data.get('clustering_methods', []))}")
                else:
                    print(f"❌ Clustering Analysis Failed: HTTP {response.status}")
                    
        except Exception as e:
            print(f"❌ Clustering Error: {e}")
    
    async def demo_sentiment_analysis(self) -> None:
        """Demo sentiment analysis."""
        print(f"\n😊 Demo: Advanced Sentiment Analysis")
        
        try:
            async with self.session.get(f"{self.base_url}/sentiment/analysis") as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ Sentiment Analysis Retrieved")
                    print(f"   Documents Analyzed: {data.get('total_documents', 0)}")
                    print(f"   Emotion Analysis: {len(data.get('emotion_analysis', []))}")
                    print(f"   Sentiment Trends: {len(data.get('sentiment_trends', []))}")
                else:
                    print(f"❌ Sentiment Analysis Failed: HTTP {response.status}")
                    
        except Exception as e:
            print(f"❌ Sentiment Analysis Error: {e}")
    
    async def demo_content_metrics(self) -> None:
        """Demo content metrics."""
        print(f"\n📏 Demo: Content Metrics")
        
        try:
            async with self.session.get(f"{self.base_url}/content-metrics/report") as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ Content Metrics Retrieved")
                    print(f"   Metrics Analyzed: {data.get('data_summary', {}).get('total_metrics_analyzed', 0)}")
                    
                    avg_metrics = data.get('average_metrics', {})
                    if avg_metrics:
                        print(f"   Average Quality: {avg_metrics.get('quality_score', 0):.2f}")
                        print(f"   Average Performance: {avg_metrics.get('performance_score', 0):.2f}")
                        print(f"   Average Readability: {avg_metrics.get('readability_score', 0):.2f}")
                        print(f"   Average Engagement: {avg_metrics.get('engagement_score', 0):.2f}")
                else:
                    print(f"❌ Content Metrics Failed: HTTP {response.status}")
                    
        except Exception as e:
            print(f"❌ Content Metrics Error: {e}")
    
    async def run_full_demo(self) -> None:
        """Run the complete demo."""
        print("=" * 80)
        print("🚀 ULTIMATE TRUTHGPT DEMO")
        print("   Final Definitive Version")
        print("=" * 80)
        
        # Check health
        if not await self.check_health():
            print("❌ Application is not healthy. Please check the logs.")
            return
        
        # Get system info
        await self.get_system_info()
        
        # Demo queries
        demo_queries = [
            "Inteligencia artificial y el futuro del trabajo",
            "Sostenibilidad y tecnología verde",
            "Innovación en el sector salud",
            "Transformación digital en empresas"
        ]
        
        # Run demos
        for i, query in enumerate(demo_queries):
            print(f"\n{'='*60}")
            print(f"DEMO {i+1}/{len(demo_queries)}")
            print(f"{'='*60}")
            
            # Bulk generation demo
            await self.demo_bulk_generation(query, 5)
            
            # Wait a bit between demos
            await asyncio.sleep(2)
        
        # Streaming demo
        print(f"\n{'='*60}")
        print("STREAMING DEMO")
        print(f"{'='*60}")
        await self.demo_streaming_generation("Tecnología del futuro", 3)
        
        # Analytics demos
        print(f"\n{'='*60}")
        print("ANALYTICS DEMOS")
        print(f"{'='*60}")
        await self.demo_analytics()
        await self.demo_ai_history()
        await self.demo_prompt_evolution()
        await self.demo_clustering()
        await self.demo_sentiment_analysis()
        await self.demo_content_metrics()
        
        print(f"\n{'='*80}")
        print("✅ DEMO COMPLETED SUCCESSFULLY!")
        print("   All features demonstrated successfully.")
        print("   Check the application logs for detailed information.")
        print(f"{'='*80}")

async def main():
    """Main demo function."""
    demo = UltimateTruthGPTDemo()
    
    async with demo:
        await demo.run_full_demo()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        sys.exit(1)
