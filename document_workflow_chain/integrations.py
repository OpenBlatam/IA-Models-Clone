"""
Integrations with Other Blatam Academy Services
==============================================

This module provides integrations with other services in the Blatam Academy
ecosystem, including content redundancy detection, SEO optimization,
brand voice consistency, and more.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import httpx
import json

# Configure logging
logger = logging.getLogger(__name__)

class BlatamServiceIntegration:
    """Base class for integrating with Blatam Academy services"""
    
    def __init__(self, service_name: str, base_url: str, api_key: Optional[str] = None):
        self.service_name = service_name
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.client = httpx.AsyncClient(timeout=30.0)
    
    async def close(self):
        """Close the HTTP client"""
        await self.client.aclose()
    
    async def _make_request(
        self, 
        method: str, 
        endpoint: str, 
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Make HTTP request to the service"""
        try:
            url = f"{self.base_url}{endpoint}"
            headers = {"Content-Type": "application/json"}
            
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            
            response = await self.client.request(
                method=method,
                url=url,
                headers=headers,
                json=data,
                params=params
            )
            
            response.raise_for_status()
            return response.json()
            
        except httpx.HTTPError as e:
            logger.error(f"HTTP error calling {self.service_name}: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error calling {self.service_name}: {str(e)}")
            raise

class ContentRedundancyDetector(BlatamServiceIntegration):
    """Integration with Content Redundancy Detector service"""
    
    def __init__(self, base_url: str = "http://localhost:8002", api_key: Optional[str] = None):
        super().__init__("Content Redundancy Detector", base_url, api_key)
    
    async def check_content_uniqueness(
        self, 
        content: str, 
        existing_content: List[str] = None
    ) -> Dict[str, Any]:
        """
        Check if content is unique compared to existing content
        
        Args:
            content: Content to check
            existing_content: List of existing content to compare against
            
        Returns:
            Dict with uniqueness score and recommendations
        """
        try:
            data = {
                "content": content,
                "existing_content": existing_content or []
            }
            
            result = await self._make_request("POST", "/api/v1/check-uniqueness", data)
            
            logger.info(f"Content uniqueness check completed: {result.get('uniqueness_score', 0)}")
            return result
            
        except Exception as e:
            logger.error(f"Error checking content uniqueness: {str(e)}")
            return {
                "uniqueness_score": 0.5,
                "is_unique": True,
                "similar_content": [],
                "recommendations": ["Unable to check uniqueness due to service error"]
            }
    
    async def suggest_improvements(self, content: str) -> List[str]:
        """Suggest improvements to make content more unique"""
        try:
            data = {"content": content}
            result = await self._make_request("POST", "/api/v1/suggest-improvements", data)
            
            return result.get("suggestions", [])
            
        except Exception as e:
            logger.error(f"Error getting improvement suggestions: {str(e)}")
            return ["Consider adding more specific details", "Include unique examples or case studies"]

class SEOOptimizer(BlatamServiceIntegration):
    """Integration with SEO service"""
    
    def __init__(self, base_url: str = "http://localhost:8018", api_key: Optional[str] = None):
        super().__init__("SEO Optimizer", base_url, api_key)
    
    async def optimize_content(
        self, 
        content: str, 
        target_keywords: List[str] = None,
        content_type: str = "blog_post"
    ) -> Dict[str, Any]:
        """
        Optimize content for SEO
        
        Args:
            content: Content to optimize
            target_keywords: Target keywords for optimization
            content_type: Type of content (blog_post, article, etc.)
            
        Returns:
            Dict with SEO optimization results
        """
        try:
            data = {
                "content": content,
                "target_keywords": target_keywords or [],
                "content_type": content_type
            }
            
            result = await self._make_request("POST", "/api/v1/optimize-content", data)
            
            logger.info(f"SEO optimization completed for {content_type}")
            return result
            
        except Exception as e:
            logger.error(f"Error optimizing content for SEO: {str(e)}")
            return {
                "optimized_content": content,
                "seo_score": 0.5,
                "keyword_density": {},
                "recommendations": ["Unable to optimize due to service error"]
            }
    
    async def generate_meta_description(self, content: str) -> str:
        """Generate SEO-optimized meta description"""
        try:
            data = {"content": content}
            result = await self._make_request("POST", "/api/v1/generate-meta-description", data)
            
            return result.get("meta_description", content[:160])
            
        except Exception as e:
            logger.error(f"Error generating meta description: {str(e)}")
            return content[:160] + "..." if len(content) > 160 else content

class BrandVoiceAnalyzer(BlatamServiceIntegration):
    """Integration with Brand Voice service"""
    
    def __init__(self, base_url: str = "http://localhost:8003", api_key: Optional[str] = None):
        super().__init__("Brand Voice Analyzer", base_url, api_key)
    
    async def analyze_brand_voice_consistency(
        self, 
        content: str, 
        brand_guidelines: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Analyze content for brand voice consistency
        
        Args:
            content: Content to analyze
            brand_guidelines: Brand voice guidelines
            
        Returns:
            Dict with brand voice analysis results
        """
        try:
            data = {
                "content": content,
                "brand_guidelines": brand_guidelines or {}
            }
            
            result = await self._make_request("POST", "/api/v1/analyze-brand-voice", data)
            
            logger.info(f"Brand voice analysis completed")
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing brand voice: {str(e)}")
            return {
                "consistency_score": 0.5,
                "tone_match": True,
                "style_match": True,
                "recommendations": ["Unable to analyze brand voice due to service error"]
            }
    
    async def adjust_content_tone(
        self, 
        content: str, 
        target_tone: str,
        brand_guidelines: Dict[str, Any] = None
    ) -> str:
        """Adjust content tone to match brand voice"""
        try:
            data = {
                "content": content,
                "target_tone": target_tone,
                "brand_guidelines": brand_guidelines or {}
            }
            
            result = await self._make_request("POST", "/api/v1/adjust-tone", data)
            
            return result.get("adjusted_content", content)
            
        except Exception as e:
            logger.error(f"Error adjusting content tone: {str(e)}")
            return content

class BlogPublisher(BlatamServiceIntegration):
    """Integration with Blog Posts service"""
    
    def __init__(self, base_url: str = "http://localhost:8015", api_key: Optional[str] = None):
        super().__init__("Blog Publisher", base_url, api_key)
    
    async def publish_blog_post(
        self, 
        title: str, 
        content: str,
        metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Publish a blog post
        
        Args:
            title: Blog post title
            content: Blog post content
            metadata: Additional metadata
            
        Returns:
            Dict with publication results
        """
        try:
            data = {
                "title": title,
                "content": content,
                "metadata": metadata or {}
            }
            
            result = await self._make_request("POST", "/api/v1/publish", data)
            
            logger.info(f"Blog post published: {title}")
            return result
            
        except Exception as e:
            logger.error(f"Error publishing blog post: {str(e)}")
            return {
                "success": False,
                "post_id": None,
                "url": None,
                "error": str(e)
            }
    
    async def schedule_blog_post(
        self, 
        title: str, 
        content: str,
        publish_date: datetime,
        metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Schedule a blog post for future publication"""
        try:
            data = {
                "title": title,
                "content": content,
                "publish_date": publish_date.isoformat(),
                "metadata": metadata or {}
            }
            
            result = await self._make_request("POST", "/api/v1/schedule", data)
            
            logger.info(f"Blog post scheduled: {title} for {publish_date}")
            return result
            
        except Exception as e:
            logger.error(f"Error scheduling blog post: {str(e)}")
            return {
                "success": False,
                "scheduled_id": None,
                "error": str(e)
            }

class WorkflowIntegrations:
    """Main class for managing all service integrations"""
    
    def __init__(self):
        self.redundancy_detector = ContentRedundancyDetector()
        self.seo_optimizer = SEOOptimizer()
        self.brand_voice_analyzer = BrandVoiceAnalyzer()
        self.blog_publisher = BlogPublisher()
        self._initialized = False
    
    async def initialize(self):
        """Initialize all integrations"""
        try:
            # Test connections to all services
            services = [
                ("Content Redundancy Detector", self.redundancy_detector),
                ("SEO Optimizer", self.seo_optimizer),
                ("Brand Voice Analyzer", self.brand_voice_analyzer),
                ("Blog Publisher", self.blog_publisher)
            ]
            
            for service_name, service in services:
                try:
                    # Simple health check
                    await service._make_request("GET", "/health")
                    logger.info(f"✅ {service_name} connection successful")
                except Exception as e:
                    logger.warning(f"⚠️ {service_name} connection failed: {str(e)}")
            
            self._initialized = True
            logger.info("Workflow integrations initialized")
            
        except Exception as e:
            logger.error(f"Error initializing integrations: {str(e)}")
            # Continue with partial initialization
    
    async def close(self):
        """Close all service connections"""
        try:
            await self.redundancy_detector.close()
            await self.seo_optimizer.close()
            await self.brand_voice_analyzer.close()
            await self.blog_publisher.close()
            logger.info("All service connections closed")
        except Exception as e:
            logger.error(f"Error closing service connections: {str(e)}")
    
    async def enhance_document(
        self, 
        content: str, 
        title: str,
        target_keywords: List[str] = None,
        brand_guidelines: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Enhance a document using all available services
        
        Args:
            content: Document content
            title: Document title
            target_keywords: SEO target keywords
            brand_guidelines: Brand voice guidelines
            
        Returns:
            Dict with enhancement results
        """
        try:
            results = {
                "original_content": content,
                "enhanced_content": content,
                "title": title,
                "enhancements": {},
                "scores": {},
                "recommendations": []
            }
            
            # Check content uniqueness
            try:
                uniqueness_result = await self.redundancy_detector.check_content_uniqueness(content)
                results["scores"]["uniqueness"] = uniqueness_result.get("uniqueness_score", 0.5)
                results["enhancements"]["uniqueness"] = uniqueness_result
                
                if uniqueness_result.get("uniqueness_score", 1.0) < 0.7:
                    results["recommendations"].extend(
                        uniqueness_result.get("recommendations", [])
                    )
            except Exception as e:
                logger.warning(f"Uniqueness check failed: {str(e)}")
            
            # SEO optimization
            try:
                seo_result = await self.seo_optimizer.optimize_content(
                    content, target_keywords
                )
                results["enhanced_content"] = seo_result.get("optimized_content", content)
                results["scores"]["seo"] = seo_result.get("seo_score", 0.5)
                results["enhancements"]["seo"] = seo_result
                
                # Generate meta description
                meta_description = await self.seo_optimizer.generate_meta_description(
                    results["enhanced_content"]
                )
                results["meta_description"] = meta_description
                
            except Exception as e:
                logger.warning(f"SEO optimization failed: {str(e)}")
            
            # Brand voice analysis
            try:
                brand_result = await self.brand_voice_analyzer.analyze_brand_voice_consistency(
                    results["enhanced_content"], brand_guidelines
                )
                results["scores"]["brand_voice"] = brand_result.get("consistency_score", 0.5)
                results["enhancements"]["brand_voice"] = brand_result
                
                if brand_result.get("consistency_score", 1.0) < 0.7:
                    results["recommendations"].extend(
                        brand_result.get("recommendations", [])
                    )
            except Exception as e:
                logger.warning(f"Brand voice analysis failed: {str(e)}")
            
            # Calculate overall quality score
            scores = results["scores"]
            if scores:
                overall_score = sum(scores.values()) / len(scores)
                results["scores"]["overall"] = overall_score
            
            logger.info(f"Document enhancement completed. Overall score: {results['scores'].get('overall', 0.5)}")
            return results
            
        except Exception as e:
            logger.error(f"Error enhancing document: {str(e)}")
            return {
                "original_content": content,
                "enhanced_content": content,
                "title": title,
                "error": str(e)
            }
    
    async def publish_workflow_chain(
        self, 
        chain_id: str,
        workflow_engine,
        publish_immediately: bool = False,
        target_keywords: List[str] = None,
        brand_guidelines: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Publish an entire workflow chain as blog posts
        
        Args:
            chain_id: Workflow chain ID
            workflow_engine: Workflow engine instance
            publish_immediately: Whether to publish immediately or schedule
            target_keywords: SEO target keywords
            brand_guidelines: Brand voice guidelines
            
        Returns:
            Dict with publication results
        """
        try:
            chain = workflow_engine.get_workflow_chain(chain_id)
            if not chain:
                return {"error": "Workflow chain not found"}
            
            history = workflow_engine.get_chain_history(chain_id)
            publication_results = []
            
            for i, document in enumerate(history):
                try:
                    # Enhance document
                    enhancement_result = await self.enhance_document(
                        content=document.content,
                        title=document.title,
                        target_keywords=target_keywords,
                        brand_guidelines=brand_guidelines
                    )
                    
                    # Prepare metadata
                    metadata = {
                        "workflow_chain_id": chain_id,
                        "document_id": document.id,
                        "sequence_number": i + 1,
                        "total_documents": len(history),
                        "enhancement_scores": enhancement_result.get("scores", {}),
                        "generated_at": document.generated_at.isoformat()
                    }
                    
                    # Publish or schedule
                    if publish_immediately:
                        publish_result = await self.blog_publisher.publish_blog_post(
                            title=enhancement_result["title"],
                            content=enhancement_result["enhanced_content"],
                            metadata=metadata
                        )
                    else:
                        # Schedule with 1-hour intervals
                        publish_date = datetime.now() + timedelta(hours=i + 1)
                        publish_result = await self.blog_publisher.schedule_blog_post(
                            title=enhancement_result["title"],
                            content=enhancement_result["enhanced_content"],
                            publish_date=publish_date,
                            metadata=metadata
                        )
                    
                    publication_results.append({
                        "document_id": document.id,
                        "title": enhancement_result["title"],
                        "success": publish_result.get("success", False),
                        "post_id": publish_result.get("post_id"),
                        "url": publish_result.get("url"),
                        "enhancement_scores": enhancement_result.get("scores", {})
                    })
                    
                except Exception as e:
                    logger.error(f"Error publishing document {document.id}: {str(e)}")
                    publication_results.append({
                        "document_id": document.id,
                        "title": document.title,
                        "success": False,
                        "error": str(e)
                    })
            
            success_count = sum(1 for result in publication_results if result.get("success", False))
            
            return {
                "chain_id": chain_id,
                "total_documents": len(history),
                "successful_publications": success_count,
                "failed_publications": len(history) - success_count,
                "publication_results": publication_results
            }
            
        except Exception as e:
            logger.error(f"Error publishing workflow chain: {str(e)}")
            return {"error": str(e)}

# Global integrations instance
workflow_integrations = WorkflowIntegrations()

# Example usage and testing
if __name__ == "__main__":
    async def test_integrations():
        """Test service integrations"""
        print("🧪 Testing Service Integrations")
        print("=" * 40)
        
        try:
            # Initialize integrations
            await workflow_integrations.initialize()
            
            # Test document enhancement
            test_content = "This is a test document about artificial intelligence and its applications in content creation."
            test_title = "AI in Content Creation"
            
            enhancement_result = await workflow_integrations.enhance_document(
                content=test_content,
                title=test_title,
                target_keywords=["AI", "content creation", "artificial intelligence"]
            )
            
            print(f"✅ Document enhancement completed")
            print(f"Overall score: {enhancement_result.get('scores', {}).get('overall', 0.5)}")
            print(f"Recommendations: {len(enhancement_result.get('recommendations', []))}")
            
            # Cleanup
            await workflow_integrations.close()
            print("✅ Integration test completed")
            
        except Exception as e:
            print(f"❌ Integration test failed: {str(e)}")
    
    # Run test
    asyncio.run(test_integrations())


