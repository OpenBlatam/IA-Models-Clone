"""
🚀 ADS System - Main Entry Point

Main entry point for the refactored advertising system that demonstrates
the consolidated architecture and Clean Architecture principles.
"""

import asyncio
import logging
from typing import Dict, Any
from uuid import uuid4

from domain.entities import Ad, AdCampaign, AdGroup
from domain.value_objects import AdStatus, AdType, Platform, Budget, TargetingCriteria, AdSchedule
from datetime import datetime, timezone, timedelta
from optimization.factory import get_optimization_factory
from optimization.base_optimizer import OptimizationContext, OptimizationStrategy, OptimizationLevel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AdsSystemDemo:
    """Demo class to showcase the refactored ADS system."""
    
    def __init__(self):
        self.optimization_factory = get_optimization_factory()
        self.ads: Dict[str, Ad] = {}
        self.campaigns: Dict[str, AdCampaign] = {}
        self.groups: Dict[str, AdGroup] = {}
        
        logger.info("ADS System Demo initialized")
    
    async def create_sample_campaign(self) -> AdCampaign:
        """Create a sample advertising campaign."""
        logger.info("Creating sample campaign")
        
        # Create targeting criteria
        targeting = TargetingCriteria(
            age_min=25,
            age_max=45,
            genders=["male", "female"],
            locations=["United States", "Canada"],
            interests=["technology", "business", "entrepreneurship"]
        )
        
        # Create budget
        budget = Budget(
            amount=1000.0,
            currency="USD",
            daily_limit=100.0,
            lifetime_limit=1000.0
        )
        
        # Create campaign
        campaign = AdCampaign(
            name="Tech Startup Launch Campaign",
            description="Campaign to promote our new tech startup",
            objective="Brand Awareness and Lead Generation",
            platform=Platform.FACEBOOK,
            targeting=targeting,
            budget=budget
        )
        
        self.campaigns[str(campaign.id)] = campaign
        logger.info(f"Created campaign: {campaign.name} (ID: {campaign.id})")
        
        return campaign
    
    async def create_sample_ad(self, campaign_id: str) -> Ad:
        """Create a sample advertisement."""
        logger.info("Creating sample advertisement")
        
        # Create targeting criteria
        targeting = TargetingCriteria(
            age_min=25,
            age_max=45,
            genders=["male", "female"],
            locations=["United States", "Canada"],
            interests=["technology", "business", "entrepreneurship"]
        )
        
        # Create budget
        budget = Budget(
            amount=100.0,
            currency="USD",
            daily_limit=20.0,
            lifetime_limit=100.0
        )
        
        # Create schedule (active from now for 30 days, 24/7 for testing)
        schedule = AdSchedule(
            start_date=datetime.now(timezone.utc),
            end_date=datetime.now(timezone.utc) + timedelta(days=30),
            # No time restrictions for testing
            timezone="UTC"
        )
        
        # Create ad
        ad = Ad(
            name="Tech Startup Launch Ad",
            description="Promote our innovative tech solution",
            ad_type=AdType.IMAGE,
            platform=Platform.FACEBOOK,
            headline="Transform Your Business with AI",
            body_text="Discover how our AI-powered platform can revolutionize your business operations and drive growth.",
            image_url="https://example.com/tech-startup-image.jpg",
            call_to_action="Learn More",
            targeting=targeting,
            budget=budget,
            schedule=schedule,
            campaign_id=campaign_id
        )
        
        self.ads[str(ad.id)] = ad
        logger.info(f"Created ad: {ad.name} (ID: {ad.id})")
        
        return ad
    
    async def demonstrate_optimization(self, ad_id: str):
        """Demonstrate the optimization system."""
        logger.info(f"Demonstrating optimization for ad: {ad_id}")
        
        # Create optimization context
        context = OptimizationContext(
            target_entity="ad",
            entity_id=ad_id,
            optimization_type=OptimizationStrategy.PERFORMANCE,
            level=OptimizationLevel.STANDARD,
            parameters={
                'optimization_focus': 'performance',
                'target_metrics': ['response_time', 'throughput', 'cpu_usage']
            }
        )
        
        # Get optimal optimizer
        optimal_optimizer_type = self.optimization_factory.get_optimal_optimizer(context)
        logger.info(f"Optimal optimizer type: {optimal_optimizer_type}")
        
        # Execute optimization
        if optimal_optimizer_type:
            result = await self.optimization_factory.execute_optimization(context, optimal_optimizer_type)
            if result:
                logger.info(f"Optimization completed successfully: {result.improvement_percentage:.2f}% improvement")
                logger.info(f"Optimization details: {result.details}")
            else:
                logger.warning("Optimization failed")
        else:
            logger.warning("No suitable optimizer found")
    
    async def demonstrate_campaign_management(self):
        """Demonstrate campaign management functionality."""
        logger.info("Demonstrating campaign management")
        
        # Create campaign
        campaign = await self.create_sample_campaign()
        
        # Create ad in campaign
        ad = await self.create_sample_ad(str(campaign.id))
        
        # Add ad to campaign
        campaign.add_ad(ad)
        
        # Demonstrate ad lifecycle
        logger.info(f"Ad status: {ad.status.value}")
        
        # Approve ad
        ad.status = AdStatus.PENDING_REVIEW
        ad.approve()
        logger.info(f"Ad approved. New status: {ad.status.value}")
        
        # Debug schedule information
        logger.info(f"Ad schedule: {ad.schedule}")
        if ad.schedule:
            logger.info(f"Schedule start: {ad.schedule.start_date}")
            logger.info(f"Schedule end: {ad.schedule.end_date}")
            logger.info(f"Schedule active: {ad.schedule.is_active()}")
        
        # Activate ad
        ad.activate()
        logger.info(f"Ad activated. New status: {ad.status.value}")
        
        # Demonstrate optimization
        await self.demonstrate_optimization(str(ad.id))
        
        return campaign, ad
    
    async def show_system_statistics(self):
        """Show system statistics."""
        logger.info("=== ADS System Statistics ===")
        
        # Show optimization statistics
        opt_stats = self.optimization_factory.get_optimization_statistics()
        logger.info(f"Total optimizers: {opt_stats['total_optimizers']}")
        logger.info(f"Active instances: {opt_stats['active_instances']}")
        logger.info(f"Optimizer types: {opt_stats['optimizer_types']}")
        
        # Show entity counts
        logger.info(f"Total campaigns: {len(self.campaigns)}")
        logger.info(f"Total ads: {len(self.ads)}")
        
        # Show available optimizers
        available_optimizers = self.optimization_factory.list_available_optimizers()
        logger.info("Available optimizers:")
        for opt in available_optimizers:
            logger.info(f"  - {opt['type']}: {opt['config']['description']}")
            logger.info(f"    Capabilities: {', '.join(opt['capabilities'])}")
    
    async def run_demo(self):
        """Run the complete demo."""
        logger.info("🚀 Starting ADS System Demo")
        
        try:
            # Demonstrate campaign management
            campaign, ad = await self.demonstrate_campaign_management()
            
            # Show system statistics
            await self.show_system_statistics()
            
            # Demonstrate different optimization levels
            logger.info("\n=== Demonstrating Different Optimization Levels ===")
            
            for level in OptimizationLevel:
                logger.info(f"\n--- {level.value.upper()} OPTIMIZATION ---")
                
                context = OptimizationContext(
                    target_entity="ad",
                    entity_id=str(ad.id),
                    optimization_type=OptimizationStrategy.PERFORMANCE,
                    level=level,
                    parameters={'level': level.value}
                )
                
                # Get optimal optimizer
                optimal_optimizer = self.optimization_factory.get_optimal_optimizer(context)
                if optimal_optimizer:
                    logger.info(f"Optimal optimizer: {optimal_optimizer}")
                    
                    # Get optimizer info
                    optimizer_info = self.optimization_factory.get_optimizer_info(optimal_optimizer)
                    if optimizer_info:
                        logger.info(f"Description: {optimizer_info['config']['description']}")
                        logger.info(f"Capabilities: {', '.join(optimizer_info['capabilities'])}")
                else:
                    logger.warning(f"No suitable optimizer found for {level.value} level")
            
            logger.info("\n✅ ADS System Demo completed successfully!")
            
        except Exception as e:
            logger.error(f"Demo failed: {e}")
            raise


async def main():
    """Main function to run the ADS system demo."""
    demo = AdsSystemDemo()
    await demo.run_demo()


if __name__ == "__main__":
    # Run the demo
    asyncio.run(main())
