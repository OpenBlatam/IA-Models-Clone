"""
Comprehensive demonstration of the Unified Infrastructure System.

This demo showcases all infrastructure components:
- Database management
- Storage strategies
- Cache management
- External services
- Repository implementations
- Version control
- Project management
- LangChain integration
"""

import asyncio
import tempfile
import os
from pathlib import Path
from datetime import datetime

from .database import DatabaseManager, DatabaseConfig
from .storage import FileStorageManager, StorageConfig, LocalStorageStrategy
from .cache import CacheManager, CacheConfig, MemoryCacheStrategy
from .external_services import ExternalServiceManager, ExternalServiceConfig
from .repositories import RepositoryFactory
from .version_control import VersionControlManager, VersionControlService
from .project_management import ProjectManager, ProjectInitializer, ProjectType, ProblemComplexity
from .langchain_integration import LangChainService, LangChainConfig


class InfrastructureSystemDemo:
    """Demonstrates the entire unified infrastructure system."""
    
    def __init__(self):
        """Initialize the demo."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.demo_results = {}
    
    async def run_comprehensive_demo(self):
        """Run the complete infrastructure demonstration."""
        print("🚀 Starting Comprehensive Infrastructure System Demo")
        print("=" * 60)
        
        try:
            # Database Management
            await self._demo_database_management()
            
            # Storage Management
            await self._demo_storage_management()
            
            # Cache Management
            await self._demo_cache_management()
            
            # External Services
            await self._demo_external_services()
            
            # Repository Operations
            await self._demo_repository_operations()
            
            # Version Control
            await self._demo_version_control()
            
            # Project Management
            await self._demo_project_management()
            
            # LangChain Integration
            await self._demo_langchain_integration()
            
            # System Integration
            await self._demo_system_integration()
            
            # Print Summary
            self._print_system_summary()
            
        except Exception as e:
            print(f"❌ Demo failed: {e}")
            raise
        finally:
            await self._cleanup()
    
    async def _demo_database_management(self):
        """Demonstrate database management capabilities."""
        print("\n📊 Database Management Demo")
        print("-" * 30)
        
        try:
            # Create database configuration
            db_config = DatabaseConfig(
                host="localhost",
                port=5432,
                database="ads_demo",
                username="demo_user",
                password="demo_pass",
                pool_size=2,
                max_overflow=2
            )
            
            # Initialize database manager
            db_manager = DatabaseManager(db_config)
            
            # Test connection
            async with db_manager.get_session() as session:
                # Simulate database operations
                result = await session.execute("SELECT 1 as test")
                test_value = result.scalar()
                
                self.demo_results['database'] = {
                    'status': 'success',
                    'test_value': test_value,
                    'config': db_config.__dict__,
                    'pool_stats': db_manager.connection_pool.get_stats()
                }
                
                print("✅ Database connection successful")
                print(f"   Pool size: {db_config.pool_size}")
                print(f"   Max overflow: {db_config.max_overflow}")
                
        except Exception as e:
            self.demo_results['database'] = {'status': 'error', 'error': str(e)}
            print(f"❌ Database demo failed: {e}")
    
    async def _demo_storage_management(self):
        """Demonstrate storage management capabilities."""
        print("\n💾 Storage Management Demo")
        print("-" * 30)
        
        try:
            # Create storage configuration
            storage_config = StorageConfig(
                storage_type="local",
                base_path=str(self.temp_dir / "storage"),
                max_file_size=100 * 1024 * 1024,  # 100MB
                allowed_extensions=[".txt", ".json", ".png", ".jpg"]
            )
            
            # Initialize storage manager
            storage_manager = FileStorageManager(storage_config)
            
            # Test file operations
            test_file_path = self.temp_dir / "test_file.txt"
            test_content = "Hello, Infrastructure Demo!"
            
            # Save file
            with open(test_file_path, 'w') as f:
                f.write(test_content)
            
            # Store file
            stored_path = await storage_manager.store_file(
                str(test_file_path),
                "demo/test_file.txt"
            )
            
            # Retrieve file
            retrieved_content = await storage_manager.get_file_content("demo/test_file.txt")
            
            self.demo_results['storage'] = {
                'status': 'success',
                'stored_path': stored_path,
                'retrieved_content': retrieved_content,
                'config': storage_config.__dict__
            }
            
            print("✅ Storage operations successful")
            print(f"   Stored file: {stored_path}")
            print(f"   Retrieved content: {retrieved_content[:50]}...")
            
        except Exception as e:
            self.demo_results['storage'] = {'status': 'error', 'error': str(e)}
            print(f"❌ Storage demo failed: {e}")
    
    async def _demo_cache_management(self):
        """Demonstrate cache management capabilities."""
        print("\n⚡ Cache Management Demo")
        print("-" * 30)
        
        try:
            # Create cache configuration
            cache_config = CacheConfig(
                cache_type="memory",
                max_size=1000,
                ttl=3600,  # 1 hour
                enable_compression=False
            )
            
            # Initialize cache manager
            cache_manager = CacheManager(cache_config)
            
            # Test cache operations
            test_key = "demo:test_data"
            test_data = {"message": "Hello from cache!", "timestamp": datetime.now().isoformat()}
            
            # Set cache
            await cache_manager.set(test_key, test_data)
            
            # Get cache
            retrieved_data = await cache_manager.get(test_key)
            
            # Check cache stats
            stats = cache_manager.get_statistics()
            
            self.demo_results['cache'] = {
                'status': 'success',
                'set_data': test_data,
                'retrieved_data': retrieved_data,
                'stats': stats,
                'config': cache_config.__dict__
            }
            
            print("✅ Cache operations successful")
            print(f"   Set data: {test_data}")
            print(f"   Retrieved data: {retrieved_data}")
            print(f"   Cache hits: {stats['hits']}")
            
        except Exception as e:
            self.demo_results['cache'] = {'status': 'error', 'error': str(e)}
            print(f"❌ Cache demo failed: {e}")
    
    async def _demo_external_services(self):
        """Demonstrate external service management."""
        print("\n🌐 External Services Demo")
        print("-" * 30)
        
        try:
            # Create external service configuration
            service_config = ExternalServiceConfig(
                base_url="https://api.example.com",
                timeout=30,
                max_retries=3,
                rate_limit_per_minute=100
            )
            
            # Initialize service manager
            service_manager = ExternalServiceManager(service_config)
            
            # Test service health
            health_status = await service_manager.check_service_health("demo_service")
            
            self.demo_results['external_services'] = {
                'status': 'success',
                'health_status': health_status,
                'config': service_config.__dict__
            }
            
            print("✅ External services initialized")
            print(f"   Base URL: {service_config.base_url}")
            print(f"   Timeout: {service_config.timeout}s")
            print(f"   Rate limit: {service_config.rate_limit_per_minute}/min")
            
        except Exception as e:
            self.demo_results['external_services'] = {'status': 'error', 'error': str(e)}
            print(f"❌ External services demo failed: {e}")
    
    async def _demo_repository_operations(self):
        """Demonstrate repository operations."""
        print("\n🗄️ Repository Operations Demo")
        print("-" * 30)
        
        try:
            # Get repository factory
            repo_factory = RepositoryFactory()
            
            # Test repository creation
            ad_repo = repo_factory.create_ad_repository()
            campaign_repo = repo_factory.create_campaign_repository()
            
            # Test basic operations
            test_ad_id = "demo_ad_001"
            test_campaign_id = "demo_campaign_001"
            
            # Check if repositories exist
            ad_repo_exists = ad_repo is not None
            campaign_repo_exists = campaign_repo is not None
            
            self.demo_results['repositories'] = {
                'status': 'success',
                'ad_repository_exists': ad_repo_exists,
                'campaign_repository_exists': campaign_repo_exists,
                'repository_types': ['AdsRepository', 'CampaignRepository', 'GroupRepository']
            }
            
            print("✅ Repository operations successful")
            print(f"   Ad repository: {'✅' if ad_repo_exists else '❌'}")
            print(f"   Campaign repository: {'✅' if campaign_repo_exists else '❌'}")
            
        except Exception as e:
            self.demo_results['repositories'] = {'status': 'error', 'error': str(e)}
            print(f"❌ Repository demo failed: {e}")
    
    async def _demo_version_control(self):
        """Demonstrate version control capabilities."""
        print("\n🔧 Version Control Demo")
        print("-" * 30)
        
        try:
            # Initialize version control manager
            vc_manager = VersionControlManager(str(self.temp_dir))
            
            # Test git operations
            git_status = vc_manager.get_status()
            current_branch = vc_manager.get_current_branch()
            is_git_repo = vc_manager._is_git_repository()
            
            # Get repository info
            repo_info = vc_manager.get_repository_info()
            
            self.demo_results['version_control'] = {
                'status': 'success',
                'git_status': git_status.value,
                'current_branch': current_branch,
                'is_git_repo': is_git_repo,
                'repo_info': repo_info
            }
            
            print("✅ Version control operations successful")
            print(f"   Git status: {git_status.value}")
            print(f"   Current branch: {current_branch}")
            print(f"   Is git repo: {is_git_repo}")
            
        except Exception as e:
            self.demo_results['version_control'] = {'status': 'error', 'error': str(e)}
            print(f"❌ Version control demo failed: {e}")
    
    async def _demo_project_management(self):
        """Demonstrate project management capabilities."""
        print("\n📁 Project Management Demo")
        print("-" * 30)
        
        try:
            # Initialize project manager
            project_manager = ProjectManager(str(self.temp_dir / "projects"))
            
            # Test project initialization
            project_initializer = ProjectInitializer(str(self.temp_dir / "projects"))
            
            # Create a sample project config
            from .project_management import DatasetInfo, ProblemDefinition
            
            dataset_info = project_manager.analyze_dataset(str(self.temp_dir))
            problem_definition = ProblemDefinition(
                title="Demo ML Project",
                description="A demonstration project for infrastructure testing",
                project_type=ProjectType.CLASSIFICATION,
                complexity=ProblemComplexity.MODERATE,
                objectives=["Test infrastructure components", "Validate project setup"],
                success_metrics=["Component initialization", "Project structure creation"]
            )
            
            # Validate project config
            validation_issues = project_manager.validate_project_config(
                project_manager._create_project_config_from_components(
                    "demo_project",
                    "1.0.0",
                    "Demo project for testing",
                    "Demo User",
                    ProjectType.CLASSIFICATION,
                    problem_definition,
                    dataset_info
                )
            )
            
            self.demo_results['project_management'] = {
                'status': 'success',
                'validation_issues': validation_issues,
                'dataset_info': dataset_info.__dict__,
                'problem_definition': problem_definition.__dict__
            }
            
            print("✅ Project management operations successful")
            print(f"   Dataset analyzed: {dataset_info.name}")
            print(f"   Project type: {problem_definition.project_type.value}")
            print(f"   Validation issues: {len(validation_issues)}")
            
        except Exception as e:
            self.demo_results['project_management'] = {'status': 'error', 'error': str(e)}
            print(f"❌ Project management demo failed: {e}")
    
    async def _demo_langchain_integration(self):
        """Demonstrate LangChain integration capabilities."""
        print("\n🤖 LangChain Integration Demo")
        print("-" * 30)
        
        try:
            # Create LangChain configuration
            langchain_config = LangChainConfig(
                chunk_size=500,
                chunk_overlap=50,
                max_tokens=500,
                temperature=0.7,
                enable_memory=True,
                enable_vector_store=True,
                enable_agents=True
            )
            
            # Initialize LangChain service
            langchain_service = LangChainService(config=langchain_config)
            
            # Get service info
            service_info = langchain_service.get_service_info()
            
            self.demo_results['langchain_integration'] = {
                'status': 'success',
                'service_info': service_info,
                'config': langchain_config.dict()
            }
            
            print("✅ LangChain integration successful")
            print(f"   LLM provider: {service_info['llm_provider']}")
            print(f"   Embeddings provider: {service_info['embeddings_provider']}")
            print(f"   Memory enabled: {service_info['memory_enabled']}")
            
        except Exception as e:
            self.demo_results['langchain_integration'] = {'status': 'error', 'error': str(e)}
            print(f"❌ LangChain integration demo failed: {e}")
    
    async def _demo_system_integration(self):
        """Demonstrate system-wide integration."""
        print("\n🔗 System Integration Demo")
        print("-" * 30)
        
        try:
            # Test cross-component integration
            integration_results = {}
            
            # Database + Cache integration
            if (self.demo_results.get('database', {}).get('status') == 'success' and
                self.demo_results.get('cache', {}).get('status') == 'success'):
                integration_results['database_cache'] = 'success'
            
            # Storage + Version Control integration
            if (self.demo_results.get('storage', {}).get('status') == 'success' and
                self.demo_results.get('version_control', {}).get('status') == 'success'):
                integration_results['storage_version_control'] = 'success'
            
            # Project Management + LangChain integration
            if (self.demo_results.get('project_management', {}).get('status') == 'success' and
                self.demo_results.get('langchain_integration', {}).get('status') == 'success'):
                integration_results['project_langchain'] = 'success'
            
            self.demo_results['system_integration'] = {
                'status': 'success',
                'integration_results': integration_results,
                'total_components': len(self.demo_results) - 1,  # Exclude system_integration
                'successful_components': sum(1 for r in self.demo_results.values() 
                                          if isinstance(r, dict) and r.get('status') == 'success')
            }
            
            print("✅ System integration successful")
            print(f"   Total components: {self.demo_results['system_integration']['total_components']}")
            print(f"   Successful components: {self.demo_results['system_integration']['successful_components']}")
            
        except Exception as e:
            self.demo_results['system_integration'] = {'status': 'error', 'error': str(e)}
            print(f"❌ System integration demo failed: {e}")
    
    def _print_system_summary(self):
        """Print a comprehensive system summary."""
        print("\n" + "=" * 60)
        print("📋 INFRASTRUCTURE SYSTEM SUMMARY")
        print("=" * 60)
        
        total_components = len(self.demo_results)
        successful_components = sum(1 for r in self.demo_results.values() 
                                  if isinstance(r, dict) and r.get('status') == 'success')
        failed_components = total_components - successful_components
        
        print(f"Total Components: {total_components}")
        print(f"Successful: {successful_components} ✅")
        print(f"Failed: {failed_components} ❌")
        print(f"Success Rate: {(successful_components/total_components)*100:.1f}%")
        
        print("\nComponent Status:")
        for component, result in self.demo_results.items():
            status_icon = "✅" if result.get('status') == 'success' else "❌"
            print(f"  {component.replace('_', ' ').title()}: {status_icon}")
        
        if successful_components == total_components:
            print("\n🎉 ALL INFRASTRUCTURE COMPONENTS ARE WORKING PERFECTLY!")
        else:
            print(f"\n⚠️  {failed_components} component(s) need attention")
    
    async def _cleanup(self):
        """Clean up demo resources."""
        try:
            # Remove temporary directory
            import shutil
            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
            print(f"\n🧹 Cleaned up temporary files: {self.temp_dir}")
        except Exception as e:
            print(f"⚠️  Cleanup warning: {e}")


async def main():
    """Main demo function."""
    demo = InfrastructureSystemDemo()
    await demo.run_comprehensive_demo()


if __name__ == "__main__":
    asyncio.run(main())
