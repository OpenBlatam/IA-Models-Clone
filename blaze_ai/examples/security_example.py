"""
Blaze AI Security Module Example

This example demonstrates how to use the Security module for comprehensive
security management including authentication, authorization, user management,
and security auditing.
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, Any

# Import the modular system
from ..modules import (
    ModuleRegistry,
    create_module_registry,
    create_security_module,
    create_cache_module,
    create_monitoring_module
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# SECURITY DEMONSTRATION
# ============================================================================

class SecurityDemo:
    """Demonstrates various security features."""
    
    def __init__(self):
        self.registry = None
        self.security_module = None
        self.cache_module = None
        self.monitoring_module = None
    
    async def setup_system(self):
        """Setup the security system."""
        logger.info("🔐 Setting up Security System...")
        
        # Create module registry
        self.registry = create_module_registry()
        
        # Create security module with enhanced configuration
        self.security_module = create_security_module(
            name="blaze_security",
            enable_password_auth=True,
            enable_api_key_auth=True,
            enable_jwt_auth=True,
            enable_oauth2_auth=False,
            enable_biometric_auth=False,
            enable_multi_factor=False,
            min_password_length=8,
            require_special_chars=True,
            require_numbers=True,
            require_uppercase=True,
            password_expiry_days=90,
            max_login_attempts=3,
            lockout_duration_minutes=15,
            jwt_expiry_hours=24,
            api_key_length=32,
            enable_audit_logging=True,
            enable_rate_limiting=True,
            session_timeout_minutes=30,
            user_storage_path="./security/users",
            audit_log_path="./security/audit",
            backup_enabled=True,
            backup_interval_hours=12,
            priority=1
        )
        
        # Create supporting modules
        self.cache_module = create_cache_module("security_cache", max_size=1000, priority=2)
        self.monitoring_module = create_monitoring_module("security_monitoring", collection_interval=5.0, priority=3)
        
        # Register modules
        await self.registry.register_module(self.security_module)
        await self.registry.register_module(self.cache_module)
        await self.registry.register_module(self.monitoring_module)
        
        logger.info("✅ Security system setup completed")
    
    async def demonstrate_user_management(self):
        """Demonstrate user creation and management."""
        logger.info("👥 Demonstrating User Management...")
        
        # Create test users
        users_to_create = [
            ("admin", "admin@blaze.ai", "Admin123!", ["admin"]),
            ("manager", "manager@blaze.ai", "Manager456!", ["manager"]),
            ("developer", "dev@blaze.ai", "Dev789!", ["developer"]),
            ("analyst", "analyst@blaze.ai", "Analyst012!", ["analyst"]),
            ("guest", "guest@blaze.ai", "Guest345!", ["guest"])
        ]
        
        created_users = []
        for username, email, password, roles in users_to_create:
            user = await self.security_module.create_user(username, email, password, roles)
            if user:
                created_users.append(user)
                logger.info(f"✅ Created user: {username} with roles: {roles}")
            else:
                logger.warning(f"⚠️ Failed to create user: {username}")
        
        logger.info(f"📊 Total users created: {len(created_users)}")
        return created_users
    
    async def demonstrate_authentication(self):
        """Demonstrate various authentication methods."""
        logger.info("🔑 Demonstrating Authentication Methods...")
        
        # Test password authentication
        logger.info("🔐 Testing Password Authentication...")
        
        # Valid credentials
        valid_user = await self.security_module.authenticate_user(
            "PASSWORD",
            {
                "username": "admin",
                "password": "Admin123!",
                "ip_address": "192.168.1.100",
                "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
            }
        )
        
        if valid_user:
            logger.info(f"✅ Password auth successful: {valid_user.username}")
        else:
            logger.warning("⚠️ Password auth failed for admin")
        
        # Invalid credentials
        invalid_user = await self.security_module.authenticate_user(
            "PASSWORD",
            {
                "username": "admin",
                "password": "wrongpassword",
                "ip_address": "192.168.1.100",
                "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
            }
        )
        
        if not invalid_user:
            logger.info("✅ Invalid password correctly rejected")
        
        # Test multiple failed attempts (should trigger lockout)
        logger.info("🚫 Testing Account Lockout...")
        for i in range(4):
            failed_auth = await self.security_module.authenticate_user(
                "PASSWORD",
                {
                    "username": "manager",
                    "password": "wrongpassword",
                    "ip_address": "192.168.1.101",
                    "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
                }
            )
            if not failed_auth:
                logger.info(f"❌ Failed attempt {i+1} for manager")
        
        # Try to authenticate with locked account
        locked_auth = await self.security_module.authenticate_user(
            "PASSWORD",
            {
                "username": "manager",
                "password": "Manager456!",
                "ip_address": "192.168.1.101",
                "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
            }
        )
        
        if not locked_auth:
            logger.info("✅ Locked account correctly rejected")
        
        # Test API key authentication (simulated)
        logger.info("🔑 Testing API Key Authentication...")
        # Note: This would require setting up API keys in the system
        
        # Test JWT authentication (simulated)
        logger.info("🎫 Testing JWT Authentication...")
        # Note: This would require generating valid JWT tokens
    
    async def demonstrate_authorization(self):
        """Demonstrate authorization and permissions."""
        logger.info("🛡️ Demonstrating Authorization System...")
        
        # Get admin user
        admin_user = None
        for user in self.security_module.users.values():
            if user.username == "admin":
                admin_user = user
                break
        
        if not admin_user:
            logger.warning("⚠️ Admin user not found")
            return
        
        # Test various permissions
        test_cases = [
            ("admin", "admin", "read", True),
            ("admin", "admin", "write", True),
            ("admin", "admin", "delete", True),
            ("admin", "users", "create", True),
            ("admin", "system", "shutdown", True),
            ("manager", "users", "read", True),
            ("manager", "users", "create", False),
            ("developer", "code", "read", True),
            ("developer", "code", "deploy", False),
            ("guest", "public", "read", True),
            ("guest", "private", "read", False)
        ]
        
        for username, resource, action, expected in test_cases:
            # Get user
            user = None
            for u in self.security_module.users.values():
                if u.username == username:
                    user = u
                    break
            
            if user:
                has_permission = await self.security_module.check_permission(user, resource, action)
                status = "✅" if has_permission == expected else "❌"
                logger.info(f"{status} {username} -> {resource}:{action} = {has_permission} (expected: {expected})")
            else:
                logger.warning(f"⚠️ User {username} not found")
    
    async def demonstrate_role_management(self):
        """Demonstrate role assignment and management."""
        logger.info("👑 Demonstrating Role Management...")
        
        # Get a user to modify
        user = None
        for u in self.security_module.users.values():
            if u.username == "developer":
                user = u
                break
        
        if not user:
            logger.warning("⚠️ Developer user not found")
            return
        
        logger.info(f"👤 User {user.username} current roles: {user.roles}")
        
        # Assign new role
        success = await self.security_module.assign_role(user.user_id, "tester")
        if success:
            logger.info(f"✅ Assigned 'tester' role to {user.username}")
            logger.info(f"👤 User {user.username} updated roles: {user.roles}")
        else:
            logger.warning(f"⚠️ Failed to assign 'tester' role to {user.username}")
        
        # Revoke role
        success = await self.security_module.revoke_role(user.user_id, "tester")
        if success:
            logger.info(f"✅ Revoked 'tester' role from {user.username}")
            logger.info(f"👤 User {user.username} final roles: {user.roles}")
        else:
            logger.warning(f"⚠️ Failed to revoke 'tester' role from {user.username}")
    
    async def demonstrate_security_auditing(self):
        """Demonstrate security event logging and auditing."""
        logger.info("📋 Demonstrating Security Auditing...")
        
        # Get recent security events
        events = await self.security_module.get_security_events()
        logger.info(f"📊 Total security events: {len(events)}")
        
        # Show recent events
        recent_events = events[:5]  # Last 5 events
        logger.info("📋 Recent Security Events:")
        for event in recent_events:
            timestamp = event.timestamp.strftime("%H:%M:%S")
            logger.info(f"   [{timestamp}] {event.event_type.value} - {event.username or 'Unknown'} - {event.details}")
        
        # Filter events by type
        login_events = await self.security_module.get_security_events(
            event_type="LOGIN_SUCCESS"
        )
        logger.info(f"🔐 Successful logins: {len(login_events)}")
        
        failed_events = await self.security_module.get_security_events(
            event_type="LOGIN_FAILURE"
        )
        logger.info(f"❌ Failed logins: {len(failed_events)}")
        
        # Show events for specific user
        admin_events = await self.security_module.get_security_events(
            user_id="admin_001"
        )
        logger.info(f"👤 Admin user events: {len(admin_events)}")
    
    async def demonstrate_security_metrics(self):
        """Demonstrate security metrics and monitoring."""
        logger.info("📈 Demonstrating Security Metrics...")
        
        # Get security metrics
        metrics = await self.security_module.get_metrics()
        
        logger.info("📊 Security Metrics:")
        logger.info(f"   Total Users: {metrics.total_users}")
        logger.info(f"   Active Users: {metrics.active_users}")
        logger.info(f"   Successful Logins: {metrics.successful_logins}")
        logger.info(f"   Failed Logins: {metrics.failed_logins}")
        logger.info(f"   Security Events: {metrics.security_events}")
        logger.info(f"   Active Sessions: {metrics.active_sessions}")
        
        # Get module health
        health = await self.security_module.health_check()
        logger.info("🏥 Security Module Health:")
        for key, value in health.items():
            logger.info(f"   {key}: {value}")
    
    async def demonstrate_security_features(self):
        """Demonstrate advanced security features."""
        logger.info("🚀 Demonstrating Advanced Security Features...")
        
        # Test session management
        logger.info("⏰ Testing Session Management...")
        initial_sessions = len(self.security_module.active_sessions)
        logger.info(f"   Active sessions: {initial_sessions}")
        
        # Wait for some time to see session cleanup
        logger.info("   Waiting 10 seconds for session cleanup...")
        await asyncio.sleep(10)
        
        current_sessions = len(self.security_module.active_sessions)
        logger.info(f"   Active sessions after cleanup: {current_sessions}")
        
        # Test user updates
        logger.info("✏️ Testing User Updates...")
        user = None
        for u in self.security_module.users.values():
            if u.username == "analyst":
                user = u
                break
        
        if user:
            # Update user metadata
            success = await self.security_module.update_user(
                user.user_id,
                {"metadata": {"department": "Data Science", "last_updated": datetime.now().isoformat()}}
            )
            if success:
                logger.info(f"✅ Updated user {user.username} metadata")
            else:
                logger.warning(f"⚠️ Failed to update user {user.username}")
        
        # Test backup functionality
        logger.info("💾 Testing Backup Functionality...")
        # The backup runs automatically in the background
        logger.info("   Backup system is running in background")
    
    async def run_demo(self):
        """Run the complete security demonstration."""
        try:
            logger.info("🚀 Starting Blaze AI Security Module Demonstration")
            
            # Setup system
            await self.setup_system()
            
            # Wait for modules to be ready
            await asyncio.sleep(2)
            
            # Run demonstrations
            await self.demonstrate_user_management()
            await asyncio.sleep(1)
            
            await self.demonstrate_authentication()
            await asyncio.sleep(1)
            
            await self.demonstrate_authorization()
            await asyncio.sleep(1)
            
            await self.demonstrate_role_management()
            await asyncio.sleep(1)
            
            await self.demonstrate_security_auditing()
            await asyncio.sleep(1)
            
            await self.demonstrate_security_metrics()
            await asyncio.sleep(1)
            
            await self.demonstrate_security_features()
            
            logger.info("🎉 Security demonstration completed successfully!")
            logger.info("🔐 Security system is now running and monitoring all activities")
            
            # Keep running for a while to see background processes
            logger.info("⏸️ System will continue running for 30 seconds to demonstrate background processes...")
            await asyncio.sleep(30)
            
        except Exception as e:
            logger.error(f"❌ Security demonstration failed: {e}")
            raise
        
        finally:
            # Shutdown system
            if self.registry:
                logger.info("🔄 Shutting down security system...")
                await self.registry.shutdown()
                logger.info("✅ Security system shutdown completed")

# ============================================================================
# MAIN EXAMPLE
# ============================================================================

async def main():
    """Main example function."""
    demo = SecurityDemo()
    await demo.run_demo()

# ============================================================================
# RUN EXAMPLE
# ============================================================================

if __name__ == "__main__":
    asyncio.run(main())
