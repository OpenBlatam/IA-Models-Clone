from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
TIMEOUT_SECONDS = 60

import argparse
import asyncio
import json
import sys
import os
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
import logging
    from security_implementation import (
    from cybersecurity.security_implementation import (
        import time
from typing import Any, List, Dict, Optional
#!/usr/bin/env python3
"""
CLI Interface for Cybersecurity Toolkit
Implements RORO (Receive an Object, Return an Object) pattern for tool control.
"""


# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
        SecurityConfig, SecureNetworkScanner, SecurityError, 
        create_secure_config, RateLimiter, AdaptiveRateLimiter,
        NetworkScanRateLimiter, SecureSecretManager
    )
except ImportError:
    # Fallback for different directory structure
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        SecurityConfig, SecureNetworkScanner, SecurityError,
        create_secure_config, RateLimiter, AdaptiveRateLimiter,
        NetworkScanRateLimiter, SecureSecretManager
    )

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class CLIRequest:
    """RORO pattern request object for CLI operations."""
    command: str
    target: Optional[str] = None
    scan_type: Optional[str] = None
    user: Optional[str] = None
    session_id: Optional[str] = None
    config: Optional[Dict[str, Any]] = None
    options: Optional[Dict[str, Any]] = None
    output_format: str = "json"
    verbose: bool = False

@dataclass
class CLIResponse:
    """RORO pattern response object for CLI operations."""
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    error_code: Optional[str] = None
    message: Optional[str] = None
    timestamp: Optional[str] = None
    execution_time: Optional[float] = None

class CybersecurityCLI:
    """CLI interface for cybersecurity toolkit using RORO pattern."""
    
    def __init__(self) -> Any:
        self.config = create_secure_config()
        self.scanner = SecureNetworkScanner(self.config)
        self.rate_limiter = RateLimiter()
        self.adaptive_limiter = AdaptiveRateLimiter()
        self.scan_limiter = NetworkScanRateLimiter()
        self.secret_manager = SecureSecretManager()
        
        # Initialize authorization for demo
        self.scanner.auth_checker.add_authorized_target("127.0.0.1", "cli_user", 
                                                       int(time.time()) + 3600, ["scan"])
        self.scanner.auth_checker.record_consent("cli_user", True, "network_scanning")
    
    async def execute_command(self, request: CLIRequest) -> CLIResponse:
        """Execute CLI command using RORO pattern."""
        start_time = time.time()
        
        try:
            # Validate request
            if not request.command:
                return CLIResponse(
                    success=False,
                    error="No command specified",
                    error_code="MISSING_COMMAND"
                )
            
            # Execute command based on type
            if request.command == "scan":
                result = await self._handle_scan_command(request)
            elif request.command == "rate-limit":
                result = await self._handle_rate_limit_command(request)
            elif request.command == "secrets":
                result = await self._handle_secrets_command(request)
            elif request.command == "config":
                result = await self._handle_config_command(request)
            elif request.command == "help":
                result = await self._handle_help_command(request)
            else:
                result = CLIResponse(
                    success=False,
                    error=f"Unknown command: {request.command}",
                    error_code="UNKNOWN_COMMAND"
                )
            
            # Add execution time
            result.execution_time = time.time() - start_time
            result.timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            
            return result
            
        except Exception as e:
            logger.error(f"CLI execution error: {e}")
            return CLIResponse(
                success=False,
                error=str(e),
                error_code="EXECUTION_ERROR",
                execution_time=time.time() - start_time,
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
            )
    
    async def _handle_scan_command(self, request: CLIRequest) -> CLIResponse:
        """Handle scan command."""
        if not request.target:
            return CLIResponse(
                success=False,
                error="Target is required for scan command",
                error_code="MISSING_TARGET"
            )
        
        scan_type = request.scan_type or "port_scan"
        user = request.user or "cli_user"
        session_id = request.session_id or "cli_session"
        
        try:
            result = await self.scanner.secure_scan(
                target=request.target,
                user=user,
                session_id=session_id,
                scan_type=scan_type
            )
            
            return CLIResponse(
                success=result.get('success', False),
                data=result,
                message=f"Scan completed for {request.target}"
            )
            
        except SecurityError as e:
            return CLIResponse(
                success=False,
                error=e.message,
                error_code=e.code
            )
    
    async def _handle_rate_limit_command(self, request: CLIRequest) -> CLIResponse:
        """Handle rate limit command."""
        if not request.target:
            return CLIResponse(
                success=False,
                error="Target is required for rate-limit command",
                error_code="MISSING_TARGET"
            )
        
        try:
            # Get rate limit statistics
            basic_stats = self.rate_limiter.get_rate_limit_stats(request.target)
            adaptive_stats = self.adaptive_limiter.get_adaptive_stats(request.target)
            scan_stats = self.scan_limiter.get_scan_stats()
            
            data = {
                "target": request.target,
                "basic_rate_limiter": basic_stats,
                "adaptive_rate_limiter": adaptive_stats,
                "scan_rate_limiter": scan_stats
            }
            
            return CLIResponse(
                success=True,
                data=data,
                message=f"Rate limit statistics for {request.target}"
            )
            
        except Exception as e:
            return CLIResponse(
                success=False,
                error=str(e),
                error_code="RATE_LIMIT_ERROR"
            )
    
    async def _handle_secrets_command(self, request: CLIRequest) -> CLIResponse:
        """Handle secrets command."""
        if not request.target:
            return CLIResponse(
                success=False,
                error="Secret name is required for secrets command",
                error_code="MISSING_SECRET_NAME"
            )
        
        try:
            source = request.options.get('source', 'env') if request.options else 'env'
            secret = self.secret_manager.get_secret(request.target, source, required=False)
            
            if secret:
                # Mask the secret for display
                masked_secret = secret[:4] + '*' * (len(secret) - 8) + secret[-4:] if len(secret) > 8 else '***'
                
                # Validate secret strength
                validation = self.secret_manager.validate_secret_strength(secret, request.target)
                
                data = {
                    "secret_name": request.target,
                    "source": source,
                    "masked_value": masked_secret,
                    "length": len(secret),
                    "strength_validation": validation
                }
                
                return CLIResponse(
                    success=True,
                    data=data,
                    message=f"Secret '{request.target}' loaded successfully"
                )
            else:
                return CLIResponse(
                    success=False,
                    error=f"Secret '{request.target}' not found in {source}",
                    error_code="SECRET_NOT_FOUND"
                )
                
        except SecurityError as e:
            return CLIResponse(
                success=False,
                error=e.message,
                error_code=e.code
            )
    
    async def _handle_config_command(self, request: CLIRequest) -> CLIResponse:
        """Handle config command."""
        try:
            # Get configuration information
            config_data = {
                "api_key_configured": bool(self.config.api_key),
                "encryption_key_configured": bool(self.config.encryption_key),
                "max_retries": self.config.max_retries,
                "timeout": self.config.timeout,
                "rate_limit": self.config.rate_limit,
                "session_timeout": self.config.session_timeout,
                "tls_version": self.config.tls_version,
                "verify_ssl": self.config.verify_ssl
            }
            
            # Validate configuration
            try:
                self.config.validate()
                config_data["validation"] = "PASSED"
            except SecurityError as e:
                config_data["validation"] = f"FAILED: {e.message}"
            
            return CLIResponse(
                success=True,
                data=config_data,
                message="Configuration information retrieved"
            )
            
        except Exception as e:
            return CLIResponse(
                success=False,
                error=str(e),
                error_code="CONFIG_ERROR"
            )
    
    async def _handle_help_command(self, request: CLIRequest) -> CLIResponse:
        """Handle help command."""
        help_data = {
            "commands": {
                "scan": {
                    "description": "Perform network scan",
                    "usage": "scan --target <target> [--scan-type <type>] [--user <user>]",
                    "options": {
                        "--target": "Target to scan (required)",
                        "--scan-type": "Type of scan (port_scan, vulnerability_scan, web_scan, network_discovery)",
                        "--user": "User performing scan",
                        "--session-id": "Session identifier"
                    }
                },
                "rate-limit": {
                    "description": "Get rate limit statistics",
                    "usage": "rate-limit --target <target>",
                    "options": {
                        "--target": "Target to check rate limits for (required)"
                    }
                },
                "secrets": {
                    "description": "Manage secrets",
                    "usage": "secrets --target <secret_name> [--source <source>]",
                    "options": {
                        "--target": "Secret name to retrieve (required)",
                        "--source": "Secret source (env, file, vault, aws, azure, gcp)"
                    }
                },
                "config": {
                    "description": "Show configuration information",
                    "usage": "config",
                    "options": {}
                },
                "help": {
                    "description": "Show this help message",
                    "usage": "help [--command <command>]",
                    "options": {
                        "--command": "Specific command to get help for"
                    }
                }
            },
            "examples": [
                "cybersecurity-cli scan --target 127.0.0.1 --scan-type port_scan",
                "cybersecurity-cli rate-limit --target example.com",
                "cybersecurity-cli secrets --target api_key --source env",
                "cybersecurity-cli config",
                "cybersecurity-cli help --command scan"
            ]
        }
        
        return CLIResponse(
            success=True,
            data=help_data,
            message="Help information retrieved"
        )

def format_output(response: CLIResponse, output_format: str = "json") -> str:
    """Format CLI response for output."""
    if output_format == "json":
        return json.dumps(asdict(response), indent=2)
    elif output_format == "text":
        if response.success:
            lines = [f"✅ {response.message}"]
            if response.data:
                lines.append("📊 Data:")
                lines.append(json.dumps(response.data, indent=2))
        else:
            lines = [f"❌ Error: {response.error}"]
            if response.error_code:
                lines.append(f"   Code: {response.error_code}")
        
        if response.execution_time:
            lines.append(f"⏱️  Execution time: {response.execution_time:.3f}s")
        
        return "\n".join(lines)
    else:
        return str(response)

async def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Cybersecurity Toolkit CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  cybersecurity-cli scan --target 127.0.0.1 --scan-type port_scan
  cybersecurity-cli rate-limit --target example.com
  cybersecurity-cli secrets --target api_key --source env
  cybersecurity-cli config
  cybersecurity-cli help
        """
    )
    
    parser.add_argument("command", help="Command to execute")
    parser.add_argument("--target", help="Target for scan or secret name")
    parser.add_argument("--scan-type", choices=["port_scan", "vulnerability_scan", "web_scan", "network_discovery"],
                       default="port_scan", help="Type of scan to perform")
    parser.add_argument("--user", help="User performing the operation")
    parser.add_argument("--session-id", help="Session identifier")
    parser.add_argument("--source", help="Source for secrets (env, file, vault, aws, azure, gcp)")
    parser.add_argument("--output-format", choices=["json", "text"], default="json",
                       help="Output format")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Create CLI request
    request = CLIRequest(
        command=args.command,
        target=args.target,
        scan_type=args.scan_type,
        user=args.user,
        session_id=args.session_id,
        options={"source": args.source} if args.source else None,
        output_format=args.output_format,
        verbose=args.verbose
    )
    
    # Execute command
    cli = CybersecurityCLI()
    response = await cli.execute_command(request)
    
    # Format and output result
    output = format_output(response, args.output_format)
    print(output)
    
    # Exit with appropriate code
    sys.exit(0 if response.success else 1)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⚠️  Operation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ CLI error: {e}")
        sys.exit(1) 