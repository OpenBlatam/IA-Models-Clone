#!/usr/bin/env python3
"""
Network Helpers Module for Video-OpusClip
Network utilities and connectivity functions
"""

import asyncio
import aiohttp
import socket
import ssl
import dns.resolver
import dns.reversename
import whois
import requests
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
import json
import ipaddress
import subprocess
import platform

class Protocol(str, Enum):
    """Network protocols"""
    TCP = "tcp"
    UDP = "udp"
    HTTP = "http"
    HTTPS = "https"
    FTP = "ftp"
    SSH = "ssh"
    SMTP = "smtp"
    DNS = "dns"

class ConnectionStatus(str, Enum):
    """Connection status"""
    CONNECTED = "connected"
    TIMEOUT = "timeout"
    REFUSED = "refused"
    UNREACHABLE = "unreachable"
    ERROR = "error"

@dataclass
class NetworkConfig:
    """Configuration for network operations"""
    timeout: float = 30.0
    max_retries: int = 3
    user_agent: str = "Video-OpusClip-Network/1.0"
    verify_ssl: bool = True
    follow_redirects: bool = True
    max_redirects: int = 5

@dataclass
class NetworkResult:
    """Network operation result"""
    success: bool
    status: ConnectionStatus
    response_time: Optional[float] = None
    data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()

class NetworkHelpers:
    """Network utilities and connectivity functions"""
    
    def __init__(self, config: Optional[NetworkConfig] = None):
        self.config = config or NetworkConfig()
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.config.timeout),
            headers={"User-Agent": self.config.user_agent}
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def check_connectivity(self, host: str, port: int, protocol: Protocol = Protocol.TCP) -> NetworkResult:
        """Check connectivity to a host and port"""
        start_time = datetime.utcnow()
        
        try:
            if protocol == Protocol.TCP:
                result = await self._check_tcp_connectivity(host, port)
            elif protocol == Protocol.UDP:
                result = await self._check_udp_connectivity(host, port)
            else:
                return NetworkResult(
                    success=False,
                    status=ConnectionStatus.ERROR,
                    error_message=f"Unsupported protocol: {protocol}"
                )
            
            # Calculate response time
            end_time = datetime.utcnow()
            result.response_time = (end_time - start_time).total_seconds()
            
            return result
            
        except Exception as e:
            return NetworkResult(
                success=False,
                status=ConnectionStatus.ERROR,
                error_message=str(e)
            )
    
    async def _check_tcp_connectivity(self, host: str, port: int) -> NetworkResult:
        """Check TCP connectivity"""
        try:
            # Use asyncio to avoid blocking
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(host, port),
                timeout=self.config.timeout
            )
            
            writer.close()
            await writer.wait_closed()
            
            return NetworkResult(
                success=True,
                status=ConnectionStatus.CONNECTED,
                data={"host": host, "port": port, "protocol": "tcp"}
            )
            
        except asyncio.TimeoutError:
            return NetworkResult(
                success=False,
                status=ConnectionStatus.TIMEOUT,
                error_message="Connection timeout"
            )
        except ConnectionRefusedError:
            return NetworkResult(
                success=False,
                status=ConnectionStatus.REFUSED,
                error_message="Connection refused"
            )
        except Exception as e:
            return NetworkResult(
                success=False,
                status=ConnectionStatus.ERROR,
                error_message=str(e)
            )
    
    async def _check_udp_connectivity(self, host: str, port: int) -> NetworkResult:
        """Check UDP connectivity"""
        try:
            # Create UDP socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.settimeout(self.config.timeout)
            
            # Try to connect
            sock.connect((host, port))
            
            # Send a test packet
            sock.send(b"test")
            
            # Try to receive response
            try:
                sock.recv(1024)
                success = True
                status = ConnectionStatus.CONNECTED
            except socket.timeout:
                # UDP might not respond, but connection was established
                success = True
                status = ConnectionStatus.CONNECTED
            
            sock.close()
            
            return NetworkResult(
                success=success,
                status=status,
                data={"host": host, "port": port, "protocol": "udp"}
            )
            
        except Exception as e:
            return NetworkResult(
                success=False,
                status=ConnectionStatus.ERROR,
                error_message=str(e)
            )
    
    async def scan_ports(self, host: str, ports: List[int], protocol: Protocol = Protocol.TCP) -> Dict[int, NetworkResult]:
        """Scan multiple ports on a host"""
        results = {}
        
        # Create tasks for concurrent scanning
        tasks = []
        for port in ports:
            task = self.check_connectivity(host, port, protocol)
            tasks.append((port, task))
        
        # Execute all tasks concurrently
        for port, task in tasks:
            result = await task
            results[port] = result
        
        return results
    
    async def scan_port_range(self, host: str, start_port: int, end_port: int, protocol: Protocol = Protocol.TCP) -> Dict[int, NetworkResult]:
        """Scan a range of ports on a host"""
        ports = list(range(start_port, end_port + 1))
        return await self.scan_ports(host, ports, protocol)
    
    async def get_dns_info(self, domain: str) -> Dict[str, Any]:
        """Get DNS information for a domain"""
        try:
            results = {
                "domain": domain,
                "a_records": [],
                "aaaa_records": [],
                "mx_records": [],
                "ns_records": [],
                "txt_records": [],
                "cname_records": [],
                "soa_record": None,
                "ptr_records": []
            }
            
            # A records
            try:
                a_records = dns.resolver.resolve(domain, 'A')
                results["a_records"] = [str(record) for record in a_records]
            except Exception:
                pass
            
            # AAAA records
            try:
                aaaa_records = dns.resolver.resolve(domain, 'AAAA')
                results["aaaa_records"] = [str(record) for record in aaaa_records]
            except Exception:
                pass
            
            # MX records
            try:
                mx_records = dns.resolver.resolve(domain, 'MX')
                results["mx_records"] = [str(record.exchange) for record in mx_records]
            except Exception:
                pass
            
            # NS records
            try:
                ns_records = dns.resolver.resolve(domain, 'NS')
                results["ns_records"] = [str(record) for record in ns_records]
            except Exception:
                pass
            
            # TXT records
            try:
                txt_records = dns.resolver.resolve(domain, 'TXT')
                results["txt_records"] = [str(record) for record in txt_records]
            except Exception:
                pass
            
            # CNAME records
            try:
                cname_records = dns.resolver.resolve(domain, 'CNAME')
                results["cname_records"] = [str(record) for record in cname_records]
            except Exception:
                pass
            
            # SOA record
            try:
                soa_records = dns.resolver.resolve(domain, 'SOA')
                if soa_records:
                    soa = soa_records[0]
                    results["soa_record"] = {
                        "mname": str(soa.mname),
                        "rname": str(soa.rname),
                        "serial": soa.serial,
                        "refresh": soa.refresh,
                        "retry": soa.retry,
                        "expire": soa.expire,
                        "minimum": soa.minimum
                    }
            except Exception:
                pass
            
            return results
            
        except Exception as e:
            return {"error": str(e), "domain": domain}
    
    async def reverse_dns_lookup(self, ip_address: str) -> Dict[str, Any]:
        """Perform reverse DNS lookup"""
        try:
            # Validate IP address
            ip = ipaddress.ip_address(ip_address)
            
            # Create reverse name
            reverse_name = dns.reversename.from_address(ip_address)
            
            # Perform PTR lookup
            ptr_records = dns.resolver.resolve(reverse_name, 'PTR')
            
            return {
                "ip_address": ip_address,
                "ptr_records": [str(record) for record in ptr_records],
                "is_valid_ip": True
            }
            
        except Exception as e:
            return {
                "ip_address": ip_address,
                "error": str(e),
                "is_valid_ip": False
            }
    
    async def get_whois_info(self, domain: str) -> Dict[str, Any]:
        """Get WHOIS information for a domain"""
        try:
            # Use asyncio to run whois in a thread
            whois_info = await asyncio.to_thread(whois.whois, domain)
            
            return {
                "domain": domain,
                "registrar": whois_info.registrar,
                "creation_date": whois_info.creation_date,
                "expiration_date": whois_info.expiration_date,
                "updated_date": whois_info.updated_date,
                "name_servers": whois_info.name_servers,
                "status": whois_info.status,
                "emails": whois_info.emails,
                "raw_data": str(whois_info)
            }
            
        except Exception as e:
            return {"error": str(e), "domain": domain}
    
    async def check_ssl_certificate(self, host: str, port: int = 443) -> Dict[str, Any]:
        """Check SSL certificate information"""
        try:
            context = ssl.create_default_context()
            
            with socket.create_connection((host, port), timeout=self.config.timeout) as sock:
                with context.wrap_socket(sock, server_hostname=host) as ssock:
                    cert = ssock.getpeercert()
                    
                    return {
                        "host": host,
                        "port": port,
                        "subject": dict(x[0] for x in cert['subject']),
                        "issuer": dict(x[0] for x in cert['issuer']),
                        "version": cert['version'],
                        "serial_number": cert['serialNumber'],
                        "not_before": cert['notBefore'],
                        "not_after": cert['notAfter'],
                        "san": cert.get('subjectAltName', []),
                        "cipher": ssock.cipher(),
                        "protocol": ssock.version()
                    }
                    
        except Exception as e:
            return {"error": str(e), "host": host, "port": port}
    
    async def http_request(self, url: str, method: str = "GET", headers: Optional[Dict[str, str]] = None, 
                          data: Optional[Dict[str, Any]] = None, timeout: Optional[float] = None) -> NetworkResult:
        """Make HTTP request"""
        if not self.session:
            raise RuntimeError("NetworkHelpers must be used as async context manager")
        
        start_time = datetime.utcnow()
        
        try:
            request_timeout = timeout or self.config.timeout
            request_headers = headers or {}
            
            if data and method.upper() == "POST":
                async with self.session.post(url, headers=request_headers, json=data, timeout=request_timeout) as response:
                    response_data = await response.text()
                    end_time = datetime.utcnow()
                    
                    return NetworkResult(
                        success=response.status < 400,
                        status=ConnectionStatus.CONNECTED if response.status < 400 else ConnectionStatus.ERROR,
                        response_time=(end_time - start_time).total_seconds(),
                        data={
                            "status_code": response.status,
                            "headers": dict(response.headers),
                            "content": response_data,
                            "url": url
                        }
                    )
            else:
                async with self.session.get(url, headers=request_headers, timeout=request_timeout) as response:
                    response_data = await response.text()
                    end_time = datetime.utcnow()
                    
                    return NetworkResult(
                        success=response.status < 400,
                        status=ConnectionStatus.CONNECTED if response.status < 400 else ConnectionStatus.ERROR,
                        response_time=(end_time - start_time).total_seconds(),
                        data={
                            "status_code": response.status,
                            "headers": dict(response.headers),
                            "content": response_data,
                            "url": url
                        }
                    )
                    
        except asyncio.TimeoutError:
            return NetworkResult(
                success=False,
                status=ConnectionStatus.TIMEOUT,
                error_message="Request timeout"
            )
        except Exception as e:
            return NetworkResult(
                success=False,
                status=ConnectionStatus.ERROR,
                error_message=str(e)
            )
    
    async def ping_host(self, host: str, count: int = 4) -> Dict[str, Any]:
        """Ping a host"""
        try:
            # Determine ping command based on platform
            if platform.system().lower() == "windows":
                cmd = ["ping", "-n", str(count), host]
            else:
                cmd = ["ping", "-c", str(count), host]
            
            # Run ping command
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                output = stdout.decode()
                
                # Parse ping results
                lines = output.split('\n')
                sent = received = lost = 0
                times = []
                
                for line in lines:
                    if "packets transmitted" in line.lower():
                        # Extract packet statistics
                        parts = line.split(',')
                        for part in parts:
                            if "packets transmitted" in part:
                                sent = int(part.split()[0])
                            elif "received" in part:
                                received = int(part.split()[0])
                            elif "lost" in part:
                                lost = int(part.split()[0])
                    
                    elif "time=" in line and "ms" in line:
                        # Extract response time
                        try:
                            time_str = line.split("time=")[1].split()[0]
                            times.append(float(time_str))
                        except (IndexError, ValueError):
                            pass
                
                return {
                    "host": host,
                    "success": True,
                    "sent": sent,
                    "received": received,
                    "lost": lost,
                    "loss_percentage": (lost / sent * 100) if sent > 0 else 0,
                    "times": times,
                    "min_time": min(times) if times else None,
                    "max_time": max(times) if times else None,
                    "avg_time": sum(times) / len(times) if times else None
                }
            else:
                return {
                    "host": host,
                    "success": False,
                    "error": stderr.decode()
                }
                
        except Exception as e:
            return {
                "host": host,
                "success": False,
                "error": str(e)
            }
    
    async def traceroute(self, host: str, max_hops: int = 30) -> List[Dict[str, Any]]:
        """Perform traceroute to a host"""
        try:
            # Determine traceroute command based on platform
            if platform.system().lower() == "windows":
                cmd = ["tracert", "-h", str(max_hops), host]
            else:
                cmd = ["traceroute", "-m", str(max_hops), host]
            
            # Run traceroute command
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                output = stdout.decode()
                lines = output.split('\n')
                
                hops = []
                for line in lines:
                    if line.strip() and not line.startswith('traceroute') and not line.startswith('tracert'):
                        # Parse hop information
                        parts = line.split()
                        if len(parts) >= 2:
                            try:
                                hop_num = int(parts[0])
                                hop_info = {
                                    "hop": hop_num,
                                    "host": parts[1] if parts[1] != "*" else None,
                                    "ip": None,
                                    "times": []
                                }
                                
                                # Extract response times
                                for part in parts[2:]:
                                    if part.endswith('ms'):
                                        try:
                                            time_val = float(part[:-2])
                                            hop_info["times"].append(time_val)
                                        except ValueError:
                                            pass
                                
                                hops.append(hop_info)
                            except ValueError:
                                continue
                
                return hops
            else:
                return [{"error": stderr.decode()}]
                
        except Exception as e:
            return [{"error": str(e)}]
    
    def get_local_ip(self) -> str:
        """Get local IP address"""
        try:
            # Connect to a remote address to determine local IP
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(("8.8.8.8", 80))
                return s.getsockname()[0]
        except Exception:
            return "127.0.0.1"
    
    def get_public_ip(self) -> str:
        """Get public IP address"""
        try:
            response = requests.get("https://api.ipify.org", timeout=self.config.timeout)
            return response.text
        except Exception:
            return "Unknown"
    
    async def check_port_common_services(self, host: str, port: int) -> Dict[str, Any]:
        """Check common services on a port"""
        common_services = {
            21: "FTP",
            22: "SSH",
            23: "Telnet",
            25: "SMTP",
            53: "DNS",
            80: "HTTP",
            110: "POP3",
            143: "IMAP",
            443: "HTTPS",
            993: "IMAPS",
            995: "POP3S",
            3306: "MySQL",
            5432: "PostgreSQL",
            6379: "Redis",
            8080: "HTTP-Proxy",
            8443: "HTTPS-Alt"
        }
        
        service_name = common_services.get(port, "Unknown")
        
        # Check connectivity
        result = await self.check_connectivity(host, port)
        
        return {
            "host": host,
            "port": port,
            "service": service_name,
            "connected": result.success,
            "status": result.status.value,
            "response_time": result.response_time,
            "error": result.error_message
        }

# Example usage
async def main():
    """Example usage of network helpers"""
    print("🌐 Network Helpers Example")
    
    async with NetworkHelpers() as network:
        # Check connectivity
        result = await network.check_connectivity("google.com", 80)
        print(f"Google.com connectivity: {result.status.value}")
        
        # Scan common ports
        common_ports = [21, 22, 23, 25, 53, 80, 110, 143, 443, 993, 995, 3306, 5432, 6379, 8080, 8443]
        scan_results = await network.scan_ports("scanme.nmap.org", common_ports)
        
        print("\nPort scan results:")
        for port, result in scan_results.items():
            if result.success:
                print(f"  Port {port}: {result.status.value}")
        
        # DNS lookup
        dns_info = await network.get_dns_info("google.com")
        print(f"\nDNS A records: {dns_info.get('a_records', [])}")
        
        # Reverse DNS
        reverse_dns = await network.reverse_dns_lookup("8.8.8.8")
        print(f"Reverse DNS: {reverse_dns.get('ptr_records', [])}")
        
        # HTTP request
        http_result = await network.http_request("https://httpbin.org/json")
        if http_result.success:
            print(f"HTTP Status: {http_result.data['status_code']}")
        
        # Ping test
        ping_result = await network.ping_host("google.com", count=3)
        if ping_result["success"]:
            print(f"Ping average: {ping_result['avg_time']:.2f}ms")
        
        # SSL certificate check
        ssl_info = await network.check_ssl_certificate("google.com")
        if "error" not in ssl_info:
            print(f"SSL Issuer: {ssl_info['issuer']}")
        
        # Local and public IP
        local_ip = network.get_local_ip()
        public_ip = network.get_public_ip()
        print(f"Local IP: {local_ip}")
        print(f"Public IP: {public_ip}")

if __name__ == "__main__":
    asyncio.run(main()) 