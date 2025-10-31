#!/usr/bin/env python3
"""
DNS Enumerator Module for Video-OpusClip
DNS reconnaissance and enumeration tools
"""

import asyncio
import dns.resolver
import dns.zone
import dns.query
import dns.reversename
import socket
import subprocess
import re
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
import aiohttp
import whois

class DNSRecordType(str, Enum):
    """DNS record types"""
    A = "A"
    AAAA = "AAAA"
    MX = "MX"
    NS = "NS"
    TXT = "TXT"
    CNAME = "CNAME"
    PTR = "PTR"
    SOA = "SOA"
    SRV = "SRV"
    CAA = "CAA"
    DNSKEY = "DNSKEY"
    DS = "DS"

class EnumerationType(str, Enum):
    """Types of DNS enumeration"""
    RECORD_ENUMERATION = "record_enumeration"
    ZONE_TRANSFER = "zone_transfer"
    REVERSE_DNS = "reverse_dns"
    SUBDOMAIN_ENUMERATION = "subdomain_enumeration"
    DNS_WILDCARD = "dns_wildcard"
    DNS_AMPLIFICATION = "dns_amplification"

@dataclass
class DNSRecord:
    """DNS record information"""
    record_type: DNSRecordType
    name: str
    value: str
    ttl: Optional[int] = None
    priority: Optional[int] = None
    discovered_at: datetime = None
    
    def __post_init__(self):
        if self.discovered_at is None:
            self.discovered_at = datetime.utcnow()

@dataclass
class SubdomainResult:
    """Subdomain enumeration result"""
    subdomain: str
    ip_address: Optional[str] = None
    status: str = "unknown"
    response_time: Optional[float] = None
    discovered_at: datetime = None
    
    def __post_init__(self):
        if self.discovered_at is None:
            self.discovered_at = datetime.utcnow()

@dataclass
class EnumerationConfig:
    """Configuration for DNS enumeration"""
    target_domain: str
    enumeration_types: List[EnumerationType] = None
    record_types: List[DNSRecordType] = None
    subdomain_wordlist: List[str] = None
    max_concurrent: int = 50
    timeout: float = 10.0
    nameservers: List[str] = None
    enable_bruteforce: bool = True
    enable_zone_transfer: bool = True
    
    def __post_init__(self):
        if self.enumeration_types is None:
            self.enumeration_types = [
                EnumerationType.RECORD_ENUMERATION,
                EnumerationType.SUBDOMAIN_ENUMERATION,
                EnumerationType.REVERSE_DNS
            ]
        if self.record_types is None:
            self.record_types = [
                DNSRecordType.A,
                DNSRecordType.AAAA,
                DNSRecordType.MX,
                DNSRecordType.NS,
                DNSRecordType.TXT,
                DNSRecordType.CNAME,
                DNSRecordType.SOA
            ]
        if self.subdomain_wordlist is None:
            self.subdomain_wordlist = self._get_default_wordlist()
        if self.nameservers is None:
            self.nameservers = ["8.8.8.8", "8.8.4.4"]  # Google DNS
    
    def _get_default_wordlist(self) -> List[str]:
        """Get default subdomain wordlist"""
        return [
            "www", "mail", "ftp", "admin", "blog", "dev", "test", "stage",
            "api", "rest", "graphql", "swagger", "docs", "support", "help",
            "ns1", "ns2", "dns1", "dns2", "mx1", "mx2", "smtp", "pop",
            "imap", "webmail", "email", "portal", "login", "secure",
            "vpn", "remote", "ssh", "telnet", "ftp", "sftp", "file",
            "backup", "db", "database", "sql", "mysql", "postgres",
            "redis", "cache", "cdn", "static", "assets", "media",
            "upload", "download", "files", "images", "img", "css",
            "js", "javascript", "app", "apps", "service", "services",
            "internal", "external", "corp", "corporate", "office",
            "home", "local", "dev", "development", "staging", "prod",
            "production", "live", "beta", "alpha", "demo", "sandbox"
        ]

class DNSEnumerator:
    """DNS enumeration and reconnaissance tool"""
    
    def __init__(self, config: EnumerationConfig):
        self.config = config
        self.results: Dict[str, Any] = {}
        self.dns_records: List[DNSRecord] = []
        self.subdomains: List[SubdomainResult] = []
        self.zone_transfer_results: List[Dict[str, Any]] = []
        self.enumeration_start_time: float = 0.0
        self.enumeration_end_time: float = 0.0
    
    async def enumerate_domain(self) -> Dict[str, Any]:
        """Perform comprehensive DNS enumeration"""
        self.enumeration_start_time = asyncio.get_event_loop().time()
        
        try:
            # Configure DNS resolver
            self._configure_resolver()
            
            # Perform enumeration based on types
            if EnumerationType.RECORD_ENUMERATION in self.config.enumeration_types:
                await self._enumerate_dns_records()
            
            if EnumerationType.ZONE_TRANSFER in self.config.enumeration_types:
                await self._attempt_zone_transfer()
            
            if EnumerationType.SUBDOMAIN_ENUMERATION in self.config.enumeration_types:
                await self._enumerate_subdomains()
            
            if EnumerationType.REVERSE_DNS in self.config.enumeration_types:
                await self._reverse_dns_lookup()
            
            if EnumerationType.DNS_WILDCARD in self.config.enumeration_types:
                await self._detect_dns_wildcards()
            
            self.enumeration_end_time = asyncio.get_event_loop().time()
            
            return {
                "success": True,
                "target_domain": self.config.target_domain,
                "enumeration_duration": self.enumeration_end_time - self.enumeration_start_time,
                "total_records": len(self.dns_records),
                "total_subdomains": len(self.subdomains),
                "zone_transfers": len(self.zone_transfer_results),
                "results": {
                    "dns_records": [self._record_to_dict(r) for r in self.dns_records],
                    "subdomains": [self._subdomain_to_dict(s) for s in self.subdomains],
                    "zone_transfers": self.zone_transfer_results
                }
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "target_domain": self.config.target_domain
            }
    
    def _configure_resolver(self) -> None:
        """Configure DNS resolver with custom nameservers"""
        resolver = dns.resolver.Resolver()
        resolver.nameservers = self.config.nameservers
        resolver.timeout = self.config.timeout
        resolver.lifetime = self.config.timeout
        dns.resolver.default_resolver = resolver
    
    async def _enumerate_dns_records(self) -> None:
        """Enumerate DNS records for the target domain"""
        for record_type in self.config.record_types:
            try:
                answers = dns.resolver.resolve(self.config.target_domain, record_type.value)
                
                for answer in answers:
                    record = DNSRecord(
                        record_type=record_type,
                        name=self.config.target_domain,
                        value=str(answer),
                        ttl=answers.rrset.ttl if hasattr(answers, 'rrset') else None
                    )
                    
                    # Add priority for MX and SRV records
                    if record_type in [DNSRecordType.MX, DNSRecordType.SRV]:
                        record.priority = answer.preference if hasattr(answer, 'preference') else None
                    
                    self.dns_records.append(record)
                    
            except dns.resolver.NXDOMAIN:
                # Domain does not exist
                continue
            except dns.resolver.NoAnswer:
                # No records of this type
                continue
            except Exception as e:
                # Other DNS errors
                continue
    
    async def _attempt_zone_transfer(self) -> None:
        """Attempt DNS zone transfer"""
        try:
            # Get nameservers for the domain
            ns_records = dns.resolver.resolve(self.config.target_domain, DNSRecordType.NS.value)
            
            for ns_record in ns_records:
                nameserver = str(ns_record)
                
                try:
                    # Attempt zone transfer
                    zone = dns.zone.from_xfr(dns.query.xfr(nameserver, self.config.target_domain))
                    
                    zone_data = {
                        "nameserver": nameserver,
                        "success": True,
                        "records": []
                    }
                    
                    # Extract zone records
                    for name, node in zone.nodes.items():
                        for rdataset in node.rdatasets:
                            for rdata in rdataset:
                                record = {
                                    "name": str(name),
                                    "type": dns.rdatatype.to_text(rdataset.rdtype),
                                    "value": str(rdata),
                                    "ttl": rdataset.ttl
                                }
                                zone_data["records"].append(record)
                    
                    self.zone_transfer_results.append(zone_data)
                    
                except Exception as e:
                    # Zone transfer failed
                    zone_data = {
                        "nameserver": nameserver,
                        "success": False,
                        "error": str(e)
                    }
                    self.zone_transfer_results.append(zone_data)
                    
        except Exception as e:
            # Could not resolve nameservers
            pass
    
    async def _enumerate_subdomains(self) -> None:
        """Enumerate subdomains using various techniques"""
        discovered_subdomains = set()
        
        # Technique 1: Brute force with wordlist
        if self.config.enable_bruteforce:
            brute_force_results = await self._brute_force_subdomains()
            discovered_subdomains.update(brute_force_results)
        
        # Technique 2: Certificate transparency logs
        ct_results = await self._certificate_transparency_search()
        discovered_subdomains.update(ct_results)
        
        # Technique 3: Search engine results
        search_results = await self._search_engine_subdomains()
        discovered_subdomains.update(search_results)
        
        # Resolve discovered subdomains
        await self._resolve_subdomains(list(discovered_subdomains))
    
    async def _brute_force_subdomains(self) -> List[str]:
        """Brute force subdomains using wordlist"""
        discovered = []
        
        # Use semaphore for concurrency control
        semaphore = asyncio.Semaphore(self.config.max_concurrent)
        
        tasks = [
            self._check_subdomain(subdomain, semaphore)
            for subdomain in self.config.subdomain_wordlist
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, str):
                discovered.append(result)
        
        return discovered
    
    async def _check_subdomain(self, subdomain: str, semaphore: asyncio.Semaphore) -> Optional[str]:
        """Check if a subdomain exists"""
        async with semaphore:
            try:
                full_domain = f"{subdomain}.{self.config.target_domain}"
                start_time = asyncio.get_event_loop().time()
                
                # Try to resolve A record
                answers = dns.resolver.resolve(full_domain, DNSRecordType.A.value)
                
                response_time = asyncio.get_event_loop().time() - start_time
                
                if answers:
                    result = SubdomainResult(
                        subdomain=full_domain,
                        ip_address=str(answers[0]),
                        status="active",
                        response_time=response_time
                    )
                    self.subdomains.append(result)
                    return full_domain
                
            except dns.resolver.NXDOMAIN:
                # Subdomain does not exist
                pass
            except Exception:
                # Other DNS errors
                pass
        
        return None
    
    async def _certificate_transparency_search(self) -> List[str]:
        """Search certificate transparency logs for subdomains"""
        discovered = []
        
        try:
            # Use crt.sh API
            url = f"https://crt.sh/?q=%.{self.config.target_domain}&output=json"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        for cert in data:
                            if 'name_value' in cert:
                                names = cert['name_value'].split('\n')
                                for name in names:
                                    name = name.strip()
                                    if name.endswith(self.config.target_domain) and name != self.config.target_domain:
                                        discovered.append(name)
        
        except Exception:
            # API request failed
            pass
        
        return list(set(discovered))
    
    async def _search_engine_subdomains(self) -> List[str]:
        """Search for subdomains using search engines"""
        discovered = []
        
        try:
            # Use Google search (simplified)
            search_query = f"site:*.{self.config.target_domain}"
            
            async with aiohttp.ClientSession() as session:
                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                }
                
                # Note: This is a simplified approach. Real implementation would need proper search API
                # For demonstration, we'll use a basic approach
                pass
                
        except Exception:
            # Search failed
            pass
        
        return list(set(discovered))
    
    async def _resolve_subdomains(self, subdomains: List[str]) -> None:
        """Resolve subdomains to IP addresses"""
        semaphore = asyncio.Semaphore(self.config.max_concurrent)
        
        tasks = [
            self._resolve_single_subdomain(subdomain, semaphore)
            for subdomain in subdomains
        ]
        
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _resolve_single_subdomain(self, subdomain: str, semaphore: asyncio.Semaphore) -> None:
        """Resolve a single subdomain"""
        async with semaphore:
            try:
                start_time = asyncio.get_event_loop().time()
                
                # Try A record first
                try:
                    answers = dns.resolver.resolve(subdomain, DNSRecordType.A.value)
                    ip_address = str(answers[0])
                    status = "active"
                except dns.resolver.NXDOMAIN:
                    # Try CNAME
                    try:
                        answers = dns.resolver.resolve(subdomain, DNSRecordType.CNAME.value)
                        ip_address = str(answers[0])
                        status = "cname"
                    except:
                        ip_address = None
                        status = "unresolved"
                
                response_time = asyncio.get_event_loop().time() - start_time
                
                result = SubdomainResult(
                    subdomain=subdomain,
                    ip_address=ip_address,
                    status=status,
                    response_time=response_time
                )
                self.subdomains.append(result)
                
            except Exception:
                # Resolution failed
                result = SubdomainResult(
                    subdomain=subdomain,
                    status="error"
                )
                self.subdomains.append(result)
    
    async def _reverse_dns_lookup(self) -> None:
        """Perform reverse DNS lookups for discovered IPs"""
        ip_addresses = set()
        
        # Collect IP addresses from DNS records and subdomains
        for record in self.dns_records:
            if record.record_type in [DNSRecordType.A, DNSRecordType.AAAA]:
                ip_addresses.add(record.value)
        
        for subdomain in self.subdomains:
            if subdomain.ip_address:
                ip_addresses.add(subdomain.ip_address)
        
        # Perform reverse DNS lookups
        semaphore = asyncio.Semaphore(self.config.max_concurrent)
        
        tasks = [
            self._reverse_lookup_ip(ip, semaphore)
            for ip in ip_addresses
        ]
        
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _reverse_lookup_ip(self, ip_address: str, semaphore: asyncio.Semaphore) -> None:
        """Perform reverse DNS lookup for an IP address"""
        async with semaphore:
            try:
                # Create reverse DNS name
                if ':' in ip_address:  # IPv6
                    reverse_name = dns.reversename.from_address(ip_address)
                else:  # IPv4
                    reverse_name = dns.reversename.from_address(ip_address)
                
                # Perform PTR lookup
                answers = dns.resolver.resolve(reverse_name, DNSRecordType.PTR.value)
                
                for answer in answers:
                    record = DNSRecord(
                        record_type=DNSRecordType.PTR,
                        name=str(reverse_name),
                        value=str(answer)
                    )
                    self.dns_records.append(record)
                    
            except Exception:
                # Reverse lookup failed
                pass
    
    async def _detect_dns_wildcards(self) -> None:
        """Detect DNS wildcard configurations"""
        try:
            # Generate random subdomain
            import random
            import string
            
            random_subdomain = ''.join(random.choices(string.ascii_lowercase, k=10))
            test_domain = f"{random_subdomain}.{self.config.target_domain}"
            
            # Try to resolve random subdomain
            try:
                answers = dns.resolver.resolve(test_domain, DNSRecordType.A.value)
                
                # If it resolves, it might be a wildcard
                wildcard_result = {
                    "wildcard_detected": True,
                    "test_subdomain": test_domain,
                    "resolved_ip": str(answers[0]),
                    "description": "DNS wildcard detected - random subdomain resolves"
                }
                
            except dns.resolver.NXDOMAIN:
                wildcard_result = {
                    "wildcard_detected": False,
                    "test_subdomain": test_domain,
                    "description": "No DNS wildcard detected"
                }
            
            self.results["wildcard_detection"] = wildcard_result
            
        except Exception as e:
            self.results["wildcard_detection"] = {
                "wildcard_detected": "unknown",
                "error": str(e)
            }
    
    def _record_to_dict(self, record: DNSRecord) -> Dict[str, Any]:
        """Convert DNSRecord to dictionary"""
        return {
            "record_type": record.record_type.value,
            "name": record.name,
            "value": record.value,
            "ttl": record.ttl,
            "priority": record.priority,
            "discovered_at": record.discovered_at.isoformat() if record.discovered_at else None
        }
    
    def _subdomain_to_dict(self, subdomain: SubdomainResult) -> Dict[str, Any]:
        """Convert SubdomainResult to dictionary"""
        return {
            "subdomain": subdomain.subdomain,
            "ip_address": subdomain.ip_address,
            "status": subdomain.status,
            "response_time": subdomain.response_time,
            "discovered_at": subdomain.discovered_at.isoformat() if subdomain.discovered_at else None
        }
    
    def get_records_by_type(self, record_type: DNSRecordType) -> List[DNSRecord]:
        """Get DNS records by type"""
        return [r for r in self.dns_records if r.record_type == record_type]
    
    def get_active_subdomains(self) -> List[SubdomainResult]:
        """Get active subdomains"""
        return [s for s in self.subdomains if s.status == "active"]
    
    def generate_report(self) -> str:
        """Generate DNS enumeration report"""
        report = f"DNS Enumeration Report for {self.config.target_domain}\n"
        report += "=" * 60 + "\n"
        report += f"Enumeration Duration: {self.enumeration_end_time - self.enumeration_start_time:.2f} seconds\n"
        report += f"Total DNS Records: {len(self.dns_records)}\n"
        report += f"Total Subdomains: {len(self.subdomains)}\n"
        report += f"Active Subdomains: {len(self.get_active_subdomains())}\n"
        report += f"Zone Transfers: {len(self.zone_transfer_results)}\n\n"
        
        # DNS Records by type
        report += "DNS Records by Type:\n"
        report += "-" * 30 + "\n"
        for record_type in DNSRecordType:
            records = self.get_records_by_type(record_type)
            if records:
                report += f"{record_type.value} Records ({len(records)}):\n"
                for record in records:
                    report += f"  • {record.name} -> {record.value}"
                    if record.ttl:
                        report += f" (TTL: {record.ttl})"
                    report += "\n"
                report += "\n"
        
        # Subdomains
        active_subdomains = self.get_active_subdomains()
        if active_subdomains:
            report += "Active Subdomains:\n"
            report += "-" * 20 + "\n"
            for subdomain in active_subdomains:
                report += f"• {subdomain.subdomain} -> {subdomain.ip_address}"
                if subdomain.response_time:
                    report += f" ({subdomain.response_time:.3f}s)"
                report += "\n"
            report += "\n"
        
        # Zone Transfer Results
        successful_transfers = [zt for zt in self.zone_transfer_results if zt.get("success")]
        if successful_transfers:
            report += "Successful Zone Transfers:\n"
            report += "-" * 30 + "\n"
            for transfer in successful_transfers:
                report += f"• Nameserver: {transfer['nameserver']}\n"
                report += f"  Records: {len(transfer.get('records', []))}\n"
            report += "\n"
        
        return report

# Example usage
async def main():
    """Example usage of DNS enumerator"""
    print("🔍 DNS Enumerator Example")
    
    # Create enumeration configuration
    config = EnumerationConfig(
        target_domain="example.com",
        enumeration_types=[
            EnumerationType.RECORD_ENUMERATION,
            EnumerationType.SUBDOMAIN_ENUMERATION,
            EnumerationType.REVERSE_DNS,
            EnumerationType.DNS_WILDCARD
        ],
        max_concurrent=20,
        timeout=5.0
    )
    
    # Create enumerator
    enumerator = DNSEnumerator(config)
    
    # Perform enumeration
    print(f"Enumerating DNS for {config.target_domain}...")
    result = await enumerator.enumerate_domain()
    
    if result["success"]:
        print(f"✅ Enumeration completed in {result['enumeration_duration']:.2f} seconds")
        print(f"📊 Found {result['total_records']} DNS records")
        print(f"🌐 Found {result['total_subdomains']} subdomains")
        print(f"🔄 Attempted {result['zone_transfers']} zone transfers")
        
        # Print some results
        if result['results']['dns_records']:
            print("\n📋 DNS Records:")
            for record in result['results']['dns_records'][:5]:  # Show first 5
                print(f"  {record['record_type']}: {record['name']} -> {record['value']}")
        
        if result['results']['subdomains']:
            print("\n🌐 Subdomains:")
            for subdomain in result['results']['subdomains'][:5]:  # Show first 5
                print(f"  {subdomain['subdomain']} -> {subdomain['ip_address']}")
        
        # Generate report
        print("\n📋 DNS Enumeration Report:")
        print(enumerator.generate_report())
        
    else:
        print(f"❌ Enumeration failed: {result['error']}")

if __name__ == "__main__":
    asyncio.run(main()) 