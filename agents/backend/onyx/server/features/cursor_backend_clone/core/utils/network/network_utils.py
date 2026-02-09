"""
Network Utils - Utilidades de Networking
=========================================

Utilidades para operaciones de red y conectividad.
"""

import logging
import socket
import asyncio
from typing import Optional, Tuple, List, Dict, Any
from urllib.parse import urlparse
import ipaddress

logger = logging.getLogger(__name__)

# Intentar importar httpx para requests async
try:
    import httpx
    _has_httpx = True
except ImportError:
    _has_httpx = False


def is_port_open(host: str, port: int, timeout: float = 2.0) -> bool:
    """
    Verificar si un puerto está abierto.
    
    Args:
        host: Host a verificar
        port: Puerto a verificar
        timeout: Timeout en segundos
        
    Returns:
        True si el puerto está abierto
    """
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except Exception as e:
        logger.debug(f"Error checking port {host}:{port}: {e}")
        return False


async def is_port_open_async(host: str, port: int, timeout: float = 2.0) -> bool:
    """
    Verificar si un puerto está abierto (async).
    
    Args:
        host: Host a verificar
        port: Puerto a verificar
        timeout: Timeout en segundos
        
    Returns:
        True si el puerto está abierto
    """
    try:
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(host, port),
            timeout=timeout
        )
        writer.close()
        await writer.wait_closed()
        return True
    except (asyncio.TimeoutError, OSError):
        return False


def get_local_ip() -> str:
    """
    Obtener IP local.
    
    Returns:
        IP local
    """
    try:
        # Conectar a un servidor externo para obtener IP local
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def is_valid_ip(ip: str) -> bool:
    """
    Verificar si una IP es válida.
    
    Args:
        ip: IP a verificar
        
    Returns:
        True si es válida
    """
    try:
        ipaddress.ip_address(ip)
        return True
    except ValueError:
        return False


def is_valid_url(url: str) -> bool:
    """
    Verificar si una URL es válida.
    
    Args:
        url: URL a verificar
        
    Returns:
        True si es válida
    """
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except Exception:
        return False


def parse_url(url: str) -> Dict[str, Any]:
    """
    Parsear URL en componentes.
    
    Args:
        url: URL a parsear
        
    Returns:
        Diccionario con componentes
    """
    try:
        parsed = urlparse(url)
        return {
            "scheme": parsed.scheme,
            "netloc": parsed.netloc,
            "path": parsed.path,
            "params": parsed.params,
            "query": parsed.query,
            "fragment": parsed.fragment,
            "hostname": parsed.hostname,
            "port": parsed.port,
            "username": parsed.username,
            "password": parsed.password
        }
    except Exception as e:
        logger.error(f"Error parsing URL: {e}")
        return {}


async def check_connectivity(
    url: str,
    timeout: float = 5.0
) -> Tuple[bool, Optional[str]]:
    """
    Verificar conectividad a URL.
    
    Args:
        url: URL a verificar
        timeout: Timeout en segundos
        
    Returns:
        Tupla (conectado, mensaje_error)
    """
    if not _has_httpx:
        return False, "httpx not available"
    
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(url)
            return response.status_code < 500, None
    except httpx.TimeoutException:
        return False, "Connection timeout"
    except httpx.ConnectError:
        return False, "Connection error"
    except Exception as e:
        return False, str(e)


def get_hostname() -> str:
    """
    Obtener hostname del sistema.
    
    Returns:
        Hostname
    """
    try:
        return socket.gethostname()
    except Exception:
        return "unknown"


def resolve_hostname(hostname: str) -> Optional[str]:
    """
    Resolver hostname a IP.
    
    Args:
        hostname: Hostname a resolver
        
    Returns:
        IP o None si falla
    """
    try:
        return socket.gethostbyname(hostname)
    except socket.gaierror:
        return None


def get_network_interfaces() -> List[Dict[str, Any]]:
    """
    Obtener interfaces de red.
    
    Returns:
        Lista de interfaces con información
    """
    interfaces = []
    
    try:
        import psutil
        net_if_addrs = psutil.net_if_addrs()
        
        for interface_name, addresses in net_if_addrs.items():
            interface_info = {
                "name": interface_name,
                "addresses": []
            }
            
            for addr in addresses:
                interface_info["addresses"].append({
                    "family": str(addr.family),
                    "address": addr.address,
                    "netmask": addr.netmask,
                    "broadcast": addr.broadcast
                })
            
            interfaces.append(interface_info)
    except ImportError:
        logger.warning("psutil not available for network interfaces")
    except Exception as e:
        logger.error(f"Error getting network interfaces: {e}")
    
    return interfaces


def build_url(
    scheme: str,
    host: str,
    port: Optional[int] = None,
    path: str = "/",
    query: Optional[Dict[str, Any]] = None,
    fragment: Optional[str] = None
) -> str:
    """
    Construir URL desde componentes.
    
    Args:
        scheme: Esquema (http, https)
        host: Host
        port: Puerto (opcional)
        path: Path
        query: Query parameters (opcional)
        fragment: Fragment (opcional)
        
    Returns:
        URL construida
    """
    from urllib.parse import urlencode, urlunparse
    
    netloc = host
    if port:
        netloc = f"{host}:{port}"
    
    query_str = urlencode(query) if query else ""
    
    return urlunparse((
        scheme,
        netloc,
        path,
        "",
        query_str,
        fragment or ""
    ))




