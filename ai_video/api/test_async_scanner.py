from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
import pytest
import asyncio
from unittest.mock import patch, AsyncMock
from api.async_services import scan_ports

from typing import Any, List, Dict, Optional
import logging
@pytest.mark.asyncio
async def test_scan_ports_success():
    
    """test_scan_ports_success function."""
# Mock check_port_open to always return True
    with patch('api.async_services.check_port_open', new=AsyncMock(return_value=True)):
        result = await scan_ports('127.0.0.1', [22, 80])
        assert all(r['is_open'] for r in result['results'])

@pytest.mark.asyncio
async def test_scan_ports_failure():
    
    """test_scan_ports_failure function."""
# Mock check_port_open to always return False
    with patch('api.async_services.check_port_open', new=AsyncMock(return_value=False)):
        result = await scan_ports('127.0.0.1', [22, 80])
        assert not any(r['is_open'] for r in result['results'])

@pytest.mark.asyncio
async def test_scan_ports_mixed():
    
    """test_scan_ports_mixed function."""
# Mock check_port_open to alternate True/False
    async def side_effect(host, port, timeout=1.0) -> Any:
        return port % 2 == 0
    with patch('api.async_services.check_port_open', new=AsyncMock(side_effect=side_effect)):
        result = await scan_ports('127.0.0.1', [22, 23, 24])
        assert result['results'][0]['is_open']
        assert not result['results'][1]['is_open']
        assert result['results'][2]['is_open']

@pytest.mark.asyncio
async def test_scan_ports_empty():
    
    """test_scan_ports_empty function."""
result = await scan_ports('127.0.0.1', [])
    assert result['results'] == []

@pytest.mark.asyncio
async def test_scan_ports_invalid_host():
    
    """test_scan_ports_invalid_host function."""
# Simulate exception in check_port_open
    with patch('api.async_services.check_port_open', new=AsyncMock(side_effect=Exception('fail'))):
        result = await scan_ports('invalid_host', [22])
        assert not result['results'][0]['is_open'] 