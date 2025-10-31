from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from typing import Any, List, Dict, Optional
import logging
import asyncio
def test_scrape(client) -> Any:
    payload = {"url": "https://ejemplo.com"}
    response = client.post("/seo/scrape", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["data"]["title"] == "Ejemplo"
    assert "description" in data["data"] 