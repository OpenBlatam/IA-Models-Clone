from typing import TYPE_CHECKING

from langchain.schema.messages import AIMessage
from langchain.schema.messages import BaseMessage
from langchain.schema.messages import HumanMessage
from langchain.schema.messages import SystemMessage
from pydantic import BaseModel
from typing import List, Dict
from onyx.utils.langchain import LangchainField
from onyx.utils.logger import setup_logger

logger = setup_logger()

class BrandKit(BaseModel):
    """Simplified version of `BrandKit` following Onyx backend conventions."""

    brand_name: str = LangchainField(description="Nombre de la marca")
    slogan: str = LangchainField(description="Slogan o tagline de la marca")
    primary_colors: List[Dict[str, str]] = LangchainField(
        description="Lista de colores primarios, ej. [{'name': 'Azul Corporativo', 'hex': '#005A9C'}]"
    )
    secondary_colors: List[Dict[str, str]] = LangchainField(
        description="Lista de colores secundarios/acento"
    )
    main_typography: Dict[str, str] = LangchainField(
        description="Tipografía principal, ej. {'name': 'Montserrat', 'style': 'Sans-serif moderna y versátil'}"
    )
    secondary_typography: Dict[str, str] = LangchainField(
        description="Tipografía secundaria"
    )
    tone_of_voice: str = LangchainField(
        description="Tono de voz general descrito en 2-3 adjetivos"
    )
    target_audience_keywords: List[str] = LangchainField(
        description="Palabras clave del público objetivo principal"
    )
    logo_description: str = LangchainField(
        description="Descripción del logo si es visible o se describe"
    )
    mission_vision_summary: str = LangchainField(
        description="Misión o visión resumida si se infiere"
    )

    class Config:
        arbitrary_types_allowed = True

