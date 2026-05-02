# app/llm_interface.py
import logging
import httpx
from config import settings
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from typing import List, Dict, Any
import json
import asyncio

logger = logging.getLogger(__name__)

# DeepSeek API Configuration
DEEPSEEK_API_KEY = settings.DEEPSEEK_API_KEY
DEEPSEEK_API_URL = settings.DEEPSEEK_API_URL
DEEPSEEK_MODEL_NAME = settings.DEEPSEEK_MODEL_NAME

async def call_deepseek_api(prompt: str, system_prompt: str = None, temperature: float = 0.7) -> str:
    """Call DeepSeek API with the given prompt and return the response."""
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    
    data = {
        "model": DEEPSEEK_MODEL_NAME,
        "messages": messages,
        "temperature": temperature
    }
    
    try:
        logger.info(f"Calling DeepSeek API with model: {DEEPSEEK_MODEL_NAME}")
        logger.debug(f"Request data: {data}")
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{DEEPSEEK_API_URL}/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=settings.LLM_REQUEST_TIMEOUT_SECONDS
            )
            
            logger.debug(f"Response status: {response.status_code}")
            logger.debug(f"Response headers: {response.headers}")
            
            if response.status_code != 200:
                logger.error(f"DeepSeek API error: {response.status_code} - {response.text}")
                raise Exception(f"DeepSeek API error: {response.status_code} - {response.text}")
            
            response_data = response.json()
            logger.debug(f"Response data: {response_data}")
            
            if not response_data.get("choices") or not response_data["choices"][0].get("message"):
                logger.error(f"Invalid response format: {response_data}")
                raise Exception("Invalid response format from DeepSeek API")
            
            return response_data["choices"][0]["message"]["content"]
    except Exception as e:
        logger.error(f"Error calling DeepSeek API: {str(e)}", exc_info=True)
        raise

async def call_deepseek_api_stream(prompt: str, system_prompt: str = None, temperature: float = 0.3, top_p: float = 0.8, max_tokens: int = 60):
    """Call DeepSeek API with streaming enabled and yield content chunks as they arrive."""
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    data = {
        "model": DEEPSEEK_MODEL_NAME,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "stream": True
    }
    try:
        logger.info(f"Calling DeepSeek API (stream) with model: {DEEPSEEK_MODEL_NAME}")
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream("POST", f"{DEEPSEEK_API_URL}/v1/chat/completions", headers=headers, json=data) as response:
                async for line in response.aiter_lines():
                    if not line or line.strip() == "data: [DONE]":
                        continue
                    if line.startswith("data: "):
                        line = line[len("data: ") :]
                    try:
                        chunk = json.loads(line)
                        delta = chunk["choices"][0]["delta"]
                        content = delta.get("content", "")
                        if content:
                            yield content
                    except Exception as e:
                        logger.warning(f"Error parsing DeepSeek stream chunk: {e}")
                        continue
    except Exception as e:
        logger.error(f"Error calling DeepSeek API (stream): {str(e)}", exc_info=True)
        raise

# --- Definición de Estructura para Brand Kit ---
class BrandKitPydantic(BaseModel):
    brand_name: str = Field(description="Nombre de la marca o empresa, si es evidente")
    slogan: str = Field(description="Slogan principal de la marca, si se detecta")
    primary_colors: List[Dict[str, str]] = Field(description="Lista de colores primarios, ej. [{'name': 'Azul Corporativo', 'hex': '#005A9C'}]")
    secondary_colors: List[Dict[str, str]] = Field(description="Lista de colores secundarios/acento")
    main_typography: Dict[str, str] = Field(description="Tipografía principal, ej. {'name': 'Montserrat', 'style': 'Sans-serif moderna y versátil'}")
    secondary_typography: Dict[str, str] = Field(description="Tipografía secundaria")
    tone_of_voice: str = Field(description="Tono de voz general descrito en 2-3 adjetivos")
    target_audience_keywords: List[str] = Field(description="Palabras clave del público objetivo principal")
    logo_description: str = Field(description="Descripción del logo si es visible o se describe")
    mission_vision_summary: str = Field(description="Misión o visión resumida si se infiere")

# --- Output Parsers ---
str_output_parser = StrOutputParser()
json_output_parser = JsonOutputParser()
pydantic_brand_kit_parser = JsonOutputParser(pydantic_object=BrandKitPydantic)

# --- Cadenas LCEL ---

# 1. Cadena para Generar Anuncios
ads_prompt_template_text = """
System: {system_prompt}
User: Basado en el siguiente contenido de un sitio web, genera EXACTAMENTE 3 copys de anuncios concisos, atractivos y optimizados para redes sociales (Facebook, Instagram).
Cada copy debe ser independiente y estar en una nueva línea. NO uses numeración, viñetas, ni introducciones como "Aquí tienes los anuncios:".
Enfócate en un lenguaje persuasivo y directo.

Contenido del sitio web:
---
{website_content}
---
Anuncios Generados:
"""
ads_prompt = ChatPromptTemplate.from_template(ads_prompt_template_text)

def parse_newline_separated_list(text: str) -> List[str]:
    return [line.strip() for line in text.split('\n') if line.strip()]

async def generate_ads_lcel(website_content: str) -> List[str]:
    try:
        logger.info(f"Generando anuncios para contenido de {len(website_content)} chars.")
        result = await call_deepseek_api(
            prompt=ads_prompt.format(
                system_prompt="Eres un copywriter experto en anuncios de alto impacto para redes sociales.",
                website_content=website_content
            )
        )
        ads = parse_newline_separated_list(result)
        logger.info(f"Anuncios generados: {len(ads)}")
        return ads
    except Exception as e:
        logger.error(f"Error generando anuncios: {e}", exc_info=True)
        return []

async def generate_ads_lcel_streaming(website_content: str):
    prompt = ads_prompt.format(
        system_prompt="Eres un copywriter experto en anuncios de alto impacto para redes sociales.",
        website_content=website_content
    )
    async for chunk in call_deepseek_api_stream(prompt=prompt):
        yield chunk

async def generate_ads_lcel_streaming_parallel(website_content: str, n_ads: int = 3):
    prompt_template = """System: Eres un copywriter experto en anuncios de alto impacto para redes sociales.\nUser: Basado en el siguiente contenido de un sitio web, genera UN copy de anuncio conciso, atractivo y optimizado para redes sociales (Facebook, Instagram). El copy debe tener un máximo de 20 palabras.\nNO uses numeración, viñetas, ni introducciones.\nContenido del sitio web:\n---\n{website_content}\n---\nAnuncio generado:"""
    async def single_ad():
        prompt = prompt_template.format(website_content=website_content)
        ad = ""
        async for chunk in call_deepseek_api_stream(prompt=prompt, temperature=0.3, top_p=0.8, max_tokens=60):
            ad += chunk
        return ad.strip()
    tasks = [asyncio.create_task(single_ad()) for _ in range(n_ads)]
    for idx, coro in enumerate(asyncio.as_completed(tasks), 1):
        ad = await coro
        yield f"{ad}\n"

# 2. Cadena para Generar Brand Kit (JSON con Pydantic Parser)
brand_kit_json_prompt_template_text = """
System: {system_prompt}
User: Analiza el siguiente contenido de un sitio web y extrae un brand kit.
Debes formatear tu respuesta EXCLUSIVAMENTE como un objeto JSON válido que se ajuste al siguiente esquema Pydantic:
```json
{json_schema}
```
"""
brand_kit_json_prompt = ChatPromptTemplate.from_template(brand_kit_json_prompt_template_text)

async def generate_brand_kit_lcel(website_content: str) -> str:
    try:
        logger.info(f"Generando brand kit para contenido de {len(website_content)} chars.")
        result = await call_deepseek_api(
            prompt=brand_kit_json_prompt.format(
                system_prompt="Eres un experto en branding y diseño de marca.",
                json_schema=BrandKitPydantic.schema_json(indent=2),
                website_content=website_content
            )
        )
        logger.info("Brand kit generado exitosamente")
        return result
    except Exception as e:
        logger.error(f"Error generando brand kit: {e}", exc_info=True)
        return ""

# 3. Cadena para Generar Contenido Personalizado
custom_content_prompt_template_text = """
System: {system_prompt}
User: Genera contenido personalizado para el siguiente contenido de un sitio web.

Contenido del sitio web:
---
{website_content}
---

Instrucciones específicas:
{prompt}"""
custom_content_prompt = ChatPromptTemplate.from_template(custom_content_prompt_template_text)

async def generate_custom_content_lcel(prompt: str, website_content: str) -> str:
    try:
        logger.info(f"Generando contenido personalizado para contenido de {len(website_content)} chars.")
        result = await call_deepseek_api(
            prompt=custom_content_prompt.format(
                system_prompt="Eres un experto en marketing digital y creación de contenido.",
                website_content=website_content,
                prompt=prompt
            )
        )
        logger.info("Contenido personalizado generado exitosamente")
        return result
    except Exception as e:
        logger.error(f"Error generando contenido personalizado: {e}", exc_info=True)
        return ""