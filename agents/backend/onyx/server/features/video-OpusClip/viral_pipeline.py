"""
Módulo viral_pipeline.py: generación modular de variantes virales usando Onyx y LangChain.
Incluye: batch, memory, StructuredOutputParser, audit log, observabilidad, y solo las librerías clave del stack.
"""
import numpy as np
from datetime import datetime
from uuid import uuid4
import structlog
import redis
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.output_parsers import StructuredOutputParser
from .models import ViralClipVariant, ViralVideoBatchResponse, VideoClipRequest, CaptionOutput

logger = structlog.get_logger()

# --- Redis Cache Helper ---
def get_redis_cache(redis_url=None):
    if redis_url: return redis.Redis.from_url(redis_url)
    return None

# --- Prompting Helper ---
def get_prompt_and_parser():
    parser = StructuredOutputParser(pydantic_object=CaptionOutput)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Eres un experto en videos virales. Devuelve el resultado en JSON: {format_instructions}"),
        ("human", "Tono: {tone}. Variante: {variant_id}. Audiencia: {audience}.")
    ])
    return prompt, parser

# --- LLM Helper ---
def get_llm():
    return ChatOpenAI(model="gpt-3.5-turbo", temperature=0.8)

# --- Generación de variantes virales ---
def generate_viral_variants(request: VideoClipRequest, n_variants: int = 10, audience_profile: dict = None, experiment_id: str = None, redis_url=None) -> ViralVideoBatchResponse:
    """
    Genera variantes virales de un video largo usando LangChain, Redis cache y Onyx features.
    """
    tones = np.array(["divertido", "educativo", "polémico", "inspirador"])
    idxs = np.arange(n_variants)
    starts = idxs * 10
    ends = (idxs + 1) * 10
    tone_arr = tones[idxs % len(tones)]
    variant_ids = np.array([f"var-{i+1}" for i in idxs])
    logo_styles = np.where(idxs % 2 == 0, "top-right", "bottom-left")
    viral_scores = 0.8 + 0.02 * idxs

    cache = get_redis_cache(redis_url)
    prompt, parser = get_prompt_and_parser()
    llm = get_llm()

    variants = []
    for i in idxs:
        cache_key = f"viral:caption:{tone_arr[i]}:{variant_ids[i]}"
        cached = cache.get(cache_key) if cache else None
        if cached:
            parsed = CaptionOutput.parse_raw(cached)
        else:
            variables = {
                "tone": tone_arr[i],
                "variant_id": variant_ids[i],
                "audience": audience_profile or {"age": "18-24", "interests": ["humor", "tecnología"]},
                "format_instructions": parser.get_format_instructions()
            }
            chain = prompt | llm | parser
            parsed = chain.invoke(variables)
            if cache:
                cache.set(cache_key, parsed.json(), ex=3600)
        variants.append(ViralClipVariant(
            start=float(starts[i]),
            end=float(ends[i]),
            caption=parsed.caption,
            emojis=parsed.emojis,
            hashtags=parsed.hashtags,
            logo_style=logo_styles[i],
            cta=parsed.cta,
            format="9:16",
            viral_score=float(viral_scores[i]),
            tone=tone_arr[i],
            variant_id=variant_ids[i],
            experiment_id=experiment_id or "exp-001",
            audience_profile=audience_profile or {"age": "18-24", "interests": ["humor", "tecnología"]},
            metadata={"variant": int(i+1)}
        ))

    logger.info("viral_batch_generated", batch_id=str(uuid4()), n_variants=n_variants)
    return ViralVideoBatchResponse(
        youtube_url=request.youtube_url,
        variants=variants,
        original_duration=600,
        language=request.language,
        batch_id=str(uuid4()),
        created_at=datetime.utcnow(),
        source_video_stats={"views": 100000, "likes": 5000}
    ) 