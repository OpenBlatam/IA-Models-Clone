"""
Ejemplos reales de refactorización aplicando helpers al código existente.
Estos son cambios concretos que se pueden aplicar directamente.
"""

# ============================================================================
# REFACTORIZACIÓN REAL #1: services/profile_extractor.py
# Método: _extract_tiktok_videos
# ============================================================================

# CÓDIGO ORIGINAL (líneas 259-324)
"""
async def _extract_tiktok_videos(
    self,
    username: str
) -> List[VideoContent]:
    videos: List[VideoContent] = []
    max_videos = self.settings.max_videos_per_profile
    
    try:
        video_list = await self.tiktok_connector.get_videos(
            username,
            limit=max_videos
        )
        
        for video_data in video_list:
            try:
                video = VideoContent(
                    video_id=video_data.get("video_id"),
                    url=video_data.get("url"),
                    title=video_data.get("title"),
                    description=video_data.get("description"),
                    duration=video_data.get("duration"),
                    views=video_data.get("views"),
                    likes=video_data.get("likes"),
                    comments=video_data.get("comments"),
                    created_at=video_data.get("created_at"),
                    hashtags=video_data.get("hashtags", []),
                    metadata=video_data
                )
                videos.append(video)
            except Exception as e:
                self.logger.warning(
                    f"Error procesando video individual: {e}"
                )
                continue
                
    except ConnectorError as e:
        self.logger.error(
            f"Error extrayendo videos de TikTok: {e}",
            exc_info=True
        )
        raise ProfileExtractionError(
            f"Error extrayendo videos de TikTok: {str(e)}",
            error_code="VIDEO_EXTRACTION_ERROR",
            details={"username": username}
        ) from e
    except Exception as e:
        self.logger.error(
            f"Error inesperado extrayendo videos: {e}",
            exc_info=True
        )
        raise ProfileExtractionError(
            f"Error inesperado: {str(e)}",
            error_code="UNEXPECTED_ERROR",
            details={"username": username}
        ) from e
    
    return videos
"""

# CÓDIGO REFACTORIZADO
"""
from ..utils.collection_helpers import safe_map
from ..utils.dict_helpers import extract_fields
from ..utils.error_handling_helpers import handle_errors
from ..core.exceptions import ProfileExtractionError, ConnectorError

@handle_errors(
    "extract_tiktok_videos",
    error_types=(ConnectorError,),
    default_error=ProfileExtractionError
)
async def _extract_tiktok_videos(
    self,
    username: str
) -> List[VideoContent]:
    max_videos = self.settings.max_videos_per_profile
    
    video_list = await self.tiktok_connector.get_videos(
        username,
        limit=max_videos
    )
    
    # Usar safe_map para procesar videos de forma segura
    videos = safe_map(
        video_list,
        lambda video_data: VideoContent(
            **extract_fields(
                video_data,
                [
                    "video_id", "url", "title", "description", "duration",
                    "views", "likes", "comments", "created_at"
                ],
                defaults={"hashtags": [], "metadata": video_data}
            )
        ),
        operation="process_tiktok_video"
    )
    
    return videos
"""

# MEJORAS:
# - Reducido de ~65 líneas a ~25 líneas (62% reducción)
# - Manejo de errores automático con decorador
# - Procesamiento seguro de items individuales
# - Extracción de campos más clara

# ============================================================================
# REFACTORIZACIÓN REAL #2: services/identity_analyzer.py
# Método: _consolidate_content
# ============================================================================

# CÓDIGO ORIGINAL (líneas 174-211)
"""
def _consolidate_content(
    self,
    tiktok_profile: Optional[SocialProfile],
    instagram_profile: Optional[SocialProfile],
    youtube_profile: Optional[SocialProfile],
) -> dict:
    content = {
        "videos": [],
        "posts": [],
        "comments": [],
        "texts": []
    }
    
    if tiktok_profile:
        content["videos"].extend(tiktok_profile.videos)
        for video in tiktok_profile.videos:
            if video.transcript:
                content["texts"].append(video.transcript)
            if video.description:
                content["texts"].append(video.description)
    
    if instagram_profile:
        content["posts"].extend(instagram_profile.posts)
        for post in instagram_profile.posts:
            if post.caption:
                content["texts"].append(post.caption)
        content["comments"].extend(instagram_profile.comments)
    
    if youtube_profile:
        content["videos"].extend(youtube_profile.videos)
        for video in youtube_profile.videos:
            if video.transcript:
                content["texts"].append(video.transcript)
            if video.description:
                content["texts"].append(video.description)
    
    return content
"""

# CÓDIGO REFACTORIZADO
"""
from ..utils.data_consolidation_helpers import (
    consolidate_lists,
    extract_text_fields,
    merge_content_dicts
)

def _consolidate_content(
    self,
    tiktok_profile: Optional[SocialProfile],
    instagram_profile: Optional[SocialProfile],
    youtube_profile: Optional[SocialProfile],
) -> dict:
    # Consolidar videos
    all_videos = consolidate_lists(
        tiktok_profile.videos if tiktok_profile else None,
        youtube_profile.videos if youtube_profile else None
    )
    
    # Consolidar posts
    all_posts = consolidate_lists(
        instagram_profile.posts if instagram_profile else None
    )
    
    # Consolidar comments
    all_comments = consolidate_lists(
        instagram_profile.comments if instagram_profile else None
    )
    
    # Extraer textos de videos
    video_texts = extract_text_fields(
        all_videos,
        [
            lambda v: v.transcript,
            lambda v: v.description
        ]
    )
    
    # Extraer textos de posts
    post_texts = extract_text_fields(
        all_posts,
        [lambda p: p.caption]
    )
    
    # Combinar todos los textos
    all_texts = video_texts + post_texts
    
    return {
        "videos": all_videos,
        "posts": all_posts,
        "comments": all_comments,
        "texts": all_texts
    }
"""

# MEJORAS:
# - Reducido de ~38 líneas a ~35 líneas (pero mucho más claro)
# - Lógica de consolidación reutilizable
# - Extracción de texto más declarativa
# - Fácil agregar nuevas fuentes

# ============================================================================
# REFACTORIZACIÓN REAL #3: services/profile_extractor.py
# Método: extract_multiple_profiles
# ============================================================================

# CÓDIGO ORIGINAL (líneas 383-459)
"""
async def extract_multiple_profiles(
    self,
    tiktok_usernames: Optional[List[str]] = None,
    instagram_usernames: Optional[List[str]] = None,
    youtube_channels: Optional[List[str]] = None,
    use_cache: bool = True,
    max_concurrent: int = 5
) -> Dict[str, List[SocialProfile]]:
    results = {
        "tiktok": [],
        "instagram": [],
        "youtube": []
    }
    
    tasks = []
    
    # TikTok profiles
    if tiktok_usernames:
        for username in tiktok_usernames:
            tasks.append(self.extract_tiktok_profile(username, use_cache))
    
    # Instagram profiles
    if instagram_usernames:
        for username in instagram_usernames:
            tasks.append(self.extract_instagram_profile(username, use_cache))
    
    # YouTube channels
    if youtube_channels:
        for channel_id in youtube_channels:
            tasks.append(self.extract_youtube_profile(channel_id, use_cache))
    
    # Procesar en paralelo con límite de concurrencia
    if tasks:
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def bounded_extract(task):
            async with semaphore:
                try:
                    return await task
                except Exception as e:
                    self.logger.error(f"Error en extracción paralela: {e}")
                    return None
        
        completed = await asyncio.gather(*[bounded_extract(task) for task in tasks])
        
        # Organizar resultados por plataforma
        idx = 0
        if tiktok_usernames:
            results["tiktok"] = [p for p in completed[idx:idx+len(tiktok_usernames)] if p]
            idx += len(tiktok_usernames)
        if instagram_usernames:
            results["instagram"] = [p for p in completed[idx:idx+len(instagram_usernames)] if p]
            idx += len(instagram_usernames)
        if youtube_channels:
            results["youtube"] = [p for p in completed[idx:idx+len(youtube_channels)] if p]
    
    self.logger.info(
        f"Extracción paralela completada: "
        f"{len(results['tiktok'])} TikTok, "
        f"{len(results['instagram'])} Instagram, "
        f"{len(results['youtube'])} YouTube"
    )
    
    return results
"""

# CÓDIGO REFACTORIZADO
"""
from ..utils.async_helpers import safe_map_async
from ..utils.collection_helpers import partition

async def extract_multiple_profiles(
    self,
    tiktok_usernames: Optional[List[str]] = None,
    instagram_usernames: Optional[List[str]] = None,
    youtube_channels: Optional[List[str]] = None,
    use_cache: bool = True,
    max_concurrent: int = 5
) -> Dict[str, List[SocialProfile]]:
    results = {
        "tiktok": [],
        "instagram": [],
        "youtube": []
    }
    
    # Extraer perfiles de TikTok
    if tiktok_usernames:
        results["tiktok"] = await safe_map_async(
            tiktok_usernames,
            lambda username: self.extract_tiktok_profile(username, use_cache),
            max_concurrent=max_concurrent,
            operation="extract_tiktok_profiles"
        )
    
    # Extraer perfiles de Instagram
    if instagram_usernames:
        results["instagram"] = await safe_map_async(
            instagram_usernames,
            lambda username: self.extract_instagram_profile(username, use_cache),
            max_concurrent=max_concurrent,
            operation="extract_instagram_profiles"
        )
    
    # Extraer canales de YouTube
    if youtube_channels:
        results["youtube"] = await safe_map_async(
            youtube_channels,
            lambda channel_id: self.extract_youtube_profile(channel_id, use_cache),
            max_concurrent=max_concurrent,
            operation="extract_youtube_profiles"
        )
    
    self.logger.info(
        f"Extracción paralela completada: "
        f"{len(results['tiktok'])} TikTok, "
        f"{len(results['instagram'])} Instagram, "
        f"{len(results['youtube'])} YouTube"
    )
    
    return results
"""

# MEJORAS:
# - Reducido de ~77 líneas a ~40 líneas (48% reducción)
# - Manejo de concurrencia más simple
# - Manejo de errores automático
# - Código más declarativo

# ============================================================================
# REFACTORIZACIÓN REAL #4: services/storage_service.py
# Método: _save_social_profile
# ============================================================================

# CÓDIGO ORIGINAL (líneas 112-155)
"""
def _save_social_profile(self, db: Session, identity_id: str, profile: SocialProfile):
    existing = db.query(SocialProfileModel).filter_by(
        identity_profile_id=identity_id,
        platform=profile.platform.value
    ).first()
    
    # Serializar contenido
    videos_data = [v.model_dump() for v in profile.videos] if profile.videos else None
    posts_data = [p.model_dump() for p in profile.posts] if profile.posts else None
    comments_data = [c.model_dump() for c in profile.comments] if profile.comments else None
    
    if existing:
        existing.username = profile.username
        existing.display_name = profile.display_name
        existing.bio = profile.bio
        existing.profile_image_url = profile.profile_image_url
        existing.followers_count = profile.followers_count
        existing.following_count = profile.following_count
        existing.posts_count = profile.posts_count
        existing.videos_data = videos_data
        existing.posts_data = posts_data
        existing.comments_data = comments_data
        existing.metadata = profile.metadata
        existing.updated_at = datetime.utcnow()
    else:
        social_model = SocialProfileModel(
            id=str(uuid.uuid4()),
            identity_profile_id=identity_id,
            platform=profile.platform.value,
            username=profile.username,
            display_name=profile.display_name,
            bio=profile.bio,
            profile_image_url=profile.profile_image_url,
            followers_count=profile.followers_count,
            following_count=profile.following_count,
            posts_count=profile.posts_count,
            videos_data=videos_data,
            posts_data=posts_data,
            comments_data=comments_data,
            metadata=profile.metadata,
            extracted_at=profile.extracted_at
        )
        db.add(social_model)
"""

# CÓDIGO REFACTORIZADO
"""
from ..db.model_helpers import upsert_model
from ..utils.serialization_helpers import serialize_models
from ..utils.id_helpers import generate_id
from ..utils.condition_helpers import if_none

def _save_social_profile(self, db: Session, identity_id: str, profile: SocialProfile):
    # Serializar contenido usando helpers
    videos_data = serialize_models(profile.videos) if profile.videos else None
    posts_data = serialize_models(profile.posts) if profile.posts else None
    comments_data = serialize_models(profile.comments) if profile.comments else None
    
    # Upsert usando helper
    upsert_model(
        db,
        SocialProfileModel,
        identifier={
            "identity_profile_id": identity_id,
            "platform": profile.platform.value
        },
        update_data={
            "username": profile.username,
            "display_name": profile.display_name,
            "bio": profile.bio,
            "profile_image_url": profile.profile_image_url,
            "followers_count": profile.followers_count,
            "following_count": profile.following_count,
            "posts_count": profile.posts_count,
            "videos_data": videos_data,
            "posts_data": posts_data,
            "comments_data": comments_data,
            "metadata": profile.metadata
        },
        create_data={
            "id": generate_id("social_profile"),
            "extracted_at": profile.extracted_at
        }
    )
"""

# MEJORAS:
# - Reducido de ~44 líneas a ~25 líneas (43% reducción)
# - Upsert automático
# - Serialización consistente
# - Timestamps automáticos

# ============================================================================
# REFACTORIZACIÓN REAL #5: services/identity_analyzer.py
# Método: _determine_username
# ============================================================================

# CÓDIGO ORIGINAL (probablemente similar a este patrón)
"""
def _determine_username(
    self,
    tiktok_profile: Optional[SocialProfile],
    instagram_profile: Optional[SocialProfile],
    youtube_profile: Optional[SocialProfile]
) -> str:
    if tiktok_profile and tiktok_profile.username:
        return tiktok_profile.username
    elif instagram_profile and instagram_profile.username:
        return instagram_profile.username
    elif youtube_profile and youtube_profile.username:
        return youtube_profile.username
    else:
        return "unknown"
"""

# CÓDIGO REFACTORIZADO
"""
from ..utils.condition_helpers import first_not_none, coalesce

def _determine_username(
    self,
    tiktok_profile: Optional[SocialProfile],
    instagram_profile: Optional[SocialProfile],
    youtube_profile: Optional[SocialProfile]
) -> str:
    return coalesce(
        tiktok_profile.username if tiktok_profile else None,
        instagram_profile.username if instagram_profile else None,
        youtube_profile.username if youtube_profile else None,
        default="unknown"
    )
"""

# MEJORAS:
# - Reducido de ~10 líneas a ~6 líneas (40% reducción)
# - Más declarativo
# - Fácil agregar más fuentes

# ============================================================================
# RESUMEN DE REFACTORIZACIONES REALES
# ============================================================================

"""
REFACTORIZACIÓN #1 (_extract_tiktok_videos):
- Líneas: 65 → 25 (62% reducción)
- Helpers usados: safe_map, extract_fields, handle_errors
- Patrones: loop con try/except, extracción de campos

REFACTORIZACIÓN #2 (_consolidate_content):
- Líneas: 38 → 35 (8% reducción, pero mucho más claro)
- Helpers usados: consolidate_lists, extract_text_fields
- Patrones: consolidación de múltiples fuentes

REFACTORIZACIÓN #3 (extract_multiple_profiles):
- Líneas: 77 → 40 (48% reducción)
- Helpers usados: safe_map_async
- Patrones: procesamiento paralelo con manejo de errores

REFACTORIZACIÓN #4 (_save_social_profile):
- Líneas: 44 → 25 (43% reducción)
- Helpers usados: upsert_model, serialize_models, generate_id
- Patrones: upsert, serialización

REFACTORIZACIÓN #5 (_determine_username):
- Líneas: 10 → 6 (40% reducción)
- Helpers usados: coalesce
- Patrones: múltiples fallbacks

TOTAL ESTIMADO:
- Líneas reducidas: ~234 → ~131 (44% reducción promedio)
- Helpers diferentes usados: 10
- Código más mantenible y claro
"""








