"""
Wrappers de compatibilidad para integrar el módulo Rust con el sistema Python existente.

Este módulo proporciona clases que mantienen la misma interfaz que los servicios
Python originales pero utilizan las implementaciones Rust internamente.
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

try:
    from .faceless_video_core import (
        video as rust_video,
        crypto as rust_crypto,
        text as rust_text,
        image as rust_image,
        batch as rust_batch,
    )
    RUST_AVAILABLE = True
    logger.info("Rust core module loaded successfully")
except ImportError as e:
    RUST_AVAILABLE = False
    logger.warning(f"Rust core not available, using Python fallback: {e}")


class VideoCompositorWrapper:
    """
    Wrapper compatible con el VideoCompositor Python existente.
    Utiliza Rust internamente cuando está disponible.
    """

    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = Path(output_dir) if output_dir else Path("/tmp/faceless_video/output")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if RUST_AVAILABLE:
            self._rust_processor = rust_video.VideoProcessor(
                output_dir=str(self.output_dir)
            )
        else:
            self._rust_processor = None

    async def composite_video(
        self,
        image_sequence: List[Dict[str, Any]],
        audio_path: str,
        subtitles: List[Dict[str, Any]],
        video_config: Dict[str, Any],
        subtitle_config: Dict[str, Any],
        output_path: Optional[Path] = None
    ) -> Path:
        """Composita video final desde componentes."""
        if not RUST_AVAILABLE:
            from services.video_compositor import VideoCompositor
            fallback = VideoCompositor(output_dir=str(self.output_dir))
            return await fallback.composite_video(
                image_sequence, audio_path, subtitles, video_config, subtitle_config, output_path
            )

        output = output_path or self.output_dir / "final_video.mp4"
        
        frames = [
            rust_video.FrameSequence(
                image_path=f["image_path"],
                duration=f["duration"],
                transition=f.get("transition"),
                effect=f.get("effect"),
            )
            for f in image_sequence
        ]
        
        config = rust_video.VideoConfig(
            width=int(video_config.get("resolution", "1920x1080").split("x")[0]),
            height=int(video_config.get("resolution", "1920x1080").split("x")[1]),
            fps=video_config.get("fps", 30),
        )
        
        video_path = self._rust_processor.create_video_from_images(frames, config)
        video_with_audio = self._rust_processor.add_audio_to_video(video_path, audio_path)
        
        if subtitles and subtitle_config.get("enabled", True):
            from .faceless_video_core import text as rust_text
            text_proc = rust_text.TextProcessor()
            
            subtitle_entries = []
            for i, sub in enumerate(subtitles):
                subtitle_entries.append(rust_text.SubtitleEntry(
                    index=i,
                    text=sub["text"],
                    start_time=sub["start_time"],
                    end_time=sub["end_time"],
                ))
            
            srt_path = str(self.output_dir / "subtitles.srt")
            text_proc.export_srt(subtitle_entries, srt_path)
            
            final_video = self._rust_processor.add_subtitles(
                video_with_audio,
                srt_path,
                font_size=subtitle_config.get("font_size", 48),
                font_color=subtitle_config.get("font_color", "#FFFFFF"),
                output_path=str(output),
            )
        else:
            final_video = video_with_audio
            if str(final_video) != str(output):
                import shutil
                shutil.move(final_video, output)
                final_video = str(output)

        return Path(final_video)


class VideoOptimizerWrapper:
    """Wrapper compatible con el VideoOptimizer Python existente."""

    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = Path(output_dir) if output_dir else Path("/tmp/faceless_video/optimized")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if RUST_AVAILABLE:
            self._rust_processor = rust_video.VideoProcessor(
                output_dir=str(self.output_dir)
            )
        else:
            self._rust_processor = None

    async def optimize_video(
        self,
        video_path: Path,
        quality: str = "high",
        target_size_mb: Optional[float] = None,
        output_path: Optional[Path] = None
    ) -> Path:
        """Optimiza video para tamaño y calidad."""
        if not RUST_AVAILABLE:
            from services.video_optimizer import VideoOptimizer
            fallback = VideoOptimizer(output_dir=str(self.output_dir))
            return await fallback.optimize_video(video_path, quality, target_size_mb, output_path)

        output = output_path or self.output_dir / f"optimized_{video_path.stem}.mp4"
        
        result = self._rust_processor.optimize_video(
            video_path=str(video_path),
            quality=quality,
            target_size_mb=target_size_mb,
            output_path=str(output),
        )
        
        return Path(result)

    async def generate_thumbnail(
        self,
        video_path: Path,
        output_path: Optional[Path] = None,
        time_offset: float = 1.0,
        width: int = 320,
        height: int = 180
    ) -> Path:
        """Genera thumbnail del video."""
        if not RUST_AVAILABLE:
            from services.video_optimizer import VideoOptimizer
            fallback = VideoOptimizer(output_dir=str(self.output_dir))
            return await fallback.generate_thumbnail(video_path, output_path, time_offset, width, height)

        output = output_path or self.output_dir / f"thumbnail_{video_path.stem}.jpg"
        
        result = self._rust_processor.generate_thumbnail(
            video_path=str(video_path),
            time_offset=time_offset,
            width=width,
            height=height,
            output_path=str(output),
        )
        
        return Path(result)


class EncryptionServiceWrapper:
    """Wrapper compatible con el EncryptionService Python existente."""

    def __init__(self, key: Optional[bytes] = None):
        if RUST_AVAILABLE:
            key_str = None
            if key:
                import base64
                key_str = base64.b64encode(key).decode()
            self._rust_crypto = rust_crypto.CryptoService(key=key_str)
        else:
            self._rust_crypto = None
            from services.security.encryption import EncryptionService
            self._fallback = EncryptionService(key=key)

    def encrypt(self, data: str) -> str:
        """Encripta datos."""
        if RUST_AVAILABLE:
            return self._rust_crypto.encrypt(data)
        return self._fallback.encrypt(data)

    def decrypt(self, encrypted_data: str) -> str:
        """Desencripta datos."""
        if RUST_AVAILABLE:
            return self._rust_crypto.decrypt(encrypted_data)
        return self._fallback.decrypt(encrypted_data)

    def encrypt_file(self, file_path: str, output_path: Optional[str] = None) -> str:
        """Encripta archivo."""
        if RUST_AVAILABLE:
            return self._rust_crypto.encrypt_file(file_path, output_path)
        return self._fallback.encrypt_file(file_path, output_path)

    def decrypt_file(self, encrypted_path: str, output_path: Optional[str] = None) -> str:
        """Desencripta archivo."""
        if RUST_AVAILABLE:
            return self._rust_crypto.decrypt_file(encrypted_path, output_path)
        return self._fallback.decrypt_file(encrypted_path, output_path)


class SubtitleGeneratorWrapper:
    """Wrapper compatible con el SubtitleGenerator Python existente."""

    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = Path(output_dir) if output_dir else Path("/tmp/faceless_video/subtitles")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if RUST_AVAILABLE:
            self._rust_text = rust_text.TextProcessor()
        else:
            self._rust_text = None

    def generate_subtitles(
        self,
        segments: List[Dict[str, Any]],
        config: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Genera entradas de subtítulos desde segmentos."""
        if not RUST_AVAILABLE:
            from services.subtitle_generator import SubtitleGenerator
            fallback = SubtitleGenerator(output_dir=str(self.output_dir))
            return fallback.generate_subtitles(segments, config)

        if not config.get("enabled", True):
            return []

        style = rust_text.SubtitleStyle(
            name=config.get("style", "modern"),
            font_size=config.get("font_size", 48),
            font_color=config.get("font_color", "#FFFFFF"),
            position=config.get("position", "bottom"),
            animation=config.get("animation", True),
        )

        text_segments = []
        for seg in segments:
            text_segments.append(rust_text.TextSegment(
                index=seg.get("index", 0),
                text=seg.get("text", ""),
                start_time=seg.get("start_time", 0.0),
                end_time=seg.get("end_time", 0.0),
                language=seg.get("language", "es"),
            ))

        subtitle_entries = self._rust_text.generate_subtitles(text_segments, style)

        return [
            {
                "index": sub.index,
                "text": sub.text,
                "start_time": sub.start_time,
                "end_time": sub.end_time,
                "duration": sub.end_time - sub.start_time,
                "style": config.get("style", "modern"),
                "font_size": config.get("font_size", 48),
                "font_color": config.get("font_color", "#FFFFFF"),
                "position": config.get("position", "bottom"),
                "animation": config.get("animation", True),
            }
            for sub in subtitle_entries
        ]

    def export_srt(self, subtitles: List[Dict[str, Any]], output_path: Path) -> Path:
        """Exporta subtítulos a formato SRT."""
        if RUST_AVAILABLE:
            entries = [
                rust_text.SubtitleEntry(
                    index=s["index"],
                    text=s["text"],
                    start_time=s["start_time"],
                    end_time=s["end_time"],
                )
                for s in subtitles
            ]
            self._rust_text.export_srt(entries, str(output_path))
            return output_path
        
        from services.subtitle_generator import SubtitleGenerator
        fallback = SubtitleGenerator(output_dir=str(self.output_dir))
        return fallback.export_srt(subtitles, output_path)

    def export_vtt(self, subtitles: List[Dict[str, Any]], output_path: Path) -> Path:
        """Exporta subtítulos a formato VTT."""
        if RUST_AVAILABLE:
            entries = [
                rust_text.SubtitleEntry(
                    index=s["index"],
                    text=s["text"],
                    start_time=s["start_time"],
                    end_time=s["end_time"],
                )
                for s in subtitles
            ]
            self._rust_text.export_vtt(entries, str(output_path))
            return output_path
        
        from services.subtitle_generator import SubtitleGenerator
        fallback = SubtitleGenerator(output_dir=str(self.output_dir))
        return fallback.export_vtt(subtitles, output_path)


class WatermarkServiceWrapper:
    """Wrapper compatible con el WatermarkService Python existente."""

    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = Path(output_dir) if output_dir else Path("/tmp/faceless_video/watermarked")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if RUST_AVAILABLE:
            self._rust_image = rust_image.ImageProcessor(output_dir=str(self.output_dir))
        else:
            self._rust_image = None

    async def add_watermark(
        self,
        video_path: Path,
        watermark_text: Optional[str] = None,
        watermark_image: Optional[Path] = None,
        position: str = "bottom-right",
        opacity: float = 0.7,
        size: float = 0.1,
        output_path: Optional[Path] = None
    ) -> Path:
        """Agrega watermark a video (primero a imágenes extraídas)."""
        if not RUST_AVAILABLE:
            from services.watermarking import WatermarkService
            fallback = WatermarkService(output_dir=str(self.output_dir))
            return await fallback.add_watermark(
                video_path, watermark_text, watermark_image, position, opacity, size, output_path
            )

        if watermark_text:
            config = rust_image.WatermarkConfig(
                text=watermark_text,
                position=position,
                opacity=opacity,
                size=size,
            )
            result = self._rust_image.add_text_watermark(str(video_path), config)
        elif watermark_image:
            config = rust_image.WatermarkConfig(
                image_path=str(watermark_image),
                position=position,
                opacity=opacity,
                size=size,
            )
            result = self._rust_image.add_image_watermark(str(video_path), config)
        else:
            return video_path

        return Path(result)


class ScriptProcessorWrapper:
    """Wrapper compatible con el ScriptProcessor Python existente."""

    def __init__(self):
        if RUST_AVAILABLE:
            self._rust_text = rust_text.TextProcessor()
        else:
            self._rust_text = None

    def process_script(self, script) -> List[Dict[str, Any]]:
        """Procesa script en segmentos."""
        if script.segments:
            return script.segments

        if not RUST_AVAILABLE:
            from services.script_processor import ScriptProcessor
            fallback = ScriptProcessor()
            return fallback.process_script(script)

        segments = self._rust_text.process_script(script.text, script.language)

        return [
            {
                "index": seg.index,
                "text": seg.text,
                "word_count": seg.word_count,
                "char_count": seg.char_count,
                "start_time": seg.start_time,
                "end_time": seg.end_time,
                "duration": seg.duration,
                "keywords": seg.keywords,
                "language": seg.language,
            }
            for seg in segments
        ]

    def estimate_total_duration(self, segments: List[Dict[str, Any]]) -> float:
        """Estima duración total."""
        if not segments:
            return 0.0
        return segments[-1].get("end_time", 0.0)


class BatchProcessorWrapper:
    """Wrapper compatible con el BatchProcessor Python existente."""

    def __init__(self, max_concurrent: int = 3):
        self.max_concurrent = max_concurrent
        
        if RUST_AVAILABLE:
            self._rust_batch = rust_batch.BatchProcessor(
                max_concurrent=max_concurrent,
                timeout_seconds=300
            )
        else:
            self._rust_batch = None

    async def process_batch(
        self,
        requests: List[Any],
        webhook_url: Optional[str] = None
    ) -> Dict[str, Any]:
        """Procesa múltiples requests en batch."""
        if not RUST_AVAILABLE:
            from services.batch_processor import BatchProcessor
            fallback = BatchProcessor(max_concurrent=self.max_concurrent)
            return await fallback.process_batch(requests, webhook_url)

        scripts = [r.script.text for r in requests]
        result = self._rust_batch.process_scripts_batch(scripts, "es")

        return {
            "batch_id": result.batch_id,
            "total": result.total,
            "started": result.completed,
            "failed": result.failed,
            "jobs": [
                {
                    "video_id": job.id,
                    "status": job.status,
                }
                for job in result.jobs
            ],
        }


def is_rust_available() -> bool:
    """Verifica si el módulo Rust está disponible."""
    return RUST_AVAILABLE


def get_rust_version() -> Optional[str]:
    """Obtiene la versión del módulo Rust."""
    if RUST_AVAILABLE:
        from .faceless_video_core import __version__
        return __version__
    return None




