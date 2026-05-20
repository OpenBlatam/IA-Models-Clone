"""
Faceless Video Core - High Performance Rust Core

Este módulo proporciona implementaciones de alto rendimiento en Rust para:
- Procesamiento de video (composición, optimización, efectos)
- Criptografía (encriptación/desencriptación AES-GCM)
- Procesamiento de texto (segmentación, subtítulos)
- Procesamiento de imágenes (watermarking, efectos)
- Procesamiento por lotes (paralelización)

Ejemplo de uso:
    
    from faceless_video_core import video, crypto, text, image, batch

    # Procesamiento de video
    processor = video.VideoProcessor()
    processor.optimize_video("input.mp4", "high")

    # Encriptación
    crypto_service = crypto.CryptoService()
    encrypted = crypto_service.encrypt("datos sensibles")

    # Procesamiento de texto
    text_proc = text.TextProcessor()
    segments = text_proc.process_script("Mi script...", "es")

    # Procesamiento de imágenes
    img_proc = image.ImageProcessor()
    img_proc.add_text_watermark("image.png", config)
"""

from .faceless_video_core import (
    video,
    crypto,
    text,
    image,
    batch,
    __version__,
    __author__,
)

# Video processing exports
VideoProcessor = video.VideoProcessor
VideoConfig = video.VideoConfig
FrameSequence = video.FrameSequence
TransitionEffect = video.TransitionEffect

# Crypto exports
CryptoService = crypto.CryptoService
HashResult = crypto.HashResult

# Text processing exports
TextProcessor = text.TextProcessor
TextSegment = text.TextSegment
SubtitleEntry = text.SubtitleEntry
SubtitleStyle = text.SubtitleStyle

# Image processing exports
ImageProcessor = image.ImageProcessor
WatermarkConfig = image.WatermarkConfig
ColorGrading = image.ColorGrading

# Batch processing exports
BatchProcessor = batch.BatchProcessor
BatchJob = batch.BatchJob
BatchResult = batch.BatchResult

__all__ = [
    # Modules
    "video",
    "crypto",
    "text",
    "image",
    "batch",
    # Video
    "VideoProcessor",
    "VideoConfig",
    "FrameSequence",
    "TransitionEffect",
    # Crypto
    "CryptoService",
    "HashResult",
    # Text
    "TextProcessor",
    "TextSegment",
    "SubtitleEntry",
    "SubtitleStyle",
    # Image
    "ImageProcessor",
    "WatermarkConfig",
    "ColorGrading",
    # Batch
    "BatchProcessor",
    "BatchJob",
    "BatchResult",
    # Meta
    "__version__",
    "__author__",
]




