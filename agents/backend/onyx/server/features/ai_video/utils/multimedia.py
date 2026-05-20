from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
import msgspec
from typing import List, Optional

from typing import Any, List, Dict, Optional
import logging
import asyncio
class MultimediaInfo(msgspec.Struct, frozen=True, slots=True):
    """
    Información multimedia y de IA asociada al video.
    """
    thumbnails: List[str] = msgspec.field(default_factory=list)
    embedding: Optional[List[float]] = None
    transcript: Optional[str] = None
    detected_language: Optional[str] = None
    speech_analysis: Optional[dict] = None
    image_analysis: Optional[dict] = None
    ocr_text: Optional[str] = None
    emotion_analysis: Optional[dict] = None

    def with_thumbnails(self, thumbnails: List[str]) -> 'MultimediaInfo':
        return self.update(thumbnails=thumbnails)

    def with_embedding(self, embedding: List[float]) -> 'MultimediaInfo':
        return self.update(embedding=embedding)

    def with_transcript(self, transcript: str) -> 'MultimediaInfo':
        return self.update(transcript=transcript)

    def with_detected_language(self, lang: str) -> 'MultimediaInfo':
        return self.update(detected_language=lang)

    def with_speech_analysis(self, analysis: dict) -> 'MultimediaInfo':
        return self.update(speech_analysis=analysis)

    def with_image_analysis(self, analysis: dict) -> 'MultimediaInfo':
        return self.update(image_analysis=analysis)

    def with_ocr_text(self, text: str) -> 'MultimediaInfo':
        return self.update(ocr_text=text)

    def with_emotion_analysis(self, analysis: dict) -> 'MultimediaInfo':
        return self.update(emotion_analysis=analysis) 