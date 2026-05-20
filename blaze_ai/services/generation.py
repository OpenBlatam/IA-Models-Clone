from __future__ import annotations

from typing import Dict, List

from .brand import BrandVoiceService


class ContentGenerationService:
    """Generates lightweight text for blog posts, posts, and emails."""

    def __init__(self, brand_voice: BrandVoiceService) -> None:
        self.brand_voice = brand_voice

    def social_post(self, topic: str, brand_name: str) -> str:
        text = f"{topic}: ideas clave para hoy. 🚀 #{topic.replace(' ', '')}"
        return self.brand_voice.apply_voice(text, brand_name)

    def email(self, subject: str, bullet_points: List[str], brand_name: str) -> Dict[str, str]:
        body = "\n".join([f"- {b}" for b in bullet_points])
        styled = self.brand_voice.apply_voice(body, brand_name)
        return {"subject": subject, "body": styled}

    def blog_outline(self, title: str, sections: List[str], brand_name: str) -> Dict[str, List[str]]:
        _ = self.brand_voice.get_voice(brand_name)
        return {"title": title, "sections": sections}


