import base64
from io import BytesIO

from langchain_core.messages import BaseMessage
from langchain_core.messages import HumanMessage
from langchain_core.messages import SystemMessage
from PIL import Image


def _encode_image_for_llm_prompt(image_data: bytes) -> str:
    """Getting the base64 string."""
    base64_encoded_data = base64.b64encode(image_data).decode("utf-8")

    return f"data:image/jpeg;base64,{base64_encoded_data}"


def _resize_image_if_needed(image_data: bytes, max_size_mb: int = 20) -> bytes:
    """Resize image if it's larger than the specified max size in MB."""
    max_size_bytes = max_size_mb * 1024 * 1024

    if len(image_data) > max_size_bytes:
        with Image.open(BytesIO(image_data)) as img:
            # Reduce dimensions for better size reduction
            img.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
            output = BytesIO()

            # Save with lower quality for compression
            img.save(output, format="JPEG", quality=85)
            resized_data = output.getvalue()

            return resized_data

    return image_data