"""
Frame Serializer - Serializador de frames MCP
==============================================

Helpers para serializar/deserializar contextos hacia MCP.
"""

import json
import base64
from typing import Any, Dict, Optional
from datetime import datetime

from .models import ContextFrame, PromptFrame, FrameMetadata


class FrameSerializer:
    """
    Serializador de frames MCP
    
    Soporta múltiples formatos:
    - JSON
    - Base64 (para binarios)
    - Compact (sin espacios)
    """
    
    @staticmethod
    def serialize_context_frame(frame: ContextFrame, format: str = "json") -> str:
        """
        Serializa un ContextFrame.
        
        Args:
            frame: Frame a serializar (debe ser instancia de ContextFrame)
            format: Formato (json, compact, base64)
            
        Returns:
            String serializado
            
        Raises:
            ValueError: Si frame es inválido o format no es soportado
            TypeError: Si frame no es una instancia de ContextFrame
        """
        if not isinstance(frame, ContextFrame):
            raise TypeError(f"frame must be an instance of ContextFrame, got {type(frame)}")
        if not format or not isinstance(format, str):
            raise ValueError("format must be a non-empty string")
        
        format = format.lower().strip()
        if format not in ["json", "compact", "base64"]:
            raise ValueError(f"Unsupported format: {format}. Supported: json, compact, base64")
        
        try:
            data = frame.to_dict()
            
            if format == "json":
                return json.dumps(data, indent=2, default=str, ensure_ascii=False)
            elif format == "compact":
                return json.dumps(data, separators=(",", ":"), default=str, ensure_ascii=False)
            elif format == "base64":
                json_str = json.dumps(data, default=str, ensure_ascii=False)
                return base64.b64encode(json_str.encode("utf-8")).decode("utf-8")
        except (TypeError, ValueError) as e:
            raise ValueError(f"Failed to serialize ContextFrame: {e}") from e
    
    @staticmethod
    def deserialize_context_frame(data: str, format: str = "json") -> ContextFrame:
        """
        Deserializa un ContextFrame.
        
        Args:
            data: String serializado (debe ser no vacío)
            format: Formato (json, compact, base64)
            
        Returns:
            ContextFrame deserializado
            
        Raises:
            ValueError: Si data es inválido, format no es soportado, o deserialización falla
            TypeError: Si data no es string
        """
        if not isinstance(data, str):
            raise TypeError(f"data must be a string, got {type(data)}")
        if not data or not data.strip():
            raise ValueError("data cannot be empty")
        if not format or not isinstance(format, str):
            raise ValueError("format must be a non-empty string")
        
        format = format.lower().strip()
        if format not in ["json", "compact", "base64"]:
            raise ValueError(f"Unsupported format: {format}. Supported: json, compact, base64")
        
        try:
            if format == "base64":
                try:
                    data = base64.b64decode(data).decode("utf-8")
                except (ValueError, UnicodeDecodeError) as e:
                    raise ValueError(f"Failed to decode base64 data: {e}") from e
                format = "json"
            
            if format in ["json", "compact"]:
                try:
                    dict_data = json.loads(data)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON data: {e}") from e
                
                try:
                    return ContextFrame(**dict_data)
                except Exception as e:
                    raise ValueError(f"Failed to create ContextFrame from data: {e}") from e
        except ValueError:
            raise
        except Exception as e:
            raise ValueError(f"Unexpected error deserializing ContextFrame: {e}") from e
    
    @staticmethod
    def serialize_prompt_frame(frame: PromptFrame, format: str = "json") -> str:
        """
        Serializa un PromptFrame.
        
        Args:
            frame: Frame a serializar (debe ser instancia de PromptFrame)
            format: Formato (json, compact, base64)
            
        Returns:
            String serializado
            
        Raises:
            ValueError: Si frame es inválido o format no es soportado
            TypeError: Si frame no es una instancia de PromptFrame
        """
        if not isinstance(frame, PromptFrame):
            raise TypeError(f"frame must be an instance of PromptFrame, got {type(frame)}")
        if not format or not isinstance(format, str):
            raise ValueError("format must be a non-empty string")
        
        format = format.lower().strip()
        if format not in ["json", "compact", "base64"]:
            raise ValueError(f"Unsupported format: {format}. Supported: json, compact, base64")
        
        try:
            data = frame.to_dict()
            
            if format == "json":
                return json.dumps(data, indent=2, default=str, ensure_ascii=False)
            elif format == "compact":
                return json.dumps(data, separators=(",", ":"), default=str, ensure_ascii=False)
            elif format == "base64":
                json_str = json.dumps(data, default=str, ensure_ascii=False)
                return base64.b64encode(json_str.encode("utf-8")).decode("utf-8")
        except (TypeError, ValueError) as e:
            raise ValueError(f"Failed to serialize PromptFrame: {e}") from e
    
    @staticmethod
    def deserialize_prompt_frame(data: str, format: str = "json") -> PromptFrame:
        """
        Deserializa un PromptFrame.
        
        Args:
            data: String serializado (debe ser no vacío)
            format: Formato (json, compact, base64)
            
        Returns:
            PromptFrame deserializado
            
        Raises:
            ValueError: Si data es inválido, format no es soportado, o deserialización falla
            TypeError: Si data no es string
        """
        if not isinstance(data, str):
            raise TypeError(f"data must be a string, got {type(data)}")
        if not data or not data.strip():
            raise ValueError("data cannot be empty")
        if not format or not isinstance(format, str):
            raise ValueError("format must be a non-empty string")
        
        format = format.lower().strip()
        if format not in ["json", "compact", "base64"]:
            raise ValueError(f"Unsupported format: {format}. Supported: json, compact, base64")
        
        try:
            if format == "base64":
                try:
                    data = base64.b64decode(data).decode("utf-8")
                except (ValueError, UnicodeDecodeError) as e:
                    raise ValueError(f"Failed to decode base64 data: {e}") from e
                format = "json"
            
            if format in ["json", "compact"]:
                try:
                    dict_data = json.loads(data)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON data: {e}") from e
                
                try:
                    return PromptFrame(**dict_data)
                except Exception as e:
                    raise ValueError(f"Failed to create PromptFrame from data: {e}") from e
        except ValueError:
            raise
        except Exception as e:
            raise ValueError(f"Unexpected error deserializing PromptFrame: {e}") from e
    
    @staticmethod
    def create_context_frame(
        content: str,
        context_type: str = "text",
        source: str = "mcp",
        max_tokens: int = 4096,
        **kwargs
    ) -> ContextFrame:
        """
        Crea un ContextFrame con valores por defecto
        
        Args:
            content: Contenido del contexto
            context_type: Tipo de contexto
            source: Fuente del contexto
            max_tokens: Límite máximo de tokens
            **kwargs: Campos adicionales
            
        Returns:
            ContextFrame creado
        """
        import uuid
        
        metadata = FrameMetadata(
            timestamp=datetime.utcnow(),
            source=source,
            version="1.0",
        )
        
        frame = ContextFrame(
            frame_id=str(uuid.uuid4()),
            content=content,
            context_type=context_type,
            max_tokens=max_tokens,
            metadata=metadata,
            fields=kwargs,
        )
        
        # Calcular tokens
        frame.token_count = frame.estimate_tokens()
        
        return frame
    
    @staticmethod
    def create_prompt_frame(
        user_prompt: str,
        system_prompt: Optional[str] = None,
        context_frames: Optional[list[ContextFrame]] = None,
        **kwargs
    ) -> PromptFrame:
        """
        Crea un PromptFrame con valores por defecto
        
        Args:
            user_prompt: Prompt del usuario
            system_prompt: Prompt del sistema (opcional)
            context_frames: Frames de contexto (opcional)
            **kwargs: Campos adicionales
            
        Returns:
            PromptFrame creado
        """
        import uuid
        
        metadata = FrameMetadata(
            timestamp=datetime.utcnow(),
            source="mcp",
            version="1.0",
        )
        
        frame = PromptFrame(
            prompt_id=str(uuid.uuid4()),
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            context_frames=context_frames or [],
            metadata=metadata,
            fields=kwargs,
        )
        
        return frame

