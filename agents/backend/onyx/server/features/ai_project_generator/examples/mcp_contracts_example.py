"""
Ejemplo de uso de contratos MCP
================================

Muestra cómo usar ContextFrame y PromptFrame para
estandarizar entrada/salida con el modelo.
"""

from mcp_server.contracts import FrameSerializer, ContextFrame, PromptFrame


def example_context_frame():
    """Ejemplo básico de ContextFrame"""
    print("=== Ejemplo: ContextFrame ===")
    
    # Crear frame de contexto
    frame = FrameSerializer.create_context_frame(
        content="def hello():\n    print('Hello, World!')",
        context_type="code",
        source="filesystem",
        max_tokens=1000,
    )
    
    print(f"Frame ID: {frame.frame_id}")
    print(f"Content: {frame.content[:50]}...")
    print(f"Tokens: {frame.token_count}")
    print(f"Within limits: {frame.is_within_limits()}")
    
    # Serializar
    json_str = FrameSerializer.serialize_context_frame(frame, format="json")
    print(f"\nSerialized (first 200 chars):\n{json_str[:200]}...")
    
    # Deserializar
    frame_restored = FrameSerializer.deserialize_context_frame(json_str, format="json")
    print(f"\nRestored frame ID: {frame_restored.frame_id}")
    print(f"Match: {frame.frame_id == frame_restored.frame_id}")


def example_prompt_frame():
    """Ejemplo básico de PromptFrame"""
    print("\n=== Ejemplo: PromptFrame ===")
    
    # Crear contexto
    context1 = FrameSerializer.create_context_frame(
        content="Project structure:\n- src/\n- tests/\n- docs/",
        context_type="text",
        source="filesystem",
    )
    
    context2 = FrameSerializer.create_context_frame(
        content="Requirements: Python 3.11+, FastAPI",
        context_type="text",
        source="filesystem",
    )
    
    # Crear prompt con contexto
    prompt = FrameSerializer.create_prompt_frame(
        system_prompt="You are a helpful AI assistant for code generation.",
        user_prompt="Generate a complete Python project structure based on the context.",
        context_frames=[context1, context2],
        temperature=0.7,
        max_tokens=2048,
    )
    
    print(f"Prompt ID: {prompt.prompt_id}")
    print(f"User prompt: {prompt.user_prompt}")
    print(f"Context frames: {len(prompt.context_frames)}")
    print(f"Total context tokens: {prompt.get_total_context_tokens()}")
    print(f"Within limits: {prompt.is_within_limits()}")
    
    # Serializar
    json_str = FrameSerializer.serialize_prompt_frame(prompt, format="json")
    print(f"\nSerialized (first 300 chars):\n{json_str[:300]}...")


def example_multiple_frames():
    """Ejemplo con múltiples frames relacionados"""
    print("\n=== Ejemplo: Múltiples Frames Relacionados ===")
    
    # Frame padre
    parent = FrameSerializer.create_context_frame(
        content="Main project configuration",
        context_type="text",
        source="filesystem",
    )
    
    # Frames hijos
    child1 = FrameSerializer.create_context_frame(
        content="Backend configuration",
        context_type="text",
        source="filesystem",
    )
    child1.parent_frame_id = parent.frame_id
    
    child2 = FrameSerializer.create_context_frame(
        content="Frontend configuration",
        context_type="text",
        source="filesystem",
    )
    child2.parent_frame_id = parent.frame_id
    
    # Relacionar frames
    parent.related_frames = [child1.frame_id, child2.frame_id]
    
    print(f"Parent frame: {parent.frame_id}")
    print(f"Child frames: {[child1.frame_id, child2.frame_id]}")
    print(f"Related frames: {parent.related_frames}")


def example_validation():
    """Ejemplo de validación de límites"""
    print("\n=== Ejemplo: Validación de Límites ===")
    
    # Frame dentro de límites
    small_frame = FrameSerializer.create_context_frame(
        content="Small content",
        context_type="text",
        max_tokens=1000,
    )
    print(f"Small frame within limits: {small_frame.is_within_limits()}")
    
    # Frame grande (simulado)
    large_content = "x" * 50000  # ~12,500 tokens
    large_frame = FrameSerializer.create_context_frame(
        content=large_content,
        context_type="text",
        max_tokens=4096,  # Límite menor
    )
    print(f"Large frame within limits: {large_frame.is_within_limits()}")
    print(f"Large frame tokens: {large_frame.token_count}")
    print(f"Large frame max_tokens: {large_frame.max_tokens}")


if __name__ == "__main__":
    example_context_frame()
    example_prompt_frame()
    example_multiple_frames()
    example_validation()

