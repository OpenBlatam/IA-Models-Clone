"""
Tests básicos para el editor de contenido
"""

import pytest
import asyncio
from addition_removal_ai.core.editor import ContentEditor


@pytest.fixture
def editor():
    """Fixture para crear un editor"""
    return ContentEditor()


@pytest.mark.asyncio
async def test_add_content(editor):
    """Test agregar contenido"""
    content = "Texto original"
    addition = "Nuevo texto"
    
    result = await editor.add(content, addition, "end")
    
    assert result["success"] is True
    assert addition in result["content"]
    assert "validation" in result


@pytest.mark.asyncio
async def test_remove_content(editor):
    """Test eliminar contenido"""
    content = "Texto con elemento a eliminar"
    pattern = "elemento a eliminar"
    
    result = await editor.remove(content, pattern)
    
    assert result["success"] is True
    assert pattern not in result["content"]


@pytest.mark.asyncio
async def test_batch_add(editor):
    """Test operación batch de adición"""
    content = "Texto original"
    additions = [
        {"addition": "Primero", "position": "start"},
        {"addition": "Segundo", "position": "end"}
    ]
    
    result = await editor.batch_add(content, additions)
    
    assert result["success"] is True
    assert "Primero" in result["content"]
    assert "Segundo" in result["content"]


def test_format_detection(editor):
    """Test detección de formato"""
    markdown = "# Título\n\nContenido"
    json_content = '{"key": "value"}'
    plain = "Texto plano"
    
    assert editor.formatter.detect_format(markdown) == "markdown"
    assert editor.formatter.detect_format(json_content) == "json"
    assert editor.formatter.detect_format(plain) == "plain_text"


def test_diff_computation(editor):
    """Test cálculo de diferencias"""
    original = "Texto original"
    modified = "Texto modificado"
    
    diff = editor.diff.compute_diff(original, modified)
    
    assert "changes" in diff
    assert "summary" in diff
    assert len(diff["changes"]) > 0


def test_undo_redo(editor):
    """Test sistema undo/redo"""
    content1 = "Estado 1"
    content2 = "Estado 2"
    
    editor.undo_redo.save_state(content1, "add")
    editor.undo_redo.save_state(content2, "add")
    
    assert editor.undo_redo.can_undo() is True
    
    previous = editor.undo_redo.undo(content2)
    assert previous is not None
    assert previous["content"] == content1
    
    assert editor.undo_redo.can_redo() is True






