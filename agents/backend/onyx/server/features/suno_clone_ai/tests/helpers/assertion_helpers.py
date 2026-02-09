"""
Helpers para aserciones personalizadas
"""

import numpy as np
from typing import Dict, Any, List, Optional
from pathlib import Path


def assert_song_response_valid(response: Dict[str, Any]) -> None:
    """
    Verifica que una respuesta de canción sea válida
    
    Args:
        response: Diccionario de respuesta
    """
    assert "song_id" in response, "Missing song_id in response"
    assert "status" in response, "Missing status in response"
    assert "message" in response, "Missing message in response"
    
    assert isinstance(response["song_id"], str), "song_id must be string"
    assert isinstance(response["status"], str), "status must be string"
    assert isinstance(response["message"], str), "message must be string"
    
    valid_statuses = ["processing", "completed", "failed", "not_found"]
    assert response["status"] in valid_statuses, f"Invalid status: {response['status']}"


def assert_audio_files_equal(
    file1: Path,
    file2: Path,
    tolerance: float = 1e-5
) -> None:
    """
    Verifica que dos archivos de audio sean iguales
    
    Args:
        file1: Primer archivo
        file2: Segundo archivo
        tolerance: Tolerancia para comparación numérica
    """
    import soundfile as sf
    
    assert file1.exists(), f"File 1 not found: {file1}"
    assert file2.exists(), f"File 2 not found: {file2}"
    
    audio1, sr1 = sf.read(str(file1))
    audio2, sr2 = sf.read(str(file2))
    
    assert sr1 == sr2, f"Sample rates differ: {sr1} != {sr2}"
    assert len(audio1) == len(audio2), f"Lengths differ: {len(audio1)} != {len(audio2)}"
    
    if len(audio1) > 0:
        diff = np.abs(audio1 - audio2)
        max_diff = np.max(diff)
        assert max_diff < tolerance, f"Audio differs by {max_diff} > {tolerance}"


def assert_audio_processed(
    original: np.ndarray,
    processed: np.ndarray,
    should_differ: bool = True
) -> None:
    """
    Verifica que el audio haya sido procesado
    
    Args:
        original: Audio original
        processed: Audio procesado
        should_differ: Si True, verifica que sean diferentes
    """
    assert original is not None, "Original audio is None"
    assert processed is not None, "Processed audio is None"
    
    if should_differ:
        assert not np.array_equal(original, processed), "Audio was not processed"
    else:
        assert np.array_equal(original, processed), "Audio should not differ"


def assert_song_list_valid(
    song_list: List[Dict[str, Any]],
    min_count: int = 0,
    max_count: Optional[int] = None
) -> None:
    """
    Verifica que una lista de canciones sea válida
    
    Args:
        song_list: Lista de canciones
        min_count: Cantidad mínima esperada
        max_count: Cantidad máxima esperada
    """
    assert isinstance(song_list, list), "song_list must be a list"
    assert len(song_list) >= min_count, f"Too few songs: {len(song_list)} < {min_count}"
    
    if max_count is not None:
        assert len(song_list) <= max_count, f"Too many songs: {len(song_list)} > {max_count}"
    
    for song in song_list:
        assert "song_id" in song, "Missing song_id in song"
        assert isinstance(song["song_id"], str), "song_id must be string"


def assert_error_response(
    response: Dict[str, Any],
    expected_status_code: int,
    expected_detail: Optional[str] = None
) -> None:
    """
    Verifica que una respuesta de error sea válida
    
    Args:
        response: Diccionario de respuesta
        expected_status_code: Código de estado esperado
        expected_detail: Detalle esperado (opcional)
    """
    assert "detail" in response, "Missing detail in error response"
    
    if expected_detail:
        assert expected_detail in response["detail"], f"Detail mismatch: {response['detail']}"


def assert_metadata_valid(metadata: Dict[str, Any]) -> None:
    """
    Verifica que metadata sea válida
    
    Args:
        metadata: Diccionario de metadata
    """
    assert isinstance(metadata, dict), "Metadata must be a dictionary"
    # Metadata puede estar vacía, así que solo verificamos que sea dict


def assert_file_exists_and_valid(file_path: Path, min_size: int = 1) -> None:
    """
    Verifica que un archivo exista y sea válido
    
    Args:
        file_path: Ruta del archivo
        min_size: Tamaño mínimo en bytes
    """
    assert file_path.exists(), f"File does not exist: {file_path}"
    assert file_path.is_file(), f"Path is not a file: {file_path}"
    assert file_path.stat().st_size >= min_size, f"File too small: {file_path.stat().st_size} < {min_size}"

