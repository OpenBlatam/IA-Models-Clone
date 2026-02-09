"""I/O utilities."""

from pathlib import Path
from typing import Optional, List
import aiofiles


async def read_file_async(file_path: Path, encoding: str = 'utf-8') -> str:
    """
    Read file asynchronously.
    
    Args:
        file_path: Path to file
        encoding: Text encoding
        
    Returns:
        File contents as string
    """
    async with aiofiles.open(file_path, 'r', encoding=encoding) as f:
        return await f.read()


async def write_file_async(
    file_path: Path,
    content: str,
    encoding: str = 'utf-8'
) -> None:
    """
    Write file asynchronously.
    
    Args:
        file_path: Path to file
        content: Content to write
        encoding: Text encoding
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)
    async with aiofiles.open(file_path, 'w', encoding=encoding) as f:
        await f.write(content)


async def read_file_lines_async(
    file_path: Path,
    encoding: str = 'utf-8'
) -> List[str]:
    """
    Read file lines asynchronously.
    
    Args:
        file_path: Path to file
        encoding: Text encoding
        
    Returns:
        List of lines
    """
    async with aiofiles.open(file_path, 'r', encoding=encoding) as f:
        return await f.readlines()


async def append_file_async(
    file_path: Path,
    content: str,
    encoding: str = 'utf-8'
) -> None:
    """
    Append to file asynchronously.
    
    Args:
        file_path: Path to file
        content: Content to append
        encoding: Text encoding
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)
    async with aiofiles.open(file_path, 'a', encoding=encoding) as f:
        await f.write(content)


def read_file_sync(file_path: Path, encoding: str = 'utf-8') -> str:
    """
    Read file synchronously.
    
    Args:
        file_path: Path to file
        encoding: Text encoding
        
    Returns:
        File contents
    """
    with open(file_path, 'r', encoding=encoding) as f:
        return f.read()


def write_file_sync(
    file_path: Path,
    content: str,
    encoding: str = 'utf-8'
) -> None:
    """
    Write file synchronously.
    
    Args:
        file_path: Path to file
        content: Content to write
        encoding: Text encoding
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w', encoding=encoding) as f:
        f.write(content)

